"""多进程版轮胎段 PPO 微调：环境放到独立子进程，父进程只做 actor/critic 前向与更新。

训练目标：从 21.5m 起跑，越过 24.5m 轮胎、过 27.6m 里程碑，并保持中线。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"E:\desktop-rl-apl\rl_finetune_v5")
sys.path.insert(0, r"E:\desktop-rl-apl\aptrl_go2w")
from course_reward import RewardWeights, step_reward, terminal_reward  # noqa: E402
from finetune_full_course import (DEC, TRACK, Episode, build_from_checkpoint,  # noqa: E402
                                  ppo_update, save_ckpt, trunk_state, contact_names, compute_gae,
                                  EXTERO, PROPRIO, verify_target_scene)
from common.eval_protocol import material_command_speed  # noqa: E402
from common.material_route_monitor import MaterialRouteMonitor, frame_overlap_top  # noqa: E402
from stage3_rl.env import Go2WEnv  # noqa: E402
import mujoco  # noqa: E402

REW_SCALE = 0.01
W = None
GOAL_X = 27.6
STUCK_AFTER = 700
MAX_STEPS = 9000


class EnvWrap:
    def __init__(self, seed, start_x):
        kw = dict(seed=seed, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
                  eval_protocol="pose-v1", decoder_path=str(DEC),
                  material_aux_limit=6.0, material_wheel_control="residual_v1")
        self.start_x = None if (start_x is None or float(start_x) <= 0) else float(start_x)
        if self.start_x is not None:
            kw["material_eval_start"] = float(start_x) if self.start_x in (21.5, 30.2) else 21.5
        self.env = Go2WEnv(**kw)
        m, d = self.env.model, self.env.data
        self.wheels = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot_0")
                       for l in ("FL", "FR", "RL", "RR")]
        self.reset()

    def reset(self):
        self.obs = self.env.reset()
        if self.start_x is not None and self.start_x not in (21.5, 30.2):
            d = self.env.data
            d.qpos[0] = self.start_x
            mujoco.mj_forward(self.env.model, d)
            self.env.vx_cmd = material_command_speed(0, self.start_x, "start_ramp_100_v1")
            self.obs = self.env._get_obs()
        self.route = MaterialRouteMonitor()
        d = self.env.data
        self.route.update(float(d.qpos[0]), float(d.qpos[1]), d.geom_xpos[self.wheels, :2],
                          contact_names(self.env.model, d),
                          frame_overlap_top(self.env.model, d, 44.98, 45.02))
        if getattr(self.env, "material_eval_start", None) is not None:
            self.route.started = True
            self.route.violations.discard("invalid_start")
        self.prev_action = None
        self.stuck_ref = None
        self.parts_acc = {k: 0.0 for k in ("progress", "upright", "height", "smooth", "torque",
                                           "lane", "jump", "climb", "stall", "energy", "total")}
        return self.obs


_ENV = None


def _worker_init(seed, start_x, goal_x, wdict, stuck_after=700, max_steps=9000):
    global _ENV, GOAL_X, W, STUCK_AFTER, MAX_STEPS
    MAX_STEPS = max_steps
    from multiprocessing import current_process
    GOAL_X = goal_x
    STUCK_AFTER = stuck_after
    W = RewardWeights(**wdict)
    try:
        idx = current_process()._identity[0] - 1
    except Exception:
        idx = 0
    _ENV = EnvWrap(seed + idx, start_x)


def _get_obs(step):
    e = _ENV
    d = e.env.data
    e.env.vx_cmd = material_command_speed(step, float(d.qpos[0]), "start_ramp_100_v1")
    return e.env._get_obs()


def _step(payload):
    step, action = payload
    e = _ENV
    env = e.env
    d = env.data
    roll, pitch, yaw, vx, vy, vz, z = trunk_state(env)
    parts = step_reward(vx=vx, vx_cmd=env.vx_cmd, roll=roll, pitch=pitch, z=z, vz=vz,
                        y=float(d.qpos[1]), x=float(d.qpos[0]),
                        tau_leg=d.ctrl[env.leg_act].copy(),
                        tau_wheel=d.ctrl[env.wheel_act].copy(),
                        wheel_action=float(action[29]), action=action,
                        prev_action=e.prev_action, w=W)
    e.prev_action = np.asarray(action, dtype=np.float64).copy()
    obs2, _r, done, info = env.step(np.asarray(action, dtype=np.float64))
    e.route.update(float(d.qpos[0]), float(d.qpos[1]), d.geom_xpos[e.wheels, :2],
                   contact_names(env.model, d), frame_overlap_top(env.model, d, 44.98, 45.02))
    for k in e.parts_acc:
        if k != "total":
            e.parts_acc[k] += parts.get(k, 0.0)
    x0 = float(d.qpos[0])
    rew = parts["total"]
    e.parts_acc["total"] += rew
    stuck = ((step > STUCK_AFTER and x0 < GOAL_X - 0.4 and e.stuck_ref is not None
             and (x0 - e.stuck_ref) < 0.15) or step >= MAX_STEPS)
    finished = False
    ok = False
    if step % 100 == 0:
        e.stuck_ref = x0
    if done or e.route.violations or x0 >= GOAL_X or stuck:
        ok = (x0 >= GOAL_X) and not e.route.violations
        term = terminal_reward(ok, W)
        rew += term
        e.parts_acc["total"] += term
        finished = True
    if finished:
        viol = sorted(e.route.violations)
        return (None, rew * REW_SCALE, finished, ok, round(x0, 2),
                str(info.get("reason")), {k: round(v, 1) for k, v in e.parts_acc.items()}, viol)
    env.vx_cmd = material_command_speed(step + 1, float(d.qpos[0]), "start_ramp_100_v1")
    return (env._get_obs(), rew * REW_SCALE, finished, ok, round(x0, 2),
            str(info.get("reason")), {k: round(v, 1) for k, v in e.parts_acc.items()}, [])


def _reset(_ignored=None):
    return _ENV.reset()


def ppo_update_anchored(enc, actor, critic, opt, episodes, nm, ns, actor_cpu, device,
                        ref_actor=None, bc_coef=0.3, clip=0.2, epochs=1, chunk=512,
                        ent_coef=0.0, lr_floor=5e-6):
    """PPO 更新 + 对 BC 策略均值的锚定项，防止在轮胎前退化成"站桩"。"""
    import torch.nn as nn
    advs = []
    for ep in episodes:
        adv, ret = compute_gae(ep.rew, ep.done, ep.value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        advs.append((adv, ret))
    stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "bc": 0.0, "n": 0}
    for _ in range(epochs):
        for i, ep in enumerate(episodes):
            adv, ret = advs[i]
            obs_n = np.clip((ep.obs - nm) / ns, -10, 10).astype(np.float32)
            actor.reset_context()
            if ref_actor is not None:
                ref_actor.reset_context()
            T = len(ep.rew)
            for c0 in range(0, T, chunk):
                c1 = min(c0 + chunk, T)
                lp_new, ents, bc_terms = [], [], []
                for t in range(c0, c1):
                    ob_t = torch.from_numpy(obs_n[t][None]).to(device)
                    ao = torch.cat([ob_t[:, :PROPRIO], enc(ob_t[:, PROPRIO:PROPRIO + EXTERO]),
                                    ob_t[:, PROPRIO + EXTERO:]], dim=1)
                    dist = actor.dist(ao)
                    a_t = torch.from_numpy(ep.act[t][None]).to(device)
                    lp_new.append(dist.log_prob(a_t).sum(-1))
                    ents.append(dist.entropy().sum(-1))
                    if ref_actor is not None:
                        with torch.no_grad():
                            ref_mean = ref_actor.dist(ao).mean
                        bc_terms.append(((dist.mean - ref_mean) ** 2).mean())
                lp_new_t = torch.stack(lp_new).reshape(-1)
                ent_t = torch.stack(ents).reshape(-1)
                lp_old = torch.from_numpy(ep.logp[c0:c1]).to(device)
                adv_t = torch.from_numpy(adv[c0:c1]).to(device)
                ret_t = torch.from_numpy(ret[c0:c1]).to(device)
                log_ratio = lp_new_t - lp_old
                ratio = log_ratio.exp()
                pi_loss = -torch.min(ratio * adv_t,
                                     torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t).mean()
                v = critic(torch.from_numpy(obs_n[c0:c1]).to(device))
                v_loss = 0.5 * nn.functional.huber_loss(v, ret_t, delta=10.0)
                ent = ent_t.mean()
                loss = pi_loss + v_loss - ent_coef * ent
                bc_val = 0.0
                if bc_terms:
                    bc_val = torch.stack(bc_terms).mean()
                    loss = loss + bc_coef * bc_val
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in list(enc.parameters()) + list(actor.parameters())
                                          if p.requires_grad], 1.0)
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt.step()
                stats["pi"] += float(pi_loss.detach()); stats["v"] += float(v_loss.detach())
                stats["ent"] += float(ent.detach()); stats["bc"] += float(bc_val)
                kl_c = float(((log_ratio.exp() - 1.0) - log_ratio).mean().detach())
                stats["kl"] += kl_c; stats["n"] += 1
                if kl_c > 0.06:
                    for g in opt.param_groups:
                        g["lr"] = max(g["lr"] * 0.7, lr_floor)
                    break
    n = max(1, stats.pop("n"))
    for k in ("pi", "v", "ent", "kl", "bc"):
        stats[k] /= n
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--start-x", type=float, default=0.0)
    ap.add_argument("--goal-x", type=float, default=27.6)
    ap.add_argument("--bc-anchor", type=float, default=0.0)
    ap.add_argument("--stuck-after", type=int, default=700)
    ap.add_argument("--std-scale", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=9000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global W
    verify_target_scene(str(TRACK))
    torch.manual_seed(7); np.random.seed(7)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck, arch, passthrough, enc, actor, nm, ns = build_from_checkpoint(Path(args.init), dev)
    _, _, _, enc_cpu, actor_cpu, _, _ = build_from_checkpoint(Path(args.init), "cpu")
    ref_actor = None
    if args.bc_anchor > 0:
        _, _, _, _, ref_actor, _, _ = build_from_checkpoint(Path(args.init), dev)
        ref_actor.eval()
        for p_ in ref_actor.parameters():
            p_.requires_grad_(False)
    from stage3_rl.ppo_train import Critic
    critic = Critic().to(dev)
    with torch.no_grad():
        import math
        actor.log_std.add_(math.log(args.std_scale))
        if hasattr(actor, "wheel_log_std"):
            actor.wheel_log_std.add_(math.log(args.std_scale))
    opt = torch.optim.AdamW(
        [{"params": [p for p in list(enc.parameters()) + list(actor.parameters()) if p.requires_grad],
          "lr": args.lr}, {"params": critic.parameters(), "lr": 3e-4}], weight_decay=1e-4)

    W = RewardWeights(lane=60.0, lane_deadband=0.08, lane_block_mult=5.0)
    ctx = get_context("spawn")
    pool = ctx.Pool(args.envs, initializer=_worker_init,
                    initargs=(8800, args.start_x, args.goal_x, dict(W.__dict__), args.stuck_after,
                              args.max_steps))
    (out / "run_config.json").write_text(json.dumps(
        {"init": args.init, "arch": arch, "iters": args.iters, "envs": args.envs,
         "start_x": args.start_x, "goal_x": args.goal_x, "lr": args.lr,
         "track": str(TRACK), "reward": W.__dict__}, indent=2), encoding="utf-8")

    metrics = (out / "metrics.jsonl").open("a", encoding="utf-8")
    dbg = (out / "episodes.jsonl").open("a", encoding="utf-8")
    t0 = time.perf_counter()

    for it in range(1, args.iters + 1):
        obs_now = pool.map_async(_reset, range(args.envs)).get()
        actor.reset_context()
        recs = [{"obs": [], "act": [], "logp": [], "rew": [], "done": [], "value": [],
                 "success": False, "max_x": 0.0, "reason": "", "parts": {}} for _ in range(args.envs)]
        for step in range(9000):
            obs_list = obs_now
            ob = np.stack(obs_list).astype(np.float32)
            ob_n = np.clip((ob - nm) / ns, -10, 10).astype(np.float32)
            ob_t = torch.from_numpy(ob_n).to(dev)
            with torch.no_grad():
                ao = torch.cat([ob_t[:, :PROPRIO], enc(ob_t[:, PROPRIO:PROPRIO + EXTERO]),
                                ob_t[:, PROPRIO + EXTERO:]], dim=1)
                dist = actor.dist(ao)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                vals = critic(ob_t)
            act_np = act.cpu().numpy()
            outs = pool.map(_step, [(step, a) for a in act_np])
            alive = 0
            for i, (obs2, rew, finished, ok, x0, reason, parts, viol) in enumerate(outs):
                r = recs[i]
                r["obs"].append(obs_list[i]); r["act"].append(act_np[i])
                r["logp"].append(float(logp[i])); r["rew"].append(rew)
                r["done"].append(1.0 if finished else 0.0); r["value"].append(float(vals[i]))
                if finished:
                    r["success"] = ok; r["max_x"] = x0; r["reason"] = reason; r["parts"] = parts
                    r["viol"] = viol
                    dbg.write(json.dumps({"iter": it, "env": i, "max_x": x0, "ok": ok,
                                          "reason": reason, "viol": viol,
                                          "lane": parts.get("lane"), "climb": parts.get("climb"),
                                          "rew": round(float(np.sum(r["rew"])), 2)}) + "\n")
                else:
                    alive += 1
                    if obs2 is not None:
                        obs_now[i] = obs2
            if alive == 0:
                break
        episodes = [Episode(obs=np.stack(r["obs"]), act=np.stack(r["act"]),
                            logp=np.asarray(r["logp"], np.float32),
                            rew=np.asarray(r["rew"], np.float32),
                            done=np.asarray(r["done"], np.float32),
                            value=np.asarray(r["value"], np.float32),
                            success=bool(r["success"]), max_x=float(r["max_x"]),
                            reason=str(r["reason"]), parts=r["parts"]) for r in recs]
        stat = ppo_update_anchored(enc, actor, critic, opt, episodes, nm, ns, actor_cpu, dev,
                               ref_actor=ref_actor, bc_coef=args.bc_anchor, epochs=1)
        n_ok = sum(1 for e in episodes if e.success)
        rec = {"iter": it, "pass": n_ok, "envs": args.envs,
               "mean_x": round(float(np.mean([e.max_x for e in episodes])), 2),
               "max_x": round(float(np.max([e.max_x for e in episodes])), 2),
               "mean_rew": round(float(np.mean([e.rew.sum() for e in episodes])), 2),
               "lane": round(float(np.mean([e.parts.get("lane", 0) for e in episodes])), 1),
               "climb": round(float(np.mean([e.parts.get("climb", 0) for e in episodes])), 1),
               **{k: round(v, 5) for k, v in stat.items()},
               "sec": round(time.perf_counter() - t0, 1)}
        metrics.write(json.dumps(rec) + "\n"); metrics.flush(); dbg.flush()
        print("[iter %d] %s" % (it, json.dumps(rec)), flush=True)
        if it % 10 == 0:
            save_ckpt(out / "model_latest.pt", enc, actor, critic, ck, nm, ns, it, arch, passthrough, W)


if __name__ == "__main__":
    main()
