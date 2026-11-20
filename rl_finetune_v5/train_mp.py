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
                                  ppo_update, save_ckpt, trunk_state, contact_names,
                                  EXTERO, PROPRIO, verify_target_scene)
from common.eval_protocol import material_command_speed  # noqa: E402
from common.material_route_monitor import MaterialRouteMonitor, frame_overlap_top  # noqa: E402
from stage3_rl.env import Go2WEnv  # noqa: E402
import mujoco  # noqa: E402

REW_SCALE = 0.01
W = None
GOAL_X = 27.6


class EnvWrap:
    def __init__(self, seed, start_x):
        kw = dict(seed=seed, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
                  eval_protocol="pose-v1", decoder_path=str(DEC),
                  material_aux_limit=6.0, material_wheel_control="residual_v1")
        if start_x is not None:
            kw["material_eval_start"] = float(start_x)
        self.env = Go2WEnv(**kw)
        m, d = self.env.model, self.env.data
        self.wheels = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot_0")
                       for l in ("FL", "FR", "RL", "RR")]
        self.reset()

    def reset(self):
        self.obs = self.env.reset()
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


def _worker_init(seed, start_x, goal_x, wdict):
    global _ENV, GOAL_X, W
    GOAL_X = goal_x
    W = RewardWeights(**wdict)
    _ENV = EnvWrap(seed, start_x)


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
    stuck = (step > 700 and x0 < GOAL_X - 0.4 and e.stuck_ref is not None
             and (x0 - e.stuck_ref) < 0.15)
    finished = False
    ok = False
    if step % 200 == 0:
        e.stuck_ref = x0
    if done or e.route.violations or x0 >= GOAL_X or stuck:
        ok = (x0 >= GOAL_X) and not e.route.violations
        term = terminal_reward(ok, W)
        rew += term
        e.parts_acc["total"] += term
        finished = True
    return (obs2 if not finished else None, rew * REW_SCALE, finished, ok,
            round(x0, 2), str(info.get("reason")), {k: round(v, 1) for k, v in e.parts_acc.items()})


def _reset(_ignored=None):
    return _ENV.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--start-x", type=float, default=21.5)
    ap.add_argument("--goal-x", type=float, default=27.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global W
    verify_target_scene(str(TRACK))
    torch.manual_seed(7); np.random.seed(7)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck, arch, passthrough, enc, actor, nm, ns = build_from_checkpoint(Path(args.init), dev)
    _, _, _, enc_cpu, actor_cpu, _, _ = build_from_checkpoint(Path(args.init), "cpu")
    from stage3_rl.ppo_train import Critic
    critic = Critic().to(dev)
    with torch.no_grad():
        actor.log_std[:16] = 0.40; actor.log_std[16:28] = 0.24; actor.log_std[28:] = 0.60
        if hasattr(actor, "wheel_log_std"):
            actor.wheel_log_std.fill_(0.20)
    opt = torch.optim.AdamW(
        [{"params": [p for p in list(enc.parameters()) + list(actor.parameters()) if p.requires_grad],
          "lr": args.lr}, {"params": critic.parameters(), "lr": 3e-4}], weight_decay=1e-4)

    W = RewardWeights(lane=60.0, lane_deadband=0.08, lane_block_mult=5.0)
    ctx = get_context("spawn")
    pool = ctx.Pool(args.envs, initializer=_worker_init,
                    initargs=(8800, args.start_x, args.goal_x, dict(W.__dict__)))
    (out / "run_config.json").write_text(json.dumps(
        {"init": args.init, "arch": arch, "iters": args.iters, "envs": args.envs,
         "start_x": args.start_x, "goal_x": args.goal_x, "lr": args.lr,
         "track": str(TRACK), "reward": W.__dict__}, indent=2), encoding="utf-8")

    metrics = (out / "metrics.jsonl").open("a", encoding="utf-8")
    t0 = time.perf_counter()

    for it in range(1, args.iters + 1):
        obs_now = pool.map_async(_reset, range(args.envs)).get()
        actor.reset_context()
        recs = [{"obs": [], "act": [], "logp": [], "rew": [], "done": [], "value": [],
                 "success": False, "max_x": 0.0, "reason": "", "parts": {}} for _ in range(args.envs)]
        for step in range(9000):
            obs_list = pool.map(_get_obs, [step] * args.envs)
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
            for i, (obs2, rew, finished, ok, x0, reason, parts) in enumerate(outs):
                r = recs[i]
                r["obs"].append(obs_list[i]); r["act"].append(act_np[i])
                r["logp"].append(float(logp[i])); r["rew"].append(rew)
                r["done"].append(1.0 if finished else 0.0); r["value"].append(float(vals[i]))
                if finished:
                    r["success"] = ok; r["max_x"] = x0; r["reason"] = reason; r["parts"] = parts
                else:
                    alive += 1
            if alive == 0:
                break
        episodes = [Episode(obs=np.stack(r["obs"]), act=np.stack(r["act"]),
                            logp=np.asarray(r["logp"], np.float32),
                            rew=np.asarray(r["rew"], np.float32),
                            done=np.asarray(r["done"], np.float32),
                            value=np.asarray(r["value"], np.float32),
                            success=bool(r["success"]), max_x=float(r["max_x"]),
                            reason=str(r["reason"]), parts=r["parts"]) for r in recs]
        stat = ppo_update(enc, actor, critic, opt, episodes, nm, ns, actor_cpu, dev, epochs=1)
        n_ok = sum(1 for e in episodes if e.success)
        rec = {"iter": it, "pass": n_ok, "envs": args.envs,
               "mean_x": round(float(np.mean([e.max_x for e in episodes])), 2),
               "max_x": round(float(np.max([e.max_x for e in episodes])), 2),
               "mean_rew": round(float(np.mean([e.rew.sum() for e in episodes])), 2),
               "lane": round(float(np.mean([e.parts.get("lane", 0) for e in episodes])), 1),
               "climb": round(float(np.mean([e.parts.get("climb", 0) for e in episodes])), 1),
               **{k: round(v, 5) for k, v in stat.items()},
               "sec": round(time.perf_counter() - t0, 1)}
        metrics.write(json.dumps(rec) + "\n"); metrics.flush()
        print("[iter %d] %s" % (it, json.dumps(rec)), flush=True)
        if it % 10 == 0:
            save_ckpt(out / "model_latest.pt", enc, actor, critic, ck, nm, ns, it, arch, passthrough, W)


if __name__ == "__main__":
    main()
