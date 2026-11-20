"""并行版全程 PPO 微调（同样的 temporal actor 与奖励，但 10 个环境用线程并行取观测/步进）。

相比 finetune_full_course.py：那里 10 个环境串行，每轮 ~150s；这里环境 I/O 并行后每轮 ~10-20s。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"E:\desktop-rl-apl\rl_finetune_v5")
from course_reward import RewardWeights, step_reward, terminal_reward  # noqa: E402
from finetune_full_course import (DEC, TRACK, Episode, build_from_checkpoint,  # noqa: E402
                                  compute_gae, precompute_kin, ppo_update, save_ckpt,
                                  trunk_state, contact_names, EXTERO, PROPRIO,
                                  verify_target_scene)
from common.eval_protocol import material_command_speed  # noqa: E402
from common.material_route_monitor import MaterialRouteMonitor, frame_overlap_top  # noqa: E402
from stage3_rl.env import Go2WEnv  # noqa: E402
import mujoco  # noqa: E402

REW_SCALE = 0.01


class EnvWrap:
    def __init__(self, seed, start_x=None, episode_len=9000):
        kw = dict(seed=seed, fixed_track=str(TRACK), eval_mode=True, episode_len=episode_len,
                  eval_protocol="pose-v1", decoder_path=str(DEC),
                  material_aux_limit=6.0, material_wheel_control="residual_v1")
        if start_x is not None:
            kw["material_eval_start"] = float(start_x)
        self.env = Go2WEnv(**kw)
        m, d = self.env.model, self.env.data
        self.wheels = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot_0")
                       for l in ("FL", "FR", "RL", "RR")]
        self.obs = None
        self.route = None
        self.prev_action = None
        self.parts_acc = None
        self.stuck_ref = None
        self.rec = None
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
        self.parts_acc = {k: 0.0 for k in ("progress", "upright", "height", "smooth", "torque",
                                           "lane", "jump", "climb", "stall", "energy", "total")}
        self.rec = {"obs": [], "act": [], "logp": [], "rew": [], "done": [], "value": []}

    # ---------------- 线程里执行的两件事 ----------------
    def get_obs(self, step):
        d = self.env.data
        self.env.vx_cmd = material_command_speed(step, float(d.qpos[0]), "start_ramp_100_v1")
        return self.env._get_obs()

    def compute_and_step(self, args):
        action, value, step, goal_x, w = args
        env = self.env
        d = env.data
        roll, pitch, yaw, vx, vy, vz, z = trunk_state(env)
        parts = step_reward(vx=vx, vx_cmd=env.vx_cmd, roll=roll, pitch=pitch, z=z, vz=vz,
                            y=float(d.qpos[1]), x=float(d.qpos[0]),
                            tau_leg=d.ctrl[env.leg_act].copy(),
                            tau_wheel=d.ctrl[env.wheel_act].copy(),
                            wheel_action=float(action[29]), action=action,
                            prev_action=self.prev_action, w=w)
        self.prev_action = np.asarray(action, dtype=np.float64).copy()
        x0 = float(d.qpos[0])
        obs2, _r, done, info = env.step(np.asarray(action, dtype=np.float64))
        self.route.update(float(d.qpos[0]), float(d.qpos[1]), d.geom_xpos[self.wheels, :2],
                          contact_names(env.model, d),
                          frame_overlap_top(env.model, d, 44.98, 45.02))
        rew = parts["total"]
        for k in self.parts_acc:
            if k != "total":
                self.parts_acc[k] += parts.get(k, 0.0)
        self.parts_acc["total"] += parts["total"]

        finished = False
        ok = False
        stuck = (step > 600 and x0 < goal_x - 0.4
                 and self.stuck_ref is not None and (x0 - self.stuck_ref) < 0.15)
        if done or self.route.violations or x0 >= goal_x or stuck:
            ok = (x0 >= goal_x) and not self.route.violations
            term = terminal_reward(ok, w)
            rew += term
            self.parts_acc["total"] += term
            finished = True
        if step % 200 == 0:
            self.stuck_ref = x0
        return obs2, rew * REW_SCALE, float(value), finished, ok, info.get("reason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--iters", type=int, default=140)
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--start-x", type=float, default=None)
    ap.add_argument("--goal-x", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    verify_target_scene(str(TRACK))
    torch.manual_seed(7); np.random.seed(7)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck, arch, passthrough, enc, actor, nm, ns = build_from_checkpoint(Path(args.init), dev)
    _, _, _, enc_cpu, actor_cpu, _, _ = build_from_checkpoint(Path(args.init), "cpu")
    from stage3_rl.ppo_train import Critic
    critic = Critic().to(dev)
    if "critic" in ck and isinstance(ck["critic"], dict) and ck["critic"]:
        try:
            critic.load_state_dict(ck["critic"])
        except Exception:
            pass
    with torch.no_grad():
        actor.log_std[:16] = 0.40; actor.log_std[16:28] = 0.24; actor.log_std[28:] = 0.60
        if hasattr(actor, "wheel_log_std"):
            actor.wheel_log_std.fill_(0.20)
    opt = torch.optim.AdamW([{"params": [p for p in list(enc.parameters()) + list(actor.parameters()) if p.requires_grad],
                              "lr": args.lr}, {"params": critic.parameters(), "lr": 3e-4}], weight_decay=1e-4)
    w = RewardWeights(lane=60.0, lane_deadband=0.08, lane_block_mult=5.0)
    (out / "run_config.json").write_text(json.dumps(
        {"init": args.init, "arch": arch, "iters": args.iters, "envs": args.envs,
         "start_x": args.start_x, "goal_x": args.goal_x, "lr": args.lr,
         "track": str(TRACK), "reward": w.__dict__}, indent=2), encoding="utf-8")

    envs = [EnvWrap(8800 + i, start_x=args.start_x, episode_len=9000) for i in range(args.envs)]
    pool = ThreadPoolExecutor(max_workers=args.envs)
    metrics = (out / "metrics.jsonl").open("a", encoding="utf-8")
    t0 = time.perf_counter()

    for it in range(1, args.iters + 1):
        for e in envs:
            e.reset()
        actor.reset_context()
        for step in range(9000):
            obs_list = list(pool.map(lambda e: e.get_obs(step), envs))
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
            tasks = [(i, (act_np[i], float(vals[i]), step, args.goal_x, w)) for i in range(len(envs))]
            outs = list(pool.map(lambda a: envs[a[0]].compute_and_step(a[1]), tasks))
            any_alive = False
            for i, (obs2, rew, val, finished, ok, reason) in enumerate(outs):
                e = envs[i]
                e.rec["obs"].append(obs_list[i]); e.rec["act"].append(act_np[i])
                e.rec["logp"].append(float(logp[i])); e.rec["rew"].append(rew)
                e.rec["done"].append(1.0 if finished else 0.0); e.rec["value"].append(val)
                if finished:
                    e.rec["success"] = ok
                    e.rec["parts"] = dict(e.parts_acc)
                    e.rec["max_x"] = float(e.env.data.qpos[0])
                    e.rec["reason"] = reason
                else:
                    any_alive = True
            if not any_alive:
                break
        episodes = []
        for e in envs:
            r = e.rec
            episodes.append(Episode(obs=np.stack(r["obs"]), act=np.stack(r["act"]),
                                    logp=np.asarray(r["logp"], np.float32),
                                    rew=np.asarray(r["rew"], np.float32),
                                    done=np.asarray(r["done"], np.float32),
                                    value=np.asarray(r["value"], np.float32),
                                    success=bool(r.get("success", False)),
                                    max_x=float(r.get("max_x", 0.0)),
                                    reason=str(r.get("reason")),
                                    parts=r.get("parts", {})))
        stat = ppo_update(enc, actor, critic, opt, episodes, nm, ns, actor_cpu, dev, epochs=1)
        n_ok = sum(1 for e in episodes if e.success)
        rec = {"iter": it, "pass": n_ok, "envs": len(episodes),
               "mean_x": round(float(np.mean([e.max_x for e in episodes])), 2),
               "mean_rew": round(float(np.mean([e.rew.sum() for e in episodes])), 2),
               "lane": round(float(np.mean([e.parts.get("lane", 0) for e in episodes])), 1),
               "climb": round(float(np.mean([e.parts.get("climb", 0) for e in episodes])), 1),
               **{k: round(v, 5) for k, v in stat.items()},
               "sec": round(time.perf_counter() - t0, 1)}
        metrics.write(json.dumps(rec) + "\n"); metrics.flush()
        print("[iter %d] %s" % (it, json.dumps(rec)), flush=True)
        if it % 10 == 0:
            save_ckpt(out / "model_latest.pt", enc, actor, critic, ck, nm, ns, it, arch, passthrough, w)


if __name__ == "__main__":
    main()


