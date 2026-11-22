"""从现有 checkpoint 出发的**全程** PPO 微调（v5 赛道，自定义奖励）。

用法（先冒烟，再正式训练）::

    $env:PYTHONPATH="E:\\desktop-rl-apl\\aptrl_go2w"
    python E:\\desktop-rl-apl\\rl_finetune_v5\\finetune_full_course.py --smoke

    python E:\\desktop-rl-apl\\rl_finetune_v5\\finetune_full_course.py `
        --init <checkpoint.pt> --iters 200 --envs 6 --out E:\\desktop-rl-apl\\rl_finetune_v5\\runs\\run1

实现要点（便于审阅）:
  * 策略/编码器复用 aprtl_go2w 的 `ExteroEncoder` / `TemporalWheelResidualActor` /
    `FeedforwardKinActor`，与 `stage3_rl/evaluate.py::PolicyRunner` 完全同构，
    因此训练产物可直接被官方验收器加载。
  * 奖励完全由本目录 `course_reward.py` 计算（环境自带奖励只用来越过 done 判定）。
  * 每个 episode 从起点开始（pose-v1），最多 9000 步，跑满全程。
  * 更新阶段按 episode 片段做序列前向（kinematics 状态可离线重放，和推理一致）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
REPO = Path(r"E:\desktop-rl-apl\aptrl_go2w")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from course_reward import RewardWeights, step_reward, terminal_reward, weights_dict  # noqa: E402
from reference_traj import ReferenceTrajectory, reference_reward, anneal_weight  # noqa: E402
from common.eval_protocol import material_command_speed  # noqa: E402
from common.material_acceptance import file_sha256, verify_target_scene  # noqa: E402
from common.material_route_monitor import MaterialRouteMonitor, frame_overlap_top  # noqa: E402
from stage3_rl.env import EXTERO, PROPRIO, Go2WEnv  # noqa: E402
from stage3_rl.material_acceptance import run_material_episode  # noqa: E402
from stage3_rl.ppo_train import (GAIT, Critic, ExteroEncoder, FeedforwardKinActor,  # noqa: E402
                                 TemporalWheelResidualActor)
from stage3_rl.evaluate import PolicyRunner  # noqa: E402

TRACK = REPO / "results/diagnostics/robocon-material-preview-20261119-v5/scene.xml"
DEC = REPO / "stage2_tvae/candidates/20261113T153754Z/tvae_decoders.npz"
POSE_SEED0 = 8800
REW_SCALE = 0.01          # 原始奖励步均 ~1.2，缩放到 PPO 更稳；分项日志仍记录未缩放值


# ----------------------------------------------------------------- 模型装载
def build_from_checkpoint(ckpt_path: Path, device: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck.get("actor_arch")
    passthrough = bool(ck.get("extero_passthrough", False))
    enc = ExteroEncoder(split_frozen=True, passthrough_raw=passthrough).to(device)
    if arch == "temporal_residual_v4":
        actor_cls = TemporalWheelResidualActor
    elif arch == "feedforward_kin_v4":
        actor_cls = FeedforwardKinActor
    else:
        raise ValueError(f"unsupported actor_arch for this trainer: {arch}")
    nm = np.asarray(ck["norm"]["mean"], dtype=np.float64)
    ns = np.sqrt(np.asarray(ck["norm"]["var"], dtype=np.float64) + 1e-8)
    actor = actor_cls(hidden=int(ck.get("actor_hidden", 768)),
                      gru_hidden=int(ck.get("gru_hidden", 256)),
                      raw_norm_mean=nm, raw_norm_std=ns,
                      extero_passthrough=passthrough).to(device)
    enc.load_state_dict(ck["enc"])
    try:
        actor.load_state_dict(ck["actor"])
    except RuntimeError:
        actor.load_state_dict(ck["actor"], strict=False)
    for p in actor.log_std.parameters() if isinstance(actor.log_std, nn.Module) else [actor.log_std]:
        p.requires_grad_(False)
    if hasattr(actor, "wheel_log_std"):
        actor.wheel_log_std.requires_grad_(False)
    return ck, arch, passthrough, enc, actor, nm, ns


# ----------------------------------------------------------------- 工具
def trunk_state(env):
    d = env.data
    R = d.xmat[env.body_root].reshape(3, 3)
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    pitch = float(-np.arcsin(np.clip(R[2, 0], -1, 1)))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    v_world = R @ d.qvel[0:3]
    return roll, pitch, yaw, float(v_world[0]), float(v_world[1]), float(v_world[2]), float(d.qpos[2])


def contact_names(model, data):
    import mujoco
    return [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "",
             mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "") for c in data.contact]


def precompute_kin(actor_cpu, ob_norm):
    """按推理顺序重放 causality 特征（与 dist() 内部逐步计算一致）。"""
    actor_cpu.reset_context()
    out = []
    with torch.no_grad():
        t = torch.from_numpy(ob_norm)
        for i in range(len(ob_norm)):
            out.append(actor_cpu._compute_kin_step(t[i:i + 1, :PROPRIO],
                                                   t[i:i + 1, PROPRIO:PROPRIO + EXTERO]))
    return torch.cat(out, 0).numpy().astype(np.float32)


@dataclass
class Episode:
    obs: np.ndarray        # (T, 426) raw
    act: np.ndarray        # (T, 30)
    logp: np.ndarray       # (T,)
    rew: np.ndarray        # (T,)
    done: np.ndarray       # (T,)
    value: np.ndarray      # (T,)
    success: bool
    max_x: float
    reason: str
    parts: dict            # 各奖励项累计


# ----------------------------------------------------------------- 采样
@torch.no_grad()
def rollout(env, enc, actor, critic, nm, ns, device, cmd_profile, w: RewardWeights, goal_x=100.0,
            ref=None, ref_w=0.0, ref_act_w=0.0):
    import mujoco
    obs = env.reset()
    actor.reset_context()
    m, d = env.model, env.data
    wheels = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot_0") for l in ("FL", "FR", "RL", "RR")]
    route = MaterialRouteMonitor()
    route.update(float(d.qpos[0]), float(d.qpos[1]), d.geom_xpos[wheels, :2],
                 contact_names(m, d), frame_overlap_top(m, d, 44.98, 45.02))
    if getattr(env, "material_eval_start", None) is not None:
        # 课程学习从半程起跑：跑线监控的"必须从起点线开始"不适用于本阶段
        route.started = True
        route.violations.discard("invalid_start")
    obs_l, act_l, logp_l, rew_l, done_l, val_l = [], [], [], [], [], []
    acc = {k: 0.0 for k in ("progress", "upright", "height", "smooth", "torque",
                            "lane", "jump", "energy", "ref", "total")}
    prev_action = None
    info = {"reason": None, "success": False}
    steps = 0
    for step in range(9000):
        x = float(d.qpos[0])
        env.vx_cmd = material_command_speed(step, x, cmd_profile)
        obs = env._get_obs()
        ob_n = np.clip((obs - nm) / ns, -10, 10).astype(np.float32)
        ob_t = torch.from_numpy(ob_n[None]).to(device)
        with torch.no_grad():
            ao = torch.cat([ob_t[:, :PROPRIO], enc(ob_t[:, PROPRIO:PROPRIO + EXTERO]),
                            ob_t[:, PROPRIO + EXTERO:]], dim=1)
            dist = actor.dist(ao)
            action = dist.sample()            # 需要探索：确定性均值在停滞点永远学不动
            logp = dist.log_prob(action).sum(-1)
            value = float(critic(ob_t).item())
        a_np = action.cpu().numpy()[0]

        roll, pitch, yaw, vx, vy, vz, z = trunk_state(env)
        parts = step_reward(vx=vx, vx_cmd=env.vx_cmd, roll=roll, pitch=pitch, z=z, vz=vz,
                            y=float(d.qpos[1]), x=x,
                            tau_leg=d.ctrl[env.leg_act].copy(),
                            tau_wheel=d.ctrl[env.wheel_act].copy(),
                            wheel_action=float(a_np[29]),
                            action=a_np, prev_action=prev_action, w=w)
        if ref is not None and ref_w > 0.0:
            parts["ref"] = reference_reward(ref, x, float(d.qpos[1]), yaw,
                                            ref_w, act=a_np, act_weight=ref_act_w)
            parts["total"] += parts["ref"]
        else:
            parts["ref"] = 0.0
        prev_action = a_np.copy()

        obs_used = obs
        obs2, _env_rew, done, info = env.step(a_np)
        obs = obs2
        steps = step + 1
        route.update(float(d.qpos[0]), float(d.qpos[1]), d.geom_xpos[wheels, :2],
                     contact_names(m, d), frame_overlap_top(m, d, 44.98, 45.02))

        rew = parts["total"] * REW_SCALE
        route_ok = route.success()
        if done or route.violations or float(d.qpos[0]) >= goal_x:
            ok = (float(d.qpos[0]) >= goal_x) and route_ok
            term = terminal_reward(ok, w)
            rew += term * REW_SCALE
            parts["terminal"] = term
            acc = {k: acc[k] + parts.get(k, 0.0) for k in acc}
            obs_l.append(obs_used); act_l.append(a_np); logp_l.append(float(logp.item()))
            rew_l.append(rew); done_l.append(1.0); val_l.append(value)
            success = ok and not route.violations
            break
        for k in acc:
            acc[k] += parts.get(k, 0.0)
        obs_l.append(obs_used); act_l.append(a_np); logp_l.append(float(logp.item()))
        rew_l.append(rew); done_l.append(0.0); val_l.append(value)

    ep = Episode(obs=np.stack(obs_l), act=np.stack(act_l), logp=np.asarray(logp_l, np.float32),
                 rew=np.asarray(rew_l, np.float32), done=np.asarray(done_l, np.float32),
                 value=np.asarray(val_l, np.float32),
                 success=(float(d.qpos[0]) >= goal_x) and route.success(),
                 max_x=float(route.max_x), reason=str(info.get("reason")), parts=acc)
    return ep, steps


# ----------------------------------------------------------------- PPO 更新
def compute_gae(rew, done, values, gamma=0.997, lam=0.97):
    T = len(rew)
    adv = np.zeros(T, np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        next_v = values[t + 1] if t < T - 1 else 0.0
        mask = 1.0 - done[t]
        delta = rew[t] + gamma * next_v * mask - values[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    return adv, adv + values


def ppo_update(enc, actor, critic, opt, episodes, nm, ns, actor_cpu, device,
               clip=0.2, epochs=1, chunk=512, batch_eps=None, ent_coef=0.0):
    """逐步重放 rollout 时看到的同一序列（保证时序actor/history完全一致），做 PPO 裁剪更新。

    采用逐 episode、逐 chunk 的前向+反向：recurrent state 与采样时一致，
    只是把递推图截断在 chunk 边界（标准的 truncated BPTT）。
    """
    advs = []
    for ep in episodes:
        adv, ret = compute_gae(ep.rew, ep.done, ep.value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)   # 优势归一化，避免量纲失衡
        advs.append((adv, ret))
    stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "n": 0}
    for _ in range(epochs):
        for i, ep in enumerate(episodes):
            adv, ret = advs[i]
            obs_n = np.clip((ep.obs - nm) / ns, -10, 10).astype(np.float32)
            actor.reset_context()
            T = len(ep.rew)
            for c0 in range(0, T, chunk):
                c1 = min(c0 + chunk, T)
                lp_new = []
                ents = []
                for t in range(c0, c1):
                    ob_t = torch.from_numpy(obs_n[t][None]).to(device)
                    with torch.no_grad():
                        ao = torch.cat([ob_t[:, :PROPRIO], enc(ob_t[:, PROPRIO:PROPRIO + EXTERO]),
                                        ob_t[:, PROPRIO + EXTERO:]], dim=1)
                    # grad-enabled actor pass (enc frozen path is fine to reuse)
                    ao = torch.cat([ob_t[:, :PROPRIO], enc(ob_t[:, PROPRIO:PROPRIO + EXTERO]),
                                    ob_t[:, PROPRIO + EXTERO:]], dim=1)
                    dist = actor.dist(ao)
                    a_t = torch.from_numpy(ep.act[t][None]).to(device)
                    lp_new.append(dist.log_prob(a_t).sum(-1))
                    ents.append(dist.entropy().sum(-1))
                lp_new_t = torch.stack(lp_new).reshape(-1)
                ent_t = torch.stack(ents).reshape(-1)
                lp_old = torch.from_numpy(ep.logp[c0:c1]).to(device)
                adv_t = torch.from_numpy(adv[c0:c1]).to(device)
                ret_t = torch.from_numpy(ret[c0:c1]).to(device)
                log_ratio = lp_new_t - lp_old
                ratio = log_ratio.exp()
                s1 = ratio * adv_t
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t
                pi_loss = -torch.min(s1, s2).mean()
                ob_chunk = torch.from_numpy(obs_n[c0:c1]).to(device)
                v = critic(ob_chunk)
                v_loss = 0.5 * nn.functional.huber_loss(v, ret_t, delta=10.0)
                ent = ent_t.mean()
                loss = pi_loss + v_loss - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in list(enc.parameters()) + list(actor.parameters()) if p.requires_grad], 1.0)
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt.step()
                stats["pi"] += float(pi_loss.detach())
                stats["v"] += float(v_loss.detach())
                stats["ent"] += float(ent.detach())
                kl_c = float(((log_ratio.exp() - 1.0) - log_ratio).mean().detach())
                stats["kl"] += kl_c
                stats["n"] += 1
                # 自适应步长：一旦本 chunk 偏离策略过远就降学习率并停止本 episode 的继续更新
                if kl_c > 0.06:
                    for g in opt.param_groups:
                        g["lr"] = max(g["lr"] * 0.7, 5e-6)
                    break
    n = max(1, stats.pop("n"))
    for k in ("pi", "v", "ent", "kl"):
        stats[k] /= n
    return stats


# ----------------------------------------------------------------- 保存 / 评估
def save_ckpt(path: Path, enc, actor, critic, base_ck, nm, ns, it, arch, passthrough, w):
    state = {
        "enc": enc.state_dict(), "actor": actor.state_dict(), "critic": critic.state_dict(),
        "norm": {"mean": nm, "var": ns * ns, "count": float(base_ck.get("norm", {}).get("count", 1e6))},
        "iter": it, "actor_arch": arch, "extero_passthrough": passthrough,
        "actor_hidden": actor.hidden, "gru_hidden": getattr(actor, "gru_hidden", 256),
        "decoder_path": str(DEC), "decoder_sha256": file_sha256(DEC),
        "material_aux_limit": 6.0, "material_wheel_control": "residual_v1",
        "material_command_profile": "start_ramp_100_v1",
        "reward_weights": weights_dict(w), "scene_sha256": file_sha256(TRACK),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def evaluate(ckpt: Path, seeds):
    rows = []
    for s in seeds:
        r = PolicyRunner(str(ckpt), device="cpu", decoder_path=str(DEC))
        env = Go2WEnv(seed=s, fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
                      eval_protocol="pose-v1", decoder_path=str(DEC),
                      material_aux_limit=6.0, material_wheel_control="residual_v1")
        row = run_material_episode(env, r, max_steps=9000)
        rows.append((s, row["success"], round(row["max_x"], 2), row["reason"]))
    return rows


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(REPO / "stage3_rl/experiments/20261119_temporal_v4_nav_r1/model_best.pt"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "runs" / "v5_ft"))
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="2 环境 x 300 步、1 次更新，仅验证通路")
    ap.add_argument("--start-x", type=float, default=None, help="课程学习起点（21.5=轮胎前 / 30.2=上台阶前）")
    ap.add_argument("--goal-x", type=float, default=100.0, help="本阶段视为成功的 x")
    ap.add_argument("--ref-data", default=str(Path("C:/Users/Administrator/AppData/Local/Temp/opencode/v4_fullcourse_demos_v2_gclock.npz")),
                    help="参考轨迹数据集（脚本教师成功回合）")
    ap.add_argument("--ref-weight", type=float, default=0.0, help="轨迹跟随奖励初始权重（0=关闭）")
    ap.add_argument("--ref-act-weight", type=float, default=0.0, help="轨迹动作项权重系数（相对参考奖励）")
    ap.add_argument("--ref-anneal-iters", type=int, default=0, help="退火轮数：该轮数后参考权重降到 0")
    args = ap.parse_args()

    verify_target_scene(str(TRACK))
    torch.manual_seed(101)
    np.random.seed(101)
    w = RewardWeights(lane=60.0, lane_deadband=0.08, lane_block_mult=5.0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ck, arch, passthrough, enc, actor, nm, ns = build_from_checkpoint(Path(args.init), args.device)
    _, _, _, enc_cpu, actor_cpu, _, _ = build_from_checkpoint(Path(args.init), "cpu")
    critic = Critic().to(args.device)
    if "critic" in ck:
        try:
            critic.load_state_dict(ck["critic"])
        except Exception:
            pass
    # 探索方差：微调需要探索（原 checkpoint 的 latent std=1 会破坏步态，这里收紧并固定）
    with torch.no_grad():
        if hasattr(actor, "log_std"):
            actor.log_std[:16] = 0.45
            actor.log_std[16:28] = 0.28
            actor.log_std[28:] = 0.60
        if hasattr(actor, "wheel_log_std"):
            actor.wheel_log_std.fill_(0.22)
    print("[init] exploration std: latent 0.45 / aux 0.28 / gait 0.60 / wheel 0.22", flush=True)
    train_params = [p for p in list(enc.parameters()) + list(actor.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW([{"params": train_params, "lr": args.lr},
                             {"params": critic.parameters(), "lr": 3e-4}], weight_decay=1e-4)
    cfg = {"init": str(args.init), "arch": arch, "extero_passthrough": passthrough,
           "iters": args.iters, "envs": args.envs, "lr": args.lr,
           "scene": str(TRACK), "scene_sha256": file_sha256(TRACK),
           "decoder_sha256": file_sha256(DEC), "reward": weights_dict(w)}
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    metrics = (out / "metrics.jsonl").open("a", encoding="utf-8")

    env_kwargs = dict(fixed_track=str(TRACK), eval_mode=True, episode_len=9000,
                      eval_protocol="pose-v1", decoder_path=str(DEC),
                      material_aux_limit=6.0, material_wheel_control="residual_v1")
    if args.start_x is not None:
        env_kwargs["material_eval_start"] = float(args.start_x)
    envs = [Go2WEnv(seed=POSE_SEED0 + i, **env_kwargs) for i in range(args.envs)]
    profile = "start_ramp_100_v1"
    ref = None
    if args.ref_weight > 0.0:
        ref = ReferenceTrajectory.from_demos(args.ref_data)
        print(f"[ref] loaded reference trajectory: {ref.n_episodes} episodes, "
              f"{len(ref.x_centers)} x-bins, w0={args.ref_weight}", flush=True)
    t0 = time.perf_counter()
    iters = 1 if args.smoke else args.iters
    eval_seeds = list(range(8700, 8700 + args.eval_seeds))

    if args.smoke:
        ep, steps = rollout(envs[0], enc, actor, critic, nm, ns, args.device, profile, w, args.goal_x,
                            ref=ref, ref_w=args.ref_weight, ref_act_w=args.ref_act_weight)
        print(f"[smoke] rollout steps={steps} success={ep.success} max_x={ep.max_x:.2f} "
              f"parts={ {k: round(v,3) for k,v in ep.parts.items()} }", flush=True)
        stat = ppo_update(enc, actor, critic, opt, [ep], nm, ns, actor_cpu, args.device, epochs=1, batch_eps=1)
        print("[smoke] update:", {k: round(v, 5) for k, v in stat.items()}, flush=True)
        save_ckpt(out / "model_smoke.pt", enc, actor, critic, ck, nm, ns, 1, arch, passthrough, w)
        print("[smoke] saved", out / "model_smoke.pt", flush=True)
        return

    for it in range(1, iters + 1):
        eps = []
        for i, env in enumerate(envs):
            ref_w_it = anneal_weight(args.ref_weight, it, args.ref_anneal_iters) if args.ref_weight > 0 else 0.0
            ep, steps = rollout(env, enc, actor, critic, nm, ns, args.device, profile, w, args.goal_x,
                                ref=ref, ref_w=ref_w_it, ref_act_w=args.ref_act_weight)
            eps.append(ep)
            print(f"  [it{it} env{i}] steps={steps} ok={ep.success} x={ep.max_x:.2f} "
                  f"rew={ep.rew.sum():.1f} lane={ep.parts['lane']:.1f} jump={ep.parts['jump']:.1f} "
                  f"energy={ep.parts['energy']:.1f}", flush=True)
        stat = ppo_update(enc, actor, critic, opt, eps, nm, ns, actor_cpu, args.device)
        n_ok = sum(1 for e in eps if e.success)
        rec = {"iter": it, "train_pass": n_ok, "envs": len(eps),
               "mean_rew": float(np.mean([e.rew.sum() for e in eps])),
               "lane": float(np.mean([e.parts["lane"] for e in eps])),
               "jump": float(np.mean([e.parts["jump"] for e in eps])),
               "energy": float(np.mean([e.parts["energy"] for e in eps])),
               "ref": float(np.mean([e.parts.get("ref", 0.0) for e in eps])),
               "ref_w": float(ref_w_it),
               **{k: round(v, 5) for k, v in stat.items()},
               "sec": round(time.perf_counter() - t0, 1)}
        metrics.write(json.dumps(rec) + "\n"); metrics.flush(); os.fsync(metrics.fileno())
        print("[iter %d] %s" % (it, json.dumps(rec)), flush=True)
        if it % 5 == 0:
            save_ckpt(out / f"model_it{it:03d}.pt", enc, actor, critic, ck, nm, ns, it, arch, passthrough, w)
        if it % args.eval_every == 0:
            save_ckpt(out / "model_latest.pt", enc, actor, critic, ck, nm, ns, it, arch, passthrough, w)
            rows = evaluate(out / "model_latest.pt", eval_seeds)
            ok = sum(1 for r in rows if r[1])
            print(f"[eval it{it}] {ok}/{len(rows)} :: {rows}", flush=True)
            metrics.write(json.dumps({"iter": it, "eval": rows, "eval_pass": ok}) + "\n")
            metrics.flush()


if __name__ == "__main__":
    main()







