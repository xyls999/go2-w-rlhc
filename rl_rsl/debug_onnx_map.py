import sys
from pathlib import Path
import numpy as np
REPO = Path(r"E:\desktop-rl-apl\aptrl_go2w")
sys.path.insert(0, str(REPO)); sys.path.insert(0, r"E:\desktop-rl-apl\rl_rsl")
import collect_onnx_teacher_v5 as C
from stage3_rl.env import Go2WEnv, KP, KD, Q_DEFAULT, AUX_SCALE, WHEEL_R
from common.eval_protocol import material_command_speed
import onnxruntime as ort, mujoco

sess = ort.InferenceSession(str(C.ONNX), providers=["CPUExecutionProvider"])
iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
env = Go2WEnv(seed=8800, fixed_track=str(C.TRACK), eval_mode=True, episode_len=9000,
              eval_protocol="pose-v1", decoder_path=str(C.DEC), material_aux_limit=6.0,
              material_wheel_control="residual_v1")
m, d = env.model, env.data
obs = env.reset()
tob = C.TeacherObs(); last_a = np.zeros(16, np.float32)
for st in range(200):
    env.vx_cmd = material_command_speed(st, float(d.qpos[0]), "start_ramp_100_v1")
    R = d.xmat[env.body_root].reshape(3, 3)
    yaw = float(np.arctan2(R[1,0], R[0,0])); ey = float(d.qpos[1]) - 4.0
    cmd = np.array([env.vx_cmd, float(np.clip(1.0*ey,-0.5,0.5)), float(np.clip(2.0*ey+1.5*yaw,-0.8,0.8))], np.float32)
    t_obs = tob.build(ang=d.qvel[3:6], quat=d.qpos[3:7], command=cmd, elapsed=st*0.02,
                      leg_pos=d.qpos[env.leg_qadr], leg_vel=d.qvel[env.leg_vadr],
                      wheel_vel=d.qvel[env.wheel_vadr], action=last_a)
    a_t = sess.run([oname], {iname: t_obs})[0].astype(np.float32).reshape(-1)
    last_a = a_t
    q_des_t = C.ACTION_DEFAULT_Q_T + 0.5 * a_t[:12]
    q_now = d.qpos[env.leg_qadr]; dq_now = d.qvel[env.leg_vadr]
    tau_t = C.KP_T * (q_des_t - q_now) - C.KD_T * dq_now
    aux = np.clip((tau_t + KD*dq_now + KP*(q_now - Q_DEFAULT)) / (AUX_SCALE*KP), -6.0, 6.0)
    wa = float(np.clip(np.mean(35*a_t[12:]) / max(env.vx_cmd/WHEEL_R, 0.5) - 1.0, -1.0, 0.5))
    act = np.concatenate([C.Z_TROT, aux, [-8.0, wa]]).astype(np.float32)
    if st % 10 == 0:
        print(f"st={st} q_des_t[:3]={np.round(q_des_t[:3],2)} q_now[:3]={np.round(q_now[:3],2)} "
              f"tau_t[:3]={np.round(tau_t[:3],1)} aux[:3]={np.round(aux[:3],2)} max|aux|={np.abs(aux).max():.2f} wa={wa:+.2f} z={d.qpos[2]:.3f}")
    obs, r, done, info = env.step(act)
    if done:
        print("DONE", st, info.get("reason"), "terminal:", [c["geom1"]+":"+c["geom2"] for c in info.get("terminal_contacts",[])[:3]])
        break

