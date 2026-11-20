"""Convert Go2-W URDF to a torque-controlled MuJoCo MJCF.

- floating base (freejoint)
- 16 motors (12 leg + 4 wheel), ctrlrange = URDF effort limits
- named geoms (contact reporting), ground plane, home keyframe
- stance: hip 0 / thigh 0.9 / calf -1.8, standing trunk height ~0.36 m
- qpos actuator order follows URDF joint order (legs interleaved with wheels)
"""
import os
import re
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "go2w_src", "urdf", "go2w.urdf")

LEGS = ["FL", "FR", "RL", "RR"]
LEG_JOINTS = [f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf")]
WHEEL_JOINTS = [f"{leg}_foot_joint" for leg in LEGS]
ALL_JOINTS = []  # URDF order: per-leg hip, thigh, calf, wheel
for leg in LEGS:
    ALL_JOINTS += [f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint",
                   f"{leg}_foot_joint"]

JOINT_EFFORT = {"hip": 23.7, "thigh": 23.7, "calf": 35.55, "foot": 23.7}
WHEEL_RADIUS = 0.086
STANCE = {"hip": 0.0, "thigh": 0.9, "calf": -1.8, "foot": 0.0}


def clean_urdf() -> str:
    txt = open(URDF, encoding="utf-8").read()
    txt = re.sub(r"<material\s+name=\"[^\"]*\">\s*<color[^/]*/>\s*</material>", "", txt)
    txt = re.sub(r"<material\s+name=\"\"[^>]*/>", "", txt)
    txt = re.sub(r"<material\s+name=\"[^\"]*\"\s*/>", "", txt)
    tmp = os.path.join(HERE, "go2w_src", "urdf", "go2w_clean.urdf")
    open(tmp, "w", encoding="utf-8").write(txt)
    return tmp


def build_spec():
    spec = mujoco.MjSpec.from_file(clean_urdf())
    trunk = spec.body("trunk")
    trunk.add_joint(name="root", type=int(mujoco.mjtJoint.mjJNT_FREE))
    for jname in ALL_JOINTS:
        kind = "foot" if "foot" in jname else jname.split("_")[1]
        e = JOINT_EFFORT[kind]
        spec.add_actuator(name=f"act_{jname}", target=jname,
                          trntype=int(mujoco.mjtTrn.mjTRN_JOINT),
                          gear=[1.0, 0, 0, 0, 0, 0],
                          ctrlrange=[-e, e], forcerange=[-e, e])
    for b in spec.bodies:
        for i, g in enumerate(b.geoms):
            if not g.name:
                g.name = f"{b.name}_{i}"
    return spec


def apply_joint_tuning(m: mujoco.MjModel):
    """Leg damping + wheel frictionloss/damping (dof-level, by name)."""
    for jname in LEG_JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        m.dof_damping[m.jnt_dofadr[jid]] = 0.1
    for jname in WHEEL_JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        adr = m.jnt_dofadr[jid]
        m.dof_damping[adr] = 0.02
        m.dof_frictionloss[adr] = 0.8


def home_qpos(m: mujoco.MjModel, h: float = None):
    """qpos vector (23,) in model order: freejoint + URDF-ordered joints."""
    if h is None:
        h = standing_height(m)
    q = np.zeros(m.nq)
    q[2], q[3] = h, 1.0
    for jname in ALL_JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        part = "foot" if "foot" in jname else jname.split("_")[1]
        q[m.jnt_qposadr[jid]] = STANCE[part]
    return q


def standing_height(m: mujoco.MjModel) -> float:
    d = mujoco.MjData(m)
    for jname in ALL_JOINTS:
        part = "foot" if "foot" in jname else jname.split("_")[1]
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        d.qpos[m.jnt_qposadr[jid]] = STANCE[part]
    d.qpos[2] = 0.5
    mujoco.mj_forward(m, d)
    foot_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot") for leg in LEGS]
    return 0.5 - np.mean([d.xpos[b][2] for b in foot_ids]) + WHEEL_RADIUS


def main():
    spec = build_spec()
    h = standing_height(spec.compile())
    print(f"stance {STANCE} -> trunk h={h:.3f}")
    q0 = home_qpos(spec.compile(), h)

    xml = spec.to_xml()
    xml = xml.replace("<worldbody>", (
        "<worldbody>\n"
        '  <geom name="ground" type="plane" size="0 0 0.05" pos="0 0 0" '
        'friction="1.0 0.005 0.0001" condim="3"/>\n'
        '  <light directional="true" pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>\n'
    ))
    xml = xml.replace("<option ", (
        '<option timestep="0.002" integrator="implicitfast" cone="elliptic" '
        'impratio="2.0" iterations="30" ls_iterations="8" '
    ))
    kf_pos = f'<key name="home" qpos="{ " ".join(f"{v:.4f}" for v in q0) }"/>\n'
    xml = xml.replace("</mujoco>", f"<keyframe>\n{kf_pos}</keyframe>\n</mujoco>")
    out = os.path.join(HERE, "go2w.xml")
    open(out, "w", encoding="utf-8").write(xml)

    # verify: settle with PD hold
    m = mujoco.MjModel.from_xml_path(out)
    apply_joint_tuning(m)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    kp = np.array([80.0, 100.0, 120.0] * 4)
    kd = 3.5
    qd = q0[7:].copy()
    # actuators were added in ALL_JOINTS order -> actuator k <-> ALL_JOINTS[k]
    leg_act = [k for k, jn in enumerate(ALL_JOINTS) if "foot" not in jn]
    leg_qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                for jn in ALL_JOINTS if "foot" not in jn]
    leg_vadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                for jn in ALL_JOINTS if "foot" not in jn]
    effort = np.array([JOINT_EFFORT["hip" if "hip" in jn else
                                    ("thigh" if "thigh" in jn else "calf")]
                       for jn in ALL_JOINTS if "foot" not in jn])
    for i in range(3000):
        mujoco.mj_forward(m, d)
        q = d.qpos[leg_qadr]
        vq = d.qvel[leg_vadr]
        tau = np.clip(kp * (qd[leg_act] - q) - kd * vq, -effort, effort)
        d.ctrl[leg_act] = tau
        mujoco.mj_step(m, d)
    print("settle: trunk z=%.3f pitch=%.3f" % (d.qpos[2], d.qpos[4]))
    print("written", out)


if __name__ == "__main__":
    main()
