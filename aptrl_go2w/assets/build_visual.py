"""Build a visual MJCF for video recording: converts URDF DAE meshes to STL
and rebuilds go2w.xml with visual geoms (physics parameters unchanged)."""
import os
import re
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "go2w_src")
MESH_DIR = os.path.join(SRC, "meshes")
OUT_DIR = os.path.join(SRC, "meshes_stl")


def convert_meshes():
    os.makedirs(OUT_DIR, exist_ok=True)
    mapping = {}
    for f in os.listdir(MESH_DIR):
        if not f.endswith(".dae"):
            continue
        m = trimesh.load(os.path.join(MESH_DIR, f), force="mesh")
        out = os.path.join(OUT_DIR, f.replace(".dae", ".stl"))
        m.export(out)
        mapping[f] = os.path.relpath(out, SRC).replace("\\", "/")
        print("converted", f, "->", mapping[f], f"({len(m.vertices)} verts)")
    return mapping


def make_visual_urdf(mapping):
    txt = open(os.path.join(SRC, "urdf", "go2w.urdf"), encoding="utf-8").read()
    for dae, stl in mapping.items():
        txt = txt.replace(f'filename="../meshes/{dae}"', f'filename="../meshes_stl/{os.path.basename(stl)}"')
    # strip repeated materials (MuJoCo URDF parser)
    txt = re.sub(r"<material\s+name=\"[^\"]*\">\s*<color[^/]*/>\s*</material>", "", txt)
    txt = re.sub(r"<material\s+name=\"\"[^>]*/>", "", txt)
    txt = re.sub(r"<material\s+name=\"[^\"]*\"\s*/>", "", txt)
    out = os.path.join(SRC, "urdf", "go2w_visual.urdf")
    open(out, "w", encoding="utf-8").write(txt)
    return out


def main():
    mapping = convert_meshes()
    urdf = make_visual_urdf(mapping)
    print("visual urdf:", urdf)
    # build spec exactly like build_mjcf but with visual meshes kept
    import mujoco
    spec = mujoco.MjSpec.from_file(urdf)
    spec.compiler.discardvisual = False
    trunk = spec.body("trunk")
    trunk.add_joint(name="root", type=int(mujoco.mjtJoint.mjJNT_FREE))
    from build_mjcf import ALL_JOINTS, JOINT_EFFORT, home_qpos, apply_joint_tuning
    for jname in ALL_JOINTS:
        kind = "foot" if "foot" in jname else jname.split("_")[1]
        e = JOINT_EFFORT[kind]
        spec.add_actuator(name=f"act_{jname}", target=jname,
                          trntype=int(mujoco.mjtTrn.mjTRN_JOINT),
                          gear=[1.0, 0, 0, 0, 0, 0],
                          ctrlrange=[-e, e], forcerange=[-e, e])
    m0 = spec.compile()
    q0 = home_qpos(m0)
    xml = spec.to_xml()
    xml = xml.replace("<worldbody>", (
        "<worldbody>\n"
        '  <geom name="ground" type="plane" size="0 0 0.05" pos="0 0 0" '
        'friction="1.0 0.005 0.0001" condim="3"/>\n'
        '  <light directional="true" pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>\n'
    ))
    xml = xml.replace("<mujoco ", '<mujoco ').replace(
        "<compiler", '<visual><global offwidth="1920" offheight="1080"/></visual>
  <compiler')
    xml = xml.replace("<option ", (
        '<option timestep="0.002" integrator="implicitfast" cone="elliptic" '
        'impratio="2.0" iterations="30" ls_iterations="8" '
    ))
    kf = f'<key name="home" qpos="{ " ".join(f"{v:.4f}" for v in q0) }"/>\n'
    xml = xml.replace("</mujoco>", f"<keyframe>\n{kf}</keyframe>\n</mujoco>")
    out = os.path.join(HERE, "go2w_visual.xml")
    open(out, "w", encoding="utf-8").write(xml)
    print("written", out)


if __name__ == "__main__":
    main()
