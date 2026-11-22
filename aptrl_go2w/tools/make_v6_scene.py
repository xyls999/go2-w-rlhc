"""生成 v6 赛道（在 v5 基础上）：

1) 30.0-44.0m 的高差取消：删除 5 级上台阶、上平台、上下引桥、5 级下台阶、下平台，
   30-44m 段全部改为平地（保留轮胎、砖块、独木桥、桥侧墙、45m 钻框）。
2) 独木桥保留 3m 长，但宽度由 0.40m 放宽到 1.00m（横向控制精度限制）。
3) 桥侧墙缺口由 0.70m 放宽到 1.20m。

其余几何与 v5 完全一致。
"""
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path(r"E:\desktop-rl-apl\aptrl_go2w\results\diagnostics\robocon-material-preview-20261119-v5\scene.xml")
DST_DIR = Path(r"E:\desktop-rl-apl\aptrl_go2w\results\diagnostics\robocon-material-preview-20261119-v6")
DST = DST_DIR / "scene.xml"

DROP = ([f"stair_up_{i}" for i in range(5)]
        + [f"stair_down_{i}" for i in range(5)]
        + ["up_platform", "up_exit_ramp", "down_platform", "down_approach_ramp"])


def main():
    tree = ET.parse(SRC)
    root = tree.getroot()
    wb = root.find("worldbody")
    removed = []
    for g in list(wb.findall(".//geom")):
        if g.get("name") in DROP:
            for parent in wb.iter():
                if g in list(parent):
                    parent.remove(g)
                    removed.append(g.get("name"))
                    break
    for g in wb.findall(".//geom"):
        n = g.get("name")
        if n == "balance_beam":
            x = float(g.get("pos").split()[0])
            g.set("pos", "%.2f 4.0 0.015" % x)
            g.set("size", "1.5 1.9 0.015")
        elif n == "bridge_barrier_left":
            g.set("pos", "35.2 0.90 0.35")
        elif n == "bridge_barrier_right":
            g.set("pos", "35.2 7.10 0.35")
    DST_DIR.mkdir(parents=True, exist_ok=True)
    tree.write(DST, encoding="utf-8", xml_declaration=False)
    print("removed", len(removed), "->", DST)


if __name__ == "__main__":
    main()
