"""Standalone 2027-inspired material course; preview, not a training track."""
import math
import random
import xml.etree.ElementTree as ET


def build_scene():
    root = ET.Element('mujoco', model='material_track_preview')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep='0.002')
    ET.SubElement(root, 'visual').append(ET.Element('global', offwidth='1280', offheight='720'))
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='52 4 25', dir='0 0 -1', directional='true', diffuse='0.8 0.8 0.8')
    def geom(name, pos, size, rgba, **kw):
        return ET.SubElement(world, 'geom', name=name, type='box', pos=pos, size=size, rgba=rgba, **kw)
    geom('track_floor', '52 4 -0.1', '52 4 0.1', '0.33 0.37 0.36 1')
    for y in (2.05, 5.95):
        geom('lane_mark_' + str(y), f'52 {y} 0.002', '52 0.025 0.002', '0.88 0.82 0.22 1', contype='0', conaffinity='0')
    # A pair of stair/platform profiles: the 0.32 m tread is actual box geometry.
    for prefix, start in [('stair_up', 31.0), ('stair_down', 41.05)]:
        for i in range(5):
            h = 0.12 * (i + 1 if prefix == 'stair_up' else 5 - i)
            geom(f'{prefix}_{i}', f'{start + 0.16 + i * 0.32} 4 {h/2}', f'0.16 0.4 {h/2}', '0.65 0.56 0.4 1')
        # Ramps avoid introducing an unsupported vertical drop at either end.
        if prefix == 'stair_up':
            geom('up_platform', f'{start + 2.0} 4 0.3', '0.4 0.4 0.3', '0.65 0.56 0.4 1')
            geom('up_exit_ramp', f'{start + 3.225} 4 0.3', '0.878 0.4 0.02', '0.53 0.48 0.37 1', euler='0 0.349066 0')
        else:
            geom('down_approach_ramp', f'{start - 1.625} 4 0.3', '0.878 0.4 0.02', '0.53 0.48 0.37 1', euler='0 -0.349066 0')
            geom('down_platform', f'{start - 0.4} 4 0.3', '0.4 0.4 0.3', '0.65 0.56 0.4 1')
    # Eight fixed hollow tire rings, each constructed from 32 tangent boxes.
    for row in range(4):
        for col in range(2):
            cx, cy = 25.0 + row * 0.55, 3.725 + col * 0.55
            for k in range(32):
                a = 2 * math.pi * k / 32
                x, y = cx + .22 * math.cos(a), cy + .22 * math.sin(a)
                geom(f'tire_{row}_{col}_{k}', f'{x:.8f} {y:.8f} 0.0825', '0.055 0.022 0.0825', '0.075 0.075 0.08 1', euler=f'0 0 {a:.8f}', friction='1.1 0.005 0.0001')
    geom('brick_pit_base', '28.5 4 0.015', '1 1 0.015', '0.36 0.28 0.22 1')
    rng = random.Random(2027)
    # L-shaped 2x2 m preview bed: visible fixed bricks and bark, not loose physics.
    for i in range(90):
        x, y = rng.uniform(-.94, .94), rng.uniform(-.94, .94)
        if x > 0 and y > 0: continue
        h = rng.uniform(.01, .035)
        color = '0.62 0.27 0.16 1' if i % 3 == 0 else '0.32 0.21 0.13 1'
        geom(f'brick_bark_{i}', f'{28.5+x:.4f} {4+y:.4f} {0.03+h/2:.4f}', f'{rng.uniform(.025,.06):.4f} {rng.uniform(.025,.07):.4f} {h/2:.4f}', color, euler=f'0 0 {rng.uniform(-3.14,3.14):.4f}')
    geom('balance_beam', '36.7 4 0.05', '1.5 0.2 0.05', '0.55 0.37 0.20 1')
    # Entrance wings span the floor outside the beam; leaving the course itself
    # remains possible in this standalone preview and needs a lane-failure rule.
    geom('bridge_barrier_left', '35.2 1.9 0.35', '0.04 1.9 0.35', '0.32 0.40 0.48 1')
    geom('bridge_barrier_right', '35.2 6.1 0.35', '0.04 1.9 0.35', '0.32 0.40 0.48 1')
    # Provisional clearance only; no claim that the robot can crouch under it.
    for y in (3.59, 4.41):
        geom('frame_post_'+str(y), f'45 {y} 0.14', '0.02 0.02 0.14', '0.7 0.7 0.73 1')
    geom('clearance_top', '45 4 0.27', '0.02 0.43 0.02', '0.72 0.72 0.76 1')
    return ET.tostring(root, encoding='unicode')
