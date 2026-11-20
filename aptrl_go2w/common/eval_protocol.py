"""Explicit, seed-reproducible initial-pose protocols for fixed-track evaluation."""
import numpy as np


POSE_V1_RANGES = {
    "x_m": [0.45, 0.55],
    "y_m": [3.9, 4.1],
    "yaw_rad": [-0.02, 0.02],
}


def material_command_speed(step, x, profile='instant_v1'):
    """Versioned course command; obstacle speed remains unchanged."""
    if profile not in ('instant_v1', 'start_ramp_100_v1') or step < 0 or not np.isfinite(x):
        raise ValueError('invalid material command profile or state')
    if profile == 'start_ramp_100_v1' and step < 100 and x < 22:
        return 0.5 + float(step) / 100.0
    return 0.8 if 22 <= x < 47 else 1.5


def initial_pose(seed: int, protocol: str) -> tuple[float, float, float]:
    """Return requested x/y/yaw without consuming an environment RNG."""
    if protocol == "legacy":
        return 0.5, 4.0, 0.0
    if protocol != "pose-v1":
        raise ValueError(f"unknown evaluation protocol: {protocol}")
    rng = np.random.default_rng(int(seed) ^ 0x5A17E)
    return (0.5 + float(rng.uniform(-0.05, 0.05)),
            4.0 + float(rng.uniform(-0.10, 0.10)),
            float(rng.uniform(-0.02, 0.02)))
