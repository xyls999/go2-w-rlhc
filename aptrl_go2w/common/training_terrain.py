"""Private heightmap modifications used only by the progressive training course."""
import numpy as np

from common.terrain import load_npz_track


APPROVED_SINGLE_GAP_WIDTHS = (0.1, 0.2, 0.3, 0.4, 0.5)


def training_heightmap(track_path: str, width_m: float) -> np.ndarray:
    """Narrow the first gap only; never write to the original NPZ."""
    if width_m not in APPROVED_SINGLE_GAP_WIDTHS:
        raise ValueError("unsupported training gap width")
    H, _, _, _ = load_npz_track(track_path, apply_fixes=True)
    out = H.copy()
    keep = round(width_m / 0.1)
    out[265 + keep:270, :] = H[270, :]
    return out
