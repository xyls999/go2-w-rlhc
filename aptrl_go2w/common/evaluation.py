"""Helpers for immutable, auditable evaluation runs."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import tempfile

import numpy as np

from common.terrain import HS


def sha256_file(path: str) -> str:
    """Return a lowercase SHA-256 digest, reading large artifacts in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _file_record(path):
    real_path = os.path.realpath(os.fspath(path))
    return {"path": real_path, "sha256": sha256_file(real_path)}


def build_run_manifest(*, checkpoint, track, config, fixes, source_files,
                        track_summary) -> dict:
    """Build JSON-safe provenance for a single immutable evaluation run."""
    if isinstance(source_files, (str, os.PathLike)):
        source_files = [source_files]
    elif isinstance(source_files, dict):
        source_files = list(source_files.keys())

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": _file_record(checkpoint),
        "track": _file_record(track),
        "evaluation": _json_safe(config),
        "fixes": _json_safe(list(fixes)),
        "runtime": {
            "python": __import__("platform").python_version(),
            "numpy": np.__version__,
            "torch": _package_version("torch"),
            "mujoco": _package_version("mujoco"),
        },
        "source_files": [_file_record(path) for path in source_files],
        "track_summary": _json_safe(track_summary),
    }
    return manifest


def summarize_track(H, boxes, seq) -> dict:
    """Summarize each equal-length track segment and obstacle box bounds."""
    heightmap = np.asarray(H)
    if heightmap.ndim != 2 or not seq:
        raise ValueError("H must be a 2D heightmap and seq must be non-empty")
    if heightmap.shape[0] % len(seq):
        raise ValueError("heightmap rows must divide evenly across sequence")
    rows_per_segment = heightmap.shape[0] // len(seq)
    segments = []
    for index, name in enumerate(seq):
        start = index * rows_per_segment
        section = heightmap[start:start + rows_per_segment]
        segments.append({
            "index": index,
            "name": str(name),
            "start_x_m": start * HS,
            "end_x_m": (start + rows_per_segment) * HS,
            "min_m": float(section.min()),
            "max_m": float(section.max()),
        })
    return {
        "sequence": [str(name) for name in seq],
        "length_m": float(heightmap.shape[0] * HS),
        "width_m": float(heightmap.shape[1] * HS),
        "segments": segments,
        "obstacle_boxes": _json_safe(list(boxes)),
    }


def write_run_manifest(out_dir, manifest) -> str:
    """Atomically write manifest.json into a new or empty run directory.

    Existing non-empty run directories are immutable and are never reused.
    """
    out_dir = os.path.abspath(os.fspath(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    if os.listdir(out_dir):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {out_dir}")

    output_path = os.path.join(out_dir, "manifest.json")
    payload = json.dumps(_json_safe(manifest), indent=2, sort_keys=True,
                         ensure_ascii=False) + "\n"
    fd, temp_path = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, output_path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise
    return output_path
