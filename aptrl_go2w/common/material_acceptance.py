"""Integrity primitives for independent v4 material-course acceptance."""
import hashlib
import json
from pathlib import Path
import re
import secrets
import struct
import numpy as np

TARGET_SCENE_SHA256 = 'f459786e6372a0edeaa67cabe8afe32efd9734bcf67146b0bd1f955737a76227'
LEGACY_V4_SCENE_SHA256 = '2fbf48db989e56ef51066e8566a77dc627c87682382d1c95dfc89cc44b9e5b43'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_target_scene(path):
    actual = file_sha256(path)
    if actual != TARGET_SCENE_SHA256:
        raise ValueError('scene is not the frozen approved material course')
    return actual


class TrajectoryHasher:
    def __init__(self):
        self.digest = hashlib.sha256()
        self.steps = 0

    def update(self, values):
        values = np.asarray(values, dtype='<f8').reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError('non-finite trajectory state')
        self.digest.update(struct.pack('<I', values.size))
        self.digest.update(values.tobytes())
        self.steps += 1

    def hexdigest(self):
        return self.digest.hexdigest()


def trajectory_digest(states):
    h = TrajectoryHasher()
    for state in states:
        h.update(state)
    return h.hexdigest()


def qualify_summary(rows):
    rows = list(rows)
    seeds = [row.get('seed') for row in rows]
    unique = len(set(seeds)) == 100 if len(seeds) == 100 else False
    valid = sum(1 for row in rows if row.get('success') is True and
                row.get('physics_success') is True and
                row.get('route', {}).get('success') is True and
                not row.get('route', {}).get('violations'))
    return {'episodes': len(rows), 'unique_seeds': unique, 'valid_successes': valid,
            'accepted': len(rows) == 100 and unique and valid >= 91}


def policy_material_command_profile(checkpoint):
    """Read the versioned navigation schedule; old checkpoints used instant start."""
    profile = checkpoint.get('material_command_profile', 'instant_v1')
    if profile not in ('instant_v1', 'start_ramp_100_v1'):
        raise ValueError('unsupported material command profile')
    return profile


def policy_material_wheel_control(checkpoint):
    """Bind optional extra wheel control to its exact checkpoint architecture."""
    control = checkpoint.get('material_wheel_control', 'none')
    if checkpoint.get('actor_arch') in ('residual_v4', 'temporal_residual_v4', 'feedforward_kin_v4'):
        if control != 'residual_v1':
            raise ValueError('residual_v4 requires recorded wheel control')
    elif control != 'none':
        raise ValueError('wheel control does not match checkpoint architecture')
    return control


def policy_material_aux_limit(checkpoint):
    """Require the exact material action transform used to train a checkpoint."""
    if 'material_aux_limit' not in checkpoint:
        raise ValueError('checkpoint lacks material action-limit provenance')
    value = checkpoint['material_aux_limit']
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError('checkpoint material action limit is invalid')
    value = float(value)
    if not np.isfinite(value) or not 0 < value <= 8:
        raise ValueError('checkpoint material action limit must be in (0,8]')
    return value


def generate_holdout_seeds():
    """Generate 100 fresh, unique 128-bit independent pose seeds."""
    result = []
    used = set()
    while len(result) < 100:
        seed = secrets.randbits(128)
        if seed not in used:
            result.append(seed)
            used.add(seed)
    return result


def compare_replay(original, replay):
    """Require the same successful seed and exact complete state/action trace."""
    for row in (original, replay):
        if (row.get('success') is not True or row.get('physics_success') is not True or
                row.get('route', {}).get('success') is not True or
                row.get('route', {}).get('violations')):
            return False
    digest = original.get('trajectory_sha256')
    return (isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest) is not None and
            digest == replay.get('trajectory_sha256') and original.get('seed') == replay.get('seed') and
            isinstance(original.get('steps'), int) and original['steps'] > 0 and
            original['steps'] == replay.get('steps'))


def reserve_attempt(directory, checkpoint_sha256, scene_sha256):
    """Fail closed on a second acceptance attempt for the same artifacts."""
    for value in (checkpoint_sha256, scene_sha256):
        if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('artifact hash must be lowercase SHA-256')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{checkpoint_sha256}-{scene_sha256}.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump({'checkpoint_sha256': checkpoint_sha256, 'scene_sha256': scene_sha256}, stream)
    return path
