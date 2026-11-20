"""Strict v4 material-course evaluation and replay helpers.

This is separate from historical NPZ evaluation and does not alter its rules.
"""
import numpy as np
import mujoco
from common.material_acceptance import TrajectoryHasher
from common.material_route_monitor import MaterialRouteMonitor, frame_overlap_top
from common.eval_protocol import material_command_speed


def _geom_id(model, name):
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if value < 0:
        raise ValueError('required robot geometry not found: ' + name)
    return value


def _contact_names(model, data):
    pairs = []
    for contact in data.contact:
        a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ''
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ''
        pairs.append((a, b))
    return pairs


def run_material_episode(env, runner, *, max_steps=9000, recorder=None):
    """Run a deterministic from-start episode; require both physics and route success."""
    if max_steps <= 0:
        raise ValueError('max_steps must be positive')
    if not env.eval_mode or env.material_track is None:
        raise ValueError('material evaluation requires a material track in eval mode')
    obs = env.reset()
    reset_ctx = getattr(getattr(runner, 'actor', None), 'reset_context', None)
    if callable(reset_ctx):
        reset_ctx()
    model, data = env.model, env.data
    wheels = [_geom_id(model, f'{leg}_foot_0') for leg in ('FL','FR','RL','RR')]
    route = MaterialRouteMonitor()
    history = TrajectoryHasher()
    def update_route():
        route.update(float(data.qpos[0]), float(data.qpos[1]), data.geom_xpos[wheels, :2],
                     _contact_names(model, data), frame_overlap_top(model, data, 44.98, 45.02))
    update_route()
    history.update(np.concatenate([data.qpos.copy(), data.qvel.copy()]))
    steps = 0
    info = {'reason': None, 'success': False}
    for step in range(1, max_steps + 1):
        x = float(data.qpos[0])
        env.vx_cmd = material_command_speed(step - 1, x, getattr(runner, 'material_command_profile', 'instant_v1'))
        # Include the newly assigned command in the same observation.
        obs = env._get_obs()
        action = np.asarray(runner.act(obs[None], True), dtype=float).reshape(-1)
        expected_actions = 30 if env.material_wheel_control == 'residual_v1' else 29
        if action.shape != (expected_actions,) or not np.isfinite(action).all():
            raise ValueError('policy returned invalid action')
        obs, _, done, info = env.step(action)
        steps = step
        update_route()
        history.update(np.concatenate([data.qpos.copy(), data.qvel.copy(), action,
                                       [float(env.vx_cmd), float(env.gait_sel)]]))
        if recorder is not None:
            recorder(env)
        if done or route.violations:
            break
    route_result = route.report()
    physics_success = bool(info.get('success', False))
    accepted = physics_success and bool(route_result['success'])
    reason = info.get('reason') or ('route_violation' if route.violations else 'evaluation_limit')
    if physics_success and not route_result['success']:
        reason = 'route_invalid'
    return {'success': accepted, 'physics_success': physics_success, 'reason': reason,
            'steps': steps, 'max_x': route_result['max_x'],
            'final_pos': [float(x) for x in data.qpos[:3]],
            'requested_initial_pose': list(env.requested_initial_pose),
            'settled_initial_pose': list(env.settled_initial_pose),
            'terminal_contacts': info.get('terminal_contacts', []),
            'route': route_result, 'trajectory_sha256': history.hexdigest(),
            'trajectory_states_hashed': history.steps}


def main(argv=None):
    """Run exactly 100 fresh-seed episodes and verify one successful video."""
    import argparse
    import json
    import os
    import subprocess
    from pathlib import Path
    import torch
    from common.material_acceptance import (compare_replay, file_sha256,
                                            generate_holdout_seeds, qualify_summary,
                                            reserve_attempt, verify_target_scene,
                                            policy_material_aux_limit,
                                            policy_material_wheel_control,
                                            policy_material_command_profile)
    from common.evaluation import build_run_manifest, write_run_manifest
    from stage3_rl.env import Go2WEnv
    from stage3_rl.evaluate import PolicyRunner, VideoRecorder

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--decoder', required=True)
    parser.add_argument('--track', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    track, ckpt, decoder, out = map(lambda p: Path(p).resolve(),
                                    (args.track, args.checkpoint, args.decoder, args.out))
    track_hash = verify_target_scene(track)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'refusing to overwrite {out}')
    # Loading the runner validates checkpoint/decoder hashes before reservation.
    runner = PolicyRunner(str(ckpt), device='cpu', decoder_path=str(decoder))
    checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
    auxiliary_limit = policy_material_aux_limit(checkpoint)
    wheel_control = policy_material_wheel_control(checkpoint)
    command_profile = policy_material_command_profile(checkpoint)
    ckpt_hash, decoder_hash = file_sha256(ckpt), file_sha256(decoder)
    claims = root / 'results/material-acceptance/claims'
    claim = reserve_attempt(claims, ckpt_hash, track_hash)
    seeds = generate_holdout_seeds()
    sources = [root / p for p in ('stage3_rl/material_acceptance.py', 'stage3_rl/env.py',
               'stage3_rl/evaluate.py', 'stage3_rl/ppo_train.py',
               'common/material_acceptance.py', 'common/material_route_monitor.py',
               'common/material_track.py', 'common/eval_protocol.py')]
    config = {'evaluator': 'material_strict_v1', 'episodes': 100, 'episode_len': 9000,
              'eval_protocol': 'pose-v1', 'seed_policy': 'fresh_unique_128bit',
              'seeds': seeds, 'device': 'cpu', 'deterministic': True,
              'decoder_sha256': decoder_hash, 'auxiliary_action_limit': auxiliary_limit,
              'material_wheel_control': wheel_control,
              'material_command_profile': command_profile,
              'normal_command_mps': 1.5, 'obstacle_command_mps': 0.8,
              'obstacle_command_x_range_m': [22, 47], 'claim': str(claim),
              'goal_x_m': 100.0, 'lane_center_m': 4.0, 'strict_lane_half_width_m': 1.9}
    manifest = build_run_manifest(checkpoint=ckpt, track=track, config=config,
                                  fixes=[], source_files=sources,
                                  track_summary={'type':'frozen_v4_material_xml', 'length_m':104.0})
    write_run_manifest(out, manifest)
    rows = []
    try:
        with (out/'episodes.jsonl').open('x', encoding='utf-8') as stream:
            for seed in seeds:
                env = Go2WEnv(seed=seed, fixed_track=str(track), eval_mode=True,
                              episode_len=9000, eval_protocol='pose-v1', decoder_path=str(decoder),
                              material_aux_limit=auxiliary_limit, material_wheel_control=wheel_control)
                row = run_material_episode(env, runner, max_steps=9000)
                row.update(seed=seed, checkpoint_sha256=ckpt_hash,
                           decoder_sha256=decoder_hash, track_sha256=track_hash)
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                print(f'{len(rows)}/100 seed={seed} success={row["success"]} '
                      f'x={row["max_x"]:.3f} reason={row["reason"]}', flush=True)
        stats = qualify_summary(rows)
        result = {'statistical_gate': stats, 'accepted': False, 'video': None}
        if stats['accepted']:
            selected = next(row for row in rows if row['success'])
            env = Go2WEnv(seed=selected['seed'], fixed_track=str(track), eval_mode=True,
                          episode_len=9000, eval_protocol='pose-v1', decoder_path=str(decoder),
                          material_aux_limit=auxiliary_limit, material_wheel_control=wheel_control)
            video = out/'verified-success.mp4'
            recorder = VideoRecorder(env, str(video))
            recorder.start()
            try:
                replay = run_material_episode(env, runner, max_steps=9000, recorder=recorder)
                replay['seed'] = selected['seed']
                matches = compare_replay(selected, replay)
                recorder.finish(matches)
                probe = subprocess.run(['ffprobe','-v','error','-show_entries',
                    'stream=codec_name,width,height,nb_frames:format=duration',
                    '-of','json',str(video)], capture_output=True, text=True, check=True)
                video_info = json.loads(probe.stdout)
                if not video_info.get('streams') or float(video_info['format']['duration']) <= 0:
                    raise ValueError('video has no decodable stream or positive duration')
                video_record = {'path': str(video), 'sha256': file_sha256(video),
                    'checkpoint_sha256': ckpt_hash, 'decoder_sha256': decoder_hash,
                    'track_sha256': track_hash, 'seed': selected['seed'],
                    'trajectory_sha256': selected['trajectory_sha256'], 'ffprobe': video_info}
                with (out/'video-manifest.json').open('x',encoding='utf-8') as stream:
                    json.dump(video_record, stream, indent=2)
                result['video'] = video_record
                result['accepted'] = True
            except BaseException:
                if recorder.active:
                    try:
                        recorder.finish(False)
                    except Exception:
                        pass
                if video.exists():
                    video.unlink()
                raise
        with (out/'summary.json').open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2)
        print(json.dumps({'accepted':result['accepted'], **stats}), flush=True)
        return 0 if result['accepted'] else 1
    except BaseException as exc:
        with (out/'error.json').open('x',encoding='utf-8') as stream:
            json.dump({'type':type(exc).__name__, 'message':str(exc),
                       'completed_episodes':len(rows), 'accepted':False},stream,indent=2)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
