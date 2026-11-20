# Go2-W Progressive Gap Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue PPO from v21 using auditable flat, narrow-gap, original double-gap, mixed, and original full-track training stages without inserting the manually discovered pulse into learned actions.

**Architecture:** Keep the official fixed-track evaluator as the referee. Build train-only heightmaps in memory from a copy of the authorized NPZ, change only the first pit's longitudinal width for single-gap stages, and use the unmodified NPZ heightmap for double-gap/full-track stages. Sequential isolated PPO runs warm-start from the last promoted checkpoint; a separate deterministic local gate uses 10 seed-distinct starts, while original-track probes detect forgetting.

**Tech Stack:** Python 3, NumPy, MuJoCo, PyTorch, multiprocessing spawn, unittest; Windows-compatible commands.

**Spec:** `docs/superpowers/specs/2026-11-09-go2w-progressive-gap-curriculum-design.md`

## Global Constraints

- Execute `docs/superpowers/plans/2026-11-09-evaluation-safety-prerequisites.md` and save passing tests/seed-parity evidence **before** this plan's PPO pilot.
- Original NPZ SHA-256 `7876f2173e54bf73db2d2feceb0c449a2fc219298108a141a6fe6b313eb3c18f` stays immutable; original finish `x >= 100m`, strict `|y-4| < 1.9m`; final >=91/100 under frozen `pose-v1` protocol.
- Warm start `stage3_rl/runs_v21/model_latest.pt` SHA-256 `a9a6dab12c2417e15b105ebe7d7eb0f8c4c5ba07d01ee8fa237c88c42a42e43c` stays immutable.
- Stage 1/2 decoders, action space, torque limits, failure thresholds, reward and official `load_npz_track` repairs stay unchanged. No manual pulse is used for training or acceptance.
- Every run has a new empty output path; no overwriting `results/eval_summary.json`, `results/eval_best_traj.npz`, old checkpoints or videos. This directory is not a Git repo, so save logs/hashes instead of attempting commits.

## File responsibilities

- Create `common/training_terrain.py`: pure train-only gap narrowing; never write to the original NPZ and never alter official `load_npz_track`.
- Modify `stage3_rl/env.py`: explicit train-stage selection, local spawn/success overrides, local-gate seed jitter; retain default `Go2WEnv` behavior.
- Modify `stage3_rl/ppo_train.py`: CLI/worker propagation of stage, gap width, train track, episode cap; run manifest before workers.
- Create `stage3_rl/curriculum_gate.py`: checkpoint-only, deterministic ten-seed local pass measurement; no script-injected control.
- Create `tests/test_training_terrain.py`, `tests/test_training_profiles.py`, `tests/test_curriculum_gate.py`: independent geometry, environment, and gate tests.

---

### Task 1: Train-only profiles and immutable geometry

**Files:** Create `common/training_terrain.py`, `tests/test_training_terrain.py`, `tests/test_training_profiles.py`; modify `stage3_rl/env.py`.

**Interfaces:** `training_heightmap(track_path: str, width_m: float) -> np.ndarray` loads a new corrected heightmap copy; `Go2WEnv(..., train_stage: str="mixed", gap_width_m: float|None=None, train_track: str|None=None)` accepts `flat`, `rough_flat`, `single_gap`, `double_gap`, `mixed`, `full_track`. `single_gap` uses `train_track` and widths 0.1,0.2,0.3,0.4,0.5; `double_gap`/`full_track` load its exact original map. Local spawn/success `(24.5,28.3)` for single, `(24.5,32.0)` for double; the 0.5m inset avoids settling backward across the x=24 segment seam; mixed/flat/rough retain their existing strip success distance. `train_stage="mixed"` with no train track preserves current default behavior.

```python
def training_heightmap(track_path: str, width_m: float) -> np.ndarray:
    if width_m not in (0.1, 0.2, 0.3, 0.4, 0.5):
        raise ValueError("unsupported training gap width")
    H, _, _, _ = load_npz_track(track_path, apply_fixes=True)
    out = H.copy()
    out[265 + round(width_m / 0.1):270, :] = H[270, :]
    return out

assert np.array_equal(training_heightmap(track, 0.5), original_H)
```

- [x] **Step 1: Write failing geometry tests.** Read original H through `load_npz_track`. For each width `w`, assert the training copy has rows `265:265+round(w/.1)` identical to original first-gap rows, rows `265+round(w/.1):270` replaced by a copy of original row 270, all rows outside `265:270` unchanged, all source files unchanged by SHA-256, and `training_heightmap(...,.5)` exactly equals H. Verify original second gap rows `300:304` and side-pit cells remain original. Reject widths other than the five exact configured values.
- [x] **Step 2: Confirm red.** Run `python -m unittest tests.test_training_terrain -v`; expect missing function.
- [x] **Step 3: Implement heightmap copy.** Use `H,_,_,_=load_npz_track(track_path,apply_fixes=True)`, `out=H.copy()`, `keep=int(round(width_m/.1))`, and `out[265+keep:270,:]=H[270,:]`. Require exact membership in `(0.1,0.2,0.3,0.4,0.5)`; return `out`, never save it back to NPZ. Preserve original mesh boxes returned by the same loader.
- [x] **Step 4: Write red environment tests.** Construct a single-gap env at 0.1 and 0.5m, a double-gap env and default v21-compatible env; assert centerline pit width/depth, both original double-gap rows, `success_x`=28.3/32.0 versus fixed-track 100.0, and unchanged `leg_effort`/wheel actuator ranges. Assert local gate seeds 9000/9001 have distinct initial pose, a repeated seed is identical, and neither `eval_mode=True` nor `train_stage` can change official `Go2WEnv(seed=9000,fixed_track=track,eval_mode=True)` geometry or success rule.
- [x] **Step 5: Implement stage environment.** For `flat` and `rough_flat`, feed the same named `build_segment` for both 8m strips (not random `TERRAIN_TYPES`), disable level rotation in `reset/report`. For single/double, load `train_track` through the official loader, make a private copy of H, use a local-training marker separate from official `fixed_track`, override start/success only for that marker, and apply the existing randomization only when `eval_mode=False`. For local `eval_mode=True` gate, derive `x=24.5±.05, y=4±.1, yaw=±.02` deterministically from seed and log both requested/settled pose. In `report()`, count local training episodes/success under a stable stage name without altering terrain level; existing mixed adaptive reports remain unchanged. Snapshot nominal geom friction on each model build and compute `nominal_friction * current_friction_gain` on every reset, because cached local-track models otherwise compound friction across episodes; lock this with a repeated-reset test. Never use the final `pose-v1` start x=0.5 for local gates.
- [x] **Step 6: Confirm green.** Run the two new unit modules plus `tests.test_track_spec` and `tests.test_track_acceptance`; compare pre/post NPZ and v21 checkpoint hashes. Save a `training_geometry.json` with width, first-gap row range, unchanged second gap, and source hash to the new experiment directory.

### Task 2: Stage-aware PPO wiring and provenance

**Files:** Modify `stage3_rl/ppo_train.py`; create `tests/test_training_cli.py`.

**Interfaces:** CLI flags `--train-stage`, `--train-track`, `--gap-width`, `--episode-len`; `Learner(..., train_stage="mixed", train_track=None, gap_width_m=None, episode_len=600)` passes config through `worker_main` to `Go2WEnv`. New empty output directory contains `run_config.json` with stage, width, `--init` hash, `--probe-track` hash, train-track hash, seed, batch/worker counts, eval protocol, max iteration, and source file hashes.

```python
def worker_env_config(args):
    return dict(train_stage=args.train_stage,
                train_track=args.train_track,
                gap_width_m=args.gap_width,
                episode_len=args.episode_len)

# In worker_main, pass only these explicit settings to Go2WEnv:
env = Go2WEnv(seed=worker_seed, eval_mode=False, **stage_config)
```

- [x] **Step 1: Write red CLI/config tests.** Check parser rejects `single_gap` without `--train-track`/`--gap-width`, `double_gap` without `--train-track`, non-member gap widths, `full_track` with `--episode-len` below 9000, and a nonexistent or unreadable train-track path. Check valid `single_gap --gap-width 0.1 --train-track <original> --episode-len 900` creates a worker config that passes `train_stage="single_gap"`, `gap_width_m=.1`, `train_track=<original>` without starting processes; a new directory refuses reuse.
- [x] **Step 2: Confirm red.** Run `python -m unittest tests.test_training_cli -v`; expect missing CLI/config functions.
- [x] **Step 3: Implement the boundary.** Add pure `validate_training_args(args)` and `worker_env_config(args)` helpers; call validation before `ensure_empty_output_dir`. Pass explicit stage settings in `Learner` worker config and construct envs with `train_stage`, `gap_width_m`, `train_track`. Full-track training uses original `train_track`, explicit `--episode-len 16000`; the normal `--probe-track` remains the original, independent path. No actor/critic, sampling distribution, reward, or action dimension changes.
- [x] **Step 4: Save auditable run config.** After output-dir emptiness check and before worker startup, save `run_config.json` with source/checkpoint/track hashes and exact CLI values using exclusive file creation. Snapshot first probe using the safety plan's fixed `pose-v1` seed policy. Smoke `python stage3_rl/ppo_train.py --help` and run `python -m unittest tests.test_training_cli tests.test_ppo_probe_config tests.test_probe_cadence -v`.

### Task 3: Measured promotion rather than automatic long training

**Files:** Create `stage3_rl/curriculum_gate.py`, `tests/test_curriculum_gate.py`.

**Interfaces:** `evaluate_gate(ckpt, track, train_stage, gap_width_m, seeds=range(9000,9010)) -> dict` runs deterministic PPO actions via `PolicyRunner`, returns per-episode `{seed,max_x,reason,success,landed_after_lip,requested_initial_pose,settled_initial_pose}`; `should_promote(gate, prior_full_median_x) -> bool` requires 8/10 local success AND current full-track `pose-v1` median max-x not below prior baseline. Original-track result comes from an explicit run manifest, not inferred from local results.

```python
landed = [False]
def contact_callback(env):
    if float(env.data.qpos[0]) >= lip_x and np.any(env._contacts()[0] > 0):
        landed[0] = True

result = run_episode(env, runner, deterministic=True, recorder=contact_callback)
passed = (result["success"] and result["reason"] == "success"
          and result["max_x"] >= finish_x and landed[0])
```

- [x] **Step 1: Write red gate tests.** Use fake episode rows to assert 8/10 passes, 7/10 fails, local success without after-lip wheel contact fails for double-gap, off-lane/body-contact fails, full-track median regression fails. Check seed list is exactly ten distinct values and 100% reproducible. Reject using a manually scripted action source.
- [x] **Step 2: Confirm red.** Run `python -m unittest tests.test_curriculum_gate -v`; expect missing gate module.
- [x] **Step 3: Implement the gate.** Construct local `Go2WEnv(train_stage=...,train_track=...,eval_mode=True,episode_len=1600)` for single/double; run `run_episode(...,deterministic=True,recorder=contact_callback)` and set `landed_after_lip` when x>=27.2 (single) or x>=30.6 (double) and any entry of `env._contacts()[0]` is positive. Require `reason="success"`, correct local finish and contact; save ten individual rows and a summary in a new directory with checkpoint/training-heightmap/original-track hashes. For flat/rough, use the named strips; for full-track, only the official `pose-v1` evaluator applies.
- [x] **Step 4: Confirm green.** Run all new tests and one v21 local baseline into a new diagnostic directory; record actual 10/10-style results even if 0/10. Never claim local pass from the human feasibility witness.

### Task 4: Bounded execution and original-track acceptance

**Files:** Create new `stage3_rl/experiments/<UTC>/stage-decisions.jsonl` and `stage3_rl/runs_curriculum_<UTC>_<stage>/` per attempted stage. Modify `REPORT.md` only after an actual acceptance run.

**Interfaces:** Consume all passing tests, the frozen `pose-v1` protocol, v21 baseline and gate records. Produce immutable short-run checkpoints and explicit promote/stop decisions.

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
python -m unittest discover -s tests -p 'test_*.py' -v
python stage3_rl/evaluate_par.py --ckpt stage3_rl/runs_v21/model_latest.pt \
  --track "C:/Users/Administrator/Desktop/go2w_race_track_seed50001.npz" \
  --eval-protocol pose-v1 --episodes 10 --workers 2 \
  --out "results/diagnostics/$RUN_ID/v21-pose-v1"
```

- [x] **Step 1: Save v21 baseline under the new protocol.** Run `python stage3_rl/evaluate_par.py --ckpt stage3_rl/runs_v21/model_latest.pt --track "C:/Users/Administrator/Desktop/go2w_race_track_seed50001.npz" --eval-protocol pose-v1 --episodes 10 --workers 2 --out results/diagnostics/<UTC>/v21-pose-v1`. Store full median x, per-seed initial states, failure causes. Also run `stage3_rl/curriculum_gate.py` for flat, rough-flat and single-gap width .1; if v21 already passes a stage, skip that stage's training and record the checkpoint/decision.
- [x] **Step 2: Start only a 25-iteration pilot from v21.** If the .1m gate fails, create a new directory and use `--init stage3_rl/runs_v21/model_latest.pt --train-stage single_gap --gap-width 0.1 --train-track "C:/Users/Administrator/Desktop/go2w_race_track_seed50001.npz" --probe-track <same path> --eval-protocol pose-v1 --episode-len 900 --batch-target 4096 --workers 4 --epw 1 --epochs 2 --iters 25`. Save full stdout/stderr log. Confirm `model_best.pt`, `model_it00025.pt`, `probe_metrics.jsonl`, and `run_config.json` exist; any missing probe/checkpoint is a failed pilot, not a reason to evaluate an unrelated file. The first 0.2m pilot predating the friction fix is labeled INVALID; use a fresh run directory and v21 init, never its checkpoints.
- [x] **Step 3: Gate and decide.** Evaluate the pilot `model_best.pt` and v21 on the same ten local seeds; compare their new full-track `pose-v1` 10-seed medians and failure rows. Append checkpoint hashes and decision to `stage-decisions.jsonl`. If no local gain or earlier full-track failure, stop and diagnose. If eligible, repeat an isolated 25-iteration stage at widths .2,.3,.4,.5, then original double gap, then mixed, then original full track; skip any stage already passing its gate on the incoming checkpoint. Each stage starts only after prior gate >=8/10, full-track median non-regression and user-visible saved evidence. No silent 100-iteration extensions.
- [ ] **Step 4: Final 100-episode gate only for a promoted candidate.** Use `evaluate_par.py --eval-protocol pose-v1 --episodes 100 --workers 16` into a new `results/acceptance/<UTC>` with a fresh video path. Require at least 91 rows with `x>=100m`, strict lane and no failure, and check the selected successful seed replays with identical steps/max x and nonempty ffprobe video. If below target, keep goal open and do not mark `REPORT.md` as success; report gap and failure distribution instead.

## Self-review and execution boundary

Coverage: geometric training stages, seed-distinct local gates, 25-iteration isolation, original-track anti-forgetting, final 100-run acceptance and replay each have a dedicated task. Run all tests before starting workers. Do not treat human pulses, training-only geometry or repeated legacy seed rollouts as learned success. Stop at the first failed gate rather than silently training longer or weakening the original 104m track.

## Execution checkpoint (2026-11-10)

- Passing regression output: `results/diagnostics/20261109T135158Z/tests-postpilot-full.log` (49 tests). Original NPZ and v21 checkpoint hashes rechecked unchanged.
- Corrected v21 local gates: flat 10/10, rough-flat 10/10, 0.1m single gap 8/10, 0.2m single gap 0/10. Earlier flat and 0.1m gate directories are explicitly marked INVALID because of training-only collision normalization / x=24 segment-seam spawn respectively.
- The first 0.2m pilot completed before a cached-model friction accumulation bug was discovered; its directory is marked INVALID and is not used for comparison. The corrected 25-iteration pilot is `stage3_rl/runs_curriculum_20261109T135158Z_single0p2_frictionfixed/`; its first probe and isolated checkpoint were verified.
- Corrected pilot local 0.2m gate: 5/10; original-track ten-seed median max-x: v21 16.804418713455846m versus pilot 18.007809253038264m, but three paired seeds regressed by over 5m. Evidence and non-promotion decision: `stage3_rl/experiments/20261109T135158Z/stage-decisions.jsonl`.
- Task 4 Step 4 is intentionally open: no candidate has passed the 8/10 local gate, so no 100-episode acceptance or success video has been attempted. Do not start another training stage without a revised, explicit decision on the regressions.
