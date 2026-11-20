# Fixed-Track Evaluation Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fixed-track comparisons genuinely seed-distinct, guarantee a 25-iteration pilot emits a probe, and make video replay physically identical to unrecorded evaluation.

**Architecture:** Add an explicit opt-in `pose-v1` evaluation protocol shared by serial, parallel, and training probes; preserve legacy defaults for historical comparability. Isolate the seed-to-pose map in one pure module; fix probe cadence as a pure helper; remove the recorder's redundant `mj_forward` call and test replay parity. Do not change track geometry, success rule, PPO reward, motor limits, or existing artifacts.

**Tech Stack:** Python 3, NumPy, MuJoCo, PyTorch, unittest; Windows paths supported.

**Spec:** `docs/superpowers/specs/2026-11-09-go2w-progressive-gap-curriculum-design.md`

## Global Constraints

- Original NPZ SHA-256 `7876f2173e54bf73db2d2feceb0c449a2fc219298108a141a6fe6b313eb3c18f` stays immutable; finish is `x >= 100m`, lane is strict `|y-4| < 1.9m`, final threshold is at least 91/100.
- Warm start `stage3_rl/runs_v21/model_latest.pt` SHA-256 `a9a6dab12c2417e15b105ebe7d7eb0f8c4c5ba07d01ee8fa237c88c42a42e43c` stays immutable.
- `pose-v1` is an **opt-in** evaluation setting used identically for checkpoint selection and final acceptance; legacy defaults remain for historical diagnostics.
- Existing `results/eval_summary.json`, `results/eval_best_traj.npz`, and videos must not be overwritten; all new outputs go to new empty run directories.
- This project has no Git metadata; replace the usual commit step with a saved test log and a source/checkpoint/track hash record.

## File responsibilities

- Create `common/eval_protocol.py`: pure deterministic seed-to-initial-pose mapping; no MuJoCo or filesystem.
- Modify `stage3_rl/env.py`: consume an evaluation protocol at reset and expose requested/settled pose.
- Modify `stage3_rl/evaluate.py` and `stage3_rl/evaluate_par.py`: CLI, manifest, per-episode protocol/pose, exact-seed replay.
- Modify `stage3_rl/ppo_train.py`: 25-iteration cadence and identical opt-in protocol for probes.
- Create `tests/test_eval_protocol.py`, `tests/test_probe_cadence.py`, `tests/test_video_replay.py`: one responsibility per test file.

---

### Task 1: Seed-distinct fixed-track initial poses

**Files:** Create `common/eval_protocol.py`, `tests/test_eval_protocol.py`; modify `stage3_rl/env.py`, `stage3_rl/evaluate.py`, `stage3_rl/evaluate_par.py`.

**Interfaces:** `initial_pose(seed: int, protocol: str) -> tuple[float, float, float]` returns requested `(x0,y0,yaw_rad)`; `Go2WEnv(..., eval_protocol="legacy")`; `run_episode` records `requested_initial_pose` and `settled_initial_pose`. Evaluator CLI `--eval-protocol {legacy,pose-v1}`; worker config passes it unchanged.

Concrete test and mapping core:

```python
assert initial_pose(9000, "pose-v1") == initial_pose(9000, "pose-v1")
assert len({initial_pose(s, "pose-v1") for s in range(9000, 9100)}) == 100
rng = np.random.default_rng(int(seed) ^ 0x5A17E)
return (0.5 + float(rng.uniform(-.05, .05)),
        4.0 + float(rng.uniform(-.10, .10)),
        float(rng.uniform(-.02, .02)))
```

- [x] **Step 1: Write failing mapping/geometry tests.** In `tests/test_eval_protocol.py`, assert `initial_pose(9000,"legacy") == (0.5,4.0,0.0)`, same seed `pose-v1` repeated returns same tuple, the 100 tuples for 9000–9099 differ, and every tuple lies within x=0.45–0.55, y=3.9–4.1, yaw=-0.02–0.02. Instantiate two fixed-track `Go2WEnv(seed=9000/9001, eval_mode=True, eval_protocol="pose-v1")`, assert requested and settled initial poses differ, and assert `track_success(99.99,4,100)` remains false and `within_track_lane(5.9)` remains false. Parse both evaluators' `--eval-protocol pose-v1` and confirm `evaluation_config(...)["eval_protocol"]` matches.
- [x] **Step 2: Confirm red.** Run `python -m unittest tests.test_eval_protocol -v`; expect import/argument/signature failures before implementation.
- [x] **Step 3: Implement the pure mapping.** In `common/eval_protocol.py`, use an independent `np.random.default_rng(int(seed) ^ 0x5A17E)` for `pose-v1`; draw `x=0.5+rng.uniform(-.05,.05)`, `y=4+rng.uniform(-.10,.10)`, `yaw=rng.uniform(-.02,.02)`; return exact legacy values for `legacy`, reject unknown names. Do not consume the environment's terrain/randomization RNG.
- [x] **Step 4: Apply pose only at fixed-track eval reset.** In `env.py`, reject `eval_protocol != "legacy"` unless `fixed_track is not None and eval_mode`; compute `requested_initial_pose` before PD settle, assign `d.qpos[0:3]=[x0,y0,h0+.36]` and `d.qpos[3:7]=[cos(yaw/2),0,0,sin(yaw/2)]`; store `settled_initial_pose` after the existing 125 settle steps. Keep the existing training reset branch unchanged.
- [x] **Step 5: Thread protocol through both evaluators.** Add the CLI choice with default `legacy`; pass it to every `Go2WEnv` construction including serial replay, parallel workers, and `_replay_episode`. Record both pose objects as float lists in `run_episode` results, put `eval_protocol`, x/y/yaw ranges and seed policy in the manifest config, and have `record_video` accept the chosen protocol. Do not change `run_episode` action generation or success logic.
- [x] **Step 6: Confirm green and serial/parallel parity.** Run `python -m unittest tests.test_eval_protocol tests.test_evaluation_manifest tests.test_track_acceptance -v`; then evaluate v21 for seeds 9000–9002 into new `results/diagnostics/<UTC>/pose-serial` and `.../pose-parallel`, both with `--eval-protocol pose-v1`, same episode cap and checkpoint. Assert each seed's initial-pose row and final/max x match between evaluators; persist their manifests/rows. Use `--workers 1` for the parallel comparison.

### Task 2: Guaranteed first pilot probe and comparable checkpoint selection

**Files:** Modify `stage3_rl/ppo_train.py`; create `tests/test_probe_cadence.py`; extend `tests/test_ppo_probe_config.py`.

**Interfaces:** `should_probe(iteration_zero_based: int, interval: int=25) -> bool`; CLI `--eval-protocol {legacy,pose-v1}`; `Learner.track_probe(..., eval_protocol="legacy")` passes protocol to its environment and returns per-episode pose metadata.

```python
def should_probe(iteration_zero_based: int, interval: int = 25) -> bool:
    if interval <= 0:
        raise ValueError("probe interval must be positive")
    return (iteration_zero_based + 1) % interval == 0

assert [i for i in range(25) if should_probe(i)] == [24]
```

- [x] **Step 1: Write red cadence tests.** Assert `should_probe(0)==False`, `should_probe(23)==False`, `should_probe(24)==True`, `should_probe(49)==True`, and `should_probe(24,0)` raises `ValueError`. Parse `--eval-protocol pose-v1`, and use a fake environment constructor to verify `Learner.track_probe` receives `eval_protocol="pose-v1"` and records its pose; no workers or learner initialization in the unit test.
- [x] **Step 2: Confirm red.** Run `python -m unittest tests.test_probe_cadence tests.test_ppo_probe_config -v`; expect helper/CLI failure.
- [x] **Step 3: Implement cadence and probe wiring.** Replace `if it % 25 == 0 and it > 0` with `if should_probe(it)`; save `it+1` in probe logs and `model_it00025.pt`, and keep `model_latest.pt` atomic. Pass `args.eval_protocol` into `track_probe`, use one frozen `seed_base=args.seed+9000` across iterations so improvements see the same scenarios; log `eval_protocol`, seed base and per-episode pose. Probe errors must log and must not manufacture `model_best.pt`.
- [x] **Step 4: Confirm green and one-iteration no-probe smoke.** Run the two unit modules and `python stage3_rl/ppo_train.py --help`; verify 25th update probes in a mocked loop without launching workers. Do not run a 25-iteration pilot until Task 1 passes.

### Task 3: Side-effect-free recorder

**Files:** Modify `stage3_rl/evaluate.py`; create `tests/test_video_replay.py`.

**Interfaces:** `VideoRecorder.__call__(env)` reads post-step `env.data` and renders without advancing or recomputing physics; `_replay_episode` still requires exact steps/reason/success/max-x parity and refuses mismatched videos.

The callback's core must be read-only with respect to physics:

```python
x, y, z = env.data.xpos[env.body_root]
self.cam.lookat[:] = [x, y, z]
self.renderer.update_scene(env.data, self.cam)
self.proc.stdin.write(self.renderer.render().tobytes())
# No mj_forward or mj_step here.
```

- [x] **Step 1: Write red replay test.** Run a v21 real fixed-track episode once without callback and once with a lightweight callback that invokes `VideoRecorder.__call__` using fake renderer/stdin objects and `frame_every=2`; compare `reason`, `steps`, and `max_x` at `1e-6`. Patch `mujoco.mj_forward` in the callback path to raise if called. Include a test that deliberately alters a replay result and confirms video is rejected/deleted.
- [x] **Step 2: Confirm red.** Run `python -m unittest tests.test_video_replay -v`; expect the recorder's extra forward call to fail.
- [x] **Step 3: Implement minimal fix.** Remove only `mujoco.mj_forward(env.model, env.data)` from `VideoRecorder.__call__` and its now-unused local import; leave the reset-time forward/settle untouched. The renderer `update_scene` must use the state after `env.step`.
- [x] **Step 4: Confirm green with real physics and preserve old files.** Run `python -m unittest discover -s tests -p 'test_*.py' -v`; run one new v21 `pose-v1` seed without video and one replay to a new diagnostic video path, compare failure location/steps, inspect ffprobe duration and file hash. Do not label it a successful race video.

## Handoff gate

Do not start the PPO pilot or claim statistical completion until Tasks 1–3 pass and the serial/parallel protocol comparison is saved. The full-track success line remains strictly `x>=100m`, lane `<1.9m`, at least 91/100; the protocol is frozen before checkpoint selection, not chosen after seeing candidate outcomes.
