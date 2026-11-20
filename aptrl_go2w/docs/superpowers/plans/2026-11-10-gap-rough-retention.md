# Gap/Rough-Slope Retention Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one isolated 25-update PPO pilot from v21 with two 0.2m-gap workers and two original-rough-slope workers, then make an auditable stop/promote decision.

**Architecture:** Add one fixed-track `rough_slope_retain` local profile and a pure 4-worker assignment for a composite learner-only `gap_rough_retain` CLI mode. Do not switch MuJoCo terrains inside a worker. Existing `single_gap`, fixed-track evaluation, actor, reward, decoder and actuator paths remain unchanged; record all worker assignments before launch and apply a three-way frozen gate afterwards.

**Tech Stack:** Python 3, NumPy, MuJoCo, PyTorch, unittest, Windows multiprocessing spawn.

**Spec:** `docs/superpowers/specs/2026-11-10-gap-rough-retention-design.md`

## Global Constraints

- The original NPZ SHA-256 is `7876f2173e54bf73db2d2feceb0c449a2fc219298108a141a6fe6b313eb3c18f`; v21 SHA-256 is `a9a6dab12c2417e15b105ebe7d7eb0f8c4c5ba07d01ee8fa237c88c42a42e43c`. Neither is modified.
- Final full-track success remains `x >= 100m`, strictly `|y-4| < 1.9m`, at least 91/100; this pilot is only a 10-seed diagnostic, never acceptance.
- Use the original track for rough-slope retention, full-track probes and gate; only the **training copy** of the single first gap is 0.2m. No human pulse, label or fallback action enters PPO.
- Limit training to `--iters 25 --workers 4 --epw 1 --batch-target 4096 --epochs 2 --episode-len 900`; initialize from original v21, not the non-promoted gap-only pilot. All output paths are fresh and exclusive.
- Preserve previous checkpoints, logs, videos, `results/eval_summary.json` and `results/eval_best_traj.npz`. This project is not a Git repository: save test logs, run config and hashes rather than commit.

## File responsibilities

- `stage3_rl/env.py`: new original-NPZ rough-slope local profile, x≈15.0 to x≥23.5, seed-distinct starts and unchanged collision/actuator rules.
- `stage3_rl/curriculum_gate.py`: run original rough-slope ten-seed local gate; pure artifact-based three-way retention decision.
- `stage3_rl/ppo_train.py`: validate 2:2 composite CLI, assign actual per-worker stages and log mapping before spawn.
- `tests/test_training_profiles.py`, `tests/test_curriculum_gate.py`, `tests/test_training_cli.py`: narrow behavioral regressions, including identical original terrain and same checkpoint across gates.

---

### Task 1: Original rough-slope local profile and baseline gate

**Files:** Modify `stage3_rl/env.py`, `stage3_rl/curriculum_gate.py`, `tests/test_training_profiles.py`, `tests/test_curriculum_gate.py`.

**Interfaces:** `Go2WEnv(..., train_stage="rough_slope_retain", train_track="C:/Users/Administrator/Desktop/go2w_race_track_seed50001.npz", eval_mode=True|False)` returns `success_x=23.5`, `requested_initial_pose[0] in [14.95,15.05]`, the exact authorized fixed-track `H`/boxes, and 10 seed-distinct settled starts. `evaluate_gate(..., train_stage="rough_slope_retain")` runs unchanged `PolicyRunner` and treats `reason="success"`, `max_x>=23.5` as a pass without a gap-lip contact requirement.

- [x] **Step 1: Write failing tests.** In `test_training_profiles.py`, compare `H`, actuator ranges and lane rule with official fixed-track env, assert local finish 23.5, seeded starts 9000/9001 differ, repeated 9000 matches, and repeated train-mode resets do not compound terrain friction. In `test_curriculum_gate.py`, assert a rough-slope row at x23.49 fails and at x23.5 with `reason="success"` passes even with `landed_after_lip=None`.

```python
retained = Go2WEnv(seed=9000, train_stage="rough_slope_retain",
                    train_track=TRACK, eval_mode=True)
np.testing.assert_array_equal(retained.H, official.H)
self.assertEqual(retained.success_x, 23.5)
self.assertTrue(14.95 <= retained.requested_initial_pose[0] <= 15.05)
```

- [x] **Step 2: Verify red.** Run `APTRL_TRACK_NPZ='C:/Users/Administrator/Desktop/go2w_race_track_seed50001.npz' python -m unittest tests.test_training_profiles tests.test_curriculum_gate -v`; expect unsupported stage/gate failures, not missing fixtures.
- [x] **Step 3: Implement only the local profile.** Add stage to env's local stage set and train-track load branch; choose base x=15.0 for retention and 24.5 for existing gaps, `success_x=23.5`. Keep original `load_npz_track` output and default terrain XML unchanged. Add stage to gate's accepted stages/track-loading branch, without lip-contact requirement. Do not change official `fixed_track` constructor behavior.
- [x] **Step 4: Verify green and collect v21 baseline.** Run the two test modules, then run `python stage3_rl/curriculum_gate.py --ckpt stage3_rl/runs_v21/model_latest.pt --track "C:/Users/Administrator/Desktop/go2w_race_track_seed50001.npz" --train-stage rough_slope_retain --out results/diagnostics/$RUN_ID/v21-rough-retain` with a new UTC RUN_ID. Verify 10 distinct initial poses and source hashes. Use the observed 6/10 v21 baseline at `results/diagnostics/20261113T131910Z/v21-rough-retain/gate.json`; its passing seed set is pre-registered as 9001,9003,9004,9005,9008,9009. Preserve the original heightmap and use this artifact rather than rerunning or selecting favorable seeds.

### Task 2: Pure 2:2 worker routing and auditable provenance

**Files:** Modify `stage3_rl/ppo_train.py`, `tests/test_training_cli.py`.

**Interfaces:** `worker_stage_for_index(train_stage: str, index: int, workers: int) -> str` returns `single_gap` for indices 0,2 and `rough_slope_retain` for 1,3 when composite is chosen; otherwise returns the existing stage. `Learner(..., train_stage="gap_rough_retain", gap_width_m=0.2, train_track=...)` passes the assigned stage (and `gap_width_m=None` for retention) to each worker. `run_config.json` records `worker_stages` in index order and `allocation="2 gap : 2 rough_slope_retain"`.

- [x] **Step 1: Write failing validation/routing tests.** Assert exact 4-element mapping; composite rejects 1/2/3/5 workers, `epw!=1`, width other than .2, or absent original `--train-track`; parsing non-composite modes remains unchanged. In an exclusive temp run directory, assert written `run_config.json` contains the mapping and checkpoint/source/track hashes before any worker starts.

```python
self.assertEqual([worker_stage_for_index("gap_rough_retain", i, 4)
                  for i in range(4)],
                 ["single_gap", "rough_slope_retain",
                  "single_gap", "rough_slope_retain"])
with self.assertRaises(ValueError):
    worker_stage_for_index("gap_rough_retain", 0, 3)
```

- [x] **Step 2: Verify red.** Run `python -m unittest tests.test_training_cli -v`; expect missing composite CLI/routing behavior.
- [x] **Step 3: Implement routing.** Add composite to parser/validation but **not** to `Go2WEnv` valid stages. Validate `workers==4`, `epw==1`, width exactly .2, train-track readable. At learner worker-cfg construction use `worker_stage_for_index(train_stage,w,n_workers)`; use `gap_width_m` only for `single_gap`, None for retention. Keep `compute_chunk_count`, PPO optimizer, action/reward and probe path unchanged.
- [x] **Step 4: Verify green and one-update worker smoke.** Run `tests.test_training_cli tests.test_ppo_probe_config tests.test_training_profiles`; run a one-iteration, one-chunk smoke with four workers and new `--out`, `--batch-target 200 --iters 1`, original v21 init and `gap_rough_retain`. Assert the config maps 2:2, each worker reaches ready, `model_final.pt` is saved, and the smoke output is not considered a candidate.

### Task 3: Artifact-based three-way gate and bounded pilot

**Files:** Modify `stage3_rl/curriculum_gate.py`, `tests/test_curriculum_gate.py`; create new `stage3_rl/runs_curriculum_$RUN_ID_gaprough/`, `results/diagnostics/$RUN_ID/`, `stage3_rl/experiments/$RUN_ID/stage-decisions.jsonl`.

**Interfaces:** `retention_decision(gap_gate: dict, rough_gate: dict, baseline_rough_gate: dict, candidate_full_manifest: str, baseline_full_manifest: str) -> dict` reads sibling `episodes.jsonl` files, checks all three candidate artifact hashes agree, exactly the same 10 seeds, `pose-v1` full-track config and original track hash. It returns `promote: bool`, per-condition booleans and per-seed full-track deltas. Conditions: gap≥8/10; rough≥6/10 **and all six v21-passing seeds (9001,9003,9004,9005,9008,9009) remain successful**; candidate full median≥v21 median; each paired candidate max-x≥baseline max-x−5m.

- [x] **Step 1: Write red decision tests.** Build temporary real manifest/episode fixtures with ten literal paired seed rows, matching checkpoint hashes and track hash. Assert all conditions pass at 8/10 gap and six specific retained rough seeds; mutate one condition at a time (7/10 gap, 5/10 rough, rough 6/10 with one previously passing seed lost and another newly passing seed, a −5.01m paired regression, lower median, mismatched candidate checkpoint hash, legacy protocol, duplicate seeds) and assert `promote=False` with a named failing condition.

```python
result = retention_decision(gap_gate, rough_gate, baseline_rough_gate,
                            str(candidate_manifest), str(baseline_manifest))
self.assertTrue(result["conditions"]["gap_8_of_10"])
self.assertTrue(result["promote"])
# In a separate fixture, candidate seed 9002 loses 5.01m:
regression = retention_decision(gap_gate, rough_gate, baseline_rough_gate,
                                str(regressed_manifest), str(baseline_manifest))
self.assertFalse(regression["conditions"]["paired_regression"])
self.assertFalse(regression["promote"])
```

- [x] **Step 2: Verify red.** Run `python -m unittest tests.test_curriculum_gate -v`; expect missing helper or failure cases.
- [x] **Step 3: Implement pure decision parser.** Read each manifest's `checkpoint.sha256`, `track.sha256`, `evaluation.eval_protocol/seed_base/episodes` and `episodes.jsonl`; compare them to the local `gate.json` checkpoint/track hashes and seed sets. Compute median and ten paired deltas directly from recorded rows. Return `promote=all(conditions.values())`; never accept caller-supplied median or success count without cross-checking rows.
- [x] **Step 4: Run bounded experiment only after the approved baseline prerequisites pass.** Verify original NPZ/v21 hashes and full suite; start from v21 into an empty `stage3_rl/runs_curriculum_$RUN_ID_gaprough` using `--train-stage gap_rough_retain --gap-width 0.2 --train-track C:/Users/Administrator/Desktop/go2w_race_track_seed50001.npz --probe-track C:/Users/Administrator/Desktop/go2w_race_track_seed50001.npz --eval-protocol pose-v1 --workers 4 --epw 1 --episode-len 900 --batch-target 4096 --epochs 2 --iters 25`. Save stdout/stderr. Require exactly one `iter=25` probe, `model_best.pt`, `model_it00025.pt`, and mapped `run_config.json`; if the probe fails or the run hangs, stop without promoting.
- [x] **Step 5: Evaluate and stop/promote.** Run 10-seed 0.2m gap and rough retention local gates with **the same best checkpoint**, then 10-seed original `pose-v1` `evaluate_par.py`. Use the saved v21 full-track and rough-retention baseline artifacts in `retention_decision`. Append exact file paths, hashes, per-seed deltas, condition booleans and decision to a new `stage-decisions.jsonl`. A failing condition ends this pilot without another automatic run; a pass only permits the next curriculum stage, not a 100m or >90% claim.

## Self-review gate

This plan implements the approved exact 2:2 worker allocation, original rough-slope local geometry, 25-update cap and three-way promotion rule. Prior single-gap pilot remains non-promoted. The only potential long operation is Task 3 Step 4; it must not start before the v21 rough retention baseline and all red/green tests pass. No acceptance video or `REPORT.md` success edit is part of this pilot.

## Execution checkpoint (2026-11-13)

Task 1 produced a valid original-NPZ `rough_slope_retain` profile and a ten-seed v21 baseline at `results/diagnostics/20261113T131910Z/v21-rough-retain/gate.json`: **6/10**. A throwaway x≈12.0/13.5m run-up diagnostic, `results/diagnostics/20261113T131910Z/rough-runup-spike.json`, yielded **7/10** and **6/10**. Full tests: `stage3_rl/experiments/20261113T131910Z/tests-prepilot-blocked.log` (51 passed). Original NPZ and v21 SHA-256 remain unchanged. The user explicitly approved replacing the unattainable local 9/10 prerequisite with preservation of all six v21-successful seeds and at least 6/10 total; Tasks 2–3 may now proceed under that frozen rule. No worker-routing code or mixed training had started at this checkpoint.

## Startup synchronization prerequisite

Before the 25-update pilot, inline review found that `main` loaded v21 only into the learner, while workers stayed random until the end of update 1. A real four-worker regression test reproduced `first collection precedes initial weight sync` (`stage3_rl/experiments/20261113T131910Z/test-initial-sync-red.log`). The minimal repair broadcasts actor/encoder/normalizer immediately after loading the checkpoint and waits for every worker's acknowledgment before collection. The same real-worker test now passes (`test-initial-sync-green.log`). The prior single-gap pilot and early 2:2 smoke are annotated with `INITIAL_SYNC_LIMITATION.md`; their checkpoint measurements are preserved, but neither is a valid all-sampling-from-v21 continuation. The newly approved pilot must use the repaired startup.

## Completed bounded pilot — STOP

Run `20261113T134122Z_gaprough` executed exactly 25 updates from original v21 after initial worker synchronization. The sole update-25 probe succeeded and the selected checkpoint SHA-256 is `2a89dd92c495fea8a4e7e899ef387e31f4cedf10706d10740e3949af694c2f28`. Observed gate results: gap **0/10** (FAIL), rough **8/10** with all six baseline-successful seeds retained (PASS), original full track **0/10**, median furthest x **25.57767720947101m** vs v21 **16.804418713455846m** (PASS), but paired seeds 9001/9004/9006 lost **15.900453622226706m / 15.140801500399455m / 15.764543494014783m** (FAIL). The saved and recomputed decision is `promote=false`, with `artifact_integrity=true`. No continuation, acceptance trial or success video was launched.

Evidence: `results/diagnostics/20261113T134122Z_gaprough/SUMMARY.md` explains all artifacts and deltas; `stage3_rl/experiments/20261113T134122Z_gaprough/stage-decisions.jsonl` contains the machine decision and hashes; `tests-postpilot-full.log` in that experiment directory records `Ran 58 tests in 12.247s` and `OK`. Protected original/old-result hashes and training source hashes matched on final verification. This non-Git project and all files remain in place; there is no branch to merge or worktree to delete.
