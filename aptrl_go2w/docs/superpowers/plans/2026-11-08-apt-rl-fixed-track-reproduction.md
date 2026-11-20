# APT-RL Go2-W Fixed-Track Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the fixed-track environment and evaluation with the approved `REPORT.md` specification, then improve and verify a Go2-W policy to achieve more than 90% full-track completion without unreported terrain or metric changes.

**Architecture:** Keep the existing Stage 1/2 data and models. First make the track loader and environment acceptance rules match the report and lock them with tests. Then make evaluation runs reproducible and diagnostic, use small PPO probes in isolated output directories, and run the full acceptance evaluation and video recording only after the earlier gates pass.

**Tech Stack:** Python, NumPy, PyTorch, MuJoCo, standard-library `unittest`, JSON manifests, existing PPO multiprocessing workers.

**Spec:** `docs/superpowers/specs/2026-11-08-apt-rl-reproduction-design.md`

## Global Constraints

- The accepted goal is the `REPORT.md` goal: fixed 104m track, finish at `x >= 100m`, throughout `|y - 4.0| < 1.9m`, and no failure-causing non-foot body contact.
- Completion rate must be strictly greater than 90%; with 100 episodes, at least 91 must succeed.
- Apply only the terrain changes documented in `REPORT.md`; any additional terrain or evaluation change requires user approval and a corresponding report update.
- Do not regenerate Stage 1/2 data or overwrite existing model checkpoints, evaluation files, or `video/pipeline_test.mp4`.
- Do not start long PPO training until the geometry and evaluation contract tests pass.
- The source tree is not a Git repository. Do not initialize Git or claim commits; keep new run outputs separate and report this limitation.
- The fixed-track input is external: `C:/Users/Administrator/Desktop/go2w_race_track_seed50001.npz`. Pass its path explicitly and record its SHA-256 in every run manifest.

---

### Task 1: Lock the report-defined fixed-track geometry

**Files:**
- Modify: `common/terrain.py::load_npz_track`
- Create: `tests/__init__.py`
- Create: `tests/test_track_spec.py`

**Interfaces:**
- Consumes: `load_npz_track(npz_path, apply_fixes=True) -> (H, boxes, seq, meta)`.
- Produces: documented cliff/crawl/stool fixes only when `apply_fixes=True`; raw heightmap and raw obstacle-component bounds when `apply_fixes=False`; `meta["fixes"]` lists only transformations actually applied.

- [x] **Step 1: Add tests that compare the loader to the source NPZ.** Use `unittest`, read the raw `height_field_raw * vertical_scale`, and assert:
  - the sequence has the documented 13 entries;
  - in `apply_fixes=True`, every height-field segment other than `stairs_cliff` and `slope_cliff` equals its source segment exactly;
  - the two cliff segments have the report's capped drops and a vertical end drop at the documented segment location;
  - the crawl-bar lower face is `z0 == 0.40` within `1e-6`;
  - stool mesh components are merged only in fixed mode and preserve their original outer bounds;
  - in `apply_fixes=False`, the heightmap is byte-for-byte equivalent after scale conversion, original crawl clearance remains, no fixes are listed, and stool components are not merged.

  Test skeleton:

  ```python
  import os
  import unittest
  import numpy as np
  from common.terrain import load_npz_track

  class TrackSpecTests(unittest.TestCase):
      @classmethod
      def setUpClass(cls):
          cls.path = os.environ["APTRL_TRACK_NPZ"]
          cls.raw = np.load(cls.path, allow_pickle=True)
          cls.expected = (cls.raw["height_field_raw"].astype(np.float32)
                          * float(cls.raw["vertical_scale"]))
          cls.fixed_h, cls.fixed_boxes, cls.seq, cls.meta = load_npz_track(cls.path, True)

      def test_unmodified_segments_match_source(self):
          for i, name in enumerate(self.seq):
              if name not in ("stairs_cliff", "slope_cliff"):
                  np.testing.assert_array_equal(
                      self.fixed_h[i * 80:(i + 1) * 80],
                      self.expected[i * 80:(i + 1) * 80])

      def test_crawl_clearance_matches_report(self):
          bars = [b for b in self.fixed_boxes
                  if b["x1"] - b["x0"] < 0.6 and b["y1"] - b["y0"] > 6.0]
          self.assertEqual(len(bars), 1)
          self.assertAlmostEqual(bars[0]["z0"], 0.40, places=6)
  ```

- [x] **Step 2: Run the new tests and confirm they fail on current v2 transformations.**

  Run from `aptrl_go2w`:

  ```bash
  export APTRL_TRACK_NPZ='C:/Users/Administrator/Desktop/go2w_race_track_seed50001.npz'
  python -m unittest tests.test_track_spec -v
  ```

  Expected: failures identify non-cliff heightmap changes, crawl clearance, and `apply_fixes=False` behavior.

- [x] **Step 3: Remove the undocumented v2 geometry transformations.** Keep only the report-authorized cliff repair, crawl clearance `0.40m`, and stool component merge. Put each transform, including stool merging, behind `apply_fixes`; leave all non-cliff heightmap segments unchanged. Build metadata from the transforms actually applied. Use the loader's documented cliff positions (x=85.8m and x=95.8m); assert the corresponding height-field rows using the established 0.1m grid convention.

- [x] **Step 4: Verify geometry tests and the full test suite.**

  Run:

  ```bash
  python -m unittest tests.test_track_spec -v
  python -m unittest discover -s tests -p 'test_*.py' -v
  ```

  Expected: all geometry assertions pass, including exact preservation of every unmodified segment.


---

### Task 2: Make environment termination match the accepted success contract

**Files:**
- Modify: `stage3_rl/env.py`
- Create: `tests/test_track_acceptance.py`

**Interfaces:**
- Add pure helpers `within_track_lane(y: float) -> bool` and `track_success(x: float, y: float, success_x: float) -> bool`.
- `Go2WEnv._check_done()` uses those helpers for the same global lane rule at every track location and the same lane rule at the finish.

- [x] **Step 1: Write boundary tests before changing the environment.**

  ```python
  import unittest
  from stage3_rl.env import within_track_lane, track_success

  class TrackAcceptanceTests(unittest.TestCase):
      def test_lane_is_strictly_within_reported_1_9m(self):
          self.assertTrue(within_track_lane(4.0))
          self.assertTrue(within_track_lane(5.89))
          self.assertFalse(within_track_lane(5.9))
          self.assertFalse(within_track_lane(2.1))

      def test_success_requires_finish_and_lane(self):
          self.assertTrue(track_success(100.0, 4.0, 100.0))
          self.assertFalse(track_success(99.99, 4.0, 100.0))
          self.assertFalse(track_success(100.0, 5.9, 100.0))
  ```

- [x] **Step 2: Confirm the tests fail because the helpers do not yet exist.**

  Run: `python -m unittest tests.test_track_acceptance -v`

  Expected: import failure for the two missing helpers.

- [x] **Step 3: Implement the two helpers and use them in `_check_done()`.** Use lane center `4.0` and strict half-width `1.9`. Remove the looser `2.5m` general bound and the tighter bridge/stool-only `1.15m` rule from the acceptance path. Keep physical body-contact, attitude, fall, and timeout detection unchanged; restrict legacy partial-progress timeout success to adaptive (non-fixed) curriculum so fixed-track evaluation requires `x >= success_x`. A lane violation must terminate as `off_lane`; reaching `success_x` only succeeds while still in the lane.

- [x] **Step 4: Run acceptance and existing smoke tests.**

  Run:

  ```bash
  python -m unittest tests.test_track_acceptance -v
  python -m unittest discover -s tests -p 'test_*.py' -v
  python -c "from stage3_rl.env import Go2WEnv; print('env import ok')"
  ```

  Expected: boundary tests pass and the environment imports successfully.

---

### Task 3: Make evaluation runs reproducible and failure-diagnostic

**Files:**
- Create: `common/evaluation.py`
- Modify: `stage3_rl/evaluate.py`
- Modify: `stage3_rl/evaluate_par.py`
- Modify: `stage3_rl/env.py` (terminal contact details only)
- Create: `tests/test_evaluation_manifest.py`

**Interfaces:**
- `common.evaluation.sha256_file(path: str) -> str` returns a lowercase SHA-256 hex digest.
- `common.evaluation.build_run_manifest(*, checkpoint, track, config, fixes, source_files, track_summary) -> dict` returns JSON-safe paths, hashes, runtime versions, evaluation configuration, applied fixes, and track geometry summary.
- `common.evaluation.summarize_track(H, boxes, seq) -> dict` returns per-segment height ranges and obstacle-box bounds.
- `common.evaluation.write_run_manifest(out_dir, manifest) -> str` atomically writes `manifest.json` and refuses to overwrite a non-empty run directory.
- Both evaluator CLIs accept `--track` and a new, empty `--out` directory; neither evaluator writes into an existing non-empty directory.
- Every episode result records seed, success, reason, final/max position, furthest segment, and terminal contact names/positions when applicable; persist one JSON object per line in `episodes.jsonl`.

- [x] **Step 1: Add manifest tests.** Create temporary checkpoint and track files, call `build_run_manifest`, and assert that it records both SHA-256 values, the evaluator settings, actual fix list, Python/NumPy/Torch/MuJoCo versions, and the supplied source-file hashes. Assert that writing a second manifest to the same non-empty run directory fails instead of overwriting it.

- [x] **Step 2: Confirm manifest tests fail because the module and APIs do not exist.**

  Run: `python -m unittest tests.test_evaluation_manifest -v`

  Expected: import failure for `common.evaluation`.

- [x] **Step 3: Implement the manifest helpers.** Use `hashlib.sha256` with chunked reads. `summarize_track` records the sequence, per-segment height minima/maxima, and obstacle-box bounds. `build_run_manifest` records Python/NumPy/Torch/MuJoCo versions, checkpoint and track paths/hashes, source-file paths/hashes, evaluation configuration, track summary, and actual fixes. `write_run_manifest` writes through a temporary file and `os.replace`; reject an output directory containing prior run files.

- [x] **Step 4: Add explicit track and output-directory arguments to both evaluators.** Remove the hard-coded desktop NPZ constant. Pass the selected path through worker configuration and video replay. Add the per-episode seed to worker results and persist each result to `episodes.jsonl`; video replay selects a successful episode and uses its seed and the exact same checkpoint/configuration. If no episode succeeds, do not emit a video labeled successful. Keep existing result JSON and trajectory formats, adding fields without deleting existing ones.

- [x] **Step 5: Capture terminal contacts from the environment.** Factor the existing bad-contact predicate so the same body-ID rule drives both termination and a JSON-safe diagnostic list. Each terminal contact item contains geom names, body names, distance, and contact position. Attach that list to `info` only on termination; leave reward computation unchanged.

- [x] **Step 6: Test CLI and metadata behavior without a long rollout.** Test argument parsing and manifest generation using temporary paths. Run `python stage3_rl/evaluate.py --ckpt stage3_rl/runs_f2/model_latest.pt --track "$APTRL_TRACK_NPZ" --episodes 1 --episode-len 16000 --seed 9000 --device cpu --out results/diagnostics/task3-serial-smoke-final` and `python stage3_rl/evaluate_par.py --ckpt stage3_rl/runs_f2/model_latest.pt --track "$APTRL_TRACK_NPZ" --episodes 1 --episode-len 16000 --seed 9000 --device cpu --workers 1 --out results/diagnostics/task3-parallel-smoke-verified`; confirm both manifests name the exact input track and list actual applied fixes. Confirm the old `results/eval_summary.json` and `results/eval_best_traj.npz` hashes are unchanged. Observed: both CPU runs with seed 9000 reported `body_contact` in `gap` at x=25.679m after 1,852 steps; both manifests recorded track SHA-256 `7876f2173e54bf73db2d2feceb0c449a2fc219298108a141a6fe6b313eb3c18f` and six fixes. The original summary and trajectory hashes remained unchanged.

---

### Task 4: Make PPO probes configurable and safe for short experiments

**Files:**
- Modify: `stage3_rl/ppo_train.py`
- Create: `tests/test_ppo_probe_config.py`

**Interfaces:**
- Add pure `compute_chunk_count(batch_target, workers, envs_per_worker, chunk_h=CHUNK_H) -> int` and use it in `Learner`.
- Add `build_arg_parser() -> argparse.ArgumentParser` so the CLI contract can be tested without starting workers.
- Add CLI flags `--batch-target` (default `65536`) and required `--probe-track`; probes must never fall back to a user-specific path.
- `Learner.track_probe(..., track_path)` evaluates the requested track and returns `best_x`, `median_x`, success count, completed-segment counts, and failure counts.
- `write_probe_record(path, record) -> None` appends one JSON object followed by a newline.
- Every training run requires a new empty `--out` directory and writes only there.
- The training loop appends each probe result to `probe_metrics.jsonl` in that run directory.

- [x] **Step 1: Add unit tests for chunk sizing, CLI parsing, and probe logging.** Test `compute_chunk_count(4096, 4, 1) == 21`, `compute_chunk_count(65536, 4, 1) == 328`, and that non-positive batch/worker/environment counts are rejected. Test `build_arg_parser().parse_args(["--probe-track", "track.npz", "--out", "new-run", "--batch-target", "4096"]).probe_track == "track.npz"`. Test that `write_probe_record(path, {"iter": 25, "best_x": 3.0})` appends one JSON object per line and does not truncate an existing record.

- [x] **Step 2: Verify the tests fail on the current implementation.**

  Run: `python -m unittest tests.test_ppo_probe_config -v`

  Expected: missing helper/argument failures.

- [x] **Step 3: Implement the helper, parser, and probe logging.** Use `compute_chunk_count` in the learner constructor. Pass `--probe-track` to periodic probes and append the returned metrics to `probe_metrics.jsonl`. Before constructing workers, reject a non-empty output directory. Do not change rewards, actor/critic architecture, or default optimizer settings in this task.

- [x] **Step 4: Verify the pure tests and training CLI help.**

  Run:

  ```bash
  python -m unittest tests.test_ppo_probe_config -v
  python stage3_rl/ppo_train.py --help
  ```

  Expected: tests pass and help lists the two new options.

---

### Task 5: Establish current-policy baseline and run a bounded PPO pilot

**Files:**
- Create: `results/diagnostics/$RUN_ID/current-policy-baseline/f2/`
- Create: `results/diagnostics/$RUN_ID/current-policy-baseline/f4/`
- Create: `results/diagnostics/$RUN_ID/current-policy-baseline/v21/`
- Create: `results/diagnostics/$RUN_ID/current-policy-baseline/hf1/`
- Create: `stage3_rl/experiments/$RUN_ID/baseline-selection.json`
- Create: `stage3_rl/runs_repro_pilot_$RUN_ID/`

**Interfaces:**
- Consume the geometry- and acceptance-tested environment, current checkpoints, explicit track path, and unique output directories.
- Produce comparable per-checkpoint episode summaries and isolated pilot checkpoints; do not alter existing checkpoints.

- [x] **Step 1: Run the current-checkpoint baseline with the same evaluator and three episodes per checkpoint.** From `aptrl_go2w`, set `RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)` and `APTRL_TRACK_NPZ`, then run the following commands:

  ```bash
  python stage3_rl/evaluate_par.py --ckpt stage3_rl/runs_f2/model_latest.pt --track "$APTRL_TRACK_NPZ" --episodes 3 --workers 1 --out results/diagnostics/$RUN_ID/current-policy-baseline/f2
  python stage3_rl/evaluate_par.py --ckpt stage3_rl/runs_f4/model_best.pt --track "$APTRL_TRACK_NPZ" --episodes 3 --workers 1 --out results/diagnostics/$RUN_ID/current-policy-baseline/f4
  python stage3_rl/evaluate_par.py --ckpt stage3_rl/runs_v21/model_latest.pt --track "$APTRL_TRACK_NPZ" --episodes 3 --workers 1 --out results/diagnostics/$RUN_ID/current-policy-baseline/v21
  python stage3_rl/evaluate_par.py --ckpt stage3_rl/runs_hf1/model_latest.pt --track "$APTRL_TRACK_NPZ" --episodes 3 --workers 1 --out results/diagnostics/$RUN_ID/current-policy-baseline/hf1
  ```

  Record failures by first failed segment and terminal contact. Do not overwrite `results/eval_summary.json` or `results/eval_best_traj.npz`.

- [x] **Step 2: Select the warm start from measured results, not checkpoint age or training reward.** Select the checkpoint with the best median `max_x`; break ties by fewer body-contact failures, then higher number of segments crossed. Record the selection and all four summaries in `stage3_rl/experiments/$RUN_ID/baseline-selection.json`.

- [ ] **Step 3: Run a small, isolated PPO pilot.** Use `--batch-target 4096 --workers 4 --epw 1 --epochs 2 --iters 25`, the selected checkpoint as `--init`, the explicit `--probe-track`, and a new output directory. Do not reuse `runs_hf1`, `runs_f2`, or any existing run folder.

  Example command:

  ```bash
  python stage3_rl/ppo_train.py \
    --init stage3_rl/runs_f2/model_latest.pt \
    --out stage3_rl/runs_repro_pilot_$RUN_ID \
    --probe-track "$APTRL_TRACK_NPZ" \
    --batch-target 4096 --workers 4 --epw 1 --epochs 2 --iters 25
  ```

  Replace the example `--init` with the checkpoint selected in Step 2.

- [ ] **Step 4: Compare the pilot with the selected baseline using three episodes each.** Run `python stage3_rl/evaluate_par.py --ckpt stage3_rl/runs_repro_pilot_$RUN_ID/model_best.pt --track "$APTRL_TRACK_NPZ" --episodes 3 --workers 1 --out results/diagnostics/$RUN_ID/ppo-pilot/evaluation`. Evaluate the pilot checkpoint with exactly the same track, lane criterion, commands, wrapper behavior, episode cap, and deterministic policy mode as the baseline. Continue training only if its median furthest progress increases or its completed-segment count improves without a new earlier failure mode. Otherwise stop the pilot, keep all artifacts, and return to failure diagnosis rather than increasing training duration blindly.

---

### Task 6: Train candidates in isolated runs and keep an auditable promotion record

**Files:**
- Create: one `stage3_rl/runs_repro_candidate_$CANDIDATE_ID/` directory per candidate
- Create: `stage3_rl/experiments/$RUN_ID/candidate-comparison.json`

**Interfaces:**
- Each candidate consumes one explicit initialization checkpoint and the accepted track/evaluator configuration.
- A candidate is eligible for full evaluation only after it improves on the measured baseline under the identical three-episode probe.

- [ ] **Step 1: Start a new candidate from the selected checkpoint or the successful pilot.** Set `INIT_CKPT` to the checkpoint path recorded in `stage3_rl/experiments/$RUN_ID/baseline-selection.json` or the accepted pilot evaluation, and set `CANDIDATE_ID=$(date -u +%Y%m%dT%H%M%SZ)`. Run `python stage3_rl/ppo_train.py --init "$INIT_CKPT" --out stage3_rl/runs_repro_candidate_$CANDIDATE_ID --probe-track "$APTRL_TRACK_NPZ" --seed 1 --batch-target 4096 --workers 4 --epw 1 --epochs 2 --iters 100`. Keep periodic `latest`/`best` files inside that directory only. Do not alter training reward, terrain, termination, or evaluation command speeds during a run.

- [ ] **Step 2: Probe at the existing 25-iteration cadence.** At each probe, save iteration, seed, mean reward, mean training success, probe max-x, completed segments, and failure counts to `probe_metrics.jsonl` in the run directory. Retain the checkpoint with the best repeated evaluation progress, not merely the highest training reward.

- [ ] **Step 3: Re-evaluate each candidate checkpoint with the three-episode fixed protocol.** Record its metrics and hashes in `stage3_rl/experiments/$RUN_ID/candidate-comparison.json`. Only promote a checkpoint if repeated episodes improve the prior candidate without changing any evaluation condition. If it improves but remains below the final target, use it as the next initialization and run another isolated 100-iteration batch with a new `CANDIDATE_ID`.

- [ ] **Step 4: Stop and request a decision if a complete 100-iteration batch fails to improve.** Do not respond to a flat result by silently easing geometry, weakening the lane criterion, changing the success location, or extending training indefinitely. Preserve logs/checkpoints and provide the evidence needed to choose between deeper controller diagnosis and an explicitly revised project specification.

---

### Task 7: Run final acceptance evaluation, record a successful episode, and update the report

**Files:**
- Modify: `REPORT.md`
- Create: `results/acceptance/$RUN_ID/`
- Create: `video/aptrl_success_$RUN_ID.mp4`

**Interfaces:**
- Consume: one promoted checkpoint, the accepted NPZ track, exact evaluator configuration, and the diagnostic metadata pipeline.
- Produce: immutable 100-episode results, manifest, best successful trajectory, exact-seed video replay, and a filled report section.

- [ ] **Step 1: Run 100 episodes into a new acceptance directory.** Set `PROMOTED_CKPT` to the path recorded in `stage3_rl/experiments/$RUN_ID/candidate-comparison.json`, then run `python stage3_rl/evaluate_par.py --ckpt "$PROMOTED_CKPT" --track "$APTRL_TRACK_NPZ" --episodes 100 --workers 16 --out results/acceptance/$RUN_ID --video video/aptrl_success_$RUN_ID.mp4`. Use deterministic policy, fixed command schedule, identical acceptance lane rule, and the recorded seed policy. Confirm the episode count is exactly 100 and all results are present in the manifest.

- [ ] **Step 2: Apply the strict success threshold.** Accept only if at least 91 of 100 episodes reach `x >= 100m`, remain within `|y - 4.0| < 1.9m` throughout, and terminate without a body-contact/fall/attitude/off-lane failure. If the threshold is missed, the goal remains unmet; return to Task 5/6 diagnosis and do not label the run a pass.

- [ ] **Step 3: Verify the video from a successful episode.** Confirm the evaluator replayed the selected successful episode with its exact seed and configuration. Verify the MP4 is non-empty with `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 video/aptrl_success_$RUN_ID.mp4`, inspect the rendered video, and match its run ID/checkpoint to a successful row in `episodes.jsonl` under that acceptance directory.

- [ ] **Step 4: Fill `REPORT.md` with observed values only.** Record checkpoint path/hash, code and track hashes, episode count, success count/rate, per-segment reach, failure breakdown, actual `meta["fixes"]`, command/wrapper settings, video path, and any residual differences from the paper. Do not claim a pass when the 91/100 threshold or video verification is absent.

- [ ] **Step 5: Final verification.** Run the full unit test suite and verify the manifest, result JSON, trajectory, video, and report all reference the same checkpoint and run ID. Leave all prior artifacts intact.

---

## Execution Notes

- Run all commands from `aptrl_go2w` and set `APTRL_TRACK_NPZ` to the existing desktop NPZ before tests and evaluation. Set `RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)` once before Task 5 and retain it for baseline, pilot, candidate comparison, acceptance, and video paths; use a new ID if an output path already exists. Set a new `CANDIDATE_ID=$(date -u +%Y%m%dT%H%M%SZ)` for each candidate run.
- `python -m unittest discover -s tests -p 'test_*.py' -v` is the required local regression command.
- The plan deliberately separates specification/test work from GPU/CPU-intensive training. The final completion-rate goal is not guaranteed by implementation alone; if measured candidate performance does not improve, stop and present the new evidence rather than making undocumented task changes.
- This directory has no Git metadata. The plan and outputs are durable files, but they cannot be committed unless the owner later places the project under version control.
- **2026-11-13 follow-up:** The reward/reset/push defects and TO phase/contact/reachability/wheel-geometry issues recorded in `docs/superpowers/research/2026-11-13-apt-rl-learning-failure-audit.md` have been corrected and tested. Final versioned TO/TVAE artifacts are under `stage1_to/dataset_periodic_20261113T153646Z/` and `stage2_tvae/candidates/20261113T153754Z/`; held-out MuJoCo open-loop cycles pass 181/181 trot and 88/88 bound. A 25-update from-scratch flat pilot under `stage3_rl/experiments/20261113T154146Z/prior-v5-flat-pilot/` did not improve over its same-decoder initialization (both 0/3, start-area body contact) and was stopped. A held-out reference latent sequence from ordinary reset completed 29/29 and 31/31 trot steps at 0.5/1.0m/s but bound at 1.5m/s failed at step 40/57 (`prior-normal-reset-probe.json`). The latest 74-test suite passes with `APTRL_TRACK_NPZ` set. No candidate has passed promotion or final acceptance.
- The former Task 5 example that initializes from v21 is no longer valid with the new decoder: v21 has no matching provenance for this candidate decoder and PPO now rejects mismatches. Keep v21 as a legacy-decoder baseline only; any new candidate must use the exact decoder hash it was trained with. Do not resume rejected pilots. The same-decoder 25-update flat gate has now failed; do not extend it. Before changing PPO initialization/training architecture, agree on an actor-bootstrap design and test gate; the current evidence says a fixed trot latent sequence can briefly start from reset, while bound transition and learned actor startup remain unresolved. The final acceptance target is unchanged.
