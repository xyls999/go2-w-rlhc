# Material-track Training Integration Plan

**Goal:** Train a hash-matched prior policy on the user-approved compact material track, with measured action limits, without modifying the original track or overwriting earlier artifacts.

**Spec:** `docs/superpowers/specs/2026-11-13-robocon-material-track-design.md` and `results/diagnostics/robocon-material-preview-20261113-v4/manifest.json`.

**Constraints:** The project has no Git metadata. The XML preview is not an NPZ course and cannot be passed to `load_npz_track`; a blind run on the old map is invalid. Existing source hashes differ from earlier evaluations. Never resume a failed pilot; record fresh matched before/after runs. Decoder and policy hashes must match.

## Preflight

- [x] Establish actual loader behavior and versioned preflight: `results/diagnostics/motion-review-20261113-v1/new-terrain-training-preflight.json`.
- [x] Establish matched action-clamp diagnostics: auxiliary clamp ±1 and ±2 never activated; ±0.2 activated on about 22–27% of outputs and remained alive for 1000 steps in each of three current-code diagnostics. These are not new-track results or proof of safe joint motion.
- [x] Implement explicit XML-material-track integration in `common/material_track.py` and `stage3_rl/env.py`; preserve original NPZ loading path. The existing 2.5D observation reports upper surfaces and cannot represent underpasses fully.
- [ ] Complete collision, sensor, out-of-lane and bridge routing tests. MuJoCo ray comparisons pass at nine points (rim/hole, stairs, ramps, bridge and frame), and the robot plus material geom names load; obstacle traversal and routing are not verified.
- [ ] Complete action-safety controls. Material XML currently activates an experimental auxiliary clamp ±0.2 in the environment (including workers and custom evaluation); run config records it. Rate limits, explicit opt-in controls, joint-limit and torque-saturation coverage, and standard serial evaluation integration remain open. The PPO policy samples raw actions and the environment deterministically clips them; this is not a truncated distribution.
- [ ] Measure robot standing and feasible low-posture clearance before treating the 0.25 m frame as feasible. Five steps are 0.60 m total; test approach and landing, don't claim compatibility without measurements.
- [x] Declare initial/final pilot gate before launch in `results/diagnostics/motion-review-20261113-v1/material-first-pilot-gate.md`.
- [x] Run first 5-update integration pilot in `stage3_rl/experiments/20261113T160601Z/actor-bootstrap/material-track-pilot-v1/`. This was launched before all preceding safety gates were complete and is therefore exploratory only, not evidence of safe or effective obstacle training.
- [x] Compare same seeds under material XML and 0.2 clamp: gate **failed**. Initial median max-x 15.129 m; latest 0.729 m; latest 3/3 body-contact at 33–59 steps. Both initial and final probes ended before the first obstacle. Full data: `results/diagnostics/motion-review-20261113-v1/material-pilot-v1-evaluation.jsonl`, `material-pilot-v1-gate-result.json`. Stop; no automatic continuation.
- [ ] Diagnose the regression and complete remaining safety gates before proposing a new run. No 100-episode success claim or video without actual success.
