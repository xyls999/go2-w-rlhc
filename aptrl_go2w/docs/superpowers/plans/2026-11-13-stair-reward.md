# Stair reward implementation plan

**Goal:** Introduce a separately selected, testable stair reward for the versioned MuJoCo material course while preserving all historical reward profiles and artifacts.

**Sources:** `docs/superpowers/research/2026-11-13-stair-reward-sources.md`. Local model API returned 401; no model output was obtained. Only copied general concepts, not source code.

**Constraints:** Do not alter old reward defaults, original map, old checkpoints or 91/100 acceptance. Stair geometry and 0.25 m clearance are not yet proven traversable. Contact bonuses cannot reward repeated contact, motion outside the stair corridor, or terminal falls. Weights are provisional task-specific experimental values, not copied or claimed paper-optimal. No automatic training.

1. DONE: state-machine tests for bounded monotonic progress, corridor/terminal guard, stuck time, actual-contact-gated lift, one-shot limits and four-wheel lateral clearance. These are synthetic tests, not physical stair traversal.
2. DONE: `stage3_rl/stair_reward.py`, opt-in `stair_v1` material-only profile, component logs and fail-closed stair layout validation. Legacy defaults and PPO distribution unchanged. Coefficients are experimental, not paper values.
3. DONE / GATE FAILED: `stair-pilot-tests.stderr.log` reports 128 tests OK. A 10-update predeclared pilot used mixed start-line/tire/stair runups and sampled 354 stair-runup steps among 8000 logged steps. No stair reward terms were emitted, so this run does not show an effect of the added reward. On seeds 9243–9245, initial and final stair crossings were both 0/3; start-line retention regressed from 3/3 to 1/3 full 1000-step runs. `stair-reward-pilot-gate-result.json` records FAIL; do not continue this pilot.
4. Diagnostic `stair-pilot-contact-trace.jsonl`: on latest checkpoint seed 9243, FR wheel first contacts `stair_up_0` at step 79 while base x=30.697m; later at step 122 RR calf hits track and terminates. Contact occurred, but the required forward progress/lift never triggered any reward; 0/3 stair success. Do not claim the reward was successfully trained or the physical geometry is solvable. Investigate feasible recovery, action distribution, and denser non-exploitable reward before a new run.
