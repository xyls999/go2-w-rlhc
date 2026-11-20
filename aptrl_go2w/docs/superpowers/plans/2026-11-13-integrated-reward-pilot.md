# Integrated material reward pilot

**Goal:** Test a versioned reward combining verifiable current legacy progress/alive terms with the existing paper-adapted and bounded stair shaping, without claiming to reconstruct the original v21 reward or paper exactly.

**Evidence:** `stage3_rl/train_v21.log` stores aggregate training reward, not per-term source or fixed-track stair results. The project has no Git metadata and no known v21 reward-source snapshot. Historical audit `docs/superpowers/research/2026-11-13-apt-rl-learning-failure-audit.md` identifies the old local terminal progress bug; never restore it.

## Invariants and design

- Opt-in `integrated_v1` applies only to material full-track training. Legacy, `paper_adapted` and `stair_v1` retain their prior behavior. Never overwrite prior checkpoints, maps or logs.
- Reuse paper-adapted base terms and terminal failure penalty -1.5. Add the current code's train-only `forward_progress = clip(v_world_x / max(vx_cmd,0.5), -0.5, 1.5)` and `alive = 0.5`; these are current legacy terms, not proven bit-for-bit v21 rewards.
- Keep `stair_v1` shaping: bounded one-shot stair progress/clearance/contact-recovery/stuck signals and layout check. Do not add hard-coded old-track milestone bonuses or absolute-position terminal rewards. This is a combined candidate, not a single-factor ablation or exact paper method.
- DONE: TDD profile/CLI checks, term presence and values, no legacy milestone/new-max, no added training bonuses during eval. Full suite: `integrated-reward-tests.stderr.log`, 131 tests OK.
- DONE / GATE FAILED: predeclared `integrated-reward-pilot-gate.md`; fresh 10-update pilot from the previously startup-gate-passing checkpoint with matched decoder. Paired evaluation stair crossings 0/3 both before and after; start-line 3/3 ran 1000 steps before and after. Logs show 8000 steps, 127 from stair runups, mean alive +0.5 and forward_progress +0.9114; no stair-specific term fired. Full data `integrated-reward-pilot-gate-result.json` and `integrated-reward-pilot-evaluation.jsonl`. Stop; do not extend or claim stairs learned. Low stair-runup sampling and absent stair signal are observed facts; they do not establish the reward as the sole cause.
