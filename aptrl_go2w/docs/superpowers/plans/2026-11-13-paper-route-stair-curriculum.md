# Paper-route stair curriculum implementation plan

**Goal:** Reuse the existing matched Go2-W prior and build a versioned gradual stair-height PPO curriculum, with flat retention and independently evaluated target geometry.

**Architecture:** Keep the original material XML intact; derive separate scenes by lowering both ascent and descent steps and rescaling ramps consistently. Use the existing `paper_adapted` reward, not `stair_v1`; existing policy, decoder, observation interfaces and physics are reused. Log exact source/map hashes and compare fixed-seed results before and after every bounded pilot.

**Tech stack:** Python, NumPy, MuJoCo, PyTorch PPO.

**Spec / sources:** User-approved paper-inspired route; APT-RL §4.3 and S3 / Table S3: <https://arxiv.org/html/2607.13579>; `docs/superpowers/research/2026-11-13-apt-rl-learning-failure-audit.md`.

## Global constraints

- Preserve all previous datasets, checkpoints, scene files and videos; project has no Git metadata.
- Do not continue failed pilots. Initialize from startup-gate-passing `material-paper-adapted-pilot-v1/model_latest.pt` with its recorded decoder SHA-256.
- The source paper uses HOUND/Isaac Gym and much larger batches; this is a small Go2-W/MuJoCo adaptation, not faithful scale or a true-robot claim.
- Existing material action clamp and imperfect paper-adapted rewards differ from paper; keep them fixed in the first experiment, record and do not attribute failure to a single cause.
- Require independent evaluation and flat retention; no auto-resume after failure; do not assert 91/100 or video success.

### Task 1: Versioned stair levels

- [x] Add failing tests and implement `stage3_rl/material_stair_curriculum.py` with 0.05/0.075/0.10/0.12m steps and adjusted ramps. Tests cover geometric heights, endpoints, non-stair preservation and immutable output.
- [x] Create `stage3_rl/experiments/20261113T160601Z/actor-bootstrap/material-stair-levels-v1/`; manifest records source/generator/scene hashes. Level-four ramp is recomputed to the same nominal dimensions, not byte-identical to source.

### Task 2: Feasibility and baseline

- [x] Compile every scene and compare eight collision vertical rays per scene to sensor heights; this does not prove underpass representation or general collision equivalence.
- [x] `paper-stair-baseline.jsonl`: original retained checkpoint produced 0/3 stair crossings on each of the four levels. On 5cm stairs, removing aux clamp also yielded 0/3. Paired start-line retention is recorded in the pilot evaluation.

### Task 3: Bounded level-one PPO pilot

- [x] Predeclared `paper-stair-l1-pilot-gate.md`, then trained a fresh 10-update 5cm level pilot from the stable checkpoint with its matched decoder and paper-adapted reward.
- [x] Matched initial/final evaluation: 0/3 crossings both before and after; start-line retention 3/3 at 1000 steps before and after. Training logs record 984 stair-runup and 1236 tire-runup steps. Full results: `paper-stair-l1-pilot-gate-result.json`.
- [x] Gate FAILED: stop; do not extend this pilot or promote to 7.5cm. No 100-episode or full-course success was claimed. Diagnose wheel-riser interaction and the limited prior/policy optimization before designing another experiment.
