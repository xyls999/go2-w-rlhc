# APT-RL Go2-W Actor Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the policy actor from validated 0.5/1.0m/s trot latent demonstrations, then allow bounded PPO pilots only when independent rollout gates pass.

**Architecture:** A standalone collector records ordinary-reset MuJoCo observations paired with fixed TVAE posterior-mean trot latents, zero auxiliary targets, and trot gate targets. A separate behavior-cloning stage trains only `ExteroEncoder` and `Actor`, then saves a normal PPO-compatible checkpoint with the exact decoder hash. A separate flat-rollout gate validates that checkpoint before PPO.

**Tech Stack:** Python, NumPy, PyTorch, MuJoCo, `unittest`, existing `stage3_rl.env.Go2WEnv`, `stage3_rl.ppo_train.Actor/ExteroEncoder/Critic/RunningNorm`, and evaluator decoder-provenance checks.

**Spec:** `docs/superpowers/specs/2026-11-13-actor-bootstrap-design.md`

## Global Constraints

- Use only TO/TVAE candidate `stage1_to/dataset_periodic_20261113T153646Z/` and `stage2_tvae/candidates/20261113T153754Z/` unless a newly versioned replacement is explicitly created.
- Bootstrap demonstrations use only trot speeds 0.5 and 1.0m/s; do not claim bound startup or gap crossing is solved.
- Demonstration and bootstrap outputs use new empty directories; never overwrite existing datasets, models, evaluations, or videos.
- A checkpoint must include the exact decoder SHA-256 and be rejected with any other decoder.
- Do not initialize from v21 or any rejected pilot checkpoint.
- Gate A requires, for each speed, at least 9/10 held-out reset episodes to run 1000 steps without body-contact/fall/off-lane and to move at least 4m (0.5m/s) or 8m (1.0m/s).
- Gate B is at most 25 PPO updates; promote only if same-decoder, same-seed 3-episode median max-x increases and no earlier/new failure mode occurs. Otherwise stop; no automatic extension.
- Final acceptance remains original fixed track, `pose-v1`, at least 91/100 successful episodes and an exact-checkpoint verified success video.
- The repository is not a Git repository; do not initialize Git or claim commits.

---

### Task 1: Deterministic teacher/action and episode-split utilities — COMPLETE

**Files:** `stage3_rl/actor_bootstrap.py`, `tests/test_actor_bootstrap.py`.

- [x] Test-first utility coverage: teacher action 29-D layout, deterministic complete episode-level split, and PPO actor feature ordering/dimension.
- [x] Implement `make_teacher_action`, `split_episode_indices`, `make_actor_input`; targeted tests pass.

### Task 2: Immutable ordinary-reset demonstrations — COMPLETE

**Files:** `stage3_rl/actor_bootstrap.py`, `tests/test_actor_bootstrap_collection.py`.

- [x] Test-first coverage for decoder/checkpoint pairing, deterministic trajectory/phase selection, ordinary reset (no reference state injection), command observation alignment, early-termination rejection, and output-directory protection.
- [x] Implement collector with trajectory hashing, per-step raw observations/teacher labels, JSONL/manifest provenance, and fail-closed minimum of 10 complete episodes per speed.
- [x] The first real collection was invalidated (first observation carried the reset's default 1.5m/s command); it is preserved with `INVALID.md` and not used.
- [x] Corrected collection: `stage3_rl/experiments/20261113T160601Z/actor-bootstrap/demonstrations-v2/`, 20/20 complete 1000-step demonstrations at each of 0.5/1.0m/s. Decoder hash matches the TVAE checkpoint's embedded export hash.

### Task 3: Actor cloning and PPO-compatible checkpoint — COMPLETE

**Files:** `stage3_rl/actor_bootstrap.py`, `tests/test_actor_bootstrap_checkpoint.py`.

- [x] Test-first episode split, per-head validation metrics, PPO-compatible keys, and decoder mismatch rejection.
- [x] Train only encoder/actor; use train-only observation normalization; save fresh critic, normalizer, exact decoder hash and provenance.
- [x] 30 epochs, batch 1024, lr 3e-4, CUDA. Held-out validation: latent MSE 0.007106, aux MSE 0.0001181, gait BCE 0.00002530. Checkpoint: `stage3_rl/experiments/20261113T160601Z/actor-bootstrap/bootstrap-v1/model_bootstrap.pt`.

### Task 4: Held-out flat reset Gate A — COMPLETE

**Files:** `stage3_rl/validate_bootstrap.py`, `tests/test_validate_bootstrap.py`.

- [x] Test-first gates for both speed-specific displacement thresholds, strict lane, early terminal, minimum sample count, and immutable output.
- [x] Implement seeded ordinary-reset closed-loop evaluation and episode/summary hashes.
- [x] Gate A passed: 10/10 at both speeds, 1000 steps each. Evidence: `results/diagnostics/20261113T134122Z_gaprough/actor-bootstrap-gate-v1/summary.json` and `episodes.jsonl`.

### Task 5: First bounded PPO pilot (Gate B) — COMPLETE / PROMOTED

**Files:** `stage3_rl/experiments/20261113T160601Z/actor-bootstrap/pilot-v1/`; corresponding diagnostics under `results/diagnostics/20261113T134122Z_gaprough/`.

- [x] Run at most 25 updates, 4 workers, 1 env/worker, batch target 4096, 2 epochs, mixed stage, episode length 900, exact bootstrap decoder.
- [x] Evaluate initial and final checkpoints on identical `pose-v1` seeds 9043–9045, track, command wrapper and 4000-step cap.
- [x] Gate B passed as specified: median max-x increased from 0.514274m to 0.615740m (+0.101465m); all episodes still ended in `body_contact`, but later (22/23/25 steps initially vs 42/31/34 steps finally), with no new/earlier failure mode. Evidence: `actor-bootstrap-pilot-v1-gateB.json`, initial/latest `episodes.jsonl` and manifests.
- [x] Full suite at that point: 92 tests pass (`results/diagnostics/20261113T134122Z_gaprough/test-suite-actor-bootstrap.stdout.log`).

### Task 6: Second mixed-stage pilot — COMPLETE / STOPPED

- [x] New 25-update mixed-stage pilot `pilot-v2` started from `pilot-v1/model_latest.pt`.
- [x] Same-seed fixed-track evaluation failed its predeclared gate: initial median max-x 0.615740m; final max-x approximately 0.6m on all three seeds, with seed 9043 failing after 26 rather than 42 steps. All were body contacts, but the earlier failure violates the gate. The in-training probe's one episode reaching 22.773m did not generalize to the fixed comparison seeds.
- [x] No automatic extension. Diagnose before further training.

**Diagnosis:** On fixed-track seeds 9043–9045, the v1 actor from ordinary fixed-track reset at an explicit 1.0m/s command completed 1000 steps on all three episodes (about 9.3–9.7m displacement); the official 1.5m/s command failed around x=0.6m. This supports command-conditioned brittleness as a relevant variable, not a unique causal proof. Evidence: `results/diagnostics/20261113T134122Z_gaprough/speed-mismatch-diagnostic.jsonl`, `actor-command-output-comparison.jsonl`, `actor-bootstrap-pilot-v2-{initial,latest}-eval/`, and `pilot-v2/probe_metrics.jsonl`.

### Task 7: Targeted fixed-1.5m/s flat pilot — COMPLETE / STOPPED

A train-only `--command-speed` option was added, restricted to flat training and the approved discrete speed grid; it does not change official evaluation. TDD coverage verifies invalid profiles/speeds and the first observation's command. The complete suite passed 96 tests with the fixed-track environment configured.

- [x] Run 25 updates maximum, 4 workers, batch-target 4096, 2 epochs, `train_stage=flat`, `command_speed=1.5`, initialized from `pilot-v1/model_latest.pt`, exact decoder hash.
- [x] Evaluate initial/final on identical `pose-v1` seeds 9043–9045. Gate C failed: initial median max-x 0.615740m; final around 0.5m; all three final episodes body-contacted in 23–26 steps, earlier than initial 31–42 steps.
- [x] Stop; do not extend the run. Evidence: `actor-bootstrap-pilot-speed15-v1-{initial,latest}-eval/` and `pilot-speed15-v1/probe_metrics.jsonl`.

### Task 8: Ramped-command flat pilot — COMPLETE / STOPPED

- [x] Add TDD-tested flat-only ramp configuration and per-step environment schedule; keep fixed-track evaluator unchanged.
- [x] Run at most 25 PPO updates, 4 workers, 1 env/worker, batch-target 4096, 2 epochs, 1.0→1.5m/s over 500 steps.
- [x] Gate D failed on official fixed-track evaluation: baseline median max-x 0.615740m; final median 0.528029m, with all episodes body-contacting earlier (23/25/27 vs 42/31/34 steps). Do not continue this run.
- [x] Full suite passed 101 tests (`test-suite-command-ramp.stdout.log`).

### Task 9: Fixed-speed action-bias diagnostics — COMPLETE

- [x] In fixed-track command-output sweep, the v1 policy at command1.0 ran 1000 steps on all 3 seeds; direct command1.1 was unstable on 2/3, 1.2–1.5 failed all 3. A 1.0→1.5m/s external command ramp was stable 3/3 for 1000 steps, but it is not the official evaluation protocol.
- [x] TDD-tested a versioned checkpoint transformation subtracting 1.0 from actor auxiliary calf outputs (action indices 18,21,24,27); no source checkpoint is overwritten.
- [x] Matched decoder / official pose-v1 3-seed diagnostic with this bias reached x≈26.7m on all seeds before `attitude`, versus x≈0.5–0.7m unmodified. The first 1000-step bias probe advanced to x≈14.3m on all seeds; full-track probes without disabling the bias later reached x≈22.8–26.8m. This is not a learned skill or success.
- [x] Local rough-slope reset gate with the biased policy remained 1/10; local reset lacks the run-up/momentum seen on the full track. Do not promote from this result.

### Task 10: Calf-bias PPO pilot — COMPLETE / STOPPED

**Gate E:** Initialize from `aux-calf-bias-minus1-v1.pt`, run at most 25 mixed-stage updates, and compare initial/final on identical official `pose-v1` seeds 9043–9045. Require final median max-x >=30.0m, no seed failing earlier than its calf-bias baseline, and no new failure reason.

- [x] Run at most 25 updates, 4 workers, 1 env/worker, batch-target 4096, 2 epochs, exact decoder hash.
- [x] Evaluate initial/final on identical seeds. Gate E failed: initial max-x about 26.7m on all seeds; final max-x about 22.7m, 0.6m, 1.0m, with earlier body-contact on seeds 9044/9045 and changed failure reason. Stop; do not extend.
- [x] Full suite after source changes: 103 tests pass (`results/diagnostics/20261113T134122Z_gaprough/test-suite-aux-bias.stdout.log`).

### Task 11: Bound/gap switch and residual diagnosis — COMPLETE / BLOCKED

- [x] Manual bound-latent switch before the gap did not reliably improve progress; see `bound-switch-gap-feasibility.jsonl`.
- [x] Calf bias works as a startup aid but did not survive the transition into rough slope/gap robustly; local rough-slope reset gate was only 1/10.
- [x] Do not start another training run on the present bootstrap/aux-bias design. Current route is blocked on a materially new high-speed startup/rough-slope/gait-transition design; fixed-track 91/100 remains unmet.

---

## Plan Self-Review

- The approved actor-bootstrap spec, collection, cloning, provenance, Gate A, and bounded pilots have been implemented and tested.
- Every failed pilot was stopped at its predeclared gate; no failed process was extended. Diagnostics are preserved in `results/diagnostics/20261113T134122Z_gaprough/actor-bootstrap-followup-diagnosis.json`.
- The best diagnostic reaches x≈26.7m, not the 100m finish. No 100-episode final evaluation or success video exists. Passing a short-term progress gate is not reproduction success.

