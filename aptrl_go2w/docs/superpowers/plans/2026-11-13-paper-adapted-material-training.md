# Paper-adapted material training (retain existing prior)

**Goal:** Reuse matched Go2-W TO/TVAE torque decoder and the pre-pilot policy; isolate an explicit paper-inspired reward profile and train/test it without claiming exact paper reproduction.

**Evidence:** `paper_text.txt` §4.3 and Table S3, `docs/superpowers/research/2026-11-13-apt-rl-learning-failure-audit.md`. The current material environment is a 104 m preview adaptation, not the paper's Isaac Gym setup. Existing decoder `5d7e8d8612874e511863eb5149378fef11f339a3c340e316d5a54f7c974d3b17` is reused. Do not continue either failed material PPO pilot.

1. DONE: tests and explicitly selected material-only `paper_adapted` profile remove old-track milestone/progress/alive bonuses and use linear tracking weight 1.5 / failure penalty -1.5; legacy default is preserved. It is an adaptation, not exact paper Table S3: velocity normalization, smoothness, style terms, PD gains, simulator, Go2-W wheels and action clamp differ.
2. DONE: profile wired through CLI, worker and training metadata; per-worker chunk-level raw reward component sums and terminal counts stored alongside checkpoints. Raw sampled PPO actions and deterministic environment clipping remain distinct.
3. DONE: prior/decoder pairing checked, MuJoCo geometry sample ray tests passed, 114 tests passed (`paper-adapted-tests-v2.stderr.log`); pilot gate declared at `paper-adapted-pilot-gate.md`. Five-update pilot `material-paper-adapted-pilot-v1` passes only the short 1000-step startup gate: initial 1/3 falls, final 0/3 falls; first terrain not reached.
4. DONE: paired observations and reward log summary in `paper-adapted-pilot-gate-result.json`. Additional 2500-step material probe `paper-adapted-pilot-material-probe.jsonl` ends 3/3 around x=24.445 due to RR calf contacting `track_geom`; no verified tire crossing. Train logs have 2500 collected steps and no logged terminal events; interpretation of aggregate reward terms alone cannot establish causality.
5. OPEN: develop and review per-obstacle curriculum, compare model initialization and evaluate safe reset / entry paths, hard frame clearance, bridge non-bypass and physical traversal. No automatic extension; no claim of exact APT-RL or 100 m success.

