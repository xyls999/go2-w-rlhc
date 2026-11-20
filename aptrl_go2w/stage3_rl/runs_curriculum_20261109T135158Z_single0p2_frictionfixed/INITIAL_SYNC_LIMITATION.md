# Training-origin limitation discovered before the mixed-retention pilot

`ppo_train.main` loaded v21 into the learner but did not broadcast actor/encoder/normalizer to workers until after the first update. The first collected batch therefore used independent random worker networks. This run does **not** satisfy the strict requirement that all pilot sampling starts from v21; do not use it as an initialization or promotion candidate.

The saved candidate's measured 5/10 gap and full-track results remain measurements of that actual checkpoint, not evidence that training was cleanly initialized. The original v21 checkpoint and its evaluation baselines are unaffected. See `stage3_rl/experiments/20261113T131910Z/test-initial-sync-red.log` for real four-worker reproduction and `test-initial-sync-green.log` for the startup-broadcast regression test after repair. This note adds provenance; no old weights or evaluations are overwritten.
