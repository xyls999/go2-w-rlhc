# Stair reward source review

## First-party references

1. [legged_gym `legged_robot.py`](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py): `_reward_tracking_lin_vel` uses an exponential of velocity error; `_reward_collision` counts configured penalised body contacts above a force threshold; `_reward_termination` distinguishes timeout; `_reward_action_rate` penalizes action change. Its `_update_terrain_curriculum` promotes/demotes according to travel distance. These are its specific functions, not proof that those settings transfer directly to Go2-W.
2. [Isaac Lab official velocity environment](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py): a sample task combines velocity tracking, joint acceleration, action rate, foot air time and undesired-contact terms; it terminates on base contact separately. Weights are task-specific, not universal. [Isaac Lab rewards implementation](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/rewards.py).
3. Local `paper_text.txt` APT-RL section 4.3 and S3/Table S3: velocity tracking, auxiliary smoothness, collision/style terms and graduated terrain difficulty; APT-RL does not prescribe this project's tire/stair event bonuses. The current `paper_adapted` profile is not exact Table S3.

## Project-specific implications

- `stage3_rl/env.py` already penalizes foot brushing, joint acceleration, auxiliary changes and body shaking; duplicates must be avoided. `_check_done` currently ends an episode on trunk/leg bad contact. Routine wheel-ground or wheel-riser contact must not be classified wholesale as a failure.
- Stair geometry in the currently previewed material XML: ascending step boxes cover x=31.0–32.6m, 0.12m rise each, 0.8m width; upper platform to x=33.4m. Descending boxes cover x=41.05–42.65m. These are versioned preview coordinates, not general terrain semantics.
- Propose a separate opt-in material stair reward rather than changing legacy or `paper_adapted`: bounded one-time qualifying progress; no credit for backward-forward oscillation, leaving the stair corridor or failing in the same step; bounded stuck detection with elapsed time and forward command only; conditional clearance shaping after actual wheel/edge contact if reliable contact context can be captured. No claim any chosen initial coefficient is validated or optimal.
- Evaluate against the same starting checkpoint and held-out initial poses; log per-term reward, contacts, crossings, stuck and previously stable flat/tire behavior. Change one reward factor at a time. Preserve old checkpoints and tracks, do not infer 100m success from local tests.

## Local model access

Unsloth Studio and `llama-server` were observed listening on loopback port 1189. `/health` returned status ok, but `/v1/models` returned HTTP 401. No credentials were accessed or bypassed; no output was obtained from the local model.
