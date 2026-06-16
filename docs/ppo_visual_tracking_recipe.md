# PPO visual tracking recipe

This note records the current PPO recipes and open validation status for the
visible-offset tracking task. The policy observation contract is:

- retinal vision from the agent eyes
- the agent's own previous action/movement command
- no global position observation
- no contact observation
- no privileged action

Reward shaping may use simulator state during training, but clean evaluation disables
the dense shaping terms and scores only task completion, time penalty, and contact
failure.

## Moving-target binocular baseline

For moving targets, the current two-fixed-eye PPO candidate is:

```bash
bash scripts/train_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_contactterm_stop_success.sh seed=0
```

Important defaults:

- two fixed eyes matched to the original three-eye angular geometry:
  `num_eyes=[1,2]`, `lon_range=[-18.75,18.75]`, `fov=[45,67.5]`,
  `sensorsize=[0.008284271247461903,0.008909048505590654]`,
  `resolution=[20,30]`
- spatial CNN visual extractor, not the global-pooled small CNN
- relative body steering action
- previous-action observation enabled
- contact observation disabled
- target/world/self positions are not policy observations
- checker ground plane
- moving target waypoints sampled from the target reset region, not only literal
  `0` cells
- target speed defaults to `GOAL_SPEED=-0.45`, which maps to a slower physical
  forward speed than the earlier `-0.15` setting
- contact with walls truncates train and eval episodes
- stop training once deterministic clean eval mean reward reaches `25`

Geometry note: this is intended to match the original three-eye setup, not create
complete forward overlap. The original baseline used eye centers `[-30,0,30]`
with horizontal FOV `45`, giving union coverage `[-52.5,52.5]` and two adjacent
overlap zones totaling `30` degrees. The matched two-eye geometry uses centers
`[-18.75,18.75]` with horizontal FOV `67.5`, which gives the same union coverage
`[-52.5,52.5]` and one `30` degree binocular zone `[-15,15]`. This depends on
`MjCambrianMultiEye` propagating the top-level FOV/sensorsize settings into the
generated per-eye cameras.

The contact truncation is the critical MDP fix. Without it, PPO repeatedly discovers
a wall/edge attractor: the policy can park or swing at the boundary while collecting
enough dense approach/respawn signal to look superficially plausible. Contact
truncation does not add privileged information to the policy; it just makes boundary
collision a task failure instead of a stable behavior mode.

The spatial CNN is also important. The original small CNN uses global average pooling,
which destroys within-eye target location. Three fixed eyes can still encode coarse
bearing by eye identity, but two wider eyes need the extractor to preserve spatial
position inside each retinal image.

### Moving-target validation status

Matched two-eye geometry is now verified under corrected FOV propagation. The
geometry target is:

- left eye yaw `-18.75`, horizontal coverage `[-52.5,15]`
- right eye yaw `18.75`, horizontal coverage `[-15,52.5]`

The first corrected matched-geometry PPO run is not a healthy baseline despite
crossing the simple eval reward threshold:

| Seed | Run | Best eval block | Stop step | Eval returns | Behavior audit |
| --- | --- | ---: | ---: | --- | --- |
| 0 | `logs/2026-06-15/exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched3geom_contactterm_seed0_80k` | `vis_3` | 20000 | `[50.0, 0.0, 20.0, 30.0, 30.0, 30.0]` | false positive: stationary fraction `0.85-0.998`, median goal distance roughly `5-7` |

The dense video/contact sheet for `vis_3` shows long periods of parking or
near-parking, not robust pursuit. Treat this run as evidence that the geometry and
training loop execute, not as a successful behavioral baseline.

The prior run below is also not valid evidence for the intended matched-geometry
claim. It fixed the camera FOV propagation, but used centers `[-45,45]` with
horizontal FOV over `100`, creating the wrong coverage geometry.

| Seed | Run | Successful eval block | Stop step | Final eval returns | Notes |
| --- | --- | ---: | ---: | --- | --- |
| 0 | `logs/2026-06-15/exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_overlap_h512_resettarget_contactterm_seed0_80k` | `vis_2` | 15000 | `[50.0, 20.0, 50.0, 20.0, 40.0, 40.0]` | all eval episodes reached 512-step horizon |

Representative corrected-overlap rollout:

- `logs/2026-06-15/exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_overlap_h512_resettarget_contactterm_seed0_80k/evaluations/vis_2.mp4`

The earlier three-seed table below is not valid evidence for the binocular-overlap
claim. Those runs used the intended wrapper arguments, but the generated per-eye
cameras stayed at the default 45 degree FOV, leaving no binocular zone. Keep them
only as evidence that the training recipe can learn a two-peripheral-eye behavior.

| Seed | Run | Successful eval block | Stop step | Final eval returns | Video/analyzer |
| --- | --- | ---: | ---: | --- | --- |
| 0 | `logs/2026-06-15/exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_resettarget_contactterm_seed0_80k` | `vis_2` | 15000 | `[29.1, 80.0, 89.5, 30.0, 40.0, 49.3]` | border `0.0`, stationary `0.161` |
| 1 | `logs/2026-06-15/exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_resettarget_contactterm_seed1_80k` | `vis_1` | 10000 | `[10.0, 60.0, 0.0, 70.0, 40.0, 9.0]` | border `0.039`, stationary `0.316` |
| 2 | `logs/2026-06-15/exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_resettarget_contactterm_seed2_80k` | `vis_2` | 15000 | `[50.0, 50.0, 70.0, 60.0, 39.3, 30.0]` | border `0.0`, stationary `0.140` |

Representative videos/contact sheets:

- `logs/2026-06-15/exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_resettarget_contactterm_seed0_80k/evaluations/vis_2.mp4`
- `logs/2026-06-15/exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_resettarget_contactterm_seed0_80k/evaluations/vis_2_contact.png`
- `logs/2026-06-15/exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_resettarget_contactterm_seed1_80k/evaluations/vis_1.mp4`
- `logs/2026-06-15/exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_resettarget_contactterm_seed1_80k/evaluations/vis_1_contact.png`
- `logs/2026-06-15/exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_resettarget_contactterm_seed2_80k/evaluations/vis_2.mp4`
- `logs/2026-06-15/exp_frozen2_goal_moving_ground_spatialcnn_relative_respawn_matched2_h512_resettarget_contactterm_seed2_80k/evaluations/vis_2_contact.png`

One bookkeeping caveat: seed 2 wrote a negative `train_fitness.txt` despite the eval
monitor and rendered eval block showing success. For this baseline, use
`eval_monitor.csv`, rendered rollouts, and `best_model.zip`/early-stop checkpoint as
the authoritative evidence.

## Recommended frozen-eye recipe

Use:

```bash
bash scripts/train_frozen2_goal_visible_offset_ground_smallcnn_relative_turncredit_stop_success.sh
```

Important defaults inherited by the wrapper stack:

- small CNN visual extractor
- textured checker ground plane
- two fixed forward eyes, 20x20 resolution, 45 degree FOV
- fixed forward action via the constant-action wrapper
- relative body steering action
- action observation enabled
- contact observation disabled
- `init_quat=[1,0,0,0]`
- clean eval videos saved as MP4
- stop training once clean eval mean reward reaches `19.5`

The stop-on-success threshold is the critical stability piece. PPO repeatedly finds
the successful policy and can later update away from it. Stopping at the first clean
successful eval preserves a good checkpoint instead of trusting the final update.

## Frozen-eye seed evidence

All runs below used the same recipe with clean eval and stopped as soon as an eval
block reached 6/6 success.

| Seed | Run | Eval success blocks | Stop step | Final eval mean |
| --- | --- | --- | ---: | ---: |
| 0 | `logs/2026-06-13/exp_frozen2_goal_visible_offset_fast_stop_success_40k` | `[0, 6]` | 5000 | 19.62 |
| 1 | `logs/2026-06-13/exp_frozen2_goal_visible_offset_fast_stop_success_seed1_40k` | `[0, 6]` | 5000 | 19.60 |
| 2 | `logs/2026-06-13/exp_frozen2_goal_visible_offset_fast_stop_success_seed2_40k` | `[0, 0, 6]` | 7500 | 19.58 |
| 3 | `logs/2026-06-13/exp_frozen2_goal_visible_offset_fast_stop_success_seed3_40k` | `[0, 0, 0, 6]` | 10000 | 19.60 |
| 4 | `logs/2026-06-13/exp_frozen2_goal_visible_offset_fast_stop_success_seed4_40k` | `[0, 0, 0, 0, 6]` | 12500 | 19.60 |
| 5 | `logs/2026-06-13/exp_frozen2_goal_visible_offset_fast_stop_success_seed5_40k` | `[6]` | 2500 | 19.60 |

Representative videos:

- `logs/2026-06-13/exp_frozen2_goal_visible_offset_fast_stop_success_seed2_40k/evaluations/vis_2.mp4`
- `logs/2026-06-13/exp_frozen2_goal_visible_offset_fast_stop_success_seed4_40k/evaluations/vis_4.mp4`

## What did not solve brittleness

Reducing PPO update aggressiveness alone did not remove policy flipping:

- `lr=3e-4`, `clip_range=0.1`, `n_epochs=4`, `ent_coef=0.01`,
  `target_kl=0.03` produced eval success blocks
  `[0, 0, 0, 0, 0, 0, 6, 0, 6, 6, 6, 0]`.
- `lr=3e-4`, `clip_range=0.1`, `n_epochs=4`, `ent_coef=0.0`,
  `target_kl=0.01`, `max_grad_norm=0.3` produced
  `[0, 0, 0, 0, 0, 0, 6, 0, 6, 0, 0, 6]`.

These variants still alternated between clean success and complete failure after
acquisition. The practical robust recipe is therefore checkpoint-and-stop on clean
success, not final-policy selection after a fixed training budget.

## Recommended actuated-eye recipe

The matching actuated-eye wrapper is:

```bash
bash scripts/train_actuated2_goal_visible_offset_ground_smallcnn_relative_turncredit_stop_success.sh
```

The actuated recipe uses the same PPO/early-stop setup as the frozen recipe, but
keeps the eye pan/tilt ranges narrow:

- pan range `[-10, 10]`
- tilt range `[-5, 5]`

This matters. A wider actuated-eye range `[-45, 45]` pan and `[-30, 30]` tilt
failed seed 3 within the 40k budget:

- `logs/2026-06-13/exp_actuated2_goal_visible_offset_fast_stop_success_seed3_40k`
- eval success blocks:
  `[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0]`

The failure videos show visible-target overshoot/circling rather than a wall-contact
trap. Constraining gaze keeps the extra action dimensions from destabilizing the
body-steering solution.

## Actuated-eye seed evidence

All runs below used the narrow-eye actuated recipe with clean eval and stopped as
soon as an eval block reached 6/6 success.

| Seed | Run | Eval success blocks | Stop step | Final eval mean |
| --- | --- | --- | ---: | ---: |
| 0 | `logs/2026-06-13/exp_actuated2_goal_visible_offset_fast_stop_success_seed0_narroweyes_40k` | `[0, 6]` | 5000 | 19.62 |
| 1 | `logs/2026-06-13/exp_actuated2_goal_visible_offset_fast_stop_success_seed1_narroweyes_40k` | `[0, 6]` | 5000 | 19.60 |
| 2 | `logs/2026-06-13/exp_actuated2_goal_visible_offset_fast_stop_success_seed2_narroweyes_40k` | `[0, 0, 0, 0, 3, 0, 0, 6]` | 20000 | 19.62 |
| 3 | `logs/2026-06-13/exp_actuated2_goal_visible_offset_fast_stop_success_seed3_narroweyes_40k` | `[0, 0, 6]` | 7500 | 19.60 |
| 4 | `logs/2026-06-13/exp_actuated2_goal_visible_offset_fast_stop_success_seed4_narroweyes_40k` | `[0, 0, 6]` | 7500 | 19.58 |
| 5 | `logs/2026-06-13/exp_actuated2_goal_visible_offset_fast_stop_success_seed5_narroweyes_40k` | `[0, 0, 6]` | 7500 | 19.60 |

Representative videos:

- `logs/2026-06-13/exp_actuated2_goal_visible_offset_fast_stop_success_seed2_narroweyes_40k/evaluations/vis_7.mp4`
- `logs/2026-06-13/exp_actuated2_goal_visible_offset_fast_stop_success_seed3_narroweyes_40k/evaluations/vis_2.mp4`
