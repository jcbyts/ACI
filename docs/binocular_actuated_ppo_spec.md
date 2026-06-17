# Binocular and Actuated-Eye PPO Spec

## Goal

Establish a robust PPO baseline for target pursuit using a vertebrate-like
binocular visual system, then extend that baseline to actuated eyes without adding
global position, target position, or privileged navigation state to the policy.

The intended policy observation contract is:

- retinal images from two eyes
- proprioception and efference copy
- contact or touch information is allowed
- no global self position
- no global target position
- no privileged action

Contact is acceptable because it is local touch feedback, not environment location.
The environment may also use simulator state privately for reward, truncation, and
diagnostics.

## Clarification: Nonstationary Observations

Body movement already makes visual observations action-conditioned. Actuated eyes add
another action-conditioned transform: the same world state can produce different
retinal images depending on recent eye commands and actual eye pose. That is the
important extra ambiguity.

This should be learnable if the policy receives enough local state:

- previous body action
- previous eye command
- actual eye pan/tilt qpos, and preferably qvel
- spatially preserved retinal features
- temporal memory, such as a GRU

So the issue is not that actuated eyes are impossible for PPO. The issue is that a
policy without gaze state and without spatial visual features is forced to infer too
much from pixels alone.

## Phase 1: Correct Binocular Frozen-Eye Baseline

Use the intended two-eye geometry before changing architecture or actuation.

Target geometry:

- two eyes
- centers: `[-18.75, 18.75]` degrees
- horizontal FOV: `67.5` degrees
- vertical FOV: `45` degrees
- union coverage: `[-52.5, 52.5]` degrees
- binocular overlap: `[-15, 15]` degrees
- resolution: `[20, 30]`

This is the two-eye analogue of the original three-eye setup:

- original centers: `[-30, 0, 30]`
- original horizontal FOV: `45` degrees
- original union coverage: `[-52.5, 52.5]`
- original total overlap: `30` degrees

Tasks:

1. Add or preserve a single canonical script for the frozen binocular moving-target
   run.
2. Add a preflight check that reads the compiled XML and verifies camera yaw,
   sensorsize/FOV, resolution, and eye count.
3. Treat any run that fails this geometry check as invalid.
4. Run at least seeds `0,1,2` with identical geometry and evaluation.

Acceptance criteria:

- compiled XML contains exactly two agent cameras
- camera geometry matches the target binocular geometry
- policy observation contains retinal inputs plus allowed local state only
- all reported baseline runs pass the geometry preflight

## Phase 2: Better Visual Architecture

Replace global-pooled visual encoders for binocular pursuit.

Problem:

The current small CNN uses global average pooling, which destroys within-eye target
location. That can still work weakly when eye identity encodes coarse bearing, but it
is the wrong inductive bias for binocular, wide-FOV, or actuated-eye vision.

Proposed architecture:

- CoordCNN image stem
- no global average pooling
- per-eye spatial feature maps retained through flattening or tokenization
- eye identity and eye pose embeddings
- GRU after visual and proprioceptive fusion
- separate policy/value MLPs after recurrent fusion

Inputs:

- left retinal image
- right retinal image
- previous body action
- previous eye action when actuated
- contact/touch bit if enabled
- actual eye pan/tilt qpos and qvel when actuated

Initial implementation:

1. Add `MjCambrianCoordCNNExtractor`.
2. Add a recurrent policy path, preferably GRU-based, compatible with Dict
   observations.
3. Add config group `coord_cnn_gru_shared.yaml`.
4. Keep the first version small and stable:
   - 2 or 3 convolution layers
   - coordinate channels appended per eye
   - flatten spatial map
   - projection to compact visual embedding
   - GRU hidden size 128 or 256

Acceptance criteria:

- extractor preserves spatial layout before projection
- no adaptive global pooling in the visual path
- architecture works with two same-shaped eyes
- architecture works with frame stack disabled or reduced
- smoke training starts and completes at least one eval block

## Phase 3: Fix Moving-Target Evaluation

Current failure mode:

The moving-target respawn task can score well with a few incidental captures over a
long horizon, including policies that park, drift, or exploit borders. Mean reward
alone is not a valid success criterion.

Add a behavior-aware evaluation gate.

Per-episode diagnostics:

- captures
- episode length
- total path length
- net displacement
- stationary fraction
- median distance to target
- minimum distance to target
- near-target fraction
- contact fraction
- near-border fraction

Success criteria should require all of the following:

- mean captures above threshold
- every eval episode has at least one capture, or a stricter configured fraction
- median target distance below threshold
- stationary fraction below threshold
- contact fraction below threshold
- near-border fraction below threshold
- no reward-threshold-only early stopping

Tasks:

1. Move the useful logic from `scripts/audit_policy_rollouts.py` into a reusable
   evaluation utility.
2. Add an SB3 callback that runs this audit after each eval block.
3. Save the first checkpoint that satisfies the behavior gate.
4. Write a compact JSON summary next to each eval video.
5. Update docs so tables report behavior metrics, not just reward.

Acceptance criteria:

- an eval block with high reward but stationary parking fails the gate
- an eval block with low stationary fraction and repeated pursuit captures passes
- training stop condition uses behavior gate, not only reward threshold

## Phase 4: Reward and MDP Cleanup

Make the moving-target reward easier to interpret.

Tasks:

1. Add an agent-centered capture reward instead of relying on reward assigned to
   `goal0` and summed through the single-agent wrapper.
2. Keep dense approach reward train-only.
3. Keep clean eval sparse and diagnostic-rich.
4. Decide whether capture should terminate, respawn, or both:
   - fixed-goal baseline: terminate on capture
   - moving-target pursuit: respawn is acceptable, but success must require repeated
     pursuit behavior
5. Keep wall/contact truncation or strong contact penalty when walls exist.

Acceptance criteria:

- reward logs clearly attribute task reward to the learning agent
- evaluation reward is not the sole success signal
- reward cannot be passed by stationary or border behavior

## Phase 5: Actuated-Eye Curriculum

Actuated eyes are required for the research program, but should be introduced after
the corrected frozen binocular baseline is measurable.

Curriculum:

1. Frozen binocular, corrected geometry.
2. Actuated-eye body with gaze clamped at zero.
3. Actuated eyes with tiny range.
4. Narrow range:
   - pan `[-10, 10]`
   - tilt `[-5, 5]`
5. Medium range.
6. Full intended range.

Required observation additions:

- actual pan qpos per eye
- actual tilt qpos per eye
- preferably pan/tilt qvel
- previous eye command

Regularization:

- eye action magnitude penalty
- eye velocity penalty
- vergence extreme penalty
- optional gaze-center prior early in curriculum

Control structure:

- keep body and eye action heads separate if practical
- allow different entropy or standard deviation schedules for body and gaze
- consider freezing or heavily regularizing gaze early in training

Acceptance criteria:

- clamped-gaze actuated model matches frozen-eye behavior
- narrow actuated eyes pass the same behavior gate as frozen eyes
- widening gaze does not pass unless behavior metrics remain healthy
- reported actuated results include gaze statistics and pursuit metrics

## Phase 6: Experiment Tracking Rules

Every claimed baseline run should record:

- git commit or diff status
- full Hydra config
- compiled XML
- geometry preflight result
- observation-space summary
- action-space summary
- eval reward table
- behavior audit JSON
- representative video
- whether it passed the behavior gate

Do not treat a run as successful if:

- geometry preflight fails
- observation contract is unclear
- only reward threshold passed
- median distance and stationary fraction indicate non-pursuit
- the checkpoint is final-policy-only after PPO has moved away from a good eval block

## Immediate Next Tasks

1. Implement the geometry preflight for the intended binocular setup.
2. Implement behavior-gated evaluation and checkpoint selection.
3. Add the CoordCNN visual extractor.
4. Add GRU-based temporal fusion.
5. Run corrected frozen binocular seeds `0,1,2`.
6. Add gaze qpos/qvel observations for actuated eyes.
7. Run clamped-gaze actuated equivalence test.
8. Run narrow-range actuated curriculum seeds `0,1,2`.

