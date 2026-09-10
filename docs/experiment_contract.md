# Binocular tracking experiment contract

> This document records the June 22 no-shaping baseline contract. Current recurrent configs use a ten-frame stack, and the recommended practical baseline adds training-only progress shaping. Follow the [student quickstart](student_quickstart.md) for the maintained workflow; the strict audit still flags that shaping.

This document defines the experiment that counts as the baseline for studying the
emergence of fixation and saccadic eye movements. A run name, launcher comment, or
video is not evidence that this contract was used. The resolved configuration,
compiled MuJoCo model, saved policy, and evaluation artifacts must agree with it.

## 1. Scientific scope

The baseline establishes that a lightweight on-policy agent can learn the original ACI
tracking behavior with two independently actuated eyes. It does **not** by itself test
sensor slowness/noise or efficient coding. Those hypotheses are introduced only after
the baseline ladder below passes.

The two eventual hypotheses are kept distinct:

1. **Noisy/slow sensing:** stable gaze improves the signal available from a sensor with
   temporal integration and/or observation noise.
2. **Efficient change coding:** an event-like or temporal-difference retina plus an
   explicit activity cost makes periods of low retinal change valuable.

A temporal-difference observation without an activity term is not an energetic coding
experiment. Conversely, a movement-dependent blur or integration rule is not a clean
sensor-slowness manipulation because it directly builds a gaze-stability preference
into the sensor.

## 2. Canonical task

The task is the repository's original `tracking` task:

- one trainable point agent named `agent`;
- one moving target named `goal0`;
- one moving hazard named `adversary0`;
- the target and hazard are both non-trainable random point seekers;
- during training, proximity to `goal0` terminates successfully and proximity to
  `adversary0` truncates unsuccessfully;
- the original contact penalty remains, but no contact termination is added;
- no approach, facing, turning, time, or goal-respawn shaping is added;
- during evaluation, both objects respawn after contact, target respawn gives the
  original positive reward, and adversary respawn gives the original negative reward.

Deleting `adversary0`, starting from `example=detection`, substituting a goal-only maze,
or disabling the original target/adversary terms makes a run a different task.

## 3. Canonical two-eye geometry

The original three-eye retina has three `20 x 20` images, horizontal eye centers
`[-30, 0, 30]` degrees, and `45` degrees horizontal FOV per eye. Therefore:

- total spatial samples: `3 * 20 * 20 = 1200` pixels;
- angular sampling: `45 / 20 = 2.25` degrees per horizontal pixel;
- union coverage: `[-30 - 22.5, 30 + 22.5] = [-52.5, 52.5]` degrees;
- adjacent overlaps: `15 + 15 = 30` degrees total.

The matched binocular control has two `20 x 30` images, centers
`[-18.75, 18.75]` degrees, and `[45, 67.5]` degrees vertical/horizontal FOV:

- total spatial samples: `2 * 20 * 30 = 1200` pixels;
- angular sampling: `67.5 / 30 = 2.25` degrees per horizontal pixel;
- union coverage: `[-18.75 - 33.75, 18.75 + 33.75] = [-52.5, 52.5]` degrees;
- binocular overlap: `[-15, 15]`, or `30` degrees.

This is a matched compute/coverage control. It is not asserted to be a uniquely
biological eye arrangement. A two-eye configuration at centers `[-30, 30]` with
`45`-degree FOV leaves a forward blind gap and is not a matched replacement.

The physical sensor size is derived from focal length and FOV rather than copied from
an intended value:

\[
  s = 2 f \tan(\mathrm{FOV}/2).
\]

Every run must pass `scripts/preflight_binocular_geometry.py` on its compiled XML.

## 4. Policy contract

### Fixed-eye controls

The policy may observe:

- two RGB retinal images;
- exact previous policy action/efference copy;
- the existing local contact bit.

It may not observe world position, target position, adversary position, privileged
navigation actions, or object bearings computed from simulator state.

### Independently actuated binocular baseline

The policy action is six-dimensional:

```text
[body_forward, body_heading,
 left_pan, left_tilt, right_pan, right_tilt]
```

The policy may additionally observe physical eye joint position and velocity. With
four eye actuators this is eight values: four normalized joint positions followed by
four normalized joint velocities. The previous-action observation must be the exact
six-dimensional policy command, not the translated MuJoCo body controls.

The baseline retina has `noise_std: 0.0` and `integration_factor: 0.0`. Frame stacking
and constant-action wrappers are disabled for recurrent PPO.

## 5. Baseline ladder

Do not jump directly from the upstream result to a new task, new sensor, new encoder,
and eye actuation simultaneously. Advance only after the preceding stage is
reproducible.

| Stage | Command mode | Purpose |
|---|---|---|
| A | `original-3eye-mlp` | Reproduce the known upstream tracking result and establish the machine/software baseline. |
| B | `fixed-2eye-mlp` | Isolate the effect of replacing three eyes with matched binocular geometry. |
| C | `fixed-2eye-rppo` | Isolate the spatial CNN and recurrent policy while keeping the eyes fixed. |
| D | `actuated-2eye-rppo` | Add independent pan/tilt actuation and eye-state proprioception. |

For each stage, use the same task, episode horizon, evaluation protocol, training
budget, and seed set. Start with seeds `0, 1, 2`; use at least five seeds for a result
that will support a scientific claim. Do not select a single visually pleasing run as
the result.

A stage passes only when:

1. the run-contract audit passes;
2. the compiled geometry preflight passes where applicable;
3. learning is reproducible across the declared seed set;
4. target captures and adversary contacts are reported separately;
5. deterministic evaluation videos agree with the numerical event log.

Mean return alone is insufficient because reward shaping or task deletion can produce
high returns without the requested behavior.

## 6. Commands

Install the package in a clean Python environment, then run one of the four explicit
modes:

```bash
bash scripts/run_tracking_baseline.sh original-3eye-mlp seed=0
bash scripts/run_tracking_baseline.sh fixed-2eye-mlp seed=0
bash scripts/run_tracking_baseline.sh fixed-2eye-rppo seed=0
bash scripts/run_tracking_baseline.sh actuated-2eye-rppo seed=0
```

Evaluate a run under the matching contract:

```bash
MODE=actuated-2eye-rppo bash scripts/eval_tracking_baseline.sh \
  logs/YYYY-MM-DD/tracking_2eye_recurrent_actuated_seed0
```

For another stage, change `MODE` to the mode used for training. Audit the resolved run
rather than the launcher:

```bash
python scripts/audit_run_contract.py RUN_DIR
python scripts/preflight_binocular_geometry.py \
  --compiled-xml RUN_DIR/compiled_env.xml
```

The contract audit is intentionally strict for Stage D. The fixed-eye and original
controls are expected to fail Stage-D-specific actuation checks; compare their
resolved configs against the stage definition above rather than relabeling them as the
actuated baseline.

## 7. Required run artifacts

A result is incomplete unless its run directory contains:

- resolved `config.yaml` and evaluation `eval_config.yaml`;
- `compiled_env.xml` and `compiled_eval_env.xml`;
- saved checkpoint and exported policy;
- training and evaluation monitor data;
- deterministic evaluation video;
- seed, package versions, Python version, MuJoCo version, and source commit;
- run-contract and geometry-preflight reports;
- event-level evaluation summary containing target captures, adversary contacts,
  wall contacts, episode length, and return.

The source tree used for a run must be clean or archived as a patch. Never infer the
experiment from shell-script names after the fact.

## 8. Hypothesis experiments after Stage D

### Sensor slowness/noise

Use a factorial design that changes only the retinal observation process:

- independent observation noise levels;
- independent first-order temporal integration time constants;
- raw instantaneous sensor control at every noise level;
- identical policy, task, geometry, reward, action cost, and compute budget.

Prefer a physical time constant `tau` and derive the per-step retention from simulation
step size, rather than treating a dimensionless integration factor as comparable
across frame rates. Log both commanded and realized gaze velocity so fixation can be
measured without relying on a rendered video.

### Efficient coding

Use a fixed retinal transform first. A minimal controlled comparison is:

1. raw RGB, no activity cost;
2. raw RGB with a matched activity cost;
3. temporal-difference or ON/OFF event channels with the same activity cost.

Define the cost before running the experiment, for example the per-step mean absolute
retinal activity multiplied by a declared coefficient. Match channel count, dynamic
range, normalization, encoder capacity, and optimizer budget. Only after the fixed
transform is understood should a learned retinal bottleneck be introduced.

Primary scientific outcomes should include task success, adversary avoidance, retinal
activity, eye/head angular velocity, fixation duration distribution, saccade amplitude
and peak velocity, and total control effort. Pre-register the fixation/saccade
segmentation thresholds or report sensitivity to those thresholds.
