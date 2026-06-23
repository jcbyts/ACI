# Repository audit — 2026-06-22

## Verdict

Neither showcased agent run implements the requested moving-target plus
moving-adversary binocular tracking experiment.

- The latest recurrent-PPO run starts from `example=detection`, selects a goal-only
  maze, removes `adversary0`, and adds several reward/done/action changes. It is a
  goal-pursuit experiment, not the original tracking task.
- The R2-Dreamer run also removes the adversary and trains moving-goal pursuit with
  shaped rewards and repeated target respawns. Its video is evidence that the
  visuomotor stack can produce pursuit behavior, not evidence for adversary avoidance.

The recurrent-PPO checkpoint has an additional implementation defect: its saved
retinal observation shape is `(20, 20, 3)`, but the first convolution expects 20 input
channels. Height was interpreted as channels. Thus the run did not use the intended
RGB spatial geometry even though its script selected a “spatial CNN.”

## Evidence from resolved artifacts

| Item | Latest recurrent PPO | R2-Dreamer |
|---|---|---|
| Requested moving `goal0` | Present, but under a modified goal-only task | Present |
| Requested moving `adversary0` | Deleted | Deleted |
| Original target/adversary evaluation | Replaced/disabled | Replaced/disabled |
| Extra shaping | Approach/facing/turn/time and contact-failure changes | Approach shaping and repeated target respawn |
| Retinal geometry | Saved as two `(20,20,3)` images; CNN consumes 20 channels | Intended wide FOV differs from compiled physical FOV |
| Scientific interpretation | Not comparable to upstream tracking | Pursuit proof of concept only |

The included `scripts/audit_run_contract.py` reproduces these conclusions from saved
configs, compiled XML, and SB3 checkpoints rather than trusting file names.

## Repository-level failures

1. **Broken provenance.** The Git object pack does not match its index and `git fsck`
   reports invalid references/reflogs. The surviving repository metadata points to
   the fork `https://github.com/jcbyts/ACI.git`; the active branch and
   `codex/r2dreamer-frozen2` ref both point to
   `e62a749c1f4f623a4b67600d7ea6efe9dd875663`, while `main` points to
   `69525c799297d836fe056031a4d5b575a7580e70`. The current checkout should not be used
   as the authoritative history.
2. **Task identity encoded in shell inheritance.** Dozens of one-off launchers layer
   overrides through nested shell calls. A descriptive final filename does not expose
   inherited task deletion or reward changes.
3. **Two incompatible PPO vision paths.** The old small CNN globally pools every
   feature map to `1 x 1`; that is a poor localization inductive bias. The later
   spatial CNN avoids pooling, but the Dict extractor supplied HWC tensors to a CHW
   convolution.
4. **Sensor hypothesis confound.** The previous temporal integration implementation
   changed its update rate as a function of frame difference, directly introducing a
   motion-dependent effect rather than a clean slow-sensor manipulation.
5. **Wrong efference copy.** Point-agent action observations exposed translated
   MuJoCo controls rather than the exact policy command. This is especially damaging
   when body heading is transformed or eye actions are appended.
6. **Observation-space type violation.** Contact observations were Python lists while
   their Gymnasium space declared a NumPy `int32` array. SB3 rejected a real default
   environment during the audit.
7. **Unsafe checkpoint loading.** Mismatched policy layers could be silently skipped,
   allowing a nominal “resume” to initialize part of a network randomly.
8. **FOV validation checked intent, not physics.** Existing preflights reported YAML
   values without deriving FOV from the compiled camera sensor size and focal length.

## Repairs in the cleanup tree

- Added matched fixed and actuated binocular eye configurations.
- Added four explicit baseline modes and one canonical training launcher.
- Added a matching evaluation/video launcher.
- Restored `tracking.yaml` to an unmodified task definition; sensor manipulations are
  no longer injected into the task.
- Corrected HWC/NHWC to CHW/NCHW conversion and added regression tests.
- Kept spatial feature maps through flattening for recurrent PPO.
- Exposed the exact policy action and physical eye joint state.
- Corrected contact observations to the declared NumPy type.
- Replaced motion-adaptive temporal integration with a validated, motion-independent
  first-order update.
- Made checkpoint loading fail closed unless partial loading is explicitly requested.
- Added run-contract and compiled-geometry audits.
- Removed 101 superseded files from the active experiment surface and preserved them
  in a checksummed legacy archive.

## Executed validation

The audit did more than parse source files:

1. Python source compilation, shell syntax, and YAML syntax checks passed.
2. Unit tests for image layout and run-contract failures passed.
3. A real MuJoCo environment instantiated with `agent`, `goal0`, and `adversary0`.
4. Its policy observation contained two `(20,30,3)` RGB retinas, a six-dimensional
   action/efference copy, an eight-dimensional eye-state vector, and the local contact
   bit.
5. The first spatial convolution had three input channels.
6. A four-step recurrent-PPO rollout and optimizer update completed; 111 policy
   tensors changed, including the recurrent and visual paths.
7. The evaluation renderer produced an H.264 MP4 through the canonical video path.

These checks prove that configuration composition, MuJoCo construction, observation
layout, recurrent forward pass, rollout buffer, backward pass, checkpoint path, and
video path execute together. Four optimizer steps do **not** demonstrate learned
tracking behavior or convergence.

## Remaining scientific and engineering risks

- The actuated-eye gimbal sign conventions and limits need deterministic geometry
  tests over the full command range, even though the one-step and optimizer smokes
  pass.
- A meaningful Stage-D training run has not been completed in this audit.
- Event-level evaluation metrics should be implemented before comparing seeds; return
  alone is not enough.
- The sensor-slowness interface should ultimately use a physical time constant tied to
  simulation `dt`.
- The efficient-coding experiment still needs a fixed event/derivative retina and an
  explicit, pre-declared activity cost.
- Dependency versions should be locked in a reproducible environment before long
  sweeps.

## Recovery procedure

Do not continue development in the corrupt Git directory. Clone the surviving fork
again, check out the intended base commit if it is still available, and then either
copy the audited cleanup tree or apply the supplied repair patch. Keep the legacy
archive outside Hydra's active search path. Commit the clean baseline before launching
new experiments, and require the run-contract/geometry reports for every result.
