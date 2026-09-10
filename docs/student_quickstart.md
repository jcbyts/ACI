# Student quickstart

**Start from `main` in `jcbyts/ACI`.** The first goal is to reproduce the current
binocular recurrent-PPO workflow and inspect its behavior. The research goal is
to determine which ingredients are necessary for fixation and saccadic sampling.

## Environment

```bash
git clone https://github.com/jcbyts/ACI.git
cd ACI
git switch main

# New environment; use Python 3.12.
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

On the existing project workstation, use `conda activate aci312` instead of
creating a new environment. MuJoCo needs a working rendering backend; the launchers
use EGL on Linux and CGL on macOS. A CUDA-capable GPU is recommended for full runs.
The setup still includes Hydra's Java prerequisite described in the upstream
installation notes. Installation also fetches a custom Hydra dependency from GitHub.

The reviewed environment used Python 3.12.13, Torch 2.6.0+cu124, MuJoCo 3.2.6,
SB3/sb3-contrib 2.4.0, and NumPy 1.26.4. The package pins many dependencies but does
not provide a complete environment lock. The install command above is a setup
recipe, not a claim that every fresh platform has been validated.

## Check the installation

From the repository root, with the environment active:

```bash
MUJOCO_GL=egl python -m pytest -q
```

Use `MUJOCO_GL=cgl` on macOS. The current suite has 20 tests. Passing it verifies
selected implementation invariants, not learned behavior.

## Run a small training smoke test

This uses the same task, eyes, architecture, and overlays as the practical baseline,
with a small rollout/update budget, two environments, and CPU inference/training.
Give every run a new directory; do not reuse an old experiment directory.

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
SMOKE_DIR="$PWD/logs/student_smoke_$(date +%Y%m%d_%H%M%S)"

bash scripts/run_tracking_baseline.sh actuated-2eye-rppo \
  'overlay=[tracking_documented_reward,textured_ground]' \
  seed=0 expname=student_smoke expdir="$SMOKE_DIR" \
  trainer.n_envs=2 trainer.total_timesteps=256 \
  trainer.model.n_steps=32 trainer.model.batch_size=64 trainer.model.n_epochs=1 \
  trainer.callbacks.eval_callback.eval_freq=128 \
  trainer.callbacks.eval_callback.n_eval_episodes=2 eval_env.n_eval_episodes=2 \
  +trainer.model.device=cpu +eval_env.renderer.save_mode=MP4
```

Expect `finished`, `config.yaml`, `compiled_env.xml`, `best_model.zip`, `policy.pt`,
`evaluations.npz`, and evaluation videos. The small budget verifies plumbing and
should not be expected to learn useful behavior. `best_model.zip` is the best SB3
checkpoint; `policy.pt` contains final policy weights.

Check physical camera geometry:

```bash
python scripts/preflight_binocular_geometry.py \
  --compiled-xml "$SMOKE_DIR/compiled_env.xml"
```

The broader `scripts/audit_run_contract.py "$SMOKE_DIR"` currently flags the
training progress reward because it enforces the historical no-shaping contract.
For this recipe, that particular failure is expected. Other failures require
investigation. The current trainer writes the evaluation environment's XML under
`compiled_env.xml`; independently saving both compiled environments remains a
handoff task.

The 256-step smoke recipe above completed on the project Linux workstation on
2026-09-10, including two evaluation episodes, checkpoint/video output, and a
passing compiled-camera geometry check. This used the existing `aci312` environment.

## Replay and analyze

Use fresh directories and the same overlays as training. These commands also work
with a completed full run: set `MODEL_PATH` to its absolute `best_model` path,
omitting the `.zip` suffix. A fresh clone contains no trained checkpoints.

```bash
MODEL_PATH="$SMOKE_DIR/best_model" MODE=actuated-2eye-rppo \
  bash scripts/eval_tracking_baseline.sh "${SMOKE_DIR}_replay" \
  'overlay=[tracking_documented_reward,textured_ground]' \
  seed=1000 eval_env.n_eval_episodes=2 +trainer.model.device=cpu

MODEL_PATH="$SMOKE_DIR/best_model" MODE=actuated-2eye-rppo EPISODES=2 \
  bash scripts/analyze_tracking_behavior.sh "${SMOKE_DIR}_analysis" \
  'overlay=[tracking_documented_reward,textured_ground]' \
  seed=1000 +trainer.model.device=cpu
```

The evaluator writes `eval_best.mp4`. The analyzer writes `behavior_steps.csv`,
`behavior_episodes.csv`, `behavior_summary.json`, and plots under
`${SMOKE_DIR}_analysis/behavior_analysis/`.

The analyzer measures task events and horizontal eye/body/gaze dynamics. Fixation
and saccade segmentation, vertical gaze, occlusion-aware visibility, and validated
energy measurements remain future work. Evaluation and analysis consume random
state differently, so the same numeric seed does not guarantee identical episode
trajectories across the two tools.

## Run the recommended baseline

After the smoke succeeds, train the current actuated-eye candidate:

```bash
RUN_DIR="$PWD/logs/student_actuated_seed0_$(date +%Y%m%d_%H%M%S)"

bash scripts/run_tracking_baseline.sh actuated-2eye-rppo \
  'overlay=[tracking_documented_reward,textured_ground]' \
  seed=0 expname=student_actuated_seed0 expdir="$RUN_DIR" \
  trainer.n_envs=8 trainer.total_timesteps=750000 \
  trainer.callbacks.eval_callback.eval_freq=3125 \
  trainer.callbacks.eval_callback.callback_after_eval.callbacks.stop_training_on_no_improvement_callback.max_no_improvement_evals=1000 \
  +trainer.model.device=cuda:0 +eval_env.renderer.save_mode=MP4
```

Use `fixed-2eye-rppo` with the same settings and a new directory for the fixed-eye
control. Evaluate it with `MODE=fixed-2eye-rppo`. Replace `cuda:0` with `cpu` only if
needed; runtime depends on the machine. The recorded 750k recurrent runs took about
1.5–1.6 hours on the project workstation. The multi-GPU sweep scripts assume
specific device assignments, so start with the single-job command above.

The recipe uses moving `goal0` and `adversary0`, matched 20×30 RGB binocular eyes,
a yaw-rate body, ten-frame history, a shared R(2+1)D visual encoder, motor-conditioned
binocular fusion, and one LSTM. Previous commands, eye state, and contact feedback
are available. Progress shaping has coefficient 0.25 during training and is disabled
during evaluation. Sensor noise, temporal integration, and metabolism costs are off.

## First research deliverables

1. Reproduce fixed and actuated controls with the same budget and seed set. Preserve
   best and late-training results and use held-out evaluation episodes.
2. Validate task-event and gaze measurements against controlled trajectories and
   videos. Define fixation/saccade criteria before interpreting plots.
3. Ablate memory, frame history, motor observations, and shaping separately while
   holding the other settings fixed. Introduce sensing/cost experiments afterward.

The [review and handoff](review_2026-09-10/README.md) explains the evidence and
remaining limitations. Motor costs and heavy-body configurations are experimental;
the eye cost currently penalizes position-command magnitude rather than measured
movement energy. No completed historical metabolism sweep is available.

For each result, record the source commit, any local source patch, seed, package
versions, train/eval configurations, checkpoint, evaluation arrays, and videos.
Do not use `train_fitness.txt` as a substitute for mean episode return.

## Where to make changes

```bash
git switch main
git pull --ff-only origin main
git switch -c codex/your-experiment
```

Open a pull request back to `main` with the change and validation. Keep each change
small enough to identify its effect on the experiment. Earlier feature branches are
historical references. The old main is preserved at `legacy-main-v0.0.0`.
See [Contributing](contributing.md) for the full workflow.
