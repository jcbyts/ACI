# R2-Dreamer Frozen Binocular Handoff

## Branch

Use:

```bash
git fetch origin
git switch codex/r2dreamer-frozen2
```

The branch is based on the binocular PPO diagnostics branch and adds a thin
ACI-side runner for the public PyTorch r2dreamer implementation.

## What Was Added

- `scripts/train_r2dreamer_frozen2_binocular.py`
- `scripts/train_r2dreamer_frozen2_binocular.sh`

The runner imports a local checkout of `NM512/r2dreamer`, builds the frozen
binocular ACI task, and presents it to r2dreamer as:

- `image`: side-by-side left/right retinal image, shape `(20, 60, 3)`, uint8
- `proprio`: previous body action / efference copy, shape `(2,)`

It does not expose global agent position, target position, or privileged target
state to the learner.

Each run writes:

- composed ACI config
- composed r2dreamer config
- compiled MuJoCo XML
- binocular geometry preflight JSON
- ACI env summary
- r2dreamer checkpoint
- TensorBoard metrics
- final behavior audit JSON
- final retinal rollout MP4

## Workstation Setup

Clone r2dreamer somewhere outside this repo:

```bash
git clone https://github.com/NM512/r2dreamer.git /path/to/r2dreamer
```

In the ACI Python environment on the workstation, install the r2dreamer-side
dependencies. Prefer adding these to the existing CUDA-enabled ACI env instead of
letting r2dreamer reinstall Torch:

```bash
python -m pip install torchrl tensordict ruamel.yaml "moviepy==1.0.3" einops tensorboard
```

Then run a smoke test:

```bash
R2DREAMER_ROOT=/path/to/r2dreamer \
EXPNAME=smoke_r2dreamer_frozen2 \
DEVICE=cuda:0 \
STEPS=1000 \
ENV_NUM=2 \
EVAL_EVERY=1000 \
EVAL_EPISODE_NUM=1 \
FINAL_EVAL_EPISODES=1 \
BATCH_SIZE=4 \
BATCH_LENGTH=16 \
TRAIN_RATIO=8 \
bash scripts/train_r2dreamer_frozen2_binocular.sh
```

If that works, start a real frozen-binocular run:

```bash
R2DREAMER_ROOT=/path/to/r2dreamer \
EXPNAME=exp_r2dreamer_frozen2_binocular_seed0_500k \
DEVICE=cuda:0 \
SEED=0 \
STEPS=500000 \
ENV_NUM=16 \
EVAL_EVERY=10000 \
EVAL_EPISODE_NUM=6 \
FINAL_EVAL_EPISODES=6 \
BATCH_SIZE=16 \
BATCH_LENGTH=64 \
TRAIN_RATIO=512 \
bash scripts/train_r2dreamer_frozen2_binocular.sh
```

## MacBook Smoke/Pilot Results

The MacBook is effectively CPU-only for this path (`cuda=false`, MPS unavailable),
so the local runs were plumbing checks, not serious evidence about Dreamer.

Completed checks:

- 64-step smoke run completed end-to-end.
- 10k CPU pilot completed after fixing reset seeding.
- Geometry preflight passed in both runs.
- Final behavior audit and retinal MP4 were written.

Corrected 10k CPU pilot:

- run: `logs/2026-06-17/codex_r2dreamer_frozen2_pilot10k_seedfix_seed0`
- `mean_captures`: `0.0`
- `mean_stationary_fraction`: `0.701`
- `mean_median_distance`: `10.638`
- `max_contact_fraction`: `0.998`
- action speed did not collapse to zero; mean physical speed command was about
  `0.475`

Interpretation: r2dreamer is wired up and training, but the short CPU run did not
solve the frozen binocular task. The failure looked like unsafe wall-contact
control rather than PPO-style parking.

## Things To Watch

- The initial policy still has unsafe raw point-agent action semantics: action
  speed near `0` maps to physical speed near `0.5`.
- If the 500k GPU run also fails, the next controlled comparison should use the
  existing action-speed-band wrapper or a fixed-speed steering-only variant to
  separate visual pursuit learning from bad speed exploration.
- r2dreamer logs internal videos through TensorBoard. Pinning `moviepy==1.0.3`
  should avoid the `moviepy.editor` warning seen on the MacBook with MoviePy 2.x.
- The script clones r2dreamer to `/private/tmp/r2dreamer` by default if
  `R2DREAMER_ROOT` is missing, but on the workstation it is cleaner to set
  `R2DREAMER_ROOT` explicitly.

