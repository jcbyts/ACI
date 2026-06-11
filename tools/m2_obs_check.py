"""STEP 1 diagnostic: is the actuated eye BLIND or is this a cosmetic overlay bug?

Instantiates the eval env, resets, steps a handful of times (the seeker targets
move on their own), and prints each trainable agent eye's observation stats
(shape, min, max, mean) plus how much it changes frame-to-frame.

Run via:
    bash scripts/run.sh tools/m2_obs_check.py example=tracking_eye \
        expname=exp_m2_obs_check trainer.n_envs=1
    bash scripts/run.sh tools/m2_obs_check.py example=tracking \
        expname=exp_m2_obs_check_base trainer.n_envs=1
"""

import numpy as np
import torch

from hydra_config import run_hydra

from cambrian import MjCambrianConfig
from cambrian.utils import get_logger


def _stats(t: torch.Tensor):
    a = t.detach().cpu().numpy().astype(np.float32)
    return a.shape, float(a.min()), float(a.max()), float(a.mean()), float(a.std())


def main(config: MjCambrianConfig, *, n_steps: int = 8, **__):
    env = config.env.instance(config.eval_env)
    env.reset(seed=config.seed)

    trainable = {n: a for n, a in env.agents.items() if a.trainable}
    log = get_logger()

    # Track previous obs per (agent, eye) to measure frame-to-frame change.
    prev = {}
    for t in range(n_steps):
        # Zero action for the trainable agent(s); the seeker targets move on their own.
        action = {n: np.zeros(a.action_space.shape[0], dtype=np.float32).tolist()
                  for n, a in trainable.items()}
        env.step(action)

        for an, agent in trainable.items():
            for en, eye in agent.eyes.items():
                obs = eye.prev_obs
                if obs is None:
                    log.info(f"[OBS] step={t} {an}/{en}: prev_obs is None")
                    continue
                shape, mn, mx, mean, std = _stats(obs)
                key = (an, en)
                delta = "n/a"
                if key in prev:
                    delta = f"{float(np.abs(obs.detach().cpu().numpy() - prev[key]).mean()):.5f}"
                prev[key] = obs.detach().cpu().numpy().copy()
                log.info(
                    f"[OBS] step={t} {an}/{en}: shape={shape} "
                    f"min={mn:.4f} max={mx:.4f} mean={mean:.4f} std={std:.4f} "
                    f"frame_delta={delta}"
                )


if __name__ == "__main__":
    run_hydra(main, config_path="pkg://cambrian/configs")
