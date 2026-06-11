"""M2 verification tool: checks the actuated-eye agent's action space and records a
scripted rollout demonstrating that the eye orientation changes independently of the
body.

Run via:
    bash scripts/run.sh tools/m2_eye_check.py example=tracking_eye \
        expname=exp_m2_eye_check trainer.n_envs=1

The rollout has two phases:
  1. Body STATIONARY (forward velocity 0) while the eye pan/tilt sweep -> the eye
     visibly rotates with the body still.
  2. Body MOVING (forward velocity > 0) while the eye command is HELD -> independent
     gaze while locomoting.

Eye joint qpos is logged each step as numeric evidence of the gimbal moving.
"""

import math

import numpy as np

from hydra_config import run_hydra

from cambrian import MjCambrianConfig
from cambrian.renderer import MjCambrianRendererSaveMode
from cambrian.utils import get_logger


def main(config: MjCambrianConfig, *, record: bool = True, **__):
    env = config.env.instance(config.eval_env)

    # --- Action-space check -------------------------------------------------
    for name, agent in env.agents.items():
        if agent.trainable:
            get_logger().info(
                f"[M2] agent '{name}' action_space={agent.action_space} "
                f"shape={agent.action_space.shape}"
            )

    if record:
        env.record(True, path=config.expdir)

    env.reset(seed=config.seed)
    env.spec.save(config.expdir / "compiled_env.xml")

    trainable = {n: a for n, a in env.agents.items() if a.trainable}
    name = next(iter(trainable))
    ndim = trainable[name].action_space.shape[0]
    get_logger().info(f"[M2] driving agent '{name}' with {ndim}-dim actions")

    n_steps = min(env.max_episode_steps, 120)
    half = n_steps // 2
    eye_qpos_log = []

    for t in range(n_steps):
        action = {}
        for n, a in trainable.items():
            act = np.zeros(a.action_space.shape[0], dtype=np.float32)
            if t < half:
                # Phase 1: body stationary (v-action = -1 -> v=0), sweep eye.
                act[0] = -1.0
                act[1] = 0.0
                if act.shape[0] >= 3:
                    act[2] = math.sin(2 * math.pi * t / half)  # pan sweep
                if act.shape[0] >= 4:
                    act[3] = 0.5 * math.sin(4 * math.pi * t / half)  # tilt sweep
            else:
                # Phase 2: body moves forward, eye command held fixed.
                act[0] = 0.5
                act[1] = 0.0
                if act.shape[0] >= 3:
                    act[2] = 0.8  # hold pan at +0.8
                if act.shape[0] >= 4:
                    act[3] = -0.3  # hold tilt
            action[n] = act.tolist()

        env.step(action)
        if record:
            env.render()

        # Log eye joint qpos (proprioceptive state of the gimbal).
        qpos = trainable[name].qpos
        eye_qpos_log.append(np.asarray(qpos[3:]).copy())

    eye_qpos_log = np.array(eye_qpos_log)
    if eye_qpos_log.size and eye_qpos_log.shape[1] >= 2:
        get_logger().info(
            f"[M2] eye joint qpos range over rollout: "
            f"pan[min={eye_qpos_log[:, 0].min():.3f}, max={eye_qpos_log[:, 0].max():.3f}] "
            f"tilt[min={eye_qpos_log[:, 1].min():.3f}, max={eye_qpos_log[:, 1].max():.3f}] "
            f"(radians)"
        )

    if record:
        env.save(
            config.expdir / "m2_eye_check",
            save_pkl=False,
            save_mode=MjCambrianRendererSaveMode.MP4 | MjCambrianRendererSaveMode.GIF,
        )
        get_logger().info(f"[M2] saved rollout to {config.expdir / 'm2_eye_check'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-record",
        action="store_false",
        dest="record",
        help="Skip recording the rollout video (fast structural check).",
    )
    run_hydra(main, config_path="pkg://cambrian/configs", parser=parser)
