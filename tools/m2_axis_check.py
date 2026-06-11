"""M2 axis-convention check: confirm that +pan rotates the eye camera HORIZONTALLY
(azimuth) and +tilt rotates it VERTICALLY (elevation), with the agent body held fixed.

This reads the eye camera's world-frame orientation directly from MuJoCo
(`data.cam_xmat`) rather than tracking scene content, so it is deterministic and
unaffected by the moving target/adversary objects.

Run:
    bash scripts/run.sh tools/m2_axis_check.py example=tracking_eye \
        expname=exp_m2_axis trainer.n_envs=1
"""

import mujoco as mj
import numpy as np

from hydra_config import run_hydra

from cambrian import MjCambrianConfig
from cambrian.utils import get_logger


def _camera_forward(env, camid):
    """World-frame forward (gaze) direction of the camera (looks down local -z)."""
    xmat = np.array(env.data.cam_xmat[camid]).reshape(3, 3)
    return -xmat[:, 2]


def _azimuth_elevation(fwd):
    az = np.degrees(np.arctan2(fwd[1], fwd[0]))  # horizontal heading
    el = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))  # vertical angle
    return az, el


def _sweep(env, agent, name, camid, axis_idx, label):
    cmds = np.linspace(-1.0, 1.0, 11)
    az_list, el_list = [], []
    for c in cmds:
        act = np.zeros(agent.action_space.shape[0], dtype=np.float32)
        act[0] = -1.0  # forward velocity -> 0 (body stationary)
        act[1] = 0.0  # fixed heading
        act[axis_idx] = float(c)
        # Step several times so the position servo settles at the commanded angle.
        for _ in range(5):
            env.step({name: act.tolist()})
        az, el = _azimuth_elevation(_camera_forward(env, camid))
        az_list.append(az)
        el_list.append(el)

    az = np.unwrap(np.radians(az_list))
    az = np.degrees(az)
    el = np.array(el_list)
    d_az = az.max() - az.min()
    d_el = el.max() - el.min()
    get_logger().info(
        f"[M2-axis] {label}: azimuth span={d_az:.1f} deg, elevation span={d_el:.1f} deg "
        f"-> dominant motion is {'HORIZONTAL (azimuth)' if d_az > d_el else 'VERTICAL (elevation)'}"
    )
    return d_az, d_el


def main(config: MjCambrianConfig, **__):
    env = config.env.instance(config.eval_env)
    env.reset(seed=config.seed)

    name, agent = next((n, a) for n, a in env.agents.items() if a.trainable)
    camid = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_CAMERA, f"{name}_eye")
    get_logger().info(
        f"[M2-axis] agent '{name}' action_space={agent.action_space} "
        f"eye camera id={camid}"
    )
    assert camid != -1, "Could not find eye camera 'agent_eye'."

    pan_az, pan_el = _sweep(
        env, agent, name, camid, axis_idx=2, label="PAN  (action[2], expect HORIZONTAL)"
    )
    env.reset(seed=config.seed)
    tilt_az, tilt_el = _sweep(
        env, agent, name, camid, axis_idx=3, label="TILT (action[3], expect VERTICAL)"
    )

    pan_ok = pan_az > pan_el
    tilt_ok = tilt_el > tilt_az
    get_logger().info(
        f"[M2-axis] VERDICT: pan->horizontal={pan_ok}, tilt->vertical={tilt_ok} "
        f"({'AXES OK' if pan_ok and tilt_ok else 'AXES NEED SWAP/SIGN FIX'})"
    )


if __name__ == "__main__":
    run_hydra(main, config_path="pkg://cambrian/configs")
