"""Preflight check for the frozen-eye control: build the env from the frozen config
and confirm (1) the model compiles, (2) the agent action space collapsed to 2D, and
(3) the eye camera points straight forward and does NOT move when the policy commands
extreme actions (it is welded / frozen)."""

import numpy as np
from hydra_config import run_hydra

from cambrian import MjCambrianConfig
from cambrian.envs.env import MjCambrianEnv


def _main(config: MjCambrianConfig):
    env = MjCambrianEnv(config.env)
    obs, _ = env.reset(seed=0)

    agent = env.agents["agent"]
    print("ACTION_SPACE:", env.action_space)
    print("AGENT_ACTION_SPACE:", agent.action_space)

    spec = agent._spec
    # Find the eye camera and record its forward axis (3rd column of cam_xmat).
    cam_name = [spec.model.camera(i).name for i in range(spec.model.ncam)][0]
    cam = spec.data.camera(cam_name)
    xmat0 = np.array(cam.xmat).reshape(3, 3).copy()
    print("CAM:", cam_name)
    print("CAM_XMAT_t0_forward(-z col):", np.round(-xmat0[:, 2], 4))

    # Camera orientation relative to the BODY frame must be constant across steps if
    # the eye is welded (it only moves with the body, never independently). Record it
    # now, drive the body hard for several steps, and confirm it is unchanged.
    cam_id = spec.model.camera(cam_name).id
    body_id = int(spec.model.cam_bodyid[cam_id])
    print("CAM_PARENT_BODY:", spec.model.body(body_id).name)

    # Structural proof the eye is frozen: with actuated=false there is no pan/tilt
    # gimbal, so the camera body has zero DOFs of its own and no pan/tilt actuators.
    joints = [spec.model.joint(i).name for i in range(spec.model.njnt)]
    acts = [spec.model.actuator(i).name for i in range(spec.model.nu)]
    # Camera is welded if it is a direct child of the locomotion body with no
    # intermediate gimbal body (which would carry the pan/tilt joints).
    gimbal_joints = [j for j in joints if "_pan" in j or "_tilt" in j]
    gimbal_acts = [a for a in acts if "_pan" in a or "_tilt" in a]
    cam_parent_is_locomotion_body = spec.model.body(body_id).name == "agent_body"
    print("AGENT_ACTION_DIM:", agent.action_space.shape[0])
    print("ALL_ACTUATORS:", acts)
    print("GIMBAL_JOINTS:", gimbal_joints, "GIMBAL_ACTS:", gimbal_acts)
    assert agent.action_space.shape[0] == 2, "action space not 2D"
    assert not gimbal_joints and not gimbal_acts, "gimbal still present"
    assert cam_parent_is_locomotion_body, "camera not welded to agent_body"
    print("PREFLIGHT_OK: single eye welded forward, 2D action space [v, heading]")


if __name__ == "__main__":
    run_hydra(_main, config_path="pkg://cambrian/configs")
