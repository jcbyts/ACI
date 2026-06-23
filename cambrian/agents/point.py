"""Point agents."""

from functools import cached_property
from typing import Optional, Tuple

import numpy as np
from gymnasium import spaces

from cambrian.agents.agent import (
    MjCambrianAgent,
    MjCambrianAgent2D,
    MjCambrianAgentConfig,
)
from cambrian.envs.maze_env import MjCambrianMapEntity, MjCambrianMazeEnv
from cambrian.utils import get_logger
from cambrian.utils.types import ActionType, ObsType


class MjCambrianAgentPoint(MjCambrianAgent2D):
    """
    This is a hardcoded class which implements the agent as actuated by a forward
    velocity and a rotational position. In mujoco, to the best of my knowledge, all
    translational joints are actuated in reference to the _global_ frame rather than
    the local frame. This means a velocity actuator applied along the x-axis will move
    the agent along the global x-axis rather than the local x-axis. Therefore, the
    agent will have 3 actuators: two for x and y global velocities and one for
    rotational position. From the perspective the calling class (i.e. MjCambrianEnv),
    this agent has two actuators: a forward velocity and a rotational position. We will
    calculate the global velocities and rotational position from these two "actuators".

    Todo:
        Will create an issue on mujoco and see if it's possible to implement this in
        xml. The issue right now is that mujoco doesn't support relative positions for
        hinge joints, so we have to implement the heading joint as a velocity actuator
        which is not ideal.
    """

    def __init__(
        self,
        config: MjCambrianAgentConfig,
        name: str,
        *,
        kp: float = 0.75,
    ):
        super().__init__(config, name)

        self._kp = kp
        # Keep the exact action supplied by the policy.  ``MjCambrianAgent`` stores the
        # translated MuJoCo actuator command in ``_last_action``; that is not an
        # efference copy of the policy action for point agents.
        self._last_policy_action = np.zeros(2, dtype=np.float32)

        assert np.all(self._actuators[0].ctrlrange == self._actuators[1].ctrlrange), (
            f"Forward velocity and lateral velocity must have the same control range, "
            f"got {self._actuators[0].ctrlrange} and {self._actuators[1].ctrlrange}"
        )
        self._v_ctrlrange = np.array([0, 1])
        self._theta_ctrlrange = self._actuators[2].ctrlrange

    def _update_obs(self, obs: ObsType) -> ObsType:
        """Creates the entire obs dict."""
        obs = super()._update_obs(obs)

        if self._config.use_action_obs:
            obs["action"] = self._last_policy_action.copy()

        return obs

    def _calc_v_theta(self, action: Tuple[float, float, float]) -> Tuple[float, float]:
        """Calculates the v and theta from the action."""
        vx, vy, _ = action
        v = np.hypot(vx, vy)
        theta = np.arctan2(vy, vx) - self.qpos[2]
        return v, theta

    def apply_action(self, action: ActionType):
        """Calls the appropriate apply action method based on the heading joint type."""
        assert len(action) == 2, f"Action must have two elements, got {len(action)}."
        self._last_policy_action = np.asarray(action, dtype=np.float32).copy()
        super().apply_action(self._body_ctrl_from_action(self._last_policy_action))

    def reset(self, *args) -> ObsType:
        """Reset the policy-space efference copy before producing the first obs."""

        self._last_policy_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return super().reset(*args)

    def _body_ctrl_from_action(self, action: ActionType) -> ActionType:
        """Map [forward_velocity, heading] to the underlying body actuators."""
        # Calculate global velocities
        v = np.interp(action[0], [-1, 1], self._v_ctrlrange)
        current_heading = self.qpos[2]
        vx = v * np.cos(current_heading)
        vy = v * np.sin(current_heading)

        return [vx, vy, action[1]]

    @property
    def last_action(self) -> ActionType:
        """Return the exact policy-space command used by diagnostics and overlays."""

        return self._last_policy_action

    @cached_property
    def action_space(self) -> spaces.Space:
        """Overrides the base implementation to only have two elements."""
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)


class MjCambrianAgentPointEye(MjCambrianAgentPoint):
    """A point agent whose eye(s) are mounted on actuated pan/tilt gimbals, decoupling
    gaze from locomotion.

    The body is driven exactly like :class:`MjCambrianAgentPoint` (forward velocity +
    heading -> 3 body actuators). Any *additional* actuators on the agent body -- the
    eye gimbal's ``*_pan_act`` / ``*_tilt_act`` position servos -- are appended to the
    action space and commanded as absolute, body-relative angles (position control).

    For a single actuated eye there are 2 extra actuators, giving a 4D action space:
    ``[forward_velocity, heading, eye_pan, eye_tilt]``.

    Note:
        The 3 body actuators are always parsed first (they live in the agent xml and the
        eye actuators are appended afterwards during xml generation), so the eye
        actuators are simply ``self._actuators[3:]``.
    """

    # Number of body actuators (x-vel, y-vel, yaw) that precede the eye actuators.
    _N_BODY_ACTUATORS = 3

    def __init__(self, config: MjCambrianAgentConfig, name: str, *, kp: float = 0.75):
        super().__init__(config, name, kp=kp)

        # At init, self._actuators only holds the body actuators (the eye actuators are
        # appended to the full model and only appear in self._actuators after reset).
        # self._numctrl, however, is computed from the full agent+eye xml, so derive the
        # eye-actuator count from it.
        self._n_eye_actuators = self._numctrl - self._N_BODY_ACTUATORS
        assert self._n_eye_actuators >= 0, (
            f"Expected at least {self._N_BODY_ACTUATORS} actuators, "
            f"got {self._numctrl}."
        )
        self._eye_action_mode = self._config.eye_action_mode
        assert self._eye_action_mode in {"independent", "binocular", "yoked"}, (
            "eye_action_mode must be 'independent', 'binocular', or 'yoked', "
            f"got '{self._eye_action_mode}'."
        )
        if self._eye_action_mode in {"binocular", "yoked"}:
            assert self._n_eye_actuators == 4, (
                "binocular eye_action_mode requires exactly two pan/tilt eyes "
                f"(4 eye actuators), got {self._n_eye_actuators}."
            )
        self._eye_action_size = (
            2 if self._eye_action_mode == "yoked" else self._n_eye_actuators
        )
        self._last_eye_action = np.zeros(self._eye_action_size, dtype=np.float32)
        self._last_policy_action = np.zeros(
            2 + self._eye_action_size, dtype=np.float32
        )

    @property
    def _eye_actuators(self):
        """The eye gimbal actuators (everything after the body actuators). Populated
        after reset, when self._actuators reflects the full model."""
        return self._actuators[self._N_BODY_ACTUATORS :]

    def _eye_joint_state_obs(self) -> np.ndarray:
        """Return normalized physical eye joint positions and velocities."""
        if self._n_eye_actuators <= 0:
            return np.zeros(0, dtype=np.float32)

        qpos: list[float] = []
        qvel: list[float] = []
        for actuator in self._eye_actuators:
            joint_id = int(actuator.trnadr)
            qpos_adr = int(self._spec.model.jnt_qposadr[joint_id])
            qvel_adr = int(self._spec.model.jnt_dofadr[joint_id])
            value = float(self._spec.data.qpos[qpos_adr])
            velocity = float(self._spec.data.qvel[qvel_adr])
            if actuator.ctrllimited:
                value = float(np.interp(value, actuator.ctrlrange, [-1.0, 1.0]))
                scale = float(np.max(np.abs(actuator.ctrlrange)))
                if scale > 1e-8:
                    velocity = velocity / scale
            qpos.append(float(np.clip(value, -5.0, 5.0)))
            qvel.append(float(np.clip(velocity, -5.0, 5.0)))
        return np.asarray([*qpos, *qvel], dtype=np.float32)

    def apply_action(self, action: ActionType):
        """Applies a [v, theta, *eye] action.

        The first two elements drive the body (identical to the parent), and the
        remaining elements set the eye gimbal position actuators directly.
        """
        n_eye = self._eye_action_size
        assert len(action) == 2 + n_eye, (
            f"Action must have {2 + n_eye} elements "
            f"(2 body + {n_eye} eye), got {len(action)}."
        )

        self._last_policy_action = np.asarray(action, dtype=np.float32).copy()

        # Body: forward velocity + heading -> global vx, vy + yaw position. Use the
        # same body-control helper as the non-eye point agent, but write only the body
        # actuators so the eye actuators can be set explicitly below.
        MjCambrianAgent.apply_action(
            self, self._body_ctrl_from_action(self._last_policy_action[:2])
        )

        # Eyes: absolute pan/tilt position control. action is normalized [-1, 1].
        self._last_eye_action = self._last_policy_action[2:].copy()
        eye_ctrl = self._map_eye_action(self._last_eye_action)
        for a, actuator in zip(eye_ctrl, self._eye_actuators):
            if actuator.ctrllimited:
                a = np.interp(a, [-1, 1], actuator.ctrlrange)
            self._spec.data.ctrl[actuator.adr] = a

    def _map_eye_action(self, eye_action: np.ndarray) -> np.ndarray:
        """Map policy eye coordinates to physical pan/tilt actuator commands.

        independent: [left_pan, left_tilt, right_pan, right_tilt]
        binocular: [pan_version, pan_vergence, tilt_version, tilt_vergence]
        yoked: [pan, tilt] applied to both eyes
        """
        if self._eye_action_mode == "independent" or eye_action.size == 0:
            return eye_action
        if self._eye_action_mode == "yoked":
            pan, tilt = eye_action
            return np.asarray([pan, tilt, pan, tilt], dtype=np.float32)

        pan_version, pan_vergence, tilt_version, tilt_vergence = eye_action
        return np.clip(
            np.array(
                [
                    pan_version + 0.5 * pan_vergence,
                    tilt_version + 0.5 * tilt_vergence,
                    pan_version - 0.5 * pan_vergence,
                    tilt_version - 0.5 * tilt_vergence,
                ],
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )

    def _update_obs(self, obs: ObsType) -> ObsType:
        """Builds the action observation as [v, theta, *eye_commands].

        Bypasses :meth:`MjCambrianAgentPoint._update_obs`, which assumes a 3-element
        body action; here ``_last_action`` may carry extra eye actuators at reset.
        """
        obs = MjCambrianAgent._update_obs(self, obs)

        if self._config.use_action_obs:
            obs["action"] = self._last_policy_action.copy()
        if self._config.use_eye_state_obs:
            obs["eye_state"] = self._eye_joint_state_obs()

        return obs

    @cached_property
    def observation_space(self) -> spaces.Space:
        observation_space = super().observation_space
        if self._config.use_eye_state_obs and self._n_eye_actuators > 0:
            observation_space.spaces["eye_state"] = spaces.Box(
                low=-5.0,
                high=5.0,
                shape=(2 * self._n_eye_actuators,),
                dtype=np.float32,
            )
        return observation_space

    @cached_property
    def action_space(self) -> spaces.Space:
        """2 body dims plus the configured policy eye-action dimensions."""
        n = 2 + self._eye_action_size
        return spaces.Box(low=-1, high=1, shape=(n,), dtype=np.float32)


class MjCambrianAgentPointRelative(MjCambrianAgentPoint):
    """Point agent whose heading action is interpreted relative to current heading."""

    def __init__(
        self,
        config: MjCambrianAgentConfig,
        name: str,
        *,
        kp: float = 0.75,
        max_relative_heading: float = 0.5,
    ):
        super().__init__(config, name, kp=kp)
        self._max_relative_heading = max_relative_heading

    def _body_ctrl_from_action(self, action: ActionType) -> ActionType:
        v = np.interp(action[0], [-1, 1], self._v_ctrlrange)
        current_heading = self.qpos[2]
        vx = v * np.cos(current_heading)
        vy = v * np.sin(current_heading)

        heading_delta = float(action[1]) * self._max_relative_heading
        target_heading = current_heading + heading_delta
        target_heading = (target_heading + np.pi) % (2 * np.pi) - np.pi
        heading_action = np.interp(target_heading, self._theta_ctrlrange, [-1, 1])
        return [vx, vy, heading_action]


class MjCambrianAgentPointRelativeEye(MjCambrianAgentPointEye):
    """Actuated-eye point agent with relative body heading control."""

    def __init__(
        self,
        config: MjCambrianAgentConfig,
        name: str,
        *,
        kp: float = 0.75,
        max_relative_heading: float = 0.5,
    ):
        super().__init__(config, name, kp=kp)
        self._max_relative_heading = max_relative_heading

    def _body_ctrl_from_action(self, action: ActionType) -> ActionType:
        v = np.interp(action[0], [-1, 1], self._v_ctrlrange)
        current_heading = self.qpos[2]
        vx = v * np.cos(current_heading)
        vy = v * np.sin(current_heading)

        heading_delta = float(action[1]) * self._max_relative_heading
        target_heading = current_heading + heading_delta
        target_heading = (target_heading + np.pi) % (2 * np.pi) - np.pi
        heading_action = np.interp(target_heading, self._theta_ctrlrange, [-1, 1])
        return [vx, vy, heading_action]


class MjCambrianAgentPointSeeker(MjCambrianAgentPoint):
    """This is an agent which is non-trainable and defines a custom policy which
    acts as a 'homing' agent in the maze environment. This agent will attempt to reach
    a target (which is either randomly placed or another specific agent) by taking
    actions that best map to the optimal trajectory which is calculated from the maze
    using bfs (or just by choosing the action which minimizes the distance to
    the target).

    Keyword Args:
        target (Optional[str]): The name of the target agent to home in on. If
            None, a random free space in the maze will be chosen as the target.
        speed (float): The speed at which the agent moves. Defaults to -0.75.
        distance_threshold (float): The distance threshold at which the agent will
            consider itself to have reached the target. Defaults to 2.0.
        use_optimal_trajectory (bool): Whether to use the optimal trajectory to the
            target. Defaults to False.
        random_target_locations (str): Which maze cells can be sampled as random
            waypoints when target is None. "empty" preserves the old behavior of only
            sampling literal 0 cells; "free" samples all non-wall cells, including
            reset cells; "reset" samples from this agent's reset region.
    """

    def __init__(
        self,
        config: MjCambrianAgentConfig,
        name: str,
        *,
        target: Optional[str],
        speed: float = -0.75,
        distance_threshold: float = 2.0,
        use_optimal_trajectory: bool = False,
        random_target_locations: str = "empty",
    ):
        super().__init__(config, name)

        self._target = target
        self._speed = speed
        self._distance_threshold = distance_threshold
        assert random_target_locations in {"empty", "free", "reset"}, (
            "random_target_locations must be 'empty', 'free', or 'reset', "
            f"got '{random_target_locations}'."
        )
        self._random_target_locations = random_target_locations

        self._optimal_trajectory: np.ndarray = None
        self._use_optimal_trajectory = use_optimal_trajectory

        self._prev_target_pos: np.ndarray = None

    def reset(self, *args) -> ObsType:
        """Resets the optimal_trajectory."""
        self._optimal_trajectory = None
        self._prev_target_pos = None
        return super().reset(*args)

    def get_action_privileged(self, env: MjCambrianMazeEnv) -> ActionType:
        if self._target is None:
            if self._optimal_trajectory is None or len(self._optimal_trajectory) == 0:
                # Generate a random position to navigate to.
                if self._random_target_locations == "empty":
                    # Historical behavior: choose only literal empty cells.
                    rows, cols = np.where(env.maze.map == "0")
                    assert rows.size > 0, "No empty spaces in the maze"
                    index = np.random.randint(rows.size)
                    target_pos = env.maze.rowcol_to_xy((rows[index], cols[index]))
                elif self._random_target_locations == "free":
                    free = np.vectorize(
                        lambda cell: MjCambrianMapEntity.parse(str(cell))[0]
                        != MjCambrianMapEntity.WALL
                    )(env.maze.map)
                    rows, cols = np.where(free)
                    assert rows.size > 0, "No free spaces in the maze"
                    index = np.random.randint(rows.size)
                    target_pos = env.maze.rowcol_to_xy((rows[index], cols[index]))
                else:
                    locations = env.maze.reset_locations_for_agent(self.name)
                    target_pos = locations[np.random.randint(len(locations))]
            else:
                target_pos = self._optimal_trajectory[0]
        else:
            assert self._target in env.agents, f"Target {self._target} not found in env"
            target_pos = env.agents[self._target].pos

        if self._prev_target_pos is None:
            self._prev_target_pos = target_pos

        # Calculate the optimal trajectory if the current trajectory is None
        if (
            self._optimal_trajectory is None
            or np.linalg.norm(target_pos - self._prev_target_pos) > 0.1
        ):
            if self._use_optimal_trajectory:
                obstacles = []
                for agent_name, agent in env.agents.items():
                    if agent_name != self._target:
                        obstacles.append(tuple(env.maze.xy_to_rowcol(agent.pos)))
                try:
                    self._optimal_trajectory = env.maze.compute_optimal_path(
                        self.pos, target_pos, obstacles=obstacles
                    )
                except IndexError:
                    # Happens if there's no path to the target
                    get_logger().warning(
                        f"No path to target {target_pos} from {self.pos}"
                    )
                    self._optimal_trajectory = np.array([target_pos[:2]])
            else:
                self._optimal_trajectory = np.array([target_pos[:2]])

        # If the optimal trajectory is empty, then set the optimal trajectory to the
        # target position
        if len(self._optimal_trajectory) == 0:
            self._optimal_trajectory = np.array([target_pos[:2]])

        # Get the current target. If the distance between the current position and the
        # target is less than the threshold, then remove the target from the optimal
        # trajectory
        target = self._optimal_trajectory[0]
        target_vector = target - self.pos[:2]
        distance = np.linalg.norm(target - self.pos[:2])
        if distance < self._distance_threshold:
            self._optimal_trajectory = self._optimal_trajectory[1:]

        # Update the previous target position
        target_theta = np.arctan2(target_vector[1], target_vector[0])
        theta_action = np.interp(target_theta, [-np.pi, np.pi], [-1, 1])

        return [self._speed, theta_action]
