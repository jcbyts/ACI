"""Reward fns. These can be used to calculate rewards for agents."""

from typing import Any, Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np

from cambrian.agents import MjCambrianAgent
from cambrian.envs import MjCambrianEnv
from cambrian.utils import agent_selected

# =====================
# Utility functions


def calc_delta(
    agent: MjCambrianAgent, info: Dict[str, Any], point: np.ndarray = np.array([0, 0])
) -> np.ndarray:
    """Calculates the delta position of the agent from a point.

    Returns:
        np.ndarray: The delta position of the agent from the point
            (i.e. current - prev).
    """

    current_distance = np.linalg.norm(agent.pos - point)
    prev_distance = np.linalg.norm(info["prev_pos"] - point)
    return current_distance - prev_distance


def calc_quickness(env: MjCambrianEnv) -> float:
    """Calculates the quickness of the agent."""
    return (
        max(env.max_episode_steps - env.episode_step, 0.0) / env.max_episode_steps
    ) ** (1 / 2)


def apply_reward_fn(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    *,
    reward_fn: Callable[..., float],
    for_agents: Optional[List[str]] = None,
    scale_by_quickness: bool = False,
    disable: bool = False,
    disable_on_max_episode_steps: bool = False,
) -> float:
    """Applies the reward function to the agent if it is in the for_agents list."""
    if disable or not agent_selected(agent, for_agents):
        return 0.0
    if disable_on_max_episode_steps and env.episode_step >= env.max_episode_steps - 1:
        return 0.0
    factor = calc_quickness(env) if scale_by_quickness else 1.0
    return reward_fn() * factor


# =====================
# Reward functions


def reward_fn_constant(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    **kwargs,
) -> float:
    """Returns a constant reward."""
    return apply_reward_fn(env, agent, reward_fn=lambda: reward, **kwargs)


def reward_fn_done(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    termination_reward: float = 0.0,
    truncation_reward: float = 0.0,
    **kwargs,
) -> float:
    """Rewards the agent if the episode is done. Termination indicates a successful
    episode, while truncation indicates an unsuccessful episode. If the time limit is
    reached, this is considered a termination. Applying a reward in this case can be
    disabled with the ``disable_on_max_episode_steps`` keyword argument.

    Keyword Args:
        termination_reward (float): The reward to give the agent if the episode is
            terminated. Defaults to 0.
        truncation_reward (float): The reward to give the agent if the episode is
            truncated. Defaults to 0.
    """

    def calc_reward():
        reward = 0.0
        if terminated:
            reward += termination_reward
        if truncated:
            reward += truncation_reward
        return reward

    return apply_reward_fn(
        env,
        agent,
        reward_fn=calc_reward,
        **kwargs,
    )


def reward_fn_euclidean_delta_from_init(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float = 1.0,
    **kwargs,
) -> float:
    """Rewards the change in distance over the previous step."""
    return apply_reward_fn(
        env,
        agent,
        reward_fn=lambda: calc_delta(agent, info, agent.init_pos) * reward,
        **kwargs,
    )


def reward_fn_euclidean_delta_to_agent(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    to_agents: Optional[List[str]] = None,
    **kwargs,
):
    """
    Rewards the change in distance to any enabled agent over the previous step.
    Convention is that a positive reward indicates getting closer to the agent.
    """

    def calc_deltas() -> float:
        accumulated_reward = 0.0
        for other_agent in env.agents.values():
            if not agent_selected(other_agent, to_agents):
                continue

            # NOTE: calc_delta returns a positive value if the agent moves away from the
            # agent. We'll multiple by -1 to flip the convention.
            delta = -reward * calc_delta(agent, info, other_agent.pos)
            accumulated_reward = delta

        return accumulated_reward

    return apply_reward_fn(env, agent, reward_fn=calc_deltas, **kwargs)


def reward_fn_heading_to_agent(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    to_agents: Optional[List[str]] = None,
    **kwargs,
) -> float:
    """Rewards facing selected agents in the 2D plane."""

    def calc_alignment() -> float:
        accumulated_reward = 0.0
        heading = agent.qpos[2]
        forward = np.array([np.cos(heading), np.sin(heading)])
        for other_agent in env.agents.values():
            if not agent_selected(other_agent, to_agents):
                continue

            delta = other_agent.pos[:2] - agent.pos[:2]
            norm = np.linalg.norm(delta)
            if norm == 0:
                continue
            accumulated_reward += reward * np.dot(forward, delta / norm)
        return accumulated_reward

    return apply_reward_fn(env, agent, reward_fn=calc_alignment, **kwargs)


def reward_fn_action_heading_to_agent(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    to_agents: Optional[List[str]] = None,
    action_index: int = 1,
    max_angle: float = np.pi / 2,
    **kwargs,
) -> float:
    """Rewards steering actions that turn toward selected agents."""

    def calc_action_alignment() -> float:
        action = info.get("action")
        if action is None or len(action) <= action_index:
            return 0.0

        accumulated_reward = 0.0
        for other_agent in env.agents.values():
            if not agent_selected(other_agent, to_agents):
                continue

            delta = other_agent.pos[:2] - agent.pos[:2]
            if np.linalg.norm(delta) == 0:
                continue

            target_heading = np.arctan2(delta[1], delta[0])
            heading_error = target_heading - agent.qpos[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            desired_turn = np.clip(heading_error / max_angle, -1.0, 1.0)
            accumulated_reward += reward * float(action[action_index]) * desired_turn
        return accumulated_reward

    return apply_reward_fn(env, agent, reward_fn=calc_action_alignment, **kwargs)


def reward_fn_agent_respawned(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    **kwargs,
) -> float:
    """This reward function rewards the agent if it has been respawned."""
    return apply_reward_fn(
        env,
        agent,
        reward_fn=lambda: reward if info.get("respawned", False) else 0.0,
        **kwargs,
    )


def reward_fn_close_to_agent(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    distance_threshold: float,
    from_agents: Optional[List[str]] = None,
    to_agents: Optional[List[str]] = None,
    **kwargs,
) -> float:
    """This reward function rewards the agent if it is close to another agent.

    Keyword Args:
        reward (float): The reward to give the agent if it is close to another agent.
            Default is 0.
        distance_threshold (float): The distance threshold to check if the agent is
            close to another agent.
        from_agents (Optional[List[str]]): The names of the agents that the reward
            should be calculated from. If None, the reward will be calculated from all
            agents.
        to_agents (Optional[List[str]]): The names of the agents that the reward
            should be calculated to. If None, the reward will be calculated to all
            agents.
    """

    def calc_deltas():
        accumulated_reward = 0
        for agent_name, agent in env.agents.items():
            if from_agents is not None and agent_name not in from_agents:
                continue

            for other_agent_name, other_agent in env.agents.items():
                if to_agents is not None and other_agent_name not in to_agents:
                    continue
                if agent_name == other_agent_name:
                    continue

                if np.linalg.norm(agent.pos - other_agent.pos) < distance_threshold:
                    accumulated_reward += reward
        return accumulated_reward

    return apply_reward_fn(env, agent, reward_fn=calc_deltas, **kwargs)


def reward_fn_has_contacts(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    **kwargs,
) -> float:
    """Rewards the agent if it has contacts."""
    return apply_reward_fn(
        env,
        agent,
        reward_fn=lambda: reward if info.get("has_contacts", False) else 0.0,
        **kwargs,
    )


def reward_fn_action(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    index: int | None = None,
    normalize: bool = False,
    absolute: bool = False,
    **kwargs,
) -> float:
    """Rewards the agent based on the action taken.

    Keyword Args:
        reward (float): The reward to give the agent if the action is taken.
        index (Optional[int]): The index of the action to use for the reward. If None,
            the sum of the action is used.
        normalize (bool): Whether to normalize the action to be in the range [0, 1).
        absolute (bool): Whether to use the absolute value of the action.
    """

    def calc_reward():
        nonlocal reward

        action = info.get("action")
        if action is None or len(action) == 0:
            return 0.0

        if normalize:
            action_space = agent.action_space
            assert isinstance(
                action_space, gym.spaces.Box
            ), "Action space must be a Box space"
            action = (action - action_space.low) / (
                action_space.high - action_space.low
            )
        if absolute:
            action = np.abs(action)

        if index is None:
            reward *= sum(action)
        else:
            assert (
                0 <= index < len(action)
            ), f"Invalid index {index} for action {action}"
            reward *= action[index]

        return reward

    return apply_reward_fn(env, agent, reward_fn=calc_reward, **kwargs)


def reward_fn_motor_metabolism(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    idle_cost: float = 0.0,
    body_speed_cost: float = 0.0,
    body_yaw_cost: float = 0.0,
    eye_cost: float = 0.0,
    action_power: float = 1.0,
    **kwargs,
) -> float:
    """Metabolic penalty for being alive and moving body/eyes.

    This reward is deliberately expressed in policy-action coordinates so it can be
    swept independently of the MuJoCo actuator ranges.  It is meant for embodied
    active-vision experiments where rotating/translating the whole body should be
    more expensive than moving the eyes.

    Action convention for point-eye agents:
        ``[forward_speed, body_yaw_rate, left_pan, left_tilt, right_pan, right_tilt]``.

    This experimental proxy penalizes commands, not measured mechanical work.
    Eye commands are positions: holding an eccentric gaze incurs a cost even when
    the eye is stationary. ``idle_cost`` applies on every step, regardless of motion.

    A positive cost becomes a negative reward. ``action_power=1`` gives an L1
    command penalty; ``action_power=2`` gives a quadratic command penalty.
    """

    def calc_cost() -> float:
        action = np.asarray(info.get("action", []), dtype=np.float64).reshape(-1)
        if action.size == 0:
            action = np.asarray(getattr(agent, "last_action", []), dtype=np.float64)
            action = action.reshape(-1)

        cost = float(idle_cost)
        if action.size >= 1:
            # The point-agent speed action maps [-1, 1] -> [0, 1] physical speed.
            normalized_speed = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
            cost += float(body_speed_cost) * normalized_speed**action_power
        if action.size >= 2:
            body_yaw = float(abs(np.clip(action[1], -1.0, 1.0)))
            cost += float(body_yaw_cost) * body_yaw**action_power
        if action.size > 2:
            eye_actions = np.abs(np.clip(action[2:], -1.0, 1.0))
            cost += float(eye_cost) * float(np.sum(eye_actions**action_power))
        return -cost

    return apply_reward_fn(env, agent, reward_fn=calc_cost, **kwargs)


def reward_combined(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    exclusive_fns: List[str] = [],
    **reward_fns,
) -> float:
    """Combines multiple reward functions into one.

    Keyword Args:
        exclusive_fns (Optional[List[str]]): If provided, only the reward functions
            with this name will be used if it's non-zero. As in, in order, the first
            function to return a non-zero reward will be returned.
    """
    accumulated_reward = 0
    for name, fn in reward_fns.items():
        reward = fn(env, agent, terminated, truncated, info)

        if name in exclusive_fns and reward != 0:
            return reward
        accumulated_reward += reward
    return accumulated_reward
