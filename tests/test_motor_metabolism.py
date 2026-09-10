import numpy as np

from cambrian.envs.reward_fns import reward_fn_motor_metabolism


class _ActionSpace:
    pass


class _Agent:
    name = "agent"
    action_space = _ActionSpace()
    last_action = np.zeros(0, dtype=np.float32)


class _Env:
    episode_step = 0
    max_episode_steps = 256


def test_motor_metabolism_penalizes_idle_body_and_eyes() -> None:
    action = np.array([0.0, -0.5, 0.25, -0.25, 0.0, 1.0], dtype=np.float32)
    reward = reward_fn_motor_metabolism(
        _Env(),
        _Agent(),
        False,
        False,
        {"action": action},
        idle_cost=0.001,
        body_speed_cost=0.002,
        body_yaw_cost=0.006,
        eye_cost=0.001,
        action_power=1.0,
    )
    expected_cost = 0.001 + 0.002 * 0.5 + 0.006 * 0.5 + 0.001 * 1.5
    assert np.isclose(reward, -expected_cost)


def test_motor_metabolism_can_be_quadratic() -> None:
    action = np.array([1.0, -0.5, 1.0, -1.0], dtype=np.float32)
    reward = reward_fn_motor_metabolism(
        _Env(),
        _Agent(),
        False,
        False,
        {"action": action},
        body_speed_cost=1.0,
        body_yaw_cost=1.0,
        eye_cost=1.0,
        action_power=2.0,
    )
    expected_cost = 1.0**2 + 0.5**2 + 1.0**2 + 1.0**2
    assert np.isclose(reward, -expected_cost)
