from __future__ import annotations

from copy import deepcopy

from scripts.audit_run_contract import Check, _check_resolved_config


def _canonical_config() -> dict:
    moving_target = {"instance": {"_target_": "cambrian.agents.point.MjCambrianAgentPointSeeker"}}
    return {
        "env": {
            "agents": {
                "agent": {
                    "instance": {"_target_": "cambrian.agents.point.MjCambrianAgentPointEye"},
                    "eye_action_mode": "independent",
                    "use_action_obs": True,
                    "use_eye_state_obs": True,
                    "eyes": {
                        "eye": {
                            "num_eyes": [1, 2],
                            "resolution": [20, 30],
                            "fov": [45.0, 67.5],
                            "lon_range": [-18.75, 18.75],
                            "gaze_lon_range": [-18.75, 18.75],
                            "actuated": True,
                            "noise_std": 0.0,
                            "integration_factor": 0.0,
                        }
                    },
                },
                "goal0": deepcopy(moving_target),
                "adversary0": deepcopy(moving_target),
            },
            "termination_fn": {
                "_target_": "combined",
                "terminate_if_close_to_goal": {
                    "for_agents": ["agent"],
                    "to_agents": ["goal0"],
                    "distance_threshold": 1.0,
                },
                "terminate_if_exceeds_max_episode_steps": {},
            },
            "truncation_fn": {
                "_target_": "combined",
                "truncate_if_close_to_adversary": {
                    "for_agents": ["agent"],
                    "to_agents": ["adversary0"],
                    "distance_threshold": 1.0,
                },
            },
            "reward_fn": {
                "_target_": "combined",
                "reward_if_done": {
                    "termination_reward": 1.0,
                    "truncation_reward": -1.0,
                    "scale_by_quickness": True,
                    "disable_on_max_episode_steps": True,
                    "for_agents": ["agent"],
                },
                "penalize_if_has_contacts": {
                    "reward": -1.0,
                    "for_agents": ["agent"],
                },
            },
        },
        "eval_env": {
            "step_fn": {
                "respawn_objects_if_agent_close": {
                    "for_agents": ["goal0", "adversary0"],
                    "to_agents": ["agent"],
                    "distance_threshold": 1.0,
                }
            },
            "reward_fn": {
                "reward_if_goal_respawned": {
                    "for_agents": ["goal0"],
                    "reward": 10.0,
                    "scale_by_quickness": True,
                },
                "penalize_if_adversary_respawned": {
                    "for_agents": ["adversary0"],
                    "reward": -20.0,
                    "scale_by_quickness": True,
                },
            },
        },
        "trainer": {
            "model": {
                "_target_": "cambrian.ml.model.MjCambrianRecurrentModel",
                "policy": "MultiInputLstmPolicy",
            },
            "wrappers": {
                "frame_stack_wrapper": None,
                "constant_action_wrapper": None,
            },
        },
    }


def _checks(config: dict) -> dict[str, Check]:
    result: list[Check] = []
    _check_resolved_config(config, result)
    return {check.name: check for check in result}


def test_canonical_resolved_config_passes_all_config_checks() -> None:
    checks = _checks(_canonical_config())
    failures = {name: check.actual for name, check in checks.items() if check.status != "pass"}
    assert failures == {}


def test_empty_adversary_target_is_detected_even_without_disable_flag() -> None:
    config = _canonical_config()
    config["env"]["truncation_fn"]["truncate_if_close_to_adversary"]["to_agents"] = []
    checks = _checks(config)
    assert checks["task.adversary_truncation_enabled"].status == "fail"


def test_contact_failure_and_reward_shaping_are_detected() -> None:
    config = _canonical_config()
    config["env"]["truncation_fn"]["truncate_if_has_contacts"] = {"for_agents": ["agent"]}
    config["env"]["reward_fn"]["approach_goal"] = {"reward": 0.1}
    checks = _checks(config)
    assert checks["task.no_added_done_events"].status == "fail"
    assert checks["task.no_added_reward_shaping"].status == "fail"
