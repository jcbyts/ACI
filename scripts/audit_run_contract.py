#!/usr/bin/env python3
"""Audit a saved experiment against the requested binocular tracking contract.

The audit reads *resolved run artifacts* rather than trusting experiment names or
launcher comments.  It intentionally avoids importing Cambrian, Hydra, Gymnasium,
MuJoCo, or Stable-Baselines3, so it can be run on a training machine or on an
archived run directory with only PyYAML (and optionally PyTorch for checkpoint
inspection).

Exit status is zero only when every required check passes.  Warnings are reported
but do not affect the exit status.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    expected: Any
    actual: Any
    detail: str
    required: bool = True


class TolerantSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that converts unknown Python tags to plain data.

    Hydra/OmegaConf may serialize ``pathlib.Path`` values with Python-specific YAML
    tags.  The audit does not instantiate those Python objects; it only needs their
    underlying scalar/sequence/mapping representation.
    """


def _construct_unknown_tag(
    loader: TolerantSafeLoader, tag_suffix: str, node: yaml.Node
) -> Any:
    if isinstance(node, ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, SequenceNode):
        values = loader.construct_sequence(node)
        if tag_suffix.endswith("pathlib.PosixPath") or tag_suffix.endswith(
            "pathlib.WindowsPath"
        ):
            return str(Path(*[str(value) for value in values]))
        return values
    if isinstance(node, MappingNode):
        return loader.construct_mapping(node)
    raise TypeError(f"Unsupported YAML node type: {type(node).__name__}")


TolerantSafeLoader.add_multi_constructor("", _construct_unknown_tag)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.load(path.read_text(), Loader=TolerantSafeLoader)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def _find_config(run_dir: Path) -> Path:
    candidates = [
        run_dir / "config.yaml",
        run_dir / "aci_config.yaml",
        run_dir / ".hydra" / "config.yaml",
    ]
    candidates.extend(sorted(run_dir.glob("hydra/**/.hydra/config.yaml")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No resolved config found in {run_dir}; looked for config.yaml, "
        "aci_config.yaml, and Hydra .hydra/config.yaml artifacts."
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _enabled(section: Mapping[str, Any], key: str) -> bool:
    value = section.get(key)
    return isinstance(value, Mapping) and not bool(value.get("disable", False))


def _listed_names(section: Mapping[str, Any], key: str) -> list[str]:
    return [str(value) for value in _sequence(section.get(key))]


def _contains_all(actual: Sequence[str], required: Sequence[str]) -> bool:
    return set(required).issubset(set(actual))


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _agent_names(config: Mapping[str, Any]) -> list[str]:
    agents = _mapping(_mapping(config.get("env")).get("agents"))
    return sorted(str(name) for name in agents)


def _prefixed(names: Iterable[str], prefix: str) -> list[str]:
    return sorted(name for name in names if name.startswith(prefix))


def _instance_target(agent: Mapping[str, Any]) -> str:
    return str(_mapping(agent.get("instance")).get("_target_", ""))


def _is_moving_seeker(agent: Mapping[str, Any]) -> bool:
    target = _instance_target(agent).lower()
    return "pointseeker" in target or "point_seeker" in target


def _as_float_pair(value: Any) -> list[float] | None:
    seq = _sequence(value)
    if len(seq) != 2:
        return None
    try:
        return [float(seq[0]), float(seq[1])]
    except (TypeError, ValueError):
        return None


def _as_int_pair(value: Any) -> list[int] | None:
    seq = _sequence(value)
    if len(seq) != 2:
        return None
    try:
        return [int(seq[0]), int(seq[1])]
    except (TypeError, ValueError):
        return None


def _add(
    checks: list[Check],
    name: str,
    passed: bool,
    *,
    expected: Any,
    actual: Any,
    detail: str,
    required: bool = True,
) -> None:
    checks.append(
        Check(
            name=name,
            status="pass" if passed else ("fail" if required else "warn"),
            expected=expected,
            actual=actual,
            detail=detail,
            required=required,
        )
    )


def _check_resolved_config(config: Mapping[str, Any], checks: list[Check]) -> None:
    env = _mapping(config.get("env"))
    eval_env = _mapping(config.get("eval_env"))
    agents = _mapping(env.get("agents"))
    names = sorted(str(name) for name in agents)
    goals = _prefixed(names, "goal")
    adversaries = _prefixed(names, "adversary")

    _add(
        checks,
        "task.primary_agent_present",
        "agent" in agents,
        expected="agent",
        actual=names,
        detail="The controllable agent must exist in the resolved environment.",
    )
    _add(
        checks,
        "task.goal_present",
        len(goals) >= 1,
        expected=">=1 goal* agent",
        actual=goals,
        detail="The original tracking task contains a target.",
    )
    _add(
        checks,
        "task.adversary_present",
        len(adversaries) >= 1,
        expected=">=1 adversary* agent",
        actual=adversaries,
        detail="A goal-only run is not the requested tracking task.",
    )

    moving_goals = [name for name in goals if _is_moving_seeker(_mapping(agents[name]))]
    moving_adversaries = [
        name for name in adversaries if _is_moving_seeker(_mapping(agents[name]))
    ]
    _add(
        checks,
        "task.goal_moves",
        len(goals) > 0 and len(moving_goals) == len(goals),
        expected="all goal* agents use a PointSeeker implementation",
        actual={name: _instance_target(_mapping(agents[name])) for name in goals},
        detail="Detection uses static objects; tracking replaces them with moving seekers.",
    )
    _add(
        checks,
        "task.adversary_moves",
        len(adversaries) > 0 and len(moving_adversaries) == len(adversaries),
        expected="all adversary* agents use a PointSeeker implementation",
        actual={
            name: _instance_target(_mapping(agents[name])) for name in adversaries
        },
        detail="The requested adversary must move, not merely be present.",
    )

    termination = _mapping(env.get("termination_fn"))
    truncation = _mapping(env.get("truncation_fn"))
    goal_termination = _mapping(termination.get("terminate_if_close_to_goal"))
    adversary_truncation = _mapping(
        truncation.get("truncate_if_close_to_adversary")
    )
    goal_from = _listed_names(goal_termination, "for_agents")
    goal_to = _listed_names(goal_termination, "to_agents")
    adversary_from = _listed_names(adversary_truncation, "for_agents")
    adversary_to = _listed_names(adversary_truncation, "to_agents")
    goal_termination_ok = (
        _enabled(termination, "terminate_if_close_to_goal")
        and "agent" in goal_from
        and bool(goals)
        and _contains_all(goal_to, goals)
        and _as_float(goal_termination.get("distance_threshold")) == 1.0
    )
    adversary_truncation_ok = (
        _enabled(truncation, "truncate_if_close_to_adversary")
        and "agent" in adversary_from
        and bool(adversaries)
        and _contains_all(adversary_to, adversaries)
        and _as_float(adversary_truncation.get("distance_threshold")) == 1.0
    )
    _add(
        checks,
        "task.goal_termination_enabled",
        goal_termination_ok,
        expected={
            "enabled": True,
            "for_agents": ["agent"],
            "to_agents": goals or ["goal*"],
            "distance_threshold": 1.0,
        },
        actual=goal_termination,
        detail=(
            "Training must retain the original agent-to-goal termination; a present "
            "but disabled or empty-target clause is not sufficient."
        ),
    )
    _add(
        checks,
        "task.adversary_truncation_enabled",
        adversary_truncation_ok,
        expected={
            "enabled": True,
            "for_agents": ["agent"],
            "to_agents": adversaries or ["adversary*"],
            "distance_threshold": 1.0,
        },
        actual=adversary_truncation,
        detail=(
            "Training must retain the original agent-to-adversary failure event; an "
            "empty to_agents list silently disables the event even without disable=true."
        ),
    )

    allowed_termination_keys = {
        "_target_",
        "_partial_",
        "terminate_if_close_to_goal",
        "terminate_if_exceeds_max_episode_steps",
    }
    allowed_truncation_keys = {
        "_target_",
        "_partial_",
        "truncate_if_close_to_adversary",
    }
    extra_done_events = {
        "termination": sorted(set(map(str, termination)) - allowed_termination_keys),
        "truncation": sorted(set(map(str, truncation)) - allowed_truncation_keys),
    }
    _add(
        checks,
        "task.no_added_done_events",
        not extra_done_events["termination"] and not extra_done_events["truncation"],
        expected={"termination": [], "truncation": []},
        actual=extra_done_events,
        detail=(
            "Contact failure or other added done events change the original tracking "
            "problem even when the reward table looks similar."
        ),
    )

    eval_step = _mapping(eval_env.get("step_fn"))
    respawn = _mapping(eval_step.get("respawn_objects_if_agent_close"))
    respawn_agents = _listed_names(respawn, "for_agents")
    respawn_to = _listed_names(respawn, "to_agents")
    respawn_ok = (
        _enabled(eval_step, "respawn_objects_if_agent_close")
        and bool(goals)
        and bool(adversaries)
        and _contains_all(respawn_agents, [*goals, *adversaries])
        and "agent" in respawn_to
        and _as_float(respawn.get("distance_threshold")) == 1.0
    )
    _add(
        checks,
        "task.eval_respawns_goal_and_adversary",
        respawn_ok,
        expected={
            "for_agents": [*(goals or ["goal*"]), *(adversaries or ["adversary*"])],
            "to_agents": ["agent"],
            "distance_threshold": 1.0,
        },
        actual=respawn,
        detail="Evaluation should score repeated target contacts and adversary contacts.",
    )

    eval_reward = _mapping(eval_env.get("reward_fn"))
    goal_eval_reward = _mapping(eval_reward.get("reward_if_goal_respawned"))
    adversary_eval_reward = _mapping(
        eval_reward.get("penalize_if_adversary_respawned")
    )
    goal_eval_ok = (
        _enabled(eval_reward, "reward_if_goal_respawned")
        and bool(goals)
        and _contains_all(_listed_names(goal_eval_reward, "for_agents"), goals)
        and _as_float(goal_eval_reward.get("reward")) == 10.0
        and goal_eval_reward.get("scale_by_quickness") is True
    )
    adversary_eval_ok = (
        _enabled(eval_reward, "penalize_if_adversary_respawned")
        and bool(adversaries)
        and _contains_all(
            _listed_names(adversary_eval_reward, "for_agents"), adversaries
        )
        and _as_float(adversary_eval_reward.get("reward")) == -20.0
        and adversary_eval_reward.get("scale_by_quickness") is True
    )
    _add(
        checks,
        "task.eval_goal_reward_enabled",
        goal_eval_ok,
        expected={
            "enabled": True,
            "for_agents": goals or ["goal*"],
            "reward": 10.0,
            "scale_by_quickness": True,
        },
        actual=goal_eval_reward,
        detail="The repeated-contact evaluation must retain the original target score.",
    )
    _add(
        checks,
        "task.eval_adversary_penalty_enabled",
        adversary_eval_ok,
        expected={
            "enabled": True,
            "for_agents": adversaries or ["adversary*"],
            "reward": -20.0,
            "scale_by_quickness": True,
        },
        actual=adversary_eval_reward,
        detail="Removing or weakening this term converts evaluation into goal-only pursuit.",
    )

    train_reward = _mapping(env.get("reward_fn"))
    done_reward = _mapping(train_reward.get("reward_if_done"))
    contact_reward = _mapping(train_reward.get("penalize_if_has_contacts"))
    original_train_reward_ok = (
        _enabled(train_reward, "reward_if_done")
        and _as_float(done_reward.get("termination_reward")) == 1.0
        and _as_float(done_reward.get("truncation_reward")) == -1.0
        and done_reward.get("scale_by_quickness") is True
        and done_reward.get("disable_on_max_episode_steps") is True
        and "agent" in _listed_names(done_reward, "for_agents")
        and _enabled(train_reward, "penalize_if_has_contacts")
        and _as_float(contact_reward.get("reward")) == -1.0
        and "agent" in _listed_names(contact_reward, "for_agents")
    )
    _add(
        checks,
        "task.original_training_reward_values",
        original_train_reward_ok,
        expected={
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
        actual={
            "reward_if_done": done_reward,
            "penalize_if_has_contacts": contact_reward,
        },
        detail=(
            "The final baseline is the upstream tracking objective, not a rescaled or "
            "shaped proxy."
        ),
    )

    original_train_reward_keys = {
        "_target_",
        "_partial_",
        "reward_if_done",
        "penalize_if_has_contacts",
    }
    reward_keys = set(str(key) for key in train_reward)
    extra_reward_keys = sorted(reward_keys - original_train_reward_keys)
    _add(
        checks,
        "task.no_added_reward_shaping",
        not extra_reward_keys,
        expected=[],
        actual=extra_reward_keys,
        detail=(
            "Approach, heading, action-alignment, time, or respawn shaping changes the "
            "scientific task and should not be in the canonical baseline."
        ),
    )

    trainer = _mapping(config.get("trainer"))
    model = _mapping(trainer.get("model"))
    model_target = str(model.get("_target_", ""))
    policy = str(model.get("policy", ""))
    _add(
        checks,
        "policy.recurrent_model",
        "recurrent" in model_target.lower() and "lstm" in policy.lower(),
        expected="MjCambrianRecurrentModel + cyclopean LSTM policy",
        actual={"_target_": model_target, "policy": policy},
        detail="The requested PPO baseline must use recurrent sequence training.",
    )

    wrappers = _mapping(trainer.get("wrappers"))
    frame_stack = _mapping(wrappers.get("frame_stack_wrapper"))
    frame_stack_size = _as_float(frame_stack.get("stack_size"))
    _add(
        checks,
        "policy.spatiotemporal_stack_enabled",
        frame_stack_size is not None and frame_stack_size >= 3,
        expected={"stack_size": ">= 3"},
        actual=frame_stack or None,
        detail=(
            "The R(2+1)D retina requires an explicit temporal stack. The LSTM adds "
            "persistent state rather than replacing local retinal motion input."
        ),
    )
    constant_wrapper = wrappers.get("constant_action_wrapper")
    constant_actions = _mapping(_mapping(constant_wrapper).get("constant_actions"))
    constant_action_free = constant_wrapper is None or len(constant_actions) == 0
    _add(
        checks,
        "policy.no_constant_actions",
        constant_action_free,
        expected="no fixed action dimensions",
        actual=constant_actions,
        detail="Fixing body speed changes the control problem and action dimensionality.",
    )

    primary = _mapping(agents.get("agent"))
    eye_groups = _mapping(primary.get("eyes"))
    eye = _mapping(eye_groups.get("eye"))
    num_eyes = _as_int_pair(eye.get("num_eyes"))
    eye_count = math.prod(num_eyes) if num_eyes else None
    _add(
        checks,
        "vision.exactly_two_eyes",
        eye_count == 2,
        expected=2,
        actual={"num_eyes": num_eyes, "product": eye_count},
        detail="The canonical baseline must be binocular, not three-eye or monocular.",
    )
    _add(
        checks,
        "vision.eyes_actuated",
        eye.get("actuated") is True,
        expected=True,
        actual=eye.get("actuated"),
        detail="The policy must control pan and tilt for both eyes.",
    )
    _add(
        checks,
        "vision.independent_eye_actions",
        str(primary.get("eye_action_mode", "")).lower() == "independent",
        expected="independent",
        actual=primary.get("eye_action_mode"),
        detail="Independent mode exposes four eye commands (left/right pan/tilt).",
    )
    _add(
        checks,
        "vision.action_efference_copy_enabled",
        primary.get("use_action_obs") is True,
        expected=True,
        actual=primary.get("use_action_obs"),
        detail="The recurrent controller should observe its previous policy-space command.",
    )
    _add(
        checks,
        "vision.eye_state_enabled",
        primary.get("use_eye_state_obs") is True,
        expected=True,
        actual=primary.get("use_eye_state_obs"),
        detail="Actuated-eye proprioception is needed to interpret retinal direction.",
    )

    resolution = _as_int_pair(eye.get("resolution"))
    fov = _as_float_pair(eye.get("fov"))
    lon = _as_float_pair(eye.get("lon_range"))
    gaze_lon = _as_float_pair(eye.get("gaze_lon_range"))
    _add(
        checks,
        "vision.matched_retinal_geometry_config",
        resolution == [20, 30]
        and fov is not None
        and all(abs(a - b) < 1e-6 for a, b in zip(fov, [45.0, 67.5]))
        and lon is not None
        and all(abs(a - b) < 1e-6 for a, b in zip(lon, [-18.75, 18.75]))
        and gaze_lon is not None
        and all(abs(a - b) < 1e-6 for a, b in zip(gaze_lon, [-18.75, 18.75])),
        expected={
            "resolution": [20, 30],
            "fov": [45.0, 67.5],
            "lon_range": [-18.75, 18.75],
            "gaze_lon_range": [-18.75, 18.75],
        },
        actual={
            "resolution": resolution,
            "fov": fov,
            "lon_range": lon,
            "gaze_lon_range": gaze_lon,
        },
        detail=(
            "This two-eye design matches the original three-eye pixel count, angular "
            "sampling, union coverage, and total overlap."
        ),
    )

    noise = eye.get("noise_std")
    integration = eye.get("integration_factor")
    _add(
        checks,
        "vision.sensor_manipulations_disabled",
        noise in (0, 0.0) and integration in (0, 0.0),
        expected={"noise_std": 0.0, "integration_factor": 0.0},
        actual={"noise_std": noise, "integration_factor": integration},
        detail="Sensor hypotheses must be introduced only after the clean baseline works.",
    )


def _parse_floats(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(value) for value in text.split()]


def _quat_rotate_wxyz(quat: Sequence[float], vector: Sequence[float]) -> list[float]:
    """Rotate a vector by a normalized wxyz quaternion without SciPy."""

    w, x, y, z = quat
    vx, vy, vz = vector
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = (value / norm for value in (w, x, y, z))

    # q * v * q^-1, expanded as v + 2w(q_vec x v) + 2(q_vec x (q_vec x v)).
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _transform_element(
    camera: ET.Element, parent_map: Mapping[ET.Element, ET.Element]
) -> ET.Element | None:
    if "quat" in camera.attrib:
        return camera
    parent = parent_map.get(camera)
    while parent is not None:
        if parent.tag == "body" and "quat" in parent.attrib:
            return parent
        parent = parent_map.get(parent)
    return None


def _camera_summaries(xml_path: Path) -> list[dict[str, Any]]:
    root = ET.parse(xml_path).getroot()
    parent_map = {child: parent for parent in root.iter() for child in parent}
    summaries: list[dict[str, Any]] = []
    for camera in root.findall(".//camera"):
        name = camera.attrib.get("name", "")
        if not name.startswith("agent_eye"):
            continue
        transform = _transform_element(camera, parent_map)
        quat = _parse_floats(transform.attrib.get("quat") if transform is not None else None)
        yaw: float | None = None
        if len(quat) == 4:
            forward = _quat_rotate_wxyz(quat, [0.0, 0.0, -1.0])
            yaw = _wrap_deg(math.degrees(math.atan2(forward[1], forward[0])))
        sensorsize = _parse_floats(camera.attrib.get("sensorsize"))
        focal = _parse_floats(camera.attrib.get("focal"))
        fov: list[float] = []
        if len(sensorsize) == 2 and len(focal) == 2:
            fov = [
                math.degrees(2.0 * math.atan(sensor / (2.0 * foc)))
                for sensor, foc in zip(sensorsize, focal)
            ]
        summaries.append(
            {
                "name": name,
                "yaw_deg": yaw,
                "resolution": [
                    int(float(value))
                    for value in camera.attrib.get("resolution", "").split()
                ],
                "sensorsize": sensorsize,
                "focal": focal,
                "fov_deg": fov,
            }
        )
    return sorted(
        summaries,
        key=lambda summary: (
            float("inf") if summary["yaw_deg"] is None else summary["yaw_deg"]
        ),
    )


def _allclose(actual: Sequence[float], expected: Sequence[float], atol: float) -> bool:
    return len(actual) == len(expected) and all(
        abs(float(a) - float(e)) <= atol for a, e in zip(actual, expected)
    )


def _check_compiled_xml(run_dir: Path, checks: list[Check]) -> None:
    xml_path = run_dir / "compiled_env.xml"
    if not xml_path.is_file():
        _add(
            checks,
            "artifact.compiled_xml_present",
            False,
            expected=str(xml_path),
            actual=None,
            detail="A run cannot be audited against the actual MuJoCo model without it.",
        )
        return

    _add(
        checks,
        "artifact.compiled_xml_present",
        True,
        expected=str(xml_path),
        actual=str(xml_path),
        detail="Checking compiled geometry rather than intended Hydra values.",
    )
    summaries = _camera_summaries(xml_path)
    _add(
        checks,
        "compiled.exactly_two_eye_cameras",
        len(summaries) == 2,
        expected=2,
        actual=[summary["name"] for summary in summaries],
        detail="The compiled model is the source of truth for camera count.",
    )

    yaws = [summary["yaw_deg"] for summary in summaries]
    yaw_ok = len(yaws) == 2 and all(value is not None for value in yaws) and _allclose(
        [float(value) for value in yaws], [-18.75, 18.75], 1e-3
    )
    _add(
        checks,
        "compiled.zero_action_camera_yaw",
        yaw_ok,
        expected=[-18.75, 18.75],
        actual=yaws,
        detail=(
            "Eye mount placement alone is insufficient: the optical axes must point "
            "outward at the matched baseline pose."
        ),
    )

    expected_resolution = [30, 20]  # MuJoCo XML stores [width, height].
    expected_fov = [45.0, 67.5]
    expected_sensorsize = [0.008284271247461903, 0.013363180779263536]
    resolution_ok = len(summaries) == 2 and all(
        summary["resolution"] == expected_resolution for summary in summaries
    )
    fov_ok = len(summaries) == 2 and all(
        _allclose(summary["fov_deg"], expected_fov, 1e-3) for summary in summaries
    )
    sensor_ok = len(summaries) == 2 and all(
        _allclose(summary["sensorsize"], expected_sensorsize, 1e-6)
        for summary in summaries
    )
    _add(
        checks,
        "compiled.retinal_resolution",
        resolution_ok,
        expected=expected_resolution,
        actual=[summary["resolution"] for summary in summaries],
        detail="Two 20x30 retinas preserve the original total pixel count.",
    )
    _add(
        checks,
        "compiled.retinal_fov",
        fov_ok and sensor_ok,
        expected={"fov_deg": expected_fov, "sensorsize": expected_sensorsize},
        actual=[
            {
                "name": summary["name"],
                "fov_deg": summary["fov_deg"],
                "sensorsize": summary["sensorsize"],
            }
            for summary in summaries
        ],
        detail="FOV is recomputed from compiled focal length and sensor size.",
    )

    root = ET.parse(xml_path).getroot()
    eye_actuators = [
        element.attrib.get("name", "")
        for actuator in root.findall("./actuator")
        for element in list(actuator)
        if element.attrib.get("name", "").startswith("agent_eye")
    ]
    _add(
        checks,
        "compiled.four_eye_actuators",
        len(eye_actuators) == 4,
        expected=4,
        actual=eye_actuators,
        detail="Independent binocular pan/tilt control requires four compiled actuators.",
    )

    body_names = [element.attrib.get("name", "") for element in root.findall(".//body")]
    compiled_goals = [name for name in body_names if name.startswith("goal")]
    compiled_adversaries = [name for name in body_names if name.startswith("adversary")]
    _add(
        checks,
        "compiled.goal_and_adversary_bodies",
        bool(compiled_goals) and bool(compiled_adversaries),
        expected="at least one goal* body and one adversary* body",
        actual={"goals": compiled_goals, "adversaries": compiled_adversaries},
        detail="This catches config/run disagreement and goal-only compiled mazes.",
    )


def _parse_eye_shapes(space_text: str) -> dict[str, list[int]]:
    pattern = re.compile(
        r"['\"](?P<name>agent_eye[^'\"]+)['\"]\s*:\s*Box\([^)]*?"
        r"\((?P<shape>\d+(?:\s*,\s*\d+){2,3})\)"
    )
    result: dict[str, list[int]] = {}
    for match in pattern.finditer(space_text):
        result[match.group("name")] = [
            int(value.strip()) for value in match.group("shape").split(",")
        ]
    return result


def _expected_channels(shape: Sequence[int]) -> int | None:
    if len(shape) not in (3, 4):
        return None
    image = list(shape[-3:])
    first, middle, last = image
    first_plausible = first < min(middle, last)
    last_plausible = last < min(first, middle)
    if first_plausible == last_plausible:
        return None
    return last if last_plausible else first


def _check_sb3_checkpoint(run_dir: Path, checks: list[Check]) -> None:
    zip_path = run_dir / "best_model.zip"
    if not zip_path.is_file():
        _add(
            checks,
            "checkpoint.sb3_archive_present",
            False,
            expected=str(zip_path),
            actual=None,
            detail="Skipped for non-SB3 runs or runs without a saved best model.",
            required=False,
        )
        return

    _add(
        checks,
        "checkpoint.sb3_archive_present",
        True,
        expected=str(zip_path),
        actual=str(zip_path),
        detail="Inspecting the serialized observation declaration and weight shapes.",
        required=False,
    )
    try:
        with ZipFile(zip_path) as archive:
            data = json.loads(archive.read("data"))
            policy_bytes = archive.read("policy.pth")
    except Exception as exc:  # pragma: no cover - defensive archive handling
        _add(
            checks,
            "checkpoint.archive_readable",
            False,
            expected="readable SB3 zip",
            actual=repr(exc),
            detail="The archive could not be inspected.",
        )
        return

    observation_data = _mapping(data.get("observation_space"))
    eye_shapes = _parse_eye_shapes(str(observation_data.get("spaces", "")))
    _add(
        checks,
        "checkpoint.two_eye_observation_tensors",
        len(eye_shapes) == 2,
        expected=2,
        actual=eye_shapes,
        detail="The saved policy should expose one image tensor per eye.",
    )

    try:
        import torch
    except ImportError:
        _add(
            checks,
            "checkpoint.cnn_channel_axis",
            False,
            expected="PyTorch available for weight inspection",
            actual="torch is not installed",
            detail="Observation shapes were read, but convolution weights were not.",
            required=False,
        )
        return

    try:
        state_dict = torch.load(
            BytesIO(policy_bytes), map_location="cpu", weights_only=True
        )
    except Exception as exc:  # pragma: no cover - version-dependent torch loading
        _add(
            checks,
            "checkpoint.cnn_channel_axis",
            False,
            expected="readable policy.pth state dict",
            actual=repr(exc),
            detail="Could not load the checkpoint with weights_only=True.",
        )
        return

    first_conv_suffixes = ("cnn.0.weight", "spatial_stem.0.weight")
    conv_candidates = [
        (name, value)
        for name, value in state_dict.items()
        if "features_extractor" in name
        and name.endswith(first_conv_suffixes)
        and getattr(value, "ndim", None) in (4, 5)
    ]
    unique_in_channels = sorted({int(value.shape[1]) for _, value in conv_candidates})
    expected_channel_counts = sorted(
        {
            channels
            for channels in (_expected_channels(shape) for shape in eye_shapes.values())
            if channels is not None
        }
    )
    channels_ok = (
        len(conv_candidates) > 0
        and expected_channel_counts == [3]
        and unique_in_channels == expected_channel_counts
    )
    _add(
        checks,
        "checkpoint.cnn_channel_axis",
        channels_ok,
        expected={"retinal_channels": expected_channel_counts or [3]},
        actual={
            "first_conv_in_channels": unique_in_channels,
            "example_weights": [
                {"name": name, "shape": list(value.shape)}
                for name, value in conv_candidates[:4]
            ],
        },
        detail=(
            "A retinal tensor must enter Conv2d/Conv3d with 3 channels. Twenty "
            "input channels means height was silently interpreted as channels."
        ),
    )


def _print_human(report: Mapping[str, Any]) -> None:
    print(f"Run: {report['run_dir']}")
    print(f"Config: {report['config_path']}")
    print(
        f"Result: {report['status'].upper()} "
        f"({report['summary']['passed']} passed, "
        f"{report['summary']['failed']} failed, "
        f"{report['summary']['warnings']} warnings)"
    )
    print()
    for item in report["checks"]:
        marker = {"pass": "PASS", "fail": "FAIL", "warn": "WARN"}[item["status"]]
        print(f"[{marker}] {item['name']}")
        if item["status"] != "pass":
            print(f"  expected: {json.dumps(item['expected'], sort_keys=True)}")
            print(f"  actual:   {json.dumps(item['actual'], sort_keys=True)}")
        print(f"  {item['detail']}")


def audit_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    config_path = _find_config(run_dir)
    config = _load_yaml(config_path)
    checks: list[Check] = []
    _check_resolved_config(config, checks)
    _check_compiled_xml(run_dir, checks)
    _check_sb3_checkpoint(run_dir, checks)

    failed = sum(check.status == "fail" for check in checks)
    warnings = sum(check.status == "warn" for check in checks)
    passed = sum(check.status == "pass" for check in checks)
    return {
        "contract": "tracking_2eye_recurrent_actuated_v1",
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "status": "pass" if failed == 0 else "fail",
        "summary": {"passed": passed, "failed": failed, "warnings": warnings},
        "checks": [asdict(check) for check in checks],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Saved experiment directory")
    parser.add_argument(
        "--json-out", type=Path, help="Optional path for a machine-readable report"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print JSON instead of the human report"
    )
    args = parser.parse_args()

    try:
        report = audit_run(args.run_dir)
    except Exception as exc:
        print(f"audit failed to execute: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
