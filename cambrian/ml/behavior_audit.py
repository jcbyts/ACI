"""Behavior-level rollout auditing for visual target pursuit policies."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


@dataclass
class BehaviorGateThresholds:
    min_mean_captures: float = 1.0
    require_all_episodes_capture: bool = True
    max_stationary_fraction: float = 0.5
    max_median_distance: float = 5.0
    max_contact_fraction: float = 0.0
    max_near_border_fraction: float = 0.25


def _first_scalar(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _first_info(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, dict):
            return first
    return {}


def _maze_border_distance(env: Any, xy: np.ndarray) -> float | None:
    maze = getattr(env, "maze", None)
    if maze is None:
        return None

    half_w = maze.map_width_scaled / 2.0
    half_h = maze.map_length_scaled / 2.0
    center_x = -float(getattr(maze, "_starting_x", 0.0))
    left = center_x - half_w
    right = center_x + half_w
    bottom = -half_h
    top = half_h
    x, y = float(xy[0]), float(xy[1])
    return min(x - left, right - x, y - bottom, top - y)


def _agent_contact_pairs(env: Any, agent_name: str = "agent") -> list[str]:
    agent = env.agents[agent_name]
    pairs: list[str] = []
    for contact in env.data.contact:
        if contact.exclude:
            continue

        geom1 = int(contact.geom[0])
        geom2 = int(contact.geom[1])
        body1 = env.model.geom_bodyid[geom1]
        body2 = env.model.geom_bodyid[geom2]
        rootbody1 = env.model.body_rootid[body1]
        rootbody2 = env.model.body_rootid[body2]
        if rootbody1 != agent._body_id and rootbody2 != agent._body_id:
            continue

        name1 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, geom1) or f"geom_{geom1}"
        name2 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, geom2) or f"geom_{geom2}"
        pairs.append("::".join(sorted((name1, name2))))
    return sorted(set(pairs))


def summarize_episode(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}

    agent = np.asarray([s["agent_xy"] for s in samples], dtype=np.float64)
    goal = np.asarray([s["goal_xy"] for s in samples], dtype=np.float64)
    actions = np.asarray([s["action"] for s in samples], dtype=np.float64)
    deltas = np.linalg.norm(np.diff(agent, axis=0), axis=1)
    goal_deltas = np.linalg.norm(np.diff(goal, axis=0), axis=1)
    distances = np.linalg.norm(agent - goal, axis=1)
    border_values = [
        s["border_distance"] for s in samples if s["border_distance"] is not None
    ]
    border = np.asarray(border_values, dtype=np.float64)
    contact_pair_counts = Counter(
        pair for sample in samples for pair in sample.get("contact_pairs", [])
    )

    summary = {
        "steps": len(samples),
        "captures": int(sum(1 for s in samples if s["goal_respawned"])),
        "mean_reward": float(np.mean([s["reward"] for s in samples])),
        "total_reward": float(np.sum([s["reward"] for s in samples])),
        "initial_distance": float(distances[0]),
        "final_distance": float(distances[-1]),
        "median_distance": float(np.median(distances)),
        "min_distance": float(np.min(distances)),
        "near_goal_fraction": float(np.mean(distances < 1.25)),
        "total_path": float(np.sum(deltas)) if len(deltas) else 0.0,
        "net_displacement": float(np.linalg.norm(agent[-1] - agent[0])),
        "stationary_fraction": float(np.mean(deltas < 0.02)) if len(deltas) else 0.0,
        "goal_total_path": float(np.sum(goal_deltas)) if len(goal_deltas) else 0.0,
        "goal_mean_step": float(np.mean(goal_deltas)) if len(goal_deltas) else 0.0,
        "goal_stationary_fraction": (
            float(np.mean(goal_deltas < 0.02)) if len(goal_deltas) else 0.0
        ),
        "contact_fraction": float(np.mean([s["has_contacts"] for s in samples])),
        "had_contact": bool(any(s["has_contacts"] for s in samples)),
        "terminal_contact": bool(samples[-1]["done"] and samples[-1]["has_contacts"]),
        "contact_pairs": dict(contact_pair_counts.most_common()),
        "terminal_contact_pairs": samples[-1].get("contact_pairs", []),
        "action_mean": actions.mean(axis=0).round(4).tolist(),
        "action_min": actions.min(axis=0).round(4).tolist(),
        "action_max": actions.max(axis=0).round(4).tolist(),
        "speed_action_mean": float(np.mean(actions[:, 0])) if actions.size else 0.0,
        "speed_action_min": float(np.min(actions[:, 0])) if actions.size else 0.0,
        "speed_action_max": float(np.max(actions[:, 0])) if actions.size else 0.0,
        "physical_speed_command_mean": (
            float(np.mean((actions[:, 0] + 1.0) / 2.0)) if actions.size else 0.0
        ),
        "near_zero_speed_command_fraction": (
            float(np.mean(((actions[:, 0] + 1.0) / 2.0) < 0.02))
            if actions.size
            else 0.0
        ),
        "turn_action_abs_mean": (
            float(np.mean(np.abs(actions[:, 1]))) if actions.shape[1] > 1 else 0.0
        ),
        "start_agent_xy": agent[0].round(3).tolist(),
        "start_goal_xy": goal[0].round(3).tolist(),
        "end_agent_xy": agent[-1].round(3).tolist(),
        "end_goal_xy": goal[-1].round(3).tolist(),
    }
    if len(border):
        summary["median_border_distance"] = float(np.median(border))
        summary["near_border_fraction"] = float(np.mean(border < 1.25))
    else:
        summary["median_border_distance"] = None
        summary["near_border_fraction"] = 0.0
    return summary


def audit_policy_behavior(
    env: VecEnv,
    model: Any,
    *,
    n_episodes: int,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run deterministic policy rollouts and summarize pursuit behavior."""
    # Step the raw eval environment instead of VecEnv.step(). DummyVecEnv auto-resets
    # on done, which makes terminal samples look like reset samples and corrupts path
    # metrics for short failed episodes.
    raw_env = env.envs[0]
    cambrian_env = raw_env.unwrapped
    obs, _ = raw_env.reset()
    episodes: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    while len(episodes) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = raw_env.step(action)

        agent = cambrian_env.agents["agent"]
        goal = cambrian_env.agents["goal0"]
        agent_xy = np.asarray(agent.pos[:2], dtype=np.float64)
        goal_xy = np.asarray(goal.pos[:2], dtype=np.float64)
        agent_info = cambrian_env._info.get("agent", {})
        goal_info = cambrian_env._info.get("goal0", {})
        step_info = _first_info(info)
        action_array = np.asarray(
            step_info.get("rescaled_action", action),
            dtype=np.float64,
        ).reshape(-1)
        has_contacts = bool(
            step_info.get(
                "has_contacts",
                agent_info.get("has_contacts", False),
            )
        )
        done_bool = bool(terminated or truncated)

        current.append(
            {
                "agent_xy": agent_xy.tolist(),
                "goal_xy": goal_xy.tolist(),
                "action": action_array.tolist(),
                "reward": _first_scalar(reward),
                "goal_respawned": bool(goal_info.get("respawned", False)),
                "has_contacts": has_contacts,
                "contact_pairs": (
                    _agent_contact_pairs(cambrian_env) if has_contacts else []
                ),
                "border_distance": _maze_border_distance(cambrian_env, agent_xy),
                "done": done_bool,
            }
        )

        if done_bool:
            episodes.append(summarize_episode(current))
            current = []
            obs, _ = raw_env.reset()

    captures = [episode.get("captures", 0) for episode in episodes]
    stationary = [episode.get("stationary_fraction", 1.0) for episode in episodes]
    median_distance = [episode.get("median_distance", np.inf) for episode in episodes]
    contact = [episode.get("contact_fraction", 1.0) for episode in episodes]
    episode_contact = [episode.get("had_contact", True) for episode in episodes]
    terminal_contact = [episode.get("terminal_contact", True) for episode in episodes]
    border = [episode.get("near_border_fraction", 0.0) for episode in episodes]

    return {
        "n_episodes": len(episodes),
        "mean_captures": float(np.mean(captures)) if captures else 0.0,
        "all_episodes_captured": bool(all(capture > 0 for capture in captures)),
        "mean_stationary_fraction": float(np.mean(stationary)) if stationary else 1.0,
        "mean_median_distance": (
            float(np.mean(median_distance)) if median_distance else float("inf")
        ),
        "max_contact_fraction": float(np.max(contact)) if contact else 1.0,
        "episode_contact_fraction": (
            float(np.mean(episode_contact)) if episode_contact else 1.0
        ),
        "terminal_contact_fraction": (
            float(np.mean(terminal_contact)) if terminal_contact else 1.0
        ),
        "max_near_border_fraction": float(np.max(border)) if border else 0.0,
        "episodes": episodes,
    }


def evaluate_behavior_gate(
    audit: dict[str, Any],
    thresholds: BehaviorGateThresholds,
) -> tuple[bool, list[str]]:
    """Return whether an audit passes the behavior gate and why failures happened."""
    failures: list[str] = []
    if audit["mean_captures"] < thresholds.min_mean_captures:
        failures.append(
            "mean_captures "
            f"{audit['mean_captures']:.3f} < {thresholds.min_mean_captures:.3f}"
        )
    if thresholds.require_all_episodes_capture and not audit["all_episodes_captured"]:
        failures.append("not all episodes captured the target")
    if audit["mean_stationary_fraction"] > thresholds.max_stationary_fraction:
        failures.append(
            "mean_stationary_fraction "
            f"{audit['mean_stationary_fraction']:.3f} > "
            f"{thresholds.max_stationary_fraction:.3f}"
        )
    if audit["mean_median_distance"] > thresholds.max_median_distance:
        failures.append(
            "mean_median_distance "
            f"{audit['mean_median_distance']:.3f} > "
            f"{thresholds.max_median_distance:.3f}"
        )
    if audit["max_contact_fraction"] > thresholds.max_contact_fraction:
        failures.append(
            "max_contact_fraction "
            f"{audit['max_contact_fraction']:.3f} > "
            f"{thresholds.max_contact_fraction:.3f}"
        )
    if audit["max_near_border_fraction"] > thresholds.max_near_border_fraction:
        failures.append(
            "max_near_border_fraction "
            f"{audit['max_near_border_fraction']:.3f} > "
            f"{thresholds.max_near_border_fraction:.3f}"
        )
    return not failures, failures


def save_behavior_audit(path: Path | str, audit: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2) + "\n")


def thresholds_from_kwargs(**kwargs: Any) -> BehaviorGateThresholds:
    return BehaviorGateThresholds(**kwargs)


def thresholds_to_dict(thresholds: BehaviorGateThresholds) -> dict[str, Any]:
    return asdict(thresholds)
