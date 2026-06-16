#!/usr/bin/env python3
"""Audit saved Cambrian PPO policies from state, without changing observations.

The policy still receives only the configured observation. This script reads true
environment state after each action so we can diagnose behavior without relying on
rendered-video heuristics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from hydra_config import run_hydra

from cambrian import MjCambrianConfig
import cambrian.envs.maze_env  # noqa: F401 - register maze structured config
import cambrian.ml.features_extractors  # noqa: F401 - make custom extractors locatable
from cambrian.ml.trainer import MjCambrianTrainer


def _as_done(done: Any) -> bool:
    arr = np.asarray(done)
    return bool(arr.any())


def _space_keys(space: Any) -> list[str]:
    if hasattr(space, "spaces"):
        return sorted(map(str, space.spaces.keys()))
    return []


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


def _summarize_episode(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}

    agent = np.asarray([s["agent_xy"] for s in samples], dtype=np.float64)
    goal = np.asarray([s["goal_xy"] for s in samples], dtype=np.float64)
    deltas = np.linalg.norm(np.diff(agent, axis=0), axis=1)
    distances = np.linalg.norm(agent - goal, axis=1)
    border = np.asarray(
        [s["border_distance"] for s in samples if s["border_distance"] is not None],
        dtype=np.float64,
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
        "contact_fraction": float(np.mean([s["has_contacts"] for s in samples])),
        "start_agent_xy": agent[0].round(3).tolist(),
        "start_goal_xy": goal[0].round(3).tolist(),
        "end_agent_xy": agent[-1].round(3).tolist(),
        "end_goal_xy": goal[-1].round(3).tolist(),
    }
    if len(border):
        summary["median_border_distance"] = float(np.median(border))
        summary["near_border_fraction"] = float(np.mean(border < 1.25))
    return summary


def _main(config: MjCambrianConfig) -> dict[str, Any]:
    trainer = MjCambrianTrainer(config)
    env = trainer._make_env(config.eval_env, 1, monitor=None)
    model = trainer._make_model(env)

    model_path = Path(os.environ.get("MODEL_PATH", config.expdir / "best_model"))
    if model_path.suffix == ".zip":
        model_path = model_path.with_suffix("")
    model = model.load(model_path)

    cambrian_env = env.envs[0].unwrapped
    obs = env.reset()
    obs_keys = _space_keys(env.observation_space)

    n_episodes = int(os.environ.get("AUDIT_EPISODES", config.eval_env.n_eval_episodes))
    episodes: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    while len(episodes) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        agent = cambrian_env.agents["agent"]
        goal = cambrian_env.agents["goal0"]
        agent_xy = np.asarray(agent.pos[:2], dtype=np.float64)
        goal_xy = np.asarray(goal.pos[:2], dtype=np.float64)
        current.append(
            {
                "agent_xy": agent_xy.tolist(),
                "goal_xy": goal_xy.tolist(),
                "reward": float(np.asarray(reward).reshape(-1)[0]),
                "goal_respawned": bool(
                    cambrian_env._info.get("goal0", {}).get("respawned", False)
                ),
                "has_contacts": bool(agent.has_contacts),
                "border_distance": _maze_border_distance(cambrian_env, agent_xy),
            }
        )

        if _as_done(done):
            episodes.append(_summarize_episode(current))
            current = []

    captures = [episode.get("captures", 0) for episode in episodes]
    result = {
        "expdir": str(config.expdir),
        "model_path": str(model_path),
        "obs_keys": obs_keys,
        "action_space": str(env.action_space),
        "n_episodes": len(episodes),
        "mean_captures": float(np.mean(captures)) if captures else 0.0,
        "episodes": episodes,
    }

    out_path = os.environ.get("AUDIT_OUT")
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    env.close()
    return result


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parents[1] / "cambrian" / "configs"
    run_hydra(_main, config_path=config_path)
