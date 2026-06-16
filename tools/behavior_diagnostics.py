"""Evaluate trained policies with behavior and geometry-based retinal velocity metrics.

This tool is intentionally separate from the training loop. It loads a saved run,
rolls out the policy in the eval environment, and writes:

- per_step.csv: positions, distances, actions, capture flags, retinal speeds
- behavior_summary.json: aggregate chase/capture/activity/saccade-fixation metrics
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cambrian.ml.model import MjCambrianModel
from cambrian.utils.wrappers import make_wrapped_env


class ConfigDict(dict):
    """Dict with attribute access, matching the config objects used at runtime."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self):
        return ConfigDict({key: _to_config(value) for key, value in self.items()})


def _to_config(value: Any) -> Any:
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            converted[key] = (
                ConfigDict() if key == "config" and item is None else _to_config(item)
            )
        return ConfigDict(converted)
    if isinstance(value, list):
        return [_to_config(item) for item in value]
    return value


def _as_scalar_done(done: Any) -> bool:
    arr = np.asarray(done)
    return bool(arr.reshape(-1)[0])


def _as_action_vector(action: Any) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    return arr.reshape(-1)


def _camera_names(model: mj.MjModel, prefix: str) -> list[tuple[int, str]]:
    cameras = []
    for cam_id in range(model.ncam):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, cam_id)
        if name and name.startswith(prefix):
            cameras.append((cam_id, name))
    return cameras


def _joint_names(model: mj.MjModel, prefix: str) -> list[tuple[int, str]]:
    joints = []
    for joint_id in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        if name and name.startswith(prefix):
            joints.append((joint_id, name))
    return joints


def _project_points(env, cam_id: int, points: np.ndarray) -> np.ndarray:
    """Project world points to normalized sensor coordinates via MuJoCo camera pose."""
    cam_pos = np.asarray(env.data.cam_xpos[cam_id], dtype=np.float64)
    cam_rot = np.asarray(env.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    local = (points - cam_pos) @ cam_rot
    depth = -local[:, 2]

    sensor = np.asarray(env.model.cam_sensorsize[cam_id], dtype=np.float64)
    focal = np.asarray(env.model.cam_intrinsic[cam_id][:2], dtype=np.float64)
    if np.any(sensor <= 0) or np.any(focal <= 0):
        fovy = np.deg2rad(float(env.model.cam_fovy[cam_id]))
        sensor = np.array([2.0 * np.tan(fovy / 2.0), 2.0 * np.tan(fovy / 2.0)])
        focal = np.array([1.0, 1.0])

    xy = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    valid = depth > 1e-6
    xy[valid, 0] = (focal[0] * local[valid, 0] / depth[valid]) / (sensor[0] / 2.0)
    xy[valid, 1] = (focal[1] * local[valid, 1] / depth[valid]) / (sensor[1] / 2.0)
    return xy


def _floor_points(env, max_points: int = 120) -> np.ndarray:
    maze = getattr(env, "maze", None)
    if maze is None:
        return np.empty((0, 3), dtype=np.float64)

    points = []
    rows, cols = maze.map.shape
    for r in range(rows):
        for c in range(cols):
            xy = maze.rowcol_to_xy(np.array([r, c]))
            points.append([xy[0], xy[1], 0.0])
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = [points[i] for i in idx]
    return np.asarray(points, dtype=np.float64)


def _visible_speeds(prev_xy: np.ndarray, xy: np.ndarray, dt: float) -> np.ndarray:
    valid = np.isfinite(prev_xy).all(axis=1) & np.isfinite(xy).all(axis=1)
    valid &= (np.abs(prev_xy) <= 1.25).all(axis=1) & (np.abs(xy) <= 1.25).all(axis=1)
    if not np.any(valid):
        return np.empty((0,), dtype=np.float64)
    return np.linalg.norm(xy[valid] - prev_xy[valid], axis=1) / dt


def _bimodality_metrics(values: np.ndarray) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    values = values[values > 1e-9]
    if values.size < 20:
        return {"n": int(values.size), "status": "too_few_samples"}

    logv = np.log10(values)
    hist, edges = np.histogram(logv, bins=32)
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] > 0:
            peaks.append(i)
    if not peaks and hist.max() > 0:
        peaks = [int(hist.argmax())]
    peaks = sorted(peaks, key=lambda i: hist[i], reverse=True)[:2]
    peaks = sorted(peaks)

    result: dict[str, Any] = {
        "n": int(values.size),
        "log10_p10": float(np.percentile(logv, 10)),
        "log10_p50": float(np.percentile(logv, 50)),
        "log10_p90": float(np.percentile(logv, 90)),
        "num_histogram_peaks": len(peaks),
    }

    if len(peaks) == 2:
        lo, hi = peaks
        valley = int(lo + np.argmin(hist[lo : hi + 1]))
        peak_min = max(1, min(hist[lo], hist[hi]))
        result.update(
            {
                "peak_log10_centers": [
                    float((edges[p] + edges[p + 1]) / 2.0) for p in (lo, hi)
                ],
                "valley_log10_center": float((edges[valley] + edges[valley + 1]) / 2.0),
                "valley_to_smaller_peak": float(hist[valley] / peak_min),
                "bimodal_candidate": bool(hist[valley] / peak_min < 0.6),
            }
        )
    else:
        result["bimodal_candidate"] = False

    return result


def _episode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    path = float(np.nansum([r["agent_step_dist"] for r in rows]))
    speeds = np.asarray([r["agent_speed"] for r in rows], dtype=np.float64)
    goal_d = np.asarray([r["goal_distance"] for r in rows], dtype=np.float64)
    adv_d = np.asarray([r["adversary_distance"] for r in rows], dtype=np.float64)
    rewards = np.asarray([r["reward"] for r in rows], dtype=np.float64)
    return {
        "steps": len(rows),
        "reward": float(np.sum(rewards)),
        "path_length": path,
        "mean_speed": float(np.nanmean(speeds)),
        "p95_speed": float(np.nanpercentile(speeds, 95)),
        "goal_captures": int(np.sum([r["goal_respawned"] for r in rows])),
        "adversary_hits": int(np.sum([r["adversary_respawned"] for r in rows])),
        "goal_distance_start": float(goal_d[0]),
        "goal_distance_end": float(goal_d[-1]),
        "goal_distance_min": float(np.nanmin(goal_d)),
        "adversary_distance_min": float(np.nanmin(adv_d)),
        "mean_forward_action": float(np.nanmean([r["action_0"] for r in rows])),
    }


def _column_metrics(
    rows: list[dict[str, Any]], suffix: str, unit_suffix: str
) -> dict[str, Any]:
    metrics = {}
    if not rows:
        return metrics

    columns = sorted(key for key in rows[0] if key.endswith(suffix))
    for column in columns:
        values = np.asarray([row[column] for row in rows], dtype=np.float64)
        metrics[column[: -len(suffix)]] = {
            f"mean_{unit_suffix}": float(np.rad2deg(np.nanmean(values))),
            f"std_{unit_suffix}": float(np.rad2deg(np.nanstd(values))),
            f"min_{unit_suffix}": float(np.rad2deg(np.nanmin(values))),
            f"max_{unit_suffix}": float(np.rad2deg(np.nanmax(values))),
            f"mean_abs_{unit_suffix}": float(np.rad2deg(np.nanmean(np.abs(values)))),
            f"p95_abs_{unit_suffix}": float(
                np.rad2deg(np.nanpercentile(np.abs(values), 95))
            ),
        }
    return metrics


def run_diagnostics(
    run_dir: Path,
    outdir: Path,
    *,
    episodes: int,
    checkpoint: str,
    deterministic: bool,
    seed: int | None,
) -> dict[str, Any]:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    eval_config = _to_config(instantiate(cfg.eval_env))
    wrappers = [instantiate(w) for w in cfg.trainer.wrappers.values() if w]
    env = make_wrapped_env(config=eval_config.copy(), wrappers=wrappers, seed=seed)()
    cambrian_env = env.unwrapped

    model = instantiate(cfg.trainer.model, env=env)
    if not hasattr(model, "predict"):
        model = model()
    if checkpoint == "best" and (run_dir / "best_model.zip").exists():
        model = MjCambrianModel.load(run_dir / "best_model", env=env)
    else:
        model.load_policy(run_dir)

    agent_name = next(name for name, agent in cambrian_env.agents.items() if agent.trainable)
    camera_ids = _camera_names(cambrian_env.model, f"{agent_name}_eye")
    eye_joint_ids = _joint_names(cambrian_env.model, f"{agent_name}_eye")
    dt = float(cambrian_env.model.opt.timestep * cambrian_env._config.frame_skip)

    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    episodes_out: list[dict[str, Any]] = []
    goal_speeds: list[float] = []
    floor_speeds: list[float] = []

    obs, _ = env.reset(seed=seed)
    floor = _floor_points(cambrian_env)
    prev_agent_pos = cambrian_env.agents[agent_name].pos.copy()
    prev_goal_proj: dict[tuple[str, str], np.ndarray] = {}
    prev_floor_proj: dict[str, np.ndarray] = {}
    episode_rows: list[dict[str, Any]] = []
    episode = 0

    while episode < episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        action_vec = _as_action_vector(action)
        obs, reward, terminated, truncated, info = env.step(action)

        agent = cambrian_env.agents[agent_name]
        agent_pos = agent.pos.copy()
        step_dist = float(np.linalg.norm(agent_pos[:2] - prev_agent_pos[:2]))
        speed = step_dist / dt
        prev_agent_pos = agent_pos.copy()

        goal = cambrian_env.agents.get("goal0")
        adversary = cambrian_env.agents.get("adversary0")
        goal_pos = goal.pos.copy() if goal else np.full(3, np.nan)
        adv_pos = adversary.pos.copy() if adversary else np.full(3, np.nan)

        goal_respawned = bool(cambrian_env._info.get("goal0", {}).get("respawned", False))
        adv_respawned = bool(
            cambrian_env._info.get("adversary0", {}).get("respawned", False)
        )

        step_goal_speeds = []
        step_floor_speeds = []
        for cam_id, cam_name in camera_ids:
            if goal is not None:
                goal_proj = _project_points(cambrian_env, cam_id, goal_pos[None, :])
                key = (cam_name, "goal")
                if key in prev_goal_proj:
                    s = _visible_speeds(prev_goal_proj[key], goal_proj, dt)
                    step_goal_speeds.extend(s.tolist())
                    goal_speeds.extend(s.tolist())
                prev_goal_proj[key] = goal_proj

            if floor.size:
                floor_proj = _project_points(cambrian_env, cam_id, floor)
                if cam_name in prev_floor_proj:
                    s = _visible_speeds(prev_floor_proj[cam_name], floor_proj, dt)
                    step_floor_speeds.extend(s.tolist())
                    floor_speeds.extend(s.tolist())
                prev_floor_proj[cam_name] = floor_proj

        row = {
            "episode": episode,
            "step": int(cambrian_env.episode_step),
            "reward": float(np.asarray(reward).reshape(-1)[0]),
            "agent_x": float(agent_pos[0]),
            "agent_y": float(agent_pos[1]),
            "agent_step_dist": step_dist,
            "agent_speed": speed,
            "goal_distance": float(np.linalg.norm(agent_pos[:2] - goal_pos[:2])),
            "adversary_distance": float(np.linalg.norm(agent_pos[:2] - adv_pos[:2])),
            "goal_respawned": int(goal_respawned),
            "adversary_respawned": int(adv_respawned),
            "goal_sensor_speed_mean": float(np.mean(step_goal_speeds))
            if step_goal_speeds
            else float("nan"),
            "floor_sensor_speed_median": float(np.median(step_floor_speeds))
            if step_floor_speeds
            else float("nan"),
            "floor_sensor_speed_p90": float(np.percentile(step_floor_speeds, 90))
            if step_floor_speeds
            else float("nan"),
        }
        for i, value in enumerate(action_vec):
            row[f"action_{i}"] = float(value)
        for joint_id, joint_name in eye_joint_ids:
            qpos_adr = cambrian_env.model.jnt_qposadr[joint_id]
            qvel_adr = cambrian_env.model.jnt_dofadr[joint_id]
            row[f"{joint_name}_qpos"] = float(cambrian_env.data.qpos[qpos_adr])
            row[f"{joint_name}_qvel"] = float(cambrian_env.data.qvel[qvel_adr])
        rows.append(row)
        episode_rows.append(row)

        done = _as_scalar_done(terminated) or _as_scalar_done(truncated)
        if done:
            episodes_out.append(_episode_summary(episode_rows))
            episode += 1
            if episode >= episodes:
                break
            obs, _ = env.reset()
            prev_agent_pos = cambrian_env.agents[agent_name].pos.copy()
            prev_goal_proj.clear()
            prev_floor_proj.clear()
            episode_rows = []

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (outdir / "per_step.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "deterministic": deterministic,
        "episodes": episodes_out,
        "aggregate": {
            "episodes": len(episodes_out),
            "mean_reward": float(np.mean([e["reward"] for e in episodes_out])),
            "mean_path_length": float(np.mean([e["path_length"] for e in episodes_out])),
            "mean_speed": float(np.mean([e["mean_speed"] for e in episodes_out])),
            "goal_captures": int(sum(e["goal_captures"] for e in episodes_out)),
            "adversary_hits": int(sum(e["adversary_hits"] for e in episodes_out)),
            "mean_goal_distance_min": float(
                np.mean([e["goal_distance_min"] for e in episodes_out])
            ),
            "mean_goal_distance_delta": float(
                np.mean(
                    [
                        e["goal_distance_end"] - e["goal_distance_start"]
                        for e in episodes_out
                    ]
                )
            ),
        },
        "retinal_velocity": {
            "goal": _bimodality_metrics(np.asarray(goal_speeds, dtype=np.float64)),
            "floor": _bimodality_metrics(np.asarray(floor_speeds, dtype=np.float64)),
        },
        "eye_joints": {
            "qpos": _column_metrics(rows, "_qpos", "deg"),
            "qvel": _column_metrics(rows, "_qvel", "deg_s"),
        },
    }
    with (outdir / "behavior_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    print(json.dumps(summary["retinal_velocity"], indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outdir = args.outdir or args.run_dir / "diagnostics" / args.checkpoint
    run_diagnostics(
        args.run_dir,
        outdir,
        episodes=args.episodes,
        checkpoint=args.checkpoint,
        deterministic=not args.stochastic,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
