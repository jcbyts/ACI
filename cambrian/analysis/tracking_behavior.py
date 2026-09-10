"""Posthoc tracking-policy behavior analysis.

This module runs deterministic evaluation episodes from a saved checkpoint and writes
CSV/JSON summaries plus gaze/body/eye movement plots.  It is intentionally separate
from the training reward so scientific analyses can compare task success, adversary
confusion, body scanning, eye movement, and saccade/fixate statistics directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from hydra_config import run_hydra

from cambrian import MjCambrianConfig, MjCambrianTrainer
from cambrian.agents import MjCambrianAgent
from cambrian.envs.env import MjCambrianEnv
from cambrian.utils.wrappers import make_wrapped_env


_EPS = 1e-8


def _angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def _unwrap(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return np.unwrap(values, axis=0)


def _as_1d(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _flatten_physical_eyes(agent: MjCambrianAgent) -> list[Any]:
    eyes = []
    for eye in agent.eyes.values():
        if hasattr(eye, "eyes"):
            eyes.extend(list(eye.eyes.values()))
        else:
            eyes.append(eye)
    return sorted(eyes, key=lambda eye: eye.name)


def _eye_base_gaze_radians(agent: MjCambrianAgent) -> tuple[np.ndarray, np.ndarray]:
    """Return base gaze longitude and horizontal FOV for each physical eye."""
    base_lon, fov_x = [], []
    for eye in _flatten_physical_eyes(agent):
        gaze = (
            eye.config.gaze_coord
            if eye.config.gaze_coord is not None
            else eye.config.coord
        )
        base_lon.append(np.deg2rad(float(gaze[1])))
        fov_x.append(np.deg2rad(float(eye.config.fov[1])))
    return np.asarray(base_lon, dtype=np.float64), np.asarray(fov_x, dtype=np.float64)


def _eye_joint_state_raw(agent: MjCambrianAgent) -> tuple[np.ndarray, np.ndarray]:
    """Return physical eye joint positions/velocities in radians and radians/s."""
    actuators = getattr(agent, "_eye_actuators", [])
    qpos, qvel = [], []
    for actuator in actuators:
        joint_id = int(actuator.trnadr)
        qpos_adr = int(agent._spec.model.jnt_qposadr[joint_id])
        qvel_adr = int(agent._spec.model.jnt_dofadr[joint_id])
        qpos.append(float(agent._spec.data.qpos[qpos_adr]))
        qvel.append(float(agent._spec.data.qvel[qvel_adr]))
    return np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)


def _pan_tilt_from_joint_state(
    agent: MjCambrianAgent, n_eyes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qpos, qvel = _eye_joint_state_raw(agent)
    if qpos.size == 0:
        return (
            np.zeros(n_eyes, dtype=np.float64),
            np.zeros(n_eyes, dtype=np.float64),
            np.zeros(n_eyes, dtype=np.float64),
            np.zeros(n_eyes, dtype=np.float64),
        )
    # Eye actuators are generated as pan, tilt for each physical eye.
    pan = qpos[0::2][:n_eyes]
    tilt = qpos[1::2][:n_eyes]
    pan_vel = qvel[0::2][:n_eyes]
    tilt_vel = qvel[1::2][:n_eyes]
    return pan, tilt, pan_vel, tilt_vel


def _object_visibility(
    agent_pos: np.ndarray,
    object_pos: np.ndarray,
    gaze_angles: np.ndarray,
    fov_x: np.ndarray,
) -> bool:
    if gaze_angles.size == 0:
        return False
    bearing = float(
        np.arctan2(object_pos[1] - agent_pos[1], object_pos[0] - agent_pos[0])
    )
    err = np.abs(_angle_diff(bearing, gaze_angles))
    return bool(np.any(err <= 0.5 * fov_x))


def _bimodality_coefficient(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 4 or float(np.std(values)) <= _EPS:
        return float("nan")
    centered = values - float(np.mean(values))
    std = float(np.std(values))
    skew = float(np.mean((centered / std) ** 3))
    kurt = float(np.mean((centered / std) ** 4))
    if kurt <= _EPS:
        return float("nan")
    return float((skew**2 + 1.0) / kurt)


def _summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_{k}": float("nan")
            for k in ("mean", "std", "median", "p95", "max")
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_hist(
    path: Path, values: np.ndarray, *, title: str, xlabel: str, bins: int = 80
) -> None:
    values = values[np.isfinite(values)]
    plt.figure(figsize=(7, 4))
    if values.size:
        plt.hist(values, bins=bins, alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_cumulative(path: Path, step_rows: list[dict[str, Any]]) -> None:
    if not step_rows:
        return
    steps = np.arange(len(step_rows))
    body = np.cumsum([float(row.get("abs_body_yaw_delta", 0.0)) for row in step_rows])
    eyes = np.cumsum(
        [float(row.get("mean_abs_eye_pan_delta", 0.0)) for row in step_rows]
    )
    plt.figure(figsize=(7, 4))
    plt.plot(steps, np.rad2deg(body), label="body yaw")
    plt.plot(steps, np.rad2deg(eyes), label="mean eye pan")
    plt.xlabel("evaluation step")
    plt.ylabel("cumulative absolute rotation (deg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _make_single_env(config: MjCambrianConfig, seed: int):
    wrappers = [w for w in config.trainer.wrappers.values() if w]
    return make_wrapped_env(
        config=config.eval_env.copy(),
        name=config.expname,
        wrappers=wrappers,
        seed=seed,
    )()


def _run_analysis(
    config: MjCambrianConfig,
    *,
    episodes: int,
    output_dir: Path,
    deterministic: bool,
    max_steps: int | None,
) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a single wrapped Gymnasium env so we can inspect the underlying Cambrian
    # state before automatic VecEnv resets obscure terminal/respawn events.
    wrapped_env = _make_single_env(config, seed=config.seed)
    cambrian_env: MjCambrianEnv = wrapped_env.unwrapped
    trainer = MjCambrianTrainer(config)
    model = trainer._make_model(wrapped_env)

    agent = cambrian_env.agents["agent"]
    base_gaze, fov_x = _eye_base_gaze_radians(agent)
    n_eyes = len(base_gaze)
    dt = float(config.eval_env.frame_skip) * float(cambrian_env.model.opt.timestep)
    if dt <= 0:
        dt = 1.0

    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    for episode in range(int(episodes)):
        # The environment is seeded once at construction; subsequent episodes
        # advance that reproducible RNG stream.
        obs, _ = wrapped_env.reset()
        state = None
        episode_start = np.ones((1,), dtype=bool)
        episode_return = 0.0
        target_respawns = 0
        adversary_respawns = 0
        contact_steps = 0
        target_visible_steps = 0
        adversary_visible_steps = 0
        prev_body_yaw = None
        prev_gaze_angles = None
        body_yaw_abs_total = 0.0
        eye_pan_abs_total = 0.0
        max_episode_steps = max_steps or int(cambrian_env.max_episode_steps)

        for step in range(max_episode_steps):
            action, state = model.predict(
                obs,
                state=state,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            action_vec = _as_1d(action)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            done = bool(terminated or truncated)
            episode_start = np.asarray([done], dtype=bool)
            episode_return += float(reward)

            all_info = info.get("_all_agents", {}) if isinstance(info, dict) else {}
            target_event = int(bool(all_info.get("goal0", {}).get("respawned", False)))
            adversary_event = int(
                bool(all_info.get("adversary0", {}).get("respawned", False))
            )
            target_respawns += target_event
            adversary_respawns += adversary_event
            contact = (
                int(bool(info.get("has_contacts", False)))
                if isinstance(info, dict)
                else 0
            )
            contact_steps += contact

            agent_pos = cambrian_env.agents["agent"].pos.copy()
            goal_pos = (
                cambrian_env.agents.get("goal0").pos.copy()
                if "goal0" in cambrian_env.agents
                else np.full(3, np.nan)
            )
            adversary_pos = (
                cambrian_env.agents.get("adversary0").pos.copy()
                if "adversary0" in cambrian_env.agents
                else np.full(3, np.nan)
            )
            body_yaw = float(cambrian_env.agents["agent"].qpos[2])
            pan, tilt, pan_vel, tilt_vel = _pan_tilt_from_joint_state(agent, n_eyes)
            gaze_angles = body_yaw + base_gaze + pan

            body_yaw_delta = (
                0.0
                if prev_body_yaw is None
                else float(_angle_diff(body_yaw, prev_body_yaw))
            )
            if prev_gaze_angles is None:
                gaze_delta = np.zeros_like(gaze_angles)
            else:
                gaze_delta = _angle_diff(gaze_angles, prev_gaze_angles)
            prev_body_yaw = body_yaw
            prev_gaze_angles = gaze_angles.copy()
            body_yaw_abs_total += abs(body_yaw_delta)
            mean_abs_eye_pan_delta = (
                float(np.mean(np.abs(pan_vel) * dt)) if pan_vel.size else 0.0
            )
            eye_pan_abs_total += mean_abs_eye_pan_delta

            target_visible = _object_visibility(agent_pos, goal_pos, gaze_angles, fov_x)
            adversary_visible = _object_visibility(
                agent_pos, adversary_pos, gaze_angles, fov_x
            )
            target_visible_steps += int(target_visible)
            adversary_visible_steps += int(adversary_visible)

            row = {
                "episode": episode,
                "step": step,
                "reward": float(reward),
                "return_so_far": float(episode_return),
                "done": int(done),
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
                "target_respawn": target_event,
                "adversary_respawn": adversary_event,
                "contact": contact,
                "agent_x": float(agent_pos[0]),
                "agent_y": float(agent_pos[1]),
                "goal_distance": float(np.linalg.norm(agent_pos[:2] - goal_pos[:2])),
                "adversary_distance": float(
                    np.linalg.norm(agent_pos[:2] - adversary_pos[:2])
                ),
                "body_yaw": body_yaw,
                "body_yaw_delta": body_yaw_delta,
                "abs_body_yaw_delta": abs(body_yaw_delta),
                "body_yaw_velocity_deg_s": float(np.rad2deg(body_yaw_delta / dt)),
                "mean_abs_eye_pan_delta": mean_abs_eye_pan_delta,
                "mean_abs_eye_pan_velocity_deg_s": float(
                    np.rad2deg(np.mean(np.abs(pan_vel))) if pan_vel.size else 0.0
                ),
                "mean_abs_gaze_velocity_deg_s": float(
                    np.rad2deg(np.mean(np.abs(gaze_delta)) / dt)
                    if gaze_delta.size
                    else 0.0
                ),
                "target_visible": int(target_visible),
                "adversary_visible": int(adversary_visible),
                "action_speed": (
                    float(action_vec[0]) if action_vec.size > 0 else float("nan")
                ),
                "action_body_yaw": (
                    float(action_vec[1]) if action_vec.size > 1 else float("nan")
                ),
                "body_yaw_saturated": int(
                    action_vec.size > 1 and abs(action_vec[1]) > 0.95
                ),
                "eye_action_abs_mean": (
                    float(np.mean(np.abs(action_vec[2:])))
                    if action_vec.size > 2
                    else 0.0
                ),
                "eye_action_saturated": int(
                    action_vec.size > 2 and np.any(np.abs(action_vec[2:]) > 0.95)
                ),
            }
            for i, value in enumerate(gaze_angles):
                row[f"gaze_angle_{i}"] = float(value)
                row[f"gaze_velocity_deg_s_{i}"] = float(np.rad2deg(gaze_delta[i] / dt))
                row[f"eye_pan_{i}"] = float(pan[i]) if i < pan.size else 0.0
                row[f"eye_pan_velocity_deg_s_{i}"] = (
                    float(np.rad2deg(pan_vel[i])) if i < pan_vel.size else 0.0
                )
            step_rows.append(row)

            if done:
                break

        denom = max(step + 1, 1)
        eye_fraction = eye_pan_abs_total / (
            eye_pan_abs_total + body_yaw_abs_total + _EPS
        )
        episode_rows.append(
            {
                "episode": episode,
                "return": float(episode_return),
                "length": int(denom),
                "target_respawns": int(target_respawns),
                "adversary_respawns": int(adversary_respawns),
                "target_minus_adversary": int(target_respawns - adversary_respawns),
                "contact_steps": int(contact_steps),
                "target_visible_fraction": float(target_visible_steps / denom),
                "adversary_visible_fraction": float(adversary_visible_steps / denom),
                "body_yaw_abs_total_deg": float(np.rad2deg(body_yaw_abs_total)),
                "eye_pan_abs_total_deg": float(np.rad2deg(eye_pan_abs_total)),
                "eye_rotation_fraction": float(eye_fraction),
            }
        )

    _write_csv(output_dir / "behavior_steps.csv", step_rows)
    _write_csv(output_dir / "behavior_episodes.csv", episode_rows)

    returns = np.asarray([row["return"] for row in episode_rows], dtype=np.float64)
    lengths = np.asarray([row["length"] for row in episode_rows], dtype=np.float64)
    target_counts = np.asarray(
        [row["target_respawns"] for row in episode_rows], dtype=np.float64
    )
    adversary_counts = np.asarray(
        [row["adversary_respawns"] for row in episode_rows], dtype=np.float64
    )
    body_yaw_vel = np.asarray(
        [abs(row["body_yaw_velocity_deg_s"]) for row in step_rows], dtype=np.float64
    )
    eye_pan_vel = np.asarray(
        [row["mean_abs_eye_pan_velocity_deg_s"] for row in step_rows], dtype=np.float64
    )
    gaze_vel = np.asarray(
        [row["mean_abs_gaze_velocity_deg_s"] for row in step_rows], dtype=np.float64
    )
    eye_fraction = np.asarray(
        [row["eye_rotation_fraction"] for row in episode_rows], dtype=np.float64
    )

    summary: dict[str, Any] = {
        "n_episodes": int(len(episode_rows)),
        "dt_seconds": float(dt),
        "return_mean": float(np.mean(returns)) if returns.size else float("nan"),
        "return_std": float(np.std(returns)) if returns.size else float("nan"),
        "length_mean": float(np.mean(lengths)) if lengths.size else float("nan"),
        "target_respawns_mean": (
            float(np.mean(target_counts)) if target_counts.size else float("nan")
        ),
        "adversary_respawns_mean": (
            float(np.mean(adversary_counts)) if adversary_counts.size else float("nan")
        ),
        "target_adversary_ratio": float(
            np.sum(target_counts) / (np.sum(adversary_counts) + _EPS)
        ),
        "target_minus_adversary_mean": (
            float(np.mean(target_counts - adversary_counts))
            if target_counts.size
            else float("nan")
        ),
        "eye_rotation_fraction_mean": (
            float(np.mean(eye_fraction)) if eye_fraction.size else float("nan")
        ),
        "gaze_velocity_abs_bimodality_coefficient": _bimodality_coefficient(gaze_vel),
        "body_yaw_velocity_abs_bimodality_coefficient": _bimodality_coefficient(
            body_yaw_vel
        ),
        "eye_pan_velocity_abs_bimodality_coefficient": _bimodality_coefficient(
            eye_pan_vel
        ),
    }
    summary.update(_summarize(gaze_vel, "gaze_velocity_abs_deg_s"))
    summary.update(_summarize(body_yaw_vel, "body_yaw_velocity_abs_deg_s"))
    summary.update(_summarize(eye_pan_vel, "eye_pan_velocity_abs_deg_s"))

    with open(output_dir / "behavior_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    _plot_hist(
        output_dir / "gaze_velocity_hist.png",
        gaze_vel,
        title="Absolute gaze velocity",
        xlabel="mean absolute gaze velocity (deg/s)",
    )
    _plot_hist(
        output_dir / "body_yaw_velocity_hist.png",
        body_yaw_vel,
        title="Absolute body yaw velocity",
        xlabel="absolute body yaw velocity (deg/s)",
    )
    _plot_hist(
        output_dir / "eye_pan_velocity_hist.png",
        eye_pan_vel,
        title="Absolute eye pan velocity",
        xlabel="mean absolute eye pan velocity (deg/s)",
    )
    _plot_cumulative(output_dir / "cumulative_body_vs_eye_rotation.png", step_rows)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return float(summary["return_mean"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)

    def _main(
        config: MjCambrianConfig,
        *,
        episodes: int,
        output_dir: Path | None,
        stochastic: bool,
        max_steps: int | None,
    ) -> float:
        outdir = output_dir or (config.expdir / "behavior_analysis")
        return _run_analysis(
            config,
            episodes=episodes,
            output_dir=Path(outdir),
            deterministic=not stochastic,
            max_steps=max_steps,
        )

    run_hydra(_main, config_path="pkg://cambrian/configs", parser=parser)


if __name__ == "__main__":
    main()
