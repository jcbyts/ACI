#!/usr/bin/env python3
"""Analyze oculomotor dynamics from an r2dreamer checkpoint rollout.

This rolls out a saved policy in the ACI task, records physical eye pan/tilt joint
positions, and writes plots/metrics for fixation-versus-saccade-like structure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.render_r2dreamer_frozen2_rollout import load_agent  # noqa: E402
from scripts.train_r2dreamer_frozen2_binocular import (  # noqa: E402
    ACIDreamerEnv,
    compose_aci_config,
    stack_obs_for_agent,
)


def force_config_device(node: Any, device: torch.device) -> None:
    if isinstance(node, DictConfig):
        for key in list(node.keys()):
            if key == "device":
                node[key] = str(device)
            else:
                force_config_device(node[key], device)
    elif isinstance(node, ListConfig):
        for item in node:
            force_config_device(item, device)


def load_aci_config(run_dir: Path) -> Any:
    overrides_path = run_dir / "aci_overrides.txt"
    overrides = (
        [line.strip() for line in overrides_path.read_text().splitlines() if line.strip()]
        if overrides_path.exists()
        else []
    )
    return compose_aci_config(overrides)


def eye_joint_values(env: ACIDreamerEnv) -> tuple[list[str], np.ndarray, np.ndarray]:
    cambrian_env = env.cambrian_env
    agent = cambrian_env.agents["agent"]
    names: list[str] = []
    qpos: list[float] = []
    qvel: list[float] = []
    for actuator in getattr(agent, "_eye_actuators", []):
        joint_id = int(actuator.trnadr)
        name = cambrian_env.model.joint(joint_id).name or f"joint_{joint_id}"
        qpos_adr = int(cambrian_env.model.jnt_qposadr[joint_id])
        qvel_adr = int(cambrian_env.model.jnt_dofadr[joint_id])
        names.append(name)
        qpos.append(float(np.rad2deg(cambrian_env.data.qpos[qpos_adr])))
        qvel.append(float(np.rad2deg(cambrian_env.data.qvel[qvel_adr])))
    return names, np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)


def collect_rollouts(
    *,
    run_dir: Path,
    r2dreamer_root: Path,
    checkpoint: Path,
    episodes: int,
    seed: int,
    max_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    r2_config = OmegaConf.load(run_dir / "r2dreamer_config.yaml")
    force_config_device(r2_config, device)
    aci_config = load_aci_config(run_dir)
    env = ACIDreamerEnv(aci_config, eval_env=True, seed=seed, name="r2dreamer_gaze_eval")
    agent = load_agent(
        r2dreamer_root=r2dreamer_root,
        r2_config=r2_config,
        observation_space=env.observation_space,
        action_space=env.action_space,
        checkpoint=checkpoint,
        device=device,
    )

    episode_records: list[dict[str, Any]] = []
    joint_names: list[str] = []
    try:
        for episode in range(episodes):
            obs = env.reset()
            state = agent.get_initial_state(1)
            done = False
            qpos_rows: list[np.ndarray] = []
            qvel_rows: list[np.ndarray] = []
            action_rows: list[np.ndarray] = []
            capture_rows: list[bool] = []
            contact_rows: list[bool] = []
            distance_rows: list[float] = []

            names, qpos, qvel = eye_joint_values(env)
            if names:
                joint_names = names
            qpos_rows.append(qpos)
            qvel_rows.append(qvel)

            while not done and len(env.last_samples) < max_steps:
                obs_td = stack_obs_for_agent(obs, device)
                with torch.no_grad():
                    action, state = agent.act(obs_td, state, eval=True)
                action_np = action.detach().cpu().numpy()[0].astype(np.float64)
                obs, _, done, _ = env.step(action_np)
                names, qpos, qvel = eye_joint_values(env)
                if names:
                    joint_names = names
                qpos_rows.append(qpos)
                qvel_rows.append(qvel)
                action_rows.append(action_np)

                sample = env.last_samples[-1]
                capture_rows.append(bool(sample["goal_respawned"]))
                contact_rows.append(bool(sample["has_contacts"]))
                ax, ay = sample["agent_xy"]
                gx, gy = sample["goal_xy"]
                distance_rows.append(float(np.hypot(ax - gx, ay - gy)))

            episode_records.append(
                {
                    "qpos_deg": np.asarray(qpos_rows, dtype=np.float64),
                    "qvel_deg_s": np.asarray(qvel_rows, dtype=np.float64),
                    "actions": np.asarray(action_rows, dtype=np.float64),
                    "captures": np.asarray(capture_rows, dtype=bool),
                    "contacts": np.asarray(contact_rows, dtype=bool),
                    "distance": np.asarray(distance_rows, dtype=np.float64),
                }
            )
    finally:
        env.close()

    return {"joint_names": joint_names, "episodes": episode_records}


def eye_speed_deg_step(qpos_deg: np.ndarray) -> np.ndarray:
    if qpos_deg.shape[0] < 2 or qpos_deg.shape[1] < 2:
        return np.zeros((0, 0), dtype=np.float64)
    deltas = np.diff(qpos_deg, axis=0)
    # Joint order follows physical actuator order: left pan/tilt, right pan/tilt.
    eye_count = deltas.shape[1] // 2
    speeds = []
    for eye_idx in range(eye_count):
        pan = deltas[:, 2 * eye_idx]
        tilt = deltas[:, 2 * eye_idx + 1]
        speeds.append(np.hypot(pan, tilt))
    return np.stack(speeds, axis=1) if speeds else np.zeros((deltas.shape[0], 0))


def bimodality_coefficient(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    if values.size < 4:
        return None
    centered = values - values.mean()
    std = values.std()
    if std <= 1e-12:
        return 0.0
    skew = float(np.mean((centered / std) ** 3))
    kurt = float(np.mean((centered / std) ** 4))
    if kurt <= 1e-12:
        return None
    return float((skew * skew + 1.0) / kurt)


def summarize(records: dict[str, Any]) -> dict[str, Any]:
    all_speeds = []
    all_qvel = []
    captures = []
    contacts = []
    for ep in records["episodes"]:
        speeds = eye_speed_deg_step(ep["qpos_deg"])
        if speeds.size:
            all_speeds.append(speeds.reshape(-1))
        if ep["qvel_deg_s"].size:
            # Per-eye angular velocity magnitude in deg/s.
            qvel = ep["qvel_deg_s"]
            eye_count = qvel.shape[1] // 2
            mags = [np.hypot(qvel[:, 2 * i], qvel[:, 2 * i + 1]) for i in range(eye_count)]
            all_qvel.append(np.stack(mags, axis=1).reshape(-1))
        captures.append(int(ep["captures"].sum()))
        contacts.append(float(ep["contacts"].mean()) if ep["contacts"].size else 0.0)

    speed = np.concatenate(all_speeds) if all_speeds else np.asarray([], dtype=np.float64)
    qvel = np.concatenate(all_qvel) if all_qvel else np.asarray([], dtype=np.float64)
    positive = speed[speed > 1e-9]
    fixation_threshold = float(max(0.05, np.percentile(positive, 25))) if positive.size else 0.05
    jump_threshold = float(np.percentile(positive, 90)) if positive.size else 0.0

    return {
        "episodes": len(records["episodes"]),
        "joint_names": records["joint_names"],
        "mean_captures": float(np.mean(captures)) if captures else 0.0,
        "mean_contact_fraction": float(np.mean(contacts)) if contacts else 0.0,
        "gaze_speed_deg_step": {
            "count": int(speed.size),
            "mean": float(np.mean(speed)) if speed.size else 0.0,
            "median": float(np.median(speed)) if speed.size else 0.0,
            "p25": float(np.percentile(speed, 25)) if speed.size else 0.0,
            "p75": float(np.percentile(speed, 75)) if speed.size else 0.0,
            "p90": float(np.percentile(speed, 90)) if speed.size else 0.0,
            "p99": float(np.percentile(speed, 99)) if speed.size else 0.0,
            "fixation_threshold": fixation_threshold,
            "jump_threshold": jump_threshold,
            "low_speed_fraction": float(np.mean(speed <= fixation_threshold)) if speed.size else 0.0,
            "high_speed_fraction": float(np.mean(speed >= jump_threshold)) if speed.size and jump_threshold > 0 else 0.0,
            "bimodality_coefficient_log_speed": bimodality_coefficient(np.log10(speed + 1e-4)),
        },
        "gaze_velocity_deg_s": {
            "mean": float(np.mean(qvel)) if qvel.size else 0.0,
            "median": float(np.median(qvel)) if qvel.size else 0.0,
            "p90": float(np.percentile(qvel, 90)) if qvel.size else 0.0,
            "p99": float(np.percentile(qvel, 99)) if qvel.size else 0.0,
        },
    }


def save_tables(records: dict[str, Any], out_prefix: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for idx, ep in enumerate(records["episodes"]):
        arrays[f"episode_{idx}_qpos_deg"] = ep["qpos_deg"]
        arrays[f"episode_{idx}_qvel_deg_s"] = ep["qvel_deg_s"]
        arrays[f"episode_{idx}_actions"] = ep["actions"]
        arrays[f"episode_{idx}_captures"] = ep["captures"]
        arrays[f"episode_{idx}_contacts"] = ep["contacts"]
        arrays[f"episode_{idx}_distance"] = ep["distance"]
        arrays[f"episode_{idx}_eye_speed_deg_step"] = eye_speed_deg_step(ep["qpos_deg"])
    np.savez_compressed(out_prefix.with_suffix(".npz"), **arrays)


def plot_diagnostics(records: dict[str, Any], summary: dict[str, Any], out_prefix: Path) -> None:
    first = records["episodes"][0]
    qpos = first["qpos_deg"]
    speeds = eye_speed_deg_step(qpos)
    all_speeds = np.concatenate([
        eye_speed_deg_step(ep["qpos_deg"]).reshape(-1)
        for ep in records["episodes"]
        if eye_speed_deg_step(ep["qpos_deg"]).size
    ])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    t = np.arange(qpos.shape[0])
    labels = records["joint_names"] or [f"eye_joint_{i}" for i in range(qpos.shape[1])]
    for idx in range(qpos.shape[1]):
        axes[0].plot(t, qpos[:, idx], linewidth=1.0, label=labels[idx] if idx < len(labels) else f"joint_{idx}")
    axes[0].set_title("Eye pan/tilt position, first rollout")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("angle (deg)")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)

    for eye_idx in range(speeds.shape[1]):
        axes[1].plot(np.arange(speeds.shape[0]), speeds[:, eye_idx], linewidth=1.0, label=f"eye {eye_idx}")
    axes[1].axhline(summary["gaze_speed_deg_step"]["fixation_threshold"], color="tab:green", linestyle="--", linewidth=1, label="fixation threshold")
    if summary["gaze_speed_deg_step"]["jump_threshold"] > 0:
        axes[1].axhline(summary["gaze_speed_deg_step"]["jump_threshold"], color="tab:red", linestyle="--", linewidth=1, label="jump threshold")
    axes[1].set_title("Gaze speed, first rollout")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("deg / step")
    axes[1].legend(loc="upper right", fontsize=8)

    if all_speeds.size:
        axes[2].hist(np.log10(all_speeds + 1e-4), bins=80, color="0.25")
    axes[2].set_title("Log gaze speed distribution across rollouts")
    axes[2].set_xlabel("log10(deg / step + 1e-4)")
    axes[2].set_ylabel("count")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_diagnostics.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if all_speeds.size:
        ax.hist(all_speeds, bins=80, color="0.3")
    ax.set_title("Gaze speed distribution")
    ax.set_xlabel("deg / step")
    ax.set_ylabel("count")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_speed_hist.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--r2dreamer-root", type=Path, default=Path("/tmp/r2dreamer"))
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = args.checkpoint or args.run_dir / "latest.pt"
    outdir = args.outdir or args.run_dir / "gaze_analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{checkpoint.stem}_seed_{args.seed}"
    out_prefix = outdir / stem

    records = collect_rollouts(
        run_dir=args.run_dir,
        r2dreamer_root=args.r2dreamer_root,
        checkpoint=checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        device=torch.device(args.device),
    )
    metrics = summarize(records)
    metrics.update({"checkpoint": str(checkpoint), "seed": args.seed, "max_steps": args.max_steps})
    out_prefix.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_tables(records, out_prefix)
    plot_diagnostics(records, metrics, out_prefix)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
