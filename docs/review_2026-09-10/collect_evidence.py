"""Rebuild the local experiment inventory and descriptive learning curves.

Run from any directory with the ACI environment. Reads saved data without importing
Cambrian or loading executable checkpoint objects. Outputs beside this script.
"""

from pathlib import Path
import csv
import re
import subprocess

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


rows = []
for run in sorted((ROOT / "logs").glob("*/*")):
    if not run.is_dir():
        continue
    cfg = (
        yaml.safe_load((run / "config.yaml").read_text())
        if (run / "config.yaml").exists()
        else {}
    )
    trainer = cfg.get("trainer", {})
    agent = cfg.get("env", {}).get("agents", {}).get("agent", {})
    row = dict(
        run=str(run.relative_to(ROOT)),
        seed=cfg.get("seed"),
        finished=(run / "finished").exists(),
        config=(run / "config.yaml").exists(),
        checkpoint=(run / "best_model.zip").exists(),
        final_policy=(run / "policy.pt").exists(),
        requested_steps=trainer.get("total_timesteps"),
        model=trainer.get("model", {}).get("_target_"),
        body=agent.get("instance", {}).get("_target_"),
        train_rewards=";".join(
            k
            for k, v in cfg.get("env", {}).get("reward_fn", {}).items()
            if isinstance(v, dict) and not v.get("disable")
        ),
        fitness_file=(
            (run / "train_fitness.txt").read_text().strip()
            if (run / "train_fitness.txt").exists()
            else ""
        ),
    )
    npz = run / "evaluations.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as data:
            means = data["results"].mean(axis=1)
            best = int(means.argmax())
            row.update(
                eval_blocks=len(means),
                episodes_per_block=data["results"].shape[1],
                first_eval_mean=float(means[0]),
                best_eval_mean=float(means[best]),
                last_eval_mean=float(means[-1]),
                best_eval_step=int(data["timesteps"][best]),
                last_eval_step=int(data["timesteps"][-1]),
                best_video=f"evaluations/vis_{best}.mp4",
            )
    monitor = run / "eval_monitor.csv"
    if monitor.exists():
        records = list(
            csv.DictReader(
                line
                for line in monitor.read_text().splitlines()
                if not line.startswith("#")
            )
        )
        if records:
            row["hours_to_last_eval"] = float(records[-1]["t"]) / 3600
    rows.append(row)
write_csv("runs.csv", rows)

launches = []
for path in sorted((ROOT / "launch_logs").glob("*.log")):
    content = path.read_text(errors="replace")
    errors = sorted(
        set(re.findall(r"\b(?:[A-Za-z]+Error|[A-Za-z]+Exception):[^\n]*", content))
    )
    launches.append(
        dict(
            log=str(path.relative_to(ROOT)),
            bytes=path.stat().st_size,
            run_directory_present=any(
                Path(row["run"]).name == path.stem for row in rows
            ),
            recorded_errors=" | ".join(errors)[:3000],
        )
    )
write_csv("launches.csv", launches)

historical = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", "e62a749", "scripts", "docs"],
    cwd=ROOT,
    text=True,
)
(OUT / "historical_files.txt").write_text(historical)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for ax, mode in zip(axes, ["fixed", "actuated"]):
    for seed, color in [(0, "#006b8f"), (100, "#b85000")]:
        for family, pattern, style in [
            ("R(2+1)D + LSTM", f"{mode}-2eye-rppo_cyclopean_lstm_seed{seed}_*", "-"),
            (
                "R(2+1)D feed-forward",
                f"{mode}-2eye-mlp_r2plus1d_yawrate_seed{seed}_*",
                "--",
            ),
        ]:
            for p in (ROOT / "logs").glob("*/" + pattern + "/evaluations.npz"):
                with np.load(p, allow_pickle=False) as d:
                    ax.plot(
                        d["timesteps"] / 1000,
                        d["results"].mean(axis=1),
                        style,
                        color=color,
                        linewidth=1.5,
                        label=f"{family}, seed {seed}",
                    )
    ax.set_title(f"{mode.capitalize()} binocular eyes")
    ax.set_xlabel("Environment steps (thousands)")
    ax.grid(alpha=0.2)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylim(-80, 85)
axes[0].set_ylabel("Mean evaluation return (6 episodes)")
axes[1].legend(fontsize=7, loc="lower right")
fig.suptitle(
    "Saved June 23–24 runs: promising best policies, unstable late performance"
)
fig.text(
    0.5,
    0.01,
    "Descriptive, not an isolated memory ablation. Early feed-forward returns below −80 are clipped. No smoothing.",
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=[0, 0.04, 1, 0.94])
fig.savefig(OUT / "learning_curves.png", dpi=160)
plt.close(fig)
print(
    f"Wrote {len(rows)} run rows, {len(launches)} launch rows, historical file list, and learning curves to {OUT}"
)
