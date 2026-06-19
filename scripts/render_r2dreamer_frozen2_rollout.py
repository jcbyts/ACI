#!/usr/bin/env python3
"""Render arena + eye-overlay rollout videos for a trained r2dreamer policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_r2dreamer_frozen2_binocular import (  # noqa: E402
    ACIDreamerEnv,
    compose_aci_config,
    stack_obs_for_agent,
)


def frame_to_uint8(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    # Cambrian's MuJoCo renderer uses OpenGL coordinates internally. The built-in
    # recorder flips on save, so do the same for manually captured frames.
    return np.flip(arr, axis=0)


def load_agent(
    *,
    r2dreamer_root: Path,
    r2_config: Any,
    observation_space: Any,
    action_space: Any,
    checkpoint: Path,
    device: torch.device,
) -> Any:
    sys.path.insert(0, str(r2dreamer_root))
    from dreamer import Dreamer

    agent = Dreamer(r2_config.model, observation_space, action_space).to(device)
    payload = torch.load(checkpoint, map_location=device)
    agent.load_state_dict(payload["agent_state_dict"])
    agent.eval()
    return agent


def render_rollout(
    *,
    run_dir: Path,
    r2dreamer_root: Path,
    checkpoint: Path,
    output: Path,
    seed: int,
    max_steps: int,
    fps: int,
    device: torch.device,
) -> dict[str, Any]:
    r2_config = OmegaConf.load(run_dir / "r2dreamer_config.yaml")
    overrides_path = run_dir / "aci_overrides.txt"
    aci_overrides = (
        [line.strip() for line in overrides_path.read_text().splitlines() if line.strip()]
        if overrides_path.exists()
        else []
    )
    aci_config = compose_aci_config(aci_overrides)

    env = ACIDreamerEnv(aci_config, eval_env=True, seed=seed, name="r2dreamer_render")
    agent = load_agent(
        r2dreamer_root=r2dreamer_root,
        r2_config=r2_config,
        observation_space=env.observation_space,
        action_space=env.action_space,
        checkpoint=checkpoint,
        device=device,
    )

    frames: list[np.ndarray] = []
    obs = env.reset()
    state = agent.get_initial_state(1)
    done = False
    frames.append(frame_to_uint8(env._env.render()))

    try:
        while not done and len(env.last_samples) < max_steps:
            obs_td = stack_obs_for_agent(obs, device)
            with torch.no_grad():
                action, state = agent.act(obs_td, state, eval=True)
            obs, _, done, _ = env.step(action.detach().cpu().numpy()[0])
            frames.append(frame_to_uint8(env._env.render()))
    finally:
        env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=fps, macro_block_size=1)

    return {
        "output": str(output),
        "frames": len(frames),
        "steps": len(env.last_samples),
        "captures": int(sum(1 for sample in env.last_samples if sample["goal_respawned"])),
        "contact_fraction": float(np.mean([sample["has_contacts"] for sample in env.last_samples]))
        if env.last_samples
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--r2dreamer-root", type=Path, default=Path("/tmp/r2dreamer"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = args.checkpoint or args.run_dir / "latest.pt"
    output = args.output or args.run_dir / "rollout.mp4"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. "
            "The current r2dreamer runner writes latest.pt after training finishes."
        )

    summary = render_rollout(
        run_dir=args.run_dir,
        r2dreamer_root=args.r2dreamer_root,
        checkpoint=checkpoint,
        output=output,
        seed=args.seed,
        max_steps=args.max_steps,
        fps=args.fps,
        device=torch.device(args.device),
    )
    print(summary)


if __name__ == "__main__":
    main()
