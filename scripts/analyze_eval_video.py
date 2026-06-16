#!/usr/bin/env python3
"""Summarize behavior from Cambrian eval MP4s.

This intentionally works from the rendered video because the question is behavioral:
is the policy moving through the arena, or parking/spinning near a wall?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    return float(xs.mean()), float(ys.mean())


def _detect_agent(frame: np.ndarray) -> tuple[float, float] | None:
    # Restrict to the top-down arena. This avoids the two retina panels and most text.
    top = frame[:330]
    r = top[..., 0].astype(np.int16)
    g = top[..., 1].astype(np.int16)
    b = top[..., 2].astype(np.int16)

    # The agent overlay is yellow/olive. The mask also catches a little trail, but the
    # centroid is stable enough for movement/parking diagnostics.
    mask = (r > 90) & (g > 90) & (b < 80) & ((r + g) > (2 * b + 120))

    # Ignore the left text overlay.
    mask[:, :90] = False
    return _centroid(mask)


def _detect_magenta(frame: np.ndarray) -> list[tuple[float, float]]:
    top = frame[:330]
    r = top[..., 0].astype(np.int16)
    g = top[..., 1].astype(np.int16)
    b = top[..., 2].astype(np.int16)
    mask = (r > 120) & (b > 110) & (g < 80)
    mask[:, :90] = False

    # Simple connected components without pulling in cv2/scipy.
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[tuple[float, float]] = []
    h, w = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack = [(int(x), int(y))]
        seen[y, x] = True
        xs: list[int] = []
        ys: list[int] = []
        while stack:
            cx, cy = stack.pop()
            xs.append(cx)
            ys.append(cy)
            for nx in range(max(0, cx - 1), min(w, cx + 2)):
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
        if len(xs) >= 10:
            comps.append((float(np.mean(xs)), float(np.mean(ys))))
    return comps


def analyze(path: Path, sample_fps: float = 5.0) -> dict:
    reader = imageio.get_reader(path)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 50.0))
    duration = float(meta.get("duration", 0.0))
    n_samples = max(1, int(duration * sample_fps))
    indices = [int(i * fps / sample_fps) for i in range(n_samples)]

    positions: list[tuple[float, float]] = []
    goal_dists: list[float] = []
    for idx in indices:
        try:
            frame = reader.get_data(idx)
        except IndexError:
            break
        pos = _detect_agent(frame)
        if pos is None:
            continue
        positions.append(pos)
        mags = _detect_magenta(frame)
        if mags:
            goal_dists.append(min(float(np.hypot(pos[0] - mx, pos[1] - my)) for mx, my in mags))

    if len(positions) < 2:
        return {
            "path": str(path),
            "frames_detected": len(positions),
            "error": "too few detected agent positions",
        }

    p = np.asarray(positions, dtype=np.float64)
    deltas = np.linalg.norm(np.diff(p, axis=0), axis=1)
    span = p.max(axis=0) - p.min(axis=0)
    arena_min = np.array([100.0, 40.0])
    arena_max = np.array([520.0, 320.0])
    border_dist = np.minimum.reduce(
        [
            p[:, 0] - arena_min[0],
            arena_max[0] - p[:, 0],
            p[:, 1] - arena_min[1],
            arena_max[1] - p[:, 1],
        ]
    )

    # More movement than displacement means loops/spins; very small displacement means
    # parking. This is approximate but useful for comparing eval videos.
    total_path = float(deltas.sum())
    net_disp = float(np.linalg.norm(p[-1] - p[0]))
    median_step = float(np.median(deltas))
    stationary_frac = float(np.mean(deltas < 1.0))
    border_frac = float(np.mean(border_dist < 25.0))
    loopiness = float(total_path / max(net_disp, 1.0))

    result = {
        "path": str(path),
        "fps": fps,
        "duration": duration,
        "frames_detected": len(positions),
        "total_path_px": round(total_path, 2),
        "net_displacement_px": round(net_disp, 2),
        "span_x_px": round(float(span[0]), 2),
        "span_y_px": round(float(span[1]), 2),
        "median_step_px": round(median_step, 2),
        "stationary_fraction": round(stationary_frac, 3),
        "border_fraction": round(border_frac, 3),
        "loopiness": round(loopiness, 2),
    }
    if goal_dists:
        result["median_goal_distance_px"] = round(float(np.median(goal_dists)), 2)
        result["near_goal_fraction"] = round(float(np.mean(np.asarray(goal_dists) < 35.0)), 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    args = parser.parse_args()

    results = [analyze(path, sample_fps=args.sample_fps) for path in args.videos]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
