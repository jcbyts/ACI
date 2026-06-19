#!/usr/bin/env python3
"""Export r2dreamer TensorBoard eval GIF summaries as readable MP4 files."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageSequence
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def decode_gif(encoded: bytes) -> list[Image.Image]:
    image = Image.open(io.BytesIO(encoded))
    return [frame.convert("RGB") for frame in ImageSequence.Iterator(image)]


def label_and_scale(
    frames: list[Image.Image],
    *,
    step: int,
    scale: int,
) -> list[np.ndarray]:
    if not frames:
        return []

    width, height = frames[0].size
    label_height = max(18, 3 * scale)
    out: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        scaled = frame.resize((width * scale, height * scale), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (scaled.width, scaled.height + label_height), "black")
        canvas.paste(scaled, (0, label_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 2), f"eval step {step} | frame {idx + 1}/{len(frames)}", fill="white")
        out.append(np.asarray(canvas))
    return out


def export_eval_videos(
    logdir: Path,
    *,
    tag: str,
    outdir: Path,
    scale: int,
    fps: int,
) -> list[Path]:
    accumulator = EventAccumulator(str(logdir), size_guidance={"images": 0})
    accumulator.Reload()
    events = accumulator.Images(tag)
    outdir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for event in events:
        frames = label_and_scale(decode_gif(event.encoded_image_string), step=event.step, scale=scale)
        if not frames:
            continue
        path = outdir / f"{tag}_step_{event.step:06d}.mp4"
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=Path)
    parser.add_argument("--tag", default="eval_video")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    outdir = args.outdir or args.logdir / "exported_eval_videos"
    written = export_eval_videos(
        args.logdir,
        tag=args.tag,
        outdir=outdir,
        scale=args.scale,
        fps=args.fps,
    )
    for path in written:
        print(path)
    print(f"exported {len(written)} videos to {outdir}")


if __name__ == "__main__":
    main()
