#!/usr/bin/env python3
"""Preflight the intended frozen binocular geometry from compiled MuJoCo XML.

This checks the run artifact, not only the Hydra values, and does not require a
rendering context.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class CameraSummary:
    name: str
    yaw_deg: float
    position_yaw_deg: float
    resolution: list[int]
    sensorsize: list[float]
    coverage_deg: list[float]


def _wrap_deg(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


def _camera_yaw_deg(quat_wxyz: np.ndarray) -> float:
    """Return camera optical-axis yaw from a MuJoCo camera quaternion."""
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    )
    forward = R.from_quat(quat_xyzw).apply([0.0, 0.0, -1.0])
    return _wrap_deg(np.rad2deg(np.arctan2(forward[1], forward[0])))


def _position_yaw_deg(pos: np.ndarray) -> float:
    return _wrap_deg(np.rad2deg(np.arctan2(pos[1], pos[0])))


def _parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split()]


def _parse_ints(text: str) -> list[int]:
    return [int(float(value)) for value in text.split()]


def _summarize_compiled_xml(path: Path) -> list[CameraSummary]:
    root = ET.parse(path).getroot()
    summaries: list[CameraSummary] = []
    for camera in root.findall(".//camera"):
        name = camera.attrib.get("name", "")
        if not name.startswith("agent_eye"):
            continue

        pos = np.asarray(_parse_floats(camera.attrib["pos"]), dtype=np.float64)
        quat = np.asarray(_parse_floats(camera.attrib["quat"]), dtype=np.float64)
        yaw = _camera_yaw_deg(quat)
        position_yaw = _position_yaw_deg(pos)
        resolution = _parse_ints(camera.attrib["resolution"])
        sensorsize = _parse_floats(camera.attrib["sensorsize"])
        horizontal_fov = 67.5
        summaries.append(
            CameraSummary(
                name=name,
                yaw_deg=yaw,
                position_yaw_deg=position_yaw,
                resolution=resolution,
                sensorsize=sensorsize,
                coverage_deg=[yaw - horizontal_fov / 2.0, yaw + horizontal_fov / 2.0],
            )
        )
    return sorted(summaries, key=lambda c: c.yaw_deg)


def _assert_close_list(
    actual: list[float],
    expected: list[float],
    *,
    atol: float,
    label: str,
) -> None:
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def _check_summaries(
    summaries: list[CameraSummary],
) -> dict[str, Any]:
    expected_yaws = [-18.75, 18.75]
    expected_resolution = [30, 20]  # MuJoCo stores [width, height].
    expected_sensorsize = [0.008284271247461903, 0.008909048505590654]
    expected_coverage = [[-52.5, 15.0], [-15.0, 52.5]]

    if len(summaries) != 2:
        raise AssertionError(f"expected exactly 2 agent cameras, got {len(summaries)}")

    actual_yaws = [round(s.yaw_deg, 6) for s in summaries]
    actual_position_yaws = [round(s.position_yaw_deg, 6) for s in summaries]
    _assert_close_list(actual_yaws, expected_yaws, atol=1e-3, label="camera yaw")
    _assert_close_list(
        actual_position_yaws,
        expected_yaws,
        atol=1e-3,
        label="camera placement yaw",
    )
    for summary in summaries:
        if summary.resolution != expected_resolution:
            raise AssertionError(
                f"{summary.name} resolution: expected {expected_resolution}, "
                f"got {summary.resolution}"
            )
        _assert_close_list(
            summary.sensorsize,
            expected_sensorsize,
            atol=1e-9,
            label=f"{summary.name} sensorsize",
        )
    for actual, expected in zip((s.coverage_deg for s in summaries), expected_coverage):
        _assert_close_list(actual, expected, atol=1e-3, label="horizontal coverage")

    result = {
        "preflight": "binocular_geometry",
        "status": "ok",
        "cameras": [asdict(summary) for summary in summaries],
        "expected_union_coverage_deg": [-52.5, 52.5],
        "expected_binocular_overlap_deg": [-15.0, 15.0],
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiled-xml",
        type=Path,
        required=True,
        help="Validate an existing compiled_env.xml file.",
    )
    args = parser.parse_args()
    _check_summaries(_summarize_compiled_xml(args.compiled_xml))
