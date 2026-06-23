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
    fov_deg: list[float]
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


def _fov_deg(sensor_size: float, focal_length: float) -> float:
    return float(np.rad2deg(2.0 * np.arctan(sensor_size / (2.0 * focal_length))))


def _camera_transform_element(
    camera: ET.Element, parent_map: dict[ET.Element, ET.Element]
) -> ET.Element:
    if "pos" in camera.attrib and "quat" in camera.attrib:
        return camera

    parent = parent_map.get(camera)
    while parent is not None:
        if parent.tag == "body" and "pos" in parent.attrib and "quat" in parent.attrib:
            return parent
        parent = parent_map.get(parent)

    raise KeyError(f"No pos/quat transform found for camera {camera.attrib.get('name')}")


def _summarize_compiled_xml(path: Path) -> list[CameraSummary]:
    root = ET.parse(path).getroot()
    parent_map = {child: parent for parent in root.iter() for child in parent}
    summaries: list[CameraSummary] = []
    for camera in root.findall(".//camera"):
        name = camera.attrib.get("name", "")
        if not name.startswith("agent_eye"):
            continue

        transform = _camera_transform_element(camera, parent_map)
        pos = np.asarray(_parse_floats(transform.attrib["pos"]), dtype=np.float64)
        quat = np.asarray(_parse_floats(transform.attrib["quat"]), dtype=np.float64)
        yaw = _camera_yaw_deg(quat)
        position_yaw = _position_yaw_deg(pos)
        resolution = _parse_ints(camera.attrib["resolution"])
        sensorsize = _parse_floats(camera.attrib["sensorsize"])
        focal = _parse_floats(camera.attrib["focal"])
        fov = [
            _fov_deg(sensorsize[0], focal[0]),
            _fov_deg(sensorsize[1], focal[1]),
        ]
        horizontal_fov = fov[1]
        summaries.append(
            CameraSummary(
                name=name,
                yaw_deg=yaw,
                position_yaw_deg=position_yaw,
                resolution=resolution,
                sensorsize=sensorsize,
                fov_deg=fov,
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
    expected_sensorsize = [0.008284271247461903, 0.013363180779263536]
    expected_fov = [45.0, 67.5]
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
            atol=1e-6,
            label=f"{summary.name} sensorsize",
        )
        _assert_close_list(
            summary.fov_deg,
            expected_fov,
            atol=1e-3,
            label=f"{summary.name} fov",
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
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optionally write the successful preflight result as JSON.",
    )
    args = parser.parse_args()
    result = _check_summaries(_summarize_compiled_xml(args.compiled_xml))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
