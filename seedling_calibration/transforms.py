from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from seedling_core.schemas import ActionTarget, SceneState

from .schemas import CalibrationArtifact


def apply_homography(point_xy: list[float] | tuple[float, float], matrix: list[list[float]]) -> list[float]:
    if len(point_xy) != 2:
        raise ValueError("point_xy must contain [x, y]")
    x, y = float(point_xy[0]), float(point_xy[1])
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("homography matrix must be 3x3")
    denom = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(denom) < 1e-12:
        raise ValueError("homography denominator is too close to zero")
    return [
        (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denom,
        (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denom,
    ]


def image_px_to_tray_mm(point_px: list[float] | tuple[float, float], artifact: CalibrationArtifact) -> list[float]:
    return apply_homography(point_px, artifact.image_to_tray_homography)


def tray_mm_to_robot_frame_mm(
    point_mm: list[float] | tuple[float, float],
    artifact: CalibrationArtifact,
    include_tool_offset: bool = True,
) -> list[float]:
    if len(point_mm) != 2:
        raise ValueError("point_mm must contain [x, y]")
    x, y = float(point_mm[0]), float(point_mm[1])
    m = artifact.tray_to_robot_transform
    denom = m[2][0] * x + m[2][1] * y + m[2][2]
    if abs(denom) < 1e-12:
        raise ValueError("tray_to_robot_transform denominator is too close to zero")
    robot = [
        (m[0][0] * x + m[0][1] * y + m[0][2]) / denom,
        (m[1][0] * x + m[1][1] * y + m[1][2]) / denom,
        0.0,
    ]
    if include_tool_offset:
        robot = [robot[index] + artifact.tool_offset_mm[index] for index in range(3)]
    return robot


def attach_calibration_to_target(target: ActionTarget, artifact: CalibrationArtifact) -> ActionTarget:
    point_mm = target.action_point_mm or image_px_to_tray_mm(target.action_point_px, artifact)
    robot_point = tray_mm_to_robot_frame_mm(point_mm, artifact)
    uncertainty = target.uncertainty_radius_mm
    if uncertainty is None:
        uncertainty = artifact.error_summary_mm.p95
    return replace(
        target,
        action_point_mm=point_mm,
        robot_point_mm=robot_point,
        uncertainty_radius_mm=uncertainty,
    )


def attach_calibration_to_targets(
    targets: Iterable[ActionTarget],
    artifact: CalibrationArtifact,
) -> list[ActionTarget]:
    return [attach_calibration_to_target(target, artifact) for target in targets]


def attach_calibration_to_scene(scene: SceneState, artifact: CalibrationArtifact) -> SceneState:
    return replace(
        scene,
        tray=replace(scene.tray, calibration_id=artifact.calibration_id),
        targets=attach_calibration_to_targets(scene.targets, artifact),
    )
