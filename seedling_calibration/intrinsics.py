from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


INTRINSICS_SCHEMA_VERSION = "camera_intrinsics_v0_1"
SUPPORTED_TARGET_TYPES = {"chessboard", "aruco", "fiducials"}


@dataclass
class IntrinsicsObservation:
    observation_id: str
    target_type: str
    image_points_px: list[list[float]]
    object_points_mm: list[list[float]]
    image_size_px: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observation_id = str(self.observation_id)
        self.target_type = str(self.target_type)
        if self.target_type not in SUPPORTED_TARGET_TYPES:
            raise ValueError(f"target_type must be one of {sorted(SUPPORTED_TARGET_TYPES)}")
        self.image_points_px = [_float_list("image_points_px[]", point, 2) for point in self.image_points_px]
        self.object_points_mm = [_float_list("object_points_mm[]", point, 3) for point in self.object_points_mm]
        if len(self.image_points_px) != len(self.object_points_mm):
            raise ValueError("image_points_px and object_points_mm must have the same length")
        if len(self.image_points_px) < 4:
            raise ValueError("at least four point correspondences are required")
        self.image_size_px = [int(value) for value in _float_list("image_size_px", self.image_size_px, 2)]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntrinsicsObservation":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraIntrinsicsArtifact:
    camera_id: str
    image_size_px: list[int]
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    target_type: str
    reprojection_error_px: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = INTRINSICS_SCHEMA_VERSION
    observations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.camera_id = str(self.camera_id)
        self.image_size_px = [int(value) for value in _float_list("image_size_px", self.image_size_px, 2)]
        self.camera_matrix = [_float_list("camera_matrix[]", row, 3) for row in self.camera_matrix]
        if len(self.camera_matrix) != 3:
            raise ValueError("camera_matrix must be 3x3")
        self.distortion_coefficients = [float(value) for value in self.distortion_coefficients]
        self.target_type = str(self.target_type)
        if self.target_type not in SUPPORTED_TARGET_TYPES:
            raise ValueError(f"target_type must be one of {sorted(SUPPORTED_TARGET_TYPES)}")
        if self.reprojection_error_px is not None:
            self.reprojection_error_px = float(self.reprojection_error_px)
            if self.reprojection_error_px < 0:
                raise ValueError("reprojection_error_px must be non-negative")
        _parse_datetime(self.created_at)
        self.observations = [str(value) for value in self.observations]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraIntrinsicsArtifact":
        payload = dict(data)
        payload.pop("schema_version", None)
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraIntrinsicsArtifact":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError(f"Camera intrinsics artifact must be a JSON object: {path}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_intrinsics_observations(path: str | Path) -> list[IntrinsicsObservation]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    raw = data.get("observations") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("Intrinsics observations must be a list or {'observations': [...]}")
    return [IntrinsicsObservation.from_dict(item) for item in raw]


def estimate_camera_intrinsics(
    observations: list[IntrinsicsObservation],
    camera_id: str,
) -> CameraIntrinsicsArtifact:
    if not observations:
        raise ValueError("at least one observation is required")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to estimate camera intrinsics") from exc

    image_size = observations[0].image_size_px
    target_type = observations[0].target_type
    if any(item.image_size_px != image_size for item in observations):
        raise ValueError("all observations must have the same image_size_px")
    if any(item.target_type != target_type for item in observations):
        raise ValueError("all observations must use the same target_type")

    object_points = [np.asarray(item.object_points_mm, dtype=np.float32) for item in observations]
    image_points = [np.asarray(item.image_points_px, dtype=np.float32) for item in observations]
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        tuple(image_size),
        None,
        None,
    )
    return CameraIntrinsicsArtifact(
        camera_id=camera_id,
        image_size_px=image_size,
        camera_matrix=[[float(value) for value in row] for row in camera_matrix.tolist()],
        distortion_coefficients=[float(value) for value in distortion.flatten().tolist()],
        target_type=target_type,
        reprojection_error_px=float(rms),
        observations=[item.observation_id for item in observations],
        metadata={"estimator": "opencv_calibrateCamera"},
    )


def validate_camera_intrinsics(
    artifact: CameraIntrinsicsArtifact,
    max_reprojection_error_px: float | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if artifact.camera_matrix[0][0] <= 0 or artifact.camera_matrix[1][1] <= 0:
        errors.append("focal lengths in camera_matrix must be positive")
    if artifact.camera_matrix[2] != [0.0, 0.0, 1.0]:
        warnings.append("camera_matrix third row is not [0, 0, 1]")
    if max_reprojection_error_px is not None:
        if artifact.reprojection_error_px is None:
            errors.append("reprojection_error_px is required")
        elif artifact.reprojection_error_px > max_reprojection_error_px:
            errors.append(
                f"reprojection_error_px={artifact.reprojection_error_px} exceeds limit {max_reprojection_error_px}"
            )
    return {
        "ok": not errors,
        "camera_id": artifact.camera_id,
        "errors": errors,
        "warnings": warnings,
        "target_type": artifact.target_type,
    }


def _float_list(name: str, value: Any, expected_length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} numeric values")
    return [float(item) for item in value]


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
