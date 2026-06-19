"""Calibration artifacts and coordinate transforms."""

from .estimation import (
    calibration_residuals,
    estimate_calibration_artifact,
    estimate_homography,
    load_calibration_config,
    write_error_map,
)
from .error_budget import (
    ERROR_BUDGET_COMPONENTS,
    ErrorBudgetReport,
    compute_error_budget,
    load_error_budget_components,
    write_error_budget_report,
)
from .intrinsics import (
    CameraIntrinsicsArtifact,
    IntrinsicsObservation,
    estimate_camera_intrinsics,
    load_intrinsics_observations,
    validate_camera_intrinsics,
)
from .schemas import CalibrationArtifact, CalibrationConfig, CalibrationValidationResult, ErrorSummary
from .transforms import (
    apply_homography,
    attach_calibration_to_scene,
    attach_calibration_to_target,
    image_px_to_tray_mm,
    tray_mm_to_robot_frame_mm,
)
from .validator import CalibrationValidator

__all__ = [
    "CalibrationArtifact",
    "CalibrationConfig",
    "CalibrationValidationResult",
    "CalibrationValidator",
    "CameraIntrinsicsArtifact",
    "ERROR_BUDGET_COMPONENTS",
    "ErrorSummary",
    "ErrorBudgetReport",
    "IntrinsicsObservation",
    "apply_homography",
    "attach_calibration_to_scene",
    "attach_calibration_to_target",
    "calibration_residuals",
    "compute_error_budget",
    "estimate_calibration_artifact",
    "estimate_camera_intrinsics",
    "estimate_homography",
    "image_px_to_tray_mm",
    "load_error_budget_components",
    "load_intrinsics_observations",
    "load_calibration_config",
    "tray_mm_to_robot_frame_mm",
    "validate_camera_intrinsics",
    "write_error_budget_report",
    "write_error_map",
]
