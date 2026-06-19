from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DomainRandomizationConfig:
    background_rgb: tuple[int, int, int] = (238, 236, 224)
    tray_line_rgb: tuple[int, int, int] = (170, 170, 160)
    lighting_scale: float = 1.0
    noise_std: float = 0.0
    plant_jitter_px: float = 0.0
    blur_radius: float = 0.0
    calibration_drift_mm: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainRandomizationConfig":
        return cls(
            background_rgb=_rgb(data.get("background_rgb", (238, 236, 224))),
            tray_line_rgb=_rgb(data.get("tray_line_rgb", (170, 170, 160))),
            lighting_scale=float(data.get("lighting_scale", 1.0)),
            noise_std=float(data.get("noise_std", 0.0)),
            plant_jitter_px=float(data.get("plant_jitter_px", 0.0)),
            blur_radius=float(data.get("blur_radius", 0.0)),
            calibration_drift_mm=_pair_float(data.get("calibration_drift_mm", (0.0, 0.0))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DOMAIN_RANDOMIZATION_PRESETS: dict[str, DomainRandomizationConfig] = {
    "none": DomainRandomizationConfig(),
    "greenhouse_default": DomainRandomizationConfig(
        background_rgb=(236, 232, 214),
        tray_line_rgb=(150, 150, 140),
        lighting_scale=1.05,
        noise_std=2.0,
        plant_jitter_px=1.5,
        calibration_drift_mm=(0.2, -0.1),
    ),
    "low_light_noisy": DomainRandomizationConfig(
        background_rgb=(210, 208, 196),
        tray_line_rgb=(130, 130, 124),
        lighting_scale=0.78,
        noise_std=8.0,
        plant_jitter_px=2.0,
        blur_radius=0.4,
        calibration_drift_mm=(0.5, -0.3),
    ),
    "wet_substrate": DomainRandomizationConfig(
        background_rgb=(105, 92, 78),
        tray_line_rgb=(80, 76, 70),
        lighting_scale=0.95,
        noise_std=3.0,
        plant_jitter_px=1.0,
        calibration_drift_mm=(-0.2, 0.3),
    ),
}


def domain_randomization_preset(name: str) -> DomainRandomizationConfig:
    try:
        return DOMAIN_RANDOMIZATION_PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown domain randomization preset {name!r}") from exc


def _rgb(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("RGB values must contain three channels")
    return tuple(max(0, min(255, int(channel))) for channel in value)


def _pair_float(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("calibration_drift_mm must contain two values")
    return (float(value[0]), float(value[1]))
