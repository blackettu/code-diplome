from __future__ import annotations

import random
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ActuatorSample:
    commanded_point_mm: list[float]
    actual_point_mm: list[float]
    drift_mm: list[float]
    latency_ms: float
    zone_error_mm: list[float]
    zone_error_active: bool = False


@dataclass
class ActuatorErrorModel:
    xy_sigma_mm: float = 0.8
    drift_sigma_mm: float = 0.2
    latency_ms_mean: float = 0.0
    latency_ms_sigma: float = 0.0
    zone_error_prob: float = 0.0
    zone_error_mm: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self.xy_sigma_mm = float(self.xy_sigma_mm)
        self.drift_sigma_mm = float(self.drift_sigma_mm)
        self.latency_ms_mean = float(self.latency_ms_mean)
        self.latency_ms_sigma = float(self.latency_ms_sigma)
        self.zone_error_prob = float(self.zone_error_prob)
        self.zone_error_mm = float(self.zone_error_mm)
        if not 0.0 <= self.zone_error_prob <= 1.0:
            raise ValueError("zone_error_prob must be in [0, 1]")
        if self.zone_error_mm < 0.0:
            raise ValueError("zone_error_mm must be non-negative")
        self._rng = random.Random(self.seed)

    def sample(self, commanded_point_mm: list[float]) -> ActuatorSample:
        if len(commanded_point_mm) != 2:
            raise ValueError("commanded_point_mm must contain [x, y]")
        drift = [
            self._rng.gauss(0.0, self.drift_sigma_mm),
            self._rng.gauss(0.0, self.drift_sigma_mm),
        ]
        noise = [
            self._rng.gauss(0.0, self.xy_sigma_mm),
            self._rng.gauss(0.0, self.xy_sigma_mm),
        ]
        zone_error = self._zone_error()
        actual = [
            float(commanded_point_mm[0]) + drift[0] + noise[0] + zone_error[0],
            float(commanded_point_mm[1]) + drift[1] + noise[1] + zone_error[1],
        ]
        latency = max(0.0, self._rng.gauss(self.latency_ms_mean, self.latency_ms_sigma))
        return ActuatorSample(
            commanded_point_mm=[float(commanded_point_mm[0]), float(commanded_point_mm[1])],
            actual_point_mm=actual,
            drift_mm=drift,
            latency_ms=latency,
            zone_error_mm=zone_error,
            zone_error_active=any(abs(value) > 0.0 for value in zone_error),
        )

    def _zone_error(self) -> list[float]:
        if self.zone_error_mm <= 0.0 or self._rng.random() >= self.zone_error_prob:
            return [0.0, 0.0]
        angle = self._rng.uniform(0.0, 2.0 * 3.141592653589793)
        return [
            self.zone_error_mm * math.cos(angle),
            self.zone_error_mm * math.sin(angle),
        ]
