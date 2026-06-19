from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RewardConfig:
    correct_remove: float = 10.0
    correct_review: float = 3.0
    episode_complete: float = 1.0
    move_mm_penalty: float = 0.01
    repeated_action: float = -2.0
    missed_target: float = -5.0
    unsafe_action: float = -100.0
    crop_damage: float = -50.0
    stop_with_remaining_target: float = -5.0

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> "RewardConfig":
        if not data:
            return cls()
        valid_fields = {field.name for field in fields(cls)}
        return cls(**{key: float(value) for key, value in data.items() if key in valid_fields})
