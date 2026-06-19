from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from seedling_rl.envs import TrayEnvConfig
from seedling_sim.scene_generator import SceneGeneratorConfig


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    episodes: int
    scene_generator: SceneGeneratorConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scene_generator"] = self.scene_generator.__dict__
        return payload


def default_curriculum(base_config: TrayEnvConfig, episodes_per_stage: int = 100) -> list[CurriculumStage]:
    stages = [
        ("easy_single_crop", {"p_empty": 0.7, "p_single_crop": 0.25, "p_multiple_crop": 0.05, "p_weed_present": 0.0, "p_unknown_present": 0.0}),
        ("mixed_multiple", {"p_empty": 0.55, "p_single_crop": 0.25, "p_multiple_crop": 0.15, "p_weed_present": 0.05, "p_unknown_present": 0.0}),
        ("uncertain_review", {"p_empty": 0.5, "p_single_crop": 0.2, "p_multiple_crop": 0.15, "p_weed_present": 0.08, "p_unknown_present": 0.07}),
    ]
    result = []
    for stage_id, overrides in stages:
        generator = replace(base_config.scene_generator, **overrides)
        result.append(CurriculumStage(stage_id=stage_id, episodes=episodes_per_stage, scene_generator=generator))
    return result


def write_curriculum_plan(
    config: TrayEnvConfig,
    output_path: str | Path,
    episodes_per_stage: int = 100,
) -> list[CurriculumStage]:
    stages = default_curriculum(config, episodes_per_stage=episodes_per_stage)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"stages": [stage.to_dict() for stage in stages]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stages
