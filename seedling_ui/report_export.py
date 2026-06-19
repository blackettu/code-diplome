from __future__ import annotations

import json
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

from seedling_core.schemas import SceneState
from seedling_sim import ReplayLog, SimScene


def write_report_export(
    scene_path: str | Path,
    output_path: str | Path,
    replay_path: str | Path | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    scene_data = _load_json(scene_path)
    scene_kind, scene = _load_scene(scene_data)
    replay = ReplayLog.from_json(replay_path) if replay_path else None
    report = build_report_payload(scene_kind, scene, replay, title=title)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".html", ".htm"}:
        output.write_text(_render_html(report), encoding="utf-8")
    else:
        output.write_text(_render_markdown(report), encoding="utf-8")
    return report


def build_report_payload(
    scene_kind: str,
    scene: SceneState | SimScene,
    replay: ReplayLog | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    scene_id = scene.scene_id
    target_rows = _target_rows(scene_kind, scene)
    payload: dict[str, Any] = {
        "title": title or f"Seedling Report: {scene_id}",
        "scene_kind": scene_kind,
        "scene_id": scene_id,
        "scene_summary": _scene_summary(scene_kind, scene),
        "distributions": _distributions(scene_kind, scene),
        "targets": target_rows,
        "replay_summary": {},
        "replay_events": [],
    }
    if replay is not None:
        payload["replay_summary"] = _replay_summary(replay)
        payload["replay_events"] = _replay_events(replay)
    return payload


def _load_scene(data: dict[str, Any]) -> tuple[str, SceneState | SimScene]:
    if isinstance(data.get("scenes"), list):
        scenes = [item for item in data["scenes"] if isinstance(item, dict)]
        if not scenes:
            raise ValueError("SceneState wrapper contains no scenes")
        return _load_scene(scenes[0])
    if "image_ref" in data and "tray" in data:
        return "scene_state", SceneState.from_dict(data)
    if "plants" in data and "cell_size_mm" in data:
        return "sim_scene", SimScene.from_dict(data)
    raise ValueError("Report export requires SceneState or SimScene JSON")


def _scene_summary(scene_kind: str, scene: SceneState | SimScene) -> list[dict[str, object]]:
    if scene_kind == "scene_state":
        assert isinstance(scene, SceneState)
        return [
            {"metric": "scene_id", "value": scene.scene_id},
            {"metric": "image_ref", "value": scene.image_ref},
            {"metric": "grid", "value": f"{scene.tray.grid_rows}x{scene.tray.grid_cols}"},
            {"metric": "detections", "value": len(scene.detections)},
            {"metric": "cells", "value": len(scene.cells)},
            {"metric": "targets", "value": len(scene.targets)},
            {"metric": "review_targets", "value": sum(1 for target in scene.targets if target.human_review_required)},
        ]
    assert isinstance(scene, SimScene)
    return [
        {"metric": "scene_id", "value": scene.scene_id},
        {"metric": "grid", "value": f"{scene.grid_rows}x{scene.grid_cols}"},
        {"metric": "cell_size_mm", "value": scene.cell_size_mm},
        {"metric": "plants", "value": len(scene.plants)},
        {"metric": "targets", "value": len(scene.targets)},
        {"metric": "processed_targets", "value": sum(1 for target in scene.targets if target.processed)},
        {"metric": "step_index", "value": scene.step_index},
    ]


def _distributions(scene_kind: str, scene: SceneState | SimScene) -> dict[str, list[dict[str, object]]]:
    if scene_kind == "scene_state":
        assert isinstance(scene, SceneState)
        return {
            "Detection Classes": _count_rows(Counter(detection.class_name for detection in scene.detections)),
            "Cell States": _count_rows(Counter(cell.state for cell in scene.cells)),
            "Target Types": _count_rows(Counter(target.target_type for target in scene.targets)),
        }
    assert isinstance(scene, SimScene)
    return {
        "Plant Classes": _count_rows(Counter(plant.class_name for plant in scene.plants)),
        "Target Types": _count_rows(Counter(target.target_type for target in scene.targets)),
    }


def _target_rows(scene_kind: str, scene: SceneState | SimScene) -> list[dict[str, object]]:
    if scene_kind == "scene_state":
        assert isinstance(scene, SceneState)
        cells_by_id = {cell.cell_id: cell for cell in scene.cells}
        return [
            {
                "target_id": target.target_id,
                "cell_id": target.cell_id,
                "object_id": target.object_id,
                "target_type": target.target_type,
                "point": target.action_point_px,
                "review": target.human_review_required,
                "risk_score": target.risk_score,
                "reasons": _target_reasons(target, cells_by_id),
            }
            for target in scene.targets
        ]
    assert isinstance(scene, SimScene)
    return [
        {
            "target_id": target.target_id,
            "object_id": target.object_id,
            "target_type": target.target_type,
            "point": target.point_mm,
            "processed": target.processed,
            "uncertainty_radius_mm": target.uncertainty_radius_mm,
        }
        for target in scene.targets
    ]


def _replay_summary(replay: ReplayLog) -> list[dict[str, object]]:
    event_counts = Counter(step.event_type for step in replay.steps)
    reason_counts: Counter[str] = Counter()
    for step in replay.steps:
        reason_counts.update(_payload_reasons(step.event_type, step.payload))
    reward_sum = sum(float(step.payload.get("reward", 0.0) or 0.0) for step in replay.steps)
    safety_blocks = sum(1 for step in replay.steps if _is_blocked(step.payload))
    return [
        {"metric": "replay_id", "value": replay.replay_id},
        {"metric": "scene_id", "value": replay.scene_id},
        {"metric": "steps", "value": len(replay.steps)},
        {"metric": "reward_sum", "value": reward_sum},
        {"metric": "safety_blocks", "value": safety_blocks},
        {"metric": "event_counts", "value": dict(sorted(event_counts.items()))},
        {"metric": "reason_counts", "value": dict(sorted(reason_counts.items()))},
    ]


def _replay_events(replay: ReplayLog) -> list[dict[str, object]]:
    rows = []
    for step in replay.steps:
        payload = step.payload
        rows.append(
            {
                "step": step.step_index,
                "event": step.event_type,
                "target_id": payload.get("target_id") or payload.get("command", {}).get("target_id"),
                "reward": payload.get("reward"),
                "ok": payload.get("ok"),
                "result": _safety_result(payload),
                "reasons": _payload_reasons(step.event_type, payload),
            }
        )
    return rows


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {report['title']}", ""]
    lines.extend(_markdown_table("Scene Summary", report["scene_summary"]))
    for title, rows in report["distributions"].items():
        lines.extend(_markdown_table(title, rows))
    lines.extend(_markdown_table("Targets", report["targets"]))
    if report["replay_summary"]:
        lines.extend(_markdown_table("Replay Summary", report["replay_summary"]))
        lines.extend(_markdown_table("Replay Events", report["replay_events"]))
    return "\n".join(lines)


def _render_html(report: dict[str, Any]) -> str:
    sections = [_html_table("Scene Summary", report["scene_summary"])]
    sections.extend(_html_table(title, rows) for title, rows in report["distributions"].items())
    sections.append(_html_table("Targets", report["targets"]))
    if report["replay_summary"]:
        sections.append(_html_table("Replay Summary", report["replay_summary"]))
        sections.append(_html_table("Replay Events", report["replay_events"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(str(report['title']))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2528; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f2f2ed; }}
  </style>
</head>
<body>
  <h1>{escape(str(report['title']))}</h1>
  {"".join(sections)}
</body>
</html>
"""


def _markdown_table(title: str, rows: list[dict[str, object]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        return lines + ["No rows.", ""]
    columns = list(rows[0].keys())
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(column)) for column in columns) + " |")
    lines.append("")
    return lines


def _html_table(title: str, rows: list[dict[str, object]]) -> str:
    if not rows:
        return f"<section><h2>{escape(title)}</h2><p>No rows.</p></section>"
    columns = list(rows[0].keys())
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(_cell(row.get(column)))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    return f"<section><h2>{escape(title)}</h2><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></section>"


def _count_rows(counter: Counter[str]) -> list[dict[str, object]]:
    return [{"name": key, "count": value} for key, value in sorted(counter.items())]


def _is_blocked(payload: dict[str, Any]) -> bool:
    safety = payload.get("safety_decision")
    if isinstance(safety, dict) and safety.get("allowed") is False:
        return True
    return payload.get("ok") is False


def _target_reasons(target: Any, cells_by_id: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if getattr(target, "human_review_required", False):
        reasons.append("human_review_required")
    forbidden_zone_ids = getattr(target, "forbidden_zone_ids", [])
    if forbidden_zone_ids:
        reasons.append("forbidden_zone_overlap")
        reasons.extend(f"forbidden_zone:{zone_id}" for zone_id in forbidden_zone_ids)
    cell = cells_by_id.get(getattr(target, "cell_id", ""))
    if cell is not None:
        if getattr(cell, "human_review_required", False):
            reasons.append("cell_review_required")
        state = getattr(cell, "state", "")
        if state == "unknown":
            reasons.append("unknown_plant_present")
        elif state in {"ambiguous", "image_quality_insufficient", "invalid_geometry"}:
            reasons.append("ambiguous_or_invalid_cell")
        elif state == "foreign_object_present":
            reasons.append("foreign_object_present")
        reasons.extend(str(flag) for flag in getattr(cell, "risk_flags", []))
    return sorted(dict.fromkeys(reason for reason in reasons if reason))


def _payload_reasons(event_type: str, payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    safety = payload.get("safety_decision")
    if isinstance(safety, dict):
        raw_reasons = safety.get("reasons")
        if isinstance(raw_reasons, list):
            reasons.extend(str(reason) for reason in raw_reasons)
        elif raw_reasons:
            reasons.append(str(raw_reasons))
        if safety.get("result"):
            reasons.append(str(safety["result"]))
    if event_type == "rl_step" and payload.get("event"):
        reasons.append(str(payload["event"]))
    return sorted(dict.fromkeys(reason for reason in reasons if reason))


def _safety_result(payload: dict[str, Any]) -> object:
    safety = payload.get("safety_decision")
    if isinstance(safety, dict):
        return safety.get("result") or safety.get("reasons")
    return payload.get("result")


def _cell(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def _load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data
