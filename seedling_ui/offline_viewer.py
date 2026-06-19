from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def write_offline_viewer(
    predictions_path: str | Path,
    output_path: str | Path,
    image_name: str | None = None,
    image_path: str | Path | None = None,
) -> None:
    prediction = _load_prediction(predictions_path, image_name)
    html = render_offline_viewer_html(prediction, image_path=image_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def render_offline_viewer_html(
    prediction: dict[str, Any],
    image_path: str | Path | None = None,
) -> str:
    if _is_scene_state(prediction):
        return _render_scene_state_viewer_html(prediction, image_path=image_path)
    width = int(prediction.get("width", 1) or 1)
    height = int(prediction.get("height", 1) or 1)
    image_ref = str(image_path or prediction.get("path") or prediction.get("image") or "")
    overlay = _overlay_svg(prediction, width, height)
    targets = _targets_table(prediction)
    detections = _detections_table(prediction)
    analysis = _analysis_table(prediction)
    feedback = _feedback_panel(
        image_id=str(prediction.get("image") or prediction.get("path") or "unknown_image"),
        items=_legacy_feedback_items(prediction),
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(str(prediction.get("image", "offline viewer")))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2528; }}
    main {{ display: grid; grid-template-columns: minmax(360px, 1fr) 420px; gap: 24px; align-items: start; }}
    .canvas {{ position: relative; border: 1px solid #ccc; display: inline-block; max-width: 100%; }}
    .canvas img, .canvas svg {{ display: block; max-width: 100%; height: auto; }}
    .canvas svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; font-size: 13px; }}
    td, th {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f2f2ed; }}
    code {{ font-family: Consolas, monospace; }}
    textarea, select, input {{ width: 100%; box-sizing: border-box; margin: 4px 0 10px; }}
    textarea {{ min-height: 64px; font-family: Consolas, monospace; }}
    .feedback-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  </style>
</head>
<body>
  <h1>{escape(str(prediction.get("image", "offline viewer")))}</h1>
  <main>
    <section>
      <div class="canvas" style="width:{width}px">
        {_image_tag(image_ref, width, height)}
        {overlay}
      </div>
    </section>
    <section>
      <h2>Container Analysis</h2>
      {analysis}
      <h2>Removal Targets</h2>
      {targets}
      {feedback}
      <h2>Detections</h2>
      {detections}
    </section>
  </main>
  {_feedback_script()}
</body>
</html>
"""


def _load_prediction(predictions_path: str | Path, image_name: str | None) -> dict[str, Any]:
    data = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"predictions.json must be a JSON object: {predictions_path}")
    if _is_scene_state(data):
        return data
    if isinstance(data.get("scenes"), list):
        scenes = [item for item in data["scenes"] if isinstance(item, dict) and _is_scene_state(item)]
        if not scenes:
            raise ValueError("SceneState bundle must contain non-empty `scenes` list")
        if image_name is None:
            return scenes[0]
        for scene in scenes:
            if image_name in {scene.get("scene_id"), scene.get("image_ref"), Path(str(scene.get("image_ref", ""))).name}:
                return scene
        raise KeyError(f"Scene or image {image_name!r} not found in {predictions_path}")
    images = data.get("images", [])
    if not isinstance(images, list) or not images:
        raise ValueError("offline viewer input must contain non-empty `images` or `scenes` list")
    if image_name is None:
        return images[0]
    for prediction in images:
        if prediction.get("image") == image_name:
            return prediction
    raise KeyError(f"Image {image_name!r} not found in {predictions_path}")


def _is_scene_state(data: dict[str, Any]) -> bool:
    return isinstance(data.get("tray"), dict) and isinstance(data.get("image_size_px"), list) and "scene_id" in data


def _render_scene_state_viewer_html(
    scene: dict[str, Any],
    image_path: str | Path | None = None,
) -> str:
    image_size = scene.get("image_size_px", [1, 1])
    width = int(image_size[0] or 1)
    height = int(image_size[1] or 1)
    image_ref = str(image_path or scene.get("image_ref") or "")
    title = escape(str(scene.get("scene_id", "scene viewer")))
    feedback = _feedback_panel(
        image_id=str(scene.get("image_ref") or scene.get("scene_id") or "unknown_scene"),
        items=_scene_feedback_items(scene),
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2528; }}
    main {{ display: grid; grid-template-columns: minmax(360px, 1fr) 480px; gap: 24px; align-items: start; }}
    .canvas {{ position: relative; border: 1px solid #ccc; display: inline-block; max-width: 100%; }}
    .canvas img, .canvas svg {{ display: block; max-width: 100%; height: auto; }}
    .canvas svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; font-size: 13px; }}
    td, th {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f2f2ed; }}
    code {{ font-family: Consolas, monospace; }}
    textarea, select, input {{ width: 100%; box-sizing: border-box; margin: 4px 0 10px; }}
    textarea {{ min-height: 64px; font-family: Consolas, monospace; }}
    .feedback-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <main>
    <section>
      <div class="canvas" style="width:{width}px">
        {_image_tag(image_ref, width, height)}
        {_scene_overlay_svg(scene, width, height)}
      </div>
    </section>
    <section>
      <h2>Scene Summary</h2>
      {_scene_summary_table(scene)}
      <h2>Cells</h2>
      {_scene_cells_table(scene)}
      <h2>Action Targets</h2>
      {_scene_targets_table(scene)}
      {feedback}
      <h2>Detections</h2>
      {_scene_detections_table(scene)}
    </section>
  </main>
  {_feedback_script()}
</body>
</html>
"""


def _scene_overlay_svg(scene: dict[str, Any], width: int, height: int) -> str:
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    tray = scene.get("tray") if isinstance(scene.get("tray"), dict) else {}
    corners = tray.get("corners_px") if isinstance(tray, dict) else None
    bbox = tray.get("bbox_xyxy_px") if isinstance(tray, dict) else None
    if isinstance(corners, list) and corners:
        parts.append(
            f'<polygon points="{_svg_points(corners)}" fill="none" stroke="#d62728" stroke-width="3">'
            f"<title>{escape(str(tray.get('tray_id', 'tray')))}</title></polygon>"
        )
    elif isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        parts.append(
            f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#d62728" stroke-width="3">'
            f"<title>{escape(str(tray.get('tray_id', 'tray')))}</title></rect>"
        )
    for cell in scene.get("cells", []):
        if not isinstance(cell, dict) or not isinstance(cell.get("polygon_px"), list):
            continue
        title = f"{cell.get('cell_id')} {cell.get('state')}"
        risk_flags = cell.get("risk_flags")
        if risk_flags:
            title = f"{title}; {risk_flags}"
        parts.append(
            f'<polygon points="{_svg_points(cell["polygon_px"])}" fill="#2ca02c22" stroke="#2ca02c" stroke-width="1">'
            f"<title>{escape(title)}</title></polygon>"
        )
    for detection in scene.get("detections", []):
        if not isinstance(detection, dict) or not isinstance(detection.get("bbox_xyxy_px"), list):
            continue
        x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy_px"]]
        label = f"{detection.get('object_id')} {detection.get('class_name')} {detection.get('confidence')}"
        parts.append(
            f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#1f77b4" stroke-width="2">'
            f"<title>{escape(label)}</title></rect>"
        )
    cells_by_id = _cells_by_id(scene)
    for target in scene.get("targets", []):
        if not isinstance(target, dict) or not isinstance(target.get("action_point_px"), list):
            continue
        x, y = [float(value) for value in target["action_point_px"]]
        reasons = _scene_target_reasons(target, cells_by_id)
        title = f"{target.get('target_id')} {target.get('target_type')}"
        if reasons:
            title = f"{title}; {'; '.join(reasons)}"
        parts.append(
            f'<path d="M {x-8} {y} L {x+8} {y} M {x} {y-8} L {x} {y+8}" stroke="#ff7f0e" stroke-width="3">'
            f"<title>{escape(title)}</title></path>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_points(points: list[Any]) -> str:
    pairs = []
    for point in points:
        if isinstance(point, list) and len(point) >= 2:
            pairs.append(f"{float(point[0])},{float(point[1])}")
    return " ".join(pairs)


def _scene_summary_table(scene: dict[str, Any]) -> str:
    tray = scene.get("tray") if isinstance(scene.get("tray"), dict) else {}
    robot = scene.get("robot") if isinstance(scene.get("robot"), dict) else {}
    safety = scene.get("safety") if isinstance(scene.get("safety"), dict) else {}
    rows = [
        {"metric": "scene_id", "value": scene.get("scene_id")},
        {"metric": "dataset_version", "value": scene.get("dataset_version")},
        {"metric": "ontology_version", "value": scene.get("ontology_version")},
        {"metric": "tray_id", "value": tray.get("tray_id")},
        {"metric": "cells", "value": len(scene.get("cells", []))},
        {"metric": "targets", "value": len(scene.get("targets", []))},
        {"metric": "robot_mode", "value": robot.get("mode")},
        {"metric": "calibration_valid", "value": safety.get("calibration_valid")},
    ]
    return _table(rows)


def _scene_cells_table(scene: dict[str, Any]) -> str:
    rows = []
    for cell in scene.get("cells", []):
        if not isinstance(cell, dict):
            continue
        rows.append(
            {
                "cell_id": cell.get("cell_id"),
                "row": cell.get("row"),
                "col": cell.get("col"),
                "state": cell.get("state"),
                "confidence": cell.get("state_confidence"),
                "objects": cell.get("object_ids"),
                "keep_object": cell.get("keep_object_id"),
                "removal_candidates": cell.get("removal_candidate_ids"),
                "review": cell.get("human_review_required"),
                "risk_flags": cell.get("risk_flags"),
            }
        )
    return _table(rows)


def _scene_targets_table(scene: dict[str, Any]) -> str:
    rows = []
    cells_by_id = _cells_by_id(scene)
    for target in scene.get("targets", []):
        if not isinstance(target, dict):
            continue
        rows.append(
            {
                "target_id": target.get("target_id"),
                "cell_id": target.get("cell_id"),
                "object_id": target.get("object_id"),
                "type": target.get("target_type"),
                "action_point_px": target.get("action_point_px"),
                "action_point_mm": target.get("action_point_mm"),
                "robot_point_mm": target.get("robot_point_mm"),
                "review": target.get("human_review_required"),
                "risk_score": target.get("risk_score"),
                "reasons": "; ".join(_scene_target_reasons(target, cells_by_id)),
            }
        )
    return _table(rows)


def _scene_detections_table(scene: dict[str, Any]) -> str:
    rows = []
    for detection in scene.get("detections", []):
        if not isinstance(detection, dict):
            continue
        rows.append(
            {
                "object_id": detection.get("object_id"),
                "class": detection.get("class_name") or detection.get("class_id"),
                "confidence": detection.get("confidence"),
                "box": detection.get("bbox_xyxy_px"),
                "attributes": detection.get("attributes"),
            }
        )
    return _table(rows)


def _cells_by_id(scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(cell.get("cell_id")): cell
        for cell in scene.get("cells", [])
        if isinstance(cell, dict) and cell.get("cell_id") is not None
    }


def _scene_target_reasons(target: dict[str, Any], cells_by_id: dict[str, dict[str, Any]]) -> list[str]:
    reasons = _target_reasons(target)
    forbidden_zones = target.get("forbidden_zone_ids")
    if isinstance(forbidden_zones, list) and forbidden_zones:
        reasons.append("forbidden_zone_overlap")
        reasons.extend(f"forbidden_zone:{zone_id}" for zone_id in forbidden_zones)
    cell = cells_by_id.get(str(target.get("cell_id")))
    if cell:
        if cell.get("human_review_required"):
            reasons.append("cell_review_required")
        state = cell.get("state")
        if state == "unknown":
            reasons.append("unknown_plant_present")
        elif state in {"ambiguous", "image_quality_insufficient", "invalid_geometry"}:
            reasons.append("ambiguous_or_invalid_cell")
        elif state == "foreign_object_present":
            reasons.append("foreign_object_present")
        risk_flags = cell.get("risk_flags")
        if isinstance(risk_flags, list):
            reasons.extend(str(flag) for flag in risk_flags)
    return sorted(dict.fromkeys(reason for reason in reasons if reason))


def _scene_feedback_items(scene: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for target in scene.get("targets", []):
        if not isinstance(target, dict):
            continue
        label = f"target {target.get('target_id')} ({target.get('target_type')})"
        items.append(
            {
                "label": label,
                "payload": {
                    "target_id": target.get("target_id"),
                    "object_id": target.get("object_id"),
                    "cell_id": target.get("cell_id"),
                    "metadata": {"source": "offline_viewer", "item_type": "target"},
                },
            }
        )
    for cell in scene.get("cells", []):
        if not isinstance(cell, dict):
            continue
        items.append(
            {
                "label": f"cell {cell.get('cell_id')} ({cell.get('state')})",
                "payload": {
                    "cell_id": cell.get("cell_id"),
                    "metadata": {"source": "offline_viewer", "item_type": "cell", "state": cell.get("state")},
                },
            }
        )
    for detection in scene.get("detections", []):
        if not isinstance(detection, dict):
            continue
        items.append(
            {
                "label": f"object {detection.get('object_id')} ({detection.get('class_name')})",
                "payload": {
                    "object_id": detection.get("object_id"),
                    "metadata": {
                        "source": "offline_viewer",
                        "item_type": "detection",
                        "class_name": detection.get("class_name"),
                    },
                },
            }
        )
    return items


def _legacy_feedback_items(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for analysis in prediction.get("container_analysis", []):
        if not isinstance(analysis, dict):
            continue
        for index, target in enumerate(analysis.get("removal_targets", []), 1):
            if not isinstance(target, dict):
                continue
            items.append(
                {
                    "label": f"target container {analysis.get('index')} r{target.get('row')} c{target.get('col')}",
                    "payload": {
                        "target_id": target.get("target_id"),
                        "object_id": target.get("object_id"),
                        "cell_id": target.get("cell_id"),
                        "metadata": {
                            "source": "offline_viewer",
                            "item_type": "legacy_target",
                            "container_index": analysis.get("index"),
                            "target_index": index,
                            "row": target.get("row"),
                            "col": target.get("col"),
                        },
                    },
                }
            )
    for index, detection in enumerate(prediction.get("detections", []), 1):
        if not isinstance(detection, dict):
            continue
        items.append(
            {
                "label": f"detection {index} ({detection.get('name') or detection.get('class_id')})",
                "payload": {
                    "object_id": detection.get("object_id"),
                    "metadata": {
                        "source": "offline_viewer",
                        "item_type": "legacy_detection",
                        "detection_index": index,
                        "class_id": detection.get("class_id"),
                        "class_name": detection.get("name"),
                    },
                },
            }
        )
    return items


def _feedback_panel(image_id: str, items: list[dict[str, Any]]) -> str:
    options = []
    for index, item in enumerate(items):
        payload = _clean_feedback_payload(item.get("payload", {}))
        options.append(
            f'<option value="{index}" data-feedback-payload="{escape(json.dumps(payload, ensure_ascii=False, sort_keys=True))}">'
            f"{escape(str(item.get('label') or f'item {index + 1}'))}</option>"
        )
    if not options:
        options.append('<option value="manual" data-feedback-payload="{}">manual feedback</option>')
    return f"""
      <h2>Annotation Feedback</h2>
      <section data-feedback-panel data-image-id="{escape(image_id)}">
        <div class="feedback-row">
          <label>Item<select data-feedback-item>{''.join(options)}</select></label>
          <label>Error type<select data-feedback-error-type>
            <option value="wrong_target">wrong_target</option>
            <option value="missed_target">missed_target</option>
            <option value="false_positive_target">false_positive_target</option>
            <option value="bad_cell_state">bad_cell_state</option>
            <option value="bad_bbox">bad_bbox</option>
            <option value="crop_damage">crop_damage</option>
            <option value="unsafe_action">unsafe_action</option>
          </select></label>
        </div>
        <label>Comment<input data-feedback-comment value=""></label>
        <textarea data-feedback-json readonly></textarea>
      </section>
"""


def _clean_feedback_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None or value == "":
            continue
        if isinstance(value, dict):
            nested = _clean_feedback_payload(value)
            if nested:
                cleaned[key] = nested
        else:
            cleaned[key] = value
    return cleaned


def _feedback_script() -> str:
    return """<script>
(() => {
  for (const panel of document.querySelectorAll('[data-feedback-panel]')) {
    const item = panel.querySelector('[data-feedback-item]');
    const errorType = panel.querySelector('[data-feedback-error-type]');
    const comment = panel.querySelector('[data-feedback-comment]');
    const output = panel.querySelector('[data-feedback-json]');
    const render = () => {
      const selected = item.selectedOptions[0];
      const payload = selected ? JSON.parse(selected.dataset.feedbackPayload || '{}') : {};
      payload.image_id = panel.dataset.imageId || 'unknown_image';
      payload.error_type = errorType.value;
      payload.comment = comment.value || '';
      payload.created_at = new Date().toISOString();
      output.value = JSON.stringify(payload);
    };
    item.addEventListener('change', render);
    errorType.addEventListener('change', render);
    comment.addEventListener('input', render);
    render();
  }
})();
</script>"""


def _image_tag(image_ref: str, width: int, height: int) -> str:
    if image_ref:
        return f'<img src="{escape(image_ref)}" width="{width}" height="{height}" alt="">'
    return f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}"><rect width="100%" height="100%" fill="#f7f7f2"/></svg>'


def _overlay_svg(prediction: dict[str, Any], width: int, height: int) -> str:
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    for detection in prediction.get("detections", []):
        x1, y1, x2, y2 = [float(value) for value in detection["box"]]
        color = "#1f77b4" if int(detection.get("class_id", -1)) == 0 else "#2ca02c"
        label = escape(str(detection.get("name") or detection.get("class_id")))
        parts.append(
            f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{color}" stroke-width="2">'
            f"<title>{label} {detection.get('confidence')}</title></rect>"
        )
    for container in prediction.get("containers", []):
        x1, y1, x2, y2 = [float(value) for value in container["box"]]
        parts.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#d62728" stroke-width="3"/>')
    for analysis in prediction.get("container_analysis", []):
        for target in analysis.get("removal_targets", []):
            x, y = [float(value) for value in target["remove_center"]]
            reasons = _target_reasons(target)
            title = f"<title>{escape('; '.join(reasons))}</title>" if reasons else ""
            parts.append(
                f'<path d="M {x-8} {y} L {x+8} {y} M {x} {y-8} L {x} {y+8}" stroke="#ff7f0e" stroke-width="3">'
                f"{title}</path>"
            )
    parts.append("</svg>")
    return "\n".join(parts)


def _analysis_table(prediction: dict[str, Any]) -> str:
    rows = []
    for analysis in prediction.get("container_analysis", []):
        matrix = analysis.get("matrix", [])
        rows.append(
            {
                "index": analysis.get("index"),
                "box": analysis.get("box"),
                "targets": len(analysis.get("removal_targets", [])),
                "rows": len(matrix),
                "cols": len(matrix[0]) if matrix else 0,
            }
        )
    return _table(rows)


def _targets_table(prediction: dict[str, Any]) -> str:
    rows = []
    for analysis in prediction.get("container_analysis", []):
        for target in analysis.get("removal_targets", []):
            rows.append(
                {
                    "container": analysis.get("index"),
                    "row": target.get("row"),
                    "col": target.get("col"),
                    "remove_center": target.get("remove_center"),
                    "review": bool(target.get("human_review_required") or target.get("review_required")),
                    "blocked": bool(target.get("blocked") or _safety_allowed(target) is False),
                    "reasons": "; ".join(_target_reasons(target)),
                }
            )
    return _table(rows)


def _target_reasons(target: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in ["reasons", "review_reasons", "block_reasons", "blocked_reasons"]:
        value = target.get(key)
        if isinstance(value, list):
            reasons.extend(str(item) for item in value)
        elif value:
            reasons.append(str(value))
    safety = target.get("safety_decision")
    if isinstance(safety, dict):
        value = safety.get("reasons")
        if isinstance(value, list):
            reasons.extend(str(item) for item in value)
        elif value:
            reasons.append(str(value))
        result = safety.get("result")
        if result:
            reasons.append(str(result))
    if target.get("human_review_required") or target.get("review_required"):
        reasons.append("human_review_required")
    return sorted(dict.fromkeys(reason for reason in reasons if reason))


def _safety_allowed(target: dict[str, Any]) -> bool | None:
    safety = target.get("safety_decision")
    if isinstance(safety, dict) and "allowed" in safety:
        return bool(safety.get("allowed"))
    return None


def _detections_table(prediction: dict[str, Any]) -> str:
    rows = [
        {
            "class": detection.get("name") or detection.get("class_id"),
            "confidence": detection.get("confidence"),
            "box": detection.get("box"),
        }
        for detection in prediction.get("detections", [])
    ]
    return _table(rows)


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    columns = list(rows[0].keys())
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
