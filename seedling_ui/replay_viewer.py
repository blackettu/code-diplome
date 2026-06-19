from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path

from seedling_sim.renderers import render_scene_svg
from seedling_sim.replay import ReplayLog
from seedling_sim.schemas import SimScene


def write_replay_viewer(scene_path: str | Path, replay_path: str | Path, output_path: str | Path) -> None:
    scene = SimScene.from_json(scene_path)
    replay = ReplayLog.from_json(replay_path)
    html = render_replay_viewer_html(scene, replay)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def render_replay_viewer_html(scene: SimScene, replay: ReplayLog) -> str:
    summary = _safety_summary_table(replay)
    timeline = _timeline_table(replay)
    controls = _timeline_controls(len(replay.steps))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(replay.replay_id)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2528; }}
    main {{ display: grid; grid-template-columns: minmax(360px, 1fr) minmax(420px, 0.8fr); gap: 24px; align-items: start; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    td, th {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f2f2ed; }}
    tr.is-active {{ background: #fff4d6; }}
    .controls {{ display: flex; gap: 8px; margin: 0 0 12px; align-items: center; }}
    .controls button {{ border: 1px solid #b8c0c8; background: #fff; padding: 6px 10px; cursor: pointer; }}
    .controls output {{ margin-left: auto; font-variant-numeric: tabular-nums; }}
    pre {{ white-space: pre-wrap; margin: 0; }}
  </style>
</head>
<body>
  <h1>{escape(replay.replay_id)}</h1>
  <main>
    <section>{render_scene_svg(scene)}</section>
    <section>
      <h2>Safety Summary</h2>
      {summary}
      <h2>Timeline</h2>
      {controls}
      {timeline}
    </section>
  </main>
  <script>
    const rows = Array.from(document.querySelectorAll("[data-timeline-row]"));
    const output = document.querySelector("[data-step-output]");
    let index = rows.length ? 0 : -1;
    let timer = null;
    function setStep(next) {{
      if (!rows.length) return;
      index = Math.max(0, Math.min(rows.length - 1, next));
      rows.forEach((row, rowIndex) => row.classList.toggle("is-active", rowIndex === index));
      if (output) output.value = `${{index + 1}} / ${{rows.length}}`;
    }}
    function stop() {{
      if (timer) window.clearInterval(timer);
      timer = null;
    }}
    document.querySelector("[data-action='reset']")?.addEventListener("click", () => {{ stop(); setStep(0); }});
    document.querySelector("[data-action='step']")?.addEventListener("click", () => {{ stop(); setStep(index + 1); }});
    document.querySelector("[data-action='play']")?.addEventListener("click", () => {{
      stop();
      timer = window.setInterval(() => {{
        if (index >= rows.length - 1) {{ stop(); return; }}
        setStep(index + 1);
      }}, 650);
    }});
    document.querySelector("[data-action='pause']")?.addEventListener("click", stop);
    setStep(index);
  </script>
</body>
</html>
"""


def _timeline_controls(step_count: int) -> str:
    return f"""
      <div class="controls" data-replay-controls>
        <button type="button" data-action="reset">Reset</button>
        <button type="button" data-action="step">Step</button>
        <button type="button" data-action="play">Play</button>
        <button type="button" data-action="pause">Pause</button>
        <output data-step-output>{'0 / 0' if step_count == 0 else f'1 / {step_count}'}</output>
      </div>
    """


def _safety_summary_table(replay: ReplayLog) -> str:
    reason_counts: Counter[str] = Counter()
    allowed_steps = 0
    blocked_steps = 0
    review_steps = 0
    for step in replay.steps:
        payload = step.payload if isinstance(step.payload, dict) else {}
        safety = payload.get("safety_decision") if isinstance(payload.get("safety_decision"), dict) else {}
        reasons = _step_reasons(step.event_type, payload, safety)
        reason_counts.update(reasons)
        event = str(payload.get("event") or "")
        if safety.get("allowed") is True:
            allowed_steps += 1
        if safety.get("allowed") is False or payload.get("ok") is False:
            blocked_steps += 1
        if event == "review" or "operator_confirmation_required" in reasons or "BLOCK_REVIEW_REQUIRED" in reasons:
            review_steps += 1
    rows = [
        {"metric": "steps", "value": len(replay.steps)},
        {"metric": "allowed_steps", "value": allowed_steps},
        {"metric": "blocked_steps", "value": blocked_steps},
        {"metric": "review_steps", "value": review_steps},
        {
            "metric": "reason_counts",
            "value": "; ".join(f"{reason}: {count}" for reason, count in sorted(reason_counts.items())),
        },
    ]
    return _table(rows)


def _step_reasons(event_type: str, payload: dict[str, object], safety: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    raw_reasons = safety.get("reasons")
    if isinstance(raw_reasons, list):
        reasons.extend(str(reason) for reason in raw_reasons)
    elif raw_reasons:
        reasons.append(str(raw_reasons))
    if safety.get("result"):
        reasons.append(str(safety["result"]))
    event = payload.get("event")
    if event_type == "rl_step" and event:
        reasons.append(str(event))
    return sorted(dict.fromkeys(reason for reason in reasons if reason))


def _timeline_table(replay: ReplayLog) -> str:
    rows = []
    for step in replay.steps:
        safety = step.payload.get("safety_decision", {}) if isinstance(step.payload, dict) else {}
        rows.append(
            {
                "step": step.step_index,
                "event": step.event_type,
                "allowed": safety.get("allowed"),
                "decision": safety.get("result"),
                "reasons": safety.get("reasons"),
                "message": step.payload.get("message") if isinstance(step.payload, dict) else "",
            }
        )
    if not rows:
        return "<p>No replay steps.</p>"
    return _table(rows, row_attr="data-timeline-row")


def _table(rows: list[dict[str, object]], row_attr: str | None = None) -> str:
    if not rows:
        return "<p>No rows.</p>"
    columns = list(rows[0].keys())
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body_rows = []
    for index, row in enumerate(rows):
        attr = f' {row_attr}="{index}"' if row_attr else ""
        cells = "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        body_rows.append(f"<tr{attr}>{cells}</tr>")
    body = "\n".join(body_rows)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
