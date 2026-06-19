from __future__ import annotations

from html import escape
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from seedling_sim.domain_randomization import DomainRandomizationConfig
from seedling_sim.replay import ReplayLog
from seedling_sim.schemas import SimScene


def render_scene_svg(scene: SimScene, width_px: int = 720) -> str:
    tray_w = scene.grid_cols * scene.cell_size_mm[0]
    tray_h = scene.grid_rows * scene.cell_size_mm[1]
    scale = width_px / tray_w if tray_w else 1.0
    height_px = max(1, int(tray_h * scale))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {tray_w:.3f} {tray_h:.3f}" width="{width_px}" height="{height_px}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#f7f7f2"/>',
    ]
    for row in range(scene.grid_rows + 1):
        y = row * scene.cell_size_mm[1]
        parts.append(f'<line x1="0" y1="{y:.3f}" x2="{tray_w:.3f}" y2="{y:.3f}" stroke="#c7c7bd" stroke-width="0.35"/>')
    for col in range(scene.grid_cols + 1):
        x = col * scene.cell_size_mm[0]
        parts.append(f'<line x1="{x:.3f}" y1="0" x2="{x:.3f}" y2="{tray_h:.3f}" stroke="#c7c7bd" stroke-width="0.35"/>')
    for plant in scene.plants:
        color = _plant_color(plant.class_name, plant.removed)
        x, y = plant.position_mm
        label = escape(plant.class_name)
        parts.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{plant.radius_mm:.3f}" fill="{color}" stroke="#222" stroke-width="0.25">'
            f"<title>{escape(plant.object_id)} {label}</title></circle>"
        )
    for target in scene.targets:
        x, y = target.point_mm
        stroke = "#bd2d2d" if not target.processed else "#555"
        parts.append(
            f'<path d="M {x-3:.3f} {y:.3f} L {x+3:.3f} {y:.3f} M {x:.3f} {y-3:.3f} L {x:.3f} {y+3:.3f}" '
            f'stroke="{stroke}" stroke-width="0.8"><title>{escape(target.target_id)}</title></path>'
        )
    rx, ry = scene.robot_position_mm[:2]
    parts.append(f'<circle cx="{rx:.3f}" cy="{ry:.3f}" r="2.5" fill="none" stroke="#1f4e79" stroke-width="1.0"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def render_scene_html(scene: SimScene, replay: ReplayLog | None = None) -> str:
    replay_rows = ""
    if replay is not None:
        replay_rows = "\n".join(
            f"<tr><td>{step.step_index}</td><td>{escape(step.event_type)}</td><td><pre>{escape(str(step.payload))}</pre></td></tr>"
            for step in replay.steps
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(scene.scene_id)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2528; }}
    main {{ display: grid; grid-template-columns: minmax(360px, 1fr) 420px; gap: 24px; align-items: start; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    td, th {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    pre {{ white-space: pre-wrap; margin: 0; }}
  </style>
</head>
<body>
  <h1>{escape(scene.scene_id)}</h1>
  <main>
    <section>{render_scene_svg(scene)}</section>
    <section>
      <h2>Scene</h2>
      <table>
        <tr><th>Grid</th><td>{scene.grid_rows} x {scene.grid_cols}</td></tr>
        <tr><th>Plants</th><td>{len(scene.plants)}</td></tr>
        <tr><th>Targets</th><td>{len(scene.targets)}</td></tr>
        <tr><th>Step</th><td>{scene.step_index}</td></tr>
      </table>
      <h2>Replay</h2>
      <table><tr><th>#</th><th>Event</th><th>Payload</th></tr>{replay_rows}</table>
    </section>
  </main>
</body>
</html>
"""


def render_scene_png(
    scene: SimScene,
    output_path: str | Path,
    width_px: int = 720,
    randomization: DomainRandomizationConfig | None = None,
    seed: int | None = None,
) -> None:
    randomization = randomization or DomainRandomizationConfig()
    rng = random.Random(seed)
    tray_w = scene.grid_cols * scene.cell_size_mm[0]
    tray_h = scene.grid_rows * scene.cell_size_mm[1]
    scale = width_px / tray_w if tray_w else 1.0
    drift_x_px = randomization.calibration_drift_mm[0] * scale
    drift_y_px = randomization.calibration_drift_mm[1] * scale
    height_px = max(1, int(tray_h * scale))
    image = Image.new("RGB", (width_px, height_px), randomization.background_rgb)
    draw = ImageDraw.Draw(image)

    for row in range(scene.grid_rows + 1):
        y = row * scene.cell_size_mm[1] * scale
        draw.line([(0, y), (width_px, y)], fill=randomization.tray_line_rgb, width=1)
    for col in range(scene.grid_cols + 1):
        x = col * scene.cell_size_mm[0] * scale
        draw.line([(x, 0), (x, height_px)], fill=randomization.tray_line_rgb, width=1)

    for plant in scene.plants:
        x = plant.position_mm[0] * scale + drift_x_px + rng.uniform(-randomization.plant_jitter_px, randomization.plant_jitter_px)
        y = plant.position_mm[1] * scale + drift_y_px + rng.uniform(-randomization.plant_jitter_px, randomization.plant_jitter_px)
        radius = max(2.0, plant.radius_mm * scale)
        color = _plant_rgb(plant.class_name, plant.removed)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(35, 55, 35))
    for target in scene.targets:
        x, y = target.point_mm[0] * scale + drift_x_px, target.point_mm[1] * scale + drift_y_px
        draw.line([(x - 5, y), (x + 5, y)], fill=(190, 40, 35), width=1)
        draw.line([(x, y - 5), (x, y + 5)], fill=(190, 40, 35), width=1)

    image = ImageEnhance.Brightness(image).enhance(randomization.lighting_scale)
    if randomization.blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(randomization.blur_radius))
    if randomization.noise_std > 0:
        array = np.asarray(image, dtype=np.float32)
        noise = np.random.default_rng(seed).normal(0.0, randomization.noise_std, array.shape)
        image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _plant_color(class_name: str, removed: bool) -> str:
    if removed:
        return "#9a9a9a"
    if class_name in {"weed"}:
        return "#c56b2c"
    if class_name in {"unknown_plant", "unknown"}:
        return "#9467bd"
    return "#2e8b57"


def _plant_rgb(class_name: str, removed: bool) -> tuple[int, int, int]:
    if removed:
        return (150, 150, 150)
    if class_name == "weed":
        return (190, 100, 35)
    if class_name in {"unknown_plant", "unknown"}:
        return (145, 95, 185)
    return (40, 145, 78)
