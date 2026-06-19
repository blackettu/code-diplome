from __future__ import annotations

import csv
import getpass
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from .changelog import audit_dataset_changelog, find_dataset_changelog


DATASET_REGISTRY_VERSION = "dataset_registry_v0_1"
ALLOWED_STATUS = {"draft", "frozen", "archived"}
DEFAULT_DATASET_REGISTRY_PATH = Path("configs/datasets/dataset_registry.yaml")


@dataclass(frozen=True)
class DatasetRegistryEntry:
    dataset_version: str
    dataset_root: str
    ontology_version: str
    annotation_guide_version: str = "annotation_guide_v0_1"
    split_version: str | None = None
    calibration_version: str | None = None
    changelog_path: str | None = None
    hashes: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = field(default_factory=getpass.getuser)
    status: str = "draft"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_STATUS:
            raise ValueError(f"dataset status must be one of {sorted(ALLOWED_STATUS)}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetRegistryEntry":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetRegistry:
    version: str = DATASET_REGISTRY_VERSION
    entries: list[DatasetRegistryEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetRegistry":
        return cls(
            version=str(data.get("version", DATASET_REGISTRY_VERSION)),
            entries=[DatasetRegistryEntry.from_dict(item) for item in data.get("entries", [])],
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetRegistry":
        registry_path = Path(path)
        if not registry_path.exists():
            return cls()
        return cls.from_dict(_load_mapping(registry_path))

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "entries": [entry.to_dict() for entry in self.entries]}

    def to_file(self, path: str | Path) -> None:
        _write_mapping(path, self.to_dict())

    def upsert(self, entry: DatasetRegistryEntry) -> DatasetRegistryEntry:
        self.entries = [item for item in self.entries if item.dataset_version != entry.dataset_version]
        self.entries.append(entry)
        self.entries.sort(key=lambda item: item.dataset_version)
        return entry

    def select(self, dataset_version: str) -> DatasetRegistryEntry:
        matches = [entry for entry in self.entries if entry.dataset_version == dataset_version]
        if not matches:
            raise KeyError(f"Dataset {dataset_version!r} is not registered")
        return matches[0]


def add_dataset(
    registry_path: str | Path,
    dataset_root: str | Path,
    name: str,
    ontology_version: str,
    annotation_guide_version: str = "annotation_guide_v0_1",
    split_version: str | None = None,
    calibration_version: str | None = None,
    changelog_path: str | Path | None = None,
    status: str = "draft",
    metadata: dict[str, Any] | None = None,
) -> DatasetRegistryEntry:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    registry = DatasetRegistry.from_file(registry_path)
    changelog = Path(changelog_path) if changelog_path else find_dataset_changelog(root)
    stored_changelog_path = _stored_changelog_path(root, changelog) if changelog else None
    hashes = collect_dataset_hashes(root)
    if changelog and changelog.exists():
        hashes[stored_changelog_path or str(changelog)] = _sha256(changelog)
    entry = DatasetRegistryEntry(
        dataset_version=name,
        dataset_root=str(root.resolve()),
        ontology_version=ontology_version,
        annotation_guide_version=annotation_guide_version,
        split_version=split_version,
        calibration_version=calibration_version,
        changelog_path=stored_changelog_path,
        hashes=hashes,
        status=status,
        metadata=metadata or {},
    )
    registry.upsert(entry)
    registry.to_file(registry_path)
    return entry


def validate_dataset_registry(
    registry_path: str | Path,
    dataset: str | None = None,
) -> dict[str, Any]:
    registry = DatasetRegistry.from_file(registry_path)
    entries = [registry.select(dataset)] if dataset else registry.entries
    errors: list[str] = []
    warnings: list[str] = []
    validated = []

    for entry in entries:
        root = Path(entry.dataset_root)
        entry_errors: list[str] = []
        entry_warnings: list[str] = []
        if entry.status not in ALLOWED_STATUS:
            entry_errors.append(f"invalid status {entry.status!r}")
        if not root.exists():
            entry_errors.append(f"dataset root does not exist: {root}")
            changelog_audit = {"ok": False, "path": None, "errors": [], "warnings": []}
        else:
            if not entry.hashes:
                entry_warnings.append("no hashes are registered")
            for relative_path, expected_hash in sorted(entry.hashes.items()):
                file_path = root / relative_path
                if not file_path.exists():
                    entry_errors.append(f"missing hashed file: {relative_path}")
                    continue
                actual_hash = _sha256(file_path)
                if actual_hash != expected_hash:
                    entry_errors.append(f"hash mismatch for {relative_path}")
            changelog_path = _resolve_changelog_path(root, entry.changelog_path)
            changelog_audit = audit_dataset_changelog(
                changelog_path,
                dataset_root=root,
                dataset_version=entry.dataset_version,
            )
            entry_errors.extend(changelog_audit.get("errors", []))
            entry_warnings.extend(changelog_audit.get("warnings", []))
        errors.extend(f"{entry.dataset_version}: {error}" for error in entry_errors)
        warnings.extend(f"{entry.dataset_version}: {warning}" for warning in entry_warnings)
        validated.append(
            {
                "dataset_version": entry.dataset_version,
                "status": entry.status,
                "hashes": len(entry.hashes),
                "changelog": changelog_audit,
                "ok": not entry_errors,
            }
        )

    return {
        "ok": not errors,
        "registry": str(Path(registry_path)),
        "datasets": validated,
        "errors": errors,
        "warnings": warnings,
    }


def write_dataset_summary(
    registry_path: str | Path,
    dataset: str,
    output_path: str | Path,
) -> dict[str, Any]:
    registry = DatasetRegistry.from_file(registry_path)
    entry = registry.select(dataset)
    root = Path(entry.dataset_root)
    manifest_stats = _manifest_stats(root)
    payload = {
        "dataset_version": entry.dataset_version,
        "dataset_root": entry.dataset_root,
        "ontology_version": entry.ontology_version,
        "annotation_guide_version": entry.annotation_guide_version,
        "split_version": entry.split_version,
        "calibration_version": entry.calibration_version,
        "changelog": audit_dataset_changelog(
            _resolve_changelog_path(root, entry.changelog_path),
            dataset_root=root,
            dataset_version=entry.dataset_version,
        ),
        "status": entry.status,
        "hashes": len(entry.hashes),
        "manifest": manifest_stats,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".html", ".htm"}:
        output.write_text(_summary_html(payload), encoding="utf-8")
    else:
        output.write_text(_summary_markdown(payload), encoding="utf-8")
    return payload


def collect_dataset_hashes(dataset_root: str | Path) -> dict[str, str]:
    root = Path(dataset_root)
    candidates: set[Path] = set()
    for relative in [
        "data.yaml",
        "dataset_version.json",
        "DATASET_CHANGELOG.md",
        "dataset_changelog.md",
        "CHANGELOG.md",
        "image_manifest.csv",
        "split_manifest.csv",
        "cell_annotations.jsonl",
        "action_points.jsonl",
        "manifests/DATASET_CHANGELOG.md",
        "manifests/dataset_changelog.md",
        "manifests/image_manifest.csv",
        "manifests/split_manifest.csv",
        "manifests/cell_annotations.jsonl",
        "manifests/action_points.jsonl",
    ]:
        path = root / relative
        if path.exists() and path.is_file():
            candidates.add(path)
    manifests_dir = root / "manifests"
    if manifests_dir.exists():
        for path in manifests_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl", ".yaml", ".yml"}:
                candidates.add(path)
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(candidates)
    }


def _manifest_stats(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifests" / "image_manifest.csv"
    if not manifest_path.exists():
        manifest_path = root / "image_manifest.csv"
    if not manifest_path.exists():
        return {"present": False}
    split_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    class_counts_by_split: dict[str, dict[str, int]] = {}
    rows = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            split = row.get("split") or "unspecified"
            split_counts[split] = split_counts.get(split, 0) + 1
            label_path = _label_path_for_manifest_row(root, row)
            if label_path is None:
                continue
            for class_id, count in _yolo_class_counts(label_path).items():
                class_counts[class_id] = class_counts.get(class_id, 0) + count
                by_split = class_counts_by_split.setdefault(split, {})
                by_split[class_id] = by_split.get(class_id, 0) + count
    return {
        "present": True,
        "path": str(manifest_path),
        "images": rows,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "class_counts_by_split": class_counts_by_split,
        "cell_annotations": _jsonl_distribution(root, ["manifests/cell_annotations.jsonl", "cell_annotations.jsonl"], "state"),
        "action_points": _jsonl_distribution(root, ["manifests/action_points.jsonl", "action_points.jsonl"], "target_type"),
    }


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Dataset Summary: {payload['dataset_version']}",
        "",
        f"- Root: `{payload['dataset_root']}`",
        f"- Status: `{payload['status']}`",
        f"- Ontology: `{payload['ontology_version']}`",
        f"- Annotation guide: `{payload['annotation_guide_version']}`",
        f"- Split version: `{payload['split_version']}`",
        f"- Calibration version: `{payload['calibration_version']}`",
        f"- Changelog: `{payload['changelog'].get('path')}`",
        f"- Registered hashes: `{payload['hashes']}`",
        "",
        "## Manifest",
    ]
    manifest = payload["manifest"]
    if not manifest.get("present"):
        lines.append("No image manifest found.")
    else:
        lines.append(f"- Images: `{manifest['images']}`")
        for split, count in sorted(manifest["split_counts"].items()):
            lines.append(f"- {split}: `{count}`")
        lines.extend(["", "## Class Distribution"])
        if manifest.get("class_counts"):
            for class_id, count in sorted(manifest["class_counts"].items(), key=lambda item: item[0]):
                lines.append(f"- class {class_id}: `{count}`")
            lines.append("")
            lines.append("| Split | Class | Count |")
            lines.append("|---|---:|---:|")
            for split, counts in sorted(manifest.get("class_counts_by_split", {}).items()):
                for class_id, count in sorted(counts.items(), key=lambda item: item[0]):
                    lines.append(f"| {split} | {class_id} | {count} |")
        else:
            lines.append("No YOLO label distribution found.")
        lines.extend(["", "## Cell Distribution"])
        cell_annotations = manifest.get("cell_annotations", {})
        if cell_annotations.get("present"):
            lines.append(f"- Rows: `{cell_annotations['rows']}`")
            for state, count in sorted(cell_annotations.get("counts", {}).items()):
                lines.append(f"- {state}: `{count}`")
        else:
            lines.append("No cell annotation distribution found.")
        lines.extend(["", "## Action Point Distribution"])
        action_points = manifest.get("action_points", {})
        if action_points.get("present"):
            lines.append(f"- Rows: `{action_points['rows']}`")
            for target_type, count in sorted(action_points.get("counts", {}).items()):
                lines.append(f"- {target_type}: `{count}`")
        else:
            lines.append("No action point distribution found.")
    lines.append("")
    return "\n".join(lines)


def _summary_html(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    sections = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Dataset Summary: {escape(str(payload['dataset_version']))}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;line-height:1.4;color:#1f2933}",
        "table{border-collapse:collapse;margin:12px 0 24px;width:100%;max-width:900px}",
        "th,td{border:1px solid #d7dde3;padding:6px 8px;text-align:left}",
        "th{background:#f3f5f7}",
        "code{background:#eef2f5;padding:1px 4px;border-radius:3px}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Dataset Summary: {escape(str(payload['dataset_version']))}</h1>",
        "<ul>",
        f"<li>Root: <code>{escape(str(payload['dataset_root']))}</code></li>",
        f"<li>Status: <code>{escape(str(payload['status']))}</code></li>",
        f"<li>Ontology: <code>{escape(str(payload['ontology_version']))}</code></li>",
        f"<li>Annotation guide: <code>{escape(str(payload['annotation_guide_version']))}</code></li>",
        f"<li>Split version: <code>{escape(str(payload['split_version']))}</code></li>",
        f"<li>Calibration version: <code>{escape(str(payload['calibration_version']))}</code></li>",
        f"<li>Changelog: <code>{escape(str(payload['changelog'].get('path')))}</code></li>",
        f"<li>Registered hashes: <code>{escape(str(payload['hashes']))}</code></li>",
        "</ul>",
        "<h2>Manifest</h2>",
    ]
    if not manifest.get("present"):
        sections.append("<p>No image manifest found.</p>")
    else:
        sections.append(f"<p>Images: <code>{escape(str(manifest['images']))}</code></p>")
        sections.append(_count_table("Split Distribution", ["Split", "Images"], manifest["split_counts"]))
        sections.append("<h2>Class Distribution</h2>")
        if manifest.get("class_counts"):
            sections.append(_count_table("Class Counts", ["Class", "Count"], manifest["class_counts"]))
            rows = [
                [split, class_id, count]
                for split, counts in sorted(manifest.get("class_counts_by_split", {}).items())
                for class_id, count in sorted(counts.items(), key=lambda item: item[0])
            ]
            sections.append(_rows_table(["Split", "Class", "Count"], rows))
        else:
            sections.append("<p>No YOLO label distribution found.</p>")
        sections.append("<h2>Cell Distribution</h2>")
        cell_annotations = manifest.get("cell_annotations", {})
        if cell_annotations.get("present"):
            sections.append(f"<p>Rows: <code>{escape(str(cell_annotations['rows']))}</code></p>")
            sections.append(_count_table("Cell State Counts", ["State", "Count"], cell_annotations.get("counts", {})))
        else:
            sections.append("<p>No cell annotation distribution found.</p>")
        sections.append("<h2>Action Point Distribution</h2>")
        action_points = manifest.get("action_points", {})
        if action_points.get("present"):
            sections.append(f"<p>Rows: <code>{escape(str(action_points['rows']))}</code></p>")
            sections.append(_count_table("Action Target Counts", ["Target Type", "Count"], action_points.get("counts", {})))
        else:
            sections.append("<p>No action point distribution found.</p>")
    sections.extend(["</body>", "</html>"])
    return "\n".join(sections)


def _count_table(caption: str, headers: list[str], counts: dict[str, int]) -> str:
    rows = [[key, value] for key, value in sorted(counts.items(), key=lambda item: item[0])]
    table = [f"<h3>{escape(caption)}</h3>", _rows_table(headers, rows)]
    return "\n".join(table)


def _rows_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["<table>", "<thead><tr>"]
    lines.extend(f"<th>{escape(str(header))}</th>" for header in headers)
    lines.append("</tr></thead>")
    lines.append("<tbody>")
    for row in rows:
        lines.append("<tr>")
        lines.extend(f"<td>{escape(str(cell))}</td>" for cell in row)
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _label_path_for_manifest_row(root: Path, row: dict[str, str]) -> Path | None:
    for column in ["label_path", "labels_path"]:
        raw = row.get(column)
        if raw:
            path = Path(raw)
            candidate = path if path.is_absolute() else root / path
            if candidate.exists():
                return candidate
    raw_image = row.get("file_path") or row.get("path")
    if not raw_image:
        return None
    image_path = Path(raw_image)
    if not image_path.is_absolute():
        image_path = root / image_path
    candidates: list[Path] = []
    parts = list(image_path.parts)
    if "images" in parts:
        index = parts.index("images")
        label_parts = parts[:]
        label_parts[index] = "labels"
        candidates.append(Path(*label_parts).with_suffix(".txt"))
    candidates.append(root / "labels" / f"{image_path.stem}.txt")
    split = row.get("split")
    if split:
        candidates.append(root / "labels" / split / f"{image_path.stem}.txt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _yolo_class_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        class_id = stripped.split()[0]
        counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def _jsonl_distribution(root: Path, relative_paths: list[str], field_name: str) -> dict[str, Any]:
    path = next((root / relative for relative in relative_paths if (root / relative).exists()), None)
    if path is None:
        return {"present": False}
    rows = 0
    counts: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            counts["invalid_json"] = counts.get("invalid_json", 0) + 1
            continue
        if not isinstance(data, dict):
            counts["invalid_row"] = counts.get("invalid_row", 0) + 1
            continue
        rows += 1
        value = str(data.get(field_name) or "unspecified")
        counts[value] = counts.get(value, 0) + 1
    return {"present": True, "path": str(path), "rows": rows, "counts": counts}


def _load_mapping(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    text = registry_path.read_text(encoding="utf-8-sig")
    if registry_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML dataset registries") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset registry must be a mapping: {registry_path}")
    return data


def _stored_changelog_path(root: Path, changelog: Path | None) -> str | None:
    if changelog is None:
        return None
    resolved_root = root.resolve()
    resolved_changelog = changelog.resolve()
    try:
        return resolved_changelog.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_changelog)


def _resolve_changelog_path(root: Path, changelog_path: str | None) -> Path | None:
    if changelog_path:
        path = Path(changelog_path)
        return path if path.is_absolute() else root / path
    return find_dataset_changelog(root)


def _write_mapping(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    try:
        import yaml
    except ImportError:
        output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    output.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
