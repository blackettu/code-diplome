from __future__ import annotations

import re
from pathlib import Path
from typing import Any


CHANGELOG_CANDIDATES = (
    "DATASET_CHANGELOG.md",
    "dataset_changelog.md",
    "CHANGELOG.md",
    "manifests/DATASET_CHANGELOG.md",
    "manifests/dataset_changelog.md",
)
REQUIRED_FIELDS = (
    "Status",
    "Date",
    "Author",
    "Source dataset root",
    "Ontology version",
    "Annotation guide version",
)
REQUIRED_SECTIONS = ("Added", "Changed", "Removed", "Known Issues", "Validation")


def find_dataset_changelog(dataset_root: str | Path) -> Path | None:
    root = Path(dataset_root)
    for relative in CHANGELOG_CANDIDATES:
        candidate = root / relative
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def audit_dataset_changelog(
    changelog_path: str | Path | None = None,
    *,
    dataset_root: str | Path | None = None,
    dataset_version: str,
) -> dict[str, Any]:
    path = Path(changelog_path) if changelog_path else None
    if path is None and dataset_root is not None:
        path = find_dataset_changelog(dataset_root)
    errors: list[str] = []
    warnings: list[str] = []
    if path is None:
        return {
            "ok": False,
            "path": None,
            "dataset_version": dataset_version,
            "section_found": False,
            "fields": {},
            "sections": {},
            "errors": [f"dataset changelog not found for {dataset_version}"],
            "warnings": [],
        }
    if not path.exists():
        return {
            "ok": False,
            "path": str(path),
            "dataset_version": dataset_version,
            "section_found": False,
            "fields": {},
            "sections": {},
            "errors": [f"dataset changelog file does not exist: {path}"],
            "warnings": [],
        }

    text = path.read_text(encoding="utf-8-sig")
    section = _version_section(text, dataset_version)
    if section is None:
        return {
            "ok": False,
            "path": str(path),
            "dataset_version": dataset_version,
            "section_found": False,
            "fields": {},
            "sections": {},
            "errors": [f"missing changelog section: ## {dataset_version}"],
            "warnings": [],
        }

    fields = _fields(section)
    for field in REQUIRED_FIELDS:
        value = fields.get(field)
        if not value or _looks_like_placeholder(value):
            errors.append(f"{dataset_version}: missing or placeholder changelog field {field!r}")
    status = fields.get("Status")
    if status and status not in {"draft", "frozen", "archived"}:
        errors.append(f"{dataset_version}: changelog Status must be draft, frozen or archived")

    sections = {name: _section_body(section, name) for name in REQUIRED_SECTIONS}
    for name, body in sections.items():
        if body is None:
            errors.append(f"{dataset_version}: missing changelog subsection {name!r}")
    changed_sections = [
        name
        for name in ("Added", "Changed", "Removed", "Known Issues")
        if _has_content(sections.get(name))
    ]
    if not changed_sections:
        warnings.append(f"{dataset_version}: changelog has no non-empty Added/Changed/Removed/Known Issues entries")
    if not _has_content(sections.get("Validation")):
        warnings.append(f"{dataset_version}: changelog Validation section is empty")

    return {
        "ok": not errors,
        "path": str(path),
        "dataset_version": dataset_version,
        "section_found": True,
        "fields": fields,
        "sections": {name: _has_content(body) for name, body in sections.items()},
        "errors": errors,
        "warnings": warnings,
    }


def _version_section(text: str, dataset_version: str) -> str | None:
    heading = re.compile(rf"^##\s+{re.escape(dataset_version)}\s*$", re.MULTILINE)
    match = heading.search(text)
    if match is None:
        return None
    next_heading = re.compile(r"^##\s+", re.MULTILINE).search(text, match.end())
    end = next_heading.start() if next_heading else len(text)
    return text[match.end() : end]


def _fields(section: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in section.splitlines():
        match = re.match(r"^\s*-\s*([^:]+):\s*(.*)\s*$", line)
        if match:
            result[match.group(1).strip()] = match.group(2).strip()
    return result


def _section_body(section: str, name: str) -> str | None:
    heading = re.compile(rf"^###\s+{re.escape(name)}\s*$", re.MULTILINE)
    match = heading.search(section)
    if match is None:
        return None
    next_heading = re.compile(r"^###\s+", re.MULTILINE).search(section, match.end())
    end = next_heading.start() if next_heading else len(section)
    return section[match.end() : end]


def _has_content(body: str | None) -> bool:
    if body is None:
        return False
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in {"-", "- TODO", "- TBD", "- N/A"}:
            continue
        if stripped.startswith("-") and not stripped.strip("- ").strip():
            continue
        return True
    return False


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered or lowered in {"-", "todo", "tbd", "n/a", "na"}:
        return True
    return lowered in {"draft / frozen / archived", "draft/frozen/archived"}
