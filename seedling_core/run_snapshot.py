from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_snapshot(packages: list[str] | None = None) -> dict[str, Any]:
    package_names = packages or [
        "ultralytics",
        "opencv-python",
        "numpy",
        "Pillow",
        "pillow-heif",
        "PyYAML",
    ]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in package_names},
    }


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_yaml(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:
        write_json(output.with_suffix(".json"), data)
        return
    output.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def save_run_snapshot(
    output_dir: str | Path,
    config: dict[str, Any],
    command: str,
    command_args: dict[str, Any] | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_config = output / "config.yaml"
    snapshot_path = output / "run_snapshot.json"
    digest = config_hash(config)
    write_yaml(snapshot_config, config)
    write_json(
        snapshot_path,
        {
            "command": command,
            "command_args": command_args or {},
            "config": config,
            "config_hash": digest,
            "environment": environment_snapshot(),
        },
    )
    register_run_artifacts(output, command, config=config, command_args=command_args)


def save_command_snapshot(
    output_dir: str | Path,
    command: str,
    *,
    command_args: dict[str, Any] | None = None,
    input_paths: Iterable[str | Path | None] | None = None,
    output_paths: Iterable[str | Path | None] | None = None,
    metadata: dict[str, Any] | None = None,
    snapshot_name: str = "run_snapshot.json",
) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = [str(path) for path in _unique_paths(input_paths or [])]
    outputs = [str(path) for path in _unique_paths(output_paths or [])]
    snapshot_config = {
        "command_args": command_args or {},
        "inputs": inputs,
        "outputs": outputs,
        "metadata": metadata or {},
    }
    snapshot_path = output / snapshot_name
    write_json(
        snapshot_path,
        {
            "schema_version": "command_run_snapshot_v0_1",
            "command": command,
            "command_args": command_args or {},
            "config": snapshot_config,
            "config_hash": config_hash(snapshot_config),
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata or {},
            "environment": environment_snapshot(),
        },
    )
    return str(snapshot_path)


def register_run_artifacts(
    output_dir: str | Path,
    command: str,
    *,
    config: dict[str, Any] | None = None,
    command_args: dict[str, Any] | None = None,
    input_paths: Iterable[str | Path | None] | None = None,
    output_paths: Iterable[str | Path | None] | None = None,
    metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
    include_snapshot: bool = True,
) -> str | None:
    """Update a run-local artifact registry without making legacy commands fail."""
    output = Path(output_dir)
    try:
        from seedling_reports.registry import ArtifactRecord, write_run_registry_records

        snapshot_config = output / "config.yaml"
        snapshot_path = output / "run_snapshot.json"
        artifacts: list[ArtifactRecord] = []

        source_config = _command_config_path(command_args)
        if source_config is not None and not _same_path(source_config, snapshot_config):
            artifacts.append(
                ArtifactRecord.from_path(
                    source_config,
                    artifact_type="config",
                    role="input",
                    command=command,
                    metadata={"source": "command_args.config"},
                )
            )
        for path in _unique_paths(input_paths or []):
            if path is None:
                continue
            if source_config is not None and _same_path(path, source_config):
                continue
            artifacts.append(
                ArtifactRecord.from_path(
                    path,
                    artifact_type=_artifact_type(Path(path)),
                    role="input",
                    command=command,
                )
            )
        if include_snapshot:
            artifacts.extend(
                [
                    ArtifactRecord.from_path(snapshot_config, artifact_type="config", role="output", command=command),
                    ArtifactRecord.from_path(snapshot_path, artifact_type="run_snapshot", role="output", command=command),
                ]
            )
        for path in _unique_paths(output_paths or []):
            if path is None:
                continue
            if include_snapshot and (_same_path(path, snapshot_config) or _same_path(path, snapshot_path)):
                continue
            artifacts.append(
                ArtifactRecord.from_path(
                    path,
                    artifact_type=_artifact_type(Path(path)),
                    role="output",
                    command=command,
                )
            )

        record_metadata = dict(metadata or {})
        if command_args is not None:
            record_metadata.setdefault("command_args", command_args)
        if config is not None:
            record_metadata.setdefault("config_hash", config_hash(config))
        write_run_registry_records(
            output,
            command,
            artifacts,
            metadata=record_metadata,
            run_id=run_id or f"{output.name or 'run'}_{_command_slug(command)}_snapshot",
        )
        return str(output / "artifact_registry.json")
    except Exception:
        # Registry creation is useful for orchestration but must not break legacy experiment commands.
        return None


def _command_config_path(command_args: dict[str, Any] | None) -> Path | None:
    if not command_args:
        return None
    value = command_args.get("config")
    if value is None or value == "":
        return None
    return Path(str(value))


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _command_slug(command: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in command).strip("_") or "command"


def _artifact_type(path: Path) -> str:
    name = path.name
    if path.is_dir():
        return "directory"
    if name.endswith(".json"):
        return name.removesuffix(".json")
    if name.endswith(".jsonl"):
        return name.removesuffix(".jsonl")
    if name.endswith(".csv"):
        return name.removesuffix(".csv")
    if name.endswith(".yaml") or name.endswith(".yml"):
        return "config"
    if name.endswith(".html"):
        return "html_report"
    if name.endswith(".md"):
        return "markdown_report"
    return path.suffix.lstrip(".") or "artifact"


def _unique_paths(paths: Iterable[str | Path | None]) -> list[str | Path]:
    seen: set[str] = set()
    result: list[str | Path] = []
    for path in paths:
        if path is None:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result
