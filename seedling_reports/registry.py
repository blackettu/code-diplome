from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    artifact_type: str
    role: str = "output"
    command: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        artifact_type: str,
        role: str = "output",
        command: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ArtifactRecord":
        artifact_path = Path(path)
        sha, size = (None, None)
        if artifact_path.exists() and artifact_path.is_file():
            sha = _sha256(artifact_path)
            size = artifact_path.stat().st_size
        return cls(
            path=str(artifact_path),
            artifact_type=artifact_type,
            role=role,
            command=command,
            sha256=sha,
            bytes=size,
            metadata=metadata or {},
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRecord":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    command: str
    run_dir: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_path: str | None = None
    snapshot_path: str | None = None
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        payload = dict(data)
        payload["artifacts"] = [ArtifactRecord.from_dict(item) for item in payload.get("artifacts", [])]
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


class ExperimentRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.runs: list[RunRecord] = []
        if self.path.exists():
            self.runs = self._load(self.path)

    def add_run(self, run: RunRecord) -> None:
        self.runs = [existing for existing in self.runs if existing.run_id != run.run_id]
        self.runs.append(run)

    def to_json(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"runs": [run.to_dict() for run in self.runs]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def to_csv(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "run_id",
                    "command",
                    "run_dir",
                    "artifact_type",
                    "role",
                    "path",
                    "sha256",
                    "bytes",
                ],
            )
            writer.writeheader()
            for run in self.runs:
                for artifact in run.artifacts:
                    writer.writerow(
                        {
                            "run_id": run.run_id,
                            "command": run.command,
                            "run_dir": run.run_dir,
                            "artifact_type": artifact.artifact_type,
                            "role": artifact.role,
                            "path": artifact.path,
                            "sha256": artifact.sha256,
                            "bytes": artifact.bytes,
                        }
                    )

    @staticmethod
    def _load(path: Path) -> list[RunRecord]:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError(f"Experiment registry must be a JSON object: {path}")
        return [RunRecord.from_dict(item) for item in data.get("runs", [])]


def write_run_registry(run_dir: str | Path, command: str, artifact_paths: list[str | Path]) -> RunRecord:
    directory = Path(run_dir)
    artifacts = [
        ArtifactRecord.from_path(path, artifact_type=_artifact_type(Path(path)), command=command)
        for path in artifact_paths
    ]
    return write_run_registry_records(directory, command, artifacts, run_id=directory.name or "run")


def write_run_registry_records(
    run_dir: str | Path,
    command: str,
    artifacts: list[ArtifactRecord],
    metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> RunRecord:
    directory = Path(run_dir)
    run = RunRecord(
        run_id=run_id or f"{directory.name or 'run'}_{_command_slug(command)}",
        command=command,
        run_dir=str(directory),
        config_path=str(directory / "config.yaml") if (directory / "config.yaml").exists() else None,
        snapshot_path=str(directory / "run_snapshot.json") if (directory / "run_snapshot.json").exists() else None,
        artifacts=artifacts,
        metadata=metadata or {},
    )
    registry = ExperimentRegistry(directory / "artifact_registry.json")
    registry.add_run(run)
    registry.to_json()
    registry.to_csv(directory / "artifact_registry.csv")
    return run


def _command_slug(command: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in command).strip("_") or "command"


def _artifact_type(path: Path) -> str:
    name = path.name
    if name.endswith(".json"):
        return name.removesuffix(".json")
    if name.endswith(".csv"):
        return name.removesuffix(".csv")
    if name.endswith(".yaml") or name.endswith(".yml"):
        return "config"
    if name.endswith(".html"):
        return "html_report"
    return path.suffix.lstrip(".") or "artifact"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
