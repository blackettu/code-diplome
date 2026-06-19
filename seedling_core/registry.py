from __future__ import annotations

import json
import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


COMPONENT_KINDS = {"detector", "segmenter", "policy", "robot", "simulator"}
MODEL_TYPES = {"detector", "segmenter", "policy", "rl_policy"}
MODEL_STATUSES = {"draft", "validated", "deprecated", "blocked"}
MODEL_SAFETY_LEVELS = {
    "offline_only",
    "simulation_only",
    "dry_run",
    "supervised",
    "production_candidate",
}
RL_POLICY_SAFETY_LEVELS = {"offline_only", "simulation_only", "dry_run"}


@dataclass(frozen=True)
class RegistryEntry:
    entry_id: str
    kind: str
    name: str
    backend: str
    uri: str | None = None
    status: str = "draft"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegistryEntry":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComponentRegistry:
    version: str
    entries: list[RegistryEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComponentRegistry":
        return cls(
            version=str(data.get("version", "registry_v0_1")),
            entries=[RegistryEntry.from_dict(item) for item in data.get("entries", [])],
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ComponentRegistry":
        data = _load_mapping(path)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "entries": [entry.to_dict() for entry in self.entries]}

    def to_file(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".json":
            output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            return
        try:
            import yaml
        except ImportError:
            output.with_suffix(".json").write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            return
        output.write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")

    def select(
        self,
        kind: str | None = None,
        name: str | None = None,
        entry_id: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> RegistryEntry:
        matches = self.filter(kind=kind, name=name, entry_id=entry_id, status=status, tags=tags)
        if not matches:
            raise KeyError("No registry entry matches selector")
        if len(matches) > 1:
            ids = ", ".join(entry.entry_id for entry in matches)
            raise ValueError(f"Selector matched multiple entries: {ids}")
        return matches[0]

    def filter(
        self,
        kind: str | None = None,
        name: str | None = None,
        entry_id: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> list[RegistryEntry]:
        tags = tags or []
        result = []
        for entry in self.entries:
            if kind is not None and entry.kind != kind:
                continue
            if name is not None and entry.name != name:
                continue
            if entry_id is not None and entry.entry_id != entry_id:
                continue
            if status is not None and entry.status != status:
                continue
            if tags and not set(tags).issubset(set(entry.tags)):
                continue
            result.append(entry)
        return result

    def validate(self, *, check_backend_imports: bool = True) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        seen_entry_ids: set[str] = set()
        seen_kind_names: set[tuple[str, str]] = set()
        for index, entry in enumerate(self.entries, 1):
            location = f"entries[{index}]/{entry.entry_id or '<missing>'}"
            if not entry.entry_id:
                errors.append(f"{location}: entry_id is required")
            if entry.entry_id in seen_entry_ids:
                errors.append(f"{location}: duplicate entry_id")
            seen_entry_ids.add(entry.entry_id)

            kind_name = (entry.kind, entry.name)
            if kind_name in seen_kind_names:
                errors.append(f"{location}: duplicate kind/name selector {entry.kind}/{entry.name}")
            seen_kind_names.add(kind_name)

            if entry.kind not in COMPONENT_KINDS:
                errors.append(f"{location}: unsupported kind {entry.kind!r}")
            if entry.status not in MODEL_STATUSES:
                errors.append(f"{location}: unsupported status {entry.status!r}")
            for field_name in ["kind", "name", "backend"]:
                if not getattr(entry, field_name):
                    errors.append(f"{location}: {field_name} is required")
            if check_backend_imports and entry.backend:
                backend_error = _backend_import_error(entry.backend)
                if backend_error:
                    errors.append(f"{location}: {backend_error}")
            if entry.kind in {"detector", "segmenter"}:
                if not entry.metadata.get("output_schema"):
                    warnings.append(f"{location}: vision component metadata.output_schema is recommended")
            if entry.kind == "policy":
                safety_level = entry.metadata.get("safety_level")
                if safety_level is None:
                    warnings.append(f"{location}: policy metadata.safety_level is recommended")
                elif safety_level not in MODEL_SAFETY_LEVELS:
                    errors.append(f"{location}: unsupported policy metadata.safety_level {safety_level!r}")
                if entry.metadata.get("requires_safety_gate") is not True:
                    warnings.append(f"{location}: policy should declare metadata.requires_safety_gate=true")
                if "rl" in entry.tags or entry.backend.endswith("RLPolicyAdapter"):
                    if safety_level not in RL_POLICY_SAFETY_LEVELS:
                        errors.append(f"{location}: rl policy component must remain offline_only/simulation_only/dry_run")
                    if entry.metadata.get("direct_hardware_access") is not False:
                        errors.append(f"{location}: rl policy component requires metadata.direct_hardware_access=false")
                    if entry.metadata.get("requires_action_mask") is not True:
                        errors.append(f"{location}: rl policy component requires metadata.requires_action_mask=true")
        return {
            "ok": not errors,
            "version": self.version,
            "entries": len(self.entries),
            "errors": errors,
            "warnings": warnings,
        }


@dataclass(frozen=True)
class ModelRegistryRecord:
    model_id: str
    model_type: str
    framework: str
    artifact_uri: str
    input_schema: str
    output_schema: str
    status: str
    safety_level: str
    config_uri: str | None = None
    metrics_uri: str | None = None
    dataset_version: str | None = None
    ontology_version: str | None = None
    calibration_requirements: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRegistryRecord":
        return cls(**data)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "model_type", str(self.model_type))
        object.__setattr__(self, "framework", str(self.framework))
        object.__setattr__(self, "artifact_uri", str(self.artifact_uri))
        object.__setattr__(self, "input_schema", str(self.input_schema))
        object.__setattr__(self, "output_schema", str(self.output_schema))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "safety_level", str(self.safety_level))
        object.__setattr__(self, "tags", [str(tag) for tag in self.tags])
        object.__setattr__(self, "calibration_requirements", dict(self.calibration_requirements))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        if self.status not in MODEL_STATUSES:
            raise ValueError(f"Unsupported model status: {self.status}")
        if self.safety_level not in MODEL_SAFETY_LEVELS:
            raise ValueError(f"Unsupported safety_level: {self.safety_level}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelRegistry:
    version: str
    models: list[ModelRegistryRecord] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRegistry":
        raw_models = data.get("models", data.get("entries", []))
        if not isinstance(raw_models, list):
            raise ValueError("Model registry requires a models list")
        return cls(
            version=str(data.get("version", "model_registry_v0_1")),
            models=[ModelRegistryRecord.from_dict(item) for item in raw_models if isinstance(item, dict)],
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelRegistry":
        return cls.from_dict(_load_mapping(path))

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "models": [model.to_dict() for model in self.models]}

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        seen: set[str] = set()
        for index, model in enumerate(self.models, 1):
            location = f"models[{index}]/{model.model_id}"
            if model.model_id in seen:
                errors.append(f"{location}: duplicate model_id")
            seen.add(model.model_id)
            for field_name in [
                "model_id",
                "model_type",
                "framework",
                "artifact_uri",
                "input_schema",
                "output_schema",
                "status",
                "safety_level",
            ]:
                if not getattr(model, field_name):
                    errors.append(f"{location}: {field_name} is required")
            if model.status == "validated" and not model.metrics_uri:
                warnings.append(f"{location}: validated model has no metrics_uri")
            if model.model_type in {"detector", "segmenter"} and not model.dataset_version:
                warnings.append(f"{location}: vision model has no dataset_version")
            if model.model_type in {"detector", "segmenter"} and not model.ontology_version:
                warnings.append(f"{location}: vision model has no ontology_version")
            if model.safety_level in {"supervised", "production_candidate"}:
                if not model.calibration_requirements:
                    errors.append(f"{location}: supervised/production model requires calibration_requirements")
                if model.status != "validated":
                    errors.append(f"{location}: supervised/production model must be validated")
            if model.safety_level == "production_candidate":
                errors.append(f"{location}: production_candidate requires separate safety review in this project")
            if model.model_type == "rl_policy":
                if model.safety_level not in RL_POLICY_SAFETY_LEVELS:
                    errors.append(f"{location}: rl_policy must remain offline_only/simulation_only/dry_run")
                if model.metadata.get("direct_hardware_access") is not False:
                    errors.append(f"{location}: rl_policy requires metadata.direct_hardware_access=false")
                if model.metadata.get("requires_action_mask") is not True:
                    errors.append(f"{location}: rl_policy requires metadata.requires_action_mask=true")
        return {
            "ok": not errors,
            "version": self.version,
            "models": len(self.models),
            "errors": errors,
            "warnings": warnings,
        }

    def select(
        self,
        model_id: str | None = None,
        model_type: str | None = None,
        status: str | None = None,
        safety_level: str | None = None,
        tags: list[str] | None = None,
    ) -> ModelRegistryRecord:
        matches = self.filter(
            model_id=model_id,
            model_type=model_type,
            status=status,
            safety_level=safety_level,
            tags=tags,
        )
        if not matches:
            raise KeyError("No model registry record matches selector")
        if len(matches) > 1:
            ids = ", ".join(model.model_id for model in matches)
            raise ValueError(f"Selector matched multiple model records: {ids}")
        return matches[0]

    def filter(
        self,
        model_id: str | None = None,
        model_type: str | None = None,
        status: str | None = None,
        safety_level: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ModelRegistryRecord]:
        tags = tags or []
        result = []
        for model in self.models:
            if model_id is not None and model.model_id != model_id:
                continue
            if model_type is not None and model.model_type != model_type:
                continue
            if status is not None and model.status != status:
                continue
            if safety_level is not None and model.safety_level != safety_level:
                continue
            if tags and not set(tags).issubset(set(model.tags)):
                continue
            result.append(model)
        return result


def _load_mapping(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    text = registry_path.read_text(encoding="utf-8-sig")
    if registry_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML registries") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Registry must be a mapping: {registry_path}")
    return data


def _backend_import_error(backend: str) -> str | None:
    module_name, separator, attribute = backend.rpartition(".")
    if not separator or not module_name or not attribute:
        return f"backend must be importable as module.attribute: {backend}"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"backend module is not importable: {module_name} ({exc})"
    if not hasattr(module, attribute):
        return f"backend attribute is missing: {backend}"
    return None
