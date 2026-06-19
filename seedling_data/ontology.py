from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_OBJECT_CLASSES = {
    0: "container",
    1: "crop_seedling",
    2: "weed",
    3: "unknown_plant",
}
REQUIRED_CELL_STATES = {
    "empty",
    "single_crop",
    "multiple_crop",
    "weed_only",
    "crop_and_weed",
    "unknown",
    "ambiguous",
}
REQUIRED_ACTION_LABELS = {
    "keep",
    "remove_weed",
    "remove_extra_crop",
    "human_review_required",
    "no_action",
    "rescan",
}
REQUIRED_SAFETY_LABELS = {
    "safe_to_act",
    "unsafe_to_act",
    "calibration_required",
}
REQUIRED_ATTRIBUTES = {
    "tiny",
    "low_confidence",
}


@dataclass(frozen=True)
class OntologyClass:
    class_id: int
    name: str
    description: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def accepted_names(self) -> set[str]:
        return {_normalize(self.name), *(_normalize(alias) for alias in self.aliases)}


@dataclass(frozen=True)
class Ontology:
    version: str
    object_classes: dict[int, OntologyClass]
    cell_states: tuple[str, ...]
    action_labels: tuple[str, ...]
    safety_labels: tuple[str, ...]
    attributes: tuple[str, ...] = field(default_factory=tuple)

    def class_names(self) -> list[str]:
        return [self.object_classes[index].name for index in sorted(self.object_classes)]

    def validate(self) -> None:
        if not self.version:
            raise ValueError("ontology.version is required")
        if not self.object_classes:
            raise ValueError("ontology.object_classes must not be empty")
        accepted_names_by_class: dict[str, int] = {}
        for class_id, item in self.object_classes.items():
            if class_id < 0:
                raise ValueError("ontology class ids must be non-negative")
            if item.class_id != class_id:
                raise ValueError(f"ontology class id mismatch for {class_id}")
            if not item.name.strip():
                raise ValueError(f"ontology object class {class_id} name is required")
            _ensure_unique(f"object_classes.{class_id}.aliases", item.aliases)
            for accepted_name in item.accepted_names:
                owner = accepted_names_by_class.get(accepted_name)
                if owner is not None and owner != class_id:
                    raise ValueError(f"ontology object class name/alias {accepted_name!r} is duplicated across classes")
                accepted_names_by_class[accepted_name] = class_id
        for class_id, required_name in REQUIRED_OBJECT_CLASSES.items():
            item = self.object_classes.get(class_id)
            if item is None:
                raise ValueError(f"ontology.object_classes.{class_id} is required")
            if _normalize(item.name) != required_name:
                raise ValueError(f"ontology.object_classes.{class_id}.name must be {required_name}")
        for label_set_name in ["cell_states", "action_labels", "safety_labels", "attributes"]:
            values = getattr(self, label_set_name)
            if not values:
                raise ValueError(f"ontology.{label_set_name} must not be empty")
            _ensure_unique(f"ontology.{label_set_name}", values)
            if any(not str(value).strip() for value in values):
                raise ValueError(f"ontology.{label_set_name} must not contain blank labels")
        _require_labels("ontology.cell_states", self.cell_states, REQUIRED_CELL_STATES)
        _require_labels("ontology.action_labels", self.action_labels, REQUIRED_ACTION_LABELS)
        _require_labels("ontology.safety_labels", self.safety_labels, REQUIRED_SAFETY_LABELS)
        _require_labels("ontology.attributes", self.attributes, REQUIRED_ATTRIBUTES)


def load_ontology(path: str | Path) -> Ontology:
    data = _load_yaml(path)
    object_classes: dict[int, OntologyClass] = {}
    for raw_id, raw_item in data.get("object_classes", {}).items():
        if not isinstance(raw_item, dict):
            raise ValueError(f"object_classes.{raw_id} must be a mapping")
        class_id = int(raw_id)
        object_classes[class_id] = OntologyClass(
            class_id=class_id,
            name=str(raw_item["name"]),
            description=str(raw_item.get("description", "")),
            aliases=tuple(str(value) for value in raw_item.get("aliases", []) or []),
        )
    ontology = Ontology(
        version=str(data.get("version", "")),
        object_classes=object_classes,
        cell_states=tuple(str(value) for value in data.get("cell_states", []) or []),
        action_labels=tuple(str(value) for value in data.get("action_labels", []) or []),
        safety_labels=tuple(str(value) for value in data.get("safety_labels", []) or []),
        attributes=tuple(str(value) for value in data.get("attributes", []) or []),
    )
    ontology.validate()
    return ontology


def validate_dataset_class_names(
    ontology: Ontology,
    dataset_class_names: list[str],
    allow_subset: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    seen_dataset_names: dict[str, int] = {}
    for class_id, class_name in enumerate(dataset_class_names):
        normalized_class_name = _normalize(class_name)
        if not normalized_class_name:
            errors.append(f"class id {class_id} dataset name is blank")
            continue
        previous_id = seen_dataset_names.get(normalized_class_name)
        if previous_id is not None:
            errors.append(f"class id {class_id} duplicates dataset class name from class id {previous_id}")
            continue
        seen_dataset_names[normalized_class_name] = class_id
        ontology_class = ontology.object_classes.get(class_id)
        if ontology_class is None:
            errors.append(f"class id {class_id} is not defined in ontology")
            continue
        if normalized_class_name not in ontology_class.accepted_names:
            errors.append(
                f"class id {class_id} dataset name {class_name!r} does not match ontology "
                f"name {ontology_class.name!r} or aliases {list(ontology_class.aliases)!r}"
            )
    if not allow_subset:
        missing = sorted(set(ontology.object_classes) - set(range(len(dataset_class_names))))
        for class_id in missing:
            warnings.append(f"ontology class id {class_id} is not present in dataset class names")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "ontology_version": ontology.version,
        "dataset_class_names": dataset_class_names,
    }


def _load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load ontology files") from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Ontology file must contain a mapping: {path}")
    return data


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _ensure_unique(field_name: str, values: tuple[str, ...]) -> None:
    seen: set[str] = set()
    for value in values:
        normalized = _normalize(value)
        if not normalized:
            continue
        if normalized in seen:
            raise ValueError(f"{field_name} contains duplicate value {value!r}")
        seen.add(normalized)


def _require_labels(field_name: str, values: tuple[str, ...], required: set[str]) -> None:
    present = {_normalize(value) for value in values}
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"{field_name} missing required labels: {', '.join(missing)}")
