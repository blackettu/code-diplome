from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a config file is syntactically valid but structurally wrong."""


@dataclass(frozen=True)
class FieldSpec:
    path: tuple[str, ...]
    expected_type: type | tuple[type, ...]
    required: bool = True

    @classmethod
    def parse(
        cls,
        dotted_path: str,
        expected_type: type | tuple[type, ...],
        required: bool = True,
    ) -> "FieldSpec":
        return cls(tuple(part for part in dotted_path.split(".") if part), expected_type, required)


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON/YAML config and require a top-level mapping."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requires PyYAML. Install base requirements or use a .json config."
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping: {config_path}")
    return data


def require_sections(config: Mapping[str, Any], section_names: list[str]) -> None:
    errors = []
    for section_name in section_names:
        value = config.get(section_name)
        if value is None:
            errors.append(f"missing section `{section_name}`")
        elif not isinstance(value, Mapping):
            errors.append(f"`{section_name}` must be a mapping, got {type(value).__name__}")
    if errors:
        raise ConfigError("; ".join(errors))


def validate_fields(config: Mapping[str, Any], specs: list[FieldSpec]) -> None:
    errors = []
    for spec in specs:
        value, exists = _lookup(config, spec.path)
        label = ".".join(spec.path)
        if not exists:
            if spec.required:
                errors.append(f"missing `{label}`")
            continue
        if not isinstance(value, spec.expected_type):
            expected = _type_label(spec.expected_type)
            errors.append(f"`{label}` must be {expected}, got {type(value).__name__}")
    if errors:
        raise ConfigError("; ".join(errors))


def validate_existing_paths(
    config: Mapping[str, Any],
    dotted_paths: list[str],
    base_dir: str | Path | None = None,
) -> None:
    errors = []
    root = Path(base_dir) if base_dir else Path.cwd()
    for dotted_path in dotted_paths:
        path_parts = tuple(part for part in dotted_path.split(".") if part)
        value, exists = _lookup(config, path_parts)
        if not exists or value is None:
            continue
        if not isinstance(value, (str, Path)):
            errors.append(f"`{dotted_path}` must be a path-like string, got {type(value).__name__}")
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            errors.append(f"`{dotted_path}` path does not exist: {candidate}")
    if errors:
        raise ConfigError("; ".join(errors))


def optional_path(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    raise ConfigError(f"`{label}` must be a path-like string, got {type(value).__name__}")


def section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"`{name}` section is required and must be a mapping")
    return value


def _lookup(config: Mapping[str, Any], path: tuple[str, ...]) -> tuple[Any, bool]:
    current: Any = config
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None, False
        current = current[part]
    return current, True


def _type_label(expected_type: type | tuple[type, ...]) -> str:
    if isinstance(expected_type, tuple):
        return " or ".join(item.__name__ for item in expected_type)
    return expected_type.__name__
