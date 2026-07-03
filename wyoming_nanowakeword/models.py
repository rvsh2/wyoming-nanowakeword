"""Model discovery and metadata helpers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_LOGGER = logging.getLogger(__name__)

_VERSION_SUFFIX = re.compile(r"^(?P<name>.+?)_v(?P<version>[0-9][0-9A-Za-z_.-]*)$")

_FUSION_MODES = {"primary_and_verifier", "weighted_average", "all"}


@dataclass(frozen=True)
class ModelMetadata:
    """Optional user-facing metadata for a wake word model."""

    name: str | None = None
    phrase: str | None = None
    language: str | None = None
    architecture: str | None = None
    version: str | None = None
    hidden: bool = False


@dataclass(frozen=True)
class EnsembleMember:
    """One ONNX model participating in an ensemble wake word."""

    model: str
    role: str = "member"
    threshold: float | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class ModelEntry:
    """A discovered NanoWakeWord ONNX model."""

    id: str
    path: Path | None
    metadata: ModelMetadata
    gate_path: Path | None = None
    members: tuple[EnsembleMember, ...] = ()
    fusion: str = "single"

    @property
    def phrase(self) -> str:
        if self.metadata.phrase:
            return self.metadata.phrase

        phrase = self.id.lower().replace("_", " ").replace("-", " ").strip()
        return " ".join(word.capitalize() for word in phrase.split())

    @property
    def is_ensemble(self) -> bool:
        return bool(self.members)


def normalize_model_id(path: Path) -> str:
    """Return a stable wake word id for a NanoWakeWord ONNX model path."""

    stem = path.stem
    if stem.endswith("_lite"):
        stem = stem[: -len("_lite")]

    match = _VERSION_SUFFIX.match(stem)
    if match:
        stem = match.group("name")

    return stem


def _read_models_mapping(model_dir: Path) -> dict[str, dict[str, Any]]:
    """Read and validate the models.yaml mapping (single parse for all uses)."""

    metadata_path = model_dir / "models.yaml"
    if not metadata_path.is_file():
        return {}

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        raw_metadata = yaml.safe_load(metadata_file) or {}

    if not isinstance(raw_metadata, dict):
        raise ValueError("models.yaml must contain a mapping or a 'models' mapping")

    models = raw_metadata.get("models", raw_metadata)
    if not isinstance(models, dict):
        raise ValueError("models.yaml must contain a mapping or a 'models' mapping")

    normalized: dict[str, dict[str, Any]] = {}
    for model_id, raw_entry in models.items():
        if raw_entry is None:
            raw_entry = {}

        if not isinstance(raw_entry, dict):
            raise ValueError(f"Metadata for {model_id!r} must be a mapping")

        normalized[str(model_id)] = raw_entry

    return normalized


def load_metadata(model_dir: Path) -> dict[str, ModelMetadata]:
    """Load optional models.yaml metadata from a model directory."""

    return _metadata_from_mapping(_read_models_mapping(model_dir))


def load_ensemble_specs(
    model_dir: Path,
) -> dict[str, tuple[str, tuple[EnsembleMember, ...]]]:
    """Load optional ensemble definitions from models.yaml."""

    return _ensembles_from_mapping(_read_models_mapping(model_dir))


def _metadata_from_mapping(
    models: dict[str, dict[str, Any]],
) -> dict[str, ModelMetadata]:
    metadata: dict[str, ModelMetadata] = {}
    for model_id, entry in models.items():
        metadata[model_id] = ModelMetadata(
            name=_optional_str(entry.get("name")),
            phrase=_optional_str(entry.get("phrase")),
            language=_optional_str(entry.get("language")),
            architecture=_optional_str(entry.get("architecture")),
            version=_optional_str(entry.get("version")),
            hidden=bool(entry.get("hidden", False)),
        )

    return metadata


def _ensembles_from_mapping(
    models: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, tuple[EnsembleMember, ...]]]:
    ensembles: dict[str, tuple[str, tuple[EnsembleMember, ...]]] = {}
    for model_id, raw_entry in models.items():
        if "members" not in raw_entry:
            continue

        raw_members = raw_entry["members"]
        if not isinstance(raw_members, list) or not raw_members:
            raise ValueError(
                f"Ensemble {model_id!r} must define a non-empty members list"
            )

        members: list[EnsembleMember] = []
        for raw_member in raw_members:
            if isinstance(raw_member, str):
                members.append(EnsembleMember(model=raw_member))
                continue

            if not isinstance(raw_member, dict):
                raise ValueError(f"Invalid ensemble member in {model_id!r}")

            member_model = _optional_str(raw_member.get("model"))
            if not member_model:
                raise ValueError(f"Ensemble member in {model_id!r} needs a model")

            member_weight = _optional_float(raw_member.get("weight"))
            members.append(
                EnsembleMember(
                    model=member_model,
                    role=_optional_str(raw_member.get("role")) or "member",
                    threshold=_optional_float(raw_member.get("threshold")),
                    weight=1.0 if member_weight is None else member_weight,
                )
            )

        fusion = _optional_str(raw_entry.get("fusion")) or "primary_and_verifier"
        if fusion not in _FUSION_MODES:
            raise ValueError(
                f"Ensemble {model_id!r} has unknown fusion {fusion!r}; "
                f"expected one of: {', '.join(sorted(_FUSION_MODES))}"
            )

        ensembles[model_id] = (fusion, tuple(members))

    return ensembles


def _version_key(path: Path) -> tuple[tuple[int, object], ...]:
    """Sort key for the _v<version> filename suffix; unversioned sorts lowest."""

    match = _VERSION_SUFFIX.match(path.stem)
    if not match:
        return ()

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"[._-]", match.group("version"))
        if part
    )


def discover_models(model_dirs: list[Path]) -> dict[str, ModelEntry]:
    """Discover ONNX models from directories.

    Lite models are associated with their main model when possible and are not
    published as separate wake words.
    """

    discovered: dict[str, Path] = {}
    gate_paths: dict[str, Path] = {}
    metadata: dict[str, ModelMetadata] = {}
    ensembles: dict[str, tuple[str, tuple[EnsembleMember, ...]]] = {}

    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue

        mapping = _read_models_mapping(model_dir)
        metadata.update(_metadata_from_mapping(mapping))
        ensembles.update(_ensembles_from_mapping(mapping))

        for model_path in sorted(model_dir.glob("*.onnx")):
            if model_path.stem.endswith("_lite"):
                # Key gates by the exact main-model stem so a gate from one
                # model version can never attach to a different version.
                gate_paths.setdefault(model_path.stem[: -len("_lite")], model_path)
                continue

            model_id = normalize_model_id(model_path)
            existing_path = discovered.get(model_id)
            if existing_path is None or existing_path == model_path:
                discovered[model_id] = model_path
                continue

            # Several versions of the same wake word: serve the newest one.
            if _version_key(model_path) > _version_key(existing_path):
                discovered[model_id] = model_path
                newer, older = model_path, existing_path
            else:
                newer, older = existing_path, model_path
            _LOGGER.warning(
                "Multiple files provide wake word %r: using %s, ignoring %s",
                model_id,
                newer.name,
                older.name,
            )

    entries = {
        model_id: ModelEntry(
            id=model_id,
            path=model_path,
            metadata=metadata.get(model_id, ModelMetadata()),
            gate_path=gate_paths.get(model_path.stem),
        )
        for model_id, model_path in discovered.items()
    }

    for model_id, (fusion, members) in ensembles.items():
        if model_id in discovered:
            raise ValueError(
                f"Ensemble {model_id!r} conflicts with model file "
                f"{discovered[model_id].name}; rename the file or the ensemble"
            )

        missing_members = [
            member.model for member in members if member.model not in discovered
        ]
        if missing_members:
            raise ValueError(
                f"Ensemble {model_id!r} references missing models: "
                + ", ".join(missing_members)
            )

        entries[model_id] = ModelEntry(
            id=model_id,
            path=None,
            metadata=metadata.get(model_id, ModelMetadata()),
            members=members,
            fusion=fusion,
        )

    return entries


def _optional_str(value: object) -> str | None:
    if value is None:
        return None

    value_str = str(value).strip()
    return value_str or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None

    if isinstance(value, (str, int, float)):
        return float(value)

    raise ValueError(f"Expected float-compatible value, got {type(value).__name__}")
