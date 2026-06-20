"""Shared server state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import ModelEntry, discover_models


@dataclass
class State:
    """Model registry and defaults shared by all Wyoming clients."""

    model_dirs: list[Path]
    default_model: str | None = None
    models: dict[str, ModelEntry] = field(default_factory=dict)
    backing_models: dict[str, ModelEntry] = field(default_factory=dict)

    def refresh(self) -> None:
        """Refresh available ONNX models from configured directories."""

        all_models = discover_models(self.model_dirs)
        self.models = {
            model_id: model_entry
            for model_id, model_entry in all_models.items()
            if not model_entry.metadata.hidden
        }
        self.backing_models = {
            model_id: model_entry
            for model_id, model_entry in all_models.items()
            if model_entry.path is not None
        }

    def get_default_model_id(self) -> str | None:
        """Return the configured default model or the first discovered model."""

        if self.default_model and self.default_model in self.models:
            return self.default_model

        return next(iter(self.models), None)
