"""Shared server state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ModelEntry, discover_models

# A score stays "peak" for this long unless a higher one arrives; after the
# window it decays to the current score. Gives threshold tuning a readable
# "how close did the last attempt get" value.
PEAK_WINDOW_SECONDS = 10.0


@dataclass
class State:
    """Model registry and defaults shared by all Wyoming clients."""

    model_dirs: list[Path]
    default_model: str | None = None
    models: dict[str, ModelEntry] = field(default_factory=dict)
    backing_models: dict[str, ModelEntry] = field(default_factory=dict)
    # Bumped on every refresh so pooled interpreters and long-lived clients
    # can notice that model files changed underneath them.
    generation: int = 0
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)

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
        self.generation += 1

        known = set(all_models)
        self.scores = {
            model_id: stats
            for model_id, stats in self.scores.items()
            if model_id in known
        }

    def get_default_model_id(self) -> str | None:
        """Return the configured default model or the first discovered model."""

        if self.default_model and self.default_model in self.models:
            return self.default_model

        return next(iter(self.models), None)

    def update_score(self, model_id: str, score: float) -> None:
        """Record an inference score (called from inference threads)."""

        now = time.monotonic()
        stats = self.scores.get(model_id)
        if stats is None:
            stats = self.scores[model_id] = {
                "last": 0.0,
                "peak": 0.0,
                "peak_at": now,
                "detections": 0,
            }

        stats["last"] = score
        if score >= stats["peak"] or now - stats["peak_at"] > PEAK_WINDOW_SECONDS:
            stats["peak"] = score
            stats["peak_at"] = now

    def record_detection(self, model_id: str) -> None:
        stats = self.scores.get(model_id)
        if stats is None:
            self.update_score(model_id, 0.0)
            stats = self.scores[model_id]

        stats["detections"] += 1
