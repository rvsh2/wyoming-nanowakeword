"""Shared server state."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ModelEntry, discover_models
from .settings import ServerSettings

# A score stays "peak" for this long unless a higher one arrives; after the
# window it decays to the current score. Gives threshold tuning a readable
# "how close did the last attempt get" value.
PEAK_WINDOW_SECONDS = 10.0


@dataclass
class State:
    """Model registry and defaults shared by all Wyoming clients."""

    model_dirs: list[Path]
    default_model: str | None = None
    settings: ServerSettings = field(default_factory=ServerSettings)
    models: dict[str, ModelEntry] = field(default_factory=dict)
    backing_models: dict[str, ModelEntry] = field(default_factory=dict)
    # Bumped on every refresh so pooled interpreters and long-lived clients
    # can notice that model files changed underneath them.
    generation: int = 0
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Connected Wyoming clients (satellites / Home Assistant).
    clients: int = 0
    # Subscribers of server events (detections), e.g. SSE connections.
    listeners: list[asyncio.Queue] = field(default_factory=list)

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

    def update_score(
        self, model_id: str, score: float, inference_ms: float | None = None
    ) -> None:
        """Record an inference score (called from inference threads)."""

        now = time.monotonic()
        stats = self.scores.get(model_id)
        if stats is None:
            stats = self.scores[model_id] = {
                "last": 0.0,
                "peak": 0.0,
                "peak_at": now,
                "detections": 0,
                "avg_ms": None,
            }

        stats["last"] = score
        if score >= stats["peak"] or now - stats["peak_at"] > PEAK_WINDOW_SECONDS:
            stats["peak"] = score
            stats["peak_at"] = now

        stats.setdefault("rejections", 0)

        if inference_ms is not None:
            previous = stats.get("avg_ms")
            # Exponential moving average keeps the number stable but current.
            if previous is None:
                stats["avg_ms"] = inference_ms
            else:
                stats["avg_ms"] = 0.9 * previous + 0.1 * inference_ms

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.listeners.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self.listeners:
            self.listeners.remove(queue)

    def publish(self, event: dict[str, Any]) -> None:
        """Broadcast a server event to all subscribers (never blocks)."""

        for queue in list(self.listeners):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    def record_detection(self, model_id: str) -> None:
        stats = self._stats_for(model_id)
        stats["detections"] += 1

    def record_rejection(self, model_id: str) -> None:
        """A candidate detection that the remote verifier refused."""

        stats = self._stats_for(model_id)
        stats["rejections"] = stats.get("rejections", 0) + 1

    def _stats_for(self, model_id: str) -> dict[str, Any]:
        stats = self.scores.get(model_id)
        if stats is None:
            self.update_score(model_id, 0.0)
            stats = self.scores[model_id]

        return stats
