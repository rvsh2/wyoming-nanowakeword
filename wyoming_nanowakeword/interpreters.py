"""Loading and pooling of NanoWakeWord interpreters.

Interpreters hold per-stream state (NanoWakeWord's AudioFeatures buffers), so
one interpreter serves one client at a time. Loading the ONNX sessions is the
expensive part, so released interpreters are pooled and handed to the next
client after a reset() instead of being reloaded — Home Assistant reconnects
often, and satellites would otherwise each pay the full load cost.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from .models import ModelEntry
from .state import State

_LOGGER = logging.getLogger(__name__)


class InterpreterManager:
    """Shared, pooled interpreter loading for all client connections."""

    def __init__(
        self,
        state: State,
        *,
        cascade: bool | None = None,
        gate_threshold: float | None = None,
        vad_threshold: float | None = None,
        factory: Callable[..., Any] | None = None,
    ) -> None:
        self.state = state
        # Explicit constructor values seed the runtime settings; afterwards
        # the settings object is the single source of truth (it can change
        # at runtime through the HTTP API).
        if cascade is not None:
            state.settings.cascade = cascade
        if gate_threshold is not None:
            state.settings.gate_threshold = gate_threshold
        if vad_threshold is not None:
            state.settings.vad_threshold = vad_threshold

        if factory is None:
            from nanowakeword.interpreter.nanointerpreter import NanoInterpreter

            factory = NanoInterpreter.load_model

        self.factory = factory
        self._idle: dict[str, list[Any]] = {}
        self._generation = state.generation
        self._lock = threading.Lock()

    def acquire_for_entry(self, model_entry: ModelEntry) -> dict[str, Any]:
        """Blocking: interpreters for a wake word (all members for ensembles)."""

        model_ids = (
            [member.model for member in model_entry.members]
            if model_entry.is_ensemble
            else [model_entry.id]
        )
        return {model_id: self._acquire(model_id) for model_id in model_ids}

    def release(self, interpreters: dict[str, Any], generation: int) -> None:
        """Return interpreters to the pool for reuse by other clients.

        Interpreters acquired before a model refresh may be backed by files
        that no longer exist; those are dropped instead of pooled.
        """

        with self._lock:
            if generation != self.state.generation:
                return

            self._sync_generation()
            for model_id, interpreter in interpreters.items():
                self._idle.setdefault(model_id, []).append(interpreter)

    def warm_up(self, model_ids: Iterable[str]) -> None:
        """Blocking: preload interpreters into the pool (startup latency)."""

        for model_id in model_ids:
            with self._lock:
                self._sync_generation()
                if self._idle.get(model_id):
                    continue

            started = time.monotonic()
            interpreter = self._load(model_id)
            with self._lock:
                self._idle.setdefault(model_id, []).append(interpreter)
            _LOGGER.info(
                "Preloaded %s in %.2fs", model_id, time.monotonic() - started
            )

    def _acquire(self, model_id: str) -> Any:
        with self._lock:
            self._sync_generation()
            idle = self._idle.get(model_id)
            interpreter = idle.pop() if idle else None

        if interpreter is not None:
            interpreter.reset()
            return interpreter

        return self._load(model_id)

    def _sync_generation(self) -> None:
        # Called with the lock held. A refresh may have replaced model files;
        # pooled interpreters from an older generation must not be reused.
        if self._generation != self.state.generation:
            self._idle.clear()
            self._generation = self.state.generation

    def _load(self, model_id: str) -> Any:
        backing_entry = self.state.backing_models[model_id]
        if backing_entry.path is None:
            raise ValueError(f"Model {model_id!r} does not have an ONNX path")

        settings = self.state.settings
        load_kwargs: dict[str, Any] = {
            "model": str(backing_entry.path),
            "cascade": settings.cascade,
            "gate_threshold": settings.gate_threshold,
        }
        if settings.cascade and backing_entry.gate_path is not None:
            # Pass the discovered <model>_lite.onnx explicitly so cascade
            # does not depend on the interpreter's own directory scan.
            load_kwargs["gate_model"] = str(backing_entry.gate_path)
        if settings.vad_threshold > 0:
            load_kwargs["vad_threshold"] = settings.vad_threshold

        return self.factory(**load_kwargs)
