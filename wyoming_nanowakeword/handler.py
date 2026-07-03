"""Wyoming event handler for NanoWakeWord clients."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler
from wyoming.wake import Detect, Detection, NotDetected

from . import __version__
from .interpreters import InterpreterManager
from .models import ModelEntry
from .state import State

_LOGGER = logging.getLogger(__name__)


@dataclass
class Detector:
    """Loaded NanoWakeWord model state for one wake word."""

    id: str
    entry: ModelEntry
    interpreters: dict[str, Any]
    triggers_left: int
    # State.generation when the interpreters were acquired.
    generation: int = 0
    is_detected: bool = False
    last_triggered: float | None = None


class NanoWakeWordEventHandler(AsyncEventHandler):
    """Handle Wyoming protocol events using NanoWakeWord inference."""

    def __init__(
        self,
        *args: Any,
        threshold: float,
        trigger_level: int,
        refractory_seconds: float,
        vad_threshold: float,
        cascade: bool,
        gate_threshold: float,
        state: State,
        interpreter_factory: Any | None = None,
        interpreter_manager: InterpreterManager | None = None,
        **kwargs: Any,
    ) -> None:
        # *args carries (reader, writer) from wyoming's handler factory; keep our
        # own params keyword-only so they never collide with those positionals.
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.threshold = threshold
        self.trigger_level = trigger_level
        self.refractory_seconds = refractory_seconds
        self.state = state
        self.converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.detectors: dict[str, Detector] = {}
        self.audio_timestamp = 0

        if interpreter_manager is None:
            interpreter_manager = InterpreterManager(
                state,
                cascade=cascade,
                gate_threshold=gate_threshold,
                vad_threshold=vad_threshold,
                factory=interpreter_factory,
            )

        self.manager = interpreter_manager
        _LOGGER.debug("Client connected: %s", self.client_id)

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._get_info().event())
            return True

        if Detect.is_type(event.type):
            await self._handle_detect(Detect.from_event(event))
        elif AudioStart.is_type(event.type):
            await self._handle_audio_start()
        elif AudioChunk.is_type(event.type):
            await self._handle_audio_chunk(AudioChunk.from_event(event))
        elif AudioStop.is_type(event.type):
            await self._handle_audio_stop()
        else:
            _LOGGER.debug("Unexpected event: type=%s, data=%s", event.type, event.data)

        return True

    async def disconnect(self) -> None:
        self._release_detectors(self.detectors)
        self.detectors = {}
        _LOGGER.debug("Client disconnected: %s", self.client_id)

    async def _handle_detect(self, detect: Detect) -> None:
        model_ids = self._resolve_model_ids(detect.names or [])

        for model_id in model_ids:
            if model_id in self.detectors:
                continue

            model_entry = self.state.models[model_id]
            generation = self.state.generation
            # ONNX session loading is blocking; keep the event loop free for
            # the other clients' audio.
            interpreters = await asyncio.to_thread(
                self.manager.acquire_for_entry, model_entry
            )
            self.detectors[model_id] = Detector(
                id=model_id,
                entry=model_entry,
                interpreters=interpreters,
                triggers_left=self.trigger_level,
                generation=generation,
            )

        removed = {
            model_id: self.detectors.pop(model_id)
            for model_id in set(self.detectors) - set(model_ids)
        }
        self._release_detectors(removed)

        _LOGGER.debug("Loaded models: %s", sorted(self.detectors))

    async def _handle_audio_start(self) -> None:
        self.audio_timestamp = 0

        current_generation = self.state.generation
        if self.detectors and any(
            detector.generation != current_generation
            for detector in self.detectors.values()
        ):
            # Model files changed (HTTP API upload/restore or re-scan):
            # reload the same wake words from the new registry.
            names = sorted(self.detectors)
            self._release_detectors(self.detectors)
            self.detectors = {}
            await self._handle_detect(Detect(names=names))

        if not self.detectors:
            # Detect is optional in the Wyoming wake protocol; fall back to
            # the default model like wyoming-openwakeword does.
            await self._handle_detect(Detect())

        for detector in self.detectors.values():
            detector.is_detected = False
            detector.triggers_left = self.trigger_level
            detector.last_triggered = None
            for interpreter in detector.interpreters.values():
                interpreter.reset()

    def _release_detectors(self, detectors: dict[str, Detector]) -> None:
        for detector in detectors.values():
            self.manager.release(detector.interpreters, detector.generation)

    async def _handle_audio_chunk(self, audio_chunk: AudioChunk) -> None:
        chunk = self.converter.convert(audio_chunk)
        audio = np.frombuffer(chunk.audio, dtype=np.int16)

        for detector in self.detectors.values():
            skip_detector = (detector.last_triggered is not None) and (
                (time.monotonic() - detector.last_triggered) < self.refractory_seconds
            )

            # ONNX inference is blocking (and releases the GIL); run it off
            # the event loop so other clients keep streaming.
            score = await asyncio.to_thread(self._predict_detector, detector, audio)

            if skip_detector:
                continue

            if score <= self.threshold:
                # Require trigger_level *consecutive* activations, like
                # wyoming-openwakeword: a miss resets the streak.
                detector.triggers_left = self.trigger_level
                continue

            detector.triggers_left -= 1
            if detector.triggers_left > 0:
                continue

            detector.is_detected = True
            detector.last_triggered = time.monotonic()
            detector.triggers_left = self.trigger_level
            self.state.record_detection(detector.id)
            await self.write_event(
                Detection(name=detector.id, timestamp=self.audio_timestamp).event()
            )
            for interpreter in detector.interpreters.values():
                interpreter.reset()
            _LOGGER.debug("Detected %s at %s", detector.id, self.audio_timestamp)

        self.audio_timestamp += chunk.milliseconds

    async def _handle_audio_stop(self) -> None:
        if not any(detector.is_detected for detector in self.detectors.values()):
            await self.write_event(NotDetected().event())

    def _resolve_model_ids(self, requested_names: list[str]) -> list[str]:
        if requested_names:
            model_ids = [
                model_name
                for model_name in requested_names
                if model_name in self.state.models
            ]
            unknown_names = [
                model_name
                for model_name in requested_names
                if model_name not in self.state.models
            ]
            if unknown_names:
                _LOGGER.warning(
                    "Unknown wake word names requested: %s (available: %s)",
                    ", ".join(unknown_names),
                    ", ".join(sorted(self.state.models)) or "(none)",
                )

            if model_ids:
                return sorted(set(model_ids))

            _LOGGER.warning("No requested wake word matched; using default model")

        default_model_id = self.state.get_default_model_id()
        return [default_model_id] if default_model_id else []

    def _predict_detector(self, detector: Detector, audio: np.ndarray) -> float:
        if not detector.entry.is_ensemble:
            model_id = next(iter(detector.interpreters))
            result = detector.interpreters[model_id].predict(audio)
            score = _score_for_detector(result, model_id)
            self.state.update_score(detector.id, score)
            return score

        scores: dict[str, float] = {}
        for member in detector.entry.members:
            result = detector.interpreters[member.model].predict(audio)
            scores[member.model] = _score_for_detector(result, member.model)
            self.state.update_score(member.model, scores[member.model])

        fused = self._fuse_scores(detector.entry, scores)
        self.state.update_score(detector.id, fused)
        return fused

    def _fuse_scores(self, entry: ModelEntry, scores: dict[str, float]) -> float:
        if entry.fusion == "weighted_average":
            total_weight = sum(max(member.weight, 0.0) for member in entry.members)
            if total_weight <= 0:
                return 0.0

            return sum(
                scores[member.model] * max(member.weight, 0.0)
                for member in entry.members
            ) / total_weight

        if entry.fusion == "all":
            return min(scores.values()) if scores else 0.0

        return self._primary_and_verifier_score(entry, scores)

    def _primary_and_verifier_score(
        self, model_entry: ModelEntry, scores: dict[str, float]
    ) -> float:
        primary = next(
            (
                member
                for member in model_entry.members
                if member.role.lower() == "primary"
            ),
            model_entry.members[0],
        )
        verifiers = [
            member
            for member in model_entry.members
            if member.model != primary.model
            and member.role.lower() in {"verifier", "confirm", "member"}
        ]

        primary_score = scores.get(primary.model, 0.0)
        primary_threshold = (
            primary.threshold if primary.threshold is not None else self.threshold
        )
        if primary_score <= primary_threshold:
            return 0.0

        for verifier in verifiers:
            verifier_score = scores.get(verifier.model, 0.0)
            verifier_threshold = (
                verifier.threshold if verifier.threshold is not None else self.threshold
            )
            if verifier_score <= verifier_threshold:
                return 0.0

        return primary_score

    def _get_info(self) -> Info:
        models: list[WakeModel] = []
        for model_entry in self.state.models.values():
            metadata = model_entry.metadata
            description_parts = [model_entry.phrase]
            if metadata.architecture:
                description_parts.append(f"Architecture: {metadata.architecture}")
            if model_entry.is_ensemble:
                description_parts.append(f"Fusion: {model_entry.fusion}")

            models.append(
                WakeModel(
                    name=model_entry.id,
                    description=" - ".join(description_parts),
                    phrase=model_entry.phrase,
                    attribution=Attribution(
                        name="Arcosoph",
                        url="https://github.com/arcosoph/nanowakeword",
                    ),
                    installed=True,
                    languages=[metadata.language] if metadata.language else [],
                    version=metadata.version or "",
                )
            )

        return Info(
            wake=[
                WakeProgram(
                    name="nanowakeword",
                    description=(
                        "NanoWakeWord ONNX inference through the Wyoming protocol"
                    ),
                    attribution=Attribution(
                        name="Arcosoph",
                        url="https://github.com/arcosoph/nanowakeword",
                    ),
                    installed=True,
                    version=__version__,
                    models=models,
                )
            ]
        )


def _score_for_detector(result: Any, detector_id: str) -> float:
    if hasattr(result, "get"):
        score = result.get(detector_id, None)
        if score is not None:
            return float(score)

    if hasattr(result, "score"):
        return float(result.score)

    if isinstance(result, dict):
        return float(result.get(detector_id, 0.0))

    return 0.0
