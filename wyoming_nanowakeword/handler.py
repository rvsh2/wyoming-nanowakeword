"""Wyoming event handler for NanoWakeWord clients."""

from __future__ import annotations

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
    is_detected: bool = False
    last_triggered: float | None = None


class NanoWakeWordEventHandler(AsyncEventHandler):
    """Handle Wyoming protocol events using NanoWakeWord inference."""

    def __init__(
        self,
        threshold: float,
        trigger_level: int,
        refractory_seconds: float,
        vad_threshold: float,
        cascade: bool,
        gate_threshold: float,
        state: State,
        interpreter_factory: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.threshold = threshold
        self.trigger_level = trigger_level
        self.refractory_seconds = refractory_seconds
        self.vad_threshold = vad_threshold
        self.cascade = cascade
        self.gate_threshold = gate_threshold
        self.state = state
        self.converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.detectors: dict[str, Detector] = {}
        self.audio_timestamp = 0

        if interpreter_factory is None:
            from nanowakeword.interpreter.nanointerpreter import NanoInterpreter

            interpreter_factory = NanoInterpreter.load_model

        self.interpreter_factory = interpreter_factory
        _LOGGER.debug("Client connected: %s", self.client_id)

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._get_info().event())
            return True

        if Detect.is_type(event.type):
            self._handle_detect(Detect.from_event(event))
        elif AudioStart.is_type(event.type):
            self._handle_audio_start()
        elif AudioChunk.is_type(event.type):
            await self._handle_audio_chunk(AudioChunk.from_event(event))
        elif AudioStop.is_type(event.type):
            await self._handle_audio_stop()
        else:
            _LOGGER.debug("Unexpected event: type=%s, data=%s", event.type, event.data)

        return True

    async def disconnect(self) -> None:
        _LOGGER.debug("Client disconnected: %s", self.client_id)

    def _handle_detect(self, detect: Detect) -> None:
        model_ids = self._resolve_model_ids(detect.names or [])

        for model_id in model_ids:
            if model_id in self.detectors:
                continue

            model_entry = self.state.models[model_id]
            interpreters = self._load_interpreters(model_entry)
            self.detectors[model_id] = Detector(
                id=model_id,
                entry=model_entry,
                interpreters=interpreters,
                triggers_left=self.trigger_level,
            )

        for other_model_id in set(self.detectors) - set(model_ids):
            self.detectors.pop(other_model_id)

        _LOGGER.debug("Loaded models: %s", sorted(self.detectors))

    def _handle_audio_start(self) -> None:
        self.audio_timestamp = 0

        for detector in self.detectors.values():
            detector.is_detected = False
            detector.triggers_left = self.trigger_level
            detector.last_triggered = None
            for interpreter in detector.interpreters.values():
                interpreter.reset()

    async def _handle_audio_chunk(self, audio_chunk: AudioChunk) -> None:
        chunk = self.converter.convert(audio_chunk)
        audio = np.frombuffer(chunk.audio, dtype=np.int16)

        for detector in self.detectors.values():
            skip_detector = (detector.last_triggered is not None) and (
                (time.monotonic() - detector.last_triggered) < self.refractory_seconds
            )

            score = self._predict_detector(detector, audio)

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
            return sorted(set(model_ids))

        default_model_id = self.state.get_default_model_id()
        return [default_model_id] if default_model_id else []

    def _load_interpreters(self, model_entry: ModelEntry) -> dict[str, Any]:
        model_ids = (
            [member.model for member in model_entry.members]
            if model_entry.is_ensemble
            else [model_entry.id]
        )

        interpreters: dict[str, Any] = {}
        for model_id in model_ids:
            backing_entry = self.state.backing_models[model_id]
            if backing_entry.path is None:
                raise ValueError(f"Model {model_id!r} does not have an ONNX path")

            load_kwargs: dict[str, Any] = {
                "model": str(backing_entry.path),
                "cascade": self.cascade,
                "gate_threshold": self.gate_threshold,
            }
            if self.cascade and backing_entry.gate_path is not None:
                # Pass the discovered <model>_lite.onnx explicitly so cascade
                # does not depend on the interpreter's own directory scan.
                load_kwargs["gate_model"] = str(backing_entry.gate_path)
            if self.vad_threshold > 0:
                load_kwargs["vad_threshold"] = self.vad_threshold

            interpreters[model_id] = self.interpreter_factory(**load_kwargs)

        return interpreters

    def _predict_detector(self, detector: Detector, audio: np.ndarray) -> float:
        if not detector.entry.is_ensemble:
            model_id = next(iter(detector.interpreters))
            result = detector.interpreters[model_id].predict(audio)
            return _score_for_detector(result, model_id)

        scores: dict[str, float] = {}
        for member in detector.entry.members:
            result = detector.interpreters[member.model].predict(audio)
            scores[member.model] = _score_for_detector(result, member.model)

        if detector.entry.fusion == "weighted_average":
            total_weight = sum(
                max(member.weight, 0.0) for member in detector.entry.members
            )
            if total_weight <= 0:
                return 0.0

            return sum(
                scores[member.model] * max(member.weight, 0.0)
                for member in detector.entry.members
            ) / total_weight

        if detector.entry.fusion == "all":
            return min(scores.values()) if scores else 0.0

        return self._primary_and_verifier_score(detector.entry, scores)

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
        primary_threshold = primary.threshold or self.threshold
        if primary_score <= primary_threshold:
            return 0.0

        for verifier in verifiers:
            verifier_score = scores.get(verifier.model, 0.0)
            verifier_threshold = verifier.threshold or self.threshold
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
