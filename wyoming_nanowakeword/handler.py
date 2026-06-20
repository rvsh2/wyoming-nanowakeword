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
from .state import State

_LOGGER = logging.getLogger(__name__)


@dataclass
class Detector:
    """Loaded NanoWakeWord model state for one wake word."""

    id: str
    interpreter: Any
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
            load_kwargs: dict[str, Any] = {
                "model": str(model_entry.path),
                "cascade": self.cascade,
                "gate_threshold": self.gate_threshold,
            }
            if self.vad_threshold > 0:
                load_kwargs["vad_threshold"] = self.vad_threshold

            interpreter = self.interpreter_factory(**load_kwargs)
            self.detectors[model_id] = Detector(
                id=model_id,
                interpreter=interpreter,
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
            detector.interpreter.reset()

    async def _handle_audio_chunk(self, audio_chunk: AudioChunk) -> None:
        chunk = self.converter.convert(audio_chunk)
        audio = np.frombuffer(chunk.audio, dtype=np.int16)

        for detector in self.detectors.values():
            skip_detector = (detector.last_triggered is not None) and (
                (time.monotonic() - detector.last_triggered) < self.refractory_seconds
            )

            result = detector.interpreter.predict(audio)
            score = _score_for_detector(result, detector.id)

            if skip_detector or score <= self.threshold:
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
            detector.interpreter.reset()
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

    def _get_info(self) -> Info:
        models: list[WakeModel] = []
        for model_entry in self.state.models.values():
            metadata = model_entry.metadata
            description_parts = [model_entry.phrase]
            if metadata.architecture:
                description_parts.append(f"Architecture: {metadata.architecture}")

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
