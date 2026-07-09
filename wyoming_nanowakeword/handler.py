"""Wyoming event handler for NanoWakeWord clients."""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler
from wyoming.wake import Detect, Detection, NotDetected

from . import __version__
from .asr_verify import verify_wake_word as asr_verify_wake_word
from .fusion import fuse_scores, score_from_result
from .interpreters import InterpreterManager
from .models import ModelEntry
from .state import State

_LOGGER = logging.getLogger(__name__)

# Seconds of buffered audio sent to the central verifier (see _buffer_as_wav).
# 1.5 s proved too short on real captures (the verifier's feature warmup can
# swallow the wake word when local detection fires late); 2.0 s scored 0.98+
# on every genuine capture while 3 s depressed some to 0.64.
VERIFY_WINDOW_SECONDS = 2.0


@dataclass
class Detector:
    """Loaded NanoWakeWord model state for one wake word."""

    id: str
    entry: ModelEntry
    interpreters: dict[str, Any]
    triggers_left: int
    # Effective values: models.yaml per-model overrides, else server-wide.
    threshold: float = 0.95
    trigger_level: int = 1
    # State.generation when the interpreters were acquired.
    generation: int = 0
    is_detected: bool = False
    last_triggered: float | None = None


class NanoWakeWordEventHandler(AsyncEventHandler):
    """Handle Wyoming protocol events using NanoWakeWord inference."""

    def __init__(
        self,
        *args: Any,
        threshold: float | None = None,
        trigger_level: int | None = None,
        refractory_seconds: float | None = None,
        vad_threshold: float | None = None,
        cascade: bool | None = None,
        gate_threshold: float | None = None,
        state: State,
        interpreter_factory: Any | None = None,
        interpreter_manager: InterpreterManager | None = None,
        capture_dir: Path | None = None,
        capture_seconds: float = 3.0,
        capture_keep: int = 200,
        **kwargs: Any,
    ) -> None:
        # *args carries (reader, writer) from wyoming's handler factory; keep our
        # own params keyword-only so they never collide with those positionals.
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.state = state

        # Explicit constructor values seed the runtime settings; from then on
        # state.settings is the single source of truth (changeable via the
        # HTTP API, i.e. from Home Assistant).
        settings = state.settings
        if threshold is not None:
            settings.threshold = threshold
        if trigger_level is not None:
            settings.trigger_level = trigger_level
        if refractory_seconds is not None:
            settings.refractory_seconds = refractory_seconds
        if capture_dir is not None:
            settings.capture = True

        self.converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.detectors: dict[str, Detector] = {}
        self.audio_timestamp = 0

        self.capture_dir = capture_dir
        self.capture_keep = capture_keep
        self._capture_max_samples = int(capture_seconds * 16000)
        self._capture_buffer: deque[np.ndarray] = deque()
        self._capture_samples = 0
        self._verify_session: Any | None = None

        state.clients += 1

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

    @property
    def _effective_capture_dir(self) -> Path | None:
        if self.capture_dir is not None:
            return self.capture_dir
        if self.state.model_dirs:
            return self.state.model_dirs[0] / "captures"
        return None

    @property
    def _buffer_needed(self) -> bool:
        settings = self.state.settings
        return settings.capture or (settings.verify and bool(settings.verify_url))

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
        self.state.clients = max(0, self.state.clients - 1)
        self._release_detectors(self.detectors)
        self.detectors = {}
        if self._verify_session is not None:
            await self._verify_session.close()
            self._verify_session = None
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
            metadata = model_entry.metadata
            settings = self.state.settings
            threshold = (
                metadata.threshold
                if metadata.threshold is not None
                else settings.threshold
            )
            trigger_level = (
                metadata.trigger_level
                if metadata.trigger_level is not None
                else settings.trigger_level
            )
            self.detectors[model_id] = Detector(
                id=model_id,
                entry=model_entry,
                interpreters=interpreters,
                triggers_left=trigger_level,
                threshold=threshold,
                trigger_level=trigger_level,
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
            detector.triggers_left = detector.trigger_level
            detector.last_triggered = None
            for interpreter in detector.interpreters.values():
                interpreter.reset()

    def _release_detectors(self, detectors: dict[str, Detector]) -> None:
        for detector in detectors.values():
            self.manager.release(detector.interpreters, detector.generation)

    async def _handle_audio_chunk(self, audio_chunk: AudioChunk) -> None:
        chunk = self.converter.convert(audio_chunk)
        audio = np.frombuffer(chunk.audio, dtype=np.int16)

        if self._buffer_needed:
            self._capture_append(audio)

        refractory_seconds = self.state.settings.refractory_seconds
        for detector in self.detectors.values():
            skip_detector = (detector.last_triggered is not None) and (
                (time.monotonic() - detector.last_triggered) < refractory_seconds
            )

            # ONNX inference is blocking (and releases the GIL); run it off
            # the event loop so other clients keep streaming.
            score = await asyncio.to_thread(self._predict_detector, detector, audio)

            if skip_detector:
                continue

            if score <= detector.threshold:
                # Require trigger_level *consecutive* activations, like
                # wyoming-openwakeword: a miss resets the streak.
                detector.triggers_left = detector.trigger_level
                continue

            detector.triggers_left -= 1
            if detector.triggers_left > 0:
                continue

            detector.last_triggered = time.monotonic()
            detector.triggers_left = detector.trigger_level

            # Hybrid satellite + server: confirm the candidate with the
            # central verifier (e.g. an ensemble) before waking the pipeline.
            verified, remote_peak = await self._verify_candidate(detector, score)
            if not verified:
                self.state.record_rejection(detector.id)
                self.state.publish(
                    {
                        "type": "rejected",
                        "model": detector.id,
                        "score": round(score, 4),
                        "remote_peak": remote_peak,
                        "timestamp": self.audio_timestamp,
                    }
                )
                if self.state.settings.capture:
                    # Rejected candidates are the most interesting audio of
                    # all: false positives of the light model, or genuine
                    # wake words the verifier thresholds miss.
                    await asyncio.to_thread(
                        self._write_capture, f"{detector.id}-rejected"
                    )
                for interpreter in detector.interpreters.values():
                    interpreter.reset()
                _LOGGER.info(
                    "Candidate %s (local %.3f) rejected by verifier (peak %s)",
                    detector.id,
                    score,
                    remote_peak,
                )
                continue

            detector.is_detected = True
            self.state.record_detection(detector.id)
            self.state.publish(
                {
                    "type": "detection",
                    "model": detector.id,
                    "score": round(score, 4),
                    "timestamp": self.audio_timestamp,
                }
            )
            await self.write_event(
                Detection(name=detector.id, timestamp=self.audio_timestamp).event()
            )
            if self.state.settings.capture:
                await asyncio.to_thread(self._write_capture, detector.id)
            for interpreter in detector.interpreters.values():
                interpreter.reset()
            _LOGGER.debug("Detected %s at %s", detector.id, self.audio_timestamp)

        self.audio_timestamp += chunk.milliseconds

    def _capture_append(self, audio: np.ndarray) -> None:
        self._capture_buffer.append(audio)
        self._capture_samples += len(audio)
        while (
            self._capture_buffer
            and self._capture_samples - len(self._capture_buffer[0])
            >= self._capture_max_samples
        ):
            self._capture_samples -= len(self._capture_buffer.popleft())

    def _write_capture(self, model_id: str) -> None:
        """Save the audio leading up to a detection as a WAV file.

        Real detections and false positives both land here — after a while
        the directory is training data for the next model version.
        """

        capture_dir = self._effective_capture_dir
        if not self._capture_buffer or capture_dir is None:
            return

        samples = np.concatenate(list(self._capture_buffer))
        capture_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = capture_dir / f"{model_id}-{timestamp}-{self.client_id[-6:]}.wav"

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(samples.tobytes())

        _LOGGER.info("Captured detection audio: %s", path.name)
        captures = sorted(capture_dir.glob("*.wav"))
        for old in captures[: max(0, len(captures) - self.capture_keep)]:
            old.unlink(missing_ok=True)

    async def _verify_candidate(
        self, detector: Detector, score: float
    ) -> tuple[bool, float | None]:
        """Confirm a candidate with the configured verifier stages.

        Stage 1 (hybrid satellite + server): a light model triggers cheaply
        and the buffered audio is confirmed by a stronger model/ensemble on
        the central server. Stage 2 (ASR): the audio is transcribed by a
        whisper server and the wake word must appear in the transcript —
        this is what turns a sensitive (high-recall) model into a
        near-zero-false-accept detector. Either stage can run alone.

        Returns (verified, stage-1 verifier peak score or None).
        """

        settings = self.state.settings
        model_verify = settings.verify and settings.verify_url
        asr_verify = settings.verify_asr and settings.verify_asr_url
        if not model_verify and not asr_verify:
            return True, None

        if not self._capture_buffer:
            _LOGGER.warning("Verification enabled but no buffered audio yet")
            return settings.verify_fail_open, None

        wav_bytes = await asyncio.to_thread(self._buffer_as_wav)

        if self._verify_session is None:
            self._verify_session = aiohttp.ClientSession()

        if model_verify:
            verified, remote_peak = await self._verify_candidate_model(
                detector, score, wav_bytes
            )
            if not verified:
                return False, remote_peak
        else:
            remote_peak = None

        if asr_verify:
            verdict, detail = await asr_verify_wake_word(
                self._verify_session, settings, wav_bytes
            )
            if verdict is None:
                _LOGGER.warning(
                    "ASR verifier unreachable (%s); %s candidate %s",
                    detail,
                    "accepting" if settings.verify_fail_open else "rejecting",
                    detector.id,
                )
                return settings.verify_fail_open, remote_peak
            _LOGGER.debug(
                "ASR verifier %s candidate %s (%s)",
                "confirmed" if verdict else "rejected",
                detector.id,
                detail,
            )
            if not verdict:
                return False, remote_peak

        return True, remote_peak

    async def _verify_candidate_model(
        self, detector: Detector, score: float, wav_bytes: bytes
    ) -> tuple[bool, float | None]:
        """Stage 1: confirm the candidate with a remote wake word model."""

        settings = self.state.settings
        assert self._verify_session is not None

        url = settings.verify_url.rstrip("/")
        if not url.endswith("/api/test"):
            url = f"{url}/api/test"
        model = settings.verify_model or detector.id
        headers = (
            {"Authorization": f"Bearer {settings.verify_token}"}
            if settings.verify_token
            else {}
        )
        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="candidate.wav")

        try:
            started = time.monotonic()
            async with self._verify_session.post(
                f"{url}?model={model}",
                data=form,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=settings.verify_timeout),
            ) as response:
                if response.status >= 400:
                    raise RuntimeError(
                        f"verifier returned {response.status}: "
                        f"{(await response.text())[:200]}"
                    )
                result = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as err:
            _LOGGER.warning(
                "Verifier unreachable (%s); %s candidate %s",
                err,
                "accepting" if settings.verify_fail_open else "rejecting",
                detector.id,
            )
            return settings.verify_fail_open, None

        verified = bool(result.get("would_detect"))
        remote_peak = result.get("peak")
        _LOGGER.debug(
            "Verifier %s candidate %s in %.0f ms (local %.3f, remote peak %s)",
            "confirmed" if verified else "rejected",
            detector.id,
            (time.monotonic() - started) * 1000,
            score,
            remote_peak,
        )
        return verified, remote_peak

    def _buffer_as_wav(self) -> bytes:
        samples = np.concatenate(list(self._capture_buffer))
        # Verify on the buffer tail only: scoring cost is ~linear in duration,
        # and leading non-wake audio depresses the verifier's score (measured
        # on real captures: 0.64 with 3 s of context vs 0.99 with 1.5 s).
        max_samples = int(VERIFY_WINDOW_SECONDS * 16000)
        if len(samples) > max_samples:
            samples = samples[-max_samples:]
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(samples.tobytes())
        return buffer.getvalue()

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
            started = time.monotonic()
            result = detector.interpreters[model_id].predict(audio)
            elapsed_ms = (time.monotonic() - started) * 1000
            score = score_from_result(result, model_id)
            self.state.update_score(detector.id, score, elapsed_ms)
            return score

        scores: dict[str, float] = {}
        total_ms = 0.0
        for member in detector.entry.members:
            started = time.monotonic()
            result = detector.interpreters[member.model].predict(audio)
            elapsed_ms = (time.monotonic() - started) * 1000
            total_ms += elapsed_ms
            scores[member.model] = score_from_result(result, member.model)
            self.state.update_score(member.model, scores[member.model], elapsed_ms)

        fused = fuse_scores(detector.entry, scores, self.state.settings.threshold)
        self.state.update_score(detector.id, fused, total_ms)
        return fused

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
