from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Describe
from wyoming.wake import Detect, Detection, NotDetected

from wyoming_nanowakeword.handler import NanoWakeWordEventHandler
from wyoming_nanowakeword.state import State


@dataclass
class FakeResult:
    score: float
    scores: dict[str, float]

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.scores.get(key, default)


class FakeInterpreter:
    def __init__(self, score: float = 0.0, model_id: str = "hey_home") -> None:
        self.score = score
        self.model_id = model_id
        self.audio_lengths: list[int] = []
        self.reset_count = 0

    def predict(self, audio: Any) -> FakeResult:
        self.audio_lengths.append(len(audio))
        return FakeResult(score=self.score, scores={self.model_id: self.score})

    def reset(self) -> None:
        self.reset_count += 1


class EventCollector:
    def __init__(self) -> None:
        self.events = []

    async def write_event(self, event) -> None:
        self.events.append(event)


def _handler(
    tmp_path: Path,
    interpreter: FakeInterpreter,
    threshold: float = 0.95,
    trigger_level: int = 1,
    refractory_seconds: float = 2.0,
) -> tuple[NanoWakeWordEventHandler, EventCollector]:
    (tmp_path / "hey_home.onnx").touch()
    state = State(model_dirs=[tmp_path], default_model="hey_home")
    state.refresh()
    collector = EventCollector()

    handler = NanoWakeWordEventHandler(
        threshold,
        trigger_level,
        refractory_seconds,
        0.0,
        False,
        0.3,
        state,
        lambda **kwargs: interpreter,
        "test",
        collector,
    )
    handler.write_event = collector.write_event  # type: ignore[method-assign]
    return handler, collector


@pytest.mark.asyncio
async def test_describe_lists_nanowakeword_program(tmp_path: Path) -> None:
    handler, collector = _handler(tmp_path, FakeInterpreter())

    await handler.handle_event(Describe().event())

    assert collector.events
    assert collector.events[0].type == "info"
    assert collector.events[0].data["wake"][0]["name"] == "nanowakeword"
    assert collector.events[0].data["wake"][0]["models"][0]["name"] == "hey_home"


@pytest.mark.asyncio
async def test_audio_chunk_passes_int16_samples_to_interpreter(tmp_path: Path) -> None:
    interpreter = FakeInterpreter(score=0.0)
    handler, _collector = _handler(tmp_path, interpreter)

    await handler.handle_event(Detect(names=["hey_home"]).event())
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(
        AudioChunk(
            rate=16000,
            width=2,
            channels=1,
            audio=(b"\x01\x00" * 333),
        ).event()
    )

    assert interpreter.audio_lengths == [333]


@pytest.mark.asyncio
async def test_detection_emits_event_and_resets_interpreter(tmp_path: Path) -> None:
    interpreter = FakeInterpreter(score=0.99)
    handler, collector = _handler(tmp_path, interpreter, threshold=0.95)

    await handler.handle_event(Detect(names=[]).event())
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(
        AudioChunk(
            rate=16000,
            width=2,
            channels=1,
            audio=(b"\x01\x00" * 1280),
        ).event()
    )

    detections = [event for event in collector.events if Detection.is_type(event.type)]
    assert len(detections) == 1
    assert detections[0].data["name"] == "hey_home"
    assert interpreter.reset_count == 2


@pytest.mark.asyncio
async def test_trigger_level_requires_multiple_hits(tmp_path: Path) -> None:
    interpreter = FakeInterpreter(score=0.99)
    handler, collector = _handler(tmp_path, interpreter, trigger_level=2)

    await handler.handle_event(Detect(names=["hey_home"]).event())
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    for _ in range(2):
        await handler.handle_event(
            AudioChunk(
                rate=16000,
                width=2,
                channels=1,
                audio=(b"\x01\x00" * 1280),
            ).event()
        )

    detections = [event for event in collector.events if Detection.is_type(event.type)]
    assert len(detections) == 1


@pytest.mark.asyncio
async def test_audio_stop_emits_not_detected_without_detection(tmp_path: Path) -> None:
    handler, collector = _handler(tmp_path, FakeInterpreter(score=0.0))

    await handler.handle_event(Detect(names=["hey_home"]).event())
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(AudioStop().event())

    assert any(NotDetected.is_type(event.type) for event in collector.events)


@pytest.mark.asyncio
async def test_ensemble_requires_primary_and_verifier(tmp_path: Path) -> None:
    (tmp_path / "agata_ebranchformer_v1.onnx").touch()
    (tmp_path / "agata_transformer_v1.onnx").touch()
    (tmp_path / "models.yaml").write_text(
        """
models:
  agata:
    phrase: "Agata"
    language: "pl"
    architecture: "ensemble:e_branchformer+transformer"
    fusion: "primary_and_verifier"
    members:
      - model: "agata_ebranchformer"
        role: "primary"
        threshold: 0.97
      - model: "agata_transformer"
        role: "verifier"
        threshold: 0.90
  agata_ebranchformer:
    hidden: true
  agata_transformer:
    hidden: true
""",
        encoding="utf-8",
    )
    state = State(model_dirs=[tmp_path], default_model="agata")
    state.refresh()
    collector = EventCollector()
    interpreters = {
        "agata_ebranchformer": FakeInterpreter(0.98, "agata_ebranchformer"),
        "agata_transformer": FakeInterpreter(0.91, "agata_transformer"),
    }

    def factory(model: str, **_kwargs: Any) -> FakeInterpreter:
        model_id = Path(model).stem.replace("_v1", "")
        return interpreters[model_id]

    handler = NanoWakeWordEventHandler(
        0.95,
        1,
        2.0,
        0.0,
        False,
        0.3,
        state,
        factory,
        "test",
        collector,
    )
    handler.write_event = collector.write_event  # type: ignore[method-assign]

    await handler.handle_event(Detect(names=["agata"]).event())
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(
        AudioChunk(
            rate=16000,
            width=2,
            channels=1,
            audio=(b"\x01\x00" * 1280),
        ).event()
    )

    detections = [event for event in collector.events if Detection.is_type(event.type)]
    assert len(detections) == 1
    assert detections[0].data["name"] == "agata"


@pytest.mark.asyncio
async def test_ensemble_rejects_when_verifier_is_low(tmp_path: Path) -> None:
    (tmp_path / "agata_ebranchformer_v1.onnx").touch()
    (tmp_path / "agata_transformer_v1.onnx").touch()
    (tmp_path / "models.yaml").write_text(
        """
models:
  agata:
    members:
      - model: "agata_ebranchformer"
        role: "primary"
        threshold: 0.97
      - model: "agata_transformer"
        role: "verifier"
        threshold: 0.90
  agata_ebranchformer:
    hidden: true
  agata_transformer:
    hidden: true
""",
        encoding="utf-8",
    )
    state = State(model_dirs=[tmp_path], default_model="agata")
    state.refresh()
    collector = EventCollector()
    interpreters = {
        "agata_ebranchformer": FakeInterpreter(0.99, "agata_ebranchformer"),
        "agata_transformer": FakeInterpreter(0.20, "agata_transformer"),
    }

    def factory(model: str, **_kwargs: Any) -> FakeInterpreter:
        model_id = Path(model).stem.replace("_v1", "")
        return interpreters[model_id]

    handler = NanoWakeWordEventHandler(
        0.95,
        1,
        2.0,
        0.0,
        False,
        0.3,
        state,
        factory,
        "test",
        collector,
    )
    handler.write_event = collector.write_event  # type: ignore[method-assign]

    await handler.handle_event(Detect(names=["agata"]).event())
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(
        AudioChunk(
            rate=16000,
            width=2,
            channels=1,
            audio=(b"\x01\x00" * 1280),
        ).event()
    )

    assert not any(Detection.is_type(event.type) for event in collector.events)
