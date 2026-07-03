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
        "test",
        collector,
        threshold=threshold,
        trigger_level=trigger_level,
        refractory_seconds=refractory_seconds,
        vad_threshold=0.0,
        cascade=False,
        gate_threshold=0.3,
        state=state,
        interpreter_factory=lambda **kwargs: interpreter,
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
async def test_trigger_level_resets_on_miss(tmp_path: Path) -> None:
    interpreter = FakeInterpreter(score=0.99)
    handler, collector = _handler(tmp_path, interpreter, trigger_level=2)

    await handler.handle_event(Detect(names=["hey_home"]).event())
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())

    async def feed(score: float) -> None:
        interpreter.score = score
        await handler.handle_event(
            AudioChunk(
                rate=16000,
                width=2,
                channels=1,
                audio=(b"\x01\x00" * 1280),
            ).event()
        )

    # One hit, then a miss resets the streak, so two hits are still required.
    await feed(0.99)
    await feed(0.0)
    await feed(0.99)
    assert not any(Detection.is_type(event.type) for event in collector.events)

    # Two consecutive hits now fire a detection.
    await feed(0.99)
    detections = [event for event in collector.events if Detection.is_type(event.type)]
    assert len(detections) == 1


@pytest.mark.asyncio
async def test_per_model_threshold_override(tmp_path: Path) -> None:
    (tmp_path / "hey_home.onnx").touch()
    (tmp_path / "models.yaml").write_text(
        'models:\n  hey_home:\n    threshold: 0.5\n    trigger_level: 2\n',
        encoding="utf-8",
    )
    state = State(model_dirs=[tmp_path], default_model="hey_home")
    state.refresh()
    collector = EventCollector()
    interpreter = FakeInterpreter(score=0.6)  # above 0.5, below global 0.95

    handler = NanoWakeWordEventHandler(
        "test",
        collector,
        threshold=0.95,
        trigger_level=1,
        refractory_seconds=2.0,
        vad_threshold=0.0,
        cascade=False,
        gate_threshold=0.3,
        state=state,
        interpreter_factory=lambda **kwargs: interpreter,
    )
    handler.write_event = collector.write_event  # type: ignore[method-assign]

    await handler.handle_event(Detect(names=["hey_home"]).event())
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    for _ in range(2):  # trigger_level override: 2 consecutive hits needed
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
async def test_capture_writes_wav_on_detection(tmp_path: Path) -> None:
    (tmp_path / "hey_home.onnx").touch()
    capture_dir = tmp_path / "captures"
    state = State(model_dirs=[tmp_path], default_model="hey_home")
    state.refresh()
    collector = EventCollector()
    interpreter = FakeInterpreter(score=0.99)

    handler = NanoWakeWordEventHandler(
        "test",
        collector,
        threshold=0.95,
        trigger_level=1,
        refractory_seconds=2.0,
        vad_threshold=0.0,
        cascade=False,
        gate_threshold=0.3,
        state=state,
        interpreter_factory=lambda **kwargs: interpreter,
        capture_dir=capture_dir,
        capture_seconds=1.0,
    )
    handler.write_event = collector.write_event  # type: ignore[method-assign]

    await handler.handle_event(Detect(names=["hey_home"]).event())
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(
        AudioChunk(
            rate=16000,
            width=2,
            channels=1,
            audio=(b"\x01\x00" * 1280),
        ).event()
    )

    captures = list(capture_dir.glob("hey_home-*.wav"))
    assert len(captures) == 1

    import wave

    with wave.open(str(captures[0]), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 1280


@pytest.mark.asyncio
async def test_detection_publishes_event(tmp_path: Path) -> None:
    interpreter = FakeInterpreter(score=0.99)
    handler, _collector = _handler(tmp_path, interpreter)
    queue = handler.state.subscribe()

    await handler.handle_event(Detect(names=["hey_home"]).event())
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(
        AudioChunk(
            rate=16000,
            width=2,
            channels=1,
            audio=(b"\x01\x00" * 1280),
        ).event()
    )

    event = queue.get_nowait()
    assert event["type"] == "detection"
    assert event["model"] == "hey_home"
    assert event["score"] == 0.99


@pytest.mark.asyncio
async def test_detectors_reload_after_model_refresh(tmp_path: Path) -> None:
    interpreter = FakeInterpreter(score=0.0)
    handler, _collector = _handler(tmp_path, interpreter)

    await handler.handle_event(Detect(names=["hey_home"]).event())
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    old_generation = handler.detectors["hey_home"].generation

    # A model upload/restore through the HTTP API bumps the generation; the
    # next pipeline run must reload detectors from the new registry.
    handler.state.refresh()
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())

    detector = handler.detectors["hey_home"]
    assert detector.generation == handler.state.generation
    assert detector.generation != old_generation


@pytest.mark.asyncio
async def test_audio_without_detect_uses_default_model(tmp_path: Path) -> None:
    interpreter = FakeInterpreter(score=0.99)
    handler, collector = _handler(tmp_path, interpreter)

    # No Detect event: the default model must be loaded on AudioStart.
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


@pytest.mark.asyncio
async def test_unknown_requested_name_falls_back_to_default(tmp_path: Path) -> None:
    interpreter = FakeInterpreter(score=0.99)
    handler, collector = _handler(tmp_path, interpreter)

    await handler.handle_event(Detect(names=["no_such_model"]).event())
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


@pytest.mark.asyncio
async def test_cascade_passes_lite_model_as_gate(tmp_path: Path) -> None:
    (tmp_path / "hey_home.onnx").touch()
    (tmp_path / "hey_home_lite.onnx").touch()
    state = State(model_dirs=[tmp_path], default_model="hey_home")
    state.refresh()
    collector = EventCollector()

    captured: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> FakeInterpreter:
        captured.append(kwargs)
        return FakeInterpreter()

    handler = NanoWakeWordEventHandler(
        "test",
        collector,
        threshold=0.95,
        trigger_level=1,
        refractory_seconds=2.0,
        vad_threshold=0.0,
        cascade=True,
        gate_threshold=0.3,
        state=state,
        interpreter_factory=factory,
    )
    handler.write_event = collector.write_event  # type: ignore[method-assign]

    await handler.handle_event(Detect(names=["hey_home"]).event())

    assert captured
    assert captured[0]["cascade"] is True
    assert captured[0]["gate_model"] == str(tmp_path / "hey_home_lite.onnx")


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
        "test",
        collector,
        threshold=0.95,
        trigger_level=1,
        refractory_seconds=2.0,
        vad_threshold=0.0,
        cascade=False,
        gate_threshold=0.3,
        state=state,
        interpreter_factory=factory,
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
        "test",
        collector,
        threshold=0.95,
        trigger_level=1,
        refractory_seconds=2.0,
        vad_threshold=0.0,
        cascade=False,
        gate_threshold=0.3,
        state=state,
        interpreter_factory=factory,
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


@pytest.mark.asyncio
async def test_ensemble_member_threshold_zero_is_respected(tmp_path: Path) -> None:
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
        threshold: 0.0
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
        # Far below the global threshold: only the explicit 0.0 lets it pass.
        "agata_transformer": FakeInterpreter(0.20, "agata_transformer"),
    }

    def factory(model: str, **_kwargs: Any) -> FakeInterpreter:
        model_id = Path(model).stem.replace("_v1", "")
        return interpreters[model_id]

    handler = NanoWakeWordEventHandler(
        "test",
        collector,
        threshold=0.95,
        trigger_level=1,
        refractory_seconds=2.0,
        vad_threshold=0.0,
        cascade=False,
        gate_threshold=0.3,
        state=state,
        interpreter_factory=factory,
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
