"""Hybrid satellite + server verification tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from test_handler import EventCollector, FakeInterpreter  # noqa: E402 - pytest sys.path
from wyoming.audio import AudioChunk, AudioStart
from wyoming.wake import Detect, Detection

from wyoming_nanowakeword.handler import NanoWakeWordEventHandler
from wyoming_nanowakeword.state import State


@asynccontextmanager
async def _verifier(would_detect: bool) -> AsyncIterator[tuple[str, list[dict]]]:
    """A fake central server exposing POST /api/test."""

    requests: list[dict] = []

    async def test_endpoint(request: web.Request) -> web.Response:
        requests.append(
            {
                "model": request.query.get("model"),
                "auth": request.headers.get("Authorization"),
            }
        )
        return web.json_response({"would_detect": would_detect, "peak": 0.5})

    app = web.Application()
    app.add_routes([web.post("/api/test", test_endpoint)])
    server = TestServer(app)
    await server.start_server()
    try:
        yield f"http://127.0.0.1:{server.port}", requests
    finally:
        await server.close()


async def _handler_with_verify(
    tmp_path: Path, verify_url: str
) -> tuple[NanoWakeWordEventHandler, EventCollector]:
    (tmp_path / "hey_home.onnx").touch()
    state = State(model_dirs=[tmp_path], default_model="hey_home")
    state.refresh()
    state.settings.verify = True
    state.settings.verify_url = verify_url
    state.settings.verify_token = "sekret"
    state.settings.verify_model = "agata"
    collector = EventCollector()

    handler = NanoWakeWordEventHandler(
        "test",
        collector,
        threshold=0.95,
        trigger_level=1,
        refractory_seconds=2.0,
        state=state,
        interpreter_factory=lambda **kwargs: FakeInterpreter(score=0.99),
    )
    handler.write_event = collector.write_event  # type: ignore[method-assign]
    return handler, collector


async def _stream(handler: NanoWakeWordEventHandler) -> None:
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


@pytest.mark.asyncio
async def test_confirmed_candidate_emits_detection(tmp_path: Path) -> None:
    async with _verifier(would_detect=True) as (url, requests):
        handler, collector = await _handler_with_verify(tmp_path, url)
        await _stream(handler)
        await handler.disconnect()

    detections = [event for event in collector.events if Detection.is_type(event.type)]
    assert len(detections) == 1
    # The verifier got the configured central model id and the token.
    assert requests[0]["model"] == "agata"
    assert requests[0]["auth"] == "Bearer sekret"


@pytest.mark.asyncio
async def test_rejected_candidate_is_suppressed(tmp_path: Path) -> None:
    async with _verifier(would_detect=False) as (url, _requests):
        handler, collector = await _handler_with_verify(tmp_path, url)
        handler.state.settings.capture = True
        queue = handler.state.subscribe()
        await _stream(handler)
        await handler.disconnect()

    assert not any(Detection.is_type(event.type) for event in collector.events)
    event = queue.get_nowait()
    assert event["type"] == "rejected"
    assert event["remote_peak"] == 0.5
    assert handler.state.scores["hey_home"]["rejections"] == 1
    assert handler.state.scores["hey_home"]["detections"] == 0
    # Rejected audio is captured for tuning/training.
    assert list((tmp_path / "captures").glob("hey_home-rejected-*.wav"))


@pytest.mark.asyncio
async def test_unreachable_verifier_fails_open_by_default(tmp_path: Path) -> None:
    handler, collector = await _handler_with_verify(
        tmp_path, "http://127.0.0.1:1"  # nothing listens here
    )
    handler.state.settings.verify_timeout = 0.2
    await _stream(handler)

    assert any(Detection.is_type(event.type) for event in collector.events)

    # Strict mode: suppress instead.
    handler.state.settings.verify_fail_open = False
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    collector.events.clear()
    await handler.handle_event(
        AudioChunk(
            rate=16000, width=2, channels=1, audio=(b"\x01\x00" * 1280)
        ).event()
    )
    await handler.disconnect()
    assert not any(Detection.is_type(event.type) for event in collector.events)


@pytest.mark.asyncio
async def test_verification_disabled_passes_through(tmp_path: Path) -> None:
    handler, collector = await _handler_with_verify(tmp_path, "http://127.0.0.1:1")
    handler.state.settings.verify = False  # the HA switch turned it off
    await _stream(handler)
    await handler.disconnect()

    assert any(Detection.is_type(event.type) for event in collector.events)
