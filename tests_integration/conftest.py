"""Fixtures for Home Assistant integration tests.

These run against a real (in-process) Home Assistant via
pytest-homeassistant-custom-component, in a separate environment from the
server unit tests — see the integration-tests job in CI.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations: None,
) -> Generator[None, None, None]:
    yield


@pytest.fixture
def mock_client() -> Generator[MagicMock, None, None]:
    """Mock the API client everywhere the integration constructs one."""

    client = MagicMock()
    client.info = AsyncMock(
        return_value={"server": "wyoming-nanowakeword", "version": "0.1.0"}
    )
    client.models = AsyncMock(
        return_value={
            "models": [
                {
                    "id": "agata",
                    "phrase": "Agata",
                    "language": "pl",
                    "architecture": "ensemble:e_branchformer+conformer",
                    "version": "v1",
                    "ensemble": True,
                    "file": None,
                    "gate": None,
                    "members": ["agata_ebranchformer", "agata_conformer"],
                }
            ],
            "files": ["agata_ebranchformer_v1.onnx", "models.yaml"],
        }
    )
    client.scores = AsyncMock(
        return_value={
            "scores": {
                "agata": {
                    "last": 0.12,
                    "peak": 0.97,
                    "peak_age_seconds": 3.2,
                    "detections": 5,
                }
            }
        }
    )
    settings_state = {
        "threshold": 0.95,
        "trigger_level": 1,
        "refractory_seconds": 2.0,
        "vad_threshold": 0.0,
        "cascade": False,
        "gate_threshold": 0.3,
        "capture": False,
        "verify": False,
        "verify_url": "",
        "verify_token": False,
        "verify_model": "",
        "verify_timeout": 3.0,
        "verify_fail_open": True,
    }

    async def _get_settings() -> dict[str, Any]:
        return dict(settings_state)

    async def _patch_settings(changes: dict[str, Any]) -> dict[str, Any]:
        settings_state.update(changes)
        return dict(settings_state)

    client.get_settings = MagicMock(side_effect=_get_settings)
    client.patch_settings = MagicMock(side_effect=_patch_settings)
    client.backup = AsyncMock(return_value=b"zip-bytes")
    client.restore = AsyncMock(return_value={})
    client.upload_model = AsyncMock(return_value={})
    client.delete_model = AsyncMock(return_value={})
    client.reload = AsyncMock(return_value={})
    client.test_recording = AsyncMock(
        return_value={
            "model": "agata",
            "duration_seconds": 1.2,
            "chunk_ms": 80,
            "threshold": 0.95,
            "peak": 0.97,
            "would_detect": True,
            "member_peaks": {"agata_ebranchformer": 0.98},
            "fused_series": [0.1, 0.97],
            "member_series": {"agata_ebranchformer": [0.1, 0.98]},
        }
    )

    async def _no_events() -> AsyncIterator[dict[str, Any]]:
        # Async generator that never yields: keeps the SSE listener idle.
        if False:
            yield {}
        await asyncio.Event().wait()

    client.listen_events = _no_events

    with (
        patch(
            "custom_components.nanowakeword.NanoWakeWordClient",
            return_value=client,
        ),
        patch(
            "custom_components.nanowakeword.config_flow.NanoWakeWordClient",
            return_value=client,
        ),
    ):
        yield client
