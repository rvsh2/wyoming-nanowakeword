"""Fixtures for Home Assistant integration tests.

These run against a real (in-process) Home Assistant via
pytest-homeassistant-custom-component, in a separate environment from the
server unit tests — see the integration-tests job in CI.
"""

from __future__ import annotations

from collections.abc import Generator
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
    client.backup = AsyncMock(return_value=b"zip-bytes")
    client.restore = AsyncMock(return_value={})
    client.upload_model = AsyncMock(return_value={})
    client.delete_model = AsyncMock(return_value={})
    client.reload = AsyncMock(return_value={})

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
