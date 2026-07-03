"""Config flow, entity, and service tests against in-process Home Assistant."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.nanowakeword.api import (
    NanoWakeWordApiError,
    NanoWakeWordAuthError,
)
from custom_components.nanowakeword.const import DOMAIN


def entry_data() -> dict[str, object]:
    return {"host": "192.0.2.10", "port": 10401, "token": "secret"}


async def _setup_entry(hass: HomeAssistant) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=entry_data(),
        title="NanoWakeWord (192.0.2.10)",
        unique_id="192.0.2.10:10401",
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_user_flow_creates_entry(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] is FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], entry_data()
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "NanoWakeWord (192.0.2.10)"
    assert result["data"]["host"] == "192.0.2.10"


async def test_user_flow_reports_connection_and_auth_errors(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )

    mock_client.info.side_effect = NanoWakeWordApiError("boom")
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], entry_data()
    )
    assert result["errors"] == {"base": "cannot_connect"}

    mock_client.info.side_effect = NanoWakeWordAuthError("nope")
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], entry_data()
    )
    assert result["errors"] == {"base": "invalid_auth"}


async def test_duplicate_server_is_rejected(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    await _setup_entry(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], entry_data()
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_setup_creates_model_and_score_entities(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    entry = await _setup_entry(hass)
    assert entry.state is ConfigEntryState.LOADED

    registry = er.async_get(hass)

    models_entity = registry.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_models"
    )
    assert models_entity is not None
    models_state = hass.states.get(models_entity)
    assert models_state is not None
    assert models_state.state == "1"
    assert models_state.attributes["models"][0]["id"] == "agata"

    score_entity = registry.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_score_agata"
    )
    assert score_entity is not None
    score_state = hass.states.get(score_entity)
    assert score_state is not None
    assert float(score_state.state) == 0.97
    assert score_state.attributes["detections"] == 5

    for key in ("backup", "reload_models"):
        assert registry.async_get_entity_id(
            "button", DOMAIN, f"{entry.entry_id}_{key}"
        )


async def test_detection_event_entity_fires_on_sse_event(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    from homeassistant.helpers.dispatcher import async_dispatcher_send

    from custom_components.nanowakeword.const import SIGNAL_DETECTION

    entry = await _setup_entry(hass)
    registry = er.async_get(hass)
    event_entity = registry.async_get_entity_id(
        "event", DOMAIN, f"{entry.entry_id}_detection"
    )
    assert event_entity is not None

    async_dispatcher_send(
        hass,
        SIGNAL_DETECTION.format(entry.entry_id),
        {"type": "detection", "model": "agata", "score": 0.98, "timestamp": 1234},
    )
    await hass.async_block_till_done()

    state = hass.states.get(event_entity)
    assert state is not None
    assert state.attributes["event_type"] == "detection"
    assert state.attributes["model"] == "agata"
    assert state.attributes["score"] == 0.98


async def test_backup_service_writes_and_rotates(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    await _setup_entry(hass)

    backup_dir = Path(hass.config.path("nanowakeword"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    for index in range(12):
        (backup_dir / f"nanowakeword-backup-2020010{index:02d}-000000.zip").touch()

    response = await hass.services.async_call(
        DOMAIN, "backup", {}, blocking=True, return_response=True
    )
    await hass.async_block_till_done()

    path = Path(response["path"])
    assert path.read_bytes() == b"zip-bytes"
    assert len(list(backup_dir.glob("nanowakeword-backup-*.zip"))) == 10

    # The user gets visible feedback: a notification and the sensor.
    from homeassistant.components import persistent_notification as pn

    notifications = pn._async_get_or_create_notifications(hass)
    assert any("nanowakeword_backup" in key for key in notifications)

    registry = er.async_get(hass)
    entry = hass.config_entries.async_entries(DOMAIN)[0]
    sensor = registry.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_last_backup"
    )
    assert sensor is not None
    state = hass.states.get(sensor)
    assert state is not None
    assert state.state not in ("unknown", "unavailable")
    assert state.attributes["path"] == str(path)


async def test_stale_score_sensors_are_removed(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    entry = await _setup_entry(hass)
    registry = er.async_get(hass)
    unique_id = f"{entry.entry_id}_score_agata"
    assert registry.async_get_entity_id("sensor", DOMAIN, unique_id) is not None

    # The wake word disappears from the server (deleted via API).
    mock_client.models.return_value = {"models": [], "files": [], "clients": 0}
    await entry.runtime_data.coordinator.async_refresh()
    await hass.async_block_till_done()

    assert registry.async_get_entity_id("sensor", DOMAIN, unique_id) is None


async def test_restore_service_sends_backup_to_server(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    await _setup_entry(hass)

    backup_dir = Path(hass.config.path("nanowakeword"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "nanowakeword-backup-20260101-000000.zip"
    backup_path.write_bytes(b"backup-bytes")

    await hass.services.async_call(
        DOMAIN,
        "restore",
        {"path": str(backup_path)},
        blocking=True,
    )

    mock_client.restore.assert_awaited_once_with(b"backup-bytes")


async def test_reload_models_service_calls_server(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    await _setup_entry(hass)

    await hass.services.async_call(DOMAIN, "reload_models", {}, blocking=True)

    mock_client.reload.assert_awaited_once()


async def test_setting_switch_patches_server(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    entry = await _setup_entry(hass)
    registry = er.async_get(hass)

    switch = registry.async_get_entity_id(
        "switch", DOMAIN, f"{entry.entry_id}_verification"
    )
    assert switch is not None
    assert hass.states.get(switch).state == "off"

    await hass.services.async_call(
        "switch", "turn_on", {"entity_id": switch}, blocking=True
    )
    await hass.async_block_till_done()

    mock_client.patch_settings.assert_called_with({"verify": True})
    assert hass.states.get(switch).state == "on"

    cascade = registry.async_get_entity_id(
        "switch", DOMAIN, f"{entry.entry_id}_cascade"
    )
    assert cascade is not None


async def test_setting_number_patches_server(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    entry = await _setup_entry(hass)
    registry = er.async_get(hass)

    number = registry.async_get_entity_id(
        "number", DOMAIN, f"{entry.entry_id}_threshold"
    )
    assert number is not None
    assert float(hass.states.get(number).state) == 0.95

    await hass.services.async_call(
        "number",
        "set_value",
        {"entity_id": number, "value": 0.7},
        blocking=True,
    )
    await hass.async_block_till_done()

    mock_client.patch_settings.assert_called_with({"threshold": 0.7})
    assert float(hass.states.get(number).state) == 0.7

    trigger = registry.async_get_entity_id(
        "number", DOMAIN, f"{entry.entry_id}_trigger_level"
    )
    await hass.services.async_call(
        "number",
        "set_value",
        {"entity_id": trigger, "value": 2},
        blocking=True,
    )
    await hass.async_block_till_done()
    mock_client.patch_settings.assert_called_with({"trigger_level": 2})
