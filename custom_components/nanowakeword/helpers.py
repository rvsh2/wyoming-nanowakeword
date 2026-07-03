"""Shared helpers for the NanoWakeWord integration."""

from __future__ import annotations

from pathlib import Path

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import BACKUP_DIR


async def async_create_backup(hass: HomeAssistant, entry: ConfigEntry) -> Path:
    """Download a model backup and store it under /config/nanowakeword."""

    content = await entry.runtime_data.client.backup()

    timestamp = dt_util.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = Path(hass.config.path(BACKUP_DIR))
    path = backup_dir / f"nanowakeword-backup-{timestamp}.zip"

    def _write() -> None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    await hass.async_add_executor_job(_write)
    return path


async def async_notify_model_change(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Refresh our data and reload Wyoming entries pointed at this server.

    Home Assistant caches the wake word list from the Wyoming `describe`
    response, so the Wyoming config entries for this host must be reloaded
    before a newly uploaded wake word can be selected in Voice Assist.
    """

    await entry.runtime_data.coordinator.async_request_refresh()

    host = entry.data[CONF_HOST]
    for wyoming_entry in hass.config_entries.async_entries("wyoming"):
        if wyoming_entry.data.get("host") == host:
            hass.async_create_task(
                hass.config_entries.async_reload(wyoming_entry.entry_id)
            )
