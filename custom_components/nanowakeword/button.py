"""Buttons for backing up models and reloading the model directory."""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .api import NanoWakeWordApiError
from .entity import NanoWakeWordEntity
from .helpers import async_create_backup, async_notify_model_change


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities(
        [NanoWakeWordBackupButton(entry), NanoWakeWordReloadButton(entry)]
    )


class NanoWakeWordBackupButton(NanoWakeWordEntity, ButtonEntity):
    """Save a backup of the model directory to /config/nanowakeword."""

    def __init__(self, entry: ConfigEntry) -> None:
        super().__init__(entry, "backup")

    async def async_press(self) -> None:
        try:
            await async_create_backup(self.hass, self.entry)
        except NanoWakeWordApiError as err:
            raise HomeAssistantError(f"Backup failed: {err}") from err


class NanoWakeWordReloadButton(NanoWakeWordEntity, ButtonEntity):
    """Re-scan the model directory on the server."""

    def __init__(self, entry: ConfigEntry) -> None:
        super().__init__(entry, "reload_models")

    async def async_press(self) -> None:
        try:
            await self.entry.runtime_data.client.reload()
        except NanoWakeWordApiError as err:
            raise HomeAssistantError(f"Reload failed: {err}") from err
        await async_notify_model_change(self.hass, self.entry)
