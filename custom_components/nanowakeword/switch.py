"""Switches controlling server behavior (persisted server-side)."""

from __future__ import annotations

from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .api import NanoWakeWordApiError
from .entity import NanoWakeWordEntity

# (translation key, settings field)
SWITCHES = [
    ("verification", "verify"),
    ("cascade", "cascade"),
    ("capture", "capture"),
]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities(
        NanoWakeWordSettingSwitch(entry, key, field) for key, field in SWITCHES
    )


class NanoWakeWordSettingSwitch(NanoWakeWordEntity, SwitchEntity):
    """One boolean server setting, changed through the HTTP API."""

    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, entry: ConfigEntry, key: str, field: str) -> None:
        super().__init__(entry, key)
        self._field = field

    @property
    def available(self) -> bool:
        return super().available and self._settings is not None

    @property
    def is_on(self) -> bool | None:
        settings = self._settings
        return bool(settings.get(self._field)) if settings else None

    async def async_turn_on(self, **kwargs: Any) -> None:
        await self._apply(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self._apply(False)

    async def _apply(self, value: bool) -> None:
        try:
            await self.coordinator.async_apply_settings({self._field: value})
        except NanoWakeWordApiError as err:
            raise HomeAssistantError(f"Cannot change setting: {err}") from err

    @property
    def _settings(self) -> dict[str, Any] | None:
        return self.coordinator.data.get("settings")
