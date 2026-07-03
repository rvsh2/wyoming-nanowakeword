"""Number entities for the server's detection settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .api import NanoWakeWordApiError
from .entity import NanoWakeWordEntity


@dataclass(frozen=True)
class SettingNumber:
    key: str  # translation key == settings field
    minimum: float
    maximum: float
    step: float
    integer: bool = False


NUMBERS = [
    SettingNumber("threshold", 0.0, 1.0, 0.01),
    SettingNumber("trigger_level", 1, 10, 1, integer=True),
    SettingNumber("refractory_seconds", 0.0, 30.0, 0.5),
    SettingNumber("vad_threshold", 0.0, 1.0, 0.01),
    SettingNumber("gate_threshold", 0.0, 1.0, 0.01),
    SettingNumber("verify_timeout", 0.5, 30.0, 0.5),
]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities(
        NanoWakeWordSettingNumber(entry, description) for description in NUMBERS
    )


class NanoWakeWordSettingNumber(NanoWakeWordEntity, NumberEntity):
    """One numeric server setting, changed through the HTTP API."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_mode = NumberMode.BOX

    def __init__(self, entry: ConfigEntry, description: SettingNumber) -> None:
        super().__init__(entry, description.key)
        self._description = description
        self._attr_native_min_value = description.minimum
        self._attr_native_max_value = description.maximum
        self._attr_native_step = description.step

    @property
    def available(self) -> bool:
        return super().available and self._settings is not None

    @property
    def native_value(self) -> float | None:
        settings = self._settings
        if not settings:
            return None
        value = settings.get(self._description.key)
        return None if value is None else float(value)

    async def async_set_native_value(self, value: float) -> None:
        payload: Any = int(value) if self._description.integer else value
        try:
            await self.coordinator.async_apply_settings(
                {self._description.key: payload}
            )
        except NanoWakeWordApiError as err:
            raise HomeAssistantError(f"Cannot change setting: {err}") from err

    @property
    def _settings(self) -> dict[str, Any] | None:
        return self.coordinator.data.get("settings")
