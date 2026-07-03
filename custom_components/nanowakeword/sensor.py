"""Sensors exposing the served wake word models."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .entity import NanoWakeWordEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities([NanoWakeWordModelsSensor(entry)])


class NanoWakeWordModelsSensor(NanoWakeWordEntity, SensorEntity):
    """Number of wake words served, with the model list as attributes."""

    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, entry: ConfigEntry) -> None:
        super().__init__(entry, "models")

    @property
    def native_value(self) -> int:
        return len(self.coordinator.data.get("models", []))

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        return {
            "models": [
                {
                    "id": model["id"],
                    "phrase": model["phrase"],
                    "language": model["language"],
                    "architecture": model["architecture"],
                    "ensemble": model["ensemble"],
                }
                for model in data.get("models", [])
            ],
            "files": data.get("files", []),
        }
