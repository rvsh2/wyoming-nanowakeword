"""Event entity firing when the server detects a wake word."""

from __future__ import annotations

from typing import Any

from homeassistant.components.event import EventEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import SIGNAL_DETECTION
from .entity import NanoWakeWordEntity

EVENT_DETECTION = "detection"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities([NanoWakeWordDetectionEvent(entry)])


class NanoWakeWordDetectionEvent(NanoWakeWordEntity, EventEntity):
    """Fires on every wake word detection, with model and score as data."""

    _attr_event_types = [EVENT_DETECTION]

    def __init__(self, entry: ConfigEntry) -> None:
        super().__init__(entry, "detection")

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_DETECTION.format(self.entry.entry_id),
                self._handle_detection,
            )
        )

    @callback
    def _handle_detection(self, event: dict[str, Any]) -> None:
        self._trigger_event(
            EVENT_DETECTION,
            {
                "model": event.get("model"),
                "score": event.get("score"),
                "timestamp": event.get("timestamp"),
            },
        )
        self.async_write_ha_state()
