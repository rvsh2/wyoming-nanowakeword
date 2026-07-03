"""Base entity for the NanoWakeWord integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import NanoWakeWordCoordinator


class NanoWakeWordEntity(CoordinatorEntity[NanoWakeWordCoordinator]):
    """Entity tied to one NanoWakeWord server."""

    _attr_has_entity_name = True

    def __init__(self, entry: ConfigEntry, key: str) -> None:
        super().__init__(entry.runtime_data.coordinator)
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_translation_key = key
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="rvsh2",
            model="wyoming-nanowakeword",
            configuration_url=(
                f"http://{entry.data[CONF_HOST]}:{entry.data[CONF_PORT]}/api/info"
            ),
        )
