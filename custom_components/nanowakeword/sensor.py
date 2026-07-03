"""Sensors exposing the served wake word models and live scores."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from homeassistant.components.sensor import (
    RestoreSensor,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import DOMAIN, SIGNAL_BACKUP
from .entity import NanoWakeWordEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator = entry.runtime_data.coordinator
    async_add_entities(
        [
            NanoWakeWordModelsSensor(entry),
            NanoWakeWordClientsSensor(entry),
            NanoWakeWordLastBackupSensor(entry),
        ]
    )

    known_models: set[str] = set()

    @callback
    def _sync_score_sensors() -> None:
        # Only trust a fresh, successful poll: a server hiccup must not
        # delete every score sensor.
        if not coordinator.last_update_success:
            return

        current = {
            model["id"] for model in (coordinator.data or {}).get("models", [])
        }

        new_models = current - known_models
        if new_models:
            async_add_entities(
                NanoWakeWordScoreSensor(entry, model_id) for model_id in new_models
            )
            known_models.update(new_models)

        # Deleted wake words: remove their sensors instead of leaving them
        # unavailable forever.
        stale = known_models - current
        if stale:
            registry = er.async_get(hass)
            for model_id in stale:
                entity_id = registry.async_get_entity_id(
                    "sensor", DOMAIN, f"{entry.entry_id}_score_{model_id}"
                )
                if entity_id:
                    registry.async_remove(entity_id)
            known_models.difference_update(stale)

    _sync_score_sensors()
    entry.async_on_unload(coordinator.async_add_listener(_sync_score_sensors))


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


class NanoWakeWordClientsSensor(NanoWakeWordEntity, SensorEntity):
    """Number of Wyoming clients (satellites) connected to the server."""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, entry: ConfigEntry) -> None:
        super().__init__(entry, "clients")

    @property
    def native_value(self) -> int:
        return self.coordinator.data.get("clients", 0)


class NanoWakeWordLastBackupSensor(NanoWakeWordEntity, RestoreSensor):
    """When the last model backup was saved, with its path as attribute."""

    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, entry: ConfigEntry) -> None:
        super().__init__(entry, "last_backup")

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()

        # Survive HA restarts: the backup file is still on disk.
        if self.entry.runtime_data.last_backup_at is None:
            if (last_state := await self.async_get_last_state()) is not None:
                self.entry.runtime_data.last_backup_at = dt_util.parse_datetime(
                    last_state.state
                )
                self.entry.runtime_data.last_backup_path = last_state.attributes.get(
                    "path"
                )

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_BACKUP.format(self.entry.entry_id),
                self.async_write_ha_state,
            )
        )

    @property
    def native_value(self) -> datetime | None:
        return self.entry.runtime_data.last_backup_at

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {"path": self.entry.runtime_data.last_backup_path}


class NanoWakeWordScoreSensor(NanoWakeWordEntity, SensorEntity):
    """Recent peak inference score of one wake word — for threshold tuning."""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_suggested_display_precision = 3

    def __init__(self, entry: ConfigEntry, model_id: str) -> None:
        super().__init__(entry, f"score_{model_id}")
        self.model_id = model_id
        self._attr_translation_key = "peak_score"
        self._attr_translation_placeholders = {"model": model_id}

    @property
    def available(self) -> bool:
        return super().available and self._stats is not None

    @property
    def native_value(self) -> float | None:
        stats = self._stats
        return stats["peak"] if stats else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        stats = self._stats or {}
        # Ensemble member scores help explain why a fused score stayed low.
        members = next(
            (
                model["members"]
                for model in self.coordinator.data.get("models", [])
                if model["id"] == self.model_id
            ),
            [],
        )
        scores = self.coordinator.data.get("scores", {})
        return {
            "last_score": stats.get("last"),
            "detections": stats.get("detections"),
            "peak_age_seconds": stats.get("peak_age_seconds"),
            "avg_inference_ms": stats.get("avg_inference_ms"),
            "member_scores": {
                member: scores.get(member, {}).get("peak") for member in members
            },
            "member_inference_ms": {
                member: scores.get(member, {}).get("avg_inference_ms")
                for member in members
            },
        }

    @property
    def _stats(self) -> dict[str, Any] | None:
        return self.coordinator.data.get("scores", {}).get(self.model_id)
