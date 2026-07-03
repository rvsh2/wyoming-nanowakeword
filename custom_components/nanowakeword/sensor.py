"""Sensors exposing the served wake word models and live scores."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .entity import NanoWakeWordEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator = entry.runtime_data.coordinator
    async_add_entities([NanoWakeWordModelsSensor(entry)])

    known_models: set[str] = set()

    @callback
    def _add_score_sensors() -> None:
        new_entities = [
            NanoWakeWordScoreSensor(entry, model["id"])
            for model in (coordinator.data or {}).get("models", [])
            if model["id"] not in known_models
        ]
        known_models.update(sensor.model_id for sensor in new_entities)
        if new_entities:
            async_add_entities(new_entities)

    _add_score_sensors()
    entry.async_on_unload(coordinator.async_add_listener(_add_score_sensors))


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
            "member_scores": {
                member: scores.get(member, {}).get("peak") for member in members
            },
        }

    @property
    def _stats(self) -> dict[str, Any] | None:
        return self.coordinator.data.get("scores", {}).get(self.model_id)
