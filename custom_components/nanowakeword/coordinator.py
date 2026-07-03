"""Data update coordinator for the NanoWakeWord integration."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)

from .api import NanoWakeWordApiError, NanoWakeWordAuthError, NanoWakeWordClient
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class NanoWakeWordCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Poll the model list from the NanoWakeWord server."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        client: NanoWakeWordClient,
    ) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{DOMAIN} {entry.title}",
            # Short enough that the per-model score sensors are usable for
            # live threshold tuning.
            update_interval=timedelta(seconds=15),
        )
        self.client = client

    async def _async_update_data(self) -> dict[str, Any]:
        try:
            data = await self.client.models()
            scores = await self.client.scores()
        except NanoWakeWordAuthError as err:
            raise ConfigEntryAuthFailed from err
        except NanoWakeWordApiError as err:
            raise UpdateFailed(str(err)) from err

        return {**data, "scores": scores.get("scores", {})}
