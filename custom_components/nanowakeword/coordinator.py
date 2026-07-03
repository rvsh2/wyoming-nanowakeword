"""Data update coordinator for the NanoWakeWord integration."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)

from .api import NanoWakeWordApiError, NanoWakeWordClient
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
            update_interval=timedelta(seconds=60),
        )
        self.client = client

    async def _async_update_data(self) -> dict[str, Any]:
        try:
            return await self.client.models()
        except NanoWakeWordApiError as err:
            raise UpdateFailed(str(err)) from err
