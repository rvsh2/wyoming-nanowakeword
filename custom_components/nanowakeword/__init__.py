"""NanoWakeWord model management integration.

Talks to the HTTP model API of a wyoming-nanowakeword server: upload wake
word models from the UI, back up and restore the model directory, and expose
the served model list as entities.
"""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.typing import ConfigType

from .api import NanoWakeWordApiError, NanoWakeWordAuthError, NanoWakeWordClient
from .const import CONF_TOKEN, DOMAIN
from .coordinator import NanoWakeWordCoordinator
from .services import async_setup_services

PLATFORMS = [Platform.BUTTON, Platform.SENSOR]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


@dataclass
class NanoWakeWordData:
    """Runtime data stored on the config entry."""

    client: NanoWakeWordClient
    coordinator: NanoWakeWordCoordinator


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    async_setup_services(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    client = NanoWakeWordClient(
        async_get_clientsession(hass),
        entry.data[CONF_HOST],
        entry.data[CONF_PORT],
        entry.data.get(CONF_TOKEN),
    )

    try:
        await client.info()
    except NanoWakeWordAuthError as err:
        raise ConfigEntryAuthFailed from err
    except NanoWakeWordApiError as err:
        raise ConfigEntryNotReady(str(err)) from err

    coordinator = NanoWakeWordCoordinator(hass, entry, client)
    await coordinator.async_config_entry_first_refresh()

    entry.runtime_data = NanoWakeWordData(client=client, coordinator=coordinator)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
