"""Services for the NanoWakeWord integration."""

from __future__ import annotations

import logging
from pathlib import Path

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError

from .api import NanoWakeWordApiError
from .const import (
    ATTR_ENTRY_ID,
    ATTR_PATH,
    BACKUP_DIR,
    CONF_FILENAME,
    DOMAIN,
    SERVICE_BACKUP,
    SERVICE_DELETE_MODEL,
    SERVICE_RELOAD_MODELS,
    SERVICE_RESTORE,
    SERVICE_UPLOAD_MODEL,
)
from .helpers import async_create_backup, async_notify_model_change

_LOGGER = logging.getLogger(__name__)

_ENTRY_SCHEMA = {vol.Optional(ATTR_ENTRY_ID): str}


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register integration services."""

    def _get_entry(call: ServiceCall) -> ConfigEntry:
        loaded = [
            entry
            for entry in hass.config_entries.async_entries(DOMAIN)
            if entry.state is ConfigEntryState.LOADED
        ]
        if entry_id := call.data.get(ATTR_ENTRY_ID):
            for entry in loaded:
                if entry.entry_id == entry_id:
                    return entry
            raise ServiceValidationError(
                f"No loaded NanoWakeWord config entry with id {entry_id!r}"
            )
        if len(loaded) == 1:
            return loaded[0]
        if not loaded:
            raise ServiceValidationError("No loaded NanoWakeWord config entry")
        raise ServiceValidationError(
            "Multiple NanoWakeWord config entries; pass entry_id"
        )

    def _resolve_local_path(call_path: str) -> Path:
        path = Path(call_path)
        if not path.is_absolute():
            path = Path(hass.config.path(call_path))

        backup_dir = Path(hass.config.path(BACKUP_DIR)).resolve()
        resolved = path.resolve()
        if not (
            resolved.is_relative_to(backup_dir)
            or hass.config.is_allowed_path(str(resolved))
        ):
            raise ServiceValidationError(
                f"Path {call_path!r} is not in {backup_dir} and not in "
                "allowlist_external_dirs"
            )
        if not resolved.is_file():
            raise ServiceValidationError(f"No such file: {call_path!r}")
        return resolved

    async def handle_backup(call: ServiceCall) -> ServiceResponse:
        entry = _get_entry(call)
        try:
            path = await async_create_backup(hass, entry)
        except NanoWakeWordApiError as err:
            raise HomeAssistantError(f"Backup failed: {err}") from err
        _LOGGER.info("Saved NanoWakeWord backup to %s", path)
        return {"path": str(path)}

    async def handle_restore(call: ServiceCall) -> None:
        entry = _get_entry(call)
        path = _resolve_local_path(call.data[ATTR_PATH])
        content = await hass.async_add_executor_job(path.read_bytes)
        try:
            await entry.runtime_data.client.restore(content)
        except NanoWakeWordApiError as err:
            raise HomeAssistantError(f"Restore failed: {err}") from err
        await async_notify_model_change(hass, entry)

    async def handle_upload_model(call: ServiceCall) -> None:
        entry = _get_entry(call)
        path = _resolve_local_path(call.data[ATTR_PATH])
        content = await hass.async_add_executor_job(path.read_bytes)
        try:
            await entry.runtime_data.client.upload_model(path.name, content)
        except NanoWakeWordApiError as err:
            raise HomeAssistantError(f"Upload failed: {err}") from err
        await async_notify_model_change(hass, entry)

    async def handle_delete_model(call: ServiceCall) -> None:
        entry = _get_entry(call)
        try:
            await entry.runtime_data.client.delete_model(call.data[CONF_FILENAME])
        except NanoWakeWordApiError as err:
            raise HomeAssistantError(f"Delete failed: {err}") from err
        await async_notify_model_change(hass, entry)

    async def handle_reload_models(call: ServiceCall) -> None:
        entry = _get_entry(call)
        try:
            await entry.runtime_data.client.reload()
        except NanoWakeWordApiError as err:
            raise HomeAssistantError(f"Reload failed: {err}") from err
        await async_notify_model_change(hass, entry)

    hass.services.async_register(
        DOMAIN,
        SERVICE_BACKUP,
        handle_backup,
        schema=vol.Schema(_ENTRY_SCHEMA),
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RESTORE,
        handle_restore,
        schema=vol.Schema({**_ENTRY_SCHEMA, vol.Required(ATTR_PATH): str}),
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_UPLOAD_MODEL,
        handle_upload_model,
        schema=vol.Schema({**_ENTRY_SCHEMA, vol.Required(ATTR_PATH): str}),
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_DELETE_MODEL,
        handle_delete_model,
        schema=vol.Schema({**_ENTRY_SCHEMA, vol.Required(CONF_FILENAME): str}),
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RELOAD_MODELS,
        handle_reload_models,
        schema=vol.Schema(_ENTRY_SCHEMA),
    )
