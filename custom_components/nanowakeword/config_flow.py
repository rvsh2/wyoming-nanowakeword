"""Config and options flow for the NanoWakeWord integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import voluptuous as vol
from homeassistant.components.file_upload import process_uploaded_file
from homeassistant.config_entries import (
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    FileSelector,
    FileSelectorConfig,
    SelectSelector,
    SelectSelectorConfig,
)
from homeassistant.helpers.service_info.zeroconf import ZeroconfServiceInfo

from .api import NanoWakeWordApiError, NanoWakeWordAuthError, NanoWakeWordClient
from .const import (
    BACKUP_DIR,
    CONF_BACKUP,
    CONF_BACKUP_FILE,
    CONF_FILENAME,
    CONF_MODEL,
    CONF_MODEL_FILE,
    CONF_RECORDING,
    CONF_TOKEN,
    CONF_VERIFY_MODEL,
    CONF_VERIFY_TOKEN,
    CONF_VERIFY_URL,
    DEFAULT_PORT,
    DOMAIN,
)
from .helpers import async_list_backups, async_notify_model_change

STEP_USER_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_HOST): str,
        vol.Required(CONF_PORT, default=DEFAULT_PORT): int,
        vol.Optional(CONF_TOKEN): str,
    }
)


class NanoWakeWordConfigFlow(ConfigFlow, domain=DOMAIN):
    """Configure a connection to a wyoming-nanowakeword HTTP model API."""

    VERSION = 1

    def __init__(self) -> None:
        self._discovered: dict[str, Any] | None = None

    async def async_step_zeroconf(
        self, discovery_info: ZeroconfServiceInfo
    ) -> ConfigFlowResult:
        host = discovery_info.host
        port = discovery_info.port or DEFAULT_PORT
        await self.async_set_unique_id(f"{host}:{port}")
        self._abort_if_unique_id_configured()

        self._discovered = {CONF_HOST: host, CONF_PORT: port}
        self.context["title_placeholders"] = {"host": host}
        return await self.async_step_zeroconf_confirm()

    async def async_step_zeroconf_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        assert self._discovered is not None

        if user_input is not None:
            data = {**self._discovered, CONF_TOKEN: user_input.get(CONF_TOKEN)}
            error = await self._async_validate(data)
            if error:
                errors["base"] = error
            else:
                return self.async_create_entry(
                    title=f"NanoWakeWord ({data[CONF_HOST]})", data=data
                )

        return self.async_show_form(
            step_id="zeroconf_confirm",
            data_schema=vol.Schema({vol.Optional(CONF_TOKEN): str}),
            errors=errors,
            description_placeholders={"host": self._discovered[CONF_HOST]},
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            error = await self._async_validate(user_input)
            if error:
                errors["base"] = error
            else:
                unique_id = f"{user_input[CONF_HOST]}:{user_input[CONF_PORT]}"
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=f"NanoWakeWord ({user_input[CONF_HOST]})",
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_SCHEMA, errors=errors
        )

    async def async_step_reauth(
        self, entry_data: dict[str, Any]
    ) -> ConfigFlowResult:
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        entry = self.hass.config_entries.async_get_entry(self.context["entry_id"])
        assert entry is not None

        if user_input is not None:
            candidate = {**entry.data, CONF_TOKEN: user_input.get(CONF_TOKEN)}
            error = await self._async_validate(candidate)
            if error:
                errors["base"] = error
            else:
                return self.async_update_reload_and_abort(
                    entry, data_updates={CONF_TOKEN: user_input.get(CONF_TOKEN)}
                )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema({vol.Optional(CONF_TOKEN): str}),
            errors=errors,
        )

    async def _async_validate(self, data: dict[str, Any]) -> str | None:
        client = NanoWakeWordClient(
            async_get_clientsession(self.hass),
            data[CONF_HOST],
            data[CONF_PORT],
            data.get(CONF_TOKEN),
        )
        try:
            await client.info()
        except NanoWakeWordAuthError:
            return "invalid_auth"
        except NanoWakeWordApiError:
            return "cannot_connect"
        return None

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: Any) -> NanoWakeWordOptionsFlow:
        return NanoWakeWordOptionsFlow()


class NanoWakeWordOptionsFlow(OptionsFlow):
    """Model management actions: upload, delete, restore."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if self.config_entry.state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="not_loaded")

        return self.async_show_menu(
            step_id="init",
            menu_options=[
                "upload_model",
                "delete_model",
                "test_recording",
                "configure_verifier",
                "restore_saved",
                "restore_backup",
            ],
        )

    async def async_step_configure_verifier(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Hybrid satellite + server: point this server at a central verifier."""

        errors: dict[str, str] = {}
        placeholders: dict[str, str] = {"error": ""}
        settings = (
            self.config_entry.runtime_data.coordinator.data or {}
        ).get("settings") or {}

        if user_input is not None:
            url = user_input.get(CONF_VERIFY_URL, "").strip()
            changes: dict[str, Any] = {
                "verify_url": url,
                "verify_model": user_input.get(CONF_VERIFY_MODEL, "").strip(),
                # A URL means the user wants verification; clearing it
                # disables the whole feature.
                "verify": bool(url),
            }
            if token := user_input.get(CONF_VERIFY_TOKEN, "").strip():
                changes["verify_token"] = token

            try:
                await self.config_entry.runtime_data.coordinator.async_apply_settings(
                    changes
                )
            except NanoWakeWordApiError as err:
                errors["base"] = "settings_failed"
                placeholders["error"] = str(err)
            else:
                return self._async_finish()

        return self.async_show_form(
            step_id="configure_verifier",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_VERIFY_URL,
                        description={
                            "suggested_value": settings.get("verify_url", "")
                        },
                    ): str,
                    vol.Optional(CONF_VERIFY_TOKEN): str,
                    vol.Optional(
                        CONF_VERIFY_MODEL,
                        description={
                            "suggested_value": settings.get("verify_model", "")
                        },
                    ): str,
                }
            ),
            errors=errors,
            description_placeholders=placeholders,
        )

    async def async_step_test_recording(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Score an uploaded WAV against a wake word model."""

        errors: dict[str, str] = {}
        placeholders: dict[str, str] = {"error": ""}

        models = [
            model["id"]
            for model in (self.config_entry.runtime_data.coordinator.data or {}).get(
                "models", []
            )
        ]
        if not models:
            return self.async_abort(reason="no_model_files")

        if user_input is not None:
            name, content = await self.hass.async_add_executor_job(
                self._read_uploaded_file, user_input[CONF_RECORDING]
            )
            try:
                result = await self._client.test_recording(
                    name, content, user_input[CONF_MODEL]
                )
            except NanoWakeWordApiError as err:
                errors["base"] = "test_failed"
                placeholders["error"] = str(err)
            else:
                member_peaks = ", ".join(
                    f"{member}: {peak:.3f}"
                    for member, peak in sorted(result["member_peaks"].items())
                )
                return self.async_abort(
                    reason="test_result",
                    description_placeholders={
                        "model": result["model"],
                        "verdict": "✅" if result["would_detect"] else "❌",
                        "peak": f"{result['peak']:.3f}",
                        "threshold": f"{result['threshold']:.3f}",
                        "duration": str(result["duration_seconds"]),
                        "member_peaks": member_peaks or "-",
                    },
                )

        return self.async_show_form(
            step_id="test_recording",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MODEL, default=models[0]): SelectSelector(
                        SelectSelectorConfig(options=models)
                    ),
                    vol.Required(CONF_RECORDING): FileSelector(
                        FileSelectorConfig(accept=".wav")
                    ),
                }
            ),
            errors=errors,
            description_placeholders=placeholders,
        )

    async def async_step_restore_saved(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Restore one of the backups saved under /config/nanowakeword."""

        errors: dict[str, str] = {}
        placeholders: dict[str, str] = {"error": ""}

        if user_input is not None:
            path = (
                Path(self.hass.config.path(BACKUP_DIR))
                / Path(user_input[CONF_BACKUP]).name
            )
            content = await self.hass.async_add_executor_job(path.read_bytes)
            try:
                await self._client.restore(content)
            except NanoWakeWordApiError as err:
                errors["base"] = "restore_failed"
                placeholders["error"] = str(err)
            else:
                await async_notify_model_change(self.hass, self.config_entry)
                return self._async_finish()

        backups = await async_list_backups(self.hass)
        if not backups:
            return self.async_abort(reason="no_backups")

        return self.async_show_form(
            step_id="restore_saved",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_BACKUP): SelectSelector(
                        SelectSelectorConfig(options=backups)
                    ),
                }
            ),
            errors=errors,
            description_placeholders=placeholders,
        )

    async def async_step_upload_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        placeholders: dict[str, str] = {"error": ""}

        if user_input is not None:
            uploaded_name, content = await self.hass.async_add_executor_job(
                self._read_uploaded_file, user_input[CONF_MODEL_FILE]
            )
            filename = (user_input.get(CONF_FILENAME) or uploaded_name).strip()
            # The server only ever reads *.onnx models and a file literally
            # named models.yaml; reject anything else up front.
            if not (filename.endswith(".onnx") or filename == "models.yaml"):
                errors["base"] = "invalid_filename"
            else:
                try:
                    await self._client.upload_model(filename, content)
                except NanoWakeWordApiError as err:
                    errors["base"] = "upload_failed"
                    placeholders["error"] = str(err)
                else:
                    await async_notify_model_change(self.hass, self.config_entry)
                    return self._async_finish()

        return self.async_show_form(
            step_id="upload_model",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MODEL_FILE): FileSelector(
                        FileSelectorConfig(accept=".onnx,.yaml")
                    ),
                    vol.Optional(CONF_FILENAME): str,
                }
            ),
            errors=errors,
            description_placeholders=placeholders,
        )

    async def async_step_delete_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        placeholders: dict[str, str] = {"error": ""}

        if user_input is not None:
            try:
                await self._client.delete_model(user_input[CONF_FILENAME])
            except NanoWakeWordApiError as err:
                errors["base"] = "delete_failed"
                placeholders["error"] = str(err)
            else:
                await async_notify_model_change(self.hass, self.config_entry)
                return self._async_finish()

        data = self.config_entry.runtime_data.coordinator.data or {}
        files = data.get("files", [])
        if not files:
            return self.async_abort(reason="no_model_files")

        return self.async_show_form(
            step_id="delete_model",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_FILENAME): SelectSelector(
                        SelectSelectorConfig(options=files)
                    ),
                }
            ),
            errors=errors,
            description_placeholders=placeholders,
        )

    async def async_step_restore_backup(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        placeholders: dict[str, str] = {"error": ""}

        if user_input is not None:
            _name, content = await self.hass.async_add_executor_job(
                self._read_uploaded_file, user_input[CONF_BACKUP_FILE]
            )
            try:
                await self._client.restore(content)
            except NanoWakeWordApiError as err:
                errors["base"] = "restore_failed"
                placeholders["error"] = str(err)
            else:
                await async_notify_model_change(self.hass, self.config_entry)
                return self._async_finish()

        return self.async_show_form(
            step_id="restore_backup",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_BACKUP_FILE): FileSelector(
                        FileSelectorConfig(accept=".zip")
                    ),
                }
            ),
            errors=errors,
            description_placeholders=placeholders,
        )

    @property
    def _client(self) -> NanoWakeWordClient:
        return self.config_entry.runtime_data.client

    def _read_uploaded_file(self, file_id: str) -> tuple[str, bytes]:
        with process_uploaded_file(self.hass, file_id) as file_path:
            return file_path.name, file_path.read_bytes()

    def _async_finish(self) -> ConfigFlowResult:
        # Options are not used for settings; keep whatever is stored.
        return self.async_create_entry(data=dict(self.config_entry.options))
