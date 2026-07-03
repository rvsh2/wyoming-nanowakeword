"""Async client for the wyoming-nanowakeword HTTP model API."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import aiohttp

# No total limit: uploads/backups of large model sets on slow links can take
# minutes. Stalled connections are cut by sock_read instead.
_TIMEOUT = aiohttp.ClientTimeout(total=None, connect=30, sock_read=120)


class NanoWakeWordApiError(Exception):
    """Raised when the server rejects a request or is unreachable."""


class NanoWakeWordAuthError(NanoWakeWordApiError):
    """Raised when the API token is missing or wrong."""


class NanoWakeWordClient:
    """Thin client for the model management endpoints."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        host: str,
        port: int,
        token: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._session = session
        self._base_url = f"http://{host}:{port}/api"
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    async def info(self) -> dict[str, Any]:
        return await self._request_json("GET", "/info")

    async def models(self) -> dict[str, Any]:
        return await self._request_json("GET", "/models")

    async def upload_model(self, filename: str, content: bytes) -> dict[str, Any]:
        form = aiohttp.FormData()
        form.add_field("file", content, filename=filename)
        return await self._request_json("POST", "/models", data=form)

    async def delete_model(self, filename: str) -> dict[str, Any]:
        return await self._request_json("DELETE", f"/models/{quote(filename, safe='')}")

    async def reload(self) -> dict[str, Any]:
        return await self._request_json("POST", "/refresh")

    async def backup(self) -> bytes:
        response = await self._request("GET", "/backup")
        try:
            return await response.read()
        finally:
            response.release()

    async def restore(self, content: bytes) -> dict[str, Any]:
        form = aiohttp.FormData()
        form.add_field("file", content, filename="backup.zip")
        return await self._request_json("POST", "/restore", data=form)

    async def _request_json(
        self, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any]:
        response = await self._request(method, path, **kwargs)
        try:
            return await response.json()
        finally:
            response.release()

    async def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> aiohttp.ClientResponse:
        try:
            response = await self._session.request(
                method,
                f"{self._base_url}{path}",
                headers=self._headers,
                timeout=_TIMEOUT,
                **kwargs,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            raise NanoWakeWordApiError(
                f"Cannot reach the NanoWakeWord API at {self._base_url}: {err}"
            ) from err

        if response.status == 401:
            response.release()
            raise NanoWakeWordAuthError("Invalid or missing API token")
        if response.status >= 400:
            text = (await response.text())[:500]
            response.release()
            raise NanoWakeWordApiError(f"Server returned {response.status}: {text}")

        return response
