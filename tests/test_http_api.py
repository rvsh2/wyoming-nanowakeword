from __future__ import annotations

import io
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer

from wyoming_nanowakeword.http_api import ModelApi
from wyoming_nanowakeword.state import State

ENSEMBLE_YAML = """
models:
  agata:
    phrase: "Agata"
    members:
      - model: "agata_primary"
        role: "primary"
      - model: "agata_verifier"
        role: "verifier"
  agata_primary:
    hidden: true
  agata_verifier:
    hidden: true
"""


@asynccontextmanager
async def _client(
    tmp_path: Path, token: str | None = None
) -> AsyncIterator[tuple[TestClient, State]]:
    state = State(model_dirs=[tmp_path])
    state.refresh()
    api = ModelApi(state, host="127.0.0.1", port=0, token=token)
    client = TestClient(TestServer(api.build_app()))
    await client.start_server()
    try:
        yield client, state
    finally:
        await client.close()


def _upload_form(filename: str, content: bytes) -> aiohttp.FormData:
    form = aiohttp.FormData()
    form.add_field("file", content, filename=filename)
    return form


@pytest.mark.asyncio
async def test_info_and_model_listing(tmp_path: Path) -> None:
    (tmp_path / "hey_home.onnx").write_bytes(b"onnx")
    (tmp_path / "hey_home_lite.onnx").write_bytes(b"onnx")

    async with _client(tmp_path) as (client, _state):
        info = await (await client.get("/api/info")).json()
        assert info["server"] == "wyoming-nanowakeword"
        assert info["models"] == ["hey_home"]

        listing = await (await client.get("/api/models")).json()
        assert listing["models"][0]["id"] == "hey_home"
        assert listing["models"][0]["gate"] == "hey_home_lite.onnx"
        assert listing["files"] == ["hey_home.onnx", "hey_home_lite.onnx"]


@pytest.mark.asyncio
async def test_upload_adds_model_and_refreshes_state(tmp_path: Path) -> None:
    async with _client(tmp_path) as (client, state):
        response = await client.post(
            "/api/models", data=_upload_form("hey_home.onnx", b"onnx")
        )

        assert response.status == 200
        assert (tmp_path / "hey_home.onnx").read_bytes() == b"onnx"
        assert "hey_home" in state.models


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_and_traversal_names(tmp_path: Path) -> None:
    async with _client(tmp_path) as (client, _state):
        response = await client.post(
            "/api/models", data=_upload_form("evil.exe", b"nope")
        )
        assert response.status == 400

        # Path components are stripped: the file lands inside model_dir.
        response = await client.post(
            "/api/models", data=_upload_form("../escape.onnx", b"onnx")
        )
        assert response.status == 200
        assert (tmp_path / "escape.onnx").is_file()
        assert not (tmp_path.parent / "escape.onnx").exists()


@pytest.mark.asyncio
async def test_upload_rejects_unsafe_and_misnamed_files(tmp_path: Path) -> None:
    async with _client(tmp_path) as (client, _state):
        # URL-special characters would be undeletable through clients.
        response = await client.post(
            "/api/models", data=_upload_form("hey#test.onnx", b"onnx")
        )
        assert response.status == 400

        # Only models.yaml is ever read by discovery; other yaml names would
        # be accepted and silently ignored.
        for name in ("models.yml", "agata.yaml"):
            response = await client.post(
                "/api/models", data=_upload_form(name, b"models: {}")
            )
            assert response.status == 400
            assert "models.yaml" in await response.text()
            assert not (tmp_path / name).exists()


@pytest.mark.asyncio
async def test_yaml_syntax_error_is_rolled_back(tmp_path: Path) -> None:
    (tmp_path / "hey_home.onnx").write_bytes(b"onnx")

    async with _client(tmp_path) as (client, state):
        response = await client.post(
            "/api/models",
            data=_upload_form("models.yaml", b"models:\n  bad\n    indent: [\n"),
        )

        assert response.status == 400
        assert not (tmp_path / "models.yaml").exists()
        assert sorted(state.models) == ["hey_home"]


@pytest.mark.asyncio
async def test_restore_rejects_zip_bombs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wyoming_nanowakeword import http_api

    (tmp_path / "hey_home.onnx").write_bytes(b"onnx")
    monkeypatch.setattr(http_api, "_MAX_RESTORE_BYTES", 1000)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("hey_home.onnx", b"\x00" * 10_000)

    async with _client(tmp_path) as (client, _state):
        response = await client.post(
            "/api/restore", data=_upload_form("backup.zip", buffer.getvalue())
        )
        assert response.status == 400
        assert "limit" in await response.text()
        assert (tmp_path / "hey_home.onnx").read_bytes() == b"onnx"


@pytest.mark.asyncio
async def test_refresh_reports_invalid_model_dir_as_400(tmp_path: Path) -> None:
    (tmp_path / "hey_home.onnx").write_bytes(b"onnx")

    async with _client(tmp_path) as (client, _state):
        (tmp_path / "models.yaml").write_text(
            "models:\n  broken:\n    members:\n      - model: 'missing'\n",
            encoding="utf-8",
        )
        response = await client.post("/api/refresh")
        assert response.status == 400
        assert "missing" in await response.text()


@pytest.mark.asyncio
async def test_invalid_models_yaml_is_rolled_back(tmp_path: Path) -> None:
    (tmp_path / "hey_home.onnx").write_bytes(b"onnx")
    broken_yaml = b"models:\n  broken:\n    members:\n      - model: 'missing'\n"

    async with _client(tmp_path) as (client, state):
        response = await client.post(
            "/api/models", data=_upload_form("models.yaml", broken_yaml)
        )

        assert response.status == 400
        assert "missing" in await response.text()
        assert not (tmp_path / "models.yaml").exists()
        assert sorted(state.models) == ["hey_home"]


@pytest.mark.asyncio
async def test_delete_removes_model_but_protects_ensemble_members(
    tmp_path: Path,
) -> None:
    (tmp_path / "agata_primary.onnx").write_bytes(b"onnx")
    (tmp_path / "agata_verifier.onnx").write_bytes(b"onnx")
    (tmp_path / "models.yaml").write_text(ENSEMBLE_YAML, encoding="utf-8")

    async with _client(tmp_path) as (client, state):
        # Deleting an ensemble member would break the ensemble: rolled back.
        response = await client.delete("/api/models/agata_verifier.onnx")
        assert response.status == 400
        assert (tmp_path / "agata_verifier.onnx").is_file()
        assert "agata" in state.models

        response = await client.delete("/api/models/models.yaml")
        assert response.status == 200
        assert sorted(state.models) == ["agata_primary", "agata_verifier"]

        response = await client.delete("/api/models/agata_verifier.onnx")
        assert response.status == 200
        assert sorted(state.models) == ["agata_primary"]

        response = await client.delete("/api/models/agata_verifier.onnx")
        assert response.status == 404


@pytest.mark.asyncio
async def test_backup_and_restore_round_trip(tmp_path: Path) -> None:
    (tmp_path / "hey_home.onnx").write_bytes(b"onnx-v1")
    (tmp_path / "models.yaml").write_text(
        'models:\n  hey_home:\n    phrase: "Hey Home"\n', encoding="utf-8"
    )

    async with _client(tmp_path) as (client, state):
        backup = await (await client.get("/api/backup")).read()
        names = sorted(zipfile.ZipFile(io.BytesIO(backup)).namelist())
        assert names == ["hey_home.onnx", "models.yaml"]

        # Diverge from the backup, then restore: extra file must disappear.
        await client.post("/api/models", data=_upload_form("other.onnx", b"onnx"))
        assert "other" in state.models

        response = await client.post(
            "/api/restore", data=_upload_form("backup.zip", backup)
        )
        assert response.status == 200
        assert sorted(state.models) == ["hey_home"]
        assert not (tmp_path / "other.onnx").exists()
        assert (tmp_path / "hey_home.onnx").read_bytes() == b"onnx-v1"


@pytest.mark.asyncio
async def test_restore_rejects_bad_archives(tmp_path: Path) -> None:
    (tmp_path / "hey_home.onnx").write_bytes(b"onnx")

    async with _client(tmp_path) as (client, state):
        response = await client.post(
            "/api/restore", data=_upload_form("backup.zip", b"not a zip")
        )
        assert response.status == 400

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("nested/dir.onnx", b"onnx")
        response = await client.post(
            "/api/restore", data=_upload_form("backup.zip", buffer.getvalue())
        )
        assert response.status == 400

        assert (tmp_path / "hey_home.onnx").is_file()
        assert sorted(state.models) == ["hey_home"]


@pytest.mark.asyncio
async def test_token_is_enforced_when_configured(tmp_path: Path) -> None:
    (tmp_path / "hey_home.onnx").write_bytes(b"onnx")

    async with _client(tmp_path, token="secret") as (client, _state):
        response = await client.get("/api/models")
        assert response.status == 401

        response = await client.get(
            "/api/models", headers={"Authorization": "Bearer wrong"}
        )
        assert response.status == 401

        response = await client.get(
            "/api/models", headers={"Authorization": "Bearer secret"}
        )
        assert response.status == 200
