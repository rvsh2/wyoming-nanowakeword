"""Optional HTTP management API for wake word models.

Runs next to the Wyoming server (see ``--http-port``) so UIs such as the
Home Assistant integration can upload, delete, back up and restore models.
The Wyoming protocol itself cannot carry files.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
import wave
import zipfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import yaml
from aiohttp import BodyPartReader, web
from wyoming.audio import AudioChunk, AudioChunkConverter

from . import __version__
from .fusion import fuse_scores, score_from_result
from .interpreters import InterpreterManager
from .models import ModelEntry
from .state import State

_LOGGER = logging.getLogger(__name__)

_METADATA_NAME = "models.yaml"
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_UPLOAD_BYTES = 256 * 1024 * 1024
_MAX_RESTORE_BYTES = 1024 * 1024 * 1024
_TEST_CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz, NanoWakeWord's feature stride


def _load_wav_as_int16(content: bytes) -> np.ndarray:
    """Decode a WAV file to 16 kHz mono int16 samples."""

    with wave.open(io.BytesIO(content), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        chunk = AudioChunk(
            rate=wav_file.getframerate(),
            width=wav_file.getsampwidth(),
            channels=wav_file.getnchannels(),
            audio=frames,
        )

    converter = AudioChunkConverter(rate=16000, width=2, channels=1)
    return np.frombuffer(converter.convert(chunk).audio, dtype=np.int16)


def _is_managed_name(name: str) -> bool:
    return name == _METADATA_NAME or name.endswith(".onnx")


def _validate_upload_name(name: str) -> None:
    if not _SAFE_FILENAME.match(name):
        raise web.HTTPBadRequest(
            text=f"Invalid filename {name!r}: use only letters, digits, . _ -"
        )
    if name.endswith((".yaml", ".yml")) and name != _METADATA_NAME:
        raise web.HTTPBadRequest(
            text=f"Metadata must be named {_METADATA_NAME!r}, got {name!r}; "
            "other YAML files are never read by the server"
        )
    if not _is_managed_name(name):
        raise web.HTTPBadRequest(
            text=f"Unsupported file type: {name!r} (expected .onnx or models.yaml)"
        )


class ModelApi:
    """HTTP API operating on the first configured model directory."""

    def __init__(
        self,
        state: State,
        host: str,
        port: int,
        token: str | None = None,
        interpreter_manager: InterpreterManager | None = None,
        default_threshold: float = 0.95,
    ) -> None:
        if not state.model_dirs:
            raise ValueError("HTTP API requires at least one model directory")

        self.state = state
        self.host = host
        self.port = port
        self.token = token
        self.interpreter_manager = interpreter_manager
        self.default_threshold = default_threshold
        self._runner: web.AppRunner | None = None

    @property
    def model_dir(self) -> Path:
        return self.state.model_dirs[0]

    def build_app(self) -> web.Application:
        token = self.token

        @web.middleware
        async def auth(
            request: web.Request,
            handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
        ) -> web.StreamResponse:
            if token and request.headers.get("Authorization") != f"Bearer {token}":
                raise web.HTTPUnauthorized(text="Invalid or missing token")
            return await handler(request)

        app = web.Application(middlewares=[auth], client_max_size=_MAX_UPLOAD_BYTES)
        app.add_routes(
            [
                web.get("/api/info", self.info),
                web.get("/api/models", self.list_models),
                web.get("/api/scores", self.scores),
                web.post("/api/models", self.upload_model),
                web.delete("/api/models/{filename}", self.delete_model),
                web.post("/api/refresh", self.refresh),
                web.get("/api/backup", self.backup),
                web.post("/api/restore", self.restore),
                web.post("/api/test", self.test_recording),
                web.get("/api/events", self.events),
            ]
        )
        return app

    async def start(self) -> None:
        runner = web.AppRunner(self.build_app())
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self._runner = runner
        _LOGGER.info("HTTP model API listening on http://%s:%s", self.host, self.port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def info(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "server": "wyoming-nanowakeword",
                "version": __version__,
                "model_dir": str(self.model_dir),
                "models": sorted(self.state.models),
                "clients": self.state.clients,
            }
        )

    async def list_models(self, request: web.Request) -> web.Response:
        return web.json_response(self._models_payload())

    async def scores(self, request: web.Request) -> web.Response:
        """Live inference scores per model — for tuning thresholds."""

        now = time.monotonic()
        payload = {
            model_id: {
                "last": round(stats["last"], 4),
                "peak": round(stats["peak"], 4),
                "peak_age_seconds": round(now - stats["peak_at"], 1),
                "detections": stats["detections"],
                "avg_inference_ms": (
                    round(stats["avg_ms"], 2)
                    if stats.get("avg_ms") is not None
                    else None
                ),
            }
            for model_id, stats in self.state.scores.items()
        }
        return web.json_response({"scores": payload})

    async def upload_model(self, request: web.Request) -> web.Response:
        filename, content = await self._read_uploaded_file(request)
        _validate_upload_name(filename)

        try:
            self._apply_changes(writes={filename: content})
        except ValueError as err:
            raise web.HTTPBadRequest(text=str(err)) from err

        _LOGGER.info("Uploaded model file %s", filename)
        return web.json_response(self._models_payload())

    async def delete_model(self, request: web.Request) -> web.Response:
        filename = Path(request.match_info["filename"]).name
        path = self.model_dir / filename
        if not _is_managed_name(filename) or not path.is_file():
            raise web.HTTPNotFound(text=f"No such model file: {filename!r}")

        try:
            self._apply_changes(deletes={filename})
        except ValueError as err:
            raise web.HTTPBadRequest(text=str(err)) from err

        _LOGGER.info("Deleted model file %s", filename)
        return web.json_response(self._models_payload())

    async def refresh(self, request: web.Request) -> web.Response:
        try:
            self.state.refresh()
        except (ValueError, yaml.YAMLError) as err:
            raise web.HTTPBadRequest(text=str(err)) from err

        return web.json_response(self._models_payload())

    async def backup(self, request: web.Request) -> web.Response:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in self._managed_files():
                archive.write(path, arcname=path.name)

        return web.Response(
            body=buffer.getvalue(),
            content_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="nanowakeword-backup.zip"'
            },
        )

    async def restore(self, request: web.Request) -> web.Response:
        _filename, content = await self._read_uploaded_file(request)

        try:
            archive = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile as err:
            raise web.HTTPBadRequest(text="Restore payload is not a zip file") from err

        entries = [entry for entry in archive.infolist() if not entry.is_dir()]

        # file_size is the declared decompressed size; the request size limit
        # only bounds the compressed payload (zip bomb protection).
        total_size = sum(entry.file_size for entry in entries)
        if total_size > _MAX_RESTORE_BYTES:
            raise web.HTTPBadRequest(
                text=f"Backup decompresses to {total_size} bytes, "
                f"limit is {_MAX_RESTORE_BYTES}"
            )

        writes: dict[str, bytes] = {}
        for entry in entries:
            name = Path(entry.filename).name
            if name != entry.filename:
                raise web.HTTPBadRequest(
                    text=f"Unexpected file in backup: {entry.filename!r}"
                )
            _validate_upload_name(name)
            data = archive.read(entry)
            if len(data) != entry.file_size:
                raise web.HTTPBadRequest(
                    text=f"Backup entry {name!r} lies about its size"
                )
            writes[name] = data

        if not writes:
            raise web.HTTPBadRequest(text="Backup archive contains no model files")

        existing = {path.name for path in self._managed_files()}
        try:
            self._apply_changes(writes=writes, deletes=existing - set(writes))
        except ValueError as err:
            raise web.HTTPBadRequest(text=str(err)) from err

        _LOGGER.info("Restored %s model files from backup", len(writes))
        return web.json_response(self._models_payload())

    async def test_recording(self, request: web.Request) -> web.Response:
        """Score an uploaded WAV recording against a wake word model.

        Returns the per-chunk score series for every ensemble member and the
        fused score, so thresholds can be tuned against real recordings
        instead of live audio.
        """

        if self.interpreter_manager is None:
            raise web.HTTPNotImplemented(
                text="Recording test requires the full server (no interpreters)"
            )

        model_id = request.query.get("model") or self.state.get_default_model_id()
        if not model_id or model_id not in self.state.models:
            raise web.HTTPNotFound(text=f"No such wake word: {model_id!r}")

        entry = self.state.models[model_id]
        filename, content = await self._read_uploaded_file(request)
        if not filename.lower().endswith(".wav"):
            raise web.HTTPBadRequest(text=f"Expected a .wav file, got {filename!r}")

        try:
            audio = _load_wav_as_int16(content)
        except (wave.Error, EOFError) as err:
            raise web.HTTPBadRequest(text=f"Cannot read WAV file: {err}") from err

        result = await asyncio.to_thread(self._score_recording, entry, audio)
        return web.json_response(result)

    def _score_recording(
        self, entry: ModelEntry, audio: np.ndarray
    ) -> dict[str, Any]:
        assert self.interpreter_manager is not None
        generation = self.state.generation
        interpreters = self.interpreter_manager.acquire_for_entry(entry)
        threshold = (
            entry.metadata.threshold
            if entry.metadata.threshold is not None
            else self.default_threshold
        )

        member_series: dict[str, list[float]] = {
            model_id: [] for model_id in interpreters
        }
        fused_series: list[float] = []
        try:
            for interpreter in interpreters.values():
                interpreter.reset()

            for start in range(0, len(audio), _TEST_CHUNK_SAMPLES):
                chunk = audio[start : start + _TEST_CHUNK_SAMPLES]
                scores: dict[str, float] = {}
                for model_id, interpreter in interpreters.items():
                    result = interpreter.predict(chunk)
                    scores[model_id] = score_from_result(result, model_id)
                    member_series[model_id].append(round(scores[model_id], 4))

                fused = (
                    fuse_scores(entry, scores, self.default_threshold)
                    if entry.is_ensemble
                    else next(iter(scores.values()))
                )
                fused_series.append(round(fused, 4))
        finally:
            self.interpreter_manager.release(interpreters, generation)

        return {
            "model": entry.id,
            "duration_seconds": round(len(audio) / 16000, 2),
            "chunk_ms": 80,
            "threshold": threshold,
            "peak": max(fused_series, default=0.0),
            "would_detect": any(score > threshold for score in fused_series),
            "member_peaks": {
                model_id: max(series, default=0.0)
                for model_id, series in member_series.items()
            },
            "fused_series": fused_series,
            "member_series": member_series,
        }

    async def events(self, request: web.Request) -> web.StreamResponse:
        """Server-sent events stream of detections."""

        queue = self.state.subscribe()
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        await response.prepare(request)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue

                await response.write(f"data: {json.dumps(event)}\n\n".encode())
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.state.unsubscribe(queue)

        return response

    def _models_payload(self) -> dict[str, Any]:
        models = []
        for entry in self.state.models.values():
            models.append(
                {
                    "id": entry.id,
                    "phrase": entry.phrase,
                    "language": entry.metadata.language,
                    "architecture": entry.metadata.architecture,
                    "version": entry.metadata.version,
                    "ensemble": entry.is_ensemble,
                    "file": entry.path.name if entry.path else None,
                    "gate": entry.gate_path.name if entry.gate_path else None,
                    "members": [member.model for member in entry.members],
                }
            )

        return {
            "models": sorted(models, key=lambda model: model["id"]),
            "files": sorted(path.name for path in self._managed_files()),
            "clients": self.state.clients,
        }

    def _managed_files(self) -> list[Path]:
        return sorted(
            path
            for path in self.model_dir.glob("*")
            if path.is_file() and _is_managed_name(path.name)
        )

    async def _read_uploaded_file(self, request: web.Request) -> tuple[str, bytes]:
        reader = await request.multipart()
        part = await reader.next()
        while part is not None and not (
            isinstance(part, BodyPartReader) and part.name == "file"
        ):
            part = await reader.next()

        if not isinstance(part, BodyPartReader) or not part.filename:
            raise web.HTTPBadRequest(
                text="Expected multipart field 'file' with a filename"
            )

        # aiohttp percent-encodes the filename; decode, then strip any path
        # components so uploads cannot escape the model directory.
        filename = Path(unquote(part.filename or "")).name
        content = bytes(await part.read())
        return filename, content

    def _apply_changes(
        self,
        writes: dict[str, bytes] | None = None,
        deletes: set[str] | None = None,
    ) -> None:
        """Write/delete model files, rolling everything back if a file
        operation fails or the resulting model set is invalid (e.g. an
        ensemble loses a member). Raises ValueError with the reason."""

        writes = writes or {}
        deletes = deletes or set()

        originals: dict[str, bytes | None] = {}
        for name in [*writes, *deletes]:
            path = self.model_dir / name
            originals[name] = path.read_bytes() if path.is_file() else None

        try:
            for name, content in writes.items():
                (self.model_dir / name).write_bytes(content)
            for name in deletes:
                (self.model_dir / name).unlink(missing_ok=True)
            self.state.refresh()
        except (ValueError, yaml.YAMLError, OSError) as err:
            self._rollback(originals)
            raise ValueError(str(err)) from err

    def _rollback(self, originals: dict[str, bytes | None]) -> None:
        for name, original in originals.items():
            path = self.model_dir / name
            try:
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(original)
            except OSError:
                _LOGGER.exception("Rollback of %s failed", name)

        try:
            self.state.refresh()
        except (ValueError, yaml.YAMLError, OSError):
            # The directory was already invalid before this request (e.g. a
            # hand-edited models.yaml); keep the previous in-memory state.
            _LOGGER.exception("Model directory is invalid after rollback")
