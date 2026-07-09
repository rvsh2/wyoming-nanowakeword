"""Command-line entry point for the Wyoming NanoWakeWord server."""

from __future__ import annotations

import argparse
import asyncio
import logging
from functools import partial
from pathlib import Path

from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .handler import NanoWakeWordEventHandler
from .interpreters import InterpreterManager
from .settings import ServerSettings, load_settings_overlay
from .state import State

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="stdio://", help="unix://, tcp:// or stdio://")
    parser.add_argument(
        "--model-dir",
        action="append",
        default=[],
        help="Directory containing NanoWakeWord .onnx models",
    )
    parser.add_argument("--default-model", help="Default wake word model id")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Wake word threshold (0.0-1.0, default: 0.95)",
    )
    parser.add_argument(
        "--trigger-level",
        type=int,
        default=1,
        help="Number of activations before detection (default: 1)",
    )
    parser.add_argument(
        "--refractory-seconds",
        type=float,
        default=2.0,
        help="Seconds before a wake word can be detected again (default: 2.0)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.0,
        help="NanoWakeWord VAD threshold; 0 disables VAD (default: 0)",
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="Enable NanoWakeWord cascade mode with <model>_lite.onnx",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.3,
        help="Cascade gate threshold (default: 0.3)",
    )
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="nanoWakeWord",
        help="Enable discovery over zeroconf with optional name",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        help="Port for the HTTP model management API (disabled unless set)",
    )
    parser.add_argument(
        "--http-host",
        default="0.0.0.0",
        help="Bind host for the HTTP model management API (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--http-token",
        help="Bearer token required by the HTTP model management API",
    )
    parser.add_argument(
        "--verify-url",
        help="Base URL of a central wyoming-nanowakeword HTTP API that "
        "verifies candidate detections (hybrid satellite + server)",
    )
    parser.add_argument(
        "--verify-token",
        help="Bearer token for the verification server",
    )
    parser.add_argument(
        "--verify-model",
        help="Wake word id on the verification server (default: same id)",
    )
    parser.add_argument(
        "--verify-asr-url",
        help="whisper.cpp-compatible /inference URL that transcribes "
        "candidate detections; the wake word must appear in the transcript",
    )
    parser.add_argument(
        "--verify-asr-keyword",
        help="Comma-separated substrings that count as the wake word in a "
        "transcript (e.g. 'agat')",
    )
    parser.add_argument(
        "--verify-asr-prompt",
        help="Optional biasing prompt for the second ASR decode pass "
        "(e.g. 'Agata? Agatka? Agato?')",
    )
    parser.add_argument(
        "--verify-asr-min-prob",
        type=float,
        default=0.68,
        help="Minimum wake word token probability in the prompted ASR pass "
        "(default: 0.68)",
    )
    parser.add_argument(
        "--verify-asr-language",
        default="pl",
        help="Language passed to the ASR verifier (default: pl)",
    )
    parser.add_argument(
        "--capture-dir",
        help="Save a WAV of the audio leading up to each detection here "
        "(training data; disabled unless set)",
    )
    parser.add_argument(
        "--capture-seconds",
        type=float,
        default=3.0,
        help="Seconds of audio to keep before a detection (default: 3.0)",
    )
    parser.add_argument(
        "--capture-keep",
        type=int,
        default=200,
        help="Newest capture files to keep (default: 200)",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format",
        default=logging.BASIC_FORMAT,
        help="Format for log messages",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=args.log_format,
    )

    if args.version:
        print(__version__)
        return

    model_dirs = [Path(model_dir) for model_dir in args.model_dir]
    state = State(model_dirs=model_dirs, default_model=args.default_model)
    state.settings = ServerSettings(
        threshold=args.threshold,
        trigger_level=args.trigger_level,
        refractory_seconds=args.refractory_seconds,
        vad_threshold=args.vad_threshold,
        cascade=args.cascade,
        gate_threshold=args.gate_threshold,
        capture=bool(args.capture_dir),
        verify=bool(args.verify_url),
        verify_url=args.verify_url or "",
        verify_token=args.verify_token or "",
        verify_model=args.verify_model or "",
        verify_asr=bool(args.verify_asr_url),
        verify_asr_url=args.verify_asr_url or "",
        verify_asr_keyword=args.verify_asr_keyword or "",
        verify_asr_prompt=args.verify_asr_prompt or "",
        verify_asr_min_prob=args.verify_asr_min_prob,
        verify_asr_language=args.verify_asr_language,
    )
    if model_dirs:
        # Settings changed from Home Assistant win over CLI defaults.
        load_settings_overlay(state.settings, model_dirs[0])
    try:
        state.refresh()
    except Exception:
        # Keep the server (and its HTTP API, if enabled) running so a broken
        # models.yaml can be fixed remotely instead of crash-looping.
        _LOGGER.exception("Failed to load models; starting with none")
    if not state.models:
        _LOGGER.warning(
            "No NanoWakeWord .onnx models found in: %s",
            ", ".join(str(model_dir) for model_dir in model_dirs) or "(none)",
        )
    else:
        _LOGGER.info("Discovered models: %s", ", ".join(sorted(state.models)))

    if args.default_model and args.default_model not in state.models:
        _LOGGER.warning(
            "Default model %r not found (available: %s); falling back to %s",
            args.default_model,
            ", ".join(sorted(state.models)) or "(none)",
            state.get_default_model_id(),
        )

    interpreter_manager = InterpreterManager(state)
    default_model_id = state.get_default_model_id()
    if default_model_id:
        # Preload the default wake word so the first Detect answers instantly.
        default_entry = state.models[default_model_id]
        backing_ids = (
            [member.model for member in default_entry.members]
            if default_entry.is_ensemble
            else [default_entry.id]
        )
        await asyncio.to_thread(interpreter_manager.warm_up, backing_ids)

    if args.http_port:
        if not model_dirs:
            raise ValueError("--http-port requires at least one --model-dir")

        from .http_api import ModelApi

        model_api = ModelApi(
            state,
            host=args.http_host,
            port=args.http_port,
            token=args.http_token,
            interpreter_manager=interpreter_manager,
            default_threshold=args.threshold,
        )
        await model_api.start()

    server = AsyncServer.from_uri(args.uri)
    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// uri")

        from wyoming.zeroconf import HomeAssistantZeroconf

        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf,
            port=server.port,
            host=server.host,
        )
        await hass_zeroconf.register_server()
        _LOGGER.debug("Zeroconf discovery enabled")

        if args.http_port:
            # Advertise the model management API so the Home Assistant
            # integration can be discovered instead of configured by hand.
            await _register_http_zeroconf(args.zeroconf, args.http_port)

    _LOGGER.info("Ready")
    try:
        await server.run(
            partial(
                NanoWakeWordEventHandler,
                state=state,
                interpreter_manager=interpreter_manager,
                capture_dir=Path(args.capture_dir) if args.capture_dir else None,
                capture_seconds=args.capture_seconds,
                capture_keep=args.capture_keep,
            )
        )
    except KeyboardInterrupt:
        pass


async def _register_http_zeroconf(name: str, http_port: int) -> None:
    import socket

    from wyoming.zeroconf import MDNS_TARGET_IP
    from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

    test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    test_sock.setblocking(False)
    test_sock.connect((MDNS_TARGET_IP, 1))
    host = test_sock.getsockname()[0]
    test_sock.close()

    service_info = AsyncServiceInfo(
        "_nanowakeword._tcp.local.",
        f"{name}._nanowakeword._tcp.local.",
        addresses=[socket.inet_aton(host)],
        port=http_port,
        properties={"version": __version__},
    )
    await AsyncZeroconf().async_register_service(service_info)
    _LOGGER.debug("HTTP API zeroconf discovery enabled on %s:%s", host, http_port)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
