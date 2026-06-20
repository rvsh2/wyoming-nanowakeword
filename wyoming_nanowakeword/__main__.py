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
    state.refresh()
    if not state.models:
        _LOGGER.warning(
            "No NanoWakeWord .onnx models found in: %s",
            ", ".join(str(model_dir) for model_dir in model_dirs) or "(none)",
        )
    else:
        _LOGGER.info("Discovered models: %s", ", ".join(sorted(state.models)))

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

    _LOGGER.info("Ready")
    try:
        await server.run(
            partial(
                NanoWakeWordEventHandler,
                args.threshold,
                args.trigger_level,
                args.refractory_seconds,
                args.vad_threshold,
                args.cascade,
                args.gate_threshold,
                state,
            )
        )
    except KeyboardInterrupt:
        pass


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
