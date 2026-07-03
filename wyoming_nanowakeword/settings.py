"""Runtime-changeable server settings, persisted to settings.json.

CLI flags (and add-on options) provide the initial defaults; anything changed
through the HTTP API (i.e. from the Home Assistant integration) is written to
``<model_dir>/settings.json`` and wins on the next start. This is what lets
the whole server be configured from Home Assistant with no terminal access.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

SETTINGS_FILE = "settings.json"


@dataclass
class ServerSettings:
    """Detection and verification behavior, adjustable at runtime."""

    threshold: float = 0.95
    trigger_level: int = 1
    refractory_seconds: float = 2.0
    vad_threshold: float = 0.0
    cascade: bool = False
    gate_threshold: float = 0.3
    # Save pre-detection audio as WAV (training data).
    capture: bool = False
    # Hybrid satellite + server: on a candidate detection, ask another
    # wyoming-nanowakeword instance to verify the buffered audio before
    # emitting the Wyoming Detection event.
    verify: bool = False
    verify_url: str = ""
    verify_token: str = ""
    verify_model: str = ""
    verify_timeout: float = 3.0
    # When the verifier is unreachable: True = accept the detection anyway
    # (voice control keeps working), False = suppress it.
    verify_fail_open: bool = True

    def apply(self, changes: dict[str, object]) -> None:
        """Validate and apply a partial update. Raises ValueError."""

        valid = {field.name: field.type for field in fields(self)}
        for key, value in changes.items():
            if key not in valid:
                raise ValueError(f"Unknown setting {key!r}")

            current = getattr(self, key)
            if isinstance(current, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"Setting {key!r} must be a boolean")
            elif isinstance(current, int) and not isinstance(current, bool):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"Setting {key!r} must be an integer")
                if value < 1:
                    raise ValueError(f"Setting {key!r} must be >= 1")
            elif isinstance(current, float):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"Setting {key!r} must be a number")
                if value < 0:
                    raise ValueError(f"Setting {key!r} must be >= 0")
                value = float(value)
            elif isinstance(current, str):
                if not isinstance(value, str):
                    raise ValueError(f"Setting {key!r} must be a string")

            setattr(self, key, value)

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        # Never leak the verifier token in GET responses; a set token is
        # reported as a boolean.
        data["verify_token"] = bool(data["verify_token"])
        return data


def load_settings_overlay(settings: ServerSettings, model_dir: Path) -> None:
    """Overlay persisted settings (if any) on top of CLI defaults."""

    path = model_dir / SETTINGS_FILE
    if not path.is_file():
        return

    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        settings.apply(stored)
        _LOGGER.info("Loaded runtime settings from %s", path)
    except (ValueError, OSError) as err:
        _LOGGER.warning("Ignoring invalid %s: %s", path, err)


def save_settings(settings: ServerSettings, model_dir: Path) -> None:
    path = model_dir / SETTINGS_FILE
    path.write_text(
        json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
