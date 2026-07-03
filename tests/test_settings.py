from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyoming_nanowakeword.settings import (
    ServerSettings,
    load_settings_overlay,
    save_settings,
)


def test_apply_validates_types_and_keys() -> None:
    settings = ServerSettings()

    settings.apply({"threshold": 0.5, "cascade": True, "verify_url": "http://x"})
    assert settings.threshold == 0.5
    assert settings.cascade is True

    with pytest.raises(ValueError, match="Unknown setting"):
        settings.apply({"nope": 1})
    with pytest.raises(ValueError, match="must be a boolean"):
        settings.apply({"cascade": "yes"})
    with pytest.raises(ValueError, match="must be an integer"):
        settings.apply({"trigger_level": 1.5})
    with pytest.raises(ValueError, match=">= 1"):
        settings.apply({"trigger_level": 0})
    with pytest.raises(ValueError, match="must be a number"):
        settings.apply({"threshold": "high"})
    with pytest.raises(ValueError, match=">= 0"):
        settings.apply({"refractory_seconds": -1})


def test_persistence_roundtrip(tmp_path: Path) -> None:
    settings = ServerSettings(threshold=0.7, verify=True, verify_token="secret")
    save_settings(settings, tmp_path)

    loaded = ServerSettings()
    load_settings_overlay(loaded, tmp_path)
    assert loaded.threshold == 0.7
    assert loaded.verify is True
    assert loaded.verify_token == "secret"


def test_invalid_overlay_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text("{broken", encoding="utf-8")

    settings = ServerSettings()
    load_settings_overlay(settings, tmp_path)
    assert settings.threshold == 0.95  # defaults kept


def test_as_dict_masks_verify_token(tmp_path: Path) -> None:
    settings = ServerSettings(verify_token="secret")
    assert settings.as_dict()["verify_token"] is True
    assert ServerSettings().as_dict()["verify_token"] is False

    # ...but the persisted file keeps the real value for restarts.
    save_settings(settings, tmp_path)
    stored = json.loads((tmp_path / "settings.json").read_text())
    assert stored["verify_token"] == "secret"
