from pathlib import Path

import pytest

from wyoming_nanowakeword.models import (
    discover_models,
    load_metadata,
    normalize_model_id,
)
from wyoming_nanowakeword.state import State


def test_normalize_model_id_removes_version_and_lite_suffix() -> None:
    assert normalize_model_id(Path("hey_home_v1.onnx")) == "hey_home"
    assert normalize_model_id(Path("hey_home_lite.onnx")) == "hey_home"
    assert normalize_model_id(Path("hey_home_v1_lite.onnx")) == "hey_home"


def test_discover_models_ignores_lite_as_public_model(tmp_path: Path) -> None:
    (tmp_path / "hey_home.onnx").touch()
    (tmp_path / "hey_home_lite.onnx").touch()
    (tmp_path / "other_v2.onnx").touch()

    models = discover_models([tmp_path])

    assert sorted(models) == ["hey_home", "other"]
    assert models["hey_home"].gate_path == tmp_path / "hey_home_lite.onnx"


def test_metadata_is_optional_and_descriptive(tmp_path: Path) -> None:
    (tmp_path / "hey_home.onnx").touch()
    (tmp_path / "models.yaml").write_text(
        """
models:
  hey_home:
    phrase: "Hey Home"
    language: "en"
    architecture: "bcresnet"
    version: "v1"
""",
        encoding="utf-8",
    )

    metadata = load_metadata(tmp_path)
    models = discover_models([tmp_path])

    assert metadata["hey_home"].architecture == "bcresnet"
    assert models["hey_home"].phrase == "Hey Home"
    assert models["hey_home"].metadata.language == "en"


def test_invalid_metadata_shape_raises(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_text("- bad\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_metadata(tmp_path)


def test_state_default_model_falls_back_to_first_model(tmp_path: Path) -> None:
    (tmp_path / "b_model.onnx").touch()
    (tmp_path / "a_model.onnx").touch()

    state = State(model_dirs=[tmp_path], default_model="missing")
    state.refresh()

    assert state.get_default_model_id() == "a_model"
