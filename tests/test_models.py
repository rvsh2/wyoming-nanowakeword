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


def test_gate_must_match_model_version(tmp_path: Path) -> None:
    (tmp_path / "agata_bcresnet_v4_lite.onnx").touch()
    (tmp_path / "agata_bcresnet_v5.onnx").touch()
    (tmp_path / "agata_bcresnet_v5_lite.onnx").touch()

    models = discover_models([tmp_path])

    assert models["agata_bcresnet"].gate_path == (
        tmp_path / "agata_bcresnet_v5_lite.onnx"
    )


def test_duplicate_model_versions_warn_and_keep_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "hey_home_v1.onnx").touch()
    (tmp_path / "hey_home_v2.onnx").touch()

    with caplog.at_level("WARNING"):
        models = discover_models([tmp_path])

    assert models["hey_home"].path == tmp_path / "hey_home_v1.onnx"
    assert any("hey_home_v2" in record.message for record in caplog.records)


def test_ensemble_id_conflicting_with_model_file_raises(tmp_path: Path) -> None:
    (tmp_path / "agata.onnx").touch()
    (tmp_path / "agata_verifier.onnx").touch()
    (tmp_path / "models.yaml").write_text(
        """
models:
  agata:
    members:
      - model: "agata"
        role: "primary"
      - model: "agata_verifier"
        role: "verifier"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicts with model file"):
        discover_models([tmp_path])


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


def test_discover_ensemble_from_metadata(tmp_path: Path) -> None:
    (tmp_path / "agata_ebranchformer_v1.onnx").touch()
    (tmp_path / "agata_transformer_v1.onnx").touch()
    (tmp_path / "models.yaml").write_text(
        """
models:
  agata:
    phrase: "Agata"
    language: "pl"
    architecture: "ensemble:e_branchformer+transformer"
    fusion: "primary_and_verifier"
    members:
      - model: "agata_ebranchformer"
        role: "primary"
        threshold: 0.97
      - model: "agata_transformer"
        role: "verifier"
        threshold: 0.90
  agata_ebranchformer:
    hidden: true
    architecture: "e_branchformer"
  agata_transformer:
    hidden: true
    architecture: "transformer"
""",
        encoding="utf-8",
    )

    state = State(model_dirs=[tmp_path], default_model="agata")
    state.refresh()

    assert sorted(state.models) == ["agata"]
    assert sorted(state.backing_models) == [
        "agata_ebranchformer",
        "agata_transformer",
    ]
    assert state.models["agata"].is_ensemble
    assert state.models["agata"].members[0].model == "agata_ebranchformer"
