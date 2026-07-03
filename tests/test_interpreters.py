from __future__ import annotations

from pathlib import Path
from typing import Any

from wyoming_nanowakeword.interpreters import InterpreterManager
from wyoming_nanowakeword.state import State


class FakeInterpreter:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def _manager(tmp_path: Path) -> tuple[InterpreterManager, State, list[Any]]:
    (tmp_path / "hey_home.onnx").touch()
    state = State(model_dirs=[tmp_path])
    state.refresh()
    loaded: list[FakeInterpreter] = []

    def factory(**_kwargs: Any) -> FakeInterpreter:
        interpreter = FakeInterpreter()
        loaded.append(interpreter)
        return interpreter

    return InterpreterManager(state, factory=factory), state, loaded


def test_released_interpreters_are_reused(tmp_path: Path) -> None:
    manager, state, loaded = _manager(tmp_path)
    entry = state.models["hey_home"]

    first = manager.acquire_for_entry(entry)
    manager.release(first, state.generation)
    second = manager.acquire_for_entry(entry)

    assert len(loaded) == 1
    assert second["hey_home"] is first["hey_home"]
    # Reused interpreters are reset before being handed out.
    assert first["hey_home"].reset_count == 1


def test_stale_interpreters_are_dropped_after_refresh(tmp_path: Path) -> None:
    manager, state, loaded = _manager(tmp_path)
    entry = state.models["hey_home"]
    acquired_at = state.generation

    first = manager.acquire_for_entry(entry)
    state.refresh()  # model files may have changed
    manager.release(first, acquired_at)
    second = manager.acquire_for_entry(state.models["hey_home"])

    assert len(loaded) == 2
    assert second["hey_home"] is not first["hey_home"]


def test_warm_up_preloads_once(tmp_path: Path) -> None:
    manager, state, loaded = _manager(tmp_path)

    manager.warm_up(["hey_home"])
    manager.warm_up(["hey_home"])
    assert len(loaded) == 1

    acquired = manager.acquire_for_entry(state.models["hey_home"])
    assert acquired["hey_home"] is loaded[0]
