from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from saps import freshness


@pytest.mark.parametrize("origin", ["frozen", "built-in", "/extension.so", None])
def test_non_python_origins_do_not_resolve_filesystem_paths(monkeypatch, origin):
    monkeypatch.setattr(
        freshness.importlib.util,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None, origin=origin),
    )

    def unavailable_cwd(self, *args, **kwargs):
        raise FileNotFoundError("working directory no longer exists")

    monkeypatch.setattr(Path, "resolve", unavailable_cwd)
    assert freshness._module_path("example") is None
