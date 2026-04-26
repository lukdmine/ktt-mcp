"""Shared pytest fixtures."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _pyktt_available() -> bool:
    env = os.environ.get("KTT_PYKTT_PATH")
    if env:
        so = Path(env)
        if so.is_file():
            parent = str(so.parent.resolve())
            if parent not in sys.path:
                sys.path.insert(0, parent)
    try:
        importlib.import_module("pyktt")
        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _pyktt_available():
        return
    skip_gpu = pytest.mark.skip(reason="pyktt not importable — set KTT_PYKTT_PATH or PYTHONPATH")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    workdir = tmp_path / ".ktt-mcp"
    workdir.mkdir()
    return workdir
