"""Centralised, robust import of the pyktt module."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def load_pyktt() -> ModuleType:
    """Return the pyktt module, raising RuntimeError with a helpful message if missing.

    Resolution order:
      1. KTT_PYKTT_PATH env var pointing at the .so file (its parent dir is added to sys.path).
      2. Already-importable `pyktt` on PYTHONPATH.
      3. `pyktt.so` next to the package (installed alongside).
    """
    env = os.environ.get("KTT_PYKTT_PATH")
    if env:
        so = Path(env)
        if so.is_file():
            parent = str(so.parent.resolve())
            if parent not in sys.path:
                sys.path.insert(0, parent)

    try:
        return importlib.import_module("pyktt")
    except ImportError:
        pkg_root = Path(__file__).resolve().parent.parent
        sibling = pkg_root / "pyktt.so"
        if sibling.is_file():
            sys.path.insert(0, str(pkg_root))
            return importlib.import_module("pyktt")
        raise RuntimeError(
            "pyktt module not importable. Set KTT_PYKTT_PATH to the path of pyktt.so, "
            "or place pyktt.so on PYTHONPATH."
        )
