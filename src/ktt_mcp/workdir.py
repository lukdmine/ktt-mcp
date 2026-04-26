"""Run directory management. Each tool call writes artefacts to one run dir."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def compute_run_id(*, spec_json: str, kernel_source: str, at: str | None = None) -> str:
    """`<yyyymmdd-hhmm>-<6-hex>` where hex = sha256(spec || kernel)[:6]."""
    digest = hashlib.sha256()
    digest.update(spec_json.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(kernel_source.encode("utf-8"))
    short = digest.hexdigest()[:6]
    when = at or _dt.datetime.now().strftime("%Y%m%d-%H%M")
    return f"{when}-{short}"


@dataclass
class RunArtefacts:
    run_id: str
    run_dir: Path


class WorkdirManager:
    """Resolves the workdir and creates per-run directories."""

    def __init__(self, *, workdir: str | None = None) -> None:
        if workdir is not None:
            root = Path(workdir)
        elif env := os.environ.get("KTT_MCP_WORKDIR"):
            root = Path(env)
        else:
            root = Path.cwd() / ".ktt-mcp"
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        (self.root / "runs").mkdir(exist_ok=True)

    def runs_dir(self) -> Path:
        return self.root / "runs"

    def run_dir(self, run_id: str) -> Path:
        return (self.runs_dir() / run_id).resolve()

    def create_run_dir(
        self,
        run_id: str,
        *,
        spec: dict[str, Any],
        kernel_source: str,
        reference_source: str | None = None,
    ) -> RunArtefacts:
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True))
        (run_dir / "kernel.cu").write_text(kernel_source)
        if reference_source is not None:
            ext = ".cu" if "__global__" in reference_source or "__device__" in reference_source else ".c"
            (run_dir / f"reference{ext}").write_text(reference_source)
        return RunArtefacts(run_id=run_id, run_dir=run_dir)
