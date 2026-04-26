"""Single-config correctness gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ktt_mcp.runtime.tuner_session import run_one
from ktt_mcp.spec import KttSpec
from ktt_mcp.workdir import RunArtefacts, WorkdirManager, compute_run_id


def validate(
    *,
    spec: KttSpec,
    config: dict[str, int | float],
    workdir_mgr: WorkdirManager,
) -> dict[str, Any]:
    spec_json = spec.model_dump_json(indent=2)
    kernel_source = Path(spec.kernel.file).read_text()
    run_id = compute_run_id(spec_json=spec_json, kernel_source=kernel_source)
    artefacts: RunArtefacts = workdir_mgr.create_run_dir(
        run_id, spec=json.loads(spec_json), kernel_source=kernel_source,
    )
    try:
        info = run_one(spec, config=config, run_dir=artefacts.run_dir, validate_only=True)
    except Exception as e:
        return {"success": False, "stage": "validate", "message": str(e), "run_id": run_id}
    valid = info["status"] in ("Ok", "Unknown")
    return {
        "success": True,
        "valid": valid,
        "status": info["status"],
        "run_id": run_id,
        "run_dir": str(artefacts.run_dir),
    }
