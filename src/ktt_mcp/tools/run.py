"""Single-configuration benchmark."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ktt_mcp.runtime.tuner_session import run_one
from ktt_mcp.spec import KttSpec
from ktt_mcp.workdir import WorkdirManager, compute_run_id


def run_config(
    *,
    spec: KttSpec,
    config: dict[str, int | float],
    workdir_mgr: WorkdirManager,
    iterations: int = 1,
) -> dict[str, Any]:
    spec_json = spec.model_dump_json(indent=2)
    kernel_source = Path(spec.kernel.file).read_text()
    run_id = compute_run_id(spec_json=spec_json, kernel_source=kernel_source)
    artefacts = workdir_mgr.create_run_dir(
        run_id, spec=json.loads(spec_json), kernel_source=kernel_source,
    )

    durations: list[float] = []
    last_status = "Ok"
    for _ in range(max(1, iterations)):
        try:
            info = run_one(spec, config=config, run_dir=artefacts.run_dir)
        except Exception as e:
            return {"success": False, "stage": "tune", "message": str(e), "run_id": run_id}
        if info["duration_us"] is not None:
            durations.append(info["duration_us"])
        last_status = info["status"]

    if not durations:
        return {
            "success": False, "stage": "launch", "run_id": run_id,
            "message": f"No timing produced; last status={last_status}",
        }

    summary = {
        "mean_us": statistics.fmean(durations),
        "min_us": min(durations),
        "max_us": max(durations),
        "stdev_us": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "iterations": len(durations),
    }
    return {"success": True, "run_id": run_id, "run_dir": str(artefacts.run_dir),
            "config": config, "timing": summary, "status": last_status}
