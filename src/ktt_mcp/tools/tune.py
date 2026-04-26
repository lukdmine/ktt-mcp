"""Full autotuning: run KTT.Tune over the configured search space."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ktt_mcp.runtime.tuner_session import run_tune
from ktt_mcp.spec import KttSpec
from ktt_mcp.tools.explain_results import explain_results
from ktt_mcp.workdir import WorkdirManager, compute_run_id


def tune(
    *,
    spec: KttSpec,
    workdir_mgr: WorkdirManager,
    top_n: int = 5,
) -> dict[str, Any]:
    spec_json = spec.model_dump_json(indent=2)
    kernel_source = Path(spec.kernel.file).read_text()
    run_id = compute_run_id(spec_json=spec_json, kernel_source=kernel_source)
    artefacts = workdir_mgr.create_run_dir(
        run_id, spec=json.loads(spec_json), kernel_source=kernel_source,
    )

    try:
        tune_info = run_tune(spec, run_dir=artefacts.run_dir)
    except Exception as e:
        return {"success": False, "stage": "tune", "message": str(e), "run_id": run_id}

    digest = explain_results(tune_info["results_path"], top_n=top_n)
    if not digest.get("success"):
        return {**digest, "run_id": run_id}

    best = digest.get("best")
    summary = (
        f"Tested {tune_info['configurations_tested']} configurations. "
        f"Best: {best['time_us']:.1f}us at {best['config']}"
        if best else
        f"Tested {tune_info['configurations_tested']} configurations. No valid result."
    )

    if best:
        (artefacts.run_dir / "best_config.json").write_text(
            json.dumps({"config": best["config"], "time_us": best["time_us"]}, indent=2)
        )

    return {
        "success": True,
        "run_id": run_id,
        "run_dir": str(artefacts.run_dir),
        "configurations_tested": tune_info["configurations_tested"],
        "valid_configurations": digest["valid_configurations"],
        "best": best,
        "top": digest["top"],
        "summary": summary,
        "results_file": tune_info["results_path"],
        "failures_by_status": digest["failures_by_status"],
    }
