"""Profile a single configuration with CUPTI counters.

Calls into runtime.tuner_session.run_profile, which loops on
HasRemainingProfilingRuns() so all requested counters get collected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ktt_mcp.runtime.tuner_session import run_profile
from ktt_mcp.spec import KttSpec
from ktt_mcp.workdir import WorkdirManager, compute_run_id

# Curated default counters useful to a kernel-rewriting LLM. Names match KTT's
# native counter set; not all are available on every device or every build.
DEFAULT_COUNTERS = [
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_bytes.sum",
    "lts__t_bytes.sum",
    "smsp__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_cycles_per_issue_active.avg",
    "smsp__inst_executed.sum",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
]


def profile(
    *,
    spec: KttSpec,
    config: dict[str, int | float],
    counters: list[str] | None,
    workdir_mgr: WorkdirManager,
) -> dict[str, Any]:
    chosen = list(counters) if counters else list(DEFAULT_COUNTERS)
    spec_json = spec.model_dump_json(indent=2)
    kernel_source = Path(spec.kernel.file).read_text()
    run_id = compute_run_id(spec_json=spec_json, kernel_source=kernel_source)
    artefacts = workdir_mgr.create_run_dir(
        run_id, spec=json.loads(spec_json), kernel_source=kernel_source,
    )

    try:
        info = run_profile(
            spec, config=config, counters=chosen, run_dir=artefacts.run_dir,
        )
    except Exception as e:
        return {"success": False, "stage": "profile", "message": str(e), "run_id": run_id}

    profile_path = artefacts.run_dir / "profile.json"
    profile_path.write_text(json.dumps({
        "config": config,
        "duration_us": info["duration_us"],
        "counters_requested": chosen,
        "counters": info["counters"],
        "counters_by_kernel": info["counters_by_kernel"],
        "profiling_status": info["profiling_status"],
        "profiling_iterations": info["profiling_iterations"],
    }, indent=2))

    return {
        "success": True,
        "run_id": run_id,
        "run_dir": str(artefacts.run_dir),
        "config": config,
        "duration_us": info["duration_us"],
        "counters": info["counters"],
        "counters_requested": chosen,
        "counters_by_kernel": info["counters_by_kernel"],
        "profiling_iterations": info["profiling_iterations"],
        "profiling_status": info["profiling_status"],
        "profiling_message": info["profiling_message"],
        "profile_file": str(profile_path),
    }
