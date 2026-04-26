from __future__ import annotations

import json
from pathlib import Path

import pytest


def _vector_add_spec(fixtures_dir: Path) -> dict:
    return {
        "kernel": {
            "file": str(fixtures_dir / "kernel_vector_add.cu"),
            "function": "vectorAdd",
        },
        "compute_api": "cuda",
        "device": {"platform": 0, "device": 0},
        "scalars": [{"name": "N", "dtype": "int32", "value": 1024}],
        "vectors": [
            {"name": "a", "dtype": "float", "size": "N", "access": "read",
             "init": "random", "init_min": -1, "init_max": 1, "validate": False},
            {"name": "b", "dtype": "float", "size": "N", "access": "read",
             "init": "random", "init_min": -1, "init_max": 1, "validate": False},
            {"name": "c", "dtype": "float", "size": "N", "access": "write",
             "init": "zeros", "validate": True},
        ],
        "grid": {"x": "N"},
        "block": {"x": 1},
        "parameters": [{"name": "BLOCK_X", "values": [32, 64, 128, 256]}],
        "constraints": [{"params": ["BLOCK_X"], "expr": "N % BLOCK_X == 0"}],
        "launch_config": {
            "grid_x": "(N + BLOCK_X - 1) // BLOCK_X",
            "grid_y": "1", "grid_z": "1",
            "block_x": "BLOCK_X", "block_y": "1", "block_z": "1",
        },
        "reference": {
            "kind": "kernel",
            "file": str(fixtures_dir / "ref_kernel_vector_add.cu"),
            "function": "vectorAdd_ref",
            "block_x": 32, "block_y": 1,
        },
        "validation": {"method": "SideBySideComparison", "tolerance": 1e-4},
        "searcher": {"name": "Random"},
        "stop": {"kind": "duration", "value": 5.0},
    }


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_validate_passes_for_correct_kernel(fixtures_dir: Path, tmp_workdir: Path):
    from ktt_mcp.server import build_server
    server = build_server(workdir=str(tmp_workdir))
    spec = _vector_add_spec(fixtures_dir)
    _content, structured = await server.call_tool(
        "ktt_validate", {"spec": spec, "config": {"BLOCK_X": 64}}
    )
    assert structured["success"] is True
    assert structured["valid"] is True


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_run_returns_timing(fixtures_dir: Path, tmp_workdir: Path):
    from ktt_mcp.server import build_server
    server = build_server(workdir=str(tmp_workdir))
    spec = _vector_add_spec(fixtures_dir)
    _content, structured = await server.call_tool(
        "ktt_run", {"spec": spec, "config": {"BLOCK_X": 128}, "iterations": 3}
    )
    assert structured["success"] is True
    assert structured["timing"]["mean_us"] > 0
    assert structured["timing"]["iterations"] == 3


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_tune_finds_best_config(fixtures_dir: Path, tmp_workdir: Path):
    from ktt_mcp.server import build_server
    server = build_server(workdir=str(tmp_workdir))
    spec = _vector_add_spec(fixtures_dir)
    spec["stop"] = {"kind": "duration", "value": 8.0}
    _content, structured = await server.call_tool("ktt_tune", {"spec": spec, "top_n": 3})
    assert structured["success"] is True
    assert structured["best"] is not None
    assert structured["best"]["time_us"] > 0
    assert len(structured["top"]) >= 1
    assert Path(structured["results_file"]).exists()


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_profile_returns_duration(fixtures_dir: Path, tmp_workdir: Path):
    from ktt_mcp.server import build_server
    server = build_server(workdir=str(tmp_workdir))
    spec = _vector_add_spec(fixtures_dir)
    _content, structured = await server.call_tool(
        "ktt_profile", {"spec": spec, "config": {"BLOCK_X": 64}}
    )
    assert structured["success"] is True
    assert structured["duration_us"] is None or structured["duration_us"] > 0
    assert isinstance(structured["counters"], dict)
    assert structured["profiling_status"] in (
        "ok", "no_profiling_data", "no_counters_returned",
    )
    if structured["profiling_status"] == "ok":
        assert structured["counters"], "profiling_status=ok but no counters returned"
    else:
        # When profiling isn't available the message must explain why.
        assert structured.get("profiling_message")


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_run_and_tune_use_same_time_unit(fixtures_dir: Path, tmp_workdir: Path):
    """Regression: ktt_run.timing.mean_us must be in microseconds, matching ktt_tune.

    KTT's KernelResult.GetTotalDuration() returns nanoseconds at the API level,
    while SaveResults applies the configured TimeUnit. If run_one forgets to
    convert, ktt_run reports values 1000x larger than ktt_tune. Catch that here.
    """
    from ktt_mcp.server import build_server
    server = build_server(workdir=str(tmp_workdir))
    spec = _vector_add_spec(fixtures_dir)
    cfg = {"BLOCK_X": 128}

    _, run_out = await server.call_tool(
        "ktt_run", {"spec": spec, "config": cfg, "iterations": 1}
    )
    spec_for_tune = dict(spec)
    spec_for_tune["parameters"] = [{"name": "BLOCK_X", "values": [128]}]
    spec_for_tune["stop"] = {"kind": "count", "value": 1}
    _, tune_out = await server.call_tool("ktt_tune", {"spec": spec_for_tune, "top_n": 1})

    assert run_out["success"] and tune_out["success"]
    run_us = run_out["timing"]["mean_us"]
    tune_us = tune_out["best"]["time_us"]
    # both should be in the same unit, so within ~5x of each other for the same kernel/config
    ratio = max(run_us, tune_us) / max(min(run_us, tune_us), 1e-9)
    assert ratio < 5.0, f"run_us={run_us}, tune_us={tune_us}, ratio={ratio} — likely a unit mismatch"
