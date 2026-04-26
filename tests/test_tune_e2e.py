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
