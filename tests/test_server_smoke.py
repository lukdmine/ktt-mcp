from __future__ import annotations

import pytest

from ktt_mcp.server import build_server


@pytest.mark.asyncio
async def test_server_lists_no_gpu_tools(tmp_workdir):
    server = build_server(workdir=str(tmp_workdir))
    tool_names = sorted(t.name for t in await server.list_tools())
    for required in [
        "ktt_search_space_size",
        "ktt_explain_results",
        "ktt_import_problem_yaml",
        "ktt_import_loader_json",
    ]:
        assert required in tool_names, f"missing {required} (got {tool_names})"


@pytest.mark.asyncio
async def test_search_space_tool_callable(tmp_workdir):
    server = build_server(workdir=str(tmp_workdir))
    spec = {
        "kernel": {"file": "k.cu", "function": "k"},
        "grid": {"x": "N"},
        "scalars": [{"name": "N", "dtype": "int32", "value": 64}],
        "parameters": [
            {"name": "A", "values": [1, 2, 3]},
            {"name": "B", "values": [4, 5]},
        ],
        "stop": {"kind": "duration", "value": 1.0},
    }
    _content, structured = await server.call_tool("ktt_search_space_size", {"spec": spec})
    assert structured["total"] == 6
    assert structured["after_constraints"] == 6


@pytest.mark.asyncio
async def test_resources_listed(tmp_workdir):
    from ktt_mcp.server import build_server
    server = build_server(workdir=str(tmp_workdir))
    uris = sorted(str(r.uri) for r in await server.list_resources())
    assert "ktt://schema/spec.json" in uris
    assert "ktt://docs/searchers" in uris
    assert "ktt://examples/vector-add/spec" in uris


@pytest.mark.asyncio
async def test_prompts_listed(tmp_workdir):
    from ktt_mcp.server import build_server
    server = build_server(workdir=str(tmp_workdir))
    names = sorted(p.name for p in await server.list_prompts())
    for required in ["prompt_tune_cuda_kernel", "prompt_iterate_on_kernel", "prompt_port_from_tuning_loader"]:
        assert required in names


@pytest.mark.asyncio
async def test_search_space_accepts_problem_dir(tmp_workdir, fixtures_dir):
    """Tools should accept a path to a legacy problem dir; auto-converts via import_yaml."""
    server = build_server(workdir=str(tmp_workdir))
    _content, structured = await server.call_tool(
        "ktt_search_space_size", {"spec": str(fixtures_dir / "legacy_problem")}
    )
    assert structured["success"] is True
    assert structured["total"] == 4  # BLOCK_X has 4 values
    assert structured["per_parameter"] == {"BLOCK_X": 4}


@pytest.mark.asyncio
async def test_search_space_accepts_json_path(tmp_workdir, tmp_path):
    """Tools should accept a path to a `.json` canonical spec."""
    spec_dict = {
        "kernel": {"file": "k.cu", "function": "k"},
        "grid": {"x": "N"},
        "scalars": [{"name": "N", "dtype": "int32", "value": 64}],
        "parameters": [{"name": "A", "values": [1, 2, 3, 4]}],
        "stop": {"kind": "duration", "value": 1.0},
    }
    p = tmp_path / "spec.json"
    import json as _json
    p.write_text(_json.dumps(spec_dict))

    server = build_server(workdir=str(tmp_workdir))
    _content, structured = await server.call_tool("ktt_search_space_size", {"spec": str(p)})
    assert structured["success"] is True
    assert structured["total"] == 4


@pytest.mark.asyncio
async def test_search_space_rejects_missing_path(tmp_workdir, tmp_path):
    """Bad spec paths produce the standard error envelope, not an exception."""
    server = build_server(workdir=str(tmp_workdir))
    _content, structured = await server.call_tool(
        "ktt_search_space_size", {"spec": str(tmp_path / "nope.json")}
    )
    assert structured["success"] is False
    assert structured["stage"] == "spec"
