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
