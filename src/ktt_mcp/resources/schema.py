"""Expose the KttSpec JSON Schema as an MCP resource."""

from __future__ import annotations

import json

from ktt_mcp.spec import KttSpec


def schema_text() -> str:
    return json.dumps(KttSpec.model_json_schema(), indent=2)
