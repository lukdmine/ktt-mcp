"""FastMCP entry point. Builds the server with all tools registered."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from ktt_mcp.spec import KttSpec
from ktt_mcp.tools.devices import describe_device as _describe_device, list_devices as _list_devices
from ktt_mcp.tools.explain_results import explain_results as _explain
from ktt_mcp.tools.import_loader import import_loader_json as _import_loader
from ktt_mcp.tools.import_yaml import import_problem_yaml as _import_yaml
from ktt_mcp.tools.run import run_config as _run_config
from ktt_mcp.tools.search_space import compute_search_space_size
from ktt_mcp.tools.validate import validate as _validate
from ktt_mcp.workdir import WorkdirManager

log = logging.getLogger(__name__)


def _coerce_spec(spec: dict[str, Any] | str) -> KttSpec:
    """Accept either an inline dict or a path to a JSON spec file."""
    if isinstance(spec, str):
        data = Path(spec).read_text()
        import json as _json
        return KttSpec.model_validate(_json.loads(data))
    return KttSpec.model_validate(spec)


def _err(stage: str, message: str, **details: Any) -> dict[str, Any]:
    return {"success": False, "stage": stage, "message": message, "details": details}


def build_server(*, workdir: str | None = None) -> FastMCP:
    mcp = FastMCP("ktt-mcp")
    workdir_mgr = WorkdirManager(workdir=workdir)
    gpu_lock = asyncio.Lock()

    # expose for downstream tasks (Task 10+) to import
    mcp.state = {"workdir": workdir_mgr, "gpu_lock": gpu_lock}  # type: ignore[attr-defined]

    @mcp.tool()
    async def ktt_search_space_size(spec: dict[str, Any] | str) -> dict[str, Any]:
        """Count valid configurations in a KTT spec without running anything.

        Returns total, after-constraints count, and per-parameter cardinality.
        """
        try:
            parsed = _coerce_spec(spec)
        except ValidationError as e:
            return _err("spec", "Invalid KttSpec.", errors=e.errors())
        return {"success": True, **compute_search_space_size(parsed)}

    @mcp.tool()
    async def ktt_explain_results(results_path: str, top_n: int = 5) -> dict[str, Any]:
        """Read a KTT JSON results file and return a structured digest.

        Top-N configurations, parameter sensitivity, failure breakdown.
        """
        return _explain(results_path, top_n=top_n)

    @mcp.tool()
    async def ktt_import_problem_yaml(problem_dir: str) -> dict[str, Any]:
        """Convert this repo's legacy problem.yaml + params.json into a KttSpec.

        problem_dir is the path to a directory containing problem.yaml and (optionally) params.json.
        """
        return _import_yaml(problem_dir)

    @mcp.tool()
    async def ktt_import_loader_json(loader_path: str) -> dict[str, Any]:
        """Convert a KTT TuningLoader / Kernel Tuner JSON file into a KttSpec.

        Returns the spec plus a list of warnings about features that didn't translate.
        """
        return _import_loader(loader_path)

    @mcp.tool()
    async def ktt_list_devices(compute_api: str = "cuda") -> dict[str, Any]:
        """List available platforms and devices for the given compute API."""
        async with gpu_lock:
            return _list_devices(compute_api=compute_api)

    @mcp.tool()
    async def ktt_describe_device(
        compute_api: str = "cuda", platform: int = 0, device: int = 0
    ) -> dict[str, Any]:
        """Get detailed capabilities for a specific device."""
        async with gpu_lock:
            return _describe_device(compute_api=compute_api, platform=platform, device=device)

    @mcp.tool()
    async def ktt_validate(
        spec: dict[str, Any] | str,
        config: dict[str, int | float],
    ) -> dict[str, Any]:
        """Compile, launch, and validate a kernel + config without timing.

        config is a flat name -> int/float map. Returns valid pass/fail and the run_id.
        """
        try:
            parsed = _coerce_spec(spec)
        except ValidationError as e:
            return _err("spec", "Invalid KttSpec.", errors=e.errors())
        async with gpu_lock:
            return _validate(spec=parsed, config=config, workdir_mgr=workdir_mgr)

    @mcp.tool()
    async def ktt_run(
        spec: dict[str, Any] | str,
        config: dict[str, int | float],
        iterations: int = 1,
    ) -> dict[str, Any]:
        """Run a single configuration N times and return timing stats."""
        try:
            parsed = _coerce_spec(spec)
        except ValidationError as e:
            return _err("spec", "Invalid KttSpec.", errors=e.errors())
        async with gpu_lock:
            return _run_config(spec=parsed, config=config, workdir_mgr=workdir_mgr, iterations=iterations)

    log.info("ktt-mcp server initialised. workdir=%s", workdir_mgr.root)
    return mcp
