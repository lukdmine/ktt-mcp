"""FastMCP entry point. Builds the server with all tools registered."""

from __future__ import annotations

import asyncio
import json
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
from ktt_mcp.tools.profile import profile as _profile
from ktt_mcp.tools.run import run_config as _run_config
from ktt_mcp.tools.search_space import compute_search_space_size
from ktt_mcp.tools.tune import tune as _tune
from ktt_mcp.tools.validate import validate as _validate
from ktt_mcp.workdir import WorkdirManager

log = logging.getLogger(__name__)


def _coerce_spec(spec: dict[str, Any] | str) -> KttSpec:
    """Resolve `spec` to a KttSpec, accepting any of:

    - inline dict already shaped like a KttSpec
    - path to a `.json` file containing a canonical KttSpec
    - path to a `.yaml`/`.yml` file containing a canonical KttSpec
    - path to a directory containing `problem.yaml` (+ optional `params.json`)
      → translated via `ktt_import_problem_yaml`
    """
    if isinstance(spec, dict):
        return KttSpec.model_validate(spec)

    path = Path(spec)
    if path.is_dir():
        if not (path / "problem.yaml").exists():
            raise ValueError(
                f"Directory {path} has no problem.yaml; expected a legacy problem dir."
            )
        result = _import_yaml(str(path))
        if not result.get("success"):
            raise ValueError(result.get("message", "problem.yaml import failed"))
        return KttSpec.model_validate(result["spec"])

    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml as _yaml
        data = _yaml.safe_load(text)
    else:
        data = json.loads(text)
    return KttSpec.model_validate(data)


def _err(stage: str, message: str, **details: Any) -> dict[str, Any]:
    return {"success": False, "stage": stage, "message": message, "details": details}


def _try_coerce_spec(spec: dict[str, Any] | str) -> tuple[KttSpec | None, dict[str, Any] | None]:
    """Run `_coerce_spec` and translate failures into the standard error envelope.

    Returns `(spec, None)` on success, `(None, error_dict)` on failure.
    """
    try:
        return _coerce_spec(spec), None
    except ValidationError as e:
        return None, _err("spec", "Invalid KttSpec.", errors=e.errors())
    except (ValueError, FileNotFoundError, OSError) as e:
        return None, _err("spec", f"Could not load spec: {e}")


def build_server(*, workdir: str | None = None) -> FastMCP:
    mcp = FastMCP("ktt-mcp")
    workdir_mgr = WorkdirManager(workdir=workdir)
    gpu_lock = asyncio.Lock()

    @mcp.tool()
    async def ktt_search_space_size(spec: dict[str, Any] | str) -> dict[str, Any]:
        """Count valid configurations in a KTT spec without running anything.

        `spec` accepts:
          - an inline KttSpec dict
          - a path to a `.json` or `.yaml` canonical KttSpec file
          - a path to a directory containing `problem.yaml` (+ optional
            `params.json`) — the legacy format is converted on the fly

        Returns total, after-constraints count, and per-parameter cardinality.
        """
        parsed, err = _try_coerce_spec(spec)
        if err:
            return err
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

        `spec` accepts: inline dict | `.json`/`.yaml` file | directory with
        `problem.yaml`. `config` is a flat name -> int/float map. Returns
        valid pass/fail and the run_id.
        """
        parsed, err = _try_coerce_spec(spec)
        if err:
            return err
        async with gpu_lock:
            return _validate(spec=parsed, config=config, workdir_mgr=workdir_mgr)

    @mcp.tool()
    async def ktt_run(
        spec: dict[str, Any] | str,
        config: dict[str, int | float],
        iterations: int = 1,
    ) -> dict[str, Any]:
        """Run a single configuration N times and return timing stats.

        `spec` accepts: inline dict | `.json`/`.yaml` file | directory with
        `problem.yaml`.
        """
        parsed, err = _try_coerce_spec(spec)
        if err:
            return err
        async with gpu_lock:
            return _run_config(spec=parsed, config=config, workdir_mgr=workdir_mgr, iterations=iterations)

    @mcp.tool()
    async def ktt_tune(
        spec: dict[str, Any] | str,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Run full autotuning. Returns best config, top-N, and a summary string.

        `spec` accepts: inline dict | `.json`/`.yaml` file | directory with
        `problem.yaml`. The full results JSON is at the returned `results_file`
        path; pass it back to ktt_explain_results for deeper analysis.
        """
        parsed, err = _try_coerce_spec(spec)
        if err:
            return err
        async with gpu_lock:
            return _tune(spec=parsed, workdir_mgr=workdir_mgr, top_n=top_n)

    @mcp.tool()
    async def ktt_profile(
        spec: dict[str, Any] | str,
        config: dict[str, int | float],
        counters: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a config with profiling counters enabled.

        `spec` accepts: inline dict | `.json`/`.yaml` file | directory with
        `problem.yaml`. `counters=None` uses the default set; see
        ktt://docs/profiling-counters.
        """
        parsed, err = _try_coerce_spec(spec)
        if err:
            return err
        async with gpu_lock:
            return _profile(spec=parsed, config=config, counters=counters, workdir_mgr=workdir_mgr)

    # ----- resources -----
    examples_dir = Path(__file__).parent / "resources" / "examples"
    from ktt_mcp.resources.schema import schema_text
    from ktt_mcp.resources.docs import (
        BEST_PRACTICES_DOC,
        PROFILING_COUNTERS_DOC,
        SEARCHERS_DOC,
        STOP_CONDITIONS_DOC,
    )

    @mcp.resource("ktt://schema/spec.json")
    async def res_schema() -> str:
        return schema_text()

    @mcp.resource("ktt://docs/searchers")
    async def res_searchers() -> str:
        return SEARCHERS_DOC

    @mcp.resource("ktt://docs/stop-conditions")
    async def res_stops() -> str:
        return STOP_CONDITIONS_DOC

    @mcp.resource("ktt://docs/profiling-counters")
    async def res_counters() -> str:
        return PROFILING_COUNTERS_DOC

    @mcp.resource("ktt://docs/best-practices")
    async def res_practices() -> str:
        return BEST_PRACTICES_DOC

    @mcp.resource("ktt://examples/vector-add/spec")
    async def res_example_spec() -> str:
        return (examples_dir / "vector_add.json").read_text()

    @mcp.resource("ktt://examples/vector-add/kernel")
    async def res_example_kernel() -> str:
        return (examples_dir / "vector_add.cu").read_text()

    @mcp.resource("ktt://examples/vector-add/reference")
    async def res_example_ref() -> str:
        return (examples_dir / "vector_add_ref.cu").read_text()

    @mcp.resource("ktt://runs/{run_id}")
    async def res_run(run_id: str) -> str:
        rd = workdir_mgr.run_dir(run_id)
        results = rd / "results.json"
        if not results.exists():
            return json.dumps({"error": f"no results.json for run_id={run_id}"})
        return results.read_text()

    from ktt_mcp.prompts.templates import (
        iterate_on_kernel,
        port_from_tuning_loader,
        tune_cuda_kernel,
    )

    @mcp.prompt()
    def prompt_tune_cuda_kernel(kernel_file: str, function_name: str = "") -> str:
        """Walk through tuning a CUDA kernel: describe device, draft spec, validate, tune."""
        return tune_cuda_kernel(kernel_file, function_name or None)

    @mcp.prompt()
    def prompt_iterate_on_kernel(kernel_file: str, iterations: int = 5) -> str:
        """Iterative kernel-optimisation loop: validate -> tune -> profile -> rewrite."""
        return iterate_on_kernel(kernel_file, iterations)

    @mcp.prompt()
    def prompt_port_from_tuning_loader(loader_json_path: str) -> str:
        """Convert a KTT TuningLoader JSON to a ktt-mcp spec."""
        return port_from_tuning_loader(loader_json_path)

    log.info("ktt-mcp server initialised. workdir=%s", workdir_mgr.root)
    return mcp
