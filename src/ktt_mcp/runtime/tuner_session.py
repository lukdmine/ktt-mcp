"""Build and run a ktt.Tuner from a KttSpec.

Single source of GPU-touching code. Each tool (validate/run/tune/profile) calls
into one of the run_*() helpers below. Per-call construction; no shared tuner
between calls.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ktt_mcp.runtime.pyktt_loader import load_pyktt
from ktt_mcp.runtime.reference import build_cpu_reference_callback
from ktt_mcp.spec import KttSpec, Vector


_COMPUTE_API = {"cuda": "CUDA", "opencl": "OpenCL", "cpp": "Cpp", "vulkan": "Vulkan"}
_ACCESS = {"read": "ReadOnly", "write": "WriteOnly", "readwrite": "ReadWrite"}
_NUMPY_BY_DTYPE = {
    "int8": np.int8, "int16": np.int16, "int32": np.int32, "int64": np.int64,
    "uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32, "uint64": np.uint64,
    "float": np.float32, "double": np.float64,
}
_VECTOR_METHOD = {
    "int8": "AddArgumentVectorChar", "int16": "AddArgumentVectorShort",
    "int32": "AddArgumentVectorInt", "int64": "AddArgumentVectorLong",
    "float": "AddArgumentVectorFloat", "double": "AddArgumentVectorDouble",
}


def _normalize_status(status: Any) -> str:
    """KTT's ResultStatus enum stringifies as 'ResultStatus.Ok'; we want just 'Ok'.

    Falls back to 'Unknown' if status is None.
    """
    if status is None:
        return "Unknown"
    name = getattr(status, "name", None)
    if name:
        return str(name)
    raw = str(status)
    return raw.rsplit(".", 1)[-1] or "Unknown"


def _eval_int(expr: Any, scalars: dict[str, int]) -> int:
    if isinstance(expr, int):
        return expr
    return int(eval(str(expr), {"__builtins__": {}}, scalars))


def _resolve_vector_size(v: Vector, scalars: dict[str, int]) -> int:
    return _eval_int(v.size, scalars)


def _make_vector_data(vectors: list[Vector], scalars: dict[str, int]) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    out: list[np.ndarray] = []
    for v in vectors:
        size = _resolve_vector_size(v, scalars)
        np_dtype = np.dtype(_NUMPY_BY_DTYPE[v.dtype])
        if v.init == "random":
            if np.issubdtype(np_dtype, np.floating):
                lo = float(v.init_min) if v.init_min is not None else -1.0
                hi = float(v.init_max) if v.init_max is not None else 1.0
                arr = rng.uniform(lo, hi, size).astype(np_dtype)
            else:
                lo = int(v.init_min) if v.init_min is not None else -2
                hi = (int(v.init_max) + 1) if v.init_max is not None else 3
                arr = rng.integers(lo, hi, size=size, dtype=np_dtype)
        elif v.init == "ones":
            arr = np.ones(size, dtype=np_dtype)
        elif v.init == "constant":
            cv = v.constant_value if v.constant_value is not None else 0
            arr = np.full(size, cv, dtype=np_dtype)
        else:  # zeros
            arr = np.zeros(size, dtype=np_dtype)
        out.append(arr)
    return out


def _scalar_defines_string(scalars: list) -> str:
    parts: list[str] = []
    for s in scalars:
        if s.dtype == "float":
            v = f"{float(s.value):.10g}"
            if "." not in v and "e" not in v.lower():
                v += ".0"
            parts.append(f"-D{s.name}={v}f")
        elif s.dtype == "double":
            v = f"{float(s.value):.10g}"
            if "." not in v and "e" not in v.lower():
                v += ".0"
            parts.append(f"-D{s.name}={v}")
        else:
            parts.append(f"-D{s.name}={int(s.value)}")
    return " ".join(parts)


@contextlib.contextmanager
def _redirect_to(stream_path: Path | None):
    if stream_path is None:
        yield
        return
    fd = os.open(stream_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _searcher(ktt, spec: KttSpec):
    name = spec.searcher.name
    opts = spec.searcher.options or {}
    if name == "Deterministic":
        return ktt.DeterministicSearcher()
    if name == "Random":
        return ktt.RandomSearcher()
    if name == "MCMC":
        seed = int(opts.get("seed", 42))
        if hasattr(ktt, "McmcSearcher"):
            return ktt.McmcSearcher(seed) if seed else ktt.McmcSearcher()
        return ktt.RandomSearcher()
    if name == "ProfileBased":
        # ProfileBased is set via SetProfileBasedSearcher rather than constructor;
        # caller handles via spec.searcher.options['model_path'].
        return ktt.RandomSearcher()
    return ktt.RandomSearcher()


def _stop_condition(ktt, spec: KttSpec):
    kind = spec.stop.kind
    val = spec.stop.value
    if kind == "duration":
        return ktt.TuningDuration(float(val))
    if kind == "count":
        return ktt.ConfigurationCount(int(val))
    if kind == "fraction":
        return ktt.ConfigurationFraction(float(val))
    if kind == "target_time":
        return ktt.ConfigurationDuration(float(val))
    return ktt.TuningDuration(60.0)


def _modifier_action(ktt, action: str):
    return getattr(ktt.ModifierAction, action)


def _modifier_type(ktt, t: str):
    return getattr(ktt.ModifierType, t)


def _modifier_dim(ktt, d: str):
    return getattr(ktt.ModifierDimension, d)


def _add_vector_arg(ktt, tuner, v: Vector, data: np.ndarray):
    method = getattr(tuner, _VECTOR_METHOD[v.dtype])
    access = getattr(ktt.ArgumentAccessType, _ACCESS[v.access])
    return method(data, access)


def _safe_eval_int(expr: str, ctx: dict[str, Any]) -> int:
    try:
        return int(eval(str(expr), {"__builtins__": {}}, ctx))
    except Exception:
        return 1


@dataclass
class BuiltSession:
    ktt: Any
    tuner: Any
    kernel_id: Any
    definition_id: Any
    arg_ids: list[Any]
    validated_vectors: list[tuple[Any, int]]
    vector_data: list[np.ndarray]
    has_cuda_ref: bool
    ref_kernel_id: Any | None


def _build(spec: KttSpec, *, run_dir: Path) -> BuiltSession:
    ktt = load_pyktt()
    api = getattr(ktt.ComputeApi, _COMPUTE_API[spec.compute_api])
    tuner = ktt.Tuner(spec.device.platform, spec.device.device, api)
    tuner.SetGlobalSizeType(ktt.GlobalSizeType.CUDA)
    tuner.SetTimeUnit(ktt.TimeUnit.Microseconds)

    scalar_values_int = {s.name: int(s.value) for s in spec.scalars if s.dtype.startswith("int")}
    base_grid_x = _eval_int(spec.grid.x, scalar_values_int)
    base_grid_y = _eval_int(spec.grid.y, scalar_values_int)
    base_grid_z = _eval_int(spec.grid.z, scalar_values_int)

    definition_id = tuner.AddKernelDefinitionFromFile(
        spec.kernel.function,
        spec.kernel.file,
        ktt.DimensionVector(base_grid_x, base_grid_y, base_grid_z),
        ktt.DimensionVector(1, 1, 1),
    )
    kernel_id = tuner.CreateSimpleKernel(spec.kernel.function, definition_id)

    compiler_opts = []
    cuda_include = os.environ.get("CUDA_INCLUDE")
    if cuda_include:
        compiler_opts.append(f"-I{cuda_include}")
    defines = _scalar_defines_string(spec.scalars)
    if defines:
        compiler_opts.append(defines)
    if spec.compiler_options:
        compiler_opts.append(spec.compiler_options)
    if compiler_opts:
        tuner.SetCompilerOptions(" ".join(compiler_opts))

    # vectors
    vector_data = _make_vector_data(spec.vectors, scalar_values_int)
    arg_ids: list[Any] = []
    validated_vectors: list[tuple[Any, int]] = []
    for i, v in enumerate(spec.vectors):
        aid = _add_vector_arg(ktt, tuner, v, vector_data[i])
        arg_ids.append(aid)
        if v.validate_output:
            validated_vectors.append((aid, i))
    tuner.SetArguments(definition_id, arg_ids)

    # parameters
    for p in spec.parameters:
        tuner.AddParameter(kernel_id, p.name, list(p.values))

    # compiler parameters
    for cp in spec.compiler_parameters:
        if cp.values:
            tuner.AddCompilerParameter(kernel_id, cp.switch, list(cp.values))
        else:
            tuner.AddCompilerParameter(kernel_id, cp.switch, [])

    # constraints
    scalar_name_set = {s.name for s in spec.scalars}
    for c in spec.constraints:
        tuning_param_names = [n for n in c.params if n not in scalar_name_set]
        if not tuning_param_names:
            continue

        def _make_constraint(expr: str, names: list[str], scalars: dict[str, Any]):
            def _fn(values):
                ns = dict(scalars)
                ns.update(dict(zip(names, values)))
                try:
                    return bool(eval(expr, {"__builtins__": {}}, ns))
                except Exception:
                    return False
            return _fn

        tuner.AddConstraint(
            kernel_id, tuning_param_names,
            _make_constraint(c.expr, tuning_param_names, {s.name: s.value for s in spec.scalars}),
        )

    # thread modifiers (if no launch_config provided)
    if spec.launch_config is None:
        for tm in spec.thread_modifiers:
            tuner.AddThreadModifier(
                kernel_id, [definition_id],
                _modifier_type(ktt, tm.type),
                _modifier_dim(ktt, tm.dim),
                tm.param,
                _modifier_action(ktt, tm.action),
            )
    else:
        # custom launcher driven by launch_config expressions
        lc = spec.launch_config
        scalar_ctx = {s.name: s.value for s in spec.scalars}

        def launcher(compute_interface):
            ctx = dict(scalar_ctx)
            for pair in compute_interface.GetCurrentConfiguration().GetPairs():
                ctx[pair.GetName()] = pair.GetValue()
            gx = _safe_eval_int(lc.grid_x, ctx)
            gy = _safe_eval_int(lc.grid_y, ctx)
            gz = _safe_eval_int(lc.grid_z, ctx)
            bx = _safe_eval_int(lc.block_x, ctx)
            by = _safe_eval_int(lc.block_y, ctx)
            bz = _safe_eval_int(lc.block_z, ctx)
            compute_interface.RunKernel(
                definition_id,
                ktt.DimensionVector(gx, gy, gz),
                ktt.DimensionVector(bx, by, bz),
            )

        tuner.SetLauncher(kernel_id, launcher)

    # reference
    has_cuda_ref = False
    ref_kernel_id = None
    if spec.reference.kind == "kernel":
        bx, by = spec.reference.block_x, spec.reference.block_y
        gx = max(1, (base_grid_x + bx - 1) // bx)
        gy = max(1, (base_grid_y + by - 1) // by)
        ref_def = tuner.AddKernelDefinitionFromFile(
            spec.reference.function,
            spec.reference.file,
            ktt.DimensionVector(gx, gy),
            ktt.DimensionVector(bx, by),
        )
        ref_kernel_id = tuner.CreateSimpleKernel(spec.reference.function + "_ref", ref_def)
        tuner.SetArguments(ref_def, arg_ids)
        has_cuda_ref = True

    # validation
    if validated_vectors:
        method_enum = getattr(ktt.ValidationMethod, spec.validation.method)
        tuner.SetValidationMethod(method_enum, float(spec.validation.tolerance))
        if spec.validation.range is not None:
            for aid, _ in validated_vectors:
                tuner.SetValidationRange(aid, int(spec.validation.range))
        if has_cuda_ref:
            for aid, _ in validated_vectors:
                tuner.SetReferenceKernel(aid, ref_kernel_id, ktt.KernelConfiguration())
        elif spec.reference.kind == "cpu_c":
            for aid, idx in validated_vectors:
                cb, ref_size = build_cpu_reference_callback(
                    source_path=Path(spec.reference.file),
                    function_name=spec.reference.function,
                    vectors=[v.model_dump() for v in spec.vectors],
                    vector_data=vector_data,
                    scalars=[s.model_dump() for s in spec.scalars],
                    validated_vector_index=idx,
                    build_dir=run_dir,
                )
                tuner.SetReferenceComputation(aid, ref_size, cb)

    return BuiltSession(
        ktt=ktt, tuner=tuner, kernel_id=kernel_id, definition_id=definition_id,
        arg_ids=arg_ids, validated_vectors=validated_vectors, vector_data=vector_data,
        has_cuda_ref=has_cuda_ref, ref_kernel_id=ref_kernel_id,
    )


# --- public entry points ---

def run_tune(spec: KttSpec, *, run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "stderr.log"
    with _redirect_to(log_path):
        sess = _build(spec, run_dir=run_dir)
        if spec.profiling.enabled:
            sess.tuner.SetProfiling(True)
            if spec.profiling.counters:
                sess.tuner.SetProfilingCounters(list(spec.profiling.counters))

        sess.tuner.SetSearcher(sess.kernel_id, _searcher(sess.ktt, spec))
        # ProfileBased searcher uses a separate API
        if spec.searcher.name == "ProfileBased":
            model_path = (spec.searcher.options or {}).get("model_path")
            if model_path:
                sess.tuner.SetProfileBasedSearcher(sess.kernel_id, str(model_path), True)

        stop = _stop_condition(sess.ktt, spec)
        results = sess.tuner.Tune(sess.kernel_id, stop)
        out_path = run_dir / "results"
        sess.tuner.SaveResults(results, str(out_path), sess.ktt.OutputFormat.JSON)
    return {"results_path": str(out_path) + ".json", "configurations_tested": len(results)}


def run_one(spec: KttSpec, *, config: dict[str, int | float], run_dir: Path,
             validate_only: bool = False, profile: bool = False,
             counters: list[str] | None = None) -> dict[str, Any]:
    log_path = run_dir / "stderr.log"
    with _redirect_to(log_path):
        sess = _build(spec, run_dir=run_dir)
        if profile:
            sess.tuner.SetProfiling(True)
            if counters:
                sess.tuner.SetProfilingCounters(counters)
        kc = sess.ktt.KernelConfiguration([
            sess.ktt.ParameterPair(name, int(value)) if isinstance(value, int)
            else sess.ktt.ParameterPair(name, float(value))
            for name, value in config.items()
        ])
        result = sess.tuner.Run(sess.kernel_id, kc, [])
    duration_us = getattr(result, "GetTotalDuration", lambda: None)()
    status = getattr(result, "GetStatus", lambda: None)()
    return {
        "duration_us": float(duration_us) if duration_us is not None else None,
        "status": _normalize_status(status),
    }
