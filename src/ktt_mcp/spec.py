"""Canonical Pydantic model for a KTT tuning specification.

Single source of truth for tool input. Importers (loader-json, problem-yaml)
convert into this; runtime consumes only this.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# --- atomic enums (string-typed for human readability in JSON) ---

ComputeApi = Literal["cuda", "opencl", "cpp", "vulkan"]
# `cuda`/`opencl`/`vulkan` → GPU backends, kernel signature is the standard
# positional `(const T* a, T* c, ...)` model.
# `cpp` → host CPU autotuning with g++ JIT; kernel signature is FIXED as
# `extern "C" void k(void** buffers, size_t* sizes)`. See KttSpec docstring.
Dtype = Literal[
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float", "double",
]
Access = Literal["read", "write", "readwrite"]
InitMode = Literal["zeros", "random", "ones", "constant"]
ValidationMethod = Literal[
    "AbsoluteDifference", "SideBySideComparison", "SideBySideRelativeComparison"
]
ReferenceKind = Literal["none", "kernel", "cpu_c"]
SearcherName = Literal["Deterministic", "Random", "MCMC", "ProfileBased"]
StopKind = Literal["duration", "count", "fraction", "target_time"]
ModifierType = Literal["Local", "Global"]
ModifierDim = Literal["X", "Y", "Z"]
ModifierAction = Literal["Add", "Subtract", "Multiply", "Divide"]


# --- nested models ---

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class KernelRef(_Strict):
    file: str = Field(
        description="Absolute path to the kernel source file. Relative paths are resolved on the SERVER's cwd, so always pass absolute."
    )
    function: str = Field(description="Name of the kernel function (e.g. `vectorAdd`).")


class DeviceRef(_Strict):
    platform: int = Field(default=0, description="Platform index from ktt_list_devices.")
    device: int = Field(default=0, description="Device index from ktt_list_devices.")


class Scalar(_Strict):
    """A problem scalar — injected as a COMPILE-TIME PREPROCESSOR DEFINE, not a runtime kernel argument.

    The scalar is added to the kernel compile flags as `-D<name>=<value>` (with the
    appropriate `f`/`.0` suffix for floats/doubles). Inside the kernel source, write
    `c[i] = a[i] + b[i]` and use `N` as a literal — DO NOT add an `int N` parameter
    to the function signature. Adding a kernel-arg parameter for a scalar will
    produce a CUDA compile error (`int 1024` after preprocessing).

    Use scalars for problem dimensions (M, N, K, batch size, tile size of the
    workload, etc.) and any constant the kernel and reference both want to see.
    They're also evaluated inside vector `size`, grid, block, and constraint expressions.
    """

    name: str = Field(description="Identifier injected as `-D<name>=<value>`. Used in expressions and inside the kernel as a preprocessor constant.")
    dtype: Dtype = Field(default="int32", description="Determines the `-D` formatting (e.g. `1.0f` vs `1` vs `1.0`).")
    value: float | int


class Vector(_Strict):
    name: str
    dtype: Dtype
    size: str | int = Field(
        description="Element count. May be an integer literal or an expression in scalar names (e.g. `M*N` if M and N are scalars)."
    )
    access: Access = Field(description="`read` for input buffers, `write` for output, `readwrite` for in-place.")
    init: InitMode = Field(default="zeros", description="How to populate the buffer before the kernel runs.")
    init_min: float | int | None = Field(
        default=None, description="Lower bound when `init=random` (default -1.0 for floats, -2 for ints)."
    )
    init_max: float | int | None = Field(
        default=None, description="Upper bound (exclusive for ints) when `init=random`."
    )
    constant_value: float | int | None = Field(
        default=None, description="Fill value when `init=constant`."
    )
    validate_output: bool = Field(
        default=False, alias="validate",
        description="If true, KTT compares this buffer against the reference after each run. AT LEAST ONE vector must have validate=true (and a reference must be set) to catch incorrect kernels."
    )


class GridSpec(_Strict):
    x: str | int = Field(description="Total threads / NDRange size in the X dim. Integer literal or expression in scalars (e.g. `N`).")
    y: str | int = 1
    z: str | int = 1


class BlockSpec(_Strict):
    x: str | int = 1
    y: str | int = 1
    z: str | int = 1


class Parameter(_Strict):
    name: str = Field(description="Tuning parameter name; injected as `-D<name>=<value>` into the kernel.")
    values: list[int | float] = Field(description="Discrete set of values to try.")
    group: str = Field(default="", description="Optional grouping for composite-kernel parameter independence.")


class CompilerParameter(_Strict):
    switch: str = Field(
        description="Compiler switch text. Empty `values=[]` makes it a binary toggle (e.g. `-use_fast_math`); non-empty appends the value (e.g. switch=`-arch=` with values=[`sm_80`]→ `-arch=sm_80`)."
    )
    values: list[str] = Field(default_factory=list)


class Constraint(_Strict):
    params: list[str] = Field(description="Parameter (and optional scalar) names referenced by `expr`.")
    expr: str = Field(
        description="Python boolean expression; combos that evaluate True are kept. Example: `BLOCK_X * BLOCK_Y <= 1024`. ZeroDivisionError → invalid (combo dropped)."
    )


class ThreadModifier(_Strict):
    type: ModifierType = Field(description="`Local` modifies block size; `Global` modifies grid size.")
    dim: ModifierDim
    param: str = Field(description="Tuning parameter whose value drives the modification.")
    action: ModifierAction


class LaunchConfig(_Strict):
    """Custom launcher: grid/block as Python expressions in scalars + parameter values.

    Use this INSTEAD of thread_modifiers when grid/block depend on multiple parameters
    (e.g. `grid_x = (N + BX - 1) // BX`). When both are present, launch_config wins.
    """
    grid_x: str = "1"
    grid_y: str = "1"
    grid_z: str = "1"
    block_x: str = "1"
    block_y: str = "1"
    block_z: str = "1"


class ValidationSpec(_Strict):
    method: ValidationMethod = Field(default="SideBySideComparison", description="How KTT compares output to reference.")
    tolerance: float = Field(default=1e-4, description="Absolute or relative tolerance depending on method.")
    range: int | None = Field(default=None, description="Optionally validate only the first N elements.")


class ReferenceSpec(_Strict):
    kind: ReferenceKind = Field(
        default="none",
        description="`none` (no validation), `kernel` (CUDA reference kernel), or `cpu_c` (compiled C/C++ reference function — wrap in `extern \"C\"`)."
    )
    file: str | None = Field(default=None, description="Absolute path to the reference source. Required when kind != none.")
    function: str | None = Field(default=None, description="Reference function name. Required when kind != none.")
    block_x: int = Field(default=8, description="Reference kernel block size X (only used when kind=kernel).")
    block_y: int = Field(default=8, description="Reference kernel block size Y.")

    @model_validator(mode="after")
    def _check_kind(self) -> "ReferenceSpec":
        if self.kind in ("kernel", "cpu_c"):
            if not self.file or not self.function:
                raise ValueError(
                    f"reference.kind={self.kind} requires both 'file' and 'function'"
                )
        return self


class SearcherSpec(_Strict):
    name: SearcherName = Field(
        default="Random",
        description="Search strategy. `Deterministic` (small spaces), `Random` (default), `MCMC` (smooth landscapes — opt `seed`), `ProfileBased` (needs `model_path`). See ktt://docs/searchers."
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Searcher-specific options (e.g. `{seed: 42}` for MCMC, `{model_path: ...}` for ProfileBased)."
    )


class StopSpec(_Strict):
    kind: StopKind = Field(
        description="`duration` (seconds), `count` (configs), `fraction` (of space), `target_time` (stop on a config faster than `value` µs)."
    )
    value: float


class ProfilingSpec(_Strict):
    enabled: bool = Field(default=False, description="If true, KTT runs each config under CUPTI; significantly slower.")
    counters: list[str] = Field(
        default_factory=list,
        description="NVPerf counter names; empty = KTT default set. See ktt://docs/profiling-counters."
    )


# --- top-level model ---

class KttSpec(_Strict):
    """Canonical KTT tuning specification.

    Workflow: ktt_describe_device → draft this spec → ktt_search_space_size →
    ktt_validate one config → ktt_tune. See server-level instructions.

    All file paths (kernel.file, reference.file) MUST be absolute — KTT
    resolves them on the server's filesystem, not the client's.

    KERNEL ARGUMENT MODEL — depends on `compute_api`:

    For `cuda` / `opencl` / `vulkan` (GPU backends):
      - `vectors` become POSITIONAL kernel arguments in the order listed. The
        kernel function signature must contain exactly those pointers, in that
        order, with matching dtypes (e.g. `const float*` for read, `float*` for write).
      - `scalars` are injected as `-D<name>=<value>` PREPROCESSOR DEFINES — they
        are NOT passed as kernel arguments. Inside the kernel use `N` as a literal
        (`if (i < N)`); do NOT add `int N` to the function signature, that will
        either fail to compile (`int 1024` after preprocessing) or silently shadow
        the constant.
      - Tuning `parameters` are also `-D<name>=<value>` defines (`#if BLOCK_X >= 64`).

    For `cpp` (host C/C++ CPU backend, JIT-compiled with g++):
      - The kernel signature is FIXED by KTT:
            `extern "C" void kernelName(void** buffers, size_t* sizes)`
        Vectors arrive through `buffers[i]` (cast to your dtype) in the order
        you declared them. `sizes[]` carries the byte length of each buffer.
      - `scalars` are still `-D` defines (so `N` is a preprocessor constant
        in the kernel body); KTT also makes them available via `sizes[N+i]`
        but using the define is simpler.
      - Tuning `parameters` are `-D` defines just like the GPU backends.
      - Use `compiler_options` to pass `-fopenmp` etc. for parallelism.
      - This path is supported by KTT but not yet exercised in our test suite —
        treat it as preview.
    """

    kernel: KernelRef
    compute_api: ComputeApi = "cuda"
    device: DeviceRef = Field(default_factory=DeviceRef)

    scalars: list[Scalar] = Field(default_factory=list)
    vectors: list[Vector] = Field(default_factory=list)

    grid: GridSpec
    block: BlockSpec = Field(default_factory=BlockSpec)

    parameters: list[Parameter] = Field(default_factory=list)
    compiler_parameters: list[CompilerParameter] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    thread_modifiers: list[ThreadModifier] = Field(default_factory=list)

    launch_config: LaunchConfig | None = Field(
        default=None,
        description="Custom Python expressions for grid/block. Use INSTEAD of thread_modifiers when dimensions depend on multiple parameters."
    )
    compiler_options: str = Field(default="", description="Free-form NVRTC/compiler options string (e.g. `-O3 -arch=sm_80`).")

    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    reference: ReferenceSpec = Field(default_factory=ReferenceSpec)

    searcher: SearcherSpec = Field(default_factory=SearcherSpec)
    stop: StopSpec
    profiling: ProfilingSpec = Field(default_factory=ProfilingSpec)
