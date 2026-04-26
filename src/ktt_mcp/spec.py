"""Canonical Pydantic model for a KTT tuning specification.

Single source of truth for tool input. Importers (loader-json, problem-yaml)
convert into this; runtime consumes only this.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# --- atomic enums (string-typed for human readability in JSON) ---

ComputeApi = Literal["cuda", "opencl", "cpp", "vulkan"]
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
    file: str
    function: str


class DeviceRef(_Strict):
    platform: int = 0
    device: int = 0


class Scalar(_Strict):
    name: str
    dtype: Dtype = "int32"
    value: float | int


class Vector(_Strict):
    name: str
    dtype: Dtype
    size: str | int                       # expression resolvable against scalars
    access: Access
    init: InitMode = "zeros"
    init_min: float | int | None = None
    init_max: float | int | None = None
    constant_value: float | int | None = None  # used when init == "constant"
    validate_output: bool = Field(default=False, alias="validate")


class GridSpec(_Strict):
    x: str | int
    y: str | int = 1
    z: str | int = 1


class BlockSpec(_Strict):
    x: str | int = 1
    y: str | int = 1
    z: str | int = 1


class Parameter(_Strict):
    name: str
    values: list[int | float]
    group: str = ""


class CompilerParameter(_Strict):
    switch: str                            # e.g. "-use_fast_math" or "-arch="
    values: list[str] = Field(default_factory=list)  # empty == binary toggle


class Constraint(_Strict):
    params: list[str]
    expr: str


class ThreadModifier(_Strict):
    type: ModifierType
    dim: ModifierDim
    param: str
    action: ModifierAction


class LaunchConfig(_Strict):
    grid_x: str = "1"
    grid_y: str = "1"
    grid_z: str = "1"
    block_x: str = "1"
    block_y: str = "1"
    block_z: str = "1"


class ValidationSpec(_Strict):
    method: ValidationMethod = "SideBySideComparison"
    tolerance: float = 1e-4
    range: int | None = None


class ReferenceSpec(_Strict):
    kind: ReferenceKind = "none"
    file: str | None = None
    function: str | None = None
    block_x: int = 8
    block_y: int = 8

    @model_validator(mode="after")
    def _check_kind(self) -> "ReferenceSpec":
        if self.kind in ("kernel", "cpu_c"):
            if not self.file or not self.function:
                raise ValueError(
                    f"reference.kind={self.kind} requires both 'file' and 'function'"
                )
        return self


class SearcherSpec(_Strict):
    name: SearcherName = "Random"
    options: dict[str, Any] = Field(default_factory=dict)


class StopSpec(_Strict):
    kind: StopKind
    value: float


class ProfilingSpec(_Strict):
    enabled: bool = False
    counters: list[str] = Field(default_factory=list)


# --- top-level model ---

class KttSpec(_Strict):
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

    launch_config: LaunchConfig | None = None
    compiler_options: str = ""

    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    reference: ReferenceSpec = Field(default_factory=ReferenceSpec)

    searcher: SearcherSpec = Field(default_factory=SearcherSpec)
    stop: StopSpec
    profiling: ProfilingSpec = Field(default_factory=ProfilingSpec)
