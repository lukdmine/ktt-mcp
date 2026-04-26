from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ktt_mcp.spec import KttSpec, SearcherSpec


VECTOR_ADD_SPEC = {
    "kernel": {"file": "kernel.cu", "function": "vectorAdd"},
    "compute_api": "cuda",
    "device": {"platform": 0, "device": 0},
    "scalars": [{"name": "N", "dtype": "int32", "value": 1024}],
    "vectors": [
        {"name": "a", "dtype": "float", "size": "N", "access": "read", "init": "random",
         "init_min": -1.0, "init_max": 1.0, "validate": False},
        {"name": "b", "dtype": "float", "size": "N", "access": "read", "init": "random",
         "init_min": -1.0, "init_max": 1.0, "validate": False},
        {"name": "c", "dtype": "float", "size": "N", "access": "write", "init": "zeros",
         "validate": True},
    ],
    "grid": {"x": "N"},
    "block": {"x": 1},
    "parameters": [{"name": "BLOCK_X", "values": [32, 64, 128, 256]}],
    "thread_modifiers": [
        {"type": "Local", "dim": "X", "param": "BLOCK_X", "action": "Multiply"},
        {"type": "Global", "dim": "X", "param": "BLOCK_X", "action": "Divide"},
    ],
    "validation": {"method": "SideBySideComparison", "tolerance": 1e-4},
    "reference": {"kind": "none"},
    "searcher": {"name": "Random"},
    "stop": {"kind": "duration", "value": 10.0},
}


def test_minimal_valid_spec_parses():
    spec = KttSpec.model_validate(VECTOR_ADD_SPEC)
    assert spec.kernel.function == "vectorAdd"
    assert len(spec.parameters) == 1
    assert spec.searcher.name == "Random"


def test_unknown_compute_api_rejected():
    bad = dict(VECTOR_ADD_SPEC)
    bad["compute_api"] = "metal"
    with pytest.raises(ValidationError):
        KttSpec.model_validate(bad)


def test_unknown_dtype_rejected():
    bad = json.loads(json.dumps(VECTOR_ADD_SPEC))
    bad["scalars"][0]["dtype"] = "bf16"
    with pytest.raises(ValidationError):
        KttSpec.model_validate(bad)


def test_searcher_options_pass_through():
    spec = KttSpec.model_validate(
        VECTOR_ADD_SPEC | {"searcher": {"name": "MCMC", "options": {"seed": 42}}}
    )
    assert spec.searcher.name == "MCMC"
    assert spec.searcher.options == {"seed": 42}


def test_stop_kinds_accepted():
    for kind, value in [
        ("duration", 5.0),
        ("count", 100),
        ("fraction", 0.25),
        ("target_time", 1e-6),
    ]:
        spec = KttSpec.model_validate(VECTOR_ADD_SPEC | {"stop": {"kind": kind, "value": value}})
        assert spec.stop.kind == kind


def test_constraint_requires_expr():
    bad = VECTOR_ADD_SPEC | {"constraints": [{"params": ["BLOCK_X"]}]}
    with pytest.raises(ValidationError):
        KttSpec.model_validate(bad)


def test_reference_kernel_requires_file_and_function():
    bad = VECTOR_ADD_SPEC | {"reference": {"kind": "kernel"}}
    with pytest.raises(ValidationError):
        KttSpec.model_validate(bad)


def test_reference_kernel_full():
    spec = KttSpec.model_validate(
        VECTOR_ADD_SPEC | {
            "reference": {
                "kind": "kernel",
                "file": "ref.cu",
                "function": "ref",
                "block_x": 8,
                "block_y": 8,
            }
        }
    )
    assert spec.reference.kind == "kernel"
    assert spec.reference.block_x == 8


def test_json_schema_generates():
    schema = KttSpec.model_json_schema()
    assert schema["title"] == "KttSpec"
    assert "kernel" in schema["properties"]
