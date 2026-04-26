from __future__ import annotations

import pytest

from ktt_mcp.spec import KttSpec
from ktt_mcp.tools.search_space import compute_search_space_size


BASE = {
    "kernel": {"file": "k.cu", "function": "k"},
    "grid": {"x": "N"},
    "scalars": [{"name": "N", "dtype": "int32", "value": 64}],
    "stop": {"kind": "duration", "value": 1.0},
}


def _spec(**overrides):
    return KttSpec.model_validate(BASE | overrides)


def test_no_parameters_returns_one_config():
    res = compute_search_space_size(_spec())
    assert res["total"] == 1
    assert res["after_constraints"] == 1
    assert res["per_parameter"] == {}


def test_simple_product():
    spec = _spec(parameters=[
        {"name": "A", "values": [1, 2, 3]},
        {"name": "B", "values": [4, 5]},
    ])
    res = compute_search_space_size(spec)
    assert res["total"] == 6
    assert res["after_constraints"] == 6
    assert res["per_parameter"] == {"A": 3, "B": 2}


def test_constraint_eliminates_some():
    spec = _spec(
        parameters=[
            {"name": "A", "values": [1, 2, 3]},
            {"name": "B", "values": [1, 2, 3]},
        ],
        constraints=[{"params": ["A", "B"], "expr": "A + B <= 4"}],
    )
    res = compute_search_space_size(spec)
    assert res["total"] == 9
    # 1+1, 1+2, 1+3, 2+1, 2+2, 3+1 -> 6
    assert res["after_constraints"] == 6


def test_constraint_uses_scalars():
    spec = _spec(
        parameters=[{"name": "BLOCK", "values": [1, 32, 64, 128]}],
        constraints=[{"params": ["BLOCK"], "expr": "N % BLOCK == 0"}],
    )
    res = compute_search_space_size(spec)
    assert res["total"] == 4
    valid = sum(1 for b in [1, 32, 64, 128] if 64 % b == 0)
    assert res["after_constraints"] == valid


def test_constraint_division_by_zero_treated_as_invalid():
    spec = _spec(
        parameters=[{"name": "A", "values": [0, 1, 2]}],
        constraints=[{"params": ["A"], "expr": "10 / A > 1"}],
    )
    res = compute_search_space_size(spec)
    # A=0 invalid (div0), A=1 -> 10>1 true, A=2 -> 5>1 true
    assert res["after_constraints"] == 2
