from __future__ import annotations

from pathlib import Path

import pytest

from ktt_mcp.tools.explain_results import explain_results


def test_explain_returns_top_n_sorted(fixtures_dir: Path):
    res = explain_results(str(fixtures_dir / "results_sample.json"), top_n=2)
    assert res["success"] is True
    assert res["total_configurations"] == 5
    assert res["valid_configurations"] == 3
    top = res["top"]
    assert len(top) == 2
    assert top[0]["time_us"] == 145.7
    assert top[0]["config"] == {"BLOCK_X": 128, "VEC": 2}
    assert top[1]["time_us"] == 180.2


def test_explain_failures_breakdown(fixtures_dir: Path):
    res = explain_results(str(fixtures_dir / "results_sample.json"), top_n=10)
    assert res["failures_by_status"] == {"ValidationFailed": 1, "CompileFailed": 1}


def test_explain_parameter_sensitivity(fixtures_dir: Path):
    res = explain_results(str(fixtures_dir / "results_sample.json"), top_n=10)
    sens = res["parameter_sensitivity"]
    assert "BLOCK_X" in sens
    assert "VEC" in sens
    assert sens["BLOCK_X"]["distinct_values"] == 3
    assert sens["VEC"]["distinct_values"] == 2


def test_explain_missing_file_returns_error(tmp_path: Path):
    res = explain_results(str(tmp_path / "nope.json"), top_n=5)
    assert res["success"] is False
    assert res["stage"] == "io"
