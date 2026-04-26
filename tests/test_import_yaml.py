from __future__ import annotations

from pathlib import Path

from ktt_mcp.tools.import_yaml import import_problem_yaml


def test_import_legacy_problem(fixtures_dir: Path):
    res = import_problem_yaml(str(fixtures_dir / "legacy_problem"))
    assert res["success"] is True
    spec = res["spec"]
    assert spec["kernel"]["file"].endswith("kernel.cu")
    assert spec["kernel"]["function"] == "vectorAdd"
    assert spec["compute_api"] == "cuda"
    assert spec["reference"]["kind"] == "kernel"
    assert spec["reference"]["function"] == "vectorAdd_ref"
    assert spec["reference"]["block_x"] == 8

    [n] = spec["scalars"]
    assert n["name"] == "N" and n["value"] == 1024 and n["dtype"] == "int32"

    names = [v["name"] for v in spec["vectors"]]
    assert names == ["a", "b", "c"]
    assert spec["vectors"][0]["init_min"] == -1.0

    assert spec["parameters"][0]["name"] == "BLOCK_X"
    assert spec["constraints"][0]["expr"] == "N % BLOCK_X == 0"
    assert spec["launch_config"]["block_x"] == "BLOCK_X"
    assert spec["validation"]["tolerance"] == 0.001
    assert spec["stop"]["kind"] == "duration"
    assert spec["stop"]["value"] == 30.0


def test_import_missing_problem(tmp_path: Path):
    res = import_problem_yaml(str(tmp_path / "no-such-dir"))
    assert res["success"] is False
    assert res["stage"] == "io"


def test_imported_spec_validates(fixtures_dir: Path):
    from ktt_mcp.spec import KttSpec
    res = import_problem_yaml(str(fixtures_dir / "legacy_problem"))
    KttSpec.model_validate(res["spec"])  # raises if invalid
