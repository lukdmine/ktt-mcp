from __future__ import annotations

from pathlib import Path

from ktt_mcp.spec import KttSpec
from ktt_mcp.tools.import_loader import import_loader_json


def test_basic_translation(fixtures_dir: Path):
    res = import_loader_json(str(fixtures_dir / "ktt_loader_sample.json"))
    assert res["success"] is True
    spec = res["spec"]

    assert spec["compute_api"] == "cuda"
    assert spec["kernel"]["function"] == "vectorAdd"
    assert spec["kernel"]["file"].endswith("kernel.cu")

    params = {p["name"]: p["values"] for p in spec["parameters"]}
    assert params == {"MWG": [16, 32, 64], "VWM": [1, 2, 4]}

    assert spec["constraints"][0]["expr"] == "MWG % VWM == 0"
    assert spec["searcher"]["name"] == "Random"
    assert spec["stop"]["kind"] == "duration"
    assert spec["stop"]["value"] == 30.0

    vectors = {v["name"] for v in spec["vectors"]}
    scalars = {s["name"] for s in spec["scalars"]}
    assert vectors == {"a", "b", "result"}
    assert scalars == {"scalar"}

    assert spec["validation"]["method"] == "SideBySideComparison"
    assert spec["validation"]["tolerance"] == 0.1
    target = next(v for v in spec["vectors"] if v["name"] == "result")
    assert target["validate"] is True


def test_imported_loader_validates(fixtures_dir: Path):
    res = import_loader_json(str(fixtures_dir / "ktt_loader_sample.json"))
    KttSpec.model_validate(res["spec"])


def test_unsupported_features_recorded(fixtures_dir: Path, tmp_path: Path):
    import json as _json
    raw = _json.loads((fixtures_dir / "ktt_loader_sample.json").read_text())
    raw["KernelSpecification"]["Arguments"][0]["FillType"] = "BinaryHDF"
    p = tmp_path / "exotic.json"
    p.write_text(_json.dumps(raw))

    res = import_loader_json(str(p))
    assert res["success"] is True
    assert any("BinaryHDF" in w for w in res["warnings"])
