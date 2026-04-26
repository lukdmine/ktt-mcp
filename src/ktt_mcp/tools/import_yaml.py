"""Convert this repo's legacy problem.yaml + params.json into KttSpec dict.

Pure data transformation — no GPU, no validation logic beyond mapping fields.
The caller validates by feeding the returned dict into KttSpec.model_validate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

_DTYPE_ALIASES = {
    "int": "int32", "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "long": "int64", "char": "int8", "short": "int16",
    "uint": "uint32", "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
    "float": "float", "float32": "float",
    "double": "double", "float64": "double",
}


def _normalize_dtype(raw: str) -> str:
    norm = _DTYPE_ALIASES.get(str(raw).lower())
    if not norm:
        raise ValueError(f"Unsupported dtype: {raw}")
    return norm


def import_problem_yaml(problem_dir: str) -> dict[str, Any]:
    base = Path(problem_dir)
    yaml_path = base / "problem.yaml"
    params_path = base / "params.json"

    if not yaml_path.exists():
        return {"success": False, "stage": "io", "message": f"problem.yaml not found in {base}"}

    problem = yaml.safe_load(yaml_path.read_text())
    params: dict[str, Any] = {}
    if params_path.exists():
        params = json.loads(params_path.read_text())

    spec: dict[str, Any] = {
        "kernel": {
            "file": str(base / problem["kernel"]["file"]),
            "function": problem["kernel"]["function"],
        },
        "compute_api": "cuda",
        "device": {"platform": 0, "device": int(problem.get("gpu", {}).get("index", 0))},
        "scalars": [
            {"name": s["name"],
             "dtype": _normalize_dtype(s.get("dtype", "int")),
             "value": s["value"]}
            for s in problem.get("scalars", [])
        ],
        "vectors": [],
        "grid": {
            "x": problem["grid"]["x"],
            "y": problem["grid"].get("y", 1),
            "z": problem["grid"].get("z", 1),
        },
        "block": {"x": 1, "y": 1, "z": 1},
    }

    for v in problem.get("vectors", []):
        vec: dict[str, Any] = {
            "name": v["name"],
            "dtype": _normalize_dtype(v["dtype"]),
            "size": v["size"],
            "access": v["access"],
            "init": v.get("init", "zeros"),
            "validate": bool(v.get("validate", False)),
        }
        if "init_min" in v:
            vec["init_min"] = v["init_min"]
        if "init_max" in v:
            vec["init_max"] = v["init_max"]
        spec["vectors"].append(vec)

    ref = problem.get("reference") or {}
    if ref:
        ref_type = ref.get("type", "cuda")
        if ref_type == "cuda":
            spec["reference"] = {
                "kind": "kernel",
                "file": str(base / ref["file"]) if not Path(ref["file"]).is_absolute() else ref["file"],
                "function": ref["function"],
                "block_x": int(ref.get("block_x", 8)),
                "block_y": int(ref.get("block_y", 8)),
            }
        elif ref_type == "cpu_c":
            spec["reference"] = {
                "kind": "cpu_c",
                "file": str(base / ref["file"]) if not Path(ref["file"]).is_absolute() else ref["file"],
                "function": ref["function"],
            }
        else:
            spec["reference"] = {"kind": "none"}
    else:
        spec["reference"] = {"kind": "none"}

    spec["validation"] = {
        "method": "SideBySideComparison",
        "tolerance": float(problem.get("validation", {}).get("tolerance", 1e-4)),
    }

    spec["parameters"] = list(params.get("parameters", []))
    spec["constraints"] = list(params.get("constraints", []))
    if "launch_config" in params:
        lc = params["launch_config"]
        spec["launch_config"] = {
            "grid_x": str(lc.get("grid_x", "1")),
            "grid_y": str(lc.get("grid_y", "1")),
            "grid_z": str(lc.get("grid_z", "1")),
            "block_x": str(lc.get("block_x", "1")),
            "block_y": str(lc.get("block_y", "1")),
            "block_z": str(lc.get("block_z", "1")),
        }

    spec["searcher"] = {"name": "Random", "options": {}}
    duration = (problem.get("tuning") or {}).get("duration_s", 60)
    spec["stop"] = {"kind": "duration", "value": float(duration)}
    spec["profiling"] = {"enabled": False, "counters": []}

    return {"success": True, "spec": spec, "source_dir": str(base.resolve())}
