"""Convert a KTT TuningLoader JSON (Kernel Tuner-compatible) into KttSpec dict."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_LANGUAGE_TO_API = {"CUDA": "cuda", "OpenCL": "opencl", "Vulkan": "vulkan"}


def _parse_values(raw: str) -> list[int | float]:
    s = raw.strip()
    if s.startswith("["):
        s = s[1:-1]
    return [int(x) if x.lstrip("-").isdigit() else float(x) for x in re.split(r"\s*,\s*", s) if x]


def _scalar_dtype(raw: str) -> str:
    table = {
        "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
        "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
        "float": "float", "double": "double",
    }
    return table.get(raw, "float")


def _stop_from_budget(budget: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not budget:
        return {"kind": "duration", "value": 60.0}
    item = budget[0]
    typ = item["Type"]
    val = float(item["BudgetValue"])
    return {
        "TuningDuration": {"kind": "duration", "value": val},
        "ConfigurationCount": {"kind": "count", "value": val},
        "ConfigurationFraction": {"kind": "fraction", "value": val},
    }.get(typ, {"kind": "duration", "value": val})


def import_loader_json(loader_path: str) -> dict[str, Any]:
    path = Path(loader_path)
    if not path.exists():
        return {"success": False, "stage": "io", "message": f"Not found: {loader_path}"}

    raw = json.loads(path.read_text())
    warnings: list[str] = []

    cs = raw.get("ConfigurationSpace", {})
    spec_kernel = raw["KernelSpecification"]
    parent = path.parent

    spec: dict[str, Any] = {
        "kernel": {
            "file": str((parent / spec_kernel["KernelFile"]).resolve()),
            "function": spec_kernel["KernelName"],
        },
        "compute_api": _LANGUAGE_TO_API.get(spec_kernel["Language"], "cuda"),
        "device": {
            "platform": int(spec_kernel.get("Device", {}).get("PlatformId", 0)),
            "device": int(spec_kernel.get("Device", {}).get("DeviceId", 0)),
        },
        "scalars": [],
        "vectors": [],
        "grid": {
            "x": spec_kernel["GlobalSize"]["X"],
            "y": spec_kernel["GlobalSize"].get("Y", 1),
            "z": spec_kernel["GlobalSize"].get("Z", 1),
        },
        "block": {
            "x": spec_kernel["LocalSize"]["X"],
            "y": spec_kernel["LocalSize"].get("Y", 1),
            "z": spec_kernel["LocalSize"].get("Z", 1),
        },
        "parameters": [
            {"name": p["Name"], "values": _parse_values(p["Values"])}
            for p in cs.get("TuningParameters", [])
        ],
        "constraints": [
            {"params": list(c["Parameters"]), "expr": c["Expression"]}
            for c in cs.get("Conditions", [])
        ],
        "compiler_options": " ".join(spec_kernel.get("CompilerOptions", []) or []).strip(),
        "validation": {"method": "SideBySideComparison", "tolerance": 1e-4},
        "reference": {"kind": "none"},
        "searcher": {"name": raw.get("Search", {}).get("Name", "Random"), "options": {}},
        "stop": _stop_from_budget(raw.get("Budget")),
        "profiling": {"enabled": bool(spec_kernel.get("Profiling", False)), "counters": []},
    }

    target_for_validation: str | None = None
    for arg in spec_kernel.get("Arguments", []):
        mem = arg["MemoryType"]
        if mem == "Scalar":
            spec["scalars"].append({
                "name": arg["Name"],
                "dtype": _scalar_dtype(arg["Type"]),
                "value": arg.get("FillValue", 0),
            })
        elif mem == "Vector":
            fill = arg.get("FillType", "Constant")
            if fill == "Constant":
                init = "constant"; init_extra = {"constant_value": float(arg.get("FillValue", 0))}
            elif fill == "Random":
                init = "random"; init_extra = {}
            else:
                init = "zeros"; init_extra = {}
                warnings.append(f"Vector {arg['Name']!r} has FillType={fill}; defaulted to 'zeros'.")
            vec = {
                "name": arg["Name"],
                "dtype": _scalar_dtype(arg["Type"]),
                "size": arg.get("Size", 0),
                "access": {"ReadOnly": "read", "WriteOnly": "write", "ReadWrite": "readwrite"}[arg["AccessType"]],
                "init": init,
                **init_extra,
                "validate": False,
            }
            spec["vectors"].append(vec)
        else:
            warnings.append(f"Argument {arg.get('Name','?')!r} MemoryType={mem!r} not yet supported.")

    for ref in spec_kernel.get("ReferenceArguments", []):
        target_for_validation = ref.get("TargetName")
        spec["validation"] = {
            "method": ref.get("ValidationMethod", "SideBySideComparison"),
            "tolerance": float(ref.get("ValidationThreshold", 1e-4)),
        }
        warnings.append(
            f"ReferenceArguments[{ref.get('Name','?')}] uses FillType={ref.get('FillType')}: "
            "no host-side reference computation was generated. Edit spec.reference manually."
        )

    if target_for_validation:
        for v in spec["vectors"]:
            if v["name"] == target_for_validation:
                v["validate"] = True

    return {"success": True, "spec": spec, "warnings": warnings}
