"""Read a KTT results.json and return a structured digest for the agent."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _parse_value(raw: str) -> int | float | str:
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _config_dict(entry: dict[str, Any]) -> dict[str, int | float | str]:
    out: dict[str, int | float | str] = {}
    for pair in entry.get("Configuration", []):
        out[pair["Name"]] = _parse_value(pair["Value"])
    return out


def explain_results(results_path: str, *, top_n: int = 5) -> dict[str, Any]:
    path = Path(results_path)
    if not path.exists():
        return {
            "success": False,
            "stage": "io",
            "message": f"Results file not found: {results_path}",
        }
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "stage": "io",
            "message": f"Failed to parse JSON: {e}",
        }

    entries = data.get("Results") or data.get("results") or []
    valid = [e for e in entries if e.get("Status", "Ok") == "Ok" and "TotalDuration" in e]
    failed = [e for e in entries if e.get("Status", "Ok") != "Ok"]

    valid_sorted = sorted(valid, key=lambda e: e["TotalDuration"])
    top = [
        {"config": _config_dict(e), "time_us": e["TotalDuration"]}
        for e in valid_sorted[:top_n]
    ]

    by_param: dict[str, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for e in valid:
        cfg = _config_dict(e)
        for k, v in cfg.items():
            by_param[k][v].append(e["TotalDuration"])

    sensitivity: dict[str, dict[str, Any]] = {}
    for name, buckets in by_param.items():
        means = [statistics.fmean(times) for times in buckets.values() if times]
        rng = max(means) - min(means) if means else 0.0
        sensitivity[name] = {
            "distinct_values": len(buckets),
            "value_means_us": {str(v): statistics.fmean(t) for v, t in buckets.items() if t},
            "range_us": rng,
        }

    failures_by_status = dict(Counter(e.get("Status", "Unknown") for e in failed))

    return {
        "success": True,
        "total_configurations": len(entries),
        "valid_configurations": len(valid),
        "top": top,
        "best": top[0] if top else None,
        "parameter_sensitivity": sensitivity,
        "failures_by_status": failures_by_status,
    }
