"""Compute total / valid configuration count from a KttSpec, no GPU needed."""

from __future__ import annotations

from itertools import product
from typing import Any

from ktt_mcp.spec import KttSpec


def _eval_constraint(expr: str, namespace: dict[str, Any]) -> bool:
    try:
        return bool(eval(expr, {"__builtins__": {}}, namespace))
    except ZeroDivisionError:
        return False
    except Exception:
        return False


def compute_search_space_size(spec: KttSpec) -> dict[str, Any]:
    if not spec.parameters:
        return {"total": 1, "after_constraints": 1, "per_parameter": {}}

    per_param = {p.name: len(p.values) for p in spec.parameters}
    total = 1
    for n in per_param.values():
        total *= n

    scalars = {s.name: s.value for s in spec.scalars}
    param_names = [p.name for p in spec.parameters]
    value_lists = [p.values for p in spec.parameters]

    if not spec.constraints:
        return {"total": total, "after_constraints": total, "per_parameter": per_param}

    valid = 0
    for combo in product(*value_lists):
        ns = dict(scalars)
        ns.update(dict(zip(param_names, combo)))
        ok = True
        for c in spec.constraints:
            if not _eval_constraint(c.expr, ns):
                ok = False
                break
        if ok:
            valid += 1

    return {"total": total, "after_constraints": valid, "per_parameter": per_param}
