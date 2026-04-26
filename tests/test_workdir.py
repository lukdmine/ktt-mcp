from __future__ import annotations

import json
import re
from pathlib import Path

from ktt_mcp.workdir import WorkdirManager, compute_run_id


def test_compute_run_id_deterministic():
    a = compute_run_id(spec_json='{"x":1}', kernel_source="int main(){}", at="20260101-1200")
    b = compute_run_id(spec_json='{"x":1}', kernel_source="int main(){}", at="20260101-1200")
    assert a == b
    assert re.fullmatch(r"\d{8}-\d{4}-[0-9a-f]{6}", a)


def test_compute_run_id_changes_with_kernel():
    a = compute_run_id(spec_json='{"x":1}', kernel_source="A", at="20260101-1200")
    b = compute_run_id(spec_json='{"x":1}', kernel_source="B", at="20260101-1200")
    assert a != b


def test_resolve_workdir_explicit(tmp_path: Path):
    target = tmp_path / "explicit"
    mgr = WorkdirManager(workdir=str(target))
    assert mgr.root == target.resolve()
    assert mgr.root.exists()


def test_resolve_workdir_env(monkeypatch, tmp_path: Path):
    target = tmp_path / "from-env"
    monkeypatch.setenv("KTT_MCP_WORKDIR", str(target))
    mgr = WorkdirManager()
    assert mgr.root == target.resolve()


def test_resolve_workdir_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("KTT_MCP_WORKDIR", raising=False)
    monkeypatch.chdir(tmp_path)
    mgr = WorkdirManager()
    assert mgr.root == (tmp_path / ".ktt-mcp").resolve()


def test_create_run_dir_writes_artifacts(tmp_workdir: Path):
    mgr = WorkdirManager(workdir=str(tmp_workdir))
    run_id = "20260426-1400-abc123"
    spec = {"hello": "world"}
    artefacts = mgr.create_run_dir(run_id, spec=spec, kernel_source="__global__ void k(){}")
    assert artefacts.run_dir == (tmp_workdir / "runs" / run_id).resolve()
    assert (artefacts.run_dir / "spec.json").read_text() == json.dumps(spec, indent=2, sort_keys=True)
    assert (artefacts.run_dir / "kernel.cu").read_text() == "__global__ void k(){}"


def test_lookup_run_dir(tmp_workdir: Path):
    mgr = WorkdirManager(workdir=str(tmp_workdir))
    run_id = "20260426-1400-abc123"
    mgr.create_run_dir(run_id, spec={}, kernel_source="")
    assert mgr.run_dir(run_id) == (tmp_workdir / "runs" / run_id).resolve()
