# ktt-mcp

MCP server exposing the [Kernel Tuning Toolkit](https://github.com/HiPerCoRe/KTT) for CUDA kernel autotuning. Drop a kernel + spec to Claude (or any MCP client); get back the best configuration, profiling metrics, or a single-config benchmark.

## Install

```bash
cd ktt-mcp
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

## Provide pyktt

The server needs `pyktt.so` (built with `--python` in KTT). Point at it:

```bash
export KTT_PYKTT_PATH=/abs/path/to/pyktt.so
```

(or place `pyktt.so` next to the `ktt_mcp` package).

## Register with Claude Code

```bash
claude mcp add --transport stdio --scope project ktt -- python -m ktt_mcp --workdir ./.ktt-mcp
```

Or write to `.mcp.json` directly (see `.mcp.json.example`).

## Tools

| Tool | Purpose |
| ---- | ------- |
| `ktt_tune` | Full autotune; returns best + top-N + summary. |
| `ktt_run`  | Run a single configuration N times, report timing stats. |
| `ktt_profile` | Run a config with CUPTI profiling counters. |
| `ktt_validate` | Compile + launch + validate without timing. |
| `ktt_search_space_size` | Count valid configs in a spec without GPU. |
| `ktt_list_devices` | List available platforms and devices for a compute API. |
| `ktt_describe_device` | Detailed capabilities of a specific device. |
| `ktt_explain_results` | Structured digest of a KTT JSON results file. |
| `ktt_import_loader_json` | Translate KTT TuningLoader JSON to a ktt-mcp spec. |
| `ktt_import_problem_yaml` | Translate this repo's `problem.yaml + params.json` to a ktt-mcp spec. |

## Resources

- `ktt://schema/spec.json` - JSON Schema for KttSpec
- `ktt://docs/searchers`, `stop-conditions`, `profiling-counters`, `best-practices`
- `ktt://examples/vector-add/spec` (kernel, reference)
- `ktt://runs/{run_id}` - read prior results

## Run dirs

Each tool call writes to `$KTT_MCP_WORKDIR/runs/<run_id>/` (default `./.ktt-mcp/runs/`):

```
spec.json  kernel.cu  results.json  best_config.json  profile.json  stderr.log
```

## Tests

```bash
pytest                                                    # CPU-only
KTT_PYKTT_PATH=/abs/path/pyktt.so pytest                  # incl. GPU smoke
```
