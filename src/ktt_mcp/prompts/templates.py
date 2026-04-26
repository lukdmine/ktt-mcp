"""User-invocable prompt templates."""

from __future__ import annotations


def tune_cuda_kernel(kernel_file: str, function_name: str | None = None) -> str:
    fn = function_name or "(infer from file)"
    return f"""\
You are tuning a CUDA kernel via the ktt-mcp MCP server.

Kernel: {kernel_file}
Function: {fn}

Steps:
1. Read the kernel source at {kernel_file}.
2. Call ktt_describe_device to learn the target GPU's capabilities.
3. Call ktt_list_devices if you need to pick a different device.
4. Read ktt://docs/best-practices and ktt://schema/spec.json.
5. Draft a KttSpec for this kernel:
   - Pick scalars and vectors that match the function signature.
   - Choose tuning parameters (block size, unroll factor, vector width) appropriate for the kernel.
   - Add a reference (CUDA kernel or CPU C function) so validation runs.
   - Use a launch_config when grid/block depend on multiple parameters.
   - Add constraints to remove invalid combinations (e.g. block size <= 1024).
6. Call ktt_search_space_size with the spec; if it's huge, prune.
7. Call ktt_validate with one config to make sure the kernel even compiles.
8. Call ktt_tune with stop=duration~30s.
9. Report the best configuration and a one-line summary.
"""


def iterate_on_kernel(kernel_file: str, iterations: int = 5) -> str:
    return f"""\
You are iteratively optimising a CUDA kernel via ktt-mcp.

Kernel: {kernel_file}
Iterations: {iterations}

Loop:
1. Read the current kernel + spec.
2. ktt_validate one promising config — bail if it doesn't compile.
3. ktt_tune the current spec, brief budget.
4. ktt_profile the best config to learn what's bottlenecking it (DRAM-bound? occupancy-bound? branch-divergent?).
5. Propose a kernel rewrite based on the profiling counters.
6. Update the kernel file and spec; loop.

Stop when speedup plateaus, or after {iterations} iterations.
"""


def port_from_tuning_loader(loader_json_path: str) -> str:
    return f"""\
You are migrating an existing KTT TuningLoader JSON to ktt-mcp.

Source: {loader_json_path}

Steps:
1. Call ktt_import_loader_json with the source path.
2. Inspect the returned spec and warnings list.
3. For each warning, edit the spec by hand (e.g. write the missing reference kernel/function).
4. Call ktt_search_space_size with the spec to confirm the configuration count matches the original.
5. Call ktt_validate with one config to confirm semantic equivalence.
"""
