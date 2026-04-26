"""Static markdown docs surfaced as MCP resources."""

SEARCHERS_DOC = """\
# KTT searchers

| Name           | When to use |
| -------------- | ----------- |
| Deterministic  | Tiny spaces (<100). Reproducible. Walks every config in fixed order. |
| Random         | Default. Use when you have a budget and no prior. |
| MCMC           | Large spaces with smooth performance landscape. Options: `seed`. |
| ProfileBased   | Requires a model. Fastest convergence on known kernel families. Options: `model_path` (str). |
"""

STOP_CONDITIONS_DOC = """\
# KTT stop conditions

| `stop.kind`    | `stop.value`                                    |
| -------------- | ----------------------------------------------- |
| duration       | seconds of wall-clock budget                    |
| count          | number of distinct configurations               |
| fraction       | fraction of total space, e.g. 0.25              |
| target_time    | stop once a config runs faster than X us        |
"""

PROFILING_COUNTERS_DOC = """\
# Default profiling counters

KTT exposes CUPTI counters under their NVPerf names. The default set is curated for kernel-perf reasoning:

- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum` - FP32 adds
- `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` - FP32 FMAs
- `dram__bytes_read.sum`, `dram__bytes_write.sum` - HBM traffic
- `l1tex__t_bytes.sum`, `lts__t_bytes.sum` - L1/L2 traffic
- `smsp__warps_active.avg.pct_of_peak_sustained_active` - achieved occupancy
- `smsp__warp_cycles_per_issue_active.avg` - issue stall pressure
- `smsp__inst_executed.sum` - total instructions
- `smsp__sass_average_branch_targets_threads_uniform.pct` - branch divergence

Pass an explicit `counters` list to ktt_profile to use a custom set.

## Two preconditions for non-empty counters

KTT profiling has two gates. If `ktt_profile` returns `profiling_status` other than `"ok"`, check both:

1. **KTT build flag.** `pyktt.so` must have been built with `--profiling=cupti` (or `--profiling=cupti-legacy`). Without it, `SetProfiling(True)` silently does nothing — `profiling_status` will be `"no_profiling_data"`. To rebuild:
   ```
   cd KTT
   ./premake5 gmake --python --profiling=cupti
   cd Build && make config=release_x86_64
   ```

2. **CUPTI permissions.** Modern NVIDIA drivers (440+) restrict CUPTI access to root by default. Allow non-root profiling once: create `/etc/modprobe.d/nvidia.conf` with:
   ```
   options nvidia NVreg_RestrictProfilingToAdminUsers=0
   ```
   then reboot. Verify with `cat /proc/driver/nvidia/params | grep RestrictProfiling`. Without this, `SetProfiling(True)` runs but counter values come back zero or missing.

## profiling_status values

`ktt_profile` returns:
- `"ok"` — counters were collected.
- `"no_profiling_data"` — no `ComputationResult` had profiling data attached. Almost always means the build flag is missing or CUPTI is denied.
- `"no_counters_returned"` — profiling ran, but the requested counter names didn't exist on this device/build. Try a smaller set or check the names match KTT's NVPerf naming.

## How counter collection works

KTT typically needs MULTIPLE Run() invocations of the same config to gather every counter — each pass collects a subset. `ktt_profile` loops on `HasRemainingProfilingRuns()` (capped at 100) and reports `profiling_iterations` so you can see how many passes were needed.
"""

BEST_PRACTICES_DOC = """\
# KTT MCP — best practices

## Kernel signature

- `vectors` are passed as positional kernel arguments in spec order. The kernel
  function signature must contain exactly those pointers (typed `const T*` for
  read, `T*` for write/readwrite).
- `scalars` and tuning `parameters` are NOT kernel arguments — they're
  `-D<name>=<value>` preprocessor defines. Use them as bare identifiers inside
  the kernel (`if (i < N)`, `#if BLOCK_X >= 64`). Do not add them to the
  function parameter list.
- The reference kernel (`reference.kind=kernel`) is invoked with the SAME
  argument list as the tuned kernel — its signature must match.

Example for a vector-add spec with scalars=[N], vectors=[a, b, c]:

```cuda
extern "C" __global__ void vectorAdd(const float* a, const float* b, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}
```

(`N` and `BLOCK_X` are visible as preprocessor constants — no kernel parameters for them.)

## CPU autotuning (compute_api: cpp)

KTT can also tune host C/C++ kernels (JIT-compiled with g++). The kernel signature is FIXED by KTT — different from the CUDA case:

```cpp
extern "C" void mykernel(void** buffers, size_t* sizes) {
    const float* a = (const float*)buffers[0];
    const float* b = (const float*)buffers[1];
    float* c       = (float*)buffers[2];
    // N is still available as a preprocessor define
    for (size_t i = 0; i < N; ++i) c[i] = a[i] + b[i];
}
```

- Vectors arrive through `buffers[i]` cast to your dtype, in the order declared.
- `sizes[i]` is the byte length of vector `i`.
- Scalars and tuning parameters are still `-D` defines (use as bare identifiers).
- Pass `-O3 -fopenmp -march=native` etc. via `spec.compiler_options`.
- Set `searcher`, `stop`, `parameters`, and `constraints` exactly like the GPU path.
- This path is supported by KTT and the spec validates it, but the MCP test suite has not yet exercised it end-to-end. Use with the understanding it's preview-quality.

## Other tips

- Always set a `validate=true` vector and a reference. Without it the tuner can pick a fast-but-wrong config.
- Make `stop.value` match the search space size. Run ktt_search_space_size first.
- Use `launch_config` (a custom launcher) when grid/block depend on multiple parameters.
- Constraints prune the space cheaply. Use them to cut block/warp/vector incompatibilities before runtime.
- For MCMC, set `searcher.options.seed` when you want reproducibility.
- For tensor-core kernels, mark vectors with the right alignment dtype (`half`, `float`).
- Wrap `__global__` definitions in `extern "C"` to avoid C++ name mangling — KTT looks up the function by exact name.
"""
