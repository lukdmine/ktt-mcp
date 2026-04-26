"""Compile a CPU C/C++ reference function and return a KTT-compatible callback."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np


_CTYPE_BY_DTYPE = {
    "int8": ctypes.c_int8, "int16": ctypes.c_int16,
    "int32": ctypes.c_int32, "int64": ctypes.c_int64,
    "uint8": ctypes.c_uint8, "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32, "uint64": ctypes.c_uint64,
    "float": ctypes.c_float, "double": ctypes.c_double,
}
_NUMPY_BY_DTYPE = {
    "int8": np.int8, "int16": np.int16, "int32": np.int32, "int64": np.int64,
    "uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32, "uint64": np.uint64,
    "float": np.float32, "double": np.float64,
}


def _build_scalar_defines(scalars: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    for s in scalars:
        name, dtype, value = s["name"], s["dtype"], s["value"]
        if dtype == "float":
            v = f"{float(value):.10g}"
            if "." not in v and "e" not in v.lower():
                v += ".0"
            flags.append(f"-D{name}={v}f")
        elif dtype == "double":
            v = f"{float(value):.10g}"
            if "." not in v and "e" not in v.lower():
                v += ".0"
            flags.append(f"-D{name}={v}")
        else:
            flags.append(f"-D{name}={int(value)}")
    return flags


def _compile_reference(source: Path, build_dir: Path, scalar_defines: list[str]) -> Path:
    out = build_dir / f"{source.stem}_ref.so"
    compiler = shutil.which("g++") or shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("Neither g++ nor gcc found on PATH; cannot compile CPU reference.")
    cmd = [compiler, "-O3", "-shared", "-fPIC", *scalar_defines, str(source), "-o", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CPU reference compile failed:\n{result.stderr}")
    return out


def build_cpu_reference_callback(
    *,
    source_path: Path,
    function_name: str,
    vectors: list[dict[str, Any]],
    vector_data: list[np.ndarray],
    scalars: list[dict[str, Any]],
    validated_vector_index: int,
    build_dir: Path,
) -> tuple[Callable[[Any], None], int]:
    so = _compile_reference(source_path, build_dir, _build_scalar_defines(scalars))
    lib = ctypes.CDLL(str(so))
    if not hasattr(lib, function_name):
        raise RuntimeError(
            f"Reference function {function_name!r} not found in {so}. "
            f"Wrap C++ definitions in extern \"C\" to avoid name mangling."
        )
    cpu_fn = getattr(lib, function_name)
    cpu_fn.argtypes = [ctypes.POINTER(_CTYPE_BY_DTYPE[v["dtype"]]) for v in vectors]
    cpu_fn.restype = None

    target_vec = vectors[validated_vector_index]
    target_dtype = np.dtype(_NUMPY_BY_DTYPE[target_vec["dtype"]])
    target_size = int(target_vec["size"])
    reference_size_bytes = target_size * target_dtype.itemsize

    def callback(buffer_view: Any) -> None:
        out_array = np.frombuffer(buffer_view, dtype=target_dtype, count=target_size)
        ptrs: list[Any] = []
        scratch: list[np.ndarray] = []
        for idx, vec in enumerate(vectors):
            ptr_t = ctypes.POINTER(_CTYPE_BY_DTYPE[vec["dtype"]])
            if idx == validated_vector_index:
                ptrs.append(out_array.ctypes.data_as(ptr_t))
            elif vec["access"] == "read":
                ptrs.append(vector_data[idx].ctypes.data_as(ptr_t))
            else:
                tmp = np.zeros(int(vec["size"]), dtype=_NUMPY_BY_DTYPE[vec["dtype"]])
                scratch.append(tmp)
                ptrs.append(tmp.ctypes.data_as(ptr_t))
        cpu_fn(*ptrs)

    return callback, reference_size_bytes
