"""Device discovery — list available CUDA/OpenCL platforms and devices."""

from __future__ import annotations

from typing import Any

from ktt_mcp.runtime.pyktt_loader import load_pyktt


_API_NAME = {"cuda": "CUDA", "opencl": "OpenCL", "cpp": "Cpp", "vulkan": "Vulkan"}


def _api_enum(ktt, api: str):
    return getattr(ktt.ComputeApi, _API_NAME[api])


def list_devices(compute_api: str = "cuda") -> dict[str, Any]:
    try:
        ktt = load_pyktt()
    except RuntimeError as e:
        return {"success": False, "stage": "io", "message": str(e)}
    try:
        # Tuner with platform=0/device=0 just to query info (cheap, no kernel).
        tuner = ktt.Tuner(0, 0, _api_enum(ktt, compute_api))
        out: list[dict[str, Any]] = []
        for plat in tuner.GetPlatformInfo():
            plat_idx = plat.GetIndex()
            for dev in tuner.GetDeviceInfo(plat_idx):
                out.append({
                    "platform_index": plat_idx,
                    "platform_name": plat.GetName(),
                    "device_index": dev.GetIndex(),
                    "device_name": dev.GetName(),
                })
        return {"success": True, "compute_api": compute_api, "devices": out}
    except Exception as e:
        return {"success": False, "stage": "io", "message": f"pyktt error: {e}"}


def describe_device(compute_api: str = "cuda", platform: int = 0, device: int = 0) -> dict[str, Any]:
    try:
        ktt = load_pyktt()
    except RuntimeError as e:
        return {"success": False, "stage": "io", "message": str(e)}
    try:
        tuner = ktt.Tuner(platform, device, _api_enum(ktt, compute_api))
        platforms = list(tuner.GetPlatformInfo())
        if platform >= len(platforms):
            return {"success": False, "stage": "io", "message": f"platform {platform} out of range"}
        devices = list(tuner.GetDeviceInfo(platform))
        if device >= len(devices):
            return {"success": False, "stage": "io", "message": f"device {device} out of range"}
        d = devices[device]
        info = {
            "name": d.GetName(),
            "index": d.GetIndex(),
            "platform_index": platform,
            "compute_api": compute_api,
            # the following are best-effort; not all KTT builds expose every getter
        }
        for attr, label in [
            ("GetVendor", "vendor"),
            ("GetDeviceType", "device_type"),
            ("GetGlobalMemorySize", "global_memory_bytes"),
            ("GetLocalMemorySize", "local_memory_bytes"),
            ("GetMaxConstantBufferSize", "max_constant_buffer_bytes"),
            ("GetMaxWorkGroupSize", "max_work_group_size"),
            ("GetMaxComputeUnits", "max_compute_units"),
        ]:
            getter = getattr(d, attr, None)
            if getter is not None:
                try:
                    info[label] = getter()
                except Exception:
                    pass
        return {"success": True, "device": info}
    except Exception as e:
        return {"success": False, "stage": "io", "message": f"pyktt error: {e}"}
