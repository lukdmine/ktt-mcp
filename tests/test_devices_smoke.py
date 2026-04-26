from __future__ import annotations

import pytest

from ktt_mcp.tools.devices import describe_device, list_devices


@pytest.mark.gpu
def test_list_devices_returns_at_least_one():
    res = list_devices("cuda")
    assert res["success"] is True
    assert len(res["devices"]) >= 1
    assert "device_name" in res["devices"][0]


@pytest.mark.gpu
def test_describe_device_zero():
    res = describe_device("cuda", 0, 0)
    assert res["success"] is True
    assert res["device"]["name"]
