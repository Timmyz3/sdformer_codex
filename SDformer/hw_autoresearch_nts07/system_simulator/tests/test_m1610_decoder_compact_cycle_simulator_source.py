#!/usr/bin/env python3
from __future__ import print_function

import importlib.util
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "scripts" /
          "build_m1610_decoder_compact_cycle_simulator_source.py")


def load_source():
    spec = importlib.util.spec_from_file_location("m1610_source", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejects(error_type, function):
    try:
        function()
    except error_type:
        return
    raise AssertionError("attack accepted")


def main():
    module = load_source()
    description = module.describe()
    assert description["implemented_miter_levels"] == ["L0", "L1"]
    assert description["missing_miter_levels"] == ["L2", "L3"]
    assert description["claim_boundary"]["actual_payload"] is False
    assert description["representation"]["hardware_resource_or_schedule_change"] is False

    result = module.synthetic_self_test()
    assert result["status"].startswith("PASS_M1610_L0_L1")
    assert result["l1"]["cycle_exact"] is True
    assert result["l1"]["count_exact"] is True
    assert result["l1"]["bytes_exact"] is True
    assert result["l1"]["commit_exact"] is True
    assert result["l1"]["address_exact"] is True
    assert len(result["l1"]["rows"]) == 12
    assert result["l1"]["pressure"]["outstanding_full_waits"] > 0
    assert result["l1"]["pressure"]["shared_1rw_serializations"] > 0
    assert result["l1"]["cache"]["evictions"] > 0
    assert result["actual_payload"] is False
    assert result["l2_actual_prefix"] is False
    assert result["l3_full_diagnostic"] is False

    rejects(module.M1610Error, lambda: module.CompactScheduler(
        module.FORBIDDEN_CONFIG))
    scheduler = module.CompactScheduler(module.CONFIGS[0])
    rejects(module.M1610Error, lambda: [scheduler.push_address(i, i)
                                        for i in range(9)])
    scheduler = module.CompactScheduler(module.CONFIGS[0])
    scheduler.push_address(221184, 0)
    rejects(module.M1610Error, lambda: scheduler.schedule_loaded(
        4, 48, 0, 0, 1, 3, 0, module.FLAG_PSUM_READ,
        0, 0, 0, 0, 0))
    cache = module.NumericWeightTileCache()
    for index in range(8):
        cache.push_key(3, 0, 0, index)
    rejects(module.M1610Error, lambda: cache.push_key(3, 0, 0, 8))
    scheduler = module.CompactScheduler(module.CONFIGS[0])
    scheduler.push_address(0, 0)
    scheduler.schedule_loaded(6, 288, 0, 0, 1, 3, 0,
                              module.FLAG_COMPUTE, 0, 0, 0, 0, 0)
    scheduler.push_address(0, 0)
    rejects(module.M1610Error, lambda: scheduler.schedule_loaded(
        6, 288, 0, 0, 1, 3, 0, module.FLAG_COMPUTE,
        0, 0, 0, 0, 0))
    rejects(module.M1610Error, lambda: module.production_release())
    print("PASS M1610 L0/L1 compact exact-miter tests rows=12 attacks=6 L2=0 L3=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
