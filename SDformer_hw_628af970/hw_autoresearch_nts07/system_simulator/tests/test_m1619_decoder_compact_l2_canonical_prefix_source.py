#!/usr/bin/env python3
from __future__ import print_function

import importlib.util
import json
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "scripts" /
          "build_m1619_decoder_compact_l2_canonical_prefix_source.py")


def load_source():
    spec = importlib.util.spec_from_file_location("m1619_source", str(SOURCE))
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
    assert description["status"].startswith("SOURCE_ONLY__L2")
    assert description["canonical_prefix"]["destination_count"] == 42
    assert description["canonical_prefix"]["last_destination_inclusive"] == 41
    assert description["canonical_prefix"]["state_reset_per_destination"] is False
    assert description["canonical_prefix"]["expected_commits_per_configuration"] == 168
    assert description["authorization"]["actual_payload"] is False
    assert description["authorization"]["l2_execution"] is False
    assert description["authorization"]["l3"] is False

    result = module.static_self_test()
    assert result["status"].startswith("PASS_M1619_L2_SOURCE_INTERFACE")
    assert result["geometry"]["parity_mask"] == 15
    assert result["geometry"]["corner"] is True
    assert result["geometry"]["edge"] is True
    assert result["geometry"]["interior"] is True
    assert result["dense_cache_history"]["hits"] > 0
    assert result["dense_cache_history"]["misses"] > 0
    assert result["dense_cache_history"]["evictions"] > 0
    assert result["dense_cache_history"]["final_entries"] == 9
    assert [row["configuration"] for row in result["synthetic_sessions"]] == list(module.CONFIGS)
    assert all(row["destinations"] == 42 and row["commits"] == 168
               for row in result["synthetic_sessions"])
    assert result["actual_payload"] is False
    assert result["l2_executed"] is False
    assert result["l3_executed"] is False

    rejects(module.M1619Error, lambda: module.actual_prefix_release())
    rejects(module.M1619Error,
            lambda: module.CanonicalPrefixMiter(module.FORBIDDEN_CONFIG))

    # Request mismatch must fail at the first differing exact field.
    reference = module.synthetic_request(module.CONFIGS[0], 0)
    compact = json.loads(json.dumps(reference))
    compact["issue_cycle"] += 1
    rejects(module.M1619Error,
            lambda: module.CanonicalPrefixMiter(
                module.CONFIGS[0]).accept_request_pair(reference, compact))

    # A skipped destination is rejected even when the two simulators agree.
    miter = module.CanonicalPrefixMiter(module.CONFIGS[0])
    for ordinal in range(module.OUTPUT_BLOCKS):
        request = module.synthetic_request(module.CONFIGS[0], ordinal)
        miter.accept_request_pair(request, dict(request))
    skipped = module.synthetic_state(module.CONFIGS[0], 1,
                                     module.OUTPUT_BLOCKS, True)
    rejects(module.M1619Error,
            lambda: miter.accept_destination_pair(skipped, dict(skipped)))

    # A per-destination reset is rejected by cumulative request/reset checks.
    miter = module.CanonicalPrefixMiter(module.CONFIGS[0])
    for ordinal in range(module.OUTPUT_BLOCKS):
        request = module.synthetic_request(module.CONFIGS[0], ordinal)
        miter.accept_request_pair(request, dict(request))
    reset = module.synthetic_state(module.CONFIGS[0], 0,
                                   module.OUTPUT_BLOCKS, True)
    reset["reset_count"] = 1
    rejects(module.M1619Error,
            lambda: miter.accept_destination_pair(reset, dict(reset)))

    source_text = SOURCE.read_text(encoding="utf-8")
    for forbidden in ("np.load", "numpy.load", "torch.load", ".npz",
                      ".tar.zst", "m1458_m1434_motion_ep34_live93_unified_hardware_capture"):
        assert forbidden not in source_text
    print("PASS M1619 source-only L2 interface destinations=42 commits=168 "
          "cache_history=1 attacks=4 actual_payload=0 L2exec=0 L3=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
