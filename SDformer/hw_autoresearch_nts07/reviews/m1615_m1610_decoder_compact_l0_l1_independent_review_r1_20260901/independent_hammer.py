#!/usr/bin/env python3
"""Independent, synthetic-only M1610 L0/L1 hammer.

This program intentionally does not know an ep34 payload path.  It binds the
reviewed source/test/contract by SHA, imports only the frozen M1539 reference
through M1610, and exercises static state bounds plus synthetic exact miters.
"""
from __future__ import print_function

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import stat
import sys
import textwrap


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/build_m1610_decoder_compact_cycle_simulator_source.py"
TEST = HW / "system_simulator/tests/test_m1610_decoder_compact_cycle_simulator_source.py"
CONTRACT = HW / "contracts/m1610_decoder_compact_cycle_simulator_source_contract_r1_20260901.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "73d4bade27612a3dfcbdc3e7417d7180397629a5be1f9e23587a58ea487b84ce",
    "test": "64abed164836fd94d4d6aebec31a32063dc3f73d94db1f9fd7aa73932d3aeeff",
    "contract": "5839c41d52af7b7b0fab9430f345bc6bb668099435ef330960564c9923a7896d",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    mode = Path(path).lstat().st_mode
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),
            label + " is not a regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def load_source():
    spec = importlib.util.spec_from_file_location("m1615_bound_m1610", str(SOURCE))
    require(spec is not None and spec.loader is not None, "cannot import M1610")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs)


def static_hot_path_hammer(module):
    functions = (
        module.validate_bank, module.port_index_for, module.queue_base_for,
        module.queue_capacity_for, module.service_bytes_for, module.latency_for,
        module.CompactScheduler.begin_addresses,
        module.CompactScheduler.push_address,
        module.CompactScheduler._count_index,
        module.CompactScheduler._active_compact,
        module.CompactScheduler.schedule_loaded,
        module.NumericWeightTileCache.begin_group,
        module.NumericWeightTileCache.push_key,
        module.NumericWeightTileCache._equal_loaded,
        module.NumericWeightTileCache._slot_matches_unique,
        module.NumericWeightTileCache._slot_is_pinned,
        module.NumericWeightTileCache._slot_key_less,
        module.NumericWeightTileCache.prepare_loaded,
        module.NumericWeightTileCache.slot_for,
    )
    forbidden_attr = set(("append", "extend", "insert", "pop", "add",
                          "dumps", "format"))
    forbidden_name = set(("dict", "set", "hash", "pickle"))
    checked = []
    for function in functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            require(not isinstance(node, (ast.Dict, ast.Set, ast.ListComp,
                                          ast.SetComp, ast.DictComp,
                                          ast.GeneratorExp)),
                    "dynamic container in compact hot path: " + function.__name__)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    require(node.func.id not in forbidden_name,
                            "forbidden hot-path call: " + node.func.id)
                if isinstance(node.func, ast.Attribute):
                    require(node.func.attr not in forbidden_attr,
                            "forbidden hot-path attribute: " + node.func.attr)
        checked.append(function.__name__)
    source_text = SOURCE.read_text(encoding="utf-8")
    for token in ("numpy.load", "np.load", "torch.load", ".npz", ".tar.zst",
                  "m1458_m1434_motion_ep34_live93_unified_hardware_capture"):
        require(token not in source_text, "actual-payload access token present: " + token)
    return checked


def pattern_bits(module, mask):
    active = []
    for flat in range(32):
        if (mask >> flat) & 1:
            channel = flat // 4
            rem = flat % 4
            active.append((channel, rem // 2, rem % 2))
    return module.make_bits(active)


def synthetic_miter_hammer(module):
    # Sixteen deterministic masks cover empty, singleton, bank-complete,
    # spatial boundaries, alternating and dense populations without payload.
    masks = (
        0x00000000, 0x00000001, 0x80000000, 0x00011111,
        0x11111111, 0x01010101, 0x80808080, 0xaaaaaaaa,
        0x55555555, 0x00ff00ff, 0xff00ff00, 0x0f0f0f0f,
        0xf0f0f0f0, 0x13579bdf, 0x2468ace0, 0xffffffff,
    )
    rows = []
    for case_index, mask in enumerate(masks):
        bits = pattern_bits(module, mask)
        case_rows = []
        for config in module.CONFIGS:
            value = module.miter_rows(
                config, module.M.synthetic_config_transactions(config, bits),
                "independent_mask_{:02d}".format(case_index))
            case_rows.append(value)
            rows.append(value)
        require(len(set(row["packed_commit_sequence_sha256"]
                        for row in case_rows)) == 1,
                "cross-configuration packed commit stream drift")
        require(case_rows[1]["kind_counts"].get("compute", 0) ==
                case_rows[2]["kind_counts"].get("compute", 0),
                "BIT equal-service/typed compute population drift")
        require(case_rows[1]["kind_counts"].get("commit", 0) ==
                case_rows[2]["kind_counts"].get("commit", 0),
                "BIT equal-service/typed commit population drift")
    pressure = module.miter_rows(
        module.CONFIGS[2], module.manual_port_pressure_rows(module.CONFIGS[2]),
        "independent_outstanding_1rw_pressure")
    require(pressure["max_active_outstanding_per_bank"] == 16,
            "external outstanding queue did not reach its fixed capacity")
    require(pressure["outstanding_full_waits"] > 0,
            "outstanding-full wait not covered")
    require(pressure["shared_1rw_serializations"] > 0,
            "shared-1RW serialization not covered")
    cache = module.cache_miter()
    require(cache["evictions"] > 0 and cache["final_entries"] == 9,
            "nine-entry cache eviction not covered")
    return {"deterministic_masks": len(masks), "miter_rows": len(rows),
            "total_requests": sum(row["requests"] for row in rows),
            "pressure": pressure, "cache": cache}


def attacks(module):
    count = 0
    for action in (
        lambda: module.CompactScheduler(module.FORBIDDEN_CONFIG),
        lambda: module.production_release(),
    ):
        try:
            action()
        except module.M1610Error:
            count += 1
        else:
            raise AssertionError("forbidden action accepted")
    scheduler = module.CompactScheduler(module.CONFIGS[0])
    for index in range(8):
        scheduler.push_address(index, index)
    try:
        scheduler.push_address(8, 8)
    except module.M1610Error:
        count += 1
    else:
        raise AssertionError("ninth address-bank scratch entry accepted")
    return count


def main():
    regular_exact(SOURCE, EXPECTED["source"], "M1610 source")
    regular_exact(TEST, EXPECTED["test"], "M1610 test")
    regular_exact(CONTRACT, EXPECTED["contract"], "M1610 contract")
    regular_exact(DOCS359, EXPECTED["docs359"], "docs359")
    contract = strict_json(CONTRACT)
    require(contract["status"] ==
            "SOURCE_ONLY__L0_L1_EXACT_MITER__NO_ACTUAL_PAYLOAD_NO_EXECUTION",
            "contract status drift")
    require(contract["fixed_state"] == {
        "next_port_entries": 24,
        "outstanding_return_slots": 129,
        "weight_cache_entries": 9,
        "address_bank_scratch_entries": 8,
        "dependency": "direct numeric ready cycle",
        "packed_address_digest": "versioned fixed-width big-endian"},
        "contract fixed-state drift")
    require(contract["claim_boundary"]["l2"] is False and
            contract["claim_boundary"]["l3"] is False and
            contract["claim_boundary"]["actual_payload"] is False and
            contract["claim_boundary"]["paper_result"] is False,
            "contract opens a forbidden claim")
    module = load_source()
    description = module.describe()
    require(description["implemented_miter_levels"] == ["L0", "L1"] and
            description["missing_miter_levels"] == ["L2", "L3"],
            "source miter-level boundary drift")
    require(description["representation"]["next_port_entries"] == 24 and
            description["representation"]["outstanding_slots"] == 129 and
            description["representation"]["weight_cache_entries"] == 9 and
            description["representation"]["address_scratch_entries"] == 8,
            "source fixed-state description drift")
    require(all(description["claim_boundary"][key] is False for key in
                ("actual_payload", "execution", "cycles", "traffic",
                 "speedup", "energy", "rtl", "eda", "ppa", "table_a",
                 "paper_result")), "source opens a forbidden claim")
    l0 = module.validate_l0()
    require(l0["actual_payload"] is False and l0["execution"] is False,
            "L0 crossed actual-payload boundary")
    built_in = module.synthetic_self_test()
    require(built_in["actual_payload"] is False and
            built_in["l2_actual_prefix"] is False and
            built_in["l3_full_diagnostic"] is False and
            built_in["pilot"] is False and built_in["production"] is False and
            built_in["paper_result"] is False,
            "built-in test crossed source-only boundary")
    output = {
        "schema": "m1615_m1610_decoder_compact_l0_l1_independent_hammer_r1_v1",
        "status": "PASS_M1610_L0_L1_SOURCE_ONLY__L2_L3_STILL_CLOSED",
        "python": sys.version,
        "input_sha256": EXPECTED,
        "fixed_state": description["representation"],
        "hot_path_functions_checked": static_hot_path_hammer(module),
        "built_in_l1_rows": len(built_in["l1"]["rows"]),
        "independent_synthetic": synthetic_miter_hammer(module),
        "attacks_rejected": attacks(module),
        "actual_payload_opened": False,
        "l2_executed": False,
        "l3_executed": False,
        "eda_gpu_executed": False,
        "paper_result": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
