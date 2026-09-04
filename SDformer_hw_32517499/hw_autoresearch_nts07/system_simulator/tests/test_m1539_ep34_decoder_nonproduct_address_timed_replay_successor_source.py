#!/usr/bin/env python3
from copy import deepcopy
import importlib.util
import json
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "system_simulator/scripts/build_m1539_ep34_decoder_nonproduct_address_timed_replay_successor_source.py"
SPEC = importlib.util.spec_from_file_location("m1539", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def rejects(function):
    try:
        function()
    except M.M1539Error:
        return
    raise AssertionError("attack accepted")


def main():
    attacks = []
    description = M.describe()
    assert description["configurations"] == list(M.CONFIGS)
    assert M.FORBIDDEN_CONFIG not in description["configurations"]
    assert description["source_capabilities"]["production_launch"] is False
    assert sum(M.COMMON_RESOURCE["partitions"].values()) == 245760

    authority = M.validate_authorities(False)
    assert authority["full_payload_verification"] is False
    assert authority["m1521"]["members"] == 122
    assert authority["resource_manifest_sha256"] == M.validate_resource()

    synthetic = M.synthetic_self_test()
    rows = synthetic["results"]
    assert [row["configuration"] for row in rows] == list(M.CONFIGS)
    assert rows[0]["kind_counts"]["compute"] > rows[2]["kind_counts"]["compute"]
    assert rows[1]["kind_counts"]["compute"] == rows[2]["kind_counts"]["compute"]
    assert rows[1]["byte_counts"]["external_read"] > rows[2]["byte_counts"]["external_read"]
    assert len(set(row["commit_sequence_sha256"] for row in rows)) == 1

    manifest = M.strict_json(M.M1521_MANIFEST)
    population = M.validate_population_manifest(manifest)
    comparator = []
    for row in rows:
        comparator.append({"configuration": row["configuration"],
            "resource_manifest_sha256": row["resource_manifest_sha256"],
            "commit_sequence_sha256": row["commit_sequence_sha256"],
            "checkpoint_sha256": M.CHECKPOINT_SHA256,
            "population_manifest_sha256": population[
                "population_projection_sha256"]})
    assert M.compare_rows(comparator)

    rejects(lambda: M.validate_config(M.FORBIDDEN_CONFIG)); attacks.append("product")
    rejects(lambda: M.validate_config("OFFICIAL_M700")); attacks.append("external")
    rejects(lambda: M.production_release()); attacks.append("production")

    bad = deepcopy(manifest)
    bad["capture"]["checkpoint_sha256"] = "0" * 64
    rejects(lambda: M.validate_population_manifest(bad)); attacks.append("checkpoint")
    bad = deepcopy(manifest)
    bad["records"][1]["layer_scale_word_uint32"] = 0x3F7FFFB3
    rejects(lambda: M.validate_population_manifest(bad)); attacks.append("old_d1")
    bad = deepcopy(manifest)
    bad["records"][0]["coerced"] = True
    rejects(lambda: M.validate_population_manifest(bad)); attacks.append("coercion")
    bad = deepcopy(manifest)
    bad["records"][0], bad["records"][1] = bad["records"][1], bad["records"][0]
    rejects(lambda: M.validate_population_manifest(bad)); attacks.append("reorder")

    bad_rows = deepcopy(comparator)
    bad_rows[2]["resource_manifest_sha256"] = "f" * 64
    rejects(lambda: M.compare_rows(bad_rows)); attacks.append("resource")
    bad_rows = deepcopy(comparator)
    bad_rows[0]["commit_sequence_sha256"] = "e" * 64
    rejects(lambda: M.compare_rows(bad_rows)); attacks.append("commit")

    scheduler = M.AddressTimedScheduler(M.CONFIGS[2])
    unresolved = M.request("missing", M.CONFIGS[2], "compute", [0], [0], 288,
                           ["absent"], "done")
    rejects(lambda: scheduler.schedule_one(unresolved)); attacks.append("dependency")
    scheduler = M.AddressTimedScheduler(M.CONFIGS[2])
    bad_bank = M.request("bank", M.CONFIGS[2], "weight_read", [0], [8], 16)
    rejects(lambda: scheduler.schedule_one(bad_bank)); attacks.append("bank")
    scheduler = M.AddressTimedScheduler(M.CONFIGS[2])
    bad_psum = M.request("psum", M.CONFIGS[2], "psum_read",
                         [221184], [0], 48)
    rejects(lambda: scheduler.schedule_one(bad_psum)); attacks.append("capacity")
    assert len(attacks) == 12
    print("PASS M1539 source tests attacks=12 configs=3 product=0 production=0")


if __name__ == "__main__":
    main()
