#!/usr/bin/env python3
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "system_simulator/scripts/build_m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source.py"
SPEC = importlib.util.spec_from_file_location("m1543", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def rejects(function):
    try:
        function()
    except (M.M1543Error, M.M.M1539Error):
        return
    raise AssertionError("attack accepted")


def main():
    attacks = []
    description = M.describe()
    assert description["pilot"] == {"call_ordinal": 0, "sample_id": 10,
        "module_ordinal": 0, "timesteps": 10, "execution": False}
    assert description["streaming"]["materialized_transaction_list"] is False
    assert description["streaming"]["preserve_bank_calendars"] is True
    assert description["streaming"]["preserve_weight_cache"] is True
    assert description["source_capabilities"]["pilot_cli"] is False
    assert description["source_capabilities"]["production_cli"] is False
    assert description["source_capabilities"]["external_plane_parameter"] is False
    assert description["source_capabilities"]["opened_fd_hash_binding"] is True
    assert description["source_capabilities"]["closure_row_snapshot"] is True
    assert description["source_capabilities"]["mutable_file_backing_during_schedule"] is False
    assert description["streaming"]["immutable_compact_plane_snapshot"] is True
    assert not hasattr(M, "stream_tensor")

    authority = M.validate_authorities(False)
    assert authority["pilot_execution"] is False
    assert authority["production"] is False
    assert authority["m1542"]["members"] == 4
    row = M.selected_pilot_record()
    assert row["global_call_ordinal"] == 0
    assert row["global_sample_id"] == 10
    assert row["module_ordinal"] == 0

    synthetic = M.synthetic_self_test()
    assert synthetic["status"] == (
        "PASS_M1556_IMMUTABLE_SNAPSHOT_STREAMING_SOURCE_SYNTHETIC_TEST__NO_PILOT_NO_PRODUCTION")
    assert synthetic["pilot_execution"] is False
    assert synthetic["production"] is False
    assert synthetic["product_capture"] is False
    results = synthetic["results"]
    assert [item["configuration"] for item in results] == list(M.M.CONFIGS)
    assert results[0]["kind_counts"]["compute"] > results[2]["kind_counts"]["compute"]
    assert results[1]["kind_counts"]["compute"] == results[2]["kind_counts"]["compute"]
    assert len(set(item["commit_sequence_sha256"] for item in results)) == 1
    assert all(item["streaming"]["peak_rss_kib"] < 8 * 1024 * 1024
               for item in results)

    rejects(lambda: M.M.validate_config(M.FORBIDDEN_CONFIG)); attacks.append("product")
    rejects(lambda: M.M.validate_config("OFFICIAL_M700")); attacks.append("external")
    rejects(lambda: M.pilot_release()); attacks.append("pilot_launch")
    rejects(lambda: M.production_release()); attacks.append("production")

    bad = deepcopy(row)
    bad["global_call_ordinal"] = 1
    manifest = M.M.strict_json(M.M.M1521_MANIFEST)
    original = manifest["records"][0]
    manifest["records"][0] = bad
    rejects(lambda: M.M.validate_population_manifest(manifest)); attacks.append("call")
    manifest["records"][0] = original

    with tempfile.TemporaryDirectory(prefix="m1543_test.") as directory:
        path = Path(directory) / "short.bitpack"
        path.write_bytes(b"\x01")
        rejects(lambda: M.ImmutableLittleBitPlane(path, (10, 1, 8, 2, 2)))
        attacks.append("payload_size")

        path = Path(directory) / "plane.bitpack"
        path.write_bytes(bytes((10 * 8 * 2 * 2 + 7) // 8))
        with M.ImmutableLittleBitPlane(path, (10, 1, 8, 2, 2)) as plane:
            rejects(lambda: plane.bit(10, 0, 0, 0)); attacks.append("timestep")
            rejects(lambda: plane.bit(0, 8, 0, 0)); attacks.append("channel")
            rejects(lambda: plane.bit(0, 0, 2, 0)); attacks.append("coordinate")

        original = bytes((10 * 8 * 2 * 2 + 7) // 8)
        replacement = bytes([0xff]) * len(original)
        bound_path = Path(directory) / "fd_bound.bitpack"
        moved_path = Path(directory) / "fd_bound.original"
        bound_path.write_bytes(original)
        with M.ImmutableLittleBitPlane(bound_path, (10, 1, 8, 2, 2)) as plane:
            original_sha = plane.opened_sha256
            bound_path.rename(moved_path)
            bound_path.write_bytes(replacement)
            assert plane.opened_sha256 == original_sha
            assert plane.bit(0, 0, 0, 0) == 0
            assert M.sha256(bound_path) != plane.opened_sha256
            attacks.append("opened_fd_binding")

        inplace = Path(directory) / "inplace.bitpack"
        inplace.write_bytes(original)
        with M.ImmutableLittleBitPlane(inplace, (10, 1, 8, 2, 2)) as plane:
            with inplace.open("r+b") as mutable:
                mutable.seek(0); mutable.write(replacement); mutable.flush()
            assert plane.opened_sha256 == original_sha
            assert plane.bit(0, 0, 0, 0) == 0
            assert M.sha256(inplace) != plane.opened_sha256
            attacks.append("inplace_inode_mutation_isolated")

        assert list(__import__("inspect").signature(
            M.stream_actual_call).parameters) == ["config"]
        attacks.append("no_plane_module_call_seam")

    scheduler = M.StreamingCallScheduler("BIT_TYPED_K8")
    unresolved = M.M.request("bad", "BIT_TYPED_K8", "compute", [0], [0],
                             288, ["absent"], "done")
    rejects(lambda: scheduler.one(unresolved)); attacks.append("dependency")
    rejects(lambda: scheduler.retire_destination(("absent",)))
    attacks.append("retire")

    scheduler = M.StreamingCallScheduler("BIT_TYPED_K8")
    bad_bank = M.M.request("bank", "BIT_TYPED_K8", "weight_read", [0], [8], 96)
    rejects(lambda: scheduler.one(bad_bank)); attacks.append("bank")

    scheduler = M.StreamingCallScheduler("BIT_TYPED_K8")
    bad_psum = M.M.request("psum", "BIT_TYPED_K8", "psum_read",
                           [221184], [0], 48)
    rejects(lambda: scheduler.one(bad_psum)); attacks.append("capacity")

    assert len(attacks) == 16
    print("PASS M1556 source tests attacks=16 configs=3 immutable_snapshot=1 pilot=0 production=0 product=0")


if __name__ == "__main__":
    main()
