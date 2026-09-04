import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as torch_functional


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / (
    "hw_autoresearch_nts07/system_simulator/scripts/"
    "map_m670_decoder_convtranspose_polyphase_workload_r2.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m670_decoder_convtranspose_polyphase_workload_mapper_r2_contract_20260828.json")


def load_target():
    spec = importlib.util.spec_from_file_location("m670_target", TARGET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M670 = load_target()


def write_bitpack(path, values):
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    path.write_bytes(np.packbits(flat, bitorder="little").tobytes())
    return path


def test_contract_binds_authored_and_frozen_inputs_without_m660_result_sha():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for group in ("authored_inputs", "frozen_predecessors"):
        for identity in contract[group].values():
            path = ROOT / identity["path"]
            assert path.is_file() and not path.is_symlink()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == \
                identity["sha256"]
    assert contract["m660_integration_boundary"][
        "unfinished_manifest_or_payload_sha256_bound"] is False
    assert contract["claim_boundary"]["cycles"] is False
    assert contract["claim_boundary"]["speedup"] is False


def test_m514_phase_tap_and_k_order_are_literal():
    assert M670.M514_PHASE_ORDER == (3, 2, 1, 0)
    assert M670.M514_SLOT_ORDER == (
        (0, 0), (0, 2), (2, 0), (2, 2),
        (0, 1), (2, 1), (1, 0), (1, 2), (1, 1))
    assert tuple(tap for bank in M670.M514_PHASE_ORDER
                 for tap in M670.M514_PHASE_TAPS[bank]) == \
        M670.M514_SLOT_ORDER
    plan = M670.build_phase_plan(3, 2, 2, 3)
    assert plan["k"] == 12
    assert plan["k_order"] == "M514_PHASE_TAP_THEN_SOURCE_CHANNEL"
    # Destination (1,1), tap (0,0) maps to source (1,1); channel is the
    # inner K dimension, so the first three indices are c*H*W + 3.
    assert plan["destination_y"][3] == 3
    assert plan["destination_x"][3] == 3
    assert plan["source_flat_index"][0, :3].tolist() == [3, 7, 11]
    # At destination (3,3), the first three taps are boundary padding and
    # only final tap (2,2) maps back to source (1,1).
    assert plan["source_flat_index"][3, :9].tolist() == [-1] * 9
    assert plan["source_flat_index"][3, 9:].tolist() == [3, 7, 11]


@pytest.mark.parametrize("seed", [665001, 665019, 665037])
def test_random_integer_polyphase_reconstructs_torch_exactly(tmp_path, seed):
    rng = np.random.default_rng(seed)
    shape = (2, 1, 3, 3, 4)
    activation = rng.integers(0, 2, size=shape, dtype=np.uint8)
    weight = rng.integers(-3, 4, size=(3, 5, 3, 3), dtype=np.int8)
    bitpack = write_bitpack(tmp_path / "activation.bitpack", activation)

    observed = M670.reconstruct_convtranspose(
        bitpack, shape, weight, tile_m=3, trusted_root=tmp_path)
    expected = torch_functional.conv_transpose2d(
        torch.from_numpy(activation[:, 0].astype(np.float64)),
        torch.from_numpy(weight.astype(np.float64)), bias=None,
        stride=(2, 2), padding=(1, 1), output_padding=(1, 1),
        groups=1, dilation=(1, 1)).numpy()
    assert observed.shape == expected.shape == (2, 5, 6, 8)
    assert np.array_equal(observed, expected)

    phases, metadata = M670.materialize_polyphase(
        bitpack, shape, tile_m=5, trusted_root=tmp_path)
    assert list(phases) == [3, 2, 1, 0]
    for bank, tap_count in ((3, 4), (2, 2), (1, 2), (0, 1)):
        assert phases[bank].shape == (2, 12, tap_count * 3)
        assert np.all((metadata[bank]["destination_y"] & 1) == (bank >> 1))
        assert np.all((metadata[bank]["destination_x"] & 1) == (bank & 1))


def test_stream_tiles_preserve_tmk_shape_and_m_partition(tmp_path):
    shape = (3, 1, 2, 2, 3)
    values = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape) & 1
    bitpack = write_bitpack(tmp_path / "x.bitpack", values)
    tiles = list(M670.iter_polyphase_tiles(
        bitpack, shape, tile_m=4, phases=(3,), trusted_root=tmp_path))
    assert [(row["m_start"], row["m_stop"]) for row in tiles] == [(0, 4),
                                                                    (4, 6)]
    assert [row["values"].shape for row in tiles] == [(3, 4, 8),
                                                       (3, 2, 8)]
    joined = np.concatenate([row["values"] for row in tiles], axis=1)
    full, _metadata = M670.materialize_polyphase(
        bitpack, shape, tile_m=99, phases=(3,), trusted_root=tmp_path)
    assert np.array_equal(joined, full[3])


def test_valid_tap_product_and_popcount_conservation(tmp_path):
    shape = (2, 1, 2, 2, 3)
    values = np.zeros(shape, dtype=np.uint8)
    values[0, 0, 0, 0, 0] = 1       # top-left: four valid taps
    values[0, 0, 1, 1, 1] = 1       # interior: nine valid taps
    values[1, 0, 0, 0, 2] = 1       # top row: six valid taps
    bitpack = write_bitpack(tmp_path / "account.bitpack", values)
    account = M670.workload_accounting(
        bitpack, shape, output_channels=7, tile_m=2,
        trusted_root=tmp_path)
    assert account["source_popcount"] == 3
    assert account["active_tap_events"] == 4 + 9 + 6
    assert account["active_products"] == (4 + 9 + 6) * 7
    # Per channel, a 2x3 source plane contains one corner (4), three edges
    # (6), and two interior-with-respect-to-top/left sites (9).
    assert account["valid_tap_slots_per_time"] == 2 * (4 + 3 * 6 + 2 * 9)
    assert account["dense_valid_products"] == (
        account["valid_tap_slots_all_time"] * 7)
    assert account["structural_padding_zero_entries_all_time"] > 0


def test_phase_weight_matrix_is_tap_then_channel():
    weight = np.arange(2 * 3 * 3 * 3, dtype=np.int64).reshape(2, 3, 3, 3)
    matrix = M670.phase_weight_matrix(weight, 3)
    expected = np.concatenate([
        weight[:, :, 0, 0], weight[:, :, 0, 2],
        weight[:, :, 2, 0], weight[:, :, 2, 2]], axis=0)
    assert np.array_equal(matrix, expected)
    # A channel-major permutation is deliberately not the admitted K order.
    channel_major = np.stack([
        weight[c, :, ky, kx] for c in range(2)
        for ky, kx in M670.M514_PHASE_TAPS[3]], axis=0)
    assert not np.array_equal(matrix, channel_major)


@pytest.mark.parametrize("field,value", [
    ("kernel_size", (2, 3)),
    ("stride", (1, 2)),
    ("padding", (0, 1)),
    ("output_padding", (0, 1)),
    ("dilation", (2, 1)),
    ("groups", 2),
])
def test_convtranspose_parameter_drift_is_rejected(field, value):
    kwargs = dict(M670.EXPECTED_SPEC)
    kwargs[field] = value
    with pytest.raises(RuntimeError, match="only exact"):
        M670.validate_convtranspose_spec(**kwargs)


def test_bit_order_shape_length_tail_and_k_order_drift_rejected(tmp_path):
    shape = (1, 1, 1, 1, 9)
    good = write_bitpack(tmp_path / "good.bitpack",
                         np.ones(shape, dtype=np.uint8))
    with pytest.raises(RuntimeError, match="non-little"):
        M670.validate_bitpack(
            good, shape, bit_order="big", trusted_root=tmp_path)
    with pytest.raises(RuntimeError, match="K-order drift"):
        M670.validate_bitpack(
            good, shape, k_order="CHANNEL_THEN_TAP", trusted_root=tmp_path)
    with pytest.raises(RuntimeError, match="batch dimension"):
        M670.validate_bitpack(
            good, (1, 2, 1, 1, 9), trusted_root=tmp_path)
    short = tmp_path / "short.bitpack"
    short.write_bytes(b"\x01")
    with pytest.raises(RuntimeError, match="byte length"):
        M670.validate_bitpack(short, shape, trusted_root=tmp_path)
    bad_tail = tmp_path / "bad_tail.bitpack"
    bad_tail.write_bytes(bytes([0xff, 0xff]))
    with pytest.raises(RuntimeError, match="tail padding"):
        M670.validate_bitpack(bad_tail, shape, trusted_root=tmp_path)


def make_m660_manifest(tmp_path, d1_scaled=True):
    shape = [1, 1, 1, 2, 4]
    payload = write_bitpack(tmp_path / "calls.bitpack",
                            np.asarray([0, 1, 0, 1, 1, 0, 0, 0],
                                       dtype=np.uint8))
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    d1_payload = write_bitpack(tmp_path / "d1.bitpack",
                               np.asarray([1, 0, 1, 0, 0, 1, 0, 0],
                                          dtype=np.uint8))
    d1_digest = hashlib.sha256(d1_payload.read_bytes()).hexdigest()
    binary_rows = []
    d1_rows = []
    for sample_id in range(10):
        for module_index in M670.M660_BINARY_MODULES:
            binary_rows.append({
                "sample_id": sample_id, "module_index": module_index,
                "name": M670.M660_MODULE_NAMES[module_index],
                "route": "EXACT_BINARY_BITPACK",
                "relative_path": "calls.bitpack", "input_shape": shape,
                "input": {"packed_bytes": 1, "packed_sha256": digest},
                "theta_binary_candidate": {
                    "packed_bytes": 1, "packed_sha256": "0" * 64},
            })
        d1_row = {
            "sample_id": sample_id, "module_index": 1,
            "name": M670.M660_MODULE_NAMES[1],
            "route": ("EXACT_SCALED_BINARY_BITPACK" if d1_scaled else
                      "COMMON_FP32_DENSE_FALLBACK"),
            "input_shape": shape,
            "input": {"packed_bytes": 7, "packed_sha256": "f" * 64},
        }
        if d1_scaled:
            d1_row.update({
                "relative_path": "d1.bitpack",
                "theta_binary_candidate": {
                    "packed_bytes": 1, "packed_sha256": d1_digest},
            })
        d1_rows.append(d1_row)
    return {
        "schema": "m660_h67_ep35_layer_static_decoder_payload_v1",
        "packing": {"values": [0, 1], "bit_order": "little",
                    "order": "C_ORDER_FLAT",
                    "whole_call_contiguous_copy_allowed": False},
        "d0_d2_d3_binary_records": binary_rows,
        "d1_records": d1_rows,
    }


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_m660_manifest_adapter_is_schema_route_name_and_lattice_bound(
        tmp_path):
    manifest = make_m660_manifest(tmp_path, d1_scaled=True)
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    records = M670.m660_bitpack_records(path, trusted_root=tmp_path)
    assert len(records) == 40
    assert [(row["sample_id"], row["module_index"])
            for row in records] == [
                (sample_id, module_index) for sample_id in range(10)
                for module_index in range(4)]
    assert all(row["name"] == M670.M660_MODULE_NAMES[
        row["module_index"]] for row in records)
    assert all(row["route"] == ("EXACT_SCALED_BINARY_BITPACK"
               if row["module_index"] == 1 else "EXACT_BINARY_BITPACK")
               for row in records)
    # The deliberately wrong D1 input identity is ignored; the admitted D1
    # route is bound exclusively to theta_binary_candidate.
    assert all(row["packed_sha256"] != "f" * 64
               for row in records if row["module_index"] == 1)

    fallback = make_m660_manifest(tmp_path, d1_scaled=False)
    write_manifest(path, fallback)
    fallback_records = M670.m660_bitpack_records(
        path, trusted_root=tmp_path)
    assert len(fallback_records) == 30
    assert all(row["module_index"] in (0, 2, 3)
               for row in fallback_records)

    manifest["packing"]["bit_order"] = "big"
    write_manifest(path, manifest)
    with pytest.raises(RuntimeError, match="packing contract drift"):
        M670.m660_bitpack_records(path, trusted_root=tmp_path)


@pytest.mark.parametrize("mutation,pattern", [
    ("binary_scaled_route", "container/module/route"),
    ("d1_binary_route", "D1 route"),
    ("d1_wrong_module", "D1 container/module"),
    ("wrong_name", "name/index"),
    ("missing_binary", "population is incomplete"),
    ("missing_d1", "containers missing"),
    ("duplicate_lattice", "duplicate or missing"),
    ("bool_sample", "non-boolean integer"),
    ("bool_module", "non-boolean integer"),
    ("bool_packed_bytes", "non-boolean integer"),
])
def test_m660_cross_route_duplicate_missing_and_bool_attacks_rejected(
        tmp_path, mutation, pattern):
    manifest = make_m660_manifest(tmp_path, d1_scaled=True)
    if mutation == "binary_scaled_route":
        manifest["d0_d2_d3_binary_records"][0]["route"] = \
            "EXACT_SCALED_BINARY_BITPACK"
    elif mutation == "d1_binary_route":
        manifest["d1_records"][0]["route"] = "EXACT_BINARY_BITPACK"
    elif mutation == "d1_wrong_module":
        manifest["d1_records"][0]["module_index"] = 0
        manifest["d1_records"][0]["name"] = M670.M660_MODULE_NAMES[0]
    elif mutation == "wrong_name":
        manifest["d0_d2_d3_binary_records"][0]["name"] = "wrong"
    elif mutation == "missing_binary":
        manifest["d0_d2_d3_binary_records"].pop()
    elif mutation == "missing_d1":
        manifest.pop("d1_records")
    elif mutation == "duplicate_lattice":
        manifest["d0_d2_d3_binary_records"][-1] = dict(
            manifest["d0_d2_d3_binary_records"][0])
    elif mutation == "bool_sample":
        manifest["d0_d2_d3_binary_records"][0]["sample_id"] = True
    elif mutation == "bool_module":
        manifest["d0_d2_d3_binary_records"][0]["module_index"] = True
    elif mutation == "bool_packed_bytes":
        manifest["d0_d2_d3_binary_records"][0]["input"][
            "packed_bytes"] = True
    path = write_manifest(tmp_path / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match=pattern):
        M670.m660_bitpack_records(path, trusted_root=tmp_path)


def test_m660_manifest_duplicate_json_key_is_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        M670.m660_bitpack_records(path, trusted_root=tmp_path)


def test_shape_product_uses_bounded_python_integers_and_rejects_aliases(
        tmp_path):
    source = TARGET.read_text(encoding="utf-8")
    assert "np.prod" not in source
    empty = tmp_path / "empty.bitpack"
    empty.write_bytes(b"")
    for shape in (
            (2 ** 32, 1, 2 ** 32, 1, 1),
            (2 ** 20, 1, 2 ** 20, 1, 1),
            (0, 1, 1, 1, 1),
            (True, 1, 1, 1, 1),
            (np.bool_(True), 1, 1, 1, 1)):
        with pytest.raises(RuntimeError):
            M670.validate_bitpack(empty, shape, trusted_root=tmp_path)


def test_all_selected_integer_interfaces_reject_bool(tmp_path):
    shape = (1, 1, 1, 1, 8)
    real = write_bitpack(tmp_path / "real.bitpack",
                         np.zeros(shape, dtype=np.uint8))
    attacks = [
        lambda: M670.phase_bank(True, 0),
        lambda: M670.build_phase_plan(True, 1, 1, 3),
        lambda: M670.build_phase_plan(1, 1, 1, np.bool_(True)),
        lambda: list(M670.iter_polyphase_tiles(
            real, shape, tile_m=True, trusted_root=tmp_path)),
        lambda: list(M670.iter_polyphase_tiles(
            real, shape, phases=(True,), trusted_root=tmp_path)),
        lambda: M670.workload_accounting(
            real, shape, output_channels=True, trusted_root=tmp_path),
        lambda: M670.phase_weight_matrix(
            np.ones((1, 1, 3, 3), dtype=np.int8), True),
        lambda: M670.validate_convtranspose_spec(groups=True),
        lambda: M670.validate_convtranspose_spec(kernel_size=(True, 3)),
    ]
    for attack in attacks:
        with pytest.raises(RuntimeError):
            attack()


def test_trusted_root_leaf_parent_manifest_and_escape_attacks_rejected(
        tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    nested = package / "nested"
    nested.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    real = write_bitpack(outside / "real.bitpack",
                         np.zeros((1, 1, 1, 1, 8), dtype=np.uint8))
    leaf_alias = package / "leaf.bitpack"
    leaf_alias.symlink_to(real)
    parent_alias = package / "alias"
    parent_alias.symlink_to(outside, target_is_directory=True)
    for candidate in (leaf_alias, parent_alias / "real.bitpack",
                      "../outside/real.bitpack", real):
        with pytest.raises(RuntimeError):
            M670.validate_bitpack(
                candidate, (1, 1, 1, 1, 8), trusted_root=package)
    with pytest.raises(RuntimeError, match="explicit trusted root"):
        M670.validate_bitpack(real, (1, 1, 1, 1, 8))

    manifest = write_manifest(
        package / "manifest.json", make_m660_manifest(package))
    manifest_alias = package / "manifest_alias.json"
    manifest_alias.symlink_to(manifest)
    with pytest.raises(RuntimeError, match="symlink component"):
        M670.m660_bitpack_records(manifest_alias, trusted_root=package)
    root_alias = tmp_path / "package_alias"
    root_alias.symlink_to(package, target_is_directory=True)
    with pytest.raises(RuntimeError, match="trusted root"):
        M670.m660_bitpack_records(manifest, trusted_root=root_alias)


def test_manifest_payload_parent_symlink_escape_is_rejected(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    manifest = make_m660_manifest(package)
    outside_payload = write_bitpack(
        outside / "x.bitpack", np.zeros((1, 1, 1, 2, 4), dtype=np.uint8))
    digest = hashlib.sha256(outside_payload.read_bytes()).hexdigest()
    alias = package / "escape"
    alias.symlink_to(outside, target_is_directory=True)
    for row in manifest["d0_d2_d3_binary_records"]:
        row["relative_path"] = "escape/x.bitpack"
        row["input"] = {"packed_bytes": 1, "packed_sha256": digest}
    path = write_manifest(package / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="symlink component"):
        M670.m660_bitpack_records(path, trusted_root=package)


def test_symlink_payload_is_rejected(tmp_path):
    real = write_bitpack(tmp_path / "real.bitpack",
                         np.zeros((1, 1, 1, 1, 8), dtype=np.uint8))
    alias = tmp_path / "alias.bitpack"
    alias.symlink_to(real)
    with pytest.raises(RuntimeError, match="symlink component"):
        M670.validate_bitpack(
            alias, (1, 1, 1, 1, 8), trusted_root=tmp_path)
