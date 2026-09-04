#!/usr/bin/env python3
"""M722 first-principles CPU fast-kill for decoder LB-FUSE.

The candidate is a K3/S2 source-order three-output-row accumulator.  The
baseline is the same 96-lane, 240-KiB, dense-commit A1-OSG coordinate and is
allowed the same source-order row lifetime and a legal D3 width stripe.  Thus
ordinary on-chip psum RMW traffic is never mislabeled as DRAM spill.

All numeric work runs on CPU.  Exact integer ConvTranspose and an independent
source/tap reconstruction are compared on the complete sealed M699 S3x10
payload.  Absolute-weight convolution gives an order-independent bound on
every partial sum, which is stronger than checking final Acc16 values alone.
"""

import argparse
from decimal import Decimal, getcontext
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

import numpy as np
import torch
import torch.nn.functional as F


getcontext().prec = 40
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CONTRACT_SCHEMA = "m722_lb_fuse_decoder_cpu_fastkill_contract_v1"
RESULT_SCHEMA = "m722_lb_fuse_decoder_cpu_fastkill_result_v1"
MODULES = {
    0: ("D0", 1536, 384, 15, 20, 30, 40, 4),
    1: ("D1", 770, 192, 30, 40, 60, 80, 2),
    2: ("D2", 386, 96, 60, 80, 120, 160, 1),
    3: ("D3", 194, 96, 120, 160, 240, 320, 1),
}
HEADLINE = {0, 2, 3}
LANES = 96
SOURCE_GROUP = 8
ACC24_BYTES = 3
ACC16_BYTES = 2
CONTROL_BYTES = 8192
BUDGET_BYTES = 240 * 1024
WEIGHT_TILE_BYTES_96 = 16 * 96 * 3 * 3
WEIGHT_TILE_BYTES_48 = 16 * 48 * 3 * 3
WEIGHT_REFILL_96 = 32 + WEIGHT_TILE_BYTES_96 // 128
WEIGHT_REFILL_48 = 32 + math.ceil(WEIGHT_TILE_BYTES_48 / 128)
GROUP_SERVICE_96 = 15
GROUP_SERVICE_48 = 9
COMMIT_CYCLES = 6 + 32 + 3
COMMIT_BYTES_PER_96 = 384
STRIPE_ALIGNMENT = 64


class Failure(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Failure(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("non-finite JSON token: " + token)))


def safe_member(name):
    member = PurePosixPath(name)
    require(member.parts and not member.is_absolute() and ".." not in member.parts
            and member.as_posix() == name, "unsafe member: " + name)
    return member


def verify_directory(path):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "bad sealed directory")
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink() and
            outer.is_file() and not outer.is_symlink(), "missing seals")
    expected_names = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "malformed manifest")
        expected, name = fields
        require(name not in expected_names, "duplicate sealed member")
        expected_names.add(name)
        member = path.joinpath(*safe_member(name).parts)
        require(member.is_file() and not member.is_symlink() and
                sha256(member) == expected, "sealed member mismatch: " + name)
    actual_names = set()
    for member in path.rglob("*"):
        require(not member.is_symlink(), "symlink in sealed directory")
        if member.is_file() and member.name not in (
                "SHA256SUMS", "SHA256SUMS.seal.sha256"):
            actual_names.add(member.relative_to(path).as_posix())
    require(actual_names == expected_names, "sealed population mismatch")
    fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(fields == [sha256(manifest), "SHA256SUMS"], "outer seal mismatch")
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer)}


def write_seal(path):
    path = Path(path)
    members = sorted(p for p in path.rglob("*") if p.is_file() and
                     p.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = path / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(
        sha256(member), member.relative_to(path).as_posix())
        for member in members), encoding="utf-8")
    (path / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")


def ratio(numerator, denominator):
    require(int(denominator) > 0, "zero denominator")
    return format(Decimal(int(numerator)) / Decimal(int(denominator)), ".12f")


def unpack_record(path, shape):
    logical = math.prod(int(value) for value in shape)
    payload = np.fromfile(str(path), dtype=np.uint8)
    bits = np.unpackbits(payload, bitorder="little")[:logical]
    require(bits.size == logical, "bitpack length mismatch")
    return bits.reshape(shape).astype(np.uint8, copy=False)


def quantized_weights(path, cin, cout):
    raw = np.fromfile(str(path), dtype="<f4")
    require(raw.size == cin * cout * 9, "weight size mismatch")
    weight = raw.reshape(cin, cout, 3, 3).astype(np.float64)
    maximum = np.max(np.abs(weight), axis=(0, 2, 3))
    scale = np.where(maximum == 0.0, 1.0, maximum / 127.0)
    quant = np.clip(np.rint(weight / scale[None, :, None, None]), -127, 127)
    quant = quant.astype(np.int8)
    return quant, {
        "policy": "local_probe_only_symmetric_per_output_channel_rint_clip127",
        "scale_f64_sha256": hashlib.sha256(scale.astype("<f8").tobytes()).hexdigest(),
        "int8_sha256": hashlib.sha256(quant.tobytes()).hexdigest(),
        "not_checkpoint_numeric_admission": True,
    }


def tap_fanout(hin, win):
    vertical = np.full(hin, 3, dtype=np.int64)
    horizontal = np.full(win, 3, dtype=np.int64)
    vertical[0] = 2
    horizontal[0] = 2
    return vertical[:, None] * horizontal[None, :]


def group_counts(bits, blocks):
    cin, hin, win = bits.shape
    tiles = (cin + 15) // 16
    padded = np.zeros((tiles * 16, hin, win), dtype=np.uint8)
    padded[:cin] = bits
    population = padded.reshape(tiles, 16, hin, win).sum(axis=1,
                                                          dtype=np.uint16)
    hout, wout = 2 * hin, 2 * win
    destination_counts = np.zeros((tiles, hout, wout), dtype=np.uint16)
    tile_ids = np.arange(tiles)
    for ky in range(3):
        sy = np.arange(hin)
        oy = 2 * sy - 1 + ky
        valid_y = (oy >= 0) & (oy < hout)
        sy, oy = sy[valid_y], oy[valid_y]
        for kx in range(3):
            sx = np.arange(win)
            ox = 2 * sx - 1 + kx
            valid_x = (ox >= 0) & (ox < wout)
            sx, ox = sx[valid_x], ox[valid_x]
            destination_counts[np.ix_(tile_ids, oy, ox)] += population[
                np.ix_(tile_ids, sy, sx)]
    contributors_one_block = int(destination_counts.sum(dtype=np.int64))
    osg_groups_one_block = int(((destination_counts + 7) // 8).sum(
        dtype=np.int64))
    fanout = tap_fanout(hin, win)
    lb_groups_one_block = int((((population + 7) // 8) *
                               fanout[None, :, :]).sum(dtype=np.int64))
    source_contributors = int((population * fanout[None, :, :]).sum(
        dtype=np.int64))
    require(source_contributors == contributors_one_block,
            "source/destination contributor conservation")
    active_tiles = int(np.count_nonzero(population.any(axis=(1, 2))))
    return {
        "contributors": contributors_one_block * blocks,
        "osg_groups": osg_groups_one_block * blocks,
        "lb_direct_groups": lb_groups_one_block * blocks,
        "active_input_tiles": active_tiles,
        "input_tiles": tiles,
    }


def source_columns_for_stripe(win, lo, hi):
    columns = []
    for sx in range(win):
        outputs = (2 * sx - 1, 2 * sx, 2 * sx + 1)
        if any(lo <= ox < hi for ox in outputs):
            columns.append(sx)
    return columns


def line_capacity(spec, accumulator_bytes, output_lanes=LANES):
    _name, _cin, _cout, _hin, _win, _hout, wout, _blocks = spec
    row_bytes = wout * output_lanes * accumulator_bytes
    row_bytes_aligned = ((row_bytes + 127) // 128) * 128
    return 3 * row_bytes_aligned


def a1_storage_plan(spec):
    name, _cin, _cout, _hin, win, _hout, wout, _blocks = spec
    available = BUDGET_BYTES - CONTROL_BYTES - WEIGHT_TILE_BYTES_96
    full = line_capacity(spec, ACC24_BYTES)
    if full <= available:
        stripe_width = wout
    else:
        raw_width = available // (3 * LANES * ACC24_BYTES)
        stripe_width = (raw_width // STRIPE_ALIGNMENT) * STRIPE_ALIGNMENT
        require(stripe_width >= STRIPE_ALIGNMENT, "no legal A1 stripe")
    stripes = [(lo, min(wout, lo + stripe_width))
               for lo in range(0, wout, stripe_width)]
    source_columns = [source_columns_for_stripe(win, lo, hi)
                      for lo, hi in stripes]
    backing = 3 * (((min(stripe_width, wout) * LANES * ACC24_BYTES + 127)
                    // 128) * 128)
    require(backing + CONTROL_BYTES + WEIGHT_TILE_BYTES_96 <= BUDGET_BYTES,
            "A1 no-spill storage exceeds budget")
    return {
        "module": name,
        "accumulator": "Acc24",
        "stripe_width": stripe_width,
        "stripe_count": len(stripes),
        "stripes": [list(pair) for pair in stripes],
        "summed_source_columns": sum(len(cols) for cols in source_columns),
        "unique_source_columns": len(set(col for cols in source_columns for col in cols)),
        "source_column_overlap": sum(len(cols) for cols in source_columns) - win,
        "onchip_psum_backing_bytes": backing,
        "control_bytes": CONTROL_BYTES,
        "weight_tile_bytes": WEIGHT_TILE_BYTES_96,
        "total_bytes": backing + CONTROL_BYTES + WEIGHT_TILE_BYTES_96,
        "offchip_psum_spill_bytes": 0,
        "model": True,
    }


def lb_storage_plans(spec, acc16_safe):
    name, _cin, _cout, _hin, _win, _hout, _wout, _blocks = spec
    acc24 = line_capacity(spec, ACC24_BYTES)
    acc16 = line_capacity(spec, ACC16_BYTES)
    half24 = line_capacity(spec, ACC24_BYTES, output_lanes=48)
    return {
        "acc24_full96": {
            "eligible": acc24 + CONTROL_BYTES + WEIGHT_TILE_BYTES_96 <= BUDGET_BYTES,
            "onchip_psum_backing_bytes": acc24,
            "total_bytes": acc24 + CONTROL_BYTES + WEIGHT_TILE_BYTES_96,
        },
        "acc16_full96": {
            "eligible": bool(acc16_safe and
                             acc16 + CONTROL_BYTES + WEIGHT_TILE_BYTES_96 <= BUDGET_BYTES),
            "trace_order_independent_acc16_safe": bool(acc16_safe),
            "onchip_psum_backing_bytes": acc16,
            "total_bytes": acc16 + CONTROL_BYTES + WEIGHT_TILE_BYTES_96,
        },
        "acc24_cout48_two_pass": {
            "eligible": half24 + CONTROL_BYTES + WEIGHT_TILE_BYTES_48 <= BUDGET_BYTES,
            "onchip_psum_backing_bytes": half24,
            "total_bytes": half24 + CONTROL_BYTES + WEIGHT_TILE_BYTES_48,
            "passes": 2,
        },
        "module": name,
    }


def lb_reconstruct(x, weight):
    require(x.device.type == "cpu" and weight.device.type == "cpu",
            "CPU only")
    n, cin, hin, win = x.shape
    require(weight.shape[0] == cin and weight.shape[2:] == (3, 3),
            "LB weight geometry")
    cout = weight.shape[1]
    flat = x.permute(0, 2, 3, 1).reshape(-1, cin)
    output = torch.zeros((n, cout, 2 * hin, 2 * win),
                         dtype=torch.float32, device="cpu")
    source_slices = (slice(1, None), slice(None), slice(None))
    output_slices = (slice(1, -1, 2), slice(0, None, 2),
                     slice(1, None, 2))
    for ky in range(3):
        for kx in range(3):
            product = torch.matmul(flat, weight[:, :, ky, kx])
            product = product.reshape(n, hin, win, cout).permute(0, 3, 1, 2)
            output[:, :, output_slices[ky], output_slices[kx]] += product[
                :, :, source_slices[ky], source_slices[kx]]
    return output


def numeric_replay(bits, quant):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU must be hidden")
    x = torch.from_numpy(bits.astype(np.float32, copy=False))
    weight = torch.from_numpy(quant.astype(np.float32, copy=False))
    require(x.device.type == "cpu" and weight.device.type == "cpu", "CPU tensors")
    with torch.inference_mode():
        a1 = F.conv_transpose2d(x, weight, stride=2, padding=1,
                                output_padding=1)
        lb = lb_reconstruct(x, weight)
        absolute_bound = F.conv_transpose2d(x, weight.abs(), stride=2,
                                            padding=1, output_padding=1)
        mismatch = int(torch.count_nonzero(a1 != lb).item())
        integer_exact = bool(torch.equal(a1, a1.round()) and
                             torch.equal(lb, lb.round()) and
                             torch.equal(absolute_bound,
                                         absolute_bound.round()))
        minima = [int(value) for value in a1.amin(dim=(1, 2, 3)).tolist()]
        maxima = [int(value) for value in a1.amax(dim=(1, 2, 3)).tolist()]
        bounds = [int(value) for value in absolute_bound.amax(
            dim=(1, 2, 3)).tolist()]
        sums = [int(value) for value in a1.sum(
            dim=(1, 2, 3), dtype=torch.float64).tolist()]
    del x, weight, a1, lb, absolute_bound
    return {"mismatches": mismatch, "integer_exact": integer_exact,
            "minima": minima, "maxima": maxima,
            "order_independent_abs_prefix_bounds": bounds,
            "integer_sums": sums}


def cycle_row(bits, spec, counts, a1_plan, chosen_lb):
    name, cin, cout, hin, win, hout, wout, blocks = spec
    plane_bits = cin * hin * win
    dense_vectors = hout * wout * blocks
    commit_cycles = dense_vectors * COMMIT_CYCLES
    commit_bytes = dense_vectors * COMMIT_BYTES_PER_96
    descriptors = counts["contributors"]
    osg_groups = counts["osg_groups"]
    lb_groups = counts["lb_direct_groups"]
    bundles = (descriptors + SOURCE_GROUP - 1) // SOURCE_GROUP

    a1_source_scan = cin * hin * a1_plan["summed_source_columns"] * blocks
    a1_weight_misses = counts["active_input_tiles"] * blocks * a1_plan["stripe_count"]
    a1 = {
        "source_scan": a1_source_scan,
        "descriptor_or_bundle": descriptors + 2 * bundles,
        "group_service": osg_groups * GROUP_SERVICE_96,
        "weight_refill": a1_weight_misses * WEIGHT_REFILL_96,
        "dense_output_commit": commit_cycles,
        "terminal_directory": (1029 * a1_plan["stripe_count"] +
                               2 * (blocks * a1_plan["stripe_count"] - 1)),
    }
    a1["total"] = sum(a1.values())

    passes = 1
    group_service = GROUP_SERVICE_96
    refill = WEIGHT_REFILL_96
    acc_bytes = ACC24_BYTES
    if chosen_lb == "acc16_full96":
        acc_bytes = ACC16_BYTES
    elif chosen_lb == "acc24_cout48_two_pass":
        passes = 2
        group_service = GROUP_SERVICE_48
        refill = WEIGHT_REFILL_48
    lb = {
        "source_scan": plane_bits * blocks * passes,
        "direct_bundle": 2 * lb_groups * passes,
        "group_service": lb_groups * group_service * passes,
        "weight_refill": counts["active_input_tiles"] * blocks * refill * passes,
        "dense_output_commit": commit_cycles,
        "row_rotation": 2 * hout * blocks * passes + 2 * (blocks * passes - 1),
    }
    lb["total"] = sum(lb.values())
    slices = LANES // 16 if passes == 1 else 48 // 16
    port_ops_per_group_pass = 2 * slices
    port_conflicts = int(group_service < port_ops_per_group_pass)
    return {
        "active_sources": int(bits.sum(dtype=np.int64)),
        "contributors": descriptors,
        "a1_osg_groups": osg_groups,
        "lb_direct_groups": lb_groups,
        "lb_over_osg_groups": ratio(lb_groups, osg_groups),
        "a1_cycles": a1,
        "lb_cycles": lb,
        "traffic": {
            "dense_commit_bytes_a1": commit_bytes,
            "dense_commit_bytes_lb": commit_bytes,
            "a1_onchip_psum_rmw_bytes": osg_groups * 2 * LANES * ACC24_BYTES,
            "lb_onchip_psum_rmw_bytes": (lb_groups * passes * 2 *
                                          (LANES // passes) * acc_bytes),
            "a1_offchip_psum_spill_bytes": 0,
            "lb_offchip_psum_spill_bytes": 0,
            "a1_weight_refill_bytes": a1_weight_misses * WEIGHT_TILE_BYTES_96,
            "lb_weight_refill_bytes": (counts["active_input_tiles"] * blocks *
                                        passes * (WEIGHT_TILE_BYTES_96 if passes == 1
                                                  else WEIGHT_TILE_BYTES_48)),
        },
        "port_model": {
            "single_1rw_onchip_psum_port": True,
            "a1_port_operations": osg_groups * 12,
            "lb_port_operations": lb_groups * passes * port_ops_per_group_pass,
            "lb_port_conflict_events": port_conflicts * lb_groups * passes,
            "serialized_group_service_covers_all_rmw": port_conflicts == 0,
            "model_not_rtl": True,
        },
    }


def summarize(rows, selector):
    selected = [row for row in rows if selector(row)]
    a1 = sum(row["a1_cycles"]["total"] for row in selected)
    lb = sum(row["lb_cycles"]["total"] for row in selected)
    return {
        "planes": len(selected),
        "a1_cycles": a1,
        "lb_cycles": lb,
        "a1_over_lb": ratio(a1, lb),
        "contributors": sum(row["contributors"] for row in selected),
        "a1_osg_groups": sum(row["a1_osg_groups"] for row in selected),
        "lb_direct_groups": sum(row["lb_direct_groups"] for row in selected),
        "lb_over_osg_groups": ratio(
            sum(row["lb_direct_groups"] for row in selected),
            sum(row["a1_osg_groups"] for row in selected)),
        "a1_onchip_psum_rmw_bytes": sum(
            row["traffic"]["a1_onchip_psum_rmw_bytes"] for row in selected),
        "lb_onchip_psum_rmw_bytes": sum(
            row["traffic"]["lb_onchip_psum_rmw_bytes"] for row in selected),
        "a1_offchip_psum_spill_bytes": 0,
        "lb_offchip_psum_spill_bytes": 0,
        "a1_commit_bytes": sum(
            row["traffic"]["dense_commit_bytes_a1"] for row in selected),
        "lb_commit_bytes": sum(
            row["traffic"]["dense_commit_bytes_lb"] for row in selected),
        "a1_port_conflict_events": 0,
        "lb_port_conflict_events": sum(
            row["port_model"]["lb_port_conflict_events"] for row in selected),
    }


def self_test():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "self-test GPU hidden")
    bits = np.asarray([[[1, 0, 1], [0, 1, 0]],
                       [[0, 1, 0], [1, 0, 1]],
                       [[1, 1, 0], [0, 0, 1]],
                       [[0, 0, 0], [1, 1, 1]],
                       [[1, 0, 0], [0, 1, 0]]], dtype=np.uint8)
    groups = group_counts(bits, 1)
    require(groups["contributors"] > 0 and groups["lb_direct_groups"] >=
            groups["osg_groups"], "group-count self-test")
    rng = np.random.default_rng(722)
    batch = rng.integers(0, 2, size=(2, 5, 2, 3), dtype=np.uint8)
    weight = rng.integers(-7, 8, size=(5, 4, 3, 3), dtype=np.int8)
    replay = numeric_replay(batch, weight)
    require(replay["mismatches"] == 0 and replay["integer_exact"],
            "numeric self-test")
    tiny = ("T", 5, 4, 2, 3, 4, 6, 1)
    plan = a1_storage_plan(tiny)
    require(plan["offchip_psum_spill_bytes"] == 0, "storage self-test")
    print("PASS M722 static CPU self-test")


def production(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden")
    require(not torch.cuda.is_available(), "GPU visible to torch")
    require(torch.get_default_dtype() == torch.float32, "torch dtype drift")
    root = Path(args.repo_root).resolve()
    hw = root / "hw_autoresearch_nts07"
    contract_path = Path(args.contract).resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") == CONTRACT_SCHEMA, "contract schema")
    require(sha256(hw / "docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA,
            "docs359 drift")

    m699 = hw / contract["inputs"]["m699_directory"]
    m705 = hw / contract["inputs"]["m705_review_directory"]
    m686 = hw / contract["inputs"]["m686_weight_directory"]
    id699 = verify_directory(m699)
    id705 = verify_directory(m705)
    id686 = verify_directory(m686)
    require(sha256(m699 / "manifest.json") ==
            contract["inputs"]["m699_manifest_sha256"] and
            id699["outer_seal_file_sha256"] ==
            contract["inputs"]["m699_outer_seal_file_sha256"],
            "M699 identity")
    require(sha256(m705 / "review.json") ==
            contract["inputs"]["m705_review_sha256"] and
            id705["outer_seal_file_sha256"] ==
            contract["inputs"]["m705_outer_seal_file_sha256"],
            "M705 identity")
    require(sha256(m686 / "manifest.json") ==
            contract["inputs"]["m686_manifest_sha256"] and
            id686["outer_seal_file_sha256"] ==
            contract["inputs"]["m686_outer_seal_file_sha256"],
            "M686 identity")
    review705 = strict_json(m705 / "review.json")
    require(review705.get("go") is True and review705.get("score") == 98 and
            review705.get("severity", {}).get("p0") == 0 and
            review705.get("severity", {}).get("p1") == 0,
            "M705 admission")
    manifest = strict_json(m699 / "manifest.json")
    require(len(manifest["records"]) == 120, "record population")

    weights = {}
    weight_ids = {}
    formal_integer_bounds = {}
    for index, spec in MODULES.items():
        name, cin, cout, _hin, _win, _hout, _wout, _blocks = spec
        weight_name = ("d1.weight.folded_theta.f32le" if index == 1
                       else "d{}.weight.f32le".format(index))
        weights[index], weight_ids[index] = quantized_weights(
            m686 / "weights" / weight_name, cin, cout)
        bound = cin * 4 * 127
        require(bound < (1 << 23) and bound < (1 << 24),
                "formal float/Acc24 bound")
        formal_integer_bounds[name] = {
            "maximum_four_tap_absolute_sum": bound,
            "below_signed_acc24": True,
            "below_float32_exact_integer_limit": True,
        }

    rows = []
    ranges = {MODULES[index][0]: {"minimum": 0, "maximum": 0,
                                  "order_independent_abs_prefix_bound": 0}
              for index in MODULES}
    numeric_signature = hashlib.sha256()
    record_numeric_mismatches = 0
    for record_index, record in enumerate(manifest["records"]):
        index = int(record["module_index"])
        spec = MODULES[index]
        name, cin, cout, hin, win, hout, wout, blocks = spec
        require(tuple(record["input_shape"]) == (10, 1, cin, hin, win),
                "record shape")
        require((index == 1 and record["route"] ==
                 "EXACT_SCALED_BINARY_BITPACK") or
                (index != 1 and record["route"] == "EXACT_BINARY_BITPACK"),
                "route drift")
        payload = m699.joinpath(*safe_member(record["relative_path"]).parts)
        packed_sha = (record["statistics"]["scaled_binary_audit"]["packed_sha256"]
                      if index == 1 else record["statistics"]["packed_sha256"])
        require(sha256(payload) == packed_sha, "payload identity")
        bits = unpack_record(payload, (10, 1, cin, hin, win))[:, 0]
        replay = numeric_replay(bits, weights[index])
        require(replay["integer_exact"], "noninteger CPU arithmetic")
        record_numeric_mismatches += replay["mismatches"]
        ranges[name]["minimum"] = min(ranges[name]["minimum"],
                                       min(replay["minima"]))
        ranges[name]["maximum"] = max(ranges[name]["maximum"],
                                       max(replay["maxima"]))
        ranges[name]["order_independent_abs_prefix_bound"] = max(
            ranges[name]["order_independent_abs_prefix_bound"],
            max(replay["order_independent_abs_prefix_bounds"]))
        numeric_signature.update(np.asarray(
            [record_index, index, replay["mismatches"]] + replay["minima"] +
            replay["maxima"] + replay["order_independent_abs_prefix_bounds"] +
            replay["integer_sums"], dtype="<i8").tobytes())
        for time in range(10):
            counts = group_counts(bits[time], blocks)
            row = cycle_row(bits[time], spec, counts,
                            a1_storage_plan(spec),
                            "acc16_full96" if index == 3 else "acc24_full96")
            row.update({
                "record_index": record_index,
                "global_sample_id": int(record["global_sample_id"]),
                "sequence": record["sequence"],
                "sequence_sample_id": int(record["sequence_sample_id"]),
                "module_index": index,
                "module": name,
                "time": time,
                "headline_eligible": index in HEADLINE,
                "numeric_minimum": replay["minima"][time],
                "numeric_maximum": replay["maxima"][time],
                "order_independent_abs_prefix_bound":
                    replay["order_independent_abs_prefix_bounds"][time],
                "a1_lb_acc24_mismatches": replay["mismatches"],
            })
            rows.append(row)
    require(len(rows) == 1200 and record_numeric_mismatches == 0,
            "full numeric miter")

    for name, observed in ranges.items():
        observed["final_values_fit_acc16"] = (
            observed["minimum"] >= -(1 << 15) and
            observed["maximum"] <= (1 << 15) - 1)
        observed["trace_all_orders_fit_acc16"] = (
            observed["order_independent_abs_prefix_bound"] <= (1 << 15) - 1)
        observed["trace_all_orders_fit_acc24"] = (
            observed["order_independent_abs_prefix_bound"] <= (1 << 23) - 1)
        observed["scope"] = "complete M705 S3x10 local-INT8 probe; not all possible inputs"

    a1_plans = {MODULES[index][0]: a1_storage_plan(MODULES[index])
                for index in MODULES}
    lb_plans = {MODULES[index][0]: lb_storage_plans(
        MODULES[index], ranges[MODULES[index][0]]["trace_all_orders_fit_acc16"])
        for index in MODULES}
    require(all(plan["offchip_psum_spill_bytes"] == 0
                for plan in a1_plans.values()), "fair A1 spills")
    require(lb_plans["D3"]["acc16_full96"]["eligible"] and
            lb_plans["D3"]["acc24_cout48_two_pass"]["eligible"],
            "D3 alternatives unavailable")

    totals = {
        "all": summarize(rows, lambda _row: True),
        "headline_d0_d2_d3": summarize(rows, lambda row:
                                        row["module_index"] in HEADLINE),
        "diagnostic_d1": summarize(rows, lambda row:
                                    row["module_index"] == 1),
    }
    per_module = {MODULES[index][0]: summarize(
        rows, lambda row, index=index: row["module_index"] == index)
        for index in MODULES}
    sequences = sorted(set(row["sequence"] for row in rows))
    per_sequence = {sequence: summarize(
        rows, lambda row, sequence=sequence:
        row["sequence"] == sequence and row["module_index"] in HEADLINE)
        for sequence in sequences}

    # D3 C48 is a separately charged sensitivity, not the chosen candidate.
    d3_rows = [row for row in rows if row["module_index"] == 3]
    c48_a1 = sum(row["a1_cycles"]["total"] for row in d3_rows)
    c48_cycles = 0
    c48_rmw = 0
    for row in d3_rows:
        bits_dummy = None
        groups = row["lb_direct_groups"]
        active_tiles = MODULES[3][1] // 16 + int(MODULES[3][1] % 16 != 0)
        # All M705 D3 tiles are active in the admitted population; assert from
        # the chosen row's exact weight traffic identity rather than hide an
        # alternate runtime oracle.
        chosen_refills = row["traffic"]["lb_weight_refill_bytes"] // WEIGHT_TILE_BYTES_96
        require(chosen_refills <= active_tiles, "D3 refill sanity")
        c48 = {
            "source_scan": MODULES[3][1] * MODULES[3][3] * MODULES[3][4] * 2,
            "direct_bundle": 4 * groups,
            "group_service": groups * GROUP_SERVICE_48 * 2,
            "weight_refill": chosen_refills * WEIGHT_REFILL_48 * 2,
            "dense_output_commit": row["a1_cycles"]["dense_output_commit"],
            "row_rotation": 2 * MODULES[3][5] * 2 + 2,
        }
        c48_cycles += sum(c48.values())
        c48_rmw += groups * 2 * 2 * 48 * ACC24_BYTES
    c48_sensitivity = {
        "admitted_as_chosen_candidate": False,
        "a1_cycles": c48_a1,
        "lb_c48_cycles": c48_cycles,
        "a1_over_lb_c48": ratio(c48_a1, c48_cycles),
        "lb_c48_onchip_psum_rmw_bytes": c48_rmw,
        "backing_plan": lb_plans["D3"]["acc24_cout48_two_pass"],
        "model": True,
    }

    headline = totals["headline_d0_d2_d3"]
    headline_ratio = Decimal(headline["a1_over_lb"])
    performance_go = (headline_ratio >= Decimal("1.20") and
                      all(Decimal(value["a1_over_lb"]) >= Decimal("1.05")
                          for value in per_sequence.values()))
    cycle_within_five_percent = headline_ratio >= Decimal(1) / Decimal("1.05")
    fair_a1_zero_spill = headline["a1_offchip_psum_spill_bytes"] == 0
    spill_reduction = Decimal("0")
    traffic_go = (cycle_within_five_percent and not fair_a1_zero_spill and
                  spill_reduction >= Decimal("0.30"))
    status = ("GO_CPU_GATE__FRESH_HAMMER_REQUIRED" if performance_go or traffic_go
              else "KILL_NO_RTL__FAIR_A1_ZERO_PSUM_SPILL")

    report = {
        "schema": RESULT_SCHEMA,
        "date": "2026-08-28",
        "status": status,
        "decision": {
            "performance_go": performance_go,
            "traffic_go": traffic_go,
            "fair_a1_zero_offchip_psum_spill": fair_a1_zero_spill,
            "cycle_within_five_percent": cycle_within_five_percent,
            "offchip_psum_spill_reduction_fraction": format(spill_reduction,
                                                              ".12f"),
            "headline_a1_over_lb": headline["a1_over_lb"],
            "minimum_sequence_a1_over_lb": min(
                value["a1_over_lb"] for value in per_sequence.values()),
            "rtl_authorized_now": False,
            "fresh_result_hammer_required": True,
            "kill_reason": "The fair source-order A1-OSG uses the same three-row lifetime and a legal D3 width stripe, so it has zero off-chip psum spill. Direct LB issue loses destination packing and cannot satisfy either gate.",
        },
        "totals": totals,
        "per_sequence_headline": per_sequence,
        "per_module": per_module,
        "numeric_exactness": {
            "records": 120,
            "planes": 1200,
            "a1_lb_acc24_mismatches": record_numeric_mismatches,
            "integer_arithmetic_exact": True,
            "numeric_signature_sha256": numeric_signature.hexdigest(),
            "dynamic_ranges": ranges,
            "formal_integer_bounds": formal_integer_bounds,
        },
        "storage": {
            "budget_bytes": BUDGET_BYTES,
            "a1_no_spill_plans": a1_plans,
            "lb_plans": lb_plans,
            "d3_chosen": "acc16_full96",
            "d3_cout48_two_pass_sensitivity": c48_sensitivity,
        },
        "first_principles": {
            "classic_prior": [
                "ordinary three-row convolution line buffers",
                "GANAX transposed-convolution zero-pattern reordering (arXiv:1806.01107)",
                "Chang and Chang transposed-convolution decomposition (arXiv:2205.02103)",
                "standard stride-2 polyphase decomposition and existing M514/M523 mapping/bundling",
            ],
            "object_delta_only": "H67 binary ATLIF descriptors, parity-asymmetric 1/2/2/4 tap sets, signed local-INT8/Acc24 arithmetic, 240-KiB storage and 96-wide issue",
            "line_buffer_or_polyphase_novelty": False,
            "a1_output_equivalence_mismatch_zero": True,
            "sequence_equivalent_to_a1_osg": False,
            "execution_collision": "A source-order direct group cannot combine contributors from up to four source positions as A1-OSG does; restoring that packing requires destination-keyed context/descriptor state and collapses back toward A1-OSG/PIDP.",
            "acc16_is_common_precision_optimization": True,
            "model_only": True,
        },
        "local_int8_probe_identities": {
            MODULES[index][0]: weight_ids[index] for index in MODULES},
        "runtime": {
            "python": os.path.realpath(os.sys.executable),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch.cuda.is_available(),
            "cpu_only": True,
        },
        "claim_boundary": contract["claim_boundary"],
        "identity": {
            "contract_path": str(contract_path),
            "contract_sha256": sha256(contract_path),
            "analyzer_path": str(Path(__file__).resolve()),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "m699_manifest_sha256": sha256(m699 / "manifest.json"),
            "m699_outer_seal_file_sha256": id699["outer_seal_file_sha256"],
            "m705_review_sha256": sha256(m705 / "review.json"),
            "m705_outer_seal_file_sha256": id705["outer_seal_file_sha256"],
            "m686_manifest_sha256": sha256(m686 / "manifest.json"),
            "m686_outer_seal_file_sha256": id686["outer_seal_file_sha256"],
            "docs359_sha256": sha256(hw / "docs/359_DATE终局冻结_20260813.md"),
        },
        "rows_file": "rows.jsonl",
    }

    output = Path(args.output).resolve()
    require(not output.exists() and not output.is_symlink(),
            "canonical output exists")
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".staging.",
                                    dir=str(output.parent)))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        with (staging / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True,
                                        separators=(",", ":"),
                                        allow_nan=False) + "\n")
        (staging / "RUN_COMPLETE.txt").write_text(status + "\n",
                                                   encoding="utf-8")
        write_seal(staging)
        verify_directory(staging)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": status,
        "headline_a1_over_lb": headline["a1_over_lb"],
        "minimum_sequence": min(value["a1_over_lb"]
                                for value in per_sequence.values()),
        "a1_zero_spill": fair_a1_zero_spill,
        "a1_lb_mismatches": record_numeric_mismatches,
        "d3_acc16_safe": ranges["D3"]["trace_all_orders_fit_acc16"],
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo-root")
    parser.add_argument("--contract")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    require(args.repo_root and args.contract and args.output,
            "production arguments required")
    production(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
