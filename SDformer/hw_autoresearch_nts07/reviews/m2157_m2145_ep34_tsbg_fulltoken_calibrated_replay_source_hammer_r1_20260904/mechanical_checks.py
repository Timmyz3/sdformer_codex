#!/opt/anaconda3/bin/python
"""Independent, CPU-only M2157 source hammer for M2145.

This checker never calls the M2145 production ``run`` path and never invokes
VCS, simv, EDA, GPU, or license tooling.  It independently reconstructs the
frozen cycle calibration, checks the production batch kernel against all
2,880 frozen VCS rows, and scans capture headers without decompressing or
replaying a production frame.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import struct
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / (
    "system_simulator/scripts/"
    "analyze_m2145_ep34_tsbg_fulltoken_calibrated_replay.py")
TEST = HW / (
    "system_simulator/tests/"
    "test_m2145_ep34_tsbg_fulltoken_calibrated_replay_source.py")
CONTRACT = HW / (
    "contracts/m2145_ep34_tsbg_fulltoken_calibrated_replay_source_"
    "contract_r1_20260904.json")
AUTHOR = HW / (
    "reviews/m2145_ep34_tsbg_fulltoken_calibrated_replay_source_"
    "author_receipt_r1_20260904")
CAPTURE = HW / (
    "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_"
    "binary_capture_s40_r1_20260901")
LOW_META = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"
LOW_MEMH = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh"
LOW_RESULT = HW / (
    "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/"
    "result.json")
HIGH_META = HW / (
    "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.json")
HIGH_MEMH = HW / (
    "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.memh")
HIGH_RESULT = HW / (
    "results/m2067_ep34_fc2_exact_continuation_vcs_r10_codex_batch_"
    "20260904/result.json")
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
FRAME = struct.Struct("<8sHH11I")

EXPECTED = {
    SOURCE: "1dc41d29ad7a0b7e175e5cd5f379c60b1320fbe08d4ff374c9d6e6ebeb37db57",
    TEST: "b0816b5bae7588289202dadba445f31531ba95e38b7ccac5dbb953055e82c547",
    CONTRACT: "c84fd342e839f5f26e2e8418856523ef9207a47c6bb0abd3f5e40a1cde9f1f38",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    CAPTURE / "fc_frames.bin": "dceb6c0c80b9c5898d10b4ad813fbcd7683fa80191b54b78eadaadda04a818b1",
    LOW_META: "3ac7048f0a97aeea0ac91627d303f4eea06b8a48bab816468825acfee180ccc5",
    LOW_MEMH: "487ca0073526b973220abd77c91d12dbc2420901443541ec5a79e36a780e1bf0",
    HIGH_META: "5b44aa6a248a8768d59a85270a50b3ba805467377365e1b6e4ad8e58eafc7b34",
    HIGH_MEMH: "c617c6311ce44f15fb820f5dba5460ebd127235a13acd56724b56ccbb10cd594",
}


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def module():
    spec = importlib.util.spec_from_file_location("m2145_reviewed", SOURCE)
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    return imported


def dense(lines: list[str], slot: int, groups: int) -> np.ndarray:
    physical = 48 if groups <= 48 else 192
    offset = slot * 4 * physical
    result = np.zeros((4, groups, 16), dtype=np.int8)
    for context in range(4):
        for group in range(groups):
            word = int(lines[offset + context * physical + group], 16)
            active = word & 0xffff
            sign_bits = (word >> 16) & 0xffff
            need(sign_bits & ~active == 0, "sign outside activity")
            for lane in range(16):
                if active & (1 << lane):
                    result[context, group, lane] = (
                        -1 if sign_bits & (1 << lane) else 1)
    return result


def independent_cycles(values: np.ndarray, mode: int) -> tuple[int, dict]:
    """Independent transcription of the reviewed M2018/M803 recurrence."""
    need(values.shape == (4, 48, 16), "independent physical geometry")
    positions = ((c, g) for c in range(4) for g in range(48)) if mode == 0 \
        else ((c, g) for g in range(48) for c in range(4))
    valid = [False] * 4
    resident = [0] * 4
    age = [0] * 4
    clock = 1
    cycle = 383
    hits = misses = evictions = live = issues = 0
    for context, group in positions:
        lower = bool(np.any(values[context, group, :8]))
        upper = bool(np.any(values[context, group, 8:]))
        if not lower and not upper:
            continue
        live += 1
        hit = next((i for i in range(4)
                    if valid[i] and resident[i] == group), None)
        if hit is None:
            misses += 1
            victim = next((i for i in range(4) if not valid[i]), None)
            if victim is None:
                victim = min(range(4), key=lambda i: (age[i], i))
                evictions += 1
            valid[victim] = True
            resident[victim] = group
            age[victim] = clock + 1
        else:
            hits += 1
            age[hit] = clock
        clock += 1
        cycle += 1
        if hit is None:
            for _beat in range(12):
                completions = []
                for bank in range(8):
                    accepted = cycle
                    while (accepted + 1 + bank * 2) % 7 == 0:
                        accepted += 1
                    completions.append(accepted + (8 - bank) + 1)
                cycle = max(completions) + 1
        issued = 6 * (int(lower) + int(upper))
        issues += issued
        for _ in range(issued):
            while cycle % 11 == 3:
                cycle += 1
            cycle += 1
    cycle += 1
    for _ in range(24):
        while cycle % 13 == 5:
            cycle += 1
        cycle += 1
    return cycle - 383, {
        "hits": hits, "misses": misses, "evictions": evictions,
        "live_rows": live, "issues": issues,
        "scalar_reads": misses * 12 * 8,
    }


def identity_and_seals(imported) -> dict:
    for path, expected in EXPECTED.items():
        need(path.is_file() and not path.is_symlink(), f"identity node {path}")
        need(sha(path) == expected, f"identity drift {path}")
    imported.verify_double_seal(AUTHOR)
    imported.verify_double_seal(CAPTURE)
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(side.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(outer.read_text().split() == [sha(side), side.name],
         "contract outer seal")
    return {"exact_files": len(EXPECTED), "author_seal": True,
            "capture_seal": True, "contract_double_seal": True}


def exhaustive_calibration(imported) -> dict:
    low_meta, low_result = load(LOW_META), load(LOW_RESULT)
    high_meta, high_result = load(HIGH_META), load(HIGH_RESULT)
    low_lines = LOW_MEMH.read_text(encoding="ascii").splitlines()
    high_lines = HIGH_MEMH.read_text(encoding="ascii").splitlines()
    need(len(low_meta["rows"]) == len(low_result["rows"]) == 1920,
         "low population")
    low_values = np.stack([dense(low_lines, slot, 48)
                           for slot in range(1920)])
    independent_mismatch = 0
    for slot, observed in enumerate(low_result["rows"]):
        for mode, field in ((0, "base_cycles"), (1, "tsbg_cycles")):
            predicted, _ = independent_cycles(low_values[slot], mode)
            independent_mismatch += int(predicted != int(observed[field]))
    lower = np.any(low_values[:, :, :, :8] != 0, axis=3)
    upper = np.any(low_values[:, :, :, 8:] != 0, axis=3)
    batch_mismatch = 0
    for mode, field in ((0, "base_cycles"), (1, "tsbg_cycles")):
        predicted = imported.batch_engine_cycles(lower, upper, mode)[:, 0]
        observed = np.asarray([row[field] for row in low_result["rows"]],
                              dtype=np.int64)
        batch_mismatch += int(np.count_nonzero(predicted != observed))

    calibrated = imported.calibration()
    need(len(high_meta["rows"]) == len(high_result["rows"]) == 960,
         "continuation population")
    high_mismatch = 0
    residuals: dict[tuple[int, int], list[int]] = defaultdict(list)
    for slot, (metadata, observed) in enumerate(
            zip(high_meta["rows"], high_result["rows"])):
        groups = int(metadata["source_groups"])
        tiles = int(metadata["output_tiles"])
        values = dense(high_lines, slot, groups)
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        key = (groups, tiles, digest)
        for mode, field in ((0, "base_cycles"), (1, "tsbg_cycles")):
            backbone = 0
            for begin in range(0, groups, 48):
                service, _ = independent_cycles(
                    values[:, begin:begin + 48, :], mode)
                backbone += 768 + service
            backbone *= tiles
            correction = calibrated["registries"][mode][key]
            reconstructed = backbone + correction
            high_mismatch += int(reconstructed != int(observed[field]))
            residuals[(groups, mode)].append(int(observed[field]) - backbone)

    expected_profiles = {
        (96, 0): (-9, 5, 13), (96, 1): (-11, 6, 36),
        (192, 0): (-27, 4, 29), (192, 1): (-39, 5, 90),
    }
    profile_rows = {}
    for key, values in residuals.items():
        ordered = sorted(values)
        profile = (min(values), ordered[len(ordered) // 2], max(values))
        need(profile == expected_profiles[key], f"residual profile {key}")
        profile_rows[f"G{key[0]}_mode{key[1]}"] = {
            "count": len(values), "min_median_max": list(profile)}

    need(independent_mismatch == batch_mismatch == high_mismatch == 0,
         "cycle calibration mismatch")
    return {
        "vcs_rows": 2880, "cycle_fields": 5760,
        "independent_recurrence_mismatches": independent_mismatch,
        "production_batch_kernel_mismatches": batch_mismatch + high_mismatch,
        "profiles": profile_rows,
    }


def population_and_header_scan(imported) -> dict:
    layers_all = load(CAPTURE / "layers.json")["layers"]
    samples = load(CAPTURE / "sample_order.json")["samples"]
    layers = [row for row in layers_all if row["target"] in ("FC1", "FC2")]
    need(len(layers) == 24 and len({row["layer_id"] for row in layers}) == 24,
         "FC layer identity")
    need(Counter(row["target"] for row in layers) == {"FC1": 12, "FC2": 12},
         "FC target inventory")
    need(len(samples) == 40 and
         {row["global_sample_id"] for row in samples} == set(range(40)),
         "sample identity")
    per_sequence = Counter(row["sequence"] for row in samples)
    need(len(per_sequence) == 4 and set(per_sequence.values()) == {10},
         "sequence balance")
    expected_pairs = [(sample["global_sample_id"], layer["layer_id"])
                      for sample in samples for layer in layers]
    pair = frame_index = token_start = frames = quartets = 0
    pair_tokens = Counter()
    with (CAPTURE / "fc_frames.bin").open("rb") as stream:
        while True:
            raw_header = stream.read(FRAME.size)
            if not raw_header:
                break
            need(len(raw_header) == FRAME.size, "truncated capture header")
            (magic, version, header_size, layer_id, sample_id, observed_frame,
             observed_start, token_count, channels, row_bytes, _nnz,
             _raw_bytes, compressed_bytes, _crc) = FRAME.unpack(raw_header)
            need(pair < len(expected_pairs), "extra pair")
            need((sample_id, layer_id) == expected_pairs[pair], "pair order")
            need(magic == b"M1558F01" and version == 1 and
                 header_size == FRAME.size, "frame identity")
            need(observed_frame == frame_index and observed_start == token_start,
                 "frame sequence")
            layer = layers[pair % len(layers)]
            need(token_count > 0 and token_count % 4 == 0 and
                 channels == int(layer["input_channels"]) and
                 channels == int(layer["weight_layout"]["source_group_count"]) * 16 and
                 row_bytes == (channels + 7) // 8, "frame geometry")
            stream.seek(compressed_bytes, os.SEEK_CUR)
            frames += 1
            quartets += token_count // 4
            pair_tokens[(sample_id, layer_id)] += token_count
            token_start += token_count
            need(token_start <= int(layer["tokens_per_call"]), "token overflow")
            if token_start == int(layer["tokens_per_call"]):
                pair += 1
                frame_index = token_start = 0
            else:
                frame_index += 1
    need(pair == 960 and token_start == 0 and quartets == 11_160_000,
         "full header population")
    need(all(pair_tokens[pair_id] == int(layers[index % 24]["tokens_per_call"])
             for index, pair_id in enumerate(expected_pairs)),
         "per-pair token population")

    calibrated = imported.calibration()
    locations = calibrated["locations_by_pair"]
    need(len(locations) == 320 and sum(map(len, locations.values())) == 960,
         "continuation calibration location population")
    layer_by_id = {int(row["layer_id"]): row for row in layers}
    for (_sample, layer_id), rows in locations.items():
        layer = layer_by_id[layer_id]
        expected = [0, int(layer["tokens_per_call"]) // 2,
                    int(layer["tokens_per_call"]) - 4]
        need(sorted(row["token_start"] for row in rows) == expected and
             all(row["token_start"] % 4 == 0 for row in rows),
             "continuation first/middle/last locations")
    return {
        "samples": 40, "sequences": dict(sorted(per_sequence.items())),
        "layers": 24, "fc1": 12, "fc2": 12, "sample_layer_pairs": 960,
        "frames_header_scanned_without_payload_decode": frames,
        "aligned_b4_quartets": quartets,
        "continuation_location_pairs": 320,
        "continuation_locations": 960,
    }


def semantics_and_mutations(imported) -> dict:
    calibrated = imported.calibration()
    high_meta = load(HIGH_META)
    high_result = load(HIGH_RESULT)
    lines = HIGH_MEMH.read_text(encoding="ascii").splitlines()
    representative = {}
    for groups in (96, 192):
        slot = next(index for index, row in enumerate(high_meta["rows"])
                    if int(row["source_groups"]) == groups)
        metadata = high_meta["rows"][slot]
        values = dense(lines, slot, groups)
        layer = {"layer_id": int(metadata["layer_id"]),
                 "weight_layout": {
                     "source_group_count": groups,
                     "output_tile_count": int(metadata["output_tiles"])}}
        exact = imported.replay_batch(
            values[None, ...], layer, calibrated,
            int(metadata["sample_id"]), int(metadata["token_start"]))
        observed = high_result["rows"][slot]
        need(bool(exact["calibration_exact_hit"][0]) and
             not bool(exact["calibration_median"][0]) and
             int(exact["base_cycles"][0]) == int(observed["base_cycles"]) and
             int(exact["tsbg_cycles"][0]) == int(observed["tsbg_cycles"]),
             "seen continuation exact path")
        unseen_values = values.copy()
        unseen_values[0, 0, 0] = 0 if unseen_values[0, 0, 0] else 1
        unseen = imported.replay_batch(
            unseen_values[None, ...], layer, calibrated, 999, 0)
        need(not bool(unseen["calibration_exact_hit"][0]) and
             bool(unseen["calibration_median"][0]),
             "unseen continuation must use median")
        ordinary = calibrated["continuation_residual_profiles"][f"G{groups}_ordinary"]
        tsbg = calibrated["continuation_residual_profiles"][f"G{groups}_tsbg"]
        need(int(unseen["base_cycles"][0] - unseen["base_cycles_low"][0]) ==
             ordinary["median"] - ordinary["minimum"] and
             int(unseen["tsbg_cycles_high"][0] - unseen["tsbg_cycles"][0]) ==
             tsbg["maximum"] - tsbg["median"],
             "unseen min/median/max separation")
        representative[f"G{groups}"] = {
            "seen_exact": True, "unseen_median": True,
            "unseen_min_max_interval": True}

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
        try:
            imported.strict_json(duplicate)
        except imported.M2145Error:
            duplicate_rejected = True
        else:
            duplicate_rejected = False
        nonfinite = root / "nonfinite.json"
        nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
        try:
            imported.strict_json(nonfinite)
        except imported.M2145Error:
            nonfinite_rejected = True
        else:
            nonfinite_rejected = False
        duplicate.unlink()
        nonfinite.unlink()
        member = root / "member"
        member.write_text("sealed\n", encoding="ascii")
        (root / "SHA256SUMS").write_text(
            f"{sha(member)}  member\n", encoding="ascii")
        (root / "SHA256SUMS.seal.sha256").write_text(
            f"{sha(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="ascii")
        imported.verify_double_seal(root)
        (root / "extra").write_text("attack\n", encoding="ascii")
        try:
            imported.verify_double_seal(root)
        except imported.M2145Error:
            extra_rejected = True
        else:
            extra_rejected = False
        (root / "extra").unlink()
        dangling = root / "dangling"
        dangling.symlink_to(root / "absent")
        try:
            imported.verify_double_seal(root)
        except imported.M2145Error:
            dangling_rejected = True
        else:
            dangling_rejected = False
        dangling.unlink()
        (root / "empty_dir").mkdir()
        try:
            imported.verify_double_seal(root)
        except imported.M2145Error:
            directory_rejected = True
        else:
            directory_rejected = False

    text = SOURCE.read_text(encoding="utf-8")
    need(all(token not in text for token in (
        "import subprocess", "import socket", "Popen(", "os.system(")),
        "execution tooling surface")
    need("1 <= workers <= 3" in text and "set_num_threads(workers)" in text,
         "worker bound")
    order = [text.index(fragment) for fragment in (
        "need(not os.path.lexists(str(output))",
        "tempfile.mkdtemp(prefix=\".m2145_stage.\"",
        "(stage / \"result.json\").write_text",
        "(stage / \"SHA256SUMS\").write_text",
        "(stage / \"SHA256SUMS.seal.sha256\").write_text",
        "os.replace(stage, output)")]
    need(order == sorted(order), "atomic output construction order")
    need("shutil.rmtree(stage, ignore_errors=True)" in text,
         "staging cleanup")
    need(not any(HW.glob("results/m2158*")), "M2158 already exists")
    need(duplicate_rejected and nonfinite_rejected and extra_rejected,
         "core fail-closed mutations")
    return {
        "continuation_boundary": representative,
        "duplicate_json_rejected": duplicate_rejected,
        "nonfinite_json_rejected": nonfinite_rejected,
        "unlisted_regular_file_rejected": extra_rejected,
        "dangling_symlink_rejected": dangling_rejected,
        "unlisted_empty_directory_rejected": directory_rejected,
        "worker_limit": 3,
        "atomic_same_parent_staging_then_replace": True,
        "production_result_absent": True,
        "execution_tool_calls": 0,
    }


def main() -> int:
    imported = module()
    result = {
        "status": "PASS_M2157_MECHANICAL_SOURCE_HAMMER",
        "identity": identity_and_seals(imported),
        "calibration": exhaustive_calibration(imported),
        "population": population_and_header_scan(imported),
        "semantics": semantics_and_mutations(imported),
        "review_execution": {
            "production_replays": 0, "payloads_decompressed": 0,
            "vcs_compiles": 0, "simv_runs": 0, "eda_runs": 0,
            "gpu_runs": 0, "license_queries": 0,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
