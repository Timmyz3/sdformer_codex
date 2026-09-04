#!/opt/anaconda3/bin/python
"""Independent, CPU-only M2184 hammer for the sealed M2175 result."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import struct
import zlib

import numpy as np
from numba import set_num_threads


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_simulator/scripts/analyze_m2175_ep34_tsbg_union_bank_fair_quickkill.py"
CONTRACT = HW / "contracts/m2175_ep34_tsbg_union_bank_fair_cpu_quickkill_contract_r1_20260904.json"
RESULT = HW / "results/m2175_ep34_tsbg_union_bank_fair_cpu_quickkill_r1_20260904"
AUTHOR = HW / "reviews/m2175_ep34_tsbg_union_bank_fair_cpu_quickkill_author_receipt_r1_20260904"
CAPTURE = HW / "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901"
BASE_RESULT = HW / "results/m2158_m2145_ep34_tsbg_fulltoken_calibrated_replay_r1_20260904"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
HEADER = struct.Struct("<8sHH11I")
CONTEXTS = 4
SOURCES = 16
SLICES = 6
CACHE_ROWS = 4
START = 383
EXPECTED_SOURCE = "121deceb266af0414781c0dcbae1f66f4007785c06c6508e55267d883abb3020"
EXPECTED_CONTRACT = "8b55349515873615ec31b3e42ec9e9d2dcc605a85016ddadeff6a097d72c3039"
EXPECTED_RESULT = "8dcd05075b57bf2bc5610d75d3ad6477f8ff1451215fad27e8e1eed8b1fdb65c"
EXPECTED_RESULT_MANIFEST = "5e49c0798fb28e98d21c33e7286c04b69e55600bf74596eaf18e21376555b56f"
EXPECTED_RESULT_OUTER = "2a4f81e22ffbc47bd7a8cbaf3060d4d14ba5e68c8b66291aaf3c42d5709414b8"
FALSE_CLAIMS = {"rtl", "vcs", "same_area", "paper_result",
                "component_speedup_admitted", "system_speedup", "energy",
                "power", "headline"}


class Failure(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, f"duplicate JSON key {key}")
            value[key] = item
        return value
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure(f"nonfinite JSON {token}")))
    need(isinstance(value, dict), f"JSON root not object {path}")
    return value


def exact_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    need(stat.S_ISREG(mode) and not path.is_symlink(), f"not regular {path}")
    need(sha(path) == expected, f"SHA drift {path}")


def verify_seal(directory: Path) -> dict[str, str]:
    need(directory.is_dir() and not directory.is_symlink(), f"bad directory {directory}")
    need(not any(path.is_symlink() for path in directory.rglob("*")),
         f"symlink in {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    outer_tokens = outer.read_text().split()
    need(outer_tokens == [sha(manifest), "SHA256SUMS"], f"outer seal {directory}")
    listed: set[str] = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        rel = Path(name)
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             f"unsafe/duplicate member {name}")
        exact_regular(directory / rel, digest)
        listed.add(name)
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == listed, f"non-exhaustive seal {directory}")
    return {"manifest_sha256": sha(manifest), "outer_sha256": sha(outer),
            "members": len(listed)}


def close(actual: float, expected: float, label: str) -> None:
    need(math.isclose(actual, expected, rel_tol=2e-15, abs_tol=2e-15),
         f"arithmetic mismatch {label}: {actual} != {expected}")


def validate_stats(stats: dict, label: str) -> None:
    dense = stats["dense_fill"]
    masked = stats["mask_aware_fair"]
    close(dense["ordinary_cycles"] / dense["tsbg_cycles"],
          dense["ratio_of_sums"], label + ".dense_ratio")
    close(masked["ordinary_cycles"] / masked["tsbg_cycles"],
          masked["ratio_of_sums"], label + ".masked_ratio")
    close(1.0 - masked["tsbg_scalar_bank_reads"] / dense["tsbg_scalar_bank_reads"],
          masked["tsbg_read_reduction_vs_dense_tsbg"], label + ".read_reduction")
    close(masked["tsbg_cycles"] / dense["tsbg_cycles"] - 1.0,
          masked["tsbg_cycle_change_vs_dense_tsbg"], label + ".cycle_change")
    close(masked["slower_than_mask_ordinary_cases"] / stats["aligned_b4_quartets"],
          masked["slower_than_mask_ordinary_rate"], label + ".slow_rate")
    close(masked["slower_than_dense_tsbg_cases"] / stats["aligned_b4_quartets"],
          masked["slower_than_dense_tsbg_rate"], label + ".dense_slow_rate")


def sum_children(children: dict[str, dict], aggregate: dict, label: str) -> None:
    need(sum(row["aligned_b4_quartets"] for row in children.values()) ==
         aggregate["aligned_b4_quartets"], f"{label} quartet sum")
    for section, fields in {
        "dense_fill": ("ordinary_cycles", "tsbg_cycles", "ordinary_scalar_bank_reads",
                       "tsbg_scalar_bank_reads"),
        "mask_aware_fair": ("ordinary_cycles", "tsbg_cycles", "ordinary_scalar_bank_reads",
                            "tsbg_scalar_bank_reads", "slower_than_mask_ordinary_cases",
                            "slower_than_dense_tsbg_cases"),
    }.items():
        for field in fields:
            need(sum(row[section][field] for row in children.values()) ==
                 aggregate[section][field], f"{label}.{section}.{field} sum")


def decode(raw: bytes, tokens: int, channels: int, row_bytes: int,
           nnz_total: int) -> np.ndarray:
    matrix_bytes = tokens * row_bytes
    need(len(raw) == 3 * matrix_bytes + 2 * tokens + nnz_total, "payload extent")
    matrices = []
    offset = 0
    for _ in range(3):
        packed = np.frombuffer(raw[offset:offset + matrix_bytes], dtype=np.uint8)
        bits = np.unpackbits(packed.reshape(tokens, row_bytes), axis=1, bitorder="little")
        need(bool((bits[:, channels:] == 0).all()), "tail bits")
        matrices.append(bits[:, :channels].astype(bool))
        offset += matrix_bytes
    support, sign, nonunit = matrices
    counts = np.frombuffer(raw[offset:offset + 2 * tokens], dtype="<u2")
    offset += 2 * tokens
    codes = np.frombuffer(raw[offset:], dtype=np.int8)
    need(bool((sign <= support).all()) and bool((nonunit <= support).all()), "support semantics")
    need(bool((counts == support.sum(axis=1)).all()) and int(counts.sum()) == nnz_total,
         "support counts")
    need(codes.size == nnz_total and bool((codes != 0).all()) and
         bool(((codes < 0) == sign[support]).all()) and not bool(nonunit.any()),
         "signed descriptor semantics")
    dense = np.zeros((tokens, channels), dtype=np.int8)
    dense[support] = codes
    need(set(map(int, np.unique(dense))).issubset({-1, 0, 1}), "non-ternary source")
    return dense


def scan_capture(layers: list[dict], samples: list[dict]) -> tuple[dict, list[tuple[np.ndarray, int]]]:
    layer_by_id = {int(row["layer_id"]): row for row in layers}
    pairs = [(int(sample["global_sample_id"]), int(layer["layer_id"]))
             for sample in samples for layer in layers]
    selections = {(0, 8): 11, (10, 9): 11, (20, 16): 11, (30, 23): 11, (39, 31): 10}
    selected: list[tuple[np.ndarray, int]] = []
    pair_index = frame_index = token_start = frames = quartets = 0
    with (CAPTURE / "fc_frames.bin").open("rb") as stream:
        while True:
            prefix = stream.read(HEADER.size)
            if not prefix:
                break
            need(len(prefix) == HEADER.size and pair_index < len(pairs), "frame header/pair extent")
            (magic, version, header_size, layer_id, sample_id, observed_frame,
             observed_start, token_count, channels, row_bytes, nnz_total,
             raw_bytes, compressed_bytes, crc32) = HEADER.unpack(prefix)
            need((sample_id, layer_id) == pairs[pair_index], "canonical pair order")
            need(magic == b"M1558F01" and version == 1 and header_size == HEADER.size,
                 "frame identity")
            need(observed_frame == frame_index and observed_start == token_start and
                 token_count > 0 and token_count % CONTEXTS == 0 and channels % SOURCES == 0,
                 "frame order/geometry")
            payload = stream.read(compressed_bytes)
            need(len(payload) == compressed_bytes, "truncated compressed payload")
            if observed_frame == 0 and (sample_id, layer_id) in selections:
                decoder = zlib.decompressobj()
                raw = decoder.decompress(payload) + decoder.flush()
                need(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and
                     len(raw) == raw_bytes and (zlib.crc32(raw) & 0xffffffff) == crc32,
                     "selected frame CRC")
                values = decode(raw, token_count, channels, row_bytes, nnz_total)
                count = selections[(sample_id, layer_id)]
                shaped = values.reshape(token_count // CONTEXTS, CONTEXTS,
                                        channels // SOURCES, SOURCES)
                need(shaped.shape[0] >= count, "selected quartet extent")
                selected.append((shaped[:count].copy(), layer_id))
            frames += 1
            quartets += token_count // CONTEXTS
            token_start += token_count
            expected_tokens = int(layer_by_id[layer_id]["tokens_per_call"])
            if token_start == expected_tokens:
                pair_index += 1
                frame_index = token_start = 0
            else:
                need(token_start < expected_tokens, "token overrun")
                frame_index += 1
        need(pair_index == len(pairs) and frame_index == token_start == 0, "capture incomplete")
    need(len(selected) == len(selections) and sum(x.shape[0] for x, _ in selected) == 54,
         "scalar sample selection")
    return {"frames": frames, "quartets": quartets, "selected_frames_crc": len(selected)}, selected


def scalar_cycle(lower: np.ndarray, upper: np.ndarray, mask: np.ndarray,
                 mode: int) -> tuple[int, ...]:
    groups = lower.shape[1]
    valid = [False] * CACHE_ROWS
    group_at = [0] * CACHE_ROWS
    age = [0] * CACHE_ROWS
    clock = 1
    cycle = START
    hits = misses = evictions = live = issues = reads = beats = 0
    for position in range(CONTEXTS * groups):
        if mode == 0:
            context, group = divmod(position, groups)
        else:
            group, context = divmod(position, CONTEXTS)
        lo, hi = bool(lower[context, group]), bool(upper[context, group])
        if not lo and not hi:
            continue
        live += 1
        hit = next((i for i in range(CACHE_ROWS)
                    if valid[i] and group_at[i] == group), -1)
        miss = hit < 0
        if miss:
            misses += 1
            victim = next((i for i in range(CACHE_ROWS) if not valid[i]), -1)
            if victim < 0:
                victim = min(range(CACHE_ROWS), key=lambda i: (age[i], i))
                evictions += 1
            valid[victim] = True
            group_at[victim] = group
            age[victim] = clock + 1
        else:
            hits += 1
            age[hit] = clock
        clock += 1
        cycle += 1
        if miss:
            mask16 = int(mask[group])
            for half in range(2):
                mask8 = (mask16 >> (half * 8)) & 0xff
                if mask8 == 0:
                    continue
                active = [bank for bank in range(8) if (mask8 >> bank) & 1]
                for _ in range(SLICES):
                    latest = 0
                    for bank in active:
                        accepted = cycle
                        while (accepted + 1 + bank * 2) % 7 == 0:
                            accepted += 1
                        latest = max(latest, accepted + (8 - bank) + 1)
                    cycle = latest + 1
                    reads += len(active)
                    beats += 1
        count = SLICES * (int(lo) + int(hi))
        issues += count
        for _ in range(count):
            while cycle % 11 == 3:
                cycle += 1
            cycle += 1
    cycle += 1
    for _ in range(CONTEXTS * SLICES):
        while cycle % 13 == 5:
            cycle += 1
        cycle += 1
    return cycle - START, hits, misses, evictions, live, issues, reads, beats


def scalar_recurrence(selected: list[tuple[np.ndarray, int]]) -> dict:
    spec = importlib.util.spec_from_file_location("m2175_candidate_for_hammer", SOURCE)
    need(spec is not None and spec.loader is not None, "candidate import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    set_num_threads(1)
    quartets = fields = mismatches = 0
    layer_ids = []
    digest = hashlib.sha256()
    for values, layer_id in selected:
        values = values[:, :, :48, :]
        lower = np.any(values[:, :, :, :8] != 0, axis=3)
        upper = np.any(values[:, :, :, 8:] != 0, axis=3)
        active = np.any(values != 0, axis=1)
        powers = np.uint16(1) << np.arange(SOURCES, dtype=np.uint16)
        masks = np.sum(active.astype(np.uint16) * powers[None, None, :],
                       axis=2, dtype=np.uint16)
        for mode in (0, 1):
            candidate = module.masked_batch_engine_cycles(lower, upper, masks, mode)
            reference = np.asarray([scalar_cycle(lower[i], upper[i], masks[i], mode)
                                    for i in range(values.shape[0])], dtype=np.int64)
            mismatches += int(np.count_nonzero(candidate != reference))
            fields += int(candidate.size)
            digest.update(reference.tobytes())
        quartets += values.shape[0]
        layer_ids.append(layer_id)
    need(quartets == 54 and fields == 864 and mismatches == 0, "scalar recurrence mismatch")
    return {"actual_quartets": quartets, "fields_exact": fields, "mismatches": mismatches,
            "modes": ["ordinary", "tsbg"], "layers": layer_ids,
            "reference_digest_sha256": digest.hexdigest(), "numba_threads": 1}


def main() -> int:
    exact_regular(SOURCE, EXPECTED_SOURCE)
    exact_regular(CONTRACT, EXPECTED_CONTRACT)
    exact_regular(RESULT / "result.json", EXPECTED_RESULT)
    need(sha(RESULT / "SHA256SUMS") == EXPECTED_RESULT_MANIFEST, "result manifest identity")
    need(sha(RESULT / "SHA256SUMS.seal.sha256") == EXPECTED_RESULT_OUTER,
         "result outer identity")
    exact_regular(DOCS359, "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
    seals = {"capture": verify_seal(CAPTURE), "base_result": verify_seal(BASE_RESULT),
             "result": verify_seal(RESULT), "author": verify_seal(AUTHOR)}
    contract = strict_json(CONTRACT)
    side = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
    outer = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256.seal.sha256")
    need(side.read_text().split() == [sha(CONTRACT), CONTRACT.name], "contract sidecar")
    need(outer.read_text().split() == [sha(side), side.name], "contract outer seal")
    result = strict_json(RESULT / "result.json")
    author = strict_json(AUTHOR / "author_receipt.json")
    need(result["status"] == "GO_CPU_QUICKKILL_ONLY_PENDING_INDEPENDENT_REVIEW_DO_NOT_CITE",
         "result status")
    need(author["status"] == "PASS_AUTHOR_RECEIPT_PENDING_INDEPENDENT_M2184_HAMMER_DO_NOT_CITE",
         "author status")
    need(all(result["claim_boundary"].get(key) is False for key in FALSE_CLAIMS),
         "result false claims")
    need(all(contract["claim_boundary"].get(key) is False for key in
             {"rtl", "same_area", "energy", "paper_result", "component_speedup_admitted",
              "system_speedup", "headline"}), "contract false claims")
    need(all(author["claim_boundary"].get(key) is False for key in FALSE_CLAIMS),
         "author false claims")

    layers = [row for row in strict_json(CAPTURE / "layers.json")["layers"]
              if row["target"] in ("FC1", "FC2")]
    samples = strict_json(CAPTURE / "sample_order.json")["samples"]
    sequences = sorted({row["sequence"] for row in samples})
    need(len(layers) == 24 and sum(x["target"] == "FC1" for x in layers) == 12 and
         sum(x["target"] == "FC2" for x in layers) == 12, "FC layer census")
    need(len(samples) == 40 and len(sequences) == 4 and
         all(sum(x["sequence"] == seq for x in samples) == 10 for seq in sequences),
         "sample/sequence census")
    expected_quartets = sum(int(row["tokens_per_call"]) // CONTEXTS for row in layers) * len(samples)
    need(expected_quartets == 11_160_000, "derived quartet population")
    scan, selected = scan_capture(layers, samples)
    need(scan == {"frames": 11040, "quartets": 11_160_000, "selected_frames_crc": 5},
         "frame scan population")

    aggregate = result["aggregate"]
    validate_stats(aggregate, "aggregate")
    for axis in ("target", "sequence", "layer_id"):
        for key, stats in result["breakdown"][axis].items():
            validate_stats(stats, f"{axis}.{key}")
        sum_children(result["breakdown"][axis], aggregate, axis)
    need(set(result["breakdown"]["target"]) == {"FC1", "FC2"}, "target keys")
    need(set(result["breakdown"]["sequence"]) == set(sequences), "sequence keys")
    need(set(map(int, result["breakdown"]["layer_id"])) == set(range(8, 32)), "layer keys")
    for layer in layers:
        row = result["breakdown"]["layer_id"][str(layer["layer_id"])]
        need(row["aligned_b4_quartets"] == int(layer["tokens_per_call"]) // 4 * 40,
             f"layer population {layer['layer_id']}")

    dense = aggregate["dense_fill"]
    masked = aggregate["mask_aware_fair"]
    need((dense["ordinary_cycles"], dense["tsbg_cycles"], dense["tsbg_scalar_bank_reads"]) ==
         (313_603_627_826, 150_234_338_522, 67_992_387_648), "dense anchors")
    need((masked["ordinary_cycles"], masked["tsbg_cycles"], masked["tsbg_scalar_bank_reads"]) ==
         (244_386_356_403, 120_075_325_155, 17_316_452_106), "masked anchors")
    gates = result["decision_gates"]
    recomputed_gates = {
        "mask_aware_tsbg_vs_mask_aware_ordinary_ratio_of_sums_ge_1p5":
            masked["ratio_of_sums"] >= 1.5,
        "mask_aware_tsbg_read_reduction_vs_dense_tsbg_ge_30pct":
            masked["tsbg_read_reduction_vs_dense_tsbg"] >= 0.30,
        "mask_aware_tsbg_cycle_degradation_vs_dense_tsbg_le_2pct":
            masked["tsbg_cycle_change_vs_dense_tsbg"] <= 0.02,
    }
    need(gates == recomputed_gates and all(gates.values()), "decision gates")
    fairness = result["fairness"]
    need(fairness == {
        "mask_definition": "OR of exact nonzero source lanes over the same aligned B4 quartet",
        "same_mask_for_ordinary_and_tsbg": True, "same_cache_rows": 4,
        "same_request_ports": 1, "same_per_bank_accept_stall_and_response_latency": True,
        "same_issue_and_commit_recurrence": True,
        "cache_row_semantics": "B4 union fully covers all four contexts for that group",
        "continuation_residual": "same frozen dense-fill axis residual added to masked successor",
    }, "fairness contract")
    source_text = SOURCE.read_text()
    need("masks = union_masks(quartets)" in source_text and
         "for mode in (0, 1):" in source_text and
         "masks[:, begin:end], mode" in source_text and "CACHE_ROWS = 4" in source_text,
         "same-mask/cache source anchors")

    recurrence = scalar_recurrence(selected)
    layers_breakdown = result["breakdown"]["layer_id"]
    layer_meta = {str(row["layer_id"]): row["target"] for row in layers}
    min_ratio_layer = min(layers_breakdown, key=lambda key:
                          layers_breakdown[key]["mask_aware_fair"]["ratio_of_sums"])
    max_slow_layer = max(layers_breakdown, key=lambda key:
                         layers_breakdown[key]["mask_aware_fair"]["slower_than_mask_ordinary_rate"])
    review = {
        "schema": "m2184_m2175_ep34_tsbg_union_bank_fair_cpu_quickkill_result_hammer_r1_v1",
        "status": "PASS_M2184_M2175_CPU_QUICKKILL_HAMMER__GO_RTL_CONSIDERATION_ONLY",
        "score_over_100": 98, "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
        "identities": {"source_sha256": sha(SOURCE), "contract_sha256": sha(CONTRACT),
                       "result_sha256": sha(RESULT / "result.json"),
                       "author_receipt_sha256": sha(AUTHOR / "author_receipt.json"),
                       "docs359_sha256": sha(DOCS359)},
        "sealed_inputs": seals,
        "population": {"aligned_b4_quartets": 11_160_000, "frames": 11040,
                       "samples": 40, "sequences": 4, "layers": 24,
                       "fc1_layers": 12, "fc2_layers": 12},
        "arithmetic": {"dense_ordinary_cycles": dense["ordinary_cycles"],
                       "dense_tsbg_cycles": dense["tsbg_cycles"],
                       "dense_tsbg_scalar_reads": dense["tsbg_scalar_bank_reads"],
                       "mask_ordinary_cycles": masked["ordinary_cycles"],
                       "mask_tsbg_cycles": masked["tsbg_cycles"],
                       "mask_tsbg_scalar_reads": masked["tsbg_scalar_bank_reads"],
                       "ratio_of_sums": masked["ratio_of_sums"],
                       "read_reduction": masked["tsbg_read_reduction_vs_dense_tsbg"],
                       "cycle_change_vs_dense_tsbg": masked["tsbg_cycle_change_vs_dense_tsbg"],
                       "all_three_gates": True},
        "independent_scalar_recurrence": recurrence,
        "fairness": {"same_b4_union_mask": True, "cache_rows": 4, "request_ports": 1,
                     "same_fetch_response_latency": True, "same_issue_commit": True,
                     "continuation_residual_same_per_axis": True},
        "tails": {"fc1_ratio_of_sums": result["breakdown"]["target"]["FC1"]["mask_aware_fair"]["ratio_of_sums"],
                  "fc2_ratio_of_sums": result["breakdown"]["target"]["FC2"]["mask_aware_fair"]["ratio_of_sums"],
                  "sequence_ratio_min": min(x["mask_aware_fair"]["ratio_of_sums"] for x in result["breakdown"]["sequence"].values()),
                  "sequence_ratio_max": max(x["mask_aware_fair"]["ratio_of_sums"] for x in result["breakdown"]["sequence"].values()),
                  "slow_cases": masked["slower_than_mask_ordinary_cases"],
                  "slow_rate": masked["slower_than_mask_ordinary_rate"],
                  "worst_workload_ratio": masked["worst_workload_ratio_vs_mask_ordinary"],
                  "minimum_layer_ratio_id": int(min_ratio_layer),
                  "minimum_layer_ratio": layers_breakdown[min_ratio_layer]["mask_aware_fair"]["ratio_of_sums"],
                  "maximum_layer_slow_rate_id": int(max_slow_layer),
                  "maximum_layer_slow_rate": layers_breakdown[max_slow_layer]["mask_aware_fair"]["slower_than_mask_ordinary_rate"],
                  "slower_than_dense_tsbg_cases": masked["slower_than_dense_tsbg_cases"]},
        "sequence_table": {
            key: {"quartets": row["aligned_b4_quartets"],
                  "ratio_of_sums": row["mask_aware_fair"]["ratio_of_sums"],
                  "read_reduction": row["mask_aware_fair"]["tsbg_read_reduction_vs_dense_tsbg"],
                  "cycle_change_vs_dense_tsbg": row["mask_aware_fair"]["tsbg_cycle_change_vs_dense_tsbg"],
                  "slow_rate": row["mask_aware_fair"]["slower_than_mask_ordinary_rate"]}
            for key, row in sorted(result["breakdown"]["sequence"].items())},
        "layer_tail_table": [
            {"layer_id": int(key), "target": layer_meta[key],
             "quartets": row["aligned_b4_quartets"],
             "ratio_of_sums": row["mask_aware_fair"]["ratio_of_sums"],
             "read_reduction": row["mask_aware_fair"]["tsbg_read_reduction_vs_dense_tsbg"],
             "cycle_change_vs_dense_tsbg": row["mask_aware_fair"]["tsbg_cycle_change_vs_dense_tsbg"],
             "slow_rate": row["mask_aware_fair"]["slower_than_mask_ordinary_rate"],
             "worst_workload_ratio": row["mask_aware_fair"]["worst_workload_ratio_vs_mask_ordinary"],
             "slower_than_dense_tsbg_cases": row["mask_aware_fair"]["slower_than_dense_tsbg_cases"]}
            for key, row in sorted(layers_breakdown.items(), key=lambda item: int(item[0]))],
        "authorization": {"go_rtl_consideration": True, "paper_admit": False,
                          "rtl_result": False, "same_area": False, "energy": False,
                          "system_speedup": False, "headline": False},
        "claim_boundary": {key: False for key in sorted(FALSE_CLAIMS)},
    }
    print(json.dumps(review, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(f"M2184_FAIL_CLOSED: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
