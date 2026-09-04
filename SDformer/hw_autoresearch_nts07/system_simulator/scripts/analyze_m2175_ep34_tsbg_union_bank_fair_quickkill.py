#!/opt/anaconda3/bin/python
"""M2175 fair CPU quick-kill for B4-union selective bank fill.

This model changes only the bank-fill mask on a frozen M2145 recurrence.  The
ordinary token-major and TSBG group-major schedules receive the *same* B4
union mask, four cache rows, one request port, response latency, issue stalls,
and commit schedule.  Therefore it cannot confuse a read-only opportunity
count with a schedule speedup.

The masked successor has no VCS or mapped implementation.  Its continuation
wrapper residual is inherited, axis by axis, from the corresponding frozen
dense-fill replay.  Every output consequently remains CPU-model-only and is
explicitly barred from paper, RTL, same-area, energy, component-speedup, and
system-speedup claims.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import zlib

import numpy as np
from numba import njit, prange, set_num_threads


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
BASE_SOURCE = HW / "system_simulator/scripts/analyze_m2145_ep34_tsbg_fulltoken_calibrated_replay.py"
CONTRACT = HW / "contracts/m2175_ep34_tsbg_union_bank_fair_cpu_quickkill_contract_r1_20260904.json"
CAPTURE = HW / "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901"
BASE_RESULT = HW / "results/m2158_m2145_ep34_tsbg_fulltoken_calibrated_replay_r1_20260904"

EXPECTED = {
    BASE_SOURCE: "1dc41d29ad7a0b7e175e5cd5f379c60b1320fbe08d4ff374c9d6e6ebeb37db57",
    CAPTURE / "fc_frames.bin": "dceb6c0c80b9c5898d10b4ad813fbcd7683fa80191b54b78eadaadda04a818b1",
    CAPTURE / "SHA256SUMS": "be0b89f9b8084baf0c2cd959805530a4e4f41c437a446335142a65bd73960a8f",
    CAPTURE / "SHA256SUMS.seal.sha256": "8d63d1054452377836c333c9848f771a6fc964e4ed4b00ed9a22f1537bd73c85",
    BASE_RESULT / "result.json": "7fd86d70db03f02900b02175b71417360527e3670c939aa7b5133f69a10db292",
    BASE_RESULT / "SHA256SUMS": "8923398927b5155ee1f4120dce073fdf4928bb9eaf7b8d292910830aafcf0f5e",
    BASE_RESULT / "SHA256SUMS.seal.sha256": "610b4fa37cd99d3a0b251cd152453647964669a79af2df4b54fbec4ab1ff9649",
}

FRAME_HEADER = struct.Struct("<8sHH11I")
CONTEXTS = 4
PHYSICAL_GROUPS = 48
SOURCES = 16
SLICES = 6
CACHE_ROWS = 4
START = 383
LOAD_CYCLES = 768
EXPECTED_QUARTETS = 11_160_000
EXPECTED_DENSE_ORDINARY_CYCLES = 313_603_627_826
EXPECTED_DENSE_TSBG_CYCLES = 150_234_338_522
EXPECTED_DENSE_TSBG_READS = 67_992_387_648


class M2175Error(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise M2175Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path: Path, digest: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise M2175Error("missing input: " + str(path)) from exc
    need(stat.S_ISREG(mode) and not path.is_symlink(),
         "input must be regular non-symlink: " + str(path))
    need(sha256(path) == digest, "identity drift: " + str(path))


def strict_json(path: Path) -> object:
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          M2175Error("nonfinite JSON: " + token)))


def verify_double_seal(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    tokens = outer.read_text(encoding="ascii").split()
    need(tokens == [sha256(manifest), "SHA256SUMS"],
         "outer seal mismatch: " + str(directory))
    listed = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe seal member")
        exact(directory / rel, digest)
        listed.add(rel.as_posix())
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == listed, "non-exhaustive seal: " + str(directory))


def load_base():
    spec = importlib.util.spec_from_file_location("m2145_frozen", BASE_SOURCE)
    need(spec is not None and spec.loader is not None, "cannot load frozen recurrence")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_inputs(require_contract: bool) -> dict:
    for path, digest in EXPECTED.items():
        exact(path, digest)
    verify_double_seal(CAPTURE)
    verify_double_seal(BASE_RESULT)
    contract_sha = None
    if require_contract:
        contract = strict_json(CONTRACT)
        need(contract["source"]["path"] == str(SOURCE.relative_to(ROOT)) and
             contract["source"]["sha256"] == sha256(SOURCE),
             "contract/source binding mismatch")
        side = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
        outer = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256.seal.sha256")
        need(side.read_text(encoding="ascii").split() ==
             [sha256(CONTRACT), CONTRACT.name], "contract sidecar mismatch")
        need(outer.read_text(encoding="ascii").split() ==
             [sha256(side), side.name], "contract outer seal mismatch")
        contract_sha = sha256(CONTRACT)
    return {"source_sha256": sha256(SOURCE),
            "base_source_sha256": sha256(BASE_SOURCE),
            "capture_sha256": sha256(CAPTURE / "fc_frames.bin"),
            "base_result_sha256": sha256(BASE_RESULT / "result.json"),
            "contract_sha256": contract_sha}


@njit(cache=False, parallel=True)
def masked_batch_engine_cycles(lower: np.ndarray, upper: np.ndarray,
                               union_mask: np.ndarray, mode: int) -> np.ndarray:
    """Fair recurrence; columns: cycles/hit/miss/evict/live/issue/read/beat."""
    population = lower.shape[0]
    groups = lower.shape[2]
    out = np.zeros((population, 8), dtype=np.int64)
    for item in prange(population):
        valid = np.zeros(CACHE_ROWS, dtype=np.uint8)
        group_at = np.zeros(CACHE_ROWS, dtype=np.int64)
        age = np.zeros(CACHE_ROWS, dtype=np.int64)
        clock = 1
        cycle = START
        hits = misses = evictions = live = issues = reads = beats = 0
        for position in range(CONTEXTS * groups):
            if mode == 0:
                context = position // groups
                group = position % groups
            else:
                group = position // CONTEXTS
                context = position % CONTEXTS
            lo = lower[item, context, group]
            hi = upper[item, context, group]
            if not lo and not hi:
                continue
            live += 1
            hit = -1
            for index in range(CACHE_ROWS):
                if valid[index] and group_at[index] == group:
                    hit = index
                    break
            miss = hit < 0
            if miss:
                misses += 1
                victim = -1
                for index in range(CACHE_ROWS):
                    if not valid[index]:
                        victim = index
                        break
                if victim < 0:
                    victim = 0
                    for index in range(1, CACHE_ROWS):
                        if (age[index] < age[victim] or
                                (age[index] == age[victim] and index < victim)):
                            victim = index
                    evictions += 1
                valid[victim] = 1
                group_at[victim] = group
                age[victim] = clock + 1
            else:
                hits += 1
                age[hit] = clock
            clock += 1
            cycle += 1  # ST_FIND
            if miss:
                mask16 = int(union_mask[item, group])
                for half in range(2):
                    mask8 = (mask16 >> (half * 8)) & 255
                    if mask8 == 0:
                        continue
                    pop = 0
                    for bank in range(8):
                        if (mask8 >> bank) & 1:
                            pop += 1
                    for _slice in range(SLICES):
                        latest = 0
                        for bank in range(8):
                            if ((mask8 >> bank) & 1) == 0:
                                continue
                            accepted = cycle
                            while (accepted + 1 + bank * 2) % 7 == 0:
                                accepted += 1
                            completion = accepted + (8 - bank) + 1
                            if completion > latest:
                                latest = completion
                        cycle = latest + 1
                        reads += pop
                        beats += 1
            count = SLICES * (int(lo) + int(hi))
            issues += count
            for _issue in range(count):
                while cycle % 11 == 3:
                    cycle += 1
                cycle += 1
        cycle += 1  # empty ST_FIND -> commit
        for _commit in range(CONTEXTS * SLICES):
            while cycle % 13 == 5:
                cycle += 1
            cycle += 1
        out[item, 0] = cycle - START
        out[item, 1] = hits
        out[item, 2] = misses
        out[item, 3] = evictions
        out[item, 4] = live
        out[item, 5] = issues
        out[item, 6] = reads
        out[item, 7] = beats
    return out


def union_masks(quartets: np.ndarray) -> np.ndarray:
    active = np.any(quartets != 0, axis=1)
    powers = (np.uint16(1) << np.arange(SOURCES, dtype=np.uint16))
    return np.sum(active.astype(np.uint16) * powers[None, None, :],
                  axis=2, dtype=np.uint16)


def masked_replay(quartets: np.ndarray, layer: dict, base_batch: dict,
                  base) -> dict:
    groups = int(layer["weight_layout"]["source_group_count"])
    tiles = int(layer["weight_layout"]["output_tile_count"])
    lower = np.any(quartets[:, :, :, :8] != 0, axis=3)
    upper = np.any(quartets[:, :, :, 8:] != 0, axis=3)
    masks = union_masks(quartets)
    population = quartets.shape[0]
    masked_backbones = []
    dense_backbones = []
    masked_reads = []
    for mode in (0, 1):
        masked_backbone = np.zeros(population, dtype=np.int64)
        dense_backbone = np.zeros(population, dtype=np.int64)
        reads = np.zeros(population, dtype=np.int64)
        for begin in range(0, groups, PHYSICAL_GROUPS):
            end = min(begin + PHYSICAL_GROUPS, groups)
            masked = masked_batch_engine_cycles(
                lower[:, :, begin:end], upper[:, :, begin:end],
                masks[:, begin:end], mode)
            dense = base.batch_engine_cycles(
                lower[:, :, begin:end], upper[:, :, begin:end], mode)
            masked_backbone += LOAD_CYCLES + masked[:, 0]
            dense_backbone += LOAD_CYCLES + dense[:, 0]
            reads += masked[:, 6] * tiles
        masked_backbones.append(masked_backbone * tiles)
        dense_backbones.append(dense_backbone * tiles)
        masked_reads.append(reads)
    # Same axis-specific wrapper residual is added to dense and masked paths.
    ordinary_residual = base_batch["base_cycles"] - dense_backbones[0]
    tsbg_residual = base_batch["tsbg_cycles"] - dense_backbones[1]
    return {
        "ordinary_cycles": masked_backbones[0] + ordinary_residual,
        "tsbg_cycles": masked_backbones[1] + tsbg_residual,
        "ordinary_reads": masked_reads[0],
        "tsbg_reads": masked_reads[1],
    }


class Totals:
    def __init__(self, keep_ratios: bool = False) -> None:
        self.rows = 0
        self.dense_o_cycles = self.dense_t_cycles = 0
        self.mask_o_cycles = self.mask_t_cycles = 0
        self.dense_o_reads = self.dense_t_reads = 0
        self.mask_o_reads = self.mask_t_reads = 0
        self.mask_slower_than_ordinary = 0
        self.mask_slower_than_dense_tsbg = 0
        self.worst_mask_schedule_ratio = float("inf")
        self.worst_mask_vs_dense_tsbg_ratio = float("inf")
        self.ratios = array("d") if keep_ratios else None

    def add(self, dense: dict, masked: dict) -> None:
        n = int(dense["base_cycles"].size)
        self.rows += n
        for attr, values in (
                ("dense_o_cycles", dense["base_cycles"]),
                ("dense_t_cycles", dense["tsbg_cycles"]),
                ("mask_o_cycles", masked["ordinary_cycles"]),
                ("mask_t_cycles", masked["tsbg_cycles"]),
                ("dense_o_reads", dense["base_scalar_reads"]),
                ("dense_t_reads", dense["tsbg_scalar_reads"]),
                ("mask_o_reads", masked["ordinary_reads"]),
                ("mask_t_reads", masked["tsbg_reads"])):
            setattr(self, attr, getattr(self, attr) + int(np.sum(values, dtype=np.int64)))
        schedule = masked["ordinary_cycles"] / masked["tsbg_cycles"]
        vs_dense = dense["tsbg_cycles"] / masked["tsbg_cycles"]
        self.mask_slower_than_ordinary += int(np.count_nonzero(
            masked["tsbg_cycles"] > masked["ordinary_cycles"]))
        self.mask_slower_than_dense_tsbg += int(np.count_nonzero(
            masked["tsbg_cycles"] > dense["tsbg_cycles"]))
        self.worst_mask_schedule_ratio = min(
            self.worst_mask_schedule_ratio, float(np.min(schedule)))
        self.worst_mask_vs_dense_tsbg_ratio = min(
            self.worst_mask_vs_dense_tsbg_ratio, float(np.min(vs_dense)))
        if self.ratios is not None:
            self.ratios.frombytes(np.ascontiguousarray(schedule, dtype=np.float64).tobytes())

    def result(self) -> dict:
        need(self.rows > 0, "empty totals")
        out = {
            "aligned_b4_quartets": self.rows,
            "dense_fill": {
                "ordinary_cycles": self.dense_o_cycles,
                "tsbg_cycles": self.dense_t_cycles,
                "ratio_of_sums": self.dense_o_cycles / self.dense_t_cycles,
                "ordinary_scalar_bank_reads": self.dense_o_reads,
                "tsbg_scalar_bank_reads": self.dense_t_reads,
            },
            "mask_aware_fair": {
                "ordinary_cycles": self.mask_o_cycles,
                "tsbg_cycles": self.mask_t_cycles,
                "ratio_of_sums": self.mask_o_cycles / self.mask_t_cycles,
                "ordinary_scalar_bank_reads": self.mask_o_reads,
                "tsbg_scalar_bank_reads": self.mask_t_reads,
                "tsbg_read_reduction_vs_dense_tsbg": 1.0 - self.mask_t_reads / self.dense_t_reads,
                "tsbg_cycle_change_vs_dense_tsbg": self.mask_t_cycles / self.dense_t_cycles - 1.0,
                "slower_than_mask_ordinary_cases": self.mask_slower_than_ordinary,
                "slower_than_mask_ordinary_rate": self.mask_slower_than_ordinary / self.rows,
                "slower_than_dense_tsbg_cases": self.mask_slower_than_dense_tsbg,
                "slower_than_dense_tsbg_rate": self.mask_slower_than_dense_tsbg / self.rows,
                "worst_workload_ratio_vs_mask_ordinary": self.worst_mask_schedule_ratio,
                "worst_workload_ratio_vs_dense_tsbg": self.worst_mask_vs_dense_tsbg_ratio,
            },
        }
        if self.ratios is not None:
            values = np.frombuffer(self.ratios, dtype=np.float64)
            out["mask_aware_fair"].update({
                "p10_workload_ratio": float(np.percentile(values, 10)),
                "p50_workload_ratio": float(np.percentile(values, 50)),
                "p90_workload_ratio": float(np.percentile(values, 90)),
            })
        return out


def selftest() -> dict:
    identity = verify_inputs(False)
    base = load_base()
    rng = np.random.default_rng(2175)
    quartets = np.zeros((7, CONTEXTS, PHYSICAL_GROUPS, SOURCES), dtype=np.int8)
    pick = rng.random(quartets.shape) < 0.11
    quartets[pick] = rng.choice(np.array([-1, 1], dtype=np.int8), int(pick.sum()))
    lower = np.any(quartets[:, :, :, :8] != 0, axis=3)
    upper = np.any(quartets[:, :, :, 8:] != 0, axis=3)
    full = np.full((quartets.shape[0], PHYSICAL_GROUPS), 0xffff, dtype=np.uint16)
    for mode in (0, 1):
        dense = base.batch_engine_cycles(lower, upper, mode)
        masked = masked_batch_engine_cycles(lower, upper, full, mode)
        need(bool(np.array_equal(dense[:, :7], masked[:, :7])),
             "full-mask recurrence does not reproduce dense recurrence")
    masks = union_masks(quartets)
    for context in range(CONTEXTS):
        active = quartets[:, context] != 0
        for lane in range(SOURCES):
            need(bool(np.all(((masks >> lane) & 1) >= active[:, :, lane])),
                 "union mask dropped live source")
    all_zero = np.zeros((1, CONTEXTS, PHYSICAL_GROUPS, SOURCES), dtype=np.int8)
    zero_masks = union_masks(all_zero)
    out = masked_batch_engine_cycles(
        np.zeros((1, CONTEXTS, PHYSICAL_GROUPS), dtype=bool),
        np.zeros((1, CONTEXTS, PHYSICAL_GROUPS), dtype=bool), zero_masks, 1)
    need(int(out[0, 6]) == 0, "zero descriptor issued a bank read")
    return {"status": "PASS_M2175_STATIC_FAIR_RECURRENCE_SELFTEST",
            "identity": identity, "synthetic_quartets": 8,
            "dense_full_mask_fields_reproduced": 98,
            "ordinary_and_tsbg_receive_same_union_mask": True,
            "production_frames_decoded": 0, "rtl_runs": 0,
            "eda_runs": 0, "gpu_jobs": 0}


def run(output: Path, workers: int, max_frames: int | None) -> dict:
    need(1 <= workers <= 3, "workers must be in [1,3]")
    need(max_frames is None or max_frames > 0, "max_frames must be positive")
    need(not os.path.lexists(str(output)), "fresh output required")
    set_num_threads(workers)
    identity = verify_inputs(True)
    base = load_base()
    calibrated = base.calibration()
    layers_payload = strict_json(CAPTURE / "layers.json")
    samples_payload = strict_json(CAPTURE / "sample_order.json")
    layers = [row for row in layers_payload["layers"] if row["target"] in ("FC1", "FC2")]
    samples = samples_payload["samples"]
    layer_by_id = {int(row["layer_id"]): row for row in layers}
    sample_by_id = {int(row["global_sample_id"]): row for row in samples}
    pairs = [(int(sample["global_sample_id"]), int(layer["layer_id"]))
             for sample in samples for layer in layers]
    aggregate = Totals(True)
    targets = {name: Totals() for name in ("FC1", "FC2")}
    sequences = {name: Totals() for name in sorted({row["sequence"] for row in samples})}
    layer_stats = {str(row["layer_id"]): Totals() for row in layers}
    frames = pair_index = frame_expected = token_expected = 0
    with (CAPTURE / "fc_frames.bin").open("rb") as stream:
        while max_frames is None or frames < max_frames:
            prefix = stream.read(FRAME_HEADER.size)
            if not prefix:
                break
            need(len(prefix) == FRAME_HEADER.size, "truncated frame header")
            (magic, version, header_size, layer_id, sample_id, frame_index,
             token_start, token_count, channels, bitrow, nnz_total, raw_bytes,
             compressed_bytes, crc32) = FRAME_HEADER.unpack(prefix)
            need(pair_index < len(pairs) and (sample_id, layer_id) == pairs[pair_index] and
                 magic == base.FRAME_MAGIC and version == base.FRAME_VERSION and
                 header_size == FRAME_HEADER.size and frame_index == frame_expected and
                 token_start == token_expected and token_count % CONTEXTS == 0,
                 "canonical frame order/identity")
            layer = layer_by_id[layer_id]
            compressed = stream.read(compressed_bytes)
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed) + decoder.flush()
            need(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and
                 len(raw) == raw_bytes and (zlib.crc32(raw) & 0xffffffff) == crc32,
                 "frame zlib/CRC")
            dense_values = base.decode_payload(raw, token_count, channels, bitrow, nnz_total)
            groups = channels // SOURCES
            quartets = dense_values.reshape(token_count // CONTEXTS, CONTEXTS, groups, SOURCES)
            dense = base.replay_batch(quartets, layer, calibrated, sample_id, token_start)
            masked = masked_replay(quartets, layer, dense, base)
            aggregate.add(dense, masked)
            targets[layer["target"]].add(dense, masked)
            sequence = sample_by_id[sample_id]["sequence"]
            sequences[sequence].add(dense, masked)
            layer_stats[str(layer_id)].add(dense, masked)
            frames += 1
            token_expected += token_count
            if token_expected == int(layer["tokens_per_call"]):
                pair_index += 1
                frame_expected = token_expected = 0
            else:
                frame_expected += 1
    full = max_frames is None
    if full:
        need(pair_index == len(pairs) and token_expected == 0 and
             aggregate.rows == EXPECTED_QUARTETS, "full replay incomplete")
    agg = aggregate.result()
    if full:
        need(agg["dense_fill"]["ordinary_cycles"] == EXPECTED_DENSE_ORDINARY_CYCLES and
             agg["dense_fill"]["tsbg_cycles"] == EXPECTED_DENSE_TSBG_CYCLES and
             agg["dense_fill"]["tsbg_scalar_bank_reads"] == EXPECTED_DENSE_TSBG_READS,
             "frozen dense replay did not reproduce M2158")
    gates = {
        "mask_aware_tsbg_vs_mask_aware_ordinary_ratio_of_sums_ge_1p5":
            agg["mask_aware_fair"]["ratio_of_sums"] >= 1.5,
        "mask_aware_tsbg_read_reduction_vs_dense_tsbg_ge_30pct":
            agg["mask_aware_fair"]["tsbg_read_reduction_vs_dense_tsbg"] >= 0.30,
        "mask_aware_tsbg_cycle_degradation_vs_dense_tsbg_le_2pct":
            agg["mask_aware_fair"]["tsbg_cycle_change_vs_dense_tsbg"] <= 0.02,
    }
    result = {
        "schema": "m2175_ep34_tsbg_union_bank_fair_cpu_quickkill_result_r1_v1",
        "status": ("CPU_PILOT_PASS_NOT_GATE_EVIDENCE" if not full else
                   ("GO_CPU_QUICKKILL_ONLY_PENDING_INDEPENDENT_REVIEW_DO_NOT_CITE"
                    if all(gates.values()) else
                    "NO_GO_CPU_QUICKKILL_PENDING_INDEPENDENT_REVIEW_DO_NOT_CITE")),
        "identity": identity,
        "population": {"full_capture": full, "frames": frames,
                       "aligned_b4_quartets": aggregate.rows,
                       "checkpoint": "motion_ep34_live93"},
        "fairness": {
            "mask_definition": "OR of exact nonzero source lanes over the same aligned B4 quartet",
            "same_mask_for_ordinary_and_tsbg": True,
            "same_cache_rows": 4, "same_request_ports": 1,
            "same_per_bank_accept_stall_and_response_latency": True,
            "same_issue_and_commit_recurrence": True,
            "cache_row_semantics": "B4 union fully covers all four contexts for that group",
            "continuation_residual": "same frozen dense-fill axis residual added to masked successor",
        },
        "aggregate": agg,
        "breakdown": {
            "target": {k: v.result() for k, v in targets.items() if v.rows},
            "sequence": {k: v.result() for k, v in sequences.items() if v.rows},
            "layer_id": {k: v.result() for k, v in layer_stats.items() if v.rows},
        },
        "decision_gates": gates,
        "claim_boundary": {
            "read_only_frozen_capture": True, "cpu_cycle_model": True,
            "complete_replay": full, "pilot_is_gate_evidence": False,
            "rtl": False, "vcs": False, "same_area": False,
            "paper_result": False, "component_speedup_admitted": False,
            "system_speedup": False, "energy": False, "power": False,
            "headline": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".m2175_stage.", dir=output.parent))
    try:
        (stage / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
        (stage / "RUN_COMPLETE.txt").write_text(result["status"] + "\n", encoding="ascii")
        members = ("RUN_COMPLETE.txt", "result.json")
        (stage / "SHA256SUMS").write_text(
            "".join(f"{sha256(stage / name)}  {name}\n" for name in members), encoding="ascii")
        (stage / "SHA256SUMS.seal.sha256").write_text(
            f"{sha256(stage / 'SHA256SUMS')}  SHA256SUMS\n", encoding="ascii")
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--static", action="store_true")
    action.add_argument("--selftest", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.static:
        need(args.output is None and args.max_frames is None, "static takes no output")
        print(json.dumps({"status": "PASS_M2175_STATIC_INPUTS",
                          "identity": verify_inputs(False)}, sort_keys=True))
    elif args.selftest:
        need(args.output is None and args.max_frames is None, "selftest takes no output")
        print(json.dumps(selftest(), sort_keys=True))
    else:
        need(args.output is not None, "--run requires --output")
        result = run(args.output.resolve(), args.workers, args.max_frames)
        print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                          "frames": result["population"]["frames"],
                          "quartets": result["population"]["aligned_b4_quartets"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
