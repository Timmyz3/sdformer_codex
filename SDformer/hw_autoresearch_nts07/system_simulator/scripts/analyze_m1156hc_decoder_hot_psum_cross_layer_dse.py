#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1156HC one-sample D0-D3 exact hot-psum cross-layer CPU replay."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, getcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import sys
import time
from typing import Any

getcontext().prec = 40
sys.dont_write_bytecode = True
HW = Path(__file__).resolve().parents[2]
CONTRACT = HW / "contracts/m1156hc_decoder_hot_psum_cross_layer_dse_contract_r1_20260830.json"
OUT = HW / "results/m1156hc_decoder_hot_psum_cross_layer_dse_r1_20260830"
GEOMETRY = {
    0: (1536, 384, 15, 20, 30, 40),
    1: (770, 192, 30, 40, 60, 80),
    2: (386, 96, 60, 80, 120, 160),
    3: (194, 96, 120, 160, 240, 320),
}


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_bytes(payload: bytes) -> Any:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key")
            output[key] = value
        return output
    return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def load_module(path: Path, expected_sha: str, name: str):
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink() and
            sha256(path) == expected_sha, "module identity drift: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module import spec")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ratio(numerator: int, denominator: int) -> str:
    require(denominator > 0, "ratio denominator")
    return format(Decimal(numerator) / Decimal(denominator), ".12f")


def replay_layer(model, mapper, canonical: dict[str, Any], call: dict[str, Any],
                 frozen: dict[str, Any]) -> dict[str, Any]:
    module = int(call["module_ordinal"]); cin, cout, _hin, _win, hout, wout = GEOMETRY[module]
    output_blocks = math.ceil(cout / 96); all_keys = range(hout * wout * output_blocks)
    payload = Path(canonical["trust_root"]["canonical_payload"]) / call["payload_relative_path"]
    require(call["payload_relative_path"] == frozen["payload_relative_path"] and
            call["payload_sha256"] == frozen["payload_sha256"] and
            sha256(payload) == frozen["payload_sha256"], "layer payload identity drift")
    candidate = model.Candidate("DIRECT_MAPPED", 1)
    baseline = {"external": model.Port(1, 32, 16), "psum": model.Port(1, 2, 8),
                "compute_next": 0, "ready": {}, "updates": 0,
                "commits": 0, "end": 0}
    weight = model.Port(8, 4, 8)
    distance_histogram = Counter(); total_updates = 0; source_terms = 0
    phase_checks = 0; phase_mismatches = 0
    for timestep in range(10):
        truth: dict[int, int] = {}; last_position: dict[int, int] = {}; position = 0
        for tile in mapper.iter_polyphase_tiles(
                payload, tuple(call["input_shape"]), tile_m=256,
                trusted_root=Path(canonical["trust_root"]["canonical_payload"]).resolve()):
            phase = int(tile["phase_bank"]); values = tile["values"][timestep]
            for local_m, (dy, dx) in enumerate(zip(tile["destination_y"],
                                                   tile["destination_x"])):
                dy = int(dy); dx = int(dx); destination = dy * wout + dx
                phase_checks += 1
                if phase != ((dy & 1) * 2 + (dx & 1)):
                    phase_mismatches += 1
                active = [int(value) for value in values[local_m].nonzero()[0]]
                queues = [[] for _ in range(8)]
                for value in active:
                    queues[(value % cin) % 8].append(value)
                groups = [tuple(queue[index] for queue in queues if index < len(queue))
                          for index in range(max((len(queue) for queue in queues), default=0))]
                source_terms += len(active) * output_blocks
                for output_block in range(output_blocks):
                    key = destination * output_blocks + output_block
                    for group in groups:
                        if key not in last_position:
                            distance_histogram["cold"] += 1
                        else:
                            distance = position - last_position[key] - 1
                            if distance == 0: bucket = "0"
                            elif distance == 1: bucket = "1"
                            elif distance <= 3: bucket = "2_3"
                            elif distance <= 7: bucket = "4_7"
                            elif distance <= 15: bucket = "8_15"
                            elif distance <= 31: bucket = "16_31"
                            elif distance <= 63: bucket = "32_63"
                            elif distance <= 255: bucket = "64_255"
                            else: bucket = "256_plus"
                            distance_histogram[bucket] += 1
                        last_position[key] = position; position += 1
                        truth[key] = truth.get(key, 0) + 1; total_updates += 1
                        weight_banks = tuple((value % cin) % 8 for value in group)
                        weight_ready = weight.issue(weight_banks, 0, 1)
                        baseline_desc = baseline["external"].issue((0,), 0, 1, 32)
                        model.baseline_update(baseline, key, baseline_desc, weight_ready)
                        candidate.update(key, candidate.descriptor(), weight_ready)
        model.baseline_finish(baseline, all_keys)
        candidate.finish_timestep(truth, all_keys)
    require(total_updates == frozen["updates"] and baseline["updates"] == total_updates and
            baseline["commits"] == frozen["commits"] and
            baseline["end"] == frozen["baseline_cycles"],
            "{} baseline mismatch updates={}/{} commits={}/{} cycles={}/{}".format(
                frozen["layer"], total_updates, frozen["updates"], baseline["commits"],
                frozen["commits"], baseline["end"], frozen["baseline_cycles"]))
    point = candidate.receipt(baseline["end"], total_updates)
    require(phase_mismatches == 0 and point["flush_mismatches"] == 0 and
            point["organization"] == "DIRECT_MAPPED" and point["entries"] == 1 and
            point["cache_total_bytes_ceil"] == 290,
            "layer phase/cache/flush invariant drift")
    return {"layer": frozen["layer"], "module_ordinal": module,
            "baseline_cycles": baseline["end"], "candidate_cycles": point["cycles"],
            "local_speedup": point["baseline_over_candidate_speedup"],
            "updates": total_updates, "source_terms": source_terms,
            "commits": baseline["commits"], "hits": point["hits"],
            "misses": point["misses"], "hit_rate": point["hit_rate"],
            "reference_distance_histogram": dict(sorted(distance_histogram.items())),
            "backing_sram_operations_baseline": point["baseline_backing_rmw_operations"],
            "backing_sram_operations_candidate": point["candidate_backing_operations"],
            "backing_operation_reduction_fraction": point["backing_operation_reduction_fraction"],
            "fills": point["fills"], "dirty_evictions": point["dirty_evictions"],
            "terminal_flush_writes": point["terminal_flush_writes"],
            "flush_mismatches": point["flush_mismatches"],
            "phase_checks": phase_checks, "phase_mismatches": phase_mismatches,
            "cache_total_bits": point["cache_total_bits"],
            "cache_total_bytes_ceil": point["cache_total_bytes_ceil"],
            "fits_unallocated_240kib_slack": point["fits_unallocated_240kib_slack"],
            "baseline_cycle_mismatch": 0}


def main() -> int:
    require(len(sys.argv) == 1 and not OUT.exists(), "zero arguments/fresh output only")
    started = time.monotonic(); contract = strict_json_bytes(CONTRACT.read_bytes())
    require(contract["schema"] ==
            "m1156hc_decoder_hot_psum_cross_layer_dse_contract_r1_v1", "contract schema")
    frozen_sources = contract["frozen_sources"]
    for relative, expected in ((contract["authorization"]["m1155hc_hammer"] +
                                "/SHA256SUMS.seal.sha256",
                                contract["authorization"]["m1155hc_outer_seal_file_sha256"]),
                               (frozen_sources["m1153hc_source"], frozen_sources["m1153hc_source_sha256"]),
                               (frozen_sources["m1105dr2_source"], frozen_sources["m1105dr2_source_sha256"]),
                               (frozen_sources["mapper"], frozen_sources["mapper_sha256"]),
                               ("docs/359_DATE终局冻结_20260813.md", frozen_sources["docs359_sha256"])):
        require(sha256(HW / relative) == expected, "authority drift: " + relative)
    hammer = strict_json_bytes((HW / contract["authorization"]["m1155hc_hammer"] /
                                "review.json").read_bytes())
    require(hammer["status"] == contract["authorization"]["m1155hc_status"] and
            hammer["authorization"]["d0_d1_d2_d3_one_call_each_cross_layer_cpu_replay_next"] is True,
            "M1155HC authorization drift")
    live = HW / contract["input"]["live_jsonl"]
    raw_rows = []
    with live.open("rb") as stream:
        for _ in range(4):
            raw_rows.append(stream.readline())
    frozen_rows = []
    for expected, (raw, identity) in enumerate(zip(raw_rows, contract["frozen_calls"])):
        require(raw.endswith(b"\n") and len(raw) == identity["line_bytes"] and
                hashlib.sha256(raw).hexdigest() == identity["line_sha256"],
                "frozen call line drift")
        row = strict_json_bytes(raw)
        require(row["global_call_ordinal"] == expected and row["module_ordinal"] == expected and
                row["sequence"] == "interlaken_01_a" and row["sequence_sample_id"] == 0 and
                row["diagnostic_cycles"] == identity["baseline_cycles"],
                "frozen call coordinate drift")
        frozen_rows.append(row)
    model = load_module(HW / frozen_sources["m1153hc_source"],
                        frozen_sources["m1153hc_source_sha256"], "m1156hc_model")
    source = load_module(HW / frozen_sources["m1105dr2_source"],
                         frozen_sources["m1105dr2_source_sha256"], "m1156hc_source")
    mapper = load_module(HW / frozen_sources["mapper"], frozen_sources["mapper_sha256"],
                         "m1156hc_mapper")
    canonical = source.build_canonical()
    layers = []
    for ordinal, identity in enumerate(contract["frozen_calls"]):
        layers.append(replay_layer(model, mapper, canonical, canonical["calls"][ordinal], identity))
    baseline_total = sum(row["baseline_cycles"] for row in layers)
    candidate_total = sum(row["candidate_cycles"] for row in layers)
    all_pass = all(row["baseline_cycle_mismatch"] == 0 and
                   row["flush_mismatches"] == 0 and
                   Decimal(row["local_speedup"]) >= Decimal("1.20")
                   for row in layers)
    result = {
        "schema": "m1156hc_decoder_hot_psum_cross_layer_dse_result_r1_v1",
        "status": ("GO_CPU_ONLY_CROSS_LAYER__DIFFERENT_AUTHOR_HAMMER_REQUIRED"
                   if all_pass else "DOWNGRADE_CROSS_LAYER_GATE_FAILED__NO_RTL"),
        "identity": {"contract_sha256": sha256(CONTRACT),
            "sequence": "interlaken_01_a", "sequence_sample_id": 0,
            "checkpoint": "H67_ep35", "call_line_sha256":
                [row["line_sha256"] for row in contract["frozen_calls"]],
            "payload_sha256": [row["payload_sha256"] for row in contract["frozen_calls"]],
            "final_checkpoint_rebind_required": True},
        "candidate": {"organization": "DIRECT_MAPPED", "entries": 1,
            "key": ["timestep", "destination", "output_block"],
            "phase_derived_from_destination_parity": True,
            "data_bits": 2304, "metadata_bits": 16, "bytes_ceil": 290},
        "layers": layers,
        "aggregate": {"baseline_cycles_sum": baseline_total,
            "candidate_cycles_sum": candidate_total,
            "four_layer_decoder_only_speedup": ratio(baseline_total, candidate_total),
            "weighting": "sum baseline cycles divided by sum candidate cycles",
            "system_speedup": False, "multi_sequence_extrapolation": False},
        "decision": {"all_four_layers_at_least_1p20x": all_pass,
            "all_baselines_zero_cycle_mismatch":
                all(row["baseline_cycle_mismatch"] == 0 for row in layers),
            "all_flush_mismatch_zero": all(row["flush_mismatches"] == 0 for row in layers),
            "rtl_authorized_now": False,
            "different_author_hammer_required": True},
        "runtime": {"wall_seconds": time.monotonic() - started},
        "claim_boundary": contract["claim_boundary"]}
    OUT.mkdir()
    (OUT / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True,
        allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(result["status"] + "\n", encoding="utf-8")
    sealed = model.seal(OUT)
    print(json.dumps({"status": result["status"], "layers": [{"layer": row["layer"],
        "speedup": row["local_speedup"]} for row in layers],
        "decoder_only_speedup": result["aggregate"]["four_layer_decoder_only_speedup"],
        "seal": sealed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
