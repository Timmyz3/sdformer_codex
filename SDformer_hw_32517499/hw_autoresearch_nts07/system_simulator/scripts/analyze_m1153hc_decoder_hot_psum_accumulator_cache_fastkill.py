#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Exact D0 hot-psum accumulator-cache CPU fast-kill.

This is a read-only diagnostic.  It replays the frozen canonical mapper and
never opens the live M1111DR2 work directory for writing.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from decimal import Decimal, getcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any

getcontext().prec = 40
sys.dont_write_bytecode = True
HW = Path(__file__).resolve().parents[2]
CONTRACT = HW / "contracts/m1153hc_decoder_hot_psum_accumulator_cache_fastkill_contract_r1_20260830.json"
OUT = HW / "results/m1153hc_decoder_hot_psum_accumulator_cache_fastkill_r1_20260830"
ENTRIES = (1, 2, 4, 8, 16, 32, 64)
FULL_LINE_BITS = 96 * 24
METADATA_BITS = 16
FIXED_RESOURCE_BYTES = 13_824 + 221_184 + 8_192
TOTAL_SRAM_BYTES = 245_760
KINDS = ("FA_LRU", "DIRECT_MAPPED")


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


@dataclass
class Port:
    banks: int
    latency: int
    outstanding_limit: int

    def __post_init__(self) -> None:
        self.next_issue = [0] * self.banks
        self.outstanding = [[] for _ in range(self.banks)]

    def issue(self, banks: tuple[int, ...], earliest: int, beats: int = 1,
              latency: int | None = None) -> int:
        initial = max([earliest] + [self.next_issue[bank] for bank in banks])
        issue = initial
        changed = True
        while changed:
            changed = False
            for bank in banks:
                occupied = sorted(value for value in self.outstanding[bank]
                                  if value > issue)
                if len(occupied) >= self.outstanding_limit:
                    proposed = occupied[len(occupied) - self.outstanding_limit]
                    if proposed > issue:
                        issue = proposed; changed = True
        returned = issue + (self.latency if latency is None else latency) + beats - 1
        for bank in banks:
            self.next_issue[bank] = issue + beats
            self.outstanding[bank] = [value for value in self.outstanding[bank]
                                      if value > issue] + [returned]
        return returned


class Candidate:
    def __init__(self, organization: str, capacity: int):
        self.organization = organization; self.capacity = capacity
        self.external = Port(1, 32, 16)
        self.psum = Port(1, 2, 8)  # six physical banks move one full vector together
        self.compute_next = 0
        self.cache = OrderedDict() if organization == "FA_LRU" else [None] * capacity
        self.backing: dict[int, int] = {}
        self.backing_ready: dict[int, int] = {}
        self.hits = self.misses = self.evictions = self.flushes = 0
        self.fills = self.sram_reads = self.sram_writes = 0
        self.compute_events = self.descriptor_events = self.commit_events = 0
        self.end_cycle = 0; self.flush_mismatches = 0

    def descriptor(self) -> int:
        self.descriptor_events += 1
        return self.external.issue((0,), 0, 1, 32)

    def _evict(self, entry) -> None:
        if entry is None:
            return
        key, value, ready, dirty = entry
        if dirty:
            returned = self.psum.issue((0,), ready, 1, 1)
            self.sram_writes += 1; self.evictions += 1
            self.backing[key] = value; self.backing_ready[key] = returned

    def _lookup(self, key: int):
        if self.organization == "FA_LRU":
            if key in self.cache:
                value = self.cache.pop(key); self.cache[key] = value
                return True, value
            return False, None
        index = key % self.capacity; entry = self.cache[index]
        if entry is not None and entry[0] == key:
            return True, entry[1:]
        return False, entry

    def _install(self, key: int, value: int, ready: int, dirty: bool) -> None:
        if self.organization == "FA_LRU":
            if len(self.cache) >= self.capacity:
                victim_key, victim = self.cache.popitem(last=False)
                self._evict((victim_key,) + victim)
            self.cache[key] = (value, ready, dirty)
        else:
            index = key % self.capacity; victim = self.cache[index]
            if victim is not None and victim[0] != key:
                self._evict(victim)
            self.cache[index] = (key, value, ready, dirty)

    def update(self, key: int, desc_ready: int, weight_ready: int) -> None:
        hit, state = self._lookup(key)
        if hit:
            self.hits += 1
            value, cache_ready, _dirty = state
            ready = cache_ready
        else:
            self.misses += 1; self.fills += 1; self.sram_reads += 1
            # Install may evict through the same 1RW backing port first.
            if self.organization == "FA_LRU" and len(self.cache) >= self.capacity:
                victim_key, victim = self.cache.popitem(last=False)
                self._evict((victim_key,) + victim)
            elif self.organization == "DIRECT_MAPPED":
                index = key % self.capacity; victim = self.cache[index]
                if victim is not None and victim[0] != key:
                    self._evict(victim); self.cache[index] = None
            ready = self.psum.issue((0,), self.backing_ready.get(key, 0), 1, 2)
            value = self.backing.get(key, 0)
        issue = max(self.compute_next, desc_ready, weight_ready, ready)
        compute_ready = issue + 1; self.compute_next = issue + 1
        self.compute_events += 1
        if self.organization == "FA_LRU":
            if hit:
                self.cache[key] = (value + 1, compute_ready, True)
            else:
                self.cache[key] = (value + 1, compute_ready, True)
        else:
            self.cache[key % self.capacity] = (key, value + 1, compute_ready, True)
        self.end_cycle = max(self.end_cycle, compute_ready)

    def finish_timestep(self, truth: dict[int, int], all_keys: range) -> None:
        entries = ([(key,) + value for key, value in self.cache.items()]
                   if self.organization == "FA_LRU"
                   else [entry for entry in self.cache if entry is not None])
        for key, value, ready, dirty in sorted(entries, key=lambda item: (item[2], item[0])):
            if dirty:
                returned = self.psum.issue((0,), ready, 1, 1)
                self.sram_writes += 1; self.flushes += 1
                self.backing[key] = value; self.backing_ready[key] = returned
        if self.backing != truth:
            self.flush_mismatches += 1
        for key in all_keys:
            dependency = self.backing_ready.get(key, 0)
            returned = self.external.issue((0,), dependency, 2, 3)
            self.commit_events += 1; self.end_cycle = max(self.end_cycle, returned + 1)
        self.cache = OrderedDict() if self.organization == "FA_LRU" else [None] * self.capacity
        self.backing.clear(); self.backing_ready.clear()

    def receipt(self, baseline_cycles: int, updates: int) -> dict[str, Any]:
        data_bits = self.capacity * FULL_LINE_BITS
        metadata_bits = self.capacity * METADATA_BITS
        cache_bytes = math.ceil((data_bits + metadata_bits) / 8)
        baseline_ops = 2 * updates
        candidate_ops = self.sram_reads + self.sram_writes
        return {
            "organization": self.organization, "entries": self.capacity,
            "hits": self.hits, "misses": self.misses,
            "hit_rate": ratio(self.hits, self.hits + self.misses),
            "fills": self.fills, "dirty_evictions": self.evictions,
            "terminal_flush_writes": self.flushes,
            "backing_sram_reads": self.sram_reads,
            "backing_sram_writes": self.sram_writes,
            "baseline_backing_rmw_operations": baseline_ops,
            "candidate_backing_operations": candidate_ops,
            "backing_operation_reduction_fraction": ratio(baseline_ops - candidate_ops, baseline_ops),
            "cycles": self.end_cycle,
            "baseline_over_candidate_speedup": ratio(baseline_cycles, self.end_cycle),
            "cache_data_bits": data_bits, "cache_metadata_bits": metadata_bits,
            "cache_total_bits": data_bits + metadata_bits,
            "cache_total_bytes_ceil": cache_bytes,
            "fits_24kib_cache_gate": cache_bytes <= 24 * 1024,
            "fits_unallocated_240kib_slack": FIXED_RESOURCE_BYTES + cache_bytes <= TOTAL_SRAM_BYTES,
            "total_sram_bytes_if_additive": FIXED_RESOURCE_BYTES + cache_bytes,
            "flush_mismatches": self.flush_mismatches,
            "exact_flush_correct": self.flush_mismatches == 0,
        }


def baseline_update(state: dict[str, Any], key: int,
                    desc_ready: int, weight_ready: int) -> None:
    previous = state["ready"].get(key, 0)
    read_ready = state["psum"].issue((0,), previous, 1, 2)
    issue = max(state["compute_next"], desc_ready, weight_ready, read_ready)
    compute_ready = issue + 1; state["compute_next"] = issue + 1
    write_ready = state["psum"].issue((0,), compute_ready, 1, 1)
    state["ready"][key] = write_ready; state["updates"] += 1
    state["end"] = max(state["end"], write_ready)


def baseline_finish(state: dict[str, Any], all_keys: range) -> None:
    for key in all_keys:
        returned = state["external"].issue((0,), state["ready"].get(key, 0), 2, 3)
        state["commits"] += 1; state["end"] = max(state["end"], returned + 1)
    state["ready"].clear()


def seal(directory: Path) -> dict[str, str]:
    members = sorted(path for path in directory.iterdir() if path.is_file() and
                     path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(sha256(path), path.name)
                                for path in members), encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text("{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer)}


def main() -> int:
    require(len(sys.argv) == 1 and not OUT.exists(), "zero arguments/fresh output only")
    started = time.monotonic()
    contract = strict_json_bytes(CONTRACT.read_bytes())
    require(contract["schema"] ==
            "m1153hc_decoder_hot_psum_accumulator_cache_fastkill_contract_r1_v1",
            "contract schema")
    auth = contract["authorities"]
    for relative, expected in (
        (auth["m1111dr2_runner"], auth["m1111dr2_runner_sha256"]),
        (auth["m1111dr2_contract"], auth["m1111dr2_contract_sha256"]),
        (auth["m1105dr2_source"], auth["m1105dr2_source_sha256"]),
        (auth["mapper"], auth["mapper_sha256"]),
        ("docs/359_DATE终局冻结_20260813.md", auth["docs359_sha256"]),
    ):
        require(sha256(HW / relative) == expected, "authority drift: " + relative)
    require(sha256(HW / "results/m1152_decoder_lbfuse_live_prefix_fastkill_r1_20260830/SHA256SUMS.seal.sha256") ==
            auth["m1152_result_outer_sha256"], "M1152 identity drift")
    frozen = contract["frozen_call"]
    live = HW / frozen["live_file"]
    with live.open("rb") as stream:
        first_raw = stream.readline()
    require(len(first_raw) == frozen["line_bytes"] and
            hashlib.sha256(first_raw).hexdigest() == frozen["line_sha256"],
            "frozen completed D0 row drift")
    frozen_row = strict_json_bytes(first_raw)
    require(frozen_row["global_call_ordinal"] == 0 and
            frozen_row["diagnostic_cycles"] == frozen["diagnostic_cycles"] and
            frozen_row["kind_summaries"]["psum_read"]["count"] ==
                frozen["exact_psum_rmw_updates"], "frozen D0 row projection drift")

    source = load_module(HW / auth["m1105dr2_source"],
                         auth["m1105dr2_source_sha256"], "m1153hc_m1105")
    mapper = load_module(HW / auth["mapper"], auth["mapper_sha256"], "m1153hc_mapper")
    canonical = source.build_canonical()
    call = canonical["calls"][0]
    require(call["global_ordinal"] == 0 and call["module_ordinal"] == 0 and
            call["sequence"] == "interlaken_01_a", "canonical D0 identity drift")
    payload = Path(canonical["trust_root"]["canonical_payload"]) / call["payload_relative_path"]
    require(sha256(payload) == call["payload_sha256"], "D0 payload drift")

    candidates = [Candidate(kind, entries) for kind in KINDS for entries in ENTRIES]
    baseline = {"external": Port(1, 32, 16), "psum": Port(1, 2, 8),
                "compute_next": 0, "ready": {}, "updates": 0,
                "commits": 0, "end": 0}
    weight = Port(8, 4, 8)
    cin, cout, _hin, _win, hout, wout = (1536, 384, 15, 20, 30, 40)
    output_blocks = cout // 96; all_keys = range(hout * wout * output_blocks)
    reference_distance = Counter(); total_updates = 0
    for timestep in range(10):
        truth: dict[int, int] = {}; last_position: dict[int, int] = {}; position = 0
        for tile in mapper.iter_polyphase_tiles(
                payload, tuple(call["input_shape"]), tile_m=256,
                trusted_root=Path(canonical["trust_root"]["canonical_payload"]).resolve()):
            values = tile["values"][timestep]
            for local_m, (dy, dx) in enumerate(zip(tile["destination_y"],
                                                   tile["destination_x"])):
                destination = int(dy) * wout + int(dx)
                active = [int(value) for value in values[local_m].nonzero()[0]]
                groups = []
                queues = [[] for _ in range(8)]
                for value in active:
                    queues[(value % cin) % 8].append(value)
                for index in range(max((len(queue) for queue in queues), default=0)):
                    groups.append(tuple(queue[index] for queue in queues if index < len(queue)))
                for output_block in range(output_blocks):
                    key = destination * output_blocks + output_block
                    for group in groups:
                        if key not in last_position:
                            reference_distance["cold"] += 1
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
                            reference_distance[bucket] += 1
                        last_position[key] = position; position += 1
                        truth[key] = truth.get(key, 0) + 1; total_updates += 1
                        weight_banks = tuple((value % cin) % 8 for value in group)
                        weight_ready = weight.issue(weight_banks, 0, 1)
                        baseline_desc = baseline["external"].issue((0,), 0, 1, 32)
                        baseline_update(baseline, key, baseline_desc, weight_ready)
                        for candidate in candidates:
                            candidate.update(key, candidate.descriptor(), weight_ready)
        baseline_finish(baseline, all_keys)
        for candidate in candidates:
            candidate.finish_timestep(truth, all_keys)

    require(total_updates == frozen["exact_psum_rmw_updates"] and
            baseline["updates"] == total_updates and baseline["commits"] == 48_000 and
            baseline["end"] == frozen["diagnostic_cycles"],
            "independent same-resource baseline mismatch: updates={}/{} commits={} cycles={}/{}".format(
                total_updates, frozen["exact_psum_rmw_updates"], baseline["commits"],
                baseline["end"], frozen["diagnostic_cycles"]))
    rows = [candidate.receipt(baseline["end"], total_updates)
            for candidate in candidates]
    eligible = [row for row in rows if row["fits_24kib_cache_gate"] and
                row["fits_unallocated_240kib_slack"] and
                Decimal(row["baseline_over_candidate_speedup"]) >= Decimal("1.20") and
                row["exact_flush_correct"]]
    best = max(eligible, key=lambda row: Decimal(row["baseline_over_candidate_speedup"]),
               default=None)
    result = {
        "schema": "m1153hc_decoder_hot_psum_accumulator_cache_fastkill_result_r1_v1",
        "status": ("GO_CPU_ONLY__DIFFERENT_AUTHOR_HAMMER_BEFORE_RTL"
                   if best else "KILL_NO_RTL__CAPACITY_OR_SPEED_GATE_FAILED"),
        "identity": {"contract_sha256": sha256(CONTRACT),
            "frozen_call_line_sha256": frozen["line_sha256"],
            "payload_sha256": call["payload_sha256"],
            "mapper_sha256": auth["mapper_sha256"],
            "checkpoint": "H67_ep35", "final_checkpoint_rebind_required": True},
        "population": {"sequence": "interlaken_01_a", "sample": 0,
            "layer": "D0", "timesteps": 10, "updates": total_updates,
            "dense_commits": baseline["commits"], "partial_population": True},
        "first_principles_width_correction": {
            "m1111_transaction_width_288_unit": "bytes",
            "full_96_lane_acc24_line_bits": FULL_LINE_BITS,
            "full_line_bytes": FULL_LINE_BITS // 8,
            "literal_288_bit_line_lanes": 12,
            "literal_288_bit_entry_is_exact_full_vector": False},
        "baseline": {"name": "M1111DR2_A1_SOURCE_ORDER_SIX_BANK_1RW",
            "cycles_frozen": frozen["diagnostic_cycles"],
            "cycles_independently_reproduced": baseline["end"],
            "cycle_mismatch": 0, "psum_rmw_updates": total_updates,
            "backing_sram_operations": 2 * total_updates,
            "offchip_weak_baseline_used": False},
        "reference_distance_histogram": dict(sorted(reference_distance.items())),
        "resource": {"total_sram_bytes": TOTAL_SRAM_BYTES,
            "fixed_existing_partition_bytes": FIXED_RESOURCE_BYTES,
            "unallocated_slack_bytes": TOTAL_SRAM_BYTES - FIXED_RESOURCE_BYTES,
            "cache_budget_gate_bytes": 24 * 1024,
            "backing_psum": "six-bank 1RW, read latency 2, write latency 1"},
        "dse": rows,
        "decision": {"minimum_local_speedup": "1.200000000000",
            "eligible_points": len(eligible), "selected": best,
            "rtl_authorized_now": False,
            "different_author_hammer_required_before_rtl": True},
        "correctness": {"all_points_zero_flush_mismatch":
            all(row["flush_mismatches"] == 0 for row in rows),
            "symbolic_update_count_preserved_per_destination_timestep": True,
            "terminal_flush_before_dense_commit": True},
        "runtime": {"wall_seconds": time.monotonic() - started},
        "claim_boundary": contract["claim_boundary"],
        "related_work_position": "Gustavson sparse-accumulator / hot-psum-cache mechanism migrated to H67 source-order K3/S2 decoder under the frozen 96-lane Acc24 and 240-KiB resource."
    }
    OUT.mkdir()
    (OUT / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True,
        allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(result["status"] + "\n", encoding="utf-8")
    sealed = seal(OUT)
    print(json.dumps({"status": result["status"], "selected": best,
                      "seal": sealed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
