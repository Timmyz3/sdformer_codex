#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author read-only first-D0 replay for M1153HC.

The subject Candidate/Port classes are never imported.  This implementation
uses only the frozen canonical decoder mapper and independently models the
M1111 baseline and one-entry direct writeback accumulator cache.  It writes no
artifact, does not alter the live producer, and invokes no RTL/VCS/DC flow.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, getcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any

getcontext().prec = 40
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/analyze_m1153hc_decoder_hot_psum_accumulator_cache_fastkill.py"
CONTRACT = HW / "contracts/m1153hc_decoder_hot_psum_accumulator_cache_fastkill_contract_r1_20260830.json"
RESULT = HW / "results/m1153hc_decoder_hot_psum_accumulator_cache_fastkill_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "6b1d07aa39e8e25bc8c80ea434ee604fea56382de2741d9006e6ee5a682d7c8c",
    "contract": "491c48b87ef85bafbb119507a98ff40be381049d24e039cbcb04df9d13ca8522",
    "contract_side": "95260919200f5f9afaeb813f9b4866de4843d75814ec79adfc713abe81e3b2cb",
    "contract_outer": "f6f95c1709765e88885e264a8a5a8f665e776f2be9f6e64a62a826fbe632deea",
    "result_manifest": "0ff980278d0a461ab069fce29d6f8bc7685ad97b2770030a5bf58180dc1db4c3",
    "result_outer": "f44747b40fe4a1a126f8051f559e7b1dbf8017f9eae3cc22dc97c9928cff955e",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
AXES = ("timestep", "destination", "output_block")
checks = 0


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise HammerFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink() and
            sha256(path) == expected, "identity drift: " + str(path))


def strict_json_bytes(payload: bytes) -> Any:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key")
            output[key] = value
        return output
    return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerFailure("nonfinite JSON: " + token)))


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "contract double seal content")


def verify_result() -> dict[str, Any]:
    manifest = RESULT / "SHA256SUMS"; outer = RESULT / "SHA256SUMS.seal.sha256"
    verify_regular(manifest, EXPECTED["result_manifest"])
    verify_regular(outer, EXPECTED["result_outer"])
    require(outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["result_manifest"], "SHA256SUMS"], "result outer content")
    listed = {}
    for row in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = row.split(maxsplit=1); name = name.lstrip("*")
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
                name not in listed and Path(name).name == name, "result manifest row")
        listed[name] = digest
    actual = {path.name for path in RESULT.iterdir() if path.is_file() and
              path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(listed) == {"RUN_COMPLETE.txt", "report.json"},
            "result exact member set")
    for name, digest in listed.items():
        verify_regular(RESULT / name, digest)
    return strict_json_bytes((RESULT / "report.json").read_bytes())


def load_module(path: Path, expected_sha: str, name: str):
    verify_regular(path, expected_sha)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module import spec")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class IndependentPort:
    banks: int
    latency: int
    outstanding_limit: int

    def __post_init__(self) -> None:
        self.next_issue = [0] * self.banks
        self.outstanding = [[] for _ in range(self.banks)]
        self.calls = 0

    def issue(self, banks: tuple[int, ...], earliest: int,
              beats: int = 1, latency: int | None = None) -> int:
        require(banks and len(banks) == len(set(banks)) and
                all(type(bank) is int and 0 <= bank < self.banks for bank in banks),
                "independent port banks")
        issue = max([earliest] + [self.next_issue[bank] for bank in banks])
        while True:
            proposed = issue
            for bank in banks:
                active = sorted(value for value in self.outstanding[bank] if value > issue)
                if len(active) >= self.outstanding_limit:
                    proposed = max(proposed, active[len(active) - self.outstanding_limit])
            if proposed == issue:
                break
            issue = proposed
        returned = issue + (self.latency if latency is None else latency) + beats - 1
        for bank in banks:
            self.next_issue[bank] = issue + beats
            self.outstanding[bank] = [value for value in self.outstanding[bank]
                                      if value > issue] + [returned]
        self.calls += 1
        return returned


def ratio(numerator: int, denominator: int) -> str:
    require(denominator > 0, "ratio denominator")
    return format(Decimal(numerator) / Decimal(denominator), ".12f")


def frozen_first_row(contract: dict[str, Any]) -> dict[str, Any]:
    frozen = contract["frozen_call"]
    live = HW / frozen["live_file"]
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(live, flags)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), "live schedule is not regular")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.readline(65537)
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
                "live schedule fd replacement")
    finally:
        os.close(fd)
    require(len(raw) == frozen["line_bytes"] and raw.endswith(b"\n") and
            hashlib.sha256(raw).hexdigest() == frozen["line_sha256"],
            "frozen first D0 line drift")
    return strict_json_bytes(raw)


def replay(contract: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    auth = contract["authorities"]
    source = load_module(HW / auth["m1105dr2_source"], auth["m1105dr2_source_sha256"],
                         "m1155hc_m1105")
    mapper = load_module(HW / auth["mapper"], auth["mapper_sha256"],
                         "m1155hc_mapper")
    canonical = source.build_canonical(); call = canonical["calls"][0]
    require((call["global_ordinal"], call["module_ordinal"], call["sequence"],
             call["sequence_sample_id"]) == (0, 0, "interlaken_01_a", 0),
            "canonical first D0 identity")
    payload = Path(canonical["trust_root"]["canonical_payload"]) / call["payload_relative_path"]
    verify_regular(payload, call["payload_sha256"])

    baseline_external = IndependentPort(1, 32, 16)
    candidate_external = IndependentPort(1, 32, 16)
    weight_port = IndependentPort(8, 4, 8)
    baseline_psum = IndependentPort(1, 2, 8)
    candidate_psum = IndependentPort(1, 2, 8)
    baseline_compute_next = 0; candidate_compute_next = 0
    baseline_end = 0; candidate_end = 0
    baseline_updates = candidate_updates = 0
    baseline_commits = candidate_commits = 0
    candidate_hits = candidate_fills = candidate_writes = 0
    dirty_evictions = terminal_flush_writes = flush_mismatches = 0
    accumulator_dependency_checks = 0
    reference_distance = Counter(); previous_key = None; event_position = 0
    unique_keys = set(); phase_by_key: dict[tuple[int, int, int], int] = {}
    output_blocks_seen = set(); timesteps_seen = set(); phases_seen = set()
    key_without_output_block = set(); key_without_timestep = set(); key_with_phase = set()
    source_terms = groups_before_output_blocks = 0
    descriptor_baseline = descriptor_candidate = weight_events = compute_baseline = compute_candidate = 0

    cin, cout, hout, wout = 1536, 384, 30, 40
    output_blocks = cout // 96
    require(output_blocks == 4, "D0 output-block geometry")
    for timestep in range(10):
        timesteps_seen.add(timestep)
        truth: dict[tuple[int, int, int], int] = {}
        baseline_ready: dict[tuple[int, int, int], int] = {}
        candidate_backing: dict[tuple[int, int, int], int] = {}
        candidate_backing_ready: dict[tuple[int, int, int], int] = {}
        cache = None  # (key, value, ready, dirty)
        timestep_keys = set()
        for tile in mapper.iter_polyphase_tiles(
                payload, tuple(call["input_shape"]), tile_m=256,
                trusted_root=Path(canonical["trust_root"]["canonical_payload"]).resolve()):
            phase = int(tile["phase_bank"]); phases_seen.add(phase)
            values = tile["values"][timestep]
            for local_m, (dy_raw, dx_raw) in enumerate(zip(
                    tile["destination_y"], tile["destination_x"])):
                dy, dx = int(dy_raw), int(dx_raw)
                destination = dy * wout + dx
                require(phase == ((dy & 1) << 1 | (dx & 1)),
                        "phase is not encoded by destination parity")
                active = [int(value) for value in values[local_m].nonzero()[0]]
                queues = [[] for _ in range(8)]
                for value in active:
                    queues[(value % cin) % 8].append(value)
                group_count = max((len(queue) for queue in queues), default=0)
                groups = [tuple(queue[index] for queue in queues if index < len(queue))
                          for index in range(group_count)]
                require(all(group and len(group) <= 8 and
                            len({(value % cin) % 8 for value in group}) == len(group)
                            for group in groups), "K8 source grouping drift")
                groups_before_output_blocks += len(groups)
                for output_block in range(output_blocks):
                    output_blocks_seen.add(output_block)
                    key = (timestep, destination, output_block)
                    timestep_keys.add(key); unique_keys.add(key)
                    key_without_output_block.add((timestep, destination))
                    key_without_timestep.add((destination, output_block))
                    key_with_phase.add((timestep, phase, destination, output_block))
                    if key in phase_by_key:
                        require(phase_by_key[key] == phase, "one destination key spans phases")
                    else:
                        phase_by_key[key] = phase
                    for group in groups:
                        source_terms += len(group)
                        if previous_key is None or key != previous_key:
                            reference_distance["cold"] += 1
                        else:
                            reference_distance["0"] += 1
                        previous_key = key; event_position += 1
                        truth[key] = truth.get(key, 0) + 1

                        banks = tuple((value % cin) % 8 for value in group)
                        weight_ready = weight_port.issue(banks, 0, 1)
                        weight_events += 1
                        b_desc = baseline_external.issue((0,), 0, 1, 32)
                        c_desc = candidate_external.issue((0,), 0, 1, 32)
                        descriptor_baseline += 1; descriptor_candidate += 1

                        b_previous = baseline_ready.get(key, 0)
                        b_read = baseline_psum.issue((0,), b_previous, 1, 2)
                        b_issue = max(baseline_compute_next, b_desc, weight_ready, b_read)
                        b_compute = b_issue + 1; baseline_compute_next = b_issue + 1
                        b_write = baseline_psum.issue((0,), b_compute, 1, 1)
                        baseline_ready[key] = b_write; baseline_updates += 1
                        compute_baseline += 1; baseline_end = max(baseline_end, b_write)

                        if cache is not None and cache[0] == key:
                            candidate_hits += 1
                            _, c_value, c_ready, _dirty = cache
                        else:
                            candidate_fills += 1
                            if cache is not None:
                                victim_key, victim_value, victim_ready, victim_dirty = cache
                                if victim_dirty:
                                    wb = candidate_psum.issue((0,), victim_ready, 1, 1)
                                    candidate_writes += 1; dirty_evictions += 1
                                    candidate_backing[victim_key] = victim_value
                                    candidate_backing_ready[victim_key] = wb
                            c_ready = candidate_psum.issue(
                                (0,), candidate_backing_ready.get(key, 0), 1, 2)
                            c_value = candidate_backing.get(key, 0)
                        c_issue = max(candidate_compute_next, c_desc, weight_ready, c_ready)
                        require(c_issue >= c_ready, "cache-hit accumulator dependency bypassed")
                        accumulator_dependency_checks += 1
                        c_compute = c_issue + 1; candidate_compute_next = c_issue + 1
                        cache = (key, c_value + 1, c_compute, True)
                        candidate_updates += 1; compute_candidate += 1
                        candidate_end = max(candidate_end, c_compute)

        require(len(timestep_keys) == hout * wout * output_blocks,
                "timestep destination/output-block population")
        if cache is not None and cache[3]:
            key, value, ready, _dirty = cache
            wb = candidate_psum.issue((0,), ready, 1, 1)
            candidate_writes += 1; terminal_flush_writes += 1
            candidate_backing[key] = value; candidate_backing_ready[key] = wb
        if candidate_backing != truth:
            flush_mismatches += 1
        for destination in range(hout * wout):
            for output_block in range(output_blocks):
                key = (timestep, destination, output_block)
                b_commit = baseline_external.issue((0,), baseline_ready.get(key, 0), 2, 3)
                c_commit = candidate_external.issue((0,), candidate_backing_ready.get(key, 0), 2, 3)
                baseline_commits += 1; candidate_commits += 1
                baseline_end = max(baseline_end, b_commit + 1)
                candidate_end = max(candidate_end, c_commit + 1)

    require(reference_distance == Counter({"0": 4_417_036, "cold": 48_000}),
            "independent reuse-distance histogram")
    require(baseline_updates == candidate_updates == 4_465_036 and
            baseline_commits == candidate_commits == 48_000,
            "update/commit count")
    require(baseline_psum.calls == 8_930_072 and baseline_end == 17_863_747,
            "M1111 same-resource baseline reproduction")
    require(candidate_hits == 4_417_036 and candidate_fills == 48_000 and
            candidate_writes == 48_000 and dirty_evictions == 47_990 and
            terminal_flush_writes == 10 and candidate_psum.calls == 96_000 and
            flush_mismatches == 0, "one-entry backing traffic/exact flush")
    require(candidate_end == 9_025_999 and
            ratio(baseline_end, candidate_end) == "1.979143472097" and
            ratio(baseline_psum.calls - candidate_psum.calls, baseline_psum.calls) ==
                "0.989249806720", "candidate cycle/traffic metric")
    require(len(unique_keys) == 48_000 and len(key_without_output_block) == 12_000 and
            len(key_without_timestep) == 4_800 and len(key_with_phase) == 48_000 and
            set(phase_by_key.values()) == {0, 1, 2, 3}, "key-axis collision audit")
    require(output_blocks_seen == {0, 1, 2, 3} and timesteps_seen == set(range(10)) and
            phases_seen == {0, 1, 2, 3}, "axis coverage")
    require(source_terms > candidate_updates and groups_before_output_blocks * output_blocks ==
            candidate_updates, "source count was confused with update count")
    require(descriptor_baseline == descriptor_candidate == weight_events ==
            compute_baseline == compute_candidate == candidate_updates and
            baseline_external.calls == candidate_external.calls == candidate_updates + 48_000,
            "descriptor/weight/compute/commit fairness")
    require(accumulator_dependency_checks == candidate_updates,
            "accumulator dependency not checked per update")

    data_bits = 96 * 24; metadata_bits = 16
    total_bits = data_bits + metadata_bits; total_bytes = math.ceil(total_bits / 8)
    fixed = 13_824 + 221_184 + 8_192; total_sram = 245_760
    require(data_bits == 2_304 and data_bits // 8 == 288 and total_bits == 2_320 and
            total_bytes == 290 and 288 // 24 == 12 and fixed == 243_200 and
            fixed + total_bytes == 243_490 and fixed + total_bytes <= total_sram and
            total_sram - (fixed + total_bytes) == 2_270,
            "line unit or 240-KiB capacity accounting")

    selected = next(row for row in report["dse"]
                    if row["organization"] == "DIRECT_MAPPED" and row["entries"] == 1)
    require(selected["hits"] == candidate_hits and selected["fills"] == candidate_fills and
            selected["backing_sram_writes"] == candidate_writes and
            selected["terminal_flush_writes"] == terminal_flush_writes and
            selected["dirty_evictions"] == dirty_evictions and
            selected["cycles"] == candidate_end and
            selected["cache_total_bytes_ceil"] == total_bytes and
            selected["total_sram_bytes_if_additive"] == fixed + total_bytes,
            "sealed report selected direct row mismatch")
    return {
        "identity": {"call": "interlaken_01_a/sample0/D0/H67_ep35",
                     "payload_sha256": call["payload_sha256"]},
        "key": {"axes": list(AXES), "unique": len(unique_keys),
                "phase_is_destination_parity_redundant": True,
                "unique_without_output_block": len(key_without_output_block),
                "unique_without_timestep": len(key_without_timestep)},
        "reuse_distance": dict(reference_distance),
        "events": {"updates": candidate_updates, "source_terms": source_terms,
                   "dense_commits": candidate_commits,
                   "descriptor_events_each": descriptor_candidate,
                   "weight_events": weight_events, "compute_events_each": compute_candidate},
        "baseline": {"cycles": baseline_end, "backing_operations": baseline_psum.calls,
                     "weak_baseline": False},
        "candidate": {"hits": candidate_hits, "fills": candidate_fills,
                      "dirty_evictions": dirty_evictions,
                      "terminal_flush_writes": terminal_flush_writes,
                      "backing_writes": candidate_writes,
                      "backing_operations": candidate_psum.calls,
                      "cycles": candidate_end,
                      "local_speedup": ratio(baseline_end, candidate_end),
                      "backing_reduction_fraction": ratio(
                          baseline_psum.calls - candidate_psum.calls, baseline_psum.calls),
                      "flush_mismatches": flush_mismatches,
                      "accumulator_dependency_checks": accumulator_dependency_checks},
        "capacity": {"data_bits": data_bits, "data_bytes": data_bits // 8,
                     "metadata_bits": metadata_bits, "total_bits": total_bits,
                     "total_bytes_ceil": total_bytes, "fixed_existing_bytes": fixed,
                     "total_with_cache_bytes": fixed + total_bytes,
                     "total_sram_bytes": total_sram, "remaining_bytes": 2_270},
    }


def main() -> int:
    verify_regular(SOURCE, EXPECTED["source"])
    verify_double(CONTRACT, (EXPECTED["contract"], EXPECTED["contract_side"],
                             EXPECTED["contract_outer"]))
    verify_regular(DOCS359, EXPECTED["docs359"])
    report = verify_result()
    contract = strict_json_bytes(CONTRACT.read_bytes())
    first = frozen_first_row(contract)
    require(first["global_call_ordinal"] == 0 and first["module_ordinal"] == 0 and
            first["diagnostic_cycles"] == 17_863_747 and
            first["kind_summaries"]["psum_read"]["count"] == 4_465_036 and
            first["kind_summaries"]["psum_write"]["count"] == 4_465_036 and
            first["kind_summaries"]["output_commit"]["count"] == 48_000,
            "frozen first row projection")
    replayed = replay(contract, report)
    require(sha256(DOCS359) == EXPECTED["docs359"], "docs359 changed during hammer")
    output = {
        "status": "PASS_M1155HC_DIFFERENT_AUTHOR_FIRST_D0_REPLAY__CROSS_LAYER_ONE_CALL_EACH_NEXT_ONLY",
        "checks": checks,
        "source_sha256": EXPECTED["source"],
        "contract_identity": [EXPECTED["contract"], EXPECTED["contract_side"],
                              EXPECTED["contract_outer"]],
        "result_outer_seal_file_sha256": EXPECTED["result_outer"],
        "replay": replayed,
        "claim_boundary": {"single_d0_call": True, "old_checkpoint": True,
                           "local_cpu_cycle_model_only": True,
                           "rtl_authorized": False, "headline_authorized": False,
                           "system_speedup_admitted": False},
        "authorization": {"d0_d1_d2_d3_one_call_each_cpu_replay_next": True,
                          "rtl": False, "vcs": False, "dc": False},
    }
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
