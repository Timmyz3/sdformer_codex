#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author read-only D0-D3 numeric hammer for M1156HC.

This does not import either M1156HC or M1153HC model code.  Only the frozen
canonical workload source and exact polyphase mapper are reused.  Four fresh
independent baseline/candidate states are replayed sequentially on CPU.  This
replay intentionally audits the frozen destination-major matrix-pull order;
it does not establish that M523's source-event stream can realize that order.
"""
from __future__ import annotations

from collections import Counter
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
SOURCE = HW / "system_simulator/scripts/analyze_m1156hc_decoder_hot_psum_cross_layer_dse.py"
CONTRACT = HW / "contracts/m1156hc_decoder_hot_psum_cross_layer_dse_contract_r1_20260830.json"
RESULT = HW / "results/m1156hc_decoder_hot_psum_cross_layer_dse_r1_20260830"
M1155 = HW / "reviews/m1155hc_m1153hc_decoder_hot_psum_cache_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HERE / "destination_major_numeric_replay.json"
EXPECTED = {
    "source": "3816aadf9770ffe59a807a8f25ad1968d8e93db5ec312cd69ee4426443e0f491",
    "contract": "53866e39159a0506aa3565608ed1f42aa7acb3c04448bf7fc2719857253bd2b6",
    "contract_side": "4f20f6b9972a88418181fc118b99edefb197caca077f00c1d7ac01e8c3a0d0ce",
    "contract_outer": "5afc184fac3cfdb077d639b8c6a35f6f7249bdc3e752c1fbaf9bbfd96c06dfca",
    "result_manifest": "ab23f2559b7448253aff14226feb6e6b4e17b1c2e93cd9a3fff23ead734b6701",
    "result_outer": "9b3f1486cea9c518be48f1b7f5e94387067f647b4b068f0e4c8627b362b1c9ce",
    "m1155_outer": "48f4ac97bd0898b18b02e93b9ecee9a6290ac2877661fb240fc6eb9ae5cec7c6",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
GEOMETRY = {
    0: ("D0", 1536, 384, 30, 40),
    1: ("D1", 770, 192, 60, 80),
    2: ("D2", 386, 96, 120, 160),
    3: ("D3", 194, 96, 240, 320),
}
EXPECTED_LAYER = {
    "D0": (4_465_036, 29_622_568, 48_000, 48_000, 4_417_036,
           47_990, 17_863_747, 9_025_999, "1.979143472097"),
    "D1": (4_647_272, 30_338_394, 96_000, 96_000, 4_551_272,
           95_990, 18_592_651, 9_486_475, "1.959911452884"),
    "D2": (5_087_981, 30_328_495, 192_000, 191_862, 4_896_119,
           191_852, 20_355_467, 10_559_856, "1.927627327494"),
    "D3": (17_288_869, 96_760_057, 768_000, 768_000, 16_520_869,
           767_990, 69_162_219, 36_113_672, "1.915125634413"),
}
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
    side = Path(str(path) + ".sha256"); outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double seal content")


def verify_sealed_result() -> dict[str, Any]:
    manifest = RESULT / "SHA256SUMS"; outer = RESULT / "SHA256SUMS.seal.sha256"
    verify_regular(manifest, EXPECTED["result_manifest"]); verify_regular(outer, EXPECTED["result_outer"])
    require(outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["result_manifest"], "SHA256SUMS"], "result outer content")
    listed = {}
    for row in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = row.split(maxsplit=1); name = name.lstrip("*")
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and name not in listed and
                Path(name).name == name, "result manifest row")
        listed[name] = digest
    require(set(listed) == {"RUN_COMPLETE.txt", "report.json"}, "result member set")
    for name, digest in listed.items():
        verify_regular(RESULT / name, digest)
    return strict_json_bytes((RESULT / "report.json").read_bytes())


def verify_m1155() -> dict[str, Any]:
    manifest = M1155 / "SHA256SUMS"; outer = M1155 / "SHA256SUMS.seal.sha256"
    verify_regular(outer, EXPECTED["m1155_outer"])
    manifest_sha, name = outer.read_text(encoding="utf-8").split()
    require(name == "SHA256SUMS", "M1155 outer content"); verify_regular(manifest, manifest_sha)
    listed = {}
    for row in manifest.read_text(encoding="utf-8").splitlines():
        digest, member = row.split(maxsplit=1); listed[member.lstrip("*")] = digest
    actual = {path.name for path in M1155.iterdir() if path.is_file() and
              path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(listed), "M1155 member set")
    for member, digest in listed.items():
        verify_regular(M1155 / member, digest)
    return strict_json_bytes((M1155 / "review.json").read_bytes())


def load_module(path: Path, expected_sha: str, name: str):
    verify_regular(path, expected_sha)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module import spec")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FastPort:
    __slots__ = ("banks", "latency", "limit", "next_issue", "outstanding", "calls")
    def __init__(self, banks: int, latency: int, limit: int):
        self.banks = banks; self.latency = latency; self.limit = limit
        self.next_issue = [0] * banks; self.outstanding = [[] for _ in range(banks)]
        self.calls = 0

    def issue(self, banks: tuple[int, ...], earliest: int,
              beats: int = 1, latency: int | None = None) -> int:
        issue = earliest
        for bank in banks:
            if self.next_issue[bank] > issue:
                issue = self.next_issue[bank]
        while True:
            proposed = issue
            for bank in banks:
                active = sorted(value for value in self.outstanding[bank] if value > issue)
                if len(active) >= self.limit:
                    value = active[len(active) - self.limit]
                    if value > proposed:
                        proposed = value
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


def read_frozen_prefix(contract: dict[str, Any]) -> list[dict[str, Any]]:
    live = HW / contract["input"]["live_jsonl"]
    fd = os.open(live, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd); require(stat.S_ISREG(before.st_mode), "live JSONL regular")
        rows = []
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for frozen in contract["frozen_calls"]:
                raw = stream.readline(65537)
                require(raw.endswith(b"\n") and len(raw) == frozen["line_bytes"] and
                        hashlib.sha256(raw).hexdigest() == frozen["line_sha256"],
                        "frozen call row identity")
                rows.append(strict_json_bytes(raw))
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
                "live JSONL fd replacement")
    finally:
        os.close(fd)
    for ordinal, (row, frozen) in enumerate(zip(rows, contract["frozen_calls"])):
        require(row["global_call_ordinal"] == row["module_ordinal"] == ordinal and
                row["sequence"] == "interlaken_01_a" and row["sequence_sample_id"] == 0 and
                row["diagnostic_cycles"] == frozen["baseline_cycles"] and
                row["kind_summaries"]["psum_read"]["count"] == frozen["updates"] and
                row["kind_summaries"]["psum_write"]["count"] == frozen["updates"] and
                row["kind_summaries"]["output_commit"]["count"] == frozen["commits"],
                "frozen row projection")
    return rows


def replay_layer(mapper, canonical: dict[str, Any], call: dict[str, Any],
                 frozen: dict[str, Any]) -> dict[str, Any]:
    module = int(call["module_ordinal"]); layer, cin, cout, hout, wout = GEOMETRY[module]
    require(layer == frozen["layer"], "layer coordinate")
    output_blocks = math.ceil(cout / 96); dense_per_timestep = hout * wout * output_blocks
    payload = Path(canonical["trust_root"]["canonical_payload"]) / call["payload_relative_path"]
    require(call["global_ordinal"] == module and call["sequence"] == "interlaken_01_a" and
            call["sequence_sample_id"] == 0 and
            call["payload_relative_path"] == frozen["payload_relative_path"] and
            call["payload_sha256"] == frozen["payload_sha256"], "canonical call identity")
    verify_regular(payload, frozen["payload_sha256"])

    b_ext = FastPort(1, 32, 16); c_ext = FastPort(1, 32, 16)
    weight = FastPort(8, 4, 8); b_psum = FastPort(1, 2, 8); c_psum = FastPort(1, 2, 8)
    require(all(port.calls == 0 and all(value == 0 for value in port.next_issue)
                for port in (b_ext, c_ext, weight, b_psum, c_psum)),
            "cross-layer state was not fresh")
    b_compute_next = c_compute_next = 0; b_end = c_end = 0
    updates = source_terms = commits = hits = fills = writes = evictions = flushes = 0
    phase_checks = phase_mismatches = flush_mismatches = dependency_checks = 0
    distance = Counter(); previous_key = None
    referenced_keys = set(); explicit_phase_keys = set()
    without_timestep = set(); without_block = set(); empty_per_timestep = []

    for timestep in range(10):
        truth = {}; b_ready = {}; backing = {}; backing_ready = {}; cache = None
        touched = set()
        for tile in mapper.iter_polyphase_tiles(
                payload, tuple(call["input_shape"]), tile_m=256,
                trusted_root=Path(canonical["trust_root"]["canonical_payload"]).resolve()):
            phase = int(tile["phase_bank"]); values = tile["values"][timestep]
            for local_m, (dy_raw, dx_raw) in enumerate(zip(tile["destination_y"], tile["destination_x"])):
                dy, dx = int(dy_raw), int(dx_raw); destination = dy * wout + dx
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
                    key = (timestep, destination, output_block)
                    if groups:
                        touched.add(key); referenced_keys.add(key)
                        explicit_phase_keys.add((timestep, phase, destination, output_block))
                        without_timestep.add((destination, output_block))
                        without_block.add((timestep, destination))
                    for group in groups:
                        if key == previous_key:
                            distance["0"] += 1; hits += 1
                        else:
                            distance["cold"] += 1; fills += 1
                        previous_key = key; updates += 1
                        truth[key] = truth.get(key, 0) + 1
                        banks = tuple((value % cin) % 8 for value in group)
                        w_ready = weight.issue(banks, 0, 1)
                        bd = b_ext.issue((0,), 0, 1, 32); cd = c_ext.issue((0,), 0, 1, 32)

                        br = b_psum.issue((0,), b_ready.get(key, 0), 1, 2)
                        bi = max(b_compute_next, bd, w_ready, br)
                        bc = bi + 1; b_compute_next = bi + 1
                        bw = b_psum.issue((0,), bc, 1, 1); b_ready[key] = bw
                        if bw > b_end: b_end = bw

                        if cache is not None and cache[0] == key:
                            _, value, ready, _dirty = cache
                        else:
                            if cache is not None and cache[3]:
                                victim_key, victim_value, victim_ready, _dirty = cache
                                wb = c_psum.issue((0,), victim_ready, 1, 1)
                                writes += 1; evictions += 1
                                backing[victim_key] = victim_value; backing_ready[victim_key] = wb
                            ready = c_psum.issue((0,), backing_ready.get(key, 0), 1, 2)
                            value = backing.get(key, 0)
                        ci = max(c_compute_next, cd, w_ready, ready)
                        if ci < ready:
                            raise HammerFailure("accumulator dependency bypass")
                        dependency_checks += 1
                        cc = ci + 1; c_compute_next = ci + 1
                        cache = (key, value + 1, cc, True)
                        if cc > c_end: c_end = cc

        empty_per_timestep.append(dense_per_timestep - len(touched))
        if cache is not None and cache[3]:
            key, value, ready, _dirty = cache
            wb = c_psum.issue((0,), ready, 1, 1)
            writes += 1; flushes += 1; backing[key] = value; backing_ready[key] = wb
        if backing != truth:
            flush_mismatches += 1
        for destination in range(hout * wout):
            for output_block in range(output_blocks):
                key = (timestep, destination, output_block)
                bc = b_ext.issue((0,), b_ready.get(key, 0), 2, 3)
                cc = c_ext.issue((0,), backing_ready.get(key, 0), 2, 3)
                commits += 1
                if bc + 1 > b_end: b_end = bc + 1
                if cc + 1 > c_end: c_end = cc + 1

    expected = EXPECTED_LAYER[layer]
    require((updates, source_terms, commits, fills, hits, evictions, b_end, c_end,
             ratio(b_end, c_end)) == expected, layer + " independent metric mismatch")
    require(b_psum.calls == 2 * updates and c_psum.calls == 2 * fills and
            writes == fills and flushes == 10 and evictions == fills - 10 and
            flush_mismatches == phase_mismatches == 0 and dependency_checks == updates,
            layer + " backing/flush/phase/dependency invariant")
    require(b_ext.calls == c_ext.calls == updates + commits and weight.calls == updates,
            layer + " descriptor/weight/commit fairness")
    require(source_terms > updates and len(referenced_keys) == fills and
            len(explicit_phase_keys) == fills, layer + " key/source invariant")
    if layer == "D2":
        require(sum(empty_per_timestep) == 138 and fills == 191_862 and commits == 192_000 and
                all(value >= 0 for value in empty_per_timestep),
                "D2 empty-destination fill/commit semantics")
    else:
        require(sum(empty_per_timestep) == 0 and fills == commits,
                layer + " unexpected empty destination")
    if layer == "D3":
        require(commits == 768_000, "D3 dense commit population")
    return {
        "layer": layer, "updates": updates, "source_terms": source_terms,
        "commits": commits, "fills": fills, "hits": hits,
        "dirty_evictions": evictions, "terminal_flush_writes": flushes,
        "backing_operations_baseline": b_psum.calls,
        "backing_operations_candidate": c_psum.calls,
        "baseline_cycles": b_end, "candidate_cycles": c_end,
        "local_speedup": ratio(b_end, c_end), "phase_checks": phase_checks,
        "phase_mismatches": phase_mismatches, "flush_mismatches": flush_mismatches,
        "empty_dense_destinations": sum(empty_per_timestep),
        "empty_destinations_per_timestep": empty_per_timestep,
        "referenced_key_count": len(referenced_keys),
        "keys_without_timestep": len(without_timestep),
        "keys_without_output_block": len(without_block),
        "explicit_phase_key_count": len(explicit_phase_keys),
        "accumulator_dependency_checks": dependency_checks,
        "state_fresh_at_layer_start": True,
    }


def main() -> int:
    verify_regular(SOURCE, EXPECTED["source"])
    verify_double(CONTRACT, (EXPECTED["contract"], EXPECTED["contract_side"], EXPECTED["contract_outer"]))
    verify_regular(DOCS359, EXPECTED["docs359"])
    report = verify_sealed_result(); prior = verify_m1155()
    require(prior["status"] ==
            "PASS_M1155HC_DIFFERENT_AUTHOR_FIRST_D0_REPLAY__D0_D1_D2_D3_ONE_CALL_EACH_NEXT_ONLY" and
            prior["authorization"]["d0_d1_d2_d3_one_call_each_cross_layer_cpu_replay_next"] is True,
            "M1155 authorization")
    contract = strict_json_bytes(CONTRACT.read_bytes()); read_frozen_prefix(contract)
    frozen = contract["frozen_sources"]
    canonical_source = load_module(HW / frozen["m1105dr2_source"],
                                   frozen["m1105dr2_source_sha256"], "m1157hc_source")
    mapper = load_module(HW / frozen["mapper"], frozen["mapper_sha256"], "m1157hc_mapper")
    canonical = canonical_source.build_canonical()
    layers = [replay_layer(mapper, canonical, canonical["calls"][ordinal], identity)
              for ordinal, identity in enumerate(contract["frozen_calls"])]
    baseline_sum = sum(row["baseline_cycles"] for row in layers)
    candidate_sum = sum(row["candidate_cycles"] for row in layers)
    require(baseline_sum == 125_974_084 and candidate_sum == 65_186_002 and
            ratio(baseline_sum, candidate_sum) == "1.932532754501",
            "aggregate sum-weighted metric")
    require([row["layer"] for row in layers] == ["D0", "D1", "D2", "D3"] and
            all(row["state_fresh_at_layer_start"] for row in layers),
            "cross-layer order/state isolation")
    sealed_layers = report["layers"]
    for observed, sealed in zip(layers, sealed_layers):
        require(observed["layer"] == sealed["layer"] and
                observed["updates"] == sealed["updates"] and
                observed["commits"] == sealed["commits"] and
                observed["fills"] == sealed["fills"] and
                observed["hits"] == sealed["hits"] and
                observed["dirty_evictions"] == sealed["dirty_evictions"] and
                observed["terminal_flush_writes"] == sealed["terminal_flush_writes"] and
                observed["baseline_cycles"] == sealed["baseline_cycles"] and
                observed["candidate_cycles"] == sealed["candidate_cycles"] and
                observed["local_speedup"] == sealed["local_speedup"],
                "sealed layer mismatch: " + observed["layer"])
    data_bits = 96 * 24; metadata_bits = 16; cache_bytes = math.ceil((data_bits + metadata_bits) / 8)
    fixed = 13_824 + 221_184 + 8_192
    require((data_bits, metadata_bits, cache_bytes, fixed, fixed + cache_bytes,
             245_760 - fixed - cache_bytes) == (2_304, 16, 290, 243_200, 243_490, 2_270),
            "cache capacity")
    require(sha256(DOCS359) == EXPECTED["docs359"], "docs359 changed")
    output = {
        "status": "PASS_M1157HC_NUMERIC_REPLAY__DESTINATION_MAJOR_UPPER_BOUND_ONLY__NO_RTL_AUTHORIZATION",
        "checks": checks,
        "identity": {"source_sha256": EXPECTED["source"],
                     "contract_identity": [EXPECTED["contract"], EXPECTED["contract_side"], EXPECTED["contract_outer"]],
                     "result_outer_seal_file_sha256": EXPECTED["result_outer"],
                     "docs359_sha256": EXPECTED["docs359"]},
        "layers": layers,
        "aggregate": {"baseline_cycles_sum": baseline_sum,
                      "candidate_cycles_sum": candidate_sum,
                      "four_call_decoder_only_speedup": ratio(baseline_sum, candidate_sum),
                      "weighting": "sum baseline cycles divided by sum candidate cycles"},
        "capacity": {"data_bits": data_bits, "metadata_bits": metadata_bits,
                     "bytes_ceil": cache_bytes, "total_with_cache_bytes": fixed + cache_bytes,
                     "remaining_bytes": 2_270},
        "ordering_boundary": {
            "replayed_order": "M1105/M672 destination-major matrix pull",
            "m523_source_event_stream_equivalence_proven": False,
            "bridge_or_reorder_cost_included": False,
            "requires_separate_protocol_and_M712_reconciliation": True},
        "authorization": {"run_accumulator_rtl_and_directed_vcs_source_design_next": False,
                          "rtl_implementation_now": False, "vcs_now": False, "dc": False},
        "claim_boundary": {"one_sequence_one_sample_four_calls": True,
                           "decoder_complete_population": False, "system_speedup": False,
                           "headline": False, "paper_ppa_ready": False},
    }
    encoded = json.dumps(output, sort_keys=True, allow_nan=False) + "\n"
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
