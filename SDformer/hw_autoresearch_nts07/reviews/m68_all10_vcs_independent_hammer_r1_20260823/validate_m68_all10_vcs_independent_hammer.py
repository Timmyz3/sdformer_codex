#!/usr/bin/env python3
"""Fail-closed independent validator for the M68 all-ten hammer review."""

from __future__ import print_function

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import struct


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REVIEW = HERE / "m68_all10_vcs_independent_hammer_review.json"
VALIDATION_RECEIPT = HERE / "m68_all10_vcs_independent_hammer_validation_receipt.json"
PRODUCER_RECEIPT = HW / (
    "results/m68_m66_all10_vcs_exact_replay_receipt_r1_20260823/"
    "m68_m66_all10_vcs_exact_replay_receipt.json")
STREAM_ROOT = HW / "results/m68_m66_all10_schedule_streams_dev_r1_20260823"
SIMV = HW / "dc_handoff/runs/m66_s00_lookahead_exact_sha_compile_r3_20260823/simv"

HEADER = struct.Struct("<8sIIQQQ")
GROUP = struct.Struct("<4sQQBBBBBBH")
DESCRIPTOR = struct.Struct("<HBB32s32s")
TRAILER = struct.Struct("<4sQQQ")

ROOT_PATHS = {
    "producer_receipt": PRODUCER_RECEIPT,
    "receipt_builder": HW / "verif_m66/build_m68_all10_vcs_receipt.py",
    "run_script": HW / "dc_handoff/scripts/run_vcs_m68_one_sample_fifo_dev.sh",
    "ledger_replayer": HW / "verif_m66/replay_m66_handshake_ledger.py",
    "schedule_generator": HW / "verif_m57/generate_m57_schedule_stream.py",
    "m53_analyzer": HW / "system_simulator/scripts/analyze_m53_adaptive_temporal_parent_k4_ctx16_dse.py",
    "m45_scheduler": HW / "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py",
    "testbench": HW / "tb_m66/tb_m66_m53_schedule_bridge_lookahead.sv",
    "m66_core_rtl": HW / "rtl_m66/qfit_k4_parent_delta_p8_l96_ctx16_lookahead.sv",
    "m66_bridge_rtl": HW / "rtl_m66/qfit_m66_m53_schedule_bridge_lookahead.sv",
    "m54_sva": HW / "verif_m54/qfit_k4_parent_delta_p8_l96_ctx16_assertions.sv",
    "m66_sva": HW / "verif_m66/qfit_k4_parent_delta_lookahead_assertions.sv",
    "filelist": HW / "dc_handoff/filelists/date_m66_m53_schedule_bridge_lookahead_vcs.f",
    "compiled_simv": SIMV,
}

EXPECTED_ROOT_SHA = {
    "producer_receipt": "c91eb47fd3a4bc021a64736874d34faa75175460c0859102a297889c812ff2b8",
    "receipt_builder": "f87fc8628c3ba3ab4561e01dfa43b289dd9e4545e60440a12b0935eb83e3d63a",
    "run_script": "5578ec0843555a9978ec6e86eff35fcbfbaa14e78f79fe80825fa7a75ca7cd59",
    "ledger_replayer": "bb7d2e3b600226e1ec09498ce64d035db6f1d4ad92fb641edfe04331362fdd1e",
    "schedule_generator": "0c80782a17eb6d9361e4f34e97fb8f45b245d4da7f37999b2581b7a3c2b675df",
    "m53_analyzer": "638809bd72ab7f66fc69b51f4cb726f2c0d1c7712f71188066b4ef04cbdda531",
    "m45_scheduler": "c1e3610ce59753f786498db46cde7b330155fa2e3c836198be165aad3eb3f38f",
    "testbench": "67d6f76182c1566ffbda9274cdbc0f01cbca19a34668290f3c086e4730c32771",
    "m66_core_rtl": "b9a2064ab73764534415f2dc54aa134807a147c6b8528f0fb041e3afc5d13f4d",
    "m66_bridge_rtl": "d1020823c328c528c5e9693cc85bd973667e143a335de2fa7a1f081f19e7c7af",
    "m54_sva": "1338421c3ee3d12f70fb2b2299e76d6651c297500920b1ffb70989c90cc2a267",
    "m66_sva": "e522c849411ab89e59037825764410e617cc642a158d3a488472272131fb3973",
    "filelist": "1a6bea2c3bc7b9a83fa69b875739f21bcb896021bc4cddcbd4089dbea311af03",
    "compiled_simv": "839d599287f63b7a973688253c815d8549448a1a0f8078e9185d6f3d098333cf",
}

EXPECTED_MODELS = [
    8117384, 8139624, 7999848, 7995672, 7870896,
    7876904, 8010640, 7948792, 7962096, 7947952,
]
EXPECTED_RTL = [
    8117392, 8139633, 7999877, 7995679, 7870920,
    7886729, 8010650, 7984551, 7974220, 7947961,
]
EXPECTED_REQUESTS = [
    7011032, 7014848, 6903792, 6756440, 6739216,
    6803648, 6937576, 6923136, 6911016, 6846392,
]
EXPECTED_FINAL_LATE = [7, 8, 28, 6, 23, 9824, 9, 35758, 12123, 8]
EXPECTED_MAX_LATE = [95319, 90966, 108051, 57782, 89038,
                     123002, 121338, 147216, 130478, 109637]
EXPECTED_NOZERO_FINAL = [7, 8, 28, 6, 22, 47, 9, 16526, 35, 8]
EXPECTED_PERFECT_COMMAND_FINAL = [5, 2, 24, 5, 10, 41, 6, 20, 35, 7]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pairs_hook(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError("non-standard JSON constant: " + raw)))


def parse_sha_manifest(path, expected_paths):
    observed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "malformed SHA manifest: " + str(path))
        expected_sha, raw_target = match.groups()
        target = Path(raw_target)
        require(target.is_absolute(), "producer manifest path unexpectedly relative")
        require(target.parent.resolve() == path.parent.resolve(),
                "manifest target escapes run directory")
        require(target.name not in observed, "duplicate manifest basename")
        require(target.is_file(), "manifest target missing: " + target.name)
        require(sha256_path(target) == expected_sha,
                "manifest target SHA drift: " + target.name)
        observed[target.name] = expected_sha
    require(set(observed) == set(expected_paths), "manifest exact path-set drift")
    return observed


def read_stream(sample):
    stream = STREAM_ROOT / "m68_s{:02d}_schedule.bin.gz".format(sample)
    manifest_path = STREAM_ROOT / "m68_s{:02d}_schedule_manifest.json".format(sample)
    manifest = strict_json(manifest_path)
    require(sha256_path(stream) == manifest["identity"]["compressed_stream_sha256"],
            "compressed stream SHA drift s{:02d}".format(sample))
    uncompressed_digest = hashlib.sha256()
    uncompressed_bytes = 0

    def take(handle, count):
        nonlocal uncompressed_bytes
        payload = handle.read(count)
        require(len(payload) == count, "truncated stream s{:02d}".format(sample))
        uncompressed_digest.update(payload)
        uncompressed_bytes += len(payload)
        return payload

    rows = []
    with gzip.open(stream, "rb") as handle:
        header = HEADER.unpack(take(handle, HEADER.size))
        require(header[0] == b"M57R1BIN" and header[1] == 1 and
                header[2] == sample, "header identity drift")
        for index in range(header[3]):
            group = GROUP.unpack(take(handle, GROUP.size))
            require(group[0] == b"GRP1" and group[2] == index and
                    group[3] == sample, "group identity drift")
            require(1 <= group[8] <= 4 and 0 <= group[9] <= 32,
                    "group geometry drift")
            # Descriptor semantics are already independently replayed from the
            # accepted-handshake ledger.  Read the exact group payload in one
            # block here so the stream SHA, population and cycle recurrence can
            # be revalidated without 25.92 million small Python reads.
            take(handle, DESCRIPTOR.size * group[8])
            rows.append({
                "target": group[1], "operator": group[4],
                "timestep": group[5], "tile": group[6], "block": group[7],
                "count": group[8], "cycles": group[9],
            })
        trailer = TRAILER.unpack(take(handle, TRAILER.size))
        require(trailer[0] == b"END1" and trailer[1] == header[3] and
                trailer[2] == header[4], "trailer population drift")
        require(handle.read(1) == b"", "stream trailing bytes")
    require(uncompressed_digest.hexdigest() ==
            manifest["identity"]["uncompressed_stream_sha256"],
            "uncompressed stream SHA drift")
    require(uncompressed_bytes == manifest["identity"]["uncompressed_stream_bytes"],
            "uncompressed byte count drift")
    require(header[4] == 2592000 and header[5] == EXPECTED_MODELS[sample] and
            trailer[3] == EXPECTED_REQUESTS[sample], "stream total drift")
    require(rows[-1]["target"] + rows[-1]["cycles"] + 4 == header[5],
            "model tail relation drift")
    return rows, header


def simulate(rows, remove_zero_exclusion=False, perfect_command=False):
    accepted = max(rows[0]["target"], rows[0]["count"])
    maximum_late = accepted - rows[0]["target"]
    target_winners = 0
    operator_end = []
    operator_max = defaultdict(lambda: -1)
    binding = Counter()
    for index in range(1, len(rows)):
        current = rows[index]
        previous = rows[index - 1]
        candidates = {"target": current["target"]}
        if not perfect_command:
            candidates["descriptor"] = accepted + current["count"] + 1
        zero_extra = int(
            not remove_zero_exclusion and previous["cycles"] > 0 and
            current["cycles"] == 0)
        candidates["engine"] = accepted + previous["cycles"] + 1 + zero_extra
        accepted = max(candidates.values())
        winners = tuple(sorted(name for name, value in candidates.items()
                               if value == accepted))
        binding[winners] += 1
        if "target" in winners:
            target_winners += 1
        late = accepted - current["target"]
        maximum_late = max(maximum_late, late)
        operator_max[current["operator"]] = max(
            operator_max[current["operator"]], late)
        if index == len(rows) - 1 or rows[index + 1]["operator"] != current["operator"]:
            operator_end.append(late)
    return {
        "final_late": accepted - rows[-1]["target"],
        "maximum_late": maximum_late,
        "target_winners": target_winners,
        "operator_end": operator_end,
        "binding": binding,
        "rtl_reconstructed": accepted + rows[-1]["cycles"] + 5,
    }


def validate_run(sample, producer_row):
    run = HW / "results/m68_m66_all10_vcs_dev_s{:02d}_r1_20260823/s{:02d}".format(
        sample, sample)
    expected_names = {
        "start_epoch.txt", "end_epoch.txt", "sim.command.txt",
        "prelaunch_input.sha256", "sim.raw.log", "sim.rc",
        "stream_gzip.rc", "ledger_gzip.rc",
        "m68_s{:02d}_handshake_ledger.compact.log.gz".format(sample),
        "replay.raw.log", "replay.rc",
        "m68_s{:02d}_ledger_replay.json".format(sample),
    }
    manifest = parse_sha_manifest(run / "output_manifest.sha256", expected_names)
    for name in ("sim.rc", "stream_gzip.rc", "ledger_gzip.rc", "replay.rc"):
        require((run / name).read_text(encoding="utf-8").strip() == "0",
                "nonzero run return code")
    complete = (run / "RUN_COMPLETE.txt").read_text(encoding="utf-8")
    require("PASS_M68_SAMPLE_{:02d}_M66_LOOKAHEAD_VCS_REPLAY".format(sample)
            in complete and "SYSTEM_SPEEDUP_ADMITTED=false" in complete and
            "PAPER_PPA_READY=false" in complete, "RUN_COMPLETE drift")
    prelaunch = (run / "prelaunch_input.sha256").read_text(encoding="utf-8")
    require(EXPECTED_ROOT_SHA["compiled_simv"] in prelaunch and
            EXPECTED_ROOT_SHA["ledger_replayer"] in prelaunch,
            "prelaunch root identity drift")
    log = (run / "sim.raw.log").read_text(encoding="utf-8", errors="strict")
    require(log.count("M66_LOOKAHEAD_ASSERTION_MODULE_ACTIVE=1") == 1 and
            log.count("M54_ASSERTION_MODULE_ACTIVE=1") == 1,
            "SVA activation marker drift")
    passes = re.findall(
        r"^PASS M66 S(\d+) groups=(\d+) commands=(\d+) requests=(\d+) "
        r"outputs=(\d+) rtl_cycles=(\d+) model_cycles=(\d+) seamless=(\d+) "
        r"max_meta=(\d+) tag_wrap=(\d+)$", log, re.M)
    require(len(passes) == 1 and int(passes[0][0]) == sample,
            "VCS terminal drift")
    require(not re.search(
        r"(?i)(protocol_error|M57_FAULT_CAUSE|assertion failure|\bfatal\b|"
        r"\berror([ :-]|$)|^FAIL )", log, re.M),
        "VCS failure signature")
    replay = strict_json(run / "m68_s{:02d}_ledger_replay.json".format(sample))
    require(sha256_path(run / "m68_s{:02d}_ledger_replay.json".format(sample)) ==
            producer_row["replay_sha256"], "replay SHA drift")
    require(replay["accepted_requests"] == replay["accepted_responses"] ==
            EXPECTED_REQUESTS[sample] and replay["accepted_outputs"] == 2592000 and
            replay["functional_mismatch_count"] == 0,
            "replay conservation drift")
    require(replay["maximum_context_occupancy"] == 8 and
            replay["maximum_metadata_occupancy"] == 1 and
            replay["maximum_complete_occupancy"] == 7 and
            replay["metadata_fifo_final_occupancy"] == 0,
            "queue bound/final occupancy drift")
    stalls = replay["stalls"]
    require(all(stalls[name] == 0 for name in
                ("command", "request", "response", "output")) and
            replay["launch_phase"]["prelaunch_artificial_bubbles"] == 0,
            "unexpected external/artificial stall")
    require(int(passes[0][3]) == EXPECTED_REQUESTS[sample] and
            int(passes[0][4]) == 2592000 and
            int(passes[0][5]) == EXPECTED_RTL[sample] and
            int(passes[0][6]) == EXPECTED_MODELS[sample],
            "VCS PASS numeric drift")
    return replay, manifest


def nearest_rank(values, percentile):
    ordered = sorted(values)
    return ordered[int(math.ceil(len(ordered) * percentile)) - 1]


def validate_all():
    for name, path in ROOT_PATHS.items():
        require(sha256_path(path) == EXPECTED_ROOT_SHA[name],
                "root SHA drift: " + name)
    receipt = strict_json(PRODUCER_RECEIPT)
    review = strict_json(REVIEW)
    require(receipt["status"] ==
            "PASS_M68_ALL10_M66_VCS_EXACT_REPLAY_MEMORY_PHYSICAL_UNADMITTED",
            "producer status drift")
    require(review["status"] ==
            "PASS_SCOPED_ALL10_ARITHMETIC_PROTOCOL_AND_MAKESPAN_REPLAY_OUTLIERS_EXPLAINED_NO_SYSTEM_SPEEDUP",
            "review status drift")
    reconstructed = []
    total_requests = 0
    for sample in range(10):
        producer_row = receipt["samples"][sample]
        require(producer_row["sample_id"] == sample, "sample order drift")
        replay, _ = validate_run(sample, producer_row)
        rows, header = read_stream(sample)
        actual = simulate(rows)
        nozero = simulate(rows, remove_zero_exclusion=True)
        perfect_command = simulate(rows, perfect_command=True)
        require(actual["final_late"] == EXPECTED_FINAL_LATE[sample] and
                actual["maximum_late"] == EXPECTED_MAX_LATE[sample] and
                actual["rtl_reconstructed"] == EXPECTED_RTL[sample],
                "independent recurrence drift s{:02d}".format(sample))
        require(nozero["final_late"] == EXPECTED_NOZERO_FINAL[sample] and
                perfect_command["final_late"] == EXPECTED_PERFECT_COMMAND_FINAL[sample],
                "counterfactual recurrence drift")
        require(producer_row["rtl_cycles"] == EXPECTED_RTL[sample] and
                producer_row["m53_transaction_model_cycles"] == EXPECTED_MODELS[sample] and
                producer_row["rtl_minus_m53_transaction_cycles"] ==
                EXPECTED_RTL[sample] - EXPECTED_MODELS[sample],
                "producer sample cycle drift")
        reconstructed.append(actual)
        total_requests += replay["accepted_requests"]
    deltas = [rtl - model for rtl, model in zip(EXPECTED_RTL, EXPECTED_MODELS)]
    require(sum(EXPECTED_MODELS) == 79869808 and sum(EXPECTED_RTL) == 79927612 and
            sum(deltas) == 57804 and total_requests == 68847096,
            "aggregate arithmetic drift")
    require(nearest_rank(deltas, 0.50) == 10 and
            nearest_rank(deltas, 0.95) == 35759 and
            nearest_rank(EXPECTED_RTL, 0.50) == 7984551 and
            nearest_rank(EXPECTED_RTL, 0.95) == 8139633,
            "nearest-rank distribution drift")
    require(sum(deltas[index] for index in (5, 7, 8)) == 57708,
            "outlier contribution drift")
    claim = receipt["claim_boundary"]
    require(all(claim[name] is False for name in (
        "date_headline", "dram_address_timed", "full_network_or_system_cycles",
        "offline_descriptor_fetch_bytes_charged", "online_selector_implemented",
        "paper_ppa_ready", "system_speedup",
        "weight_sram_ports_and_macros_implemented")),
        "producer claim boundary widened")
    require(review["protocol_bug_assessment"]["protocol_bug_found"] is False and
            len(review["findings"]["p0"]) == 0 and
            len(review["findings"]["p1"]) == 3 and
            len(review["findings"]["p2"]) == 2,
            "review finding classification drift")
    scores = review["scores"]
    require(scores["hardware_innovation"] == 43 and
            scores["performance_evidence_quality"] == 78 and
            scores["performance_advantage"] == 56 and
            scores["system_performance_advantage"] == 32,
            "review score drift")
    if VALIDATION_RECEIPT.exists():
        validation = strict_json(VALIDATION_RECEIPT)
        require(validation["status"] ==
                "PASS_M68_ALL10_INDEPENDENT_HAMMER_VALIDATION" and
                validation["metrics"]["exact_cycle_reconstruction_matches"] == 10 and
                validation["metrics"]["p0_count"] == 0 and
                validation["metrics"]["p1_count"] == 3 and
                validation["metrics"]["p2_count"] == 2,
                "validation receipt drift")
        identity = validation["identity"]
        require(identity["review_sha256"] == sha256_path(REVIEW) and
                identity["validator_sha256"] == sha256_path(Path(__file__)) and
                identity["producer_receipt_sha256"] ==
                EXPECTED_ROOT_SHA["producer_receipt"] and
                identity["compiled_simv_sha256"] ==
                EXPECTED_ROOT_SHA["compiled_simv"],
                "validation receipt identity drift")
    return {
        "model_cycles": sum(EXPECTED_MODELS),
        "rtl_cycles": sum(EXPECTED_RTL),
        "delta": sum(deltas),
        "requests": total_requests,
        "exact_cycle_reconstruction_matches": len(reconstructed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = validate_all()
    print("PASS M68 independent all10 hammer: model={} rtl={} delta={} "
          "requests=rsp={} outputs=25920000 recon={}/10 P0=0 P1=3 P2=2".format(
              result["model_cycles"], result["rtl_cycles"], result["delta"],
              result["requests"], result["exact_cycle_reconstruction_matches"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M68 independent all10 hammer: {}".format(error))
        raise SystemExit(1)
