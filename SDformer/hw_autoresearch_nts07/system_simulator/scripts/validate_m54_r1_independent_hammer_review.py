#!/usr/bin/env python3
"""Independent M54 K4-C16 hammer: reconstruct, attack, and validate evidence."""

from __future__ import print_function

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


HW_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path(
    "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/"
    "dc_handoff/runs/m54_k4_ctx16_atomic_exact_sha_vcs_r1_20260823")
RECEIPT = HW_ROOT / "contracts/m54_r1_exact_sha_vcs_receipt_r1_20260823.json"
REVIEW_DIR = HW_ROOT / "results/m54_r1_independent_hammer_20260823"
REVIEW = REVIEW_DIR / "m54_r1_independent_hammer_review.json"
VCS_RERUN_LOG = REVIEW_DIR / "m54_r1_independent_vcs_rerun.log"
VCS_RERUN_LEDGER = REVIEW_DIR / "m54_r1_independent_vcs_ledger.log"

EXPECTED_RECEIPT_SHA256 = (
    "c5ba3b3ac468ef736a478c3eb65157d61653d629c5d0fd7c29cdb58dc0c74546")
EXPECTED_REVIEW_SHA256 = (
    "5b1f66e8e0c8e235984adb1d3fd2ecf9680a8bb1e062973093c00d9190da2393")
EXPECTED_VCS_RERUN_LOG_SHA256 = (
    "b01333bf7059840f419b817112d76d838a21e018478eb4e28aca9354e70d42bd")
EXPECTED_LEDGER_SHA256 = (
    "0e08a01bb02d8dff3df7ad09db19cc810b28c4002163f06a4bfa09df4b6971ae")
EXPECTED_RECONSTRUCTION_SHA256 = (
    "83236c2fa512b9a6066938dac1252d39c1856ca3d9d7ba8ab6d9658bc273a892")
EXPECTED_ATTACKS_SHA256 = (
    "52ce8c5ae1a22eb19b2f22c33dd31e5042b79e15e9a3586f3493772ccdb137e5")

BANKS = 8
LANES = 96
ACC_W = 19


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON: {}".format(raw))

    def pairs(raw_pairs):
        result = {}
        for key, value in raw_pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def signed(value, width):
    value &= (1 << width) - 1
    return value - (1 << width) if value & (1 << (width - 1)) else value


def bits(value):
    return set(index for index in range(256) if (value >> index) & 1)


def unpack_acc(value):
    return [signed(value >> (lane * ACC_W), ACC_W) for lane in range(LANES)]


def model_weight(source, lane):
    if source < BANKS:
        return -128
    return ((source * 37 + lane * 13 + 19) % 255) - 127


def relation(rows):
    count = len(rows)
    if count == 1:
        return "K1"
    sources = [row["add"] | row["sub"] for row in rows]
    if all(item == sources[0] for item in sources[1:]):
        return "K{}_FULL_SHARE".format(count)
    if all(not (sources[left] & sources[right])
               for left in range(count)
               for right in range(left + 1, count)):
        return "K{}_NO_SHARE".format(count)
    return "K{}_PARTIAL_SHARE".format(count)


def reconstruct(lines):
    contexts = {}
    tags = {}
    expected_order = []
    output_tags = set()
    groups = []
    active = None
    commands = requests = outputs = last_requests = 0
    physical_reads = logical_updates = shared_reads = 0
    next_request_tag = 0
    end_seen = False
    last_cycle = -1
    max_context_occupancy = 0
    context_allocate_counts = Counter()

    for lineno, raw in enumerate(lines, 1):
        fields = raw.split()
        if not fields:
            continue
        kind = fields[0]
        if kind == "END":
            require(len(fields) == 3, "line {} malformed END".format(lineno))
            require(not end_seen, "line {} duplicate END".format(lineno))
            require(int(fields[1].split("=", 1)[1]) == commands,
                    "line {} END command mismatch".format(lineno))
            require(int(fields[2].split("=", 1)[1]) == outputs,
                    "line {} END output mismatch".format(lineno))
            end_seen = True
            continue

        require(not end_seen, "line {} record after END".format(lineno))
        require(len(fields) >= 2, "line {} missing cycle".format(lineno))
        cycle = int(fields[1])
        require(cycle >= last_cycle, "line {} cycle order regression".format(lineno))
        last_cycle = cycle

        if kind == "C":
            require(len(fields) == 7, "line {} malformed C".format(lineno))
            ctx, tag = int(fields[2]), int(fields[3], 16)
            add, sub = bits(int(fields[4], 16)), bits(int(fields[5], 16))
            require(0 <= ctx < 16, "line {} context out of 4b range".format(lineno))
            require(ctx not in contexts, "line {} live context reused".format(lineno))
            require(tag not in tags, "line {} command tag reused".format(lineno))
            require(not (add & sub), "line {} overlapping add/sub".format(lineno))
            row = {"tag": tag, "add": add, "sub": sub,
                   "acc": unpack_acc(int(fields[6], 16)), "count": 0,
                   "launched": False}
            contexts[ctx] = row
            tags[tag] = row
            commands += 1
            context_allocate_counts[ctx] += 1
            max_context_occupancy = max(max_context_occupancy, len(contexts))
        elif kind == "L":
            require(len(fields) == 4, "line {} malformed L".format(lineno))
            count, packed = int(fields[2]), int(fields[3], 16)
            require(1 <= count <= 4, "line {} illegal K".format(lineno))
            selected = [(packed >> (slot * 4)) & 0xf for slot in range(count)]
            require(len(set(selected)) == count,
                    "line {} duplicate launch context".format(lineno))
            require(active is None, "line {} overlapping active group".format(lineno))
            rows = []
            for ctx in selected:
                require(ctx in contexts and not contexts[ctx]["launched"],
                        "line {} unowned/relaunched context".format(lineno))
                contexts[ctx]["launched"] = True
                rows.append(contexts[ctx])
                expected_order.append(contexts[ctx]["tag"])
            group = {"contexts": selected, "relation": relation(rows),
                     "requests": 0, "physical_reads": 0,
                     "logical_updates": 0}
            groups.append(group)
            union = set()
            for row in rows:
                union |= row["add"] | row["sub"]
            if union:
                active = group
            else:
                for ctx in selected:
                    del contexts[ctx]
        elif kind == "R":
            require(len(fields) == 10, "line {} malformed R".format(lineno))
            require(active is not None, "line {} request without group".format(lineno))
            request_tag = int(fields[2], 16)
            count, packed = int(fields[3]), int(fields[4], 16)
            require(request_tag == next_request_tag,
                    "line {} request tag sequence mismatch".format(lineno))
            next_request_tag = (next_request_tag + 1) & 0xffff
            selected = [(packed >> (slot * 4)) & 0xf for slot in range(count)]
            require(count == len(active["contexts"]) and
                    selected == active["contexts"],
                    "line {} active context identity mismatch".format(lineno))
            got_bank = int(fields[5], 16)
            got_addr = int(fields[6], 16)
            got_valid = int(fields[7], 16)
            got_subtract = int(fields[8], 16)
            exp_bank = exp_valid = exp_subtract = 0
            chosen = []
            for bank in range(BANKS):
                candidates = sorted(source for ctx in selected
                                    for source in (contexts[ctx]["add"] |
                                                   contexts[ctx]["sub"])
                                    if source % BANKS == bank)
                if not candidates:
                    continue
                source = candidates[0]
                exp_bank |= 1 << bank
                require(((got_addr >> (bank * 5)) & 0x1f) == source // BANKS,
                        "line {} bank{} is not lowest row".format(lineno, bank))
                users = []
                for slot, ctx in enumerate(selected):
                    row = contexts[ctx]
                    if source in row["add"] or source in row["sub"]:
                        exp_valid |= 1 << (slot * BANKS + bank)
                        is_subtract = source in row["sub"]
                        if is_subtract:
                            exp_subtract |= 1 << (slot * BANKS + bank)
                        users.append((ctx, is_subtract))
                chosen.append((source, users))
            require(got_bank == exp_bank and got_bank != 0,
                    "line {} bank-valid mismatch".format(lineno))
            require(got_valid == exp_valid,
                    "line {} destination-valid mismatch".format(lineno))
            require(got_subtract == exp_subtract,
                    "line {} subtract mismatch".format(lineno))
            for source, users in chosen:
                physical_reads += 1
                active["physical_reads"] += 1
                if len(users) > 1:
                    shared_reads += 1
                for ctx, is_subtract in users:
                    row = contexts[ctx]
                    factor = -1 if is_subtract else 1
                    for lane in range(LANES):
                        row["acc"][lane] += factor * model_weight(source, lane)
                        require(-(1 << (ACC_W - 1)) <= row["acc"][lane]
                                < (1 << (ACC_W - 1)),
                                "line {} signed19 overflow".format(lineno))
                    row["add"].discard(source)
                    row["sub"].discard(source)
                    row["count"] += 1
                    logical_updates += 1
                    active["logical_updates"] += 1
            requests += 1
            active["requests"] += 1
            union_left = set()
            for ctx in selected:
                union_left |= contexts[ctx]["add"] | contexts[ctx]["sub"]
            expected_last = not union_left
            require(int(fields[9]) == int(expected_last),
                    "line {} last mismatch".format(lineno))
            if expected_last:
                last_requests += 1
                for ctx in selected:
                    del contexts[ctx]
                active = None
        elif kind == "O":
            require(len(fields) == 5, "line {} malformed O".format(lineno))
            tag, count = int(fields[2], 16), int(fields[3])
            require(outputs < len(expected_order),
                    "line {} output without launch".format(lineno))
            require(tag == expected_order[outputs] and tag not in output_tags and
                    tag in tags, "line {} output tag/order mismatch".format(lineno))
            row = tags[tag]
            require(count == row["count"],
                    "line {} output source-count mismatch".format(lineno))
            require(unpack_acc(int(fields[4], 16)) == row["acc"],
                    "line {} output accumulator mismatch".format(lineno))
            output_tags.add(tag)
            outputs += 1
        else:
            raise ValueError("line {} unknown record {}".format(lineno, kind))

    require(end_seen, "missing END")
    require(active is None and not contexts, "live state remains")
    require(outputs == commands == len(expected_order), "population mismatch")
    group_relations = Counter(group["relation"] for group in groups)
    required_relations = [
        "K1", "K2_FULL_SHARE", "K2_NO_SHARE", "K2_PARTIAL_SHARE",
        "K3_FULL_SHARE", "K3_NO_SHARE", "K3_PARTIAL_SHARE",
        "K4_FULL_SHARE", "K4_NO_SHARE", "K4_PARTIAL_SHARE"]
    require(all(group_relations[name] > 0 for name in required_relations),
            "group-relation coverage missing")
    reused_ids = sorted(ctx for ctx, count in context_allocate_counts.items()
                        if count > 1)
    return {
        "schema": "m54_r1_independent_ledger_reconstruction_v1",
        "status": "PASS_INDEPENDENT_RECONSTRUCTION",
        "commands": commands,
        "groups": len(groups),
        "group_relations": dict(sorted(group_relations.items())),
        "accepted_requests": requests,
        "last_requests": last_requests,
        "physical_unique_weight_row_issues": physical_reads,
        "logical_destination_updates": logical_updates,
        "shared_physical_reads": shared_reads,
        "outputs": outputs,
        "max_live_contexts": max_context_occupancy,
        "context_ids_observed": sorted(context_allocate_counts),
        "context_ids_reused": reused_ids,
        "request_tag_first": 0,
        "request_tag_last": (next_request_tag - 1) & 0xffff,
        "request_tag_rollover_exercised": requests > 65536,
        "mismatch_count": 0,
    }


def mutate_hex(fields, index, xor_value):
    fields[index] = format(int(fields[index], 16) ^ xor_value, "x")


def run_attacks(lines):
    def find(kind, occurrence=0):
        seen = 0
        for index, line in enumerate(lines):
            if line.startswith(kind + " "):
                if seen == occurrence:
                    return index
                seen += 1
        raise ValueError("missing attack record {}".format(kind))

    attacks = []

    def attack(name, mutate):
        changed = list(lines)
        mutate(changed)
        rejected = False
        diagnostic = ""
        try:
            reconstruct(changed)
        except Exception as exc:  # Expected fail-closed path.
            rejected = True
            diagnostic = str(exc)
        require(rejected, "tamper accepted: {}".format(name))
        attacks.append({"name": name, "rejected": True,
                        "diagnostic": diagnostic})

    def overlap(changed):
        for index, line in enumerate(changed):
            if not line.startswith("C "):
                continue
            fields = line.split()
            add_value = int(fields[4], 16)
            if add_value:
                fields[5] = format(add_value & -add_value, "x")
                changed[index] = " ".join(fields)
                return
        raise ValueError("no nonzero command mask for overlap attack")

    def live_context_reuse(changed):
        first, second = find("C"), find("C", 1)
        fields = changed[second].split()
        fields[2] = changed[first].split()[2]
        changed[second] = " ".join(fields)

    def illegal_k(changed):
        index = find("L")
        fields = changed[index].split()
        fields[2] = "5"
        changed[index] = " ".join(fields)

    def duplicate_launch(changed):
        index = find("L", 1)
        fields = changed[index].split()
        require(fields[2] == "2", "attack assumes second launch is K2")
        packed = int(fields[3], 16)
        first_ctx = packed & 0xf
        fields[3] = format((packed & ~0xff) | first_ctx | (first_ctx << 4), "x")
        changed[index] = " ".join(fields)

    def request_field(field_index, xor_value):
        def inner(changed):
            index = find("R")
            fields = changed[index].split()
            mutate_hex(fields, field_index, xor_value)
            changed[index] = " ".join(fields)
        return inner

    def request_last(changed):
        index = find("R")
        fields = changed[index].split()
        fields[9] = "0" if fields[9] == "1" else "1"
        changed[index] = " ".join(fields)

    def output_field(field_index, xor_value):
        def inner(changed):
            index = find("O")
            fields = changed[index].split()
            base = 16 if field_index in (2, 4) else 10
            fields[field_index] = format(int(fields[field_index], base) ^ xor_value,
                                         "x" if base == 16 else "d")
            changed[index] = " ".join(fields)
        return inner

    def remove_end(changed):
        del changed[find("END")]

    def remove_output(changed):
        del changed[find("O")]

    def reorder_outputs(changed):
        first, second = find("O"), find("O", 1)
        changed[first], changed[second] = changed[second], changed[first]

    attack("overlapping_command_masks", overlap)
    attack("live_context_reuse", live_context_reuse)
    attack("illegal_k5_launch", illegal_k)
    attack("duplicate_launch_context", duplicate_launch)
    attack("request_tag_flip", request_field(2, 1))
    attack("request_context_identity_flip", request_field(4, 1))
    attack("request_bank_valid_flip", request_field(5, 1))
    attack("request_lowest_row_address_flip", request_field(6, 1))
    attack("request_destination_valid_flip", request_field(7, 1))
    attack("request_subtract_flip", request_field(8, 1))
    attack("request_last_flip", request_last)
    attack("output_tag_flip", output_field(2, 1))
    attack("output_source_count_flip", output_field(3, 1))
    attack("output_accumulator_flip", output_field(4, 1))
    attack("missing_end", remove_end)
    attack("missing_output", remove_output)
    attack("reordered_outputs", reorder_outputs)
    return {
        "schema": "m54_r1_independent_tamper_attacks_v1",
        "status": "PASS_ALL_TAMPERS_REJECTED",
        "attack_count": len(attacks),
        "rejected_count": sum(1 for row in attacks if row["rejected"]),
        "attacks": attacks,
    }


def parse_covers(log_text):
    covers = {}
    pattern = re.compile(r'\.sva\.(cp_[A-Za-z0-9_]+),\s+\d+ attempts,\s+(\d+) match')
    for name, count in pattern.findall(log_text):
        require(name not in covers, "duplicate cover in VCS log: {}".format(name))
        covers[name] = int(count)
    require(len(covers) == 32 and all(value > 0 for value in covers.values()),
            "full SVA cover population/matches invalid")
    return covers


def check_manifest(manifest, cwd):
    result = subprocess.run(["sha256sum", "--strict", "-c", str(manifest)],
                            cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    require(result.returncode == 0,
            "manifest failure {}: {}".format(manifest, result.stderr[-500:]))


def validate(reconstruction_output, attacks_output, receipt_output, rerun_producer):
    require(sha256(RECEIPT) == EXPECTED_RECEIPT_SHA256,
            "producer receipt SHA drift")
    producer = strict_json(RECEIPT)
    require(producer["status"] ==
            "PASS_M54_R1_EXACT_SHA_VCS_SVA_PENDING_INDEPENDENT_HAMMER",
            "producer receipt status drift")
    require(RUN_DIR.is_dir() and not (RUN_DIR.stat().st_mode & 0o222),
            "canonical run missing/writable")
    for path in RUN_DIR.rglob("*"):
        if not path.is_symlink():
            require(not (path.stat().st_mode &
                         (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)),
                    "canonical path writable: {}".format(path))

    for name, row in producer["source_anchors"].items():
        path = HW_ROOT / row["path"]
        require(path.is_file() and sha256(path) == row["sha256"],
                "source anchor drift: {}".format(name))
    evidence_map = {
        "compile_log_sha256": "compile.raw.log",
        "completion_seal_sha256": "completion_seal.sha256",
        "handshake_ledger_sha256": "m54_handshake_ledger.log",
        "input_manifest_sha256": "input_sha256.txt",
        "ledger_replay_result_sha256": "m54_ledger_replay.json",
        "local_seal_sha256": "run_local_seal.sha256",
        "miter_log_sha256": "miter.raw.log",
        "output_manifest_sha256": "output_sha256.txt",
        "preflight_receipt_sha256": "preflight_receipt.json",
        "preflight_sha_checks_sha256": "preflight_sha_checks.txt",
        "run_complete_sha256": "RUN_COMPLETE.txt",
        "runner_status_sha256": "runner_status.txt",
        "simulation_log_sha256": "sim.raw.log",
        "sva_cover_receipt_sha256": "sva_cover_matches.txt",
    }
    require(set(evidence_map) == set(producer["canonical_evidence_anchors"]),
            "evidence anchor population drift")
    for key, filename in evidence_map.items():
        require(sha256(RUN_DIR / filename) ==
                producer["canonical_evidence_anchors"][key],
                "canonical evidence drift: {}".format(filename))
    check_manifest(RUN_DIR / "input_sha256.txt", HW_ROOT)
    check_manifest(RUN_DIR / "output_sha256.txt", RUN_DIR)
    check_manifest(RUN_DIR / "run_local_seal.sha256", RUN_DIR)
    check_manifest(RUN_DIR / "completion_seal.sha256", RUN_DIR)

    require(sha256(VCS_RERUN_LOG) == EXPECTED_VCS_RERUN_LOG_SHA256,
            "independent VCS rerun log drift")
    require(sha256(VCS_RERUN_LEDGER) == EXPECTED_LEDGER_SHA256 and
            sha256(RUN_DIR / "m54_handshake_ledger.log") == EXPECTED_LEDGER_SHA256,
            "independent/canonical ledger mismatch")
    rerun_log = VCS_RERUN_LOG.read_text(encoding="utf-8")
    require("Compiler version V-2023.12-SP1_Full64; Runtime version V-2023.12-SP1_Full64" in rerun_log,
            "independent VCS version mismatch")
    require("M54_ASSERTION_MODULE_ACTIVE=1" in rerun_log and
            "M54_SVA_BOUND=1" in rerun_log and
            "PASS M54 K4_CTX16_ATOMIC_UNION commands=67 outputs=67 groups=24 requests=53 context16=1 meta16=1 complete16=1 push4=1 pop13push4=1" in rerun_log,
            "independent VCS pass markers absent")
    require(not re.search(r"Offending|failed at|assertion.*(?:fail|error)|(?:Error|Fatal)",
                          rerun_log, re.IGNORECASE),
            "independent VCS failure signature")
    covers = parse_covers(rerun_log)
    require(covers == producer["sva_cover_matches"],
            "independent VCS cover counts differ")

    canonical_lines = (RUN_DIR / "m54_handshake_ledger.log").read_text().splitlines()
    reconstruction = reconstruct(canonical_lines)
    attacks = run_attacks(canonical_lines)
    require(reconstruction == reconstruct(VCS_RERUN_LEDGER.read_text().splitlines()),
            "independent VCS ledger reconstruct differs")
    require(reconstruction["commands"] == reconstruction["outputs"] == 67 and
            reconstruction["groups"] == 24 and
            reconstruction["accepted_requests"] == 53 and
            reconstruction["physical_unique_weight_row_issues"] == 381 and
            reconstruction["logical_destination_updates"] == 450 and
            reconstruction["shared_physical_reads"] == 38 and
            reconstruction["max_live_contexts"] == 16 and
            reconstruction["context_ids_observed"] == list(range(16)) and
            reconstruction["mismatch_count"] == 0,
            "independent reconstruction population mismatch")
    require(attacks["attack_count"] == attacks["rejected_count"] == 17,
            "tamper rejection population mismatch")

    if rerun_producer:
        producer_validator = HW_ROOT / "verif_m54/validate_m54_r1_exact_sha_vcs.py"
        with tempfile.TemporaryDirectory(prefix="m54_independent_producer_") as temp:
            producer_output = Path(temp) / "producer.json"
            result = subprocess.run(
                ["/usr/bin/python3.6", str(producer_validator), "--rerun-tools",
                 "--output", str(producer_output)], cwd=str(HW_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True)
            require(result.returncode == 0 and
                    "PASS M54 producer exact-SHA VCS/SVA" in result.stdout,
                    "producer validator durable rerun failed")
            producer_rerun_sha = sha256(producer_output)
    else:
        producer_rerun_sha = None

    require(not reconstruction_output.exists() and not attacks_output.exists() and
            not receipt_output.exists(), "refusing reviewer artifact overwrite")
    reconstruction_output.parent.mkdir(parents=True, exist_ok=True)
    reconstruction_output.write_text(
        json.dumps(reconstruction, indent=2, sort_keys=True) + "\n")
    attacks_output.write_text(json.dumps(attacks, indent=2, sort_keys=True) + "\n")

    if EXPECTED_REVIEW_SHA256 != "TO_BE_FROZEN":
        require(REVIEW.is_file() and sha256(REVIEW) == EXPECTED_REVIEW_SHA256,
                "independent review SHA drift")
        review = strict_json(REVIEW)
        require(review["score"] == 88 and review["finding_counts"] ==
                {"P0": 0, "P1": 1, "P2": 5} and
                review["verdict"] ==
                "GO_STANDALONE_RTL_VCS_PARTIAL_CLOSE_M52_P1_NO_PERFORMANCE_CLAIM",
                "independent review conclusion drift")
    if EXPECTED_RECONSTRUCTION_SHA256 != "TO_BE_FROZEN":
        require(sha256(reconstruction_output) == EXPECTED_RECONSTRUCTION_SHA256,
                "reconstruction byte drift")
    if EXPECTED_ATTACKS_SHA256 != "TO_BE_FROZEN":
        require(sha256(attacks_output) == EXPECTED_ATTACKS_SHA256,
                "attack receipt byte drift")

    receipt = {
        "schema": "m54_r1_independent_hammer_validation_receipt_v1",
        "status": "PASS_M54_R1_INDEPENDENT_HAMMER_VALIDATED",
        "review_sha256": sha256(REVIEW) if REVIEW.is_file() else None,
        "validator_sha256": sha256(Path(__file__)),
        "producer_receipt_sha256": sha256(RECEIPT),
        "canonical_ledger_sha256": sha256(RUN_DIR / "m54_handshake_ledger.log"),
        "independent_vcs_rerun_log_sha256": sha256(VCS_RERUN_LOG),
        "independent_vcs_rerun_ledger_sha256": sha256(VCS_RERUN_LEDGER),
        "reconstruction_sha256": sha256(reconstruction_output),
        "attacks_sha256": sha256(attacks_output),
        "producer_validator_rerun": bool(rerun_producer),
        "producer_validator_rerun_result_sha256": producer_rerun_sha,
        "vcs": {"tool": "Synopsys VCS V-2023.12-SP1_Full64",
                "commands": 67, "outputs": 67, "groups": 24,
                "accepted_requests": 53, "cover_count": 32,
                "assertion_failure_count": 0},
        "independent_reconstruction_mismatch_count": 0,
        "tamper_attack_count": 17,
        "tamper_rejected_count": 17,
        "dc_launched": False,
        "open_source_hdl_tool_used": False,
        "M52_transaction_cycles_admitted_as_RTL_or_system_cycles": False,
        "M52_P1_structural_fifo_feasibility_closed": True,
        "M52_P1_ten_trace_cycle_model_closed": False,
    }
    receipt_output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-output", type=Path, required=True)
    parser.add_argument("--attacks-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--rerun-producer", action="store_true")
    args = parser.parse_args()
    result = validate(args.reconstruction_output, args.attacks_output,
                      args.receipt_output, args.rerun_producer)
    print("PASS M54 independent hammer score=88 P0=0 P1=1 P2=5 attacks=17/17 M52-P1=PARTIAL")
    print("receipt_validator_sha256={}".format(result["validator_sha256"]))


if __name__ == "__main__":
    main()
