#!/usr/bin/env python3
"""Independent fail-closed audit for M57 S00 phase-safe compact r3.

No HDL simulator is invoked.  The validator streams the 199 MB schedule and
the compressed 8.7M-event ledger, parses VCS covers, verifies the production
manifest, and attacks review/receipt claim boundaries in memory.
"""

from __future__ import print_function

from collections import deque
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REVIEW = HERE / "m57_s00_phase_safe_r3_independent_hammer_review.json"
VALIDATION_RECEIPT = HERE / "m57_s00_phase_safe_r3_independent_hammer_validation_receipt.json"
RUN = HW / "results/m57_h67_k4c16_temporal_vcs_s00_phase_safe_full_compact_r3_20260823"
COMPILE = HW / "dc_handoff/runs/m57_diagnostics_20260823/s00_compile_r9_phase_safe"

PATHS = {
    "contract": HW / "contracts/m57_s00_phase_safe_full_compact_exact_sha_vcs_contract_r2_20260823.json",
    "receipt": RUN / "m57_s00_phase_safe_exact_sha_vcs_receipt.json",
    "replay": RUN / "m57_s00_ledger_replay.json",
    "ledger": RUN / "m57_s00_handshake_ledger.compact.log.gz",
    "sim_log": RUN / "sim.raw.log",
    "output_manifest": RUN / "output_manifest.sha256",
    "run_complete": RUN / "RUN_COMPLETE.txt",
    "prelaunch_manifest": RUN / "prelaunch_input_sha256.txt",
    "schedule_stream": HW / "dc_handoff/runs/m57_diagnostics_20260823/s00_sim_r2/input.bin",
    "schedule_manifest": HW / "results/m57_h67_k4c16_temporal_vcs_r1_20260823/m57_s00_schedule_manifest.json",
    "simv": COMPILE / "simv",
    "wrapper_rtl": HW / "rtl_m57/qfit_m57_m53_schedule_bridge.sv",
    "core_rtl": HW / "rtl_m54/qfit_k4_parent_delta_p8_l96_ctx16.sv",
    "tb": HW / "tb_m57/tb_m57_m53_schedule_bridge.sv",
    "sva": HW / "verif_m54/qfit_k4_parent_delta_p8_l96_ctx16_assertions.sv",
    "filelist": HW / "dc_handoff/filelists/date_m57_m53_schedule_bridge_vcs.f",
    "compile_log": COMPILE / "compile.raw.log",
}

EXPECTED_SHA = {
    "contract": "c0dc028dc9c8ad92ac94758446fe657187d5ae7fdfebdc8e0ad1e27e9a565e76",
    "receipt": "ad65e91ed45f171870ef6718079f4d25806111ef7004622c673f6fdbad9cbbf7",
    "replay": "6ff1f3101ae9d0c1a2331e428d133e17397005294ff54b2b16fc1caa31afec9b",
    "ledger": "ce13def66753e0a127873cfa6f102f1e08b97e82610ff83d04edbb2e25e30a98",
    "sim_log": "c046a8bec21378763121d9b068aa8fc596ff773ded3b5a168f3551ed18d7436d",
    "output_manifest": "d6a5d663a6b12cc0727eb7b27c9b74bc05b46f296f612382f60fa71e6468aad9",
    "run_complete": "7b1f9bc52963550a6cee0f389deafc84469f95e74e2539b05b28d9c3e3f982b2",
    "prelaunch_manifest": "ca9a6560c063b02800de93ad0f91e09baa7e5207c4cfca011b484cb0c0b8b0f2",
    "schedule_stream": "496706ce20dd685bbb913523d8da6e44eee6ed2c836c557d634f5f75bc45a63a",
    "schedule_manifest": "7e93928600e0ceeddf2e2103de66c7d065260e98a5845d44c0618d26c3c4c125",
    "simv": "e826ea3e2a37703f0ca87dcc3c10e6a490ef77e7877bb915e4c3213ca51e3943",
    "wrapper_rtl": "61962cb18ac82f232a0aa0b5d2649bdbe4f28e3306a18221feafd4cb64bfe460",
    "core_rtl": "e06040f6aeac3f30b2d018d415b95ae2471f01632ce801d789b0c93421e4cf0a",
    "tb": "965ccdba8abe3881a265d41576d57081244d345fedc40e5e7198dfe7e2b4f6fa",
    "sva": "1338421c3ee3d12f70fb2b2299e76d6651c297500920b1ffb70989c90cc2a267",
    "filelist": "1aedb4246f85eb0755c27a66735df7a9e61b26196ffd9ee97916973778b86a06",
    "compile_log": "a04625a7cb5eaef20462212f214d1303a7107315de1ebe2dc5227a75e1c40334",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-standard JSON constant: " + value)))


def parse_kv(line):
    fields = {}
    for token in line.strip().split()[1:]:
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def verify_path_hashes():
    for name, path in PATHS.items():
        observed = sha256_path(path)
        require(observed == EXPECTED_SHA[name],
                "{} SHA drift {} != {}".format(
                    name, observed, EXPECTED_SHA[name]))


def verify_output_manifest():
    lines = PATHS["output_manifest"].read_text(
        encoding="utf-8").splitlines()
    require(len(lines) == 18, "production output manifest population drift")
    run_prefix = str(RUN.resolve()) + "/"
    seen = set()
    for line in lines:
        match = re.match(r"^([0-9a-f]{64})  (\./.+)$", line)
        require(match is not None, "malformed output manifest line")
        expected, relative = match.groups()
        target = (RUN / relative[2:]).resolve()
        require(str(target).startswith(run_prefix),
                "output manifest path escape")
        require(relative not in seen and target.is_file(),
                "missing/duplicate sealed output " + relative)
        seen.add(relative)
        require(sha256_path(target) == expected,
                "sealed output SHA drift " + relative)
    for required in ("./m57_s00_handshake_ledger.compact.log.gz",
                     "./m57_s00_ledger_replay.json",
                     "./m57_s00_phase_safe_exact_sha_vcs_receipt.json",
                     "./sim.raw.log", "./validator.raw.log"):
        require(required in seen, "required output not sealed " + required)


def parse_schedule_stream():
    header_struct = struct.Struct("<8sIIQQQ")
    group_struct = struct.Struct("<4sQQBBBBBBH")
    descriptor_struct = struct.Struct("<HBB32s32s")
    trailer_struct = struct.Struct("<4sQQQ")
    pop_lut = [bin(value).count("1") for value in range(256)]
    group_by_k = [0, 0, 0, 0, 0]
    source_by_k = [0, 0, 0, 0, 0]
    zero_by_k = [0, 0, 0, 0, 0]
    parents = [0, 0, 0, 0]
    groups = commands = source_cycles = 0
    signed_add = signed_subtract = 0
    complete_tail = complete_wraps = 0
    first_target = last_target = None
    previous_target = -1
    nonmonotonic = 0
    with PATHS["schedule_stream"].open("rb") as handle:
        header_raw = handle.read(header_struct.size)
        require(len(header_raw) == header_struct.size, "short schedule header")
        header = header_struct.unpack(header_raw)
        require(header == (b"M57R1BIN", 1, 0, 839456, 2592000,
                           8117384), "schedule header drift")
        while groups < header[3]:
            raw = handle.read(group_struct.size)
            require(len(raw) == group_struct.size, "short group record")
            row = group_struct.unpack(raw)
            magic, target_cycle, group_id = row[:3]
            k_count, group_cycles = row[8], row[9]
            require(magic == b"GRP1" and group_id == groups and
                    1 <= k_count <= 4 and 0 <= group_cycles <= 32,
                    "group identity/geometry drift")
            if target_cycle < previous_target:
                nonmonotonic += 1
            previous_target = target_cycle
            first_target = target_cycle if first_target is None else first_target
            last_target = target_cycle
            group_by_k[k_count] += 1
            source_by_k[k_count] += group_cycles
            zero_by_k[k_count] += int(group_cycles == 0)
            commands += k_count
            source_cycles += group_cycles
            if complete_tail > 15 - k_count:
                complete_wraps += 1
            complete_tail = (complete_tail + k_count) & 15
            for unused_slot in range(k_count):
                raw = handle.read(descriptor_struct.size)
                require(len(raw) == descriptor_struct.size,
                        "short descriptor record")
                task_index, parent, unused_reserved, add_mask, sub_mask = (
                    descriptor_struct.unpack(raw))
                require(task_index < 300 and parent < 4,
                        "descriptor metadata drift")
                require(not any(a & b for a, b in zip(add_mask, sub_mask)),
                        "overlapping signed masks in schedule")
                parents[parent] += 1
                signed_add += sum(pop_lut[value] for value in add_mask)
                signed_subtract += sum(pop_lut[value] for value in sub_mask)
            groups += 1
        trailer_raw = handle.read(trailer_struct.size)
        require(len(trailer_raw) == trailer_struct.size,
                "short schedule trailer")
        trailer = trailer_struct.unpack(trailer_raw)
        extra = handle.read(1)
    require(trailer == (b"END1", groups, commands, source_cycles),
            "schedule trailer conservation drift")
    require(extra == b"", "bytes after schedule trailer")
    return {
        "groups": groups,
        "commands": commands,
        "source_cycles": source_cycles,
        "group_by_k": group_by_k,
        "source_by_k": source_by_k,
        "zero_by_k": zero_by_k,
        "parents": parents,
        "signed_add": signed_add,
        "signed_subtract": signed_subtract,
        "first_target": first_target,
        "last_target": last_target,
        "nonmonotonic": nonmonotonic,
        "complete_wraps": complete_wraps,
        "metadata_wraps": source_cycles // 16,
        "complete_tail_final": complete_tail,
    }


def parse_compact_ledger():
    flag_population = {}
    gap_population = {}
    metadata = deque()
    expected_meta = 0
    request_next_tag = 0
    output_next_tag = 0
    requests = responses = outputs = events = 0
    tag_errors = output_order_errors = 0
    max_meta = max_complete = max_context = 0
    first_cycle = last_cycle = previous_cycle = None
    begin = end = None
    with gzip.open(str(PATHS["ledger"]), "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("BEGIN "):
                require(begin is None and events == 0, "duplicate/late BEGIN")
                begin = parse_kv(line)
                continue
            if line.startswith("END "):
                require(end is None, "duplicate END")
                end = parse_kv(line)
                continue
            tokens = line.strip().split()
            require(len(tokens) == 7 and tokens[0] == "E",
                    "bad event line {}".format(line_number))
            cycle = int(tokens[1])
            flags = int(tokens[2], 16)
            occupancy = int(tokens[3], 16)
            req = flags & 1
            rsp = (flags >> 1) & 1
            out = (flags >> 2) & 1
            require(req or rsp or out, "empty compact event")
            require(last_cycle is None or cycle > last_cycle,
                    "nonmonotonic event cycle")
            if previous_cycle is not None:
                gap = cycle - previous_cycle
                gap_population[str(gap)] = gap_population.get(str(gap), 0) + 1
            previous_cycle = cycle
            first_cycle = cycle if first_cycle is None else first_cycle
            last_cycle = cycle
            key = "{}{}{}".format(req, rsp, out)
            flag_population[key] = flag_population.get(key, 0) + 1
            meta = occupancy & 31
            complete = (occupancy >> 5) & 31
            context = (occupancy >> 10) & 31
            require(meta == expected_meta and 0 <= complete <= 16 and
                    0 <= context <= 16, "occupancy replay drift")
            max_meta = max(max_meta, meta)
            max_complete = max(max_complete, complete)
            max_context = max(max_context, context)
            if rsp:
                require(metadata, "response with empty independent FIFO")
                expected_tag = metadata.popleft()
                tag_errors += int(int(tokens[5], 16) != expected_tag)
                responses += 1
            if req:
                observed = int(tokens[4], 16)
                tag_errors += int(observed != request_next_tag)
                metadata.append(observed)
                request_next_tag = (request_next_tag + 1) & 0xffff
                requests += 1
            expected_meta += req - rsp
            require(expected_meta == len(metadata) and
                    0 <= expected_meta <= 16,
                    "metadata conservation failure")
            if out:
                output_order_errors += int(
                    int(tokens[6], 16) != output_next_tag)
                output_next_tag += 1
                outputs += 1
            events += 1
    require(begin is not None and end is not None, "ledger missing BEGIN/END")
    return {
        "begin": begin,
        "end": end,
        "event_lines": events,
        "first_cycle": first_cycle,
        "last_cycle": last_cycle,
        "flag_population": flag_population,
        "gap_population": gap_population,
        "requests": requests,
        "responses": responses,
        "outputs": outputs,
        "final_meta": expected_meta,
        "tag_errors": tag_errors,
        "output_order_errors": output_order_errors,
        "max_meta": max_meta,
        "max_complete": max_complete,
        "max_context": max_context,
        "final_request_tag": request_next_tag,
        "final_output_tag": output_next_tag,
    }


def parse_sim_covers():
    text = PATHS["sim_log"].read_text(encoding="utf-8", errors="replace")
    covers = {}
    pattern = (r"m54_sva\.(cp_[A-Za-z0-9_]+),\s+"
               r"(\d+) attempts,\s+(\d+) match")
    for name, attempts, matches in re.findall(pattern, text):
        require(name not in covers, "duplicate SVA cover " + name)
        covers[name] = {"attempts": int(attempts),
                        "matches": int(matches)}
    require(len(covers) == 32, "SVA cover population drift")
    require(text.count("M54_ASSERTION_MODULE_ACTIVE=1") == 1,
            "SVA module activation drift")
    require(len(re.findall(r"^PASS M57 S0 ", text, re.MULTILINE)) == 1,
            "unique M57 PASS line absent")
    require(not re.search(r"(?i)(assertion failed|error-|fatal:)", text),
            "assertion/error/fatal signature in sim")
    return covers


def integer_end(end, key):
    return int(end[key].split(",")[0])


def validate_cross_evidence(schedule, ledger, covers):
    require(schedule["groups"] == 839456 and
            schedule["commands"] == 2592000 and
            schedule["source_cycles"] == 7011032,
            "schedule primary conservation drift")
    require(schedule["group_by_k"] == [0, 163464, 89760, 95912, 490320] and
            schedule["source_by_k"] == [0, 354512, 401848, 612288, 5642384] and
            schedule["zero_by_k"] == [0, 28840, 4720, 3280, 4192],
            "schedule K distribution drift")
    require(schedule["parents"] == [1005792, 802856, 535536, 247816] and
            sum(schedule["parents"]) == schedule["commands"] and
            schedule["signed_add"] == 39369240 and
            schedule["signed_subtract"] == 17823400,
            "schedule parent/arithmetic drift")
    require(schedule["first_target"] == 388 and
            schedule["last_target"] == 8117378 and
            schedule["nonmonotonic"] == 0 and
            schedule["complete_wraps"] == 162000 and
            schedule["metadata_wraps"] == 438189 and
            schedule["complete_tail_final"] == 0,
            "schedule target/wrap drift")
    require(ledger["begin"] == {
        "sample": "0", "groups": "839456", "commands": "2592000",
        "model_cycles": "8117384", "latency": "1",
        "ledger": "compact_v1"}, "ledger BEGIN drift")
    require(ledger["event_lines"] == 8720312 and
            ledger["requests"] == ledger["responses"] == 7011032 and
            ledger["outputs"] == 2592000 and ledger["final_meta"] == 0 and
            ledger["tag_errors"] == ledger["output_order_errors"] == 0,
            "ledger accepted-event conservation drift")
    require(ledger["flag_population"] == {
        "001": 910856, "010": 760112, "011": 38312,
        "100": 148800, "101": 649624, "110": 5219400,
        "111": 993208}, "ledger flag distribution drift")
    require(ledger["gap_population"] == {
        "1": 8668024, "2": 38359, "3": 10440, "4": 2240,
        "5": 1248}, "ledger cycle-gap distribution drift")
    require(ledger["first_cycle"] == 390 and
            ledger["last_cycle"] == 8791652 and
            ledger["max_meta"] == 1 and ledger["max_complete"] == 7 and
            ledger["max_context"] == 8 and
            ledger["final_request_tag"] == 64216 and
            ledger["final_output_tag"] == 2592000,
            "ledger cycle/occupancy/tag drift")
    end = ledger["end"]
    require(integer_end(end, "groups") == schedule["groups"] and
            integer_end(end, "commands") == schedule["commands"] and
            integer_end(end, "requests") == ledger["requests"] and
            integer_end(end, "responses") == ledger["responses"] and
            integer_end(end, "outputs") == ledger["outputs"] and
            integer_end(end, "rtl_cycles") == 8791654 and
            integer_end(end, "model_cycles") == 8117384 and
            integer_end(end, "mismatches") == 0,
            "ledger END primary drift")
    require(integer_end(end, "phase_direct") == 839444 and
            integer_end(end, "phase_aligned") == 12 and
            integer_end(end, "phase_direct") +
            integer_end(end, "phase_aligned") == schedule["groups"] and
            integer_end(end, "prelaunch_artificial_bubbles") == 0 and
            integer_end(end, "late_groups") == 839443 and
            integer_end(end, "late") == 346984961248 and
            integer_end(end, "launch_stall") == 5359724,
            "ledger phase/lateness drift")
    for key in ("cmd_stall", "req_stall", "rsp_stall", "out_stall"):
        require(integer_end(end, key) == 0, key + " drift")
    require(integer_end(end, "reuse") == 2591992 and
            integer_end(end, "tag_wrap") == 106,
            "reuse/tag-wrap drift")
    derived_cover = {
        "cp_k1": schedule["source_by_k"][1],
        "cp_k2": schedule["source_by_k"][2],
        "cp_k3": schedule["source_by_k"][3],
        "cp_k4": schedule["source_by_k"][4],
        "cp_zero_k1": schedule["zero_by_k"][1],
        "cp_zero_k2": schedule["zero_by_k"][2],
        "cp_zero_k3": schedule["zero_by_k"][3],
        "cp_zero_k4": schedule["zero_by_k"][4],
        "cp_push4": schedule["group_by_k"][4],
        "cp_meta_tail_wrap": schedule["metadata_wraps"],
        "cp_complete_tail_wrap": schedule["complete_wraps"],
    }
    for name, expected in derived_cover.items():
        require(covers[name]["matches"] == expected,
                "derived SVA cover drift " + name)
    require(all(row["attempts"] == 8791659 for row in covers.values()),
            "SVA attempts drift")
    for name in ("cp_context16", "cp_meta16", "cp_complete16",
                 "cp_complete13_pop_push4", "cp_request_stall",
                 "cp_response_stall", "cp_output_stall",
                 "cp_unexpected_response", "cp_duplicate_context_launch",
                 "cp_response_mismatch", "cp_overflow", "cp_fault"):
        require(covers[name]["matches"] == 0,
                "expected uncovered SVA cover changed " + name)
    require(sum(covers[name]["matches"] for name in (
        "cp_k2_full_share", "cp_k2_partial_share", "cp_k2_no_share")) ==
            covers["cp_k2"]["matches"], "K2 cover partition drift")
    require(covers["cp_k3"]["matches"] - sum(
        covers[name]["matches"] for name in (
            "cp_k3_full_share", "cp_k3_partial_share", "cp_k3_no_share")) ==
            139904, "K3 cover residual drift")
    require(covers["cp_k4"]["matches"] - sum(
        covers[name]["matches"] for name in (
            "cp_k4_full_share", "cp_k4_partial_share", "cp_k4_no_share")) ==
            2514920, "K4 cover residual drift")


def validate_producer_receipt(receipt, schedule, ledger, covers):
    require(receipt["status"] ==
            "PASS_EXACT_SHA_PHASE_SAFE_FULL_S00_VCS_COMPACT_REPLAY" and
            receipt["contract_sha256"] == EXPECTED_SHA["contract"],
            "producer receipt status/contract drift")
    run = receipt["run"]
    require(run["full_sample_not_sampled"] is True and
            run["sim_rc"] == run["gzip_rc"] == 0 and
            run["elapsed_seconds"] == 605,
            "producer run receipt drift")
    functional = receipt["functional_and_protocol"]
    require(functional["sample_id"] == 0 and
            functional["accepted_requests"] ==
            functional["accepted_responses"] == schedule["source_cycles"] and
            functional["accepted_outputs"] == schedule["commands"] and
            functional["event_lines"] == ledger["event_lines"] and
            functional["rtl_cycles"] == 8791654 and
            functional["m53_transaction_model_cycles"] == 8117384 and
            functional["rtl_minus_m53_transaction_cycles"] == 674270 and
            functional["functional_mismatch_count"] == 0,
            "producer functional receipt drift")
    require(receipt["sva"]["module_active"] is True and
            receipt["sva"]["assertion_failure_signatures"] == 0 and
            receipt["sva"]["coverpoints"] == covers,
            "producer SVA receipt drift")
    require(receipt["claim_boundary"] == {
        "m53_transaction_model_cycles_are_system_cycles": False,
        "paper_ppa_ready": False,
        "permitted": "exact-SHA S00 standalone VCS functional, handshake, FIFO occupancy, phase-safe launch, conservation, SVA-cover and RTL-cycle evidence",
        "power_or_energy_admitted": False,
        "system_speedup_admitted": False,
    }, "producer claim boundary widened")


def validate_review(review, schedule, ledger):
    require(review["schema"] ==
            "m57_s00_phase_safe_r3_independent_hammer_review_v1" and
            review["status"] ==
            "PASS_INDEPENDENT_S00_FUNCTIONAL_AUDIT_NO_GO_SCHEDULER_OR_DATE_HEADLINE",
            "review identity/status drift")
    require(review["scope"]["sample_id"] == 0 and
            review["scope"]["full_sample_not_sampled"] is True and
            review["scope"]["all10"] is False and
            review["scope"]["producer_evidence_modified"] is False,
            "review scope widened")
    scores = review["scores"]
    require(scores["date_accelerator_completeness_score"] == 55 and
            sum(scores["subscores"].values()) == 55 and
            scores["standalone_s00_vcs_evidence_readiness"] == 84,
            "review score drift")
    require(review["issues"]["P0"] == [] and
            [item["id"] for item in review["issues"]["P1"]] == [
                "M57-P1-01-NO-ONLINE-SCHEDULER",
                "M57-P1-02-S00-NOT-ALL10-OR-SYSTEM",
                "M57-P1-03-MODEL-GAP-AND-LATENESS",
                "M57-P1-04-UNSTRESSED-READY-MEMORY-ENVIRONMENT",
                "M57-P1-05-NO-PHYSICAL-OR-COMPARATIVE-CLOSURE"] and
            [item["id"] for item in review["issues"]["P2"]] == [
                "M57-P2-01-M54-CORE-SOURCE-SHA-OMITTED",
                "M57-P2-02-COMPACT-LEDGER-OMITS-LAUNCH-EVENTS",
                "M57-P2-03-BUBBLE-COUNTER-HARDWIRED",
                "M57-P2-04-SVA-COVERAGE-GAPS",
                "M57-P2-05-DATA-VALUE-DIVERSITY",
                "M57-P2-06-NO-NEGATIVE-PROTOCOL-ATTACKS"],
            "review issue inventory drift")
    stream = review["independent_schedule_stream_recompute"]
    require(stream["header"]["fusion_groups"] == schedule["groups"] and
            stream["header"]["descriptor_commands"] == schedule["commands"] and
            stream["source_issue_cycles_by_k"]["total"] ==
            schedule["source_cycles"] and
            stream["parent_descriptor_counts"]["total"] ==
            schedule["commands"] and
            stream["signed_add_updates"] == schedule["signed_add"] and
            stream["signed_subtract_updates"] == schedule["signed_subtract"] and
            stream["derived_complete_tail_wraps"] ==
            schedule["complete_wraps"] and
            stream["derived_metadata_tail_wraps"] ==
            schedule["metadata_wraps"], "review stream recompute drift")
    led = review["independent_compact_ledger_recompute"]
    require(led["event_lines"] == ledger["event_lines"] and
            led["flag_population_req_rsp_out"] == ledger["flag_population"] and
            led["event_cycle_gap_population"] == ledger["gap_population"] and
            led["accepted_requests"] == ledger["requests"] and
            led["accepted_responses"] == ledger["responses"] and
            led["accepted_outputs"] == ledger["outputs"] and
            led["maximum_metadata_occupancy"] == ledger["max_meta"] and
            led["maximum_complete_occupancy"] == ledger["max_complete"] and
            led["maximum_context_occupancy"] == ledger["max_context"],
            "review ledger recompute drift")
    cycle = review["cycle_and_phase_audit"]
    require(cycle["rtl_cycles"] - cycle["m53_transaction_model_cycles"] ==
            cycle["rtl_minus_model_cycles"] == 674270 and
            cycle["phase_direct_groups"] + cycle["phase_aligned_groups"] ==
            schedule["groups"] and cycle["schedule_late_groups"] == 839443 and
            cycle["launch_stall_cycles"] == 5359724,
            "review cycle/phase drift")
    architecture = review["architecture_and_environment_audit"]
    require(architecture["wrapper_contains_online_scheduler"] is False and
            architecture["weight_request_ready_forced_high"] is True and
            architecture["output_ready_forced_high"] is True and
            architecture["response_latency_cycles"] == 1 and
            architecture["observed_maximum_contexts"] == 8 and
            architecture["observed_maximum_metadata"] == 1,
            "review architecture/environment drift")
    require(review["admission_gate_for_next_milestone"]["current_result"] ==
            "NO_GO_DATE_HEADLINE_GO_ONLINE_SCHEDULER_ALL10_BACKPRESSURE_AND_SYNOPSYS_NEXT",
            "review next gate drift")


def run_attacks(receipt, review, schedule, ledger, covers):
    attacks = []
    mutant = copy.deepcopy(receipt)
    mutant["claim_boundary"]["system_speedup_admitted"] = True
    try:
        validate_producer_receipt(mutant, schedule, ledger, covers)
        rejected = False
    except ValueError:
        rejected = True
    require(rejected, "system speedup promotion accepted")
    attacks.append({"name": "producer_system_speedup_promotion",
                    "rejected": True})

    mutant = copy.deepcopy(receipt)
    mutant["functional_and_protocol"]["rtl_cycles"] = 8117384
    try:
        validate_producer_receipt(mutant, schedule, ledger, covers)
        rejected = False
    except ValueError:
        rejected = True
    require(rejected, "RTL/model gap erasure accepted")
    attacks.append({"name": "rtl_model_gap_erasure", "rejected": True})

    mutant = copy.deepcopy(review)
    mutant["architecture_and_environment_audit"][
        "wrapper_contains_online_scheduler"] = True
    try:
        validate_review(mutant, schedule, ledger)
        rejected = False
    except ValueError:
        rejected = True
    require(rejected, "online scheduler false promotion accepted")
    attacks.append({"name": "online_scheduler_false_promotion",
                    "rejected": True})

    mutant = copy.deepcopy(review)
    mutant["scope"]["all10"] = True
    try:
        validate_review(mutant, schedule, ledger)
        rejected = False
    except ValueError:
        rejected = True
    require(rejected, "all10 false promotion accepted")
    attacks.append({"name": "all10_false_promotion", "rejected": True})

    mutant = copy.deepcopy(review)
    mutant["scores"]["date_accelerator_completeness_score"] = 95
    try:
        validate_review(mutant, schedule, ledger)
        rejected = False
    except ValueError:
        rejected = True
    require(rejected, "DATE score promotion accepted")
    attacks.append({"name": "date_score_promotion", "rejected": True})
    return attacks


def validate_validation_receipt(receipt, attacks, schedule, ledger):
    require(receipt["schema"] ==
            "m57_s00_phase_safe_r3_independent_hammer_validation_receipt_v1" and
            receipt["status"] ==
            "PASS_M57_S00_PHASE_SAFE_R3_INDEPENDENT_HAMMER_VALIDATED",
            "validation receipt status drift")
    require(receipt["review_sha256"] == sha256_path(REVIEW) and
            receipt["validator_sha256"] == sha256_path(Path(__file__)) and
            receipt["producer_receipt_sha256"] == EXPECTED_SHA["receipt"] and
            receipt["producer_output_manifest_sha256"] ==
            EXPECTED_SHA["output_manifest"],
            "validation receipt identity drift")
    require(receipt["scores"] == {
        "date_accelerator_completeness": 55,
        "standalone_s00_vcs_evidence_readiness": 84},
        "validation receipt score drift")
    require(receipt["severity_counts"] == {"P0": 0, "P1": 5, "P2": 6},
            "validation receipt severity drift")
    require(receipt["conservation"] == {
        "fusion_groups": schedule["groups"],
        "descriptor_commands": schedule["commands"],
        "accepted_requests": ledger["requests"],
        "accepted_responses": ledger["responses"],
        "accepted_outputs": ledger["outputs"],
        "final_metadata_occupancy": 0},
        "validation receipt conservation drift")
    require(receipt["cycle_boundary"] == {
        "rtl_cycles": 8791654,
        "model_cycles": 8117384,
        "gap_cycles": 674270,
        "late_groups": 839443,
        "system_speedup_admitted": False},
        "validation receipt cycle boundary drift")
    require(receipt["negative_attacks"] == attacks,
            "validation receipt attack drift")
    require(receipt["admission"] == {
        "online_scheduler": False,
        "all10": False,
        "realistic_ready_memory_backpressure": False,
        "dc_sta_formality": False,
        "system_speedup": False,
        "date_headline": False}, "validation receipt admission drift")


def main():
    verify_path_hashes()
    verify_output_manifest()
    require(int((RUN / "sim.rc").read_text()) == 0 and
            int((RUN / "gzip.rc").read_text()) == 0 and
            int((RUN / "validator.rc").read_text()) == 0 and
            int((COMPILE / "compile.rc").read_text()) == 0,
            "producer return-code drift")
    contract = strict_json(PATHS["contract"])
    require("core_rtl" not in contract["exact_sha256"] and
            "m54_core_rtl" not in contract["exact_sha256"],
            "core-source omission finding no longer true")
    schedule = parse_schedule_stream()
    ledger = parse_compact_ledger()
    covers = parse_sim_covers()
    validate_cross_evidence(schedule, ledger, covers)
    receipt = strict_json(PATHS["receipt"])
    review = strict_json(REVIEW)
    validate_producer_receipt(receipt, schedule, ledger, covers)
    validate_review(review, schedule, ledger)
    attacks = run_attacks(receipt, review, schedule, ledger, covers)
    validation_receipt = strict_json(VALIDATION_RECEIPT)
    validate_validation_receipt(
        validation_receipt, attacks, schedule, ledger)
    print("PASS M57 S00 r3 independent hammer score=55 readiness=84 "
          "P0=0 P1=5 P2=6 attacks={}/{} groups=839456 commands=2592000 "
          "requests=responses=7011032 rtl_gap=674270 all10=false "
          "online_scheduler=false system_speedup=false".format(
              len(attacks), len(attacks)))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FAIL M57 S00 r3 independent hammer: {}".format(error),
              file=sys.stderr)
        sys.exit(1)
