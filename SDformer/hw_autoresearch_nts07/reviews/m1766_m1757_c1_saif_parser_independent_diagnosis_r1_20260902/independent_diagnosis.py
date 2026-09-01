#!/usr/bin/env python3
"""Read-only M1766 diagnosis of the consumed M1757 VCS/SAIF attempt.

This script never launches EDA and never writes into the M1757 namespaces.  It
pins the already-consumed authority, attempt, failure, logs and SAIF, then uses
an independent strict SAIF reader which accepts C block comments outside quoted
strings.  Acceptance of that grammar is deliberately separate from the TX=0
activity-integrity gate.
"""
from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
ATTEMPT = HW / "results/.m1757_c1_unit_delay_functional_saif_energy_attempt_consumed"
FAILURE = HW / "results/m1757_c1_unit_delay_functional_saif_energy_r1_20260901.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1757_c1_unit_delay_functional_saif_energy_r1_20260901.private_build.unsealed_do_not_cite"
CANONICAL = HW / "results/m1757_c1_unit_delay_functional_saif_energy_r1_20260901"
COMPILE_LOG = PRIVATE / "build/compile.log"
SIM_LOG = PRIVATE / "candidate/mapped_sim.log"
SAIF = PRIVATE / "candidate/m1757_c1_directed_component.saif"
RUNNER = HW / "dc_handoff/scripts/run_m1757_m1701_c1_unit_delay_functional_saif_energy_one_shot.py"
CHECKER = HW / "system_simulator/scripts/check_m1757_c1_m1701_unit_delay_functional_saif_energy_source.py"
RELEASE = HW / "contracts/m1759_m1758_m1757_m1701_c1_unit_delay_functional_saif_energy_launch_release_r1_20260901.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
TOP = "tb_m1739_c1_m1701_public_port_mapped_production_energy"

FIXED = {
    ATTEMPT / "attempt.json": "2e4756b60d489b82d5897ba77e624b679993a36c3ca072986d0676c57bce6a9c",
    ATTEMPT / "SHA256SUMS": "48a26f1ff7eaad0230756ede99a77704dad286c6e1d1a5b46d66e2fdfda3d87d",
    ATTEMPT / "SHA256SUMS.seal.sha256": "59f0b02e8dad16109bc904b92753524f98d05645479151246295d0e2a8fd9861",
    FAILURE / "failure.json": "aea36bcd319f89be78ba7a6b26f0ec02acf17f095601ad227ac194759f063427",
    FAILURE / "SHA256SUMS": "ff0eaefd6ac92f539cbcd5d01f05592e73da568731f56584c17e06dc358ad2e2",
    FAILURE / "SHA256SUMS.seal.sha256": "bda296e738e3a6e8ad8791217c9ad3ed2706e53221796eab87fbd98086312b1a",
    COMPILE_LOG: "bb2cb78d32d579acd348c599a5a85bbfb969d182a6389d07c5eb33a03a0fd5ac",
    SIM_LOG: "9eb444c81dd9bc9ecae3b217e7c5e1f419e08eaef88913f4a935cccceeaf2464",
    SAIF: "bc7d2fcf1d4c018698a7e0abb1d33ccea3086435da24ca96383ccb7520e5887e",
    RUNNER: "b7df92c54d20af892264044d9882bbdf43de1cfa79f21d57d11cbb0d613876ea",
    CHECKER: "c1b26c42896822b9903061525636aa2f36ea7a6651c1cba0e14c594808861a7b",
    RELEASE: "c5fca9c2e3a05ad48460baec52403da10c741bcb7071012649fda61ea181d190",
    Path(str(RELEASE) + ".sha256"): "a8110cb46fc601bba632c1d82ccc450066aed5416b35fd24f402a84a764fc4ef",
    Path(str(RELEASE) + ".sha256.seal.sha256"): "73e64711f6ed462da269477d925ba8075c433676ca026093bfc8a74f9d49899e",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_fixed_and_seals():
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "identity drift " + str(path))
    for root in (ATTEMPT, FAILURE):
        manifest = root / "SHA256SUMS"
        outer = root / "SHA256SUMS.seal.sha256"
        need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
             "outer seal drift " + str(root))
        for row in manifest.read_text().splitlines():
            digest, name = row.split(maxsplit=1)
            rel = Path(name.lstrip("*"))
            need(not rel.is_absolute() and ".." not in rel.parts,
                 "unsafe manifest")
            need(sha(root / rel) == digest, "manifest member drift")
    sidecar = Path(str(RELEASE) + ".sha256")
    outer = Path(str(RELEASE) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(RELEASE), RELEASE.name],
         "release sidecar content")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name],
         "release outer sidecar content")
    need(not CANONICAL.exists(), "canonical M1757 result unexpectedly exists")
    attempt = strict_json(ATTEMPT / "attempt.json")
    failure = strict_json(FAILURE / "failure.json")
    release = strict_json(RELEASE)
    need(attempt.get("status") == "M1757_ATTEMPT_CONSUMED"
         and attempt.get("automatic_retry") is False,
         "attempt disposition")
    need(failure.get("status") == "FAILED_OR_INCOMPLETE"
         and failure.get("phase") == "MAPPED_SIM_SAIF"
         and failure.get("counts") == {"vcs_compiles": 1, "simv_runs": 1,
                                         "saif_files": 0, "ptpx_runs": 0},
         "failure disposition")
    need(release.get("status") ==
         "AUTHORIZE_ONE_M1757_C1_UNIT_DELAY_FUNCTIONAL_SAIF_ENERGY_CAMPAIGN",
         "release disposition")


def strip_block_comments(text):
    """Remove only C block comments outside quoted strings; reject truncation."""
    output = []
    index = 0
    comments = 0
    while index < len(text):
        if text[index] == '"':
            start = index
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            need(index <= len(text) and text[index - 1] == '"',
                 "unterminated quoted string")
            output.append(text[start:index])
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            need(end >= 0, "unterminated block comment")
            output.append(" ")
            comments += 1
            index = end + 2
        else:
            output.append(text[index])
            index += 1
    return "".join(output), comments


def tokens(text):
    return re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+', text)


def parse_saif(text):
    value = tokens(text)
    pos = [0]
    def one():
        need(pos[0] < len(value) and value[pos[0]] == "(",
             "malformed SAIF")
        pos[0] += 1
        node = []
        while pos[0] < len(value) and value[pos[0]] != ")":
            if value[pos[0]] == "(":
                node.append(one())
            else:
                node.append(value[pos[0]])
                pos[0] += 1
        need(pos[0] < len(value), "unterminated SAIF")
        pos[0] += 1
        return node
    root = one()
    need(pos[0] == len(value), "trailing SAIF token")
    need(root and root[0] == "SAIFILE", "SAIF root")
    return root, len(value)


def forms(node, tag):
    return [item for item in node[1:]
            if isinstance(item, list) and item and item[0] == tag]


def activity_audit(root):
    tx_count = 0
    tx_nonzero = 0
    tx_sum = 0.0
    tx_max = 0.0
    tc_count = 0
    tc_positive = 0
    by_instance = {}
    records = {}

    def walk(node, instances):
        nonlocal tx_count, tx_nonzero, tx_sum, tx_max, tc_count, tc_positive
        if not isinstance(node, list) or not node:
            return
        if node[0] == "INSTANCE" and len(node) > 1:
            instances = instances + [str(node[1])]
        activity = dict((item[0], float(item[1])) for item in node[1:]
                        if isinstance(item, list) and len(item) == 2
                        and item[0] in {"T0", "T1", "TX", "TC", "IG"})
        if activity:
            key = "/".join(instances)
            records[(key, str(node[0]))] = activity
            if "TX" in activity:
                tx_count += 1
                value = activity["TX"]
                need(math.isfinite(value) and value >= 0.0,
                     "invalid TX value")
                if value != 0.0:
                    tx_nonzero += 1
                    tx_sum += value
                    tx_max = max(tx_max, value)
                    by_instance[key] = by_instance.get(key, 0) + 1
            if "TC" in activity:
                tc_count += 1
                if activity["TC"] > 0.0:
                    tc_positive += 1
        for item in node[1:]:
            if isinstance(item, list):
                walk(item, instances)
    walk(root, [])

    dut = TOP + "/dut"
    scratch = dut + "/u_parent_scratch"
    def one_record(instance, name):
        value = records.get((instance, name))
        need(value is not None, "SAIF signal absent " + instance + "/" + name)
        return value
    scratch_controls = dict((name, one_record(scratch, name)) for name in (
        "address\\[5\\]", "address\\[4\\]", "address\\[3\\]",
        "address\\[2\\]", "address\\[1\\]", "address\\[0\\]",
        "enable", "write_enable_BAR", "clk_core"))
    read_data = [one_record(scratch, "read_data\\[" + str(bit) + "\\]")
                 for bit in range(1152)]
    public_scalars = dict((name, one_record(dut, name)) for name in (
        "reset_n", "prep_valid", "prep_ready", "issue_data_valid",
        "issue_request_valid", "protocol_error", "task_done_valid",
        "psum_write_valid"))
    public_vectors = {}
    for prefix, width in (("psum_write_data", 1824),
                          ("issue_residual_data", 1152),
                          ("issue_psum_prior", 1824),
                          ("debug_parent_live_bitmap", 64),
                          ("debug_written_bitmap", 64),
                          ("issue_request_row_id", 6),
                          ("issue_request_parent_id", 6),
                          ("issue_request_source_index", 4)):
        rows = [one_record(dut, prefix + "\\[" + str(bit) + "\\]")
                for bit in range(width)]
        public_vectors[prefix] = {
            "signals": width,
            "tx_nonzero_signals": sum(row.get("TX", 0.0) != 0.0 for row in rows),
            "tx_sum_ns": sum(row.get("TX", 0.0) for row in rows)}
    need(all(row.get("TX") == 0.0 for row in scratch_controls.values()),
         "scratch control contains TX")
    need(all(row.get("TX") == 500.0 for row in read_data),
         "scratch read_data TX shape drift")
    need(all(row.get("TX") == 0.0 for row in public_scalars.values()),
         "public scalar contains TX")
    need(all(row["tx_nonzero_signals"] == 0
             for row in public_vectors.values()),
         "public vector contains TX")
    return {
        "tx_forms": tx_count,
        "tx_nonzero_forms": tx_nonzero,
        "tx_nonzero_sum_ns": tx_sum,
        "tx_max_ns": tx_max,
        "tc_forms": tc_count,
        "tc_positive_forms": tc_positive,
        "tx_nonzero_by_instance": by_instance,
        "scratch_controls": scratch_controls,
        "scratch_read_data": {
            "bits": len(read_data), "all_tx_ns": 500.0,
            "tx_nonzero_bits": sum(row["TX"] != 0.0 for row in read_data)},
        "public_scalars": public_scalars,
        "public_vectors": public_vectors,
    }


def main():
    verify_fixed_and_seals()
    compile_text = COMPILE_LOG.read_text(errors="strict")
    sim_text = SIM_LOG.read_text(errors="strict")
    need("61 modules and 3 UDPs read." in compile_text
         and "../simv up to date" in compile_text
         and "Error-[" not in compile_text,
         "compile log disposition")
    pass_token = "PASS_M1739_C1_M1701_PUBLIC_PORT_MAPPED_DIRECTED_COMPONENT_ACTIVITY"
    need(sim_text.count(pass_token) == 1 and "$fatal" not in sim_text
         and "Error-[" not in sim_text and "Assertion failed" not in sim_text,
         "runtime log disposition")
    match = re.findall(
        r"M1739_PUBLIC_COUNTERS cycles=([0-9]+) issue_accepts=([0-9]+)"
        r" parent_edges=([0-9]+) macro_reads=([0-9]+) macro_writes=([0-9]+)"
        r" forwards=([0-9]+) dead_write_elisions=([0-9]+)"
        r" psum_commits=([0-9]+) row_completions=([0-9]+)", sim_text)
    need(match == [("252", "96", "48", "46", "34", "2", "30", "64", "64")],
         "runtime counters")

    raw = SAIF.read_text(errors="strict")
    legacy_first_tokens = tokens(raw)[:4]
    need(legacy_first_tokens[0] != "(", "legacy parser unexpectedly accepts prefix")
    cleaned, comment_count = strip_block_comments(raw)
    root, token_count = parse_saif(cleaned)
    need(comment_count == 2, "VCS header comment count")
    duration = forms(root, "DURATION")
    need(duration == [["DURATION", "756.00"]], "SAIF duration")
    top_instances = forms(root, "INSTANCE")
    need(len(top_instances) == 1 and top_instances[0][1] == TOP,
         "top instance")
    dut_instances = forms(top_instances[0], "INSTANCE")
    need(len(dut_instances) == 1 and dut_instances[0][1] == "dut",
         "DUT instance")
    activity = activity_audit(root)
    need(activity["tx_forms"] == 117690
         and activity["tx_nonzero_forms"] == 37550
         and activity["tx_nonzero_sum_ns"] == 13546027.0
         and activity["tx_max_ns"] == 756.0,
         "TX audit drift")
    value = {
        "schema": "m1766_m1757_c1_saif_parser_independent_diagnosis_r1_v1",
        "status": "FAIL_M1766_SAIF_GRAMMAR_VALID_AFTER_COMMENT_STRIP__TX_NONZERO_REJECT__NO_PTPX",
        "canonical_result_absent": True,
        "attempt": {"consumed": True, "vcs_compiles": 1,
                    "simv_runs": 1, "generated_saif_files": 1,
                    "m1757_validated_saif_files": 0, "ptpx_runs": 0,
                    "automatic_retry": False},
        "runtime": {"pass_token_count": 1, "cycles": 252,
                    "duration_ns": 756.0, "issue_accepts": 96,
                    "parent_edges": 48, "macro_reads": 46,
                    "macro_writes": 34, "forwards": 2,
                    "dead_write_elisions": 30, "psum_commits": 64,
                    "row_completions": 64},
        "parser": {"legacy_first_tokens": legacy_first_tokens,
                   "legacy_failure": "first token is a C block-comment atom, not left parenthesis",
                   "strict_comment_aware_root": "SAIFILE",
                   "block_comments_skipped_outside_strings": comment_count,
                   "tokens_fully_consumed": token_count},
        "activity": activity,
        "disposition": {
            "parser_only_successor_may_reuse_existing_saif_for_diagnosis": True,
            "parser_only_successor_may_release_ptpx": False,
            "reason": "strictly parsed SAIF violates the existing all-TX-zero integrity gate",
            "recommended_first_fix": "public-port warmup/prime before enabling the measured SAIF window",
            "warmup_feasibility": {
                "sequential_epochs_supported": True,
                "epoch_rule": "second prep_task_start requires prep_epoch greater than newest_epoch_q",
                "task_completion_frees_bank": True,
                "per_task_counters_clear_on_execution_start": True,
                "scratch_storage_persists_across_task_done": True,
                "nine_macro_slices_share_enable_address_and_write_event": True,
                "recommended_epochs": {"warmup": 5944, "measured": 5945},
                "recommended_workload": "same 64 masks in warmup and measured task so every measured parent address was initialized through public execution",
                "required_success_gates": ["TX=0 for every DUT SAIF form",
                                           "100_percent_PT_read_saif_annotation",
                                           "second-task public scoreboard PASS",
                                           "measurement excludes warmup"]
            },
            "valid_gating_of_unknown_macro_output_for_ptpx": "not_admitted_without_new_proof",
            "forbidden": ["+initreg", "force", "+notimingcheck",
                          "+no_notifier", "+nospecify", "ignore_TX"],
            "eda_executed_by_m1766": False,
        },
        "claim_boundary": {"functional_runtime_pass": True,
                           "saif_grammar_valid": True,
                           "saif_activity_integrity_pass": False,
                           "ptpx": False, "component_power": False,
                           "component_energy": False, "paper_citable_power": False,
                           "system_speedup": False, "headline": False},
    }
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
