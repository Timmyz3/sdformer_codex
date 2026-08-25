#!/usr/bin/env python3
"""Independent audit of the sealed M99 directed + actual-record VCS run.

No producer, launcher, simulator, or production RTL is executed.  This script
checks frozen identities and sealed outputs, parses both VCS campaigns, and
recomputes the actual-record population directly from the input binaries.
"""

from __future__ import print_function

import collections
import hashlib
import json
from pathlib import Path
import re
import struct


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONTRACT = HW / "contracts/m99_phase_slack_metadata_compiler_vcs_contract_r1_20260824.json"
FILELIST = HW / "dc_handoff/filelists/date_m99_phase_slack_vcs.f"
LAUNCHER = HW / "dc_handoff/scripts/run_vcs_m99_phase_slack_actual_records_sva.sh"
RUN = HW / "dc_handoff/runs/m99_phase_slack_vcs_r1_sealed_20260824"
RECORDS = Path("/tmp/m85_inputs/m83_cap11_phase_records.bin")
OFFSETS = Path("/tmp/m85_inputs/m83_cap11_phase_offsets_u32le.bin")
METADATA = HW / "results/m85_canonical_74b_phase_metadata_r1_20260823/m85_phase_metadata_74b.bin"
OUTPUT = HERE / "m99_sealed_vcs_independent_audit_r2.json"

FROZEN = {
    "rtl_m82/zero_bubble_elastic_pwp_stream.sv":
        "2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f",
    "rtl_m85/guarded_wordpacked_pwp_stream.sv":
        "ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0",
    "rtl_m99/phase_slack_guarded_wordpacked_pwp_stream.sv":
        "adb2dfd95ee3dd179cb373eb5ead937d9beb4db25648325634ebba755243b082",
    "verif_m99/phase_slack_guarded_wordpacked_pwp_stream_assertions.sv":
        "461cafca5614231216652bc69a27c997e28f0d331b7a3dd958726d9297bb48de",
    "tb_m99/tb_m99_phase_slack_guarded_wordpacked_pwp_stream.sv":
        "14eeb00be94d1338aefb37190e53f81ff03edd6eb2f98eef48515433f5843aff",
    "tb_m99/tb_m99_phase_slack_actual_records.sv":
        "a3a2987164565659c8fe86aac8584651fba808eade00cbefcfa200a8fa1b3167",
    "dc_handoff/filelists/date_m99_phase_slack_vcs.f":
        "12bcb401f2779407fed42577476c8c456eaff85f742daca31f259205a0ab1975",
}
EXPECTED = {
    "contract": "a89fde382fb19b639523a0b2d0b4500b498794a09ec960a529c25c390324c420",
    "filelist": FROZEN["dc_handoff/filelists/date_m99_phase_slack_vcs.f"],
    "launcher": "836fbb8ced08039a5147e99cfda2ece314eb7f146efd263c4fc1db1e62df2009",
    "metadata": "52b700b1c17172ae5a2d08acacfd9c5bac007893332f9afd9f23c29636e468a0",
    "records": "6de1521b2ee91281eadd5945d6f69b45df2cf5f1e2cc0c93834df4ec4c87190d",
    "offsets": "1cddfc800ba18569b9c0d7f4c193f4b07ddf9046fa7688a1859f6c3e448bf30c",
    "input_manifest": "e77846b6e9c4f130bff5466a8962e221018f8704d6e01005110f34abccd6138a",
    "preflight": "84ab6578421d24aad7a8470c02ca1c4647dce48af93073ad6cf41fe629b1b930",
    "output_manifest": "489e04ddfe34e5c57f6d619d182f332828b18db951ff948eb5054c3384c4e980",
    "run_complete": "159e6b6a6a88be19dd873def3ed6b6c4a81c6bd7718ffb5e9a73e501d3c8e513",
    "compile_directed": "b118d551a735755d42d854570ddcc9ea200da4a937aa7885a6c9acaf98b0d659",
    "sim_directed": "9a17f5d2ec7d527edcc3e38fa3e4bfcfe425024ab2d1e071c7b9daa1509e9df3",
    "assert_directed": "4c61000057d3f1c1d585e27c68214ea8baae7395624e906e9bce4df80fe5d905",
    "compile_actual": "13801da5906993589c8795ec40221c61fae85548f6ddc64574005f40ad3bc5e0",
    "sim_actual": "a2b316dc4a1b786cf5d94338428a487f391a324f9a6bced6a76b1e0bd196ed64",
    "assert_actual": "28dca6e36fe84c48d720eb6a9621ecc5a404172ec45ed1a947e27e487f57fb5f",
    "rc_zero": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "disable_empty": "aecf0c9dffcd66646682ee289bbc1def497a1a7e1b3e3a8fd60586b46ad7c056",
    "simv_directed": "03038d480c8a43b17ba189c8dec1e30aae2f66a47ae30584e42c639ef19bda91",
    "simv_actual": "e9dbb86004da799b1c5cac03383a8187df5e84f613d3182aa16c0f627e3c996f",
}
RUN_FILES = {
    "input_manifest": RUN / "input_sha256.txt",
    "preflight": RUN / "preflight_sha_checks.txt",
    "output_manifest": RUN / "output_sha256.txt",
    "run_complete": RUN / "RUN_COMPLETE.txt",
    "compile_directed": RUN / "compile_directed.raw.log",
    "sim_directed": RUN / "sim_directed.raw.log",
    "assert_directed": RUN / "assert_directed.report",
    "compile_actual": RUN / "compile_actual.raw.log",
    "sim_actual": RUN / "sim_actual.raw.log",
    "assert_actual": RUN / "assert_actual.report",
    "simv_directed": RUN / "simv_directed",
    "simv_actual": RUN / "simv_actual",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key {}".format(key))
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("nonstandard JSON " + value)))


def parse_standard_sha_lines(path):
    rows = {}
    pattern = re.compile(r"^([0-9a-f]{64})  (.+)$")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            digest, name = match.groups()
            require(name not in rows, "duplicate SHA manifest path " + name)
            rows[name] = digest
    return rows


def parse_cover_report(path):
    output = {}
    pattern = re.compile(
        r"\.([A-Za-z0-9_]+),\s+([0-9]+) attempts,\s+([0-9]+) match$")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        require(match is not None, "unexpected cover report line")
        name, attempts, matches = match.groups()
        require(name not in output, "duplicate cover " + name)
        output[name] = {"attempts": int(attempts), "matches": int(matches)}
    require(len(output) == 12, "cover report population")
    return output


def pass_payload(text, prefix):
    rows = [line for line in text.splitlines() if line.startswith(prefix)]
    require(len(rows) == 1, "PASS line population for " + prefix)
    result = {}
    for token in rows[0][len(prefix):].strip().split():
        key, value = token.split("=", 1)
        result[key] = int(value)
    return rows[0], result


def recompute_actual_inputs():
    metadata = METADATA.read_bytes()
    records = RECORDS.read_bytes()
    offset_raw = OFFSETS.read_bytes()
    require(len(metadata) == 127872 and len(metadata) % 74 == 0,
            "metadata geometry")
    phases = len(metadata) // 74
    require(phases == 1728 and len(offset_raw) == (phases + 1) * 4,
            "phase/offset geometry")
    offsets = list(struct.unpack("<{}I".format(phases + 1), offset_raw))
    require(offsets[0] == 0 and offsets[-1] == len(records) and
            all(left < right for left, right in zip(offsets, offsets[1:])),
            "offset coverage/monotonicity")
    code_counts = collections.Counter()
    beats = 0
    entries = 0
    masked_nonzero = 0
    words = {0: 24, 1: 27, 2: 30, 3: 33, 4: 0}
    beat_map = {0: 3, 1: 4, 2: 4, 3: 5, 4: 1}
    for phase in range(phases):
        meta = metadata[phase * 74:(phase + 1) * 74]
        record = records[offsets[phase]:offsets[phase + 1]]
        require(record[:48] == meta[:48], "metadata/header identity")
        packed = int.from_bytes(meta, "little")
        codes = [(packed >> (entry * 3)) & 7 for entry in range(128)]
        require(all(code in words for code in codes), "reserved actual code")
        terminal = sum(words[code] for code in codes)
        require(0 < terminal <= 3680 and len(record) >= 48 + terminal * 4,
                "terminal/record geometry")
        cursor = 0
        for pattern in range(16):
            supplied = (packed >> (384 + pattern * 13)) & 0x1fff
            require(supplied == cursor, "pattern base mismatch")
            for block in range(8):
                code = codes[pattern * 8 + block]
                code_counts[code] += 1
                entries += 1
                beats += beat_map[code]
                if code != 4:
                    final_beat = beat_map[code] - 1
                    valid_words = words[code] - final_beat * 8
                    for word in range(valid_words, 8):
                        word_index = cursor + final_beat * 8 + word
                        value = 0
                        if word_index < terminal:
                            low = 48 + word_index * 4
                            value = int.from_bytes(record[low:low + 4], "little")
                        if value != 0:
                            masked_nonzero += 1
                cursor += words[code]
        require(cursor == terminal, "cursor terminal drift")
    return {
        "phases": phases,
        "entries": entries,
        "outputs": entries,
        "escape": code_counts[4],
        "beats": beats,
        "address_checks": beats,
        "masked_nonzero_words": masked_nonzero,
        "ii_checks": entries - phases,
        "parser_cycles": (phases + 3) * 128,
        "poison_attacks": 3,
        "code_population": dict((str(code), code_counts[code]) for code in range(5)),
    }


def main():
    top_paths = {"contract": CONTRACT, "filelist": FILELIST,
                 "launcher": LAUNCHER, "metadata": METADATA,
                 "records": RECORDS, "offsets": OFFSETS}
    for name, path in top_paths.items():
        require(path.is_file() and sha256(path) == EXPECTED[name], name + " SHA drift")
    for name, path in RUN_FILES.items():
        require(path.is_file() and sha256(path) == EXPECTED[name], name + " SHA drift")
    for name in ("compile_directed.rc", "sim_directed.rc",
                 "compile_actual.rc", "sim_actual.rc"):
        path = RUN / name
        require(path.read_text(encoding="utf-8") == "0\n" and
                sha256(path) == EXPECTED["rc_zero"], name + " nonzero/drift")
    for name in ("assert_directed.report.disablelog",
                 "assert_actual.report.disablelog"):
        path = RUN / name
        require(sha256(path) == EXPECTED["disable_empty"] and
                "Dynamically disabled assertions at End-of-Simulation\n\n" in
                path.read_text(encoding="utf-8"), name + " disable drift")
    require(not (RUN / "RUN_FAILED_OR_INCOMPLETE.txt").exists(), "failed marker exists")

    contract = read_json(CONTRACT)
    require(contract["frozen_sources"] == FROZEN, "contract frozen source drift")
    require(contract["input_identity"] == {
        "m83_phase_records_bytes": 23884000,
        "m83_phase_records_sha256": EXPECTED["records"],
        "m83_offsets_bytes": 6916,
        "m83_offsets_sha256": EXPECTED["offsets"],
        "m85_metadata_bytes": 127872,
        "m85_metadata_sha256": EXPECTED["metadata"],
    }, "contract input identity drift")
    for relative, digest in FROZEN.items():
        require(sha256(HW / relative) == digest, "frozen source drift " + relative)
    expected_filelist = [
        "rtl_m82/zero_bubble_elastic_pwp_stream.sv",
        "rtl_m85/guarded_wordpacked_pwp_stream.sv",
        "rtl_m99/phase_slack_guarded_wordpacked_pwp_stream.sv",
        "verif_m99/phase_slack_guarded_wordpacked_pwp_stream_assertions.sv",
        "tb_m99/tb_m99_phase_slack_guarded_wordpacked_pwp_stream.sv",
        "tb_m99/tb_m99_phase_slack_actual_records.sv",
    ]
    require(FILELIST.read_text(encoding="utf-8").splitlines() == expected_filelist,
            "filelist content/order")

    preflight = RUN_FILES["preflight"].read_text(encoding="utf-8").splitlines()
    require(len(preflight) == 9, "preflight population")
    preflight_pattern = re.compile(
        r"^path=(.+) expected=([0-9a-f]{64}) observed=([0-9a-f]{64})$")
    preflight_rows = {}
    for line in preflight:
        match = preflight_pattern.match(line)
        require(match is not None, "preflight syntax")
        name, expected_digest, observed = match.groups()
        require(expected_digest == observed and name not in preflight_rows,
                "preflight mismatch/duplicate")
        preflight_rows[name] = observed
    expected_preflight = dict(FROZEN)
    expected_preflight[
        "results/m85_canonical_74b_phase_metadata_r1_20260823/m85_phase_metadata_74b.bin"] = EXPECTED["metadata"]
    expected_preflight[
        "contracts/m99_phase_slack_metadata_compiler_vcs_contract_r1_20260824.json"] = EXPECTED["contract"]
    require(preflight_rows == expected_preflight, "preflight set drift")

    input_rows = parse_standard_sha_lines(RUN_FILES["input_manifest"])
    require(input_rows == expected_preflight, "input manifest standard rows")
    input_text = RUN_FILES["input_manifest"].read_text(encoding="utf-8")
    require("external_path={} sha256={} bytes=23884000".format(
                RECORDS, EXPECTED["records"]) in input_text and
            "external_path={} sha256={} bytes=6916".format(
                OFFSETS, EXPECTED["offsets"]) in input_text,
            "external input manifest rows")

    expected_output = {
        str(RUN / "compile_actual.raw.log"): EXPECTED["compile_actual"],
        str(RUN / "compile_directed.raw.log"): EXPECTED["compile_directed"],
        str(RUN / "sim_actual.raw.log"): EXPECTED["sim_actual"],
        str(RUN / "sim_directed.raw.log"): EXPECTED["sim_directed"],
        str(RUN / "assert_actual.report"): EXPECTED["assert_actual"],
        str(RUN / "assert_directed.report"): EXPECTED["assert_directed"],
        str(RUN / "RUN_COMPLETE.txt"): EXPECTED["run_complete"],
    }
    output_rows = parse_standard_sha_lines(RUN_FILES["output_manifest"])
    require(output_rows == expected_output and
            all(sha256(path) == digest for path, digest in output_rows.items()),
            "output manifest drift")

    launcher = LAUNCHER.read_text(encoding="utf-8")
    for token in ("set -euo pipefail", "refusing to overwrite M99 sealed VCS run",
                  "EXPECTED_SHA", "preflight_sha_checks.txt",
                  "+define+SVA_RUNTIME_ENABLED", "compile_directed.rc",
                  "sim_directed.rc", "compile_actual.rc", "sim_actual.rc",
                  "output_sha256.txt", "run_complete=1"):
        require(token in launcher, "launcher guard missing " + token)
    launcher_bound_by_contract = (
        "dc_handoff/scripts/run_vcs_m99_phase_slack_actual_records_sva.sh" in
        contract.get("frozen_sources", {}))
    launcher_bound_by_run = EXPECTED["launcher"] in (
        RUN_FILES["input_manifest"].read_text(encoding="utf-8") +
        RUN_FILES["preflight"].read_text(encoding="utf-8"))

    forbidden_compile = re.compile(r"Warning-\[|Error-\[|^Error", re.MULTILINE)
    forbidden_sim = re.compile(
        r"failed at|Offending|^Error|^Fatal|watchdog timeout", re.I | re.MULTILINE)
    for name in ("directed", "actual"):
        compile_text = RUN_FILES["compile_" + name].read_text(encoding="utf-8")
        sim_text = RUN_FILES["sim_" + name].read_text(encoding="utf-8")
        require("Version V-2023.12-SP1_Full64" in compile_text and
                "Runtime version V-2023.12-SP1_Full64" in sim_text,
                name + " VCS version")
        require(not forbidden_compile.search(compile_text), name + " compile signature")
        require(not forbidden_sim.search(sim_text), name + " sim signature")
        require("$finish at simulation time" in sim_text, name + " finish")

    directed_line, directed = pass_payload(
        RUN_FILES["sim_directed"].read_text(encoding="utf-8"),
        "PASS M99 M85-differential ")
    actual_line, actual = pass_payload(
        RUN_FILES["sim_actual"].read_text(encoding="utf-8"),
        "PASS M99 actual-record differential ")
    require(directed_line == contract["directed_vcs"]["expected_pass_line"],
            "directed contract PASS drift")
    require(actual_line == contract["actual_record_vcs"]["expected_pass_line"],
            "actual contract PASS drift")
    expected_directed = {
        "entries": 128, "beats": 436, "parser_cycles": 640, "stalls": 10,
        "poison_attacks": 3, "early_lookup_attacks": 1,
        "simultaneous_unloaded_attacks": 1,
        "simultaneous_loaded_priority_attacks": 1,
    }
    require(directed == expected_directed, "directed counter drift")

    actual_recompute = recompute_actual_inputs()
    expected_actual = dict((key, value) for key, value in actual_recompute.items()
                           if key != "code_population")
    require(actual == expected_actual, "actual input/PASS recompute drift")
    require(actual_recompute["entries"] == actual_recompute["phases"] * 128 and
            actual_recompute["parser_cycles"] == (1728 + 3) * 128 and
            actual_recompute["ii_checks"] == 1728 * 127 and
            actual_recompute["address_checks"] == actual_recompute["beats"],
            "population arithmetic")

    directed_covers = parse_cover_report(RUN_FILES["assert_directed"])
    actual_covers = parse_cover_report(RUN_FILES["assert_actual"])
    expected_directed_matches = {
        "cp_phase_load": 6, "cp_simultaneous_load_lookup": 1,
        "cp_loaded_lookup_priority": 1, "cp_parser_first_entry": 6,
        "cp_parser_middle_entry": 5, "cp_parser_final_entry": 5,
        "cp_lookup_stall": 1, "cp_escape": 28, "cp_width9": 28,
        "cp_width10": 28, "cp_width11": 27, "cp_metadata_error": 5,
    }
    require(all(directed_covers[name] == {"attempts": 1136, "matches": matches}
                for name, matches in expected_directed_matches.items()),
            "directed cover drift")
    expected_actual_matches = {
        "cp_phase_load": 1731, "cp_simultaneous_load_lookup": 0,
        "cp_loaded_lookup_priority": 0, "cp_parser_first_entry": 1731,
        "cp_parser_middle_entry": 1731, "cp_parser_final_entry": 1731,
        "cp_lookup_stall": 0, "cp_escape": actual_recompute["code_population"]["4"],
        "cp_width9": actual_recompute["code_population"]["1"],
        "cp_width10": actual_recompute["code_population"]["2"],
        "cp_width11": actual_recompute["code_population"]["3"],
        "cp_metadata_error": 5,
    }
    require(all(actual_covers[name] == {"attempts": 1063886, "matches": matches}
                for name, matches in expected_actual_matches.items()),
            "actual cover/input population drift")
    for name in ("directed", "actual"):
        sim_cover = "\n".join(
            line for line in RUN_FILES["sim_" + name].read_text(
                encoding="utf-8").splitlines() if " attempts," in line)
        require(sim_cover == RUN_FILES["assert_" + name].read_text(
            encoding="utf-8").rstrip("\n"), name + " sim/report cover drift")

    sva = (HW / "verif_m99/phase_slack_guarded_wordpacked_pwp_stream_assertions.sv").read_text(
        encoding="utf-8")
    tb_directed = (HW / "tb_m99/tb_m99_phase_slack_guarded_wordpacked_pwp_stream.sv").read_text(
        encoding="utf-8")
    assertion_labels = re.findall(r"^\s*(ap_[A-Za-z0-9_]+): assert property", sva,
                                  re.MULTILINE)
    require(len(assertion_labels) == 20 and len(set(assertion_labels)) == 20,
            "assertion label population")
    for token in ("ap_simultaneous_request_never_double_accepts",
                  "|-> !(phase_load_ready && lookup_ready)",
                  "ap_parser_starts_at_entry_zero", "ap_parser_progresses_one_entry",
                  "ap_parser_finishes_after_entry_127", "ap_parser_cursor_delta",
                  "ap_captured_metadata_stable_during_parse",
                  "ap_parser_poison_monotonic", "ap_parser_blocks_datapath_accept",
                  "ap_early_lookup_sets_sticky_error"):
        require(token in sva, "corrected/internal SVA missing " + token)
    require("cp_loaded_lookup_priority" in sva and
            "simultaneous_loaded_priority_attacks=1" in tb_directed and
            "!dut_lookup_ready" in tb_directed and
            "loaded simultaneous request did not prioritize lookup" in tb_directed,
            "loaded priority dynamic scenario drift")

    run_complete = dict(line.split("=", 1) for line in
                        RUN_FILES["run_complete"].read_text(
                            encoding="utf-8").splitlines())
    require(run_complete == {
        "status": "PASS_M99_DIRECTED_AND_ACTUAL_RECORD_VCS_SVA",
        "exact_sha": "true", "directed_entries": "128",
        "actual_phases": "1728", "actual_entries": "221184",
        "actual_outputs": "221184", "actual_beats": "835383",
        "bank_address_checks": "835383", "parser_edges_per_phase": "128",
        "current_m86_zero_incremental_parser_cycles": "false",
        "dc_admitted": "false", "paper_ppa_ready": "false",
        "system_speedup": "false", "headline": "false",
    }, "RUN_COMPLETE drift")

    output = {
        "schema": "m99_phase_slack_sealed_vcs_independent_audit_r2_v1",
        "status": "PASS_SCOPED_VCS_ADMISSION_GO_SAME_FLOW_3NS_DC_AB",
        "producer_launcher_or_simulator_executed_by_reviewer": False,
        "identity": {
            "contract_sha256": sha256(CONTRACT),
            "filelist_sha256": sha256(FILELIST),
            "launcher_observed_sha256": sha256(LAUNCHER),
            "launcher_bound_by_contract": launcher_bound_by_contract,
            "launcher_bound_by_sealed_input_manifest": launcher_bound_by_run,
            "frozen_sources": FROZEN,
            "records_sha256": sha256(RECORDS),
            "offsets_sha256": sha256(OFFSETS),
            "metadata_sha256": sha256(METADATA),
        },
        "sealed_run": {
            "all_four_rc_zero": True,
            "failed_or_incomplete_marker_absent": True,
            "preflight_nine_of_nine_exact": True,
            "input_manifest_exact": True,
            "output_manifest_seven_of_seven_exact": True,
            "assertions_disabled": False,
            "compile_warning_error_signatures": 0,
            "simulation_failure_signatures": 0,
            "directed": directed,
            "actual": actual,
        },
        "actual_input_recompute": actual_recompute,
        "cover_reports": {"directed": directed_covers, "actual": actual_covers},
        "sva_audit": {
            "assert_property_count": len(assertion_labels),
            "loaded_old_phase_priority_assertion_corrected": True,
            "loaded_old_phase_priority_cover_matches": 1,
            "parser_start_progress_finish_cursor_capture_poison_and_blocking_asserted": True,
            "actual_first_middle_final_cover_matches_each": 1731,
        },
        "manifest_scope": {
            "output_manifest_includes": sorted(Path(path).name for path in output_rows),
            "output_manifest_omits": [
                "launcher", "input_sha256.txt", "preflight_sha_checks.txt",
                "four rc files", "two assertion disablelogs", "two simv executables"
            ],
        },
        "admission": {
            "scoped_vcs_functional_admission": True,
            "go_same_flow_m99_3ns_dc_ab": True,
            "dc_or_ppa_currently_admitted": False,
            "current_m86_zero_incremental_parser_cycles": False,
            "paper_ppa_ready": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M99 sealed exact-SHA VCS independent audit r2")
    print("rc=0/0/0/0 preflight=9/9 output_manifest=7/7 assertion_failures=0")
    print("directed entries=128 beats=436 parser_cycles=640 loaded_priority=1")
    print("actual phases=1728 entries=221184 outputs=221184 beats=835383 addresses=835383 parser_cycles=221568")
    print("scoped_vcs_admission=true go_same_flow_3ns_dc_ab=true dc_admitted=false")
    print("launcher_sha={} launcher_run_bound=false".format(EXPECTED["launcher"]))
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
