#!/usr/libexec/platform-python3.6
"""Fail-closed M606 exact runner for the bounded parent-scratch energy model."""

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
PYTHON = Path("/usr/libexec/platform-python3.6")
PYTHON_SHA = "9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f"
ADAPTER_REL = "hw_autoresearch_nts07/system_simulator/scripts/analyze_m606_m597_m593_parent_scratch_generated_macro_energy_r3.py"
ADAPTER_SHA = "69d5c2c521b84aee589b28531574d95ec621dfdeeaf35d517cc0bb386e87782d"
UPSTREAM_REL = "hw_autoresearch_nts07/system_simulator/scripts/analyze_m597_m593_m528_parent_scratch_generated_macro_energy_r2.py"
UPSTREAM_SHA = "6896c8a406dc3274926e6c7d958136aca47b9df9afa3522d6c2539a142ea9cf9"
CONTRACT_REL = "hw_autoresearch_nts07/contracts/m597_m593_m528_parent_scratch_generated_macro_energy_source_contract_r2_20260828.json"
CONTRACT_SHA = "90399b6c932e28f6eac38f3408af0374b23beb369e1fd4e57e3b98d92d28b1bf"
M602_REL = "hw_autoresearch_nts07/system_simulator/scripts/run_m602_m593_parent_scratch_generated_macro_energy_r2_exact_sha.sh"
M602_SHA = "6a54d938f598835114c2e463e56eb03f4e0754947dbbeb0b33f03fd04e569b2c"
M604_REL = "hw_autoresearch_nts07/reviews/m604_m602_m593_parent_scratch_energy_exact_runner_static_hammer_r1_20260828"
M604_REVIEW_SHA = "4650c261a292c544a66c563ec123f996be65de45102a4f7d03b2447ce22e8f7a"
M604_MANIFEST_SHA = "f60a5941d3948fcc2c552e6107f351e6a9b2e05638d02d29540ec8efbfec7207"
M604_OUTER_SHA = "7a8de8aa743f47ed05ca27bb823c3fe2593bfca417bcafc6785bdfd6dcf14a2a"
DOCS_REL = "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
DOCS_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
RESULT = HW / "results/m606_m597_m593_parent_scratch_generated_macro_energy_r3_20260828"
ATTEMPT = HW / "results/m606_m597_m593_parent_scratch_generated_macro_energy_r3_20260828.attempt"
CONSUMED = HW / "results/m606_m597_m593_parent_scratch_generated_macro_energy_r3_20260828.attempt.consumed"
AUTH = HW / "contracts/m608_m606_m593_parent_scratch_energy_true_launch_admission_r1_20260828.json"
M607_REVIEW = HW / "reviews/m607_m606_m593_parent_scratch_energy_exact_runner_static_hammer_r1_20260828/review.json"
RESULT_JSON = "m597_m593_m528_parent_scratch_generated_macro_energy_result_r2.json"
CSV_NAME = "m597_parent_scratch_energy_rows_r2.csv"
COMPLETE = "RUN_COMPLETE.txt"
COMPLETE_TOKEN = "PASS_M597_R2_ANALYZER_OUTPUT_PENDING_INDEPENDENT_RESULT_HAMMER\n"


class Failure(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Failure(message)


def lexists(path):
    return os.path.lexists(str(path))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=pairs,
                          parse_constant=lambda raw: (_ for _ in ()).throw(Failure(raw)))
    require(isinstance(value, dict), "top-level JSON is not object")
    def finite(node):
        if isinstance(node, float):
            require(math.isfinite(node), "non-finite JSON")
        elif isinstance(node, dict):
            for child in node.values(): finite(child)
        elif isinstance(node, list):
            for child in node: finite(child)
    finite(value)
    return value


def plain(path, directory=False):
    path = Path(path)
    require(lexists(path), "missing path: " + str(path))
    mode = os.lstat(str(path)).st_mode
    require(not stat.S_ISLNK(mode), "symlink path: " + str(path))
    require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode),
            "wrong path type: " + str(path))


def plain_chain(path, directory=False):
    path = Path(path).resolve()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        plain(current, directory=(current != path or directory))
    return path


def rename_noreplace(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    require(hasattr(libc, "renameat2"), "renameat2 unavailable")
    fn = libc.renameat2
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                   ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    if fn(-100, os.fsencode(str(source)), -100, os.fsencode(str(target)), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def write_exclusive(path, content):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def seal_tree(directory):
    directory = Path(directory)
    manifest, outer = directory / "SHA256SUMS", directory / "SHA256SUMS.seal.sha256"
    require(not lexists(manifest) and not lexists(outer), "seal target exists")
    files = []
    for root, dirs, names in os.walk(str(directory), topdown=True, followlinks=False):
        for name in dirs:
            plain(Path(root) / name, directory=True)
        for name in names:
            path = Path(root) / name
            plain(path)
            files.append(path)
    files.sort(key=lambda item: str(item.relative_to(directory)))
    write_exclusive(manifest, "".join("%s  %s\n" %
        (sha(path), str(path.relative_to(directory))) for path in files))
    write_exclusive(outer, "%s  SHA256SUMS\n" % sha(manifest))


def verify_seal(directory, expected_members=None):
    directory = Path(directory)
    plain(directory, directory=True)
    manifest, outer = directory / "SHA256SUMS", directory / "SHA256SUMS.seal.sha256"
    plain(manifest); plain(outer)
    require(outer.read_text(encoding="utf-8").strip().split() ==
            [sha(manifest), "SHA256SUMS"], "outer seal mismatch")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1); name = name.lstrip("*")
        require(name not in listed and not os.path.isabs(name) and
                os.path.normpath(name) == name and not name.startswith(".."),
                "unsafe/duplicate manifest member")
        member = directory / name
        plain(member)
        require(sha(member) == digest, "member SHA mismatch: " + name)
        listed[name] = digest
    actual = set()
    for root, dirs, names in os.walk(str(directory), followlinks=False):
        for name in dirs: plain(Path(root) / name, directory=True)
        for name in names:
            path = Path(root) / name; plain(path)
            if path not in (manifest, outer):
                actual.add(str(path.relative_to(directory)))
    require(actual == set(listed), "sealed member set mismatch")
    if expected_members is not None:
        require(actual == set(expected_members), "unexpected exact member set")
    return sha(manifest), sha(outer)


def verify_static(shell_path, python_runner_path):
    identities = {
        REPO / ADAPTER_REL: ADAPTER_SHA,
        REPO / UPSTREAM_REL: UPSTREAM_SHA,
        REPO / CONTRACT_REL: CONTRACT_SHA,
        REPO / M602_REL: M602_SHA,
        REPO / DOCS_REL: DOCS_SHA,
        Path(python_runner_path): sha(Path(python_runner_path)),
        Path(shell_path): sha(Path(shell_path)),
    }
    for path, expected in identities.items():
        plain_chain(path)
        require(sha(path) == expected, "static SHA drift: " + str(path))
    m604 = REPO / M604_REL
    manifest_sha, outer_sha = verify_seal(m604,
        {"review.json", "review.md"})
    require(manifest_sha == M604_MANIFEST_SHA and outer_sha == M604_OUTER_SHA and
            sha(m604 / "review.json") == M604_REVIEW_SHA,
            "M604 identity drift")
    review = strict(m604 / "review.json")
    require((review.get("p0_count"), review.get("p1_count")) == (2, 2) and
            review.get("authorization", {}).get("true_launch_admission_authoring_allowed") is False,
            "M604 fail lineage drift")
    cp = subprocess.run([str(REPO / M602_REL), "--preflight-only"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        universal_newlines=True)
    require(cp.returncode == 0 and
            cp.stdout.strip() == "PASS_M602_M593_SOURCE_PREFLIGHT_ONLY__NO_RESULT_ATTEMPT_OR_LAUNCH",
            "frozen upstream identity preflight failed: " + cp.stderr)


def verify_coordinates(staging):
    plain_chain(HW / "results", directory=True)
    for path in (RESULT, ATTEMPT, CONSUMED, staging):
        require(path.parent == HW / "results", "coordinate parent drift")
        require(not lexists(path), "coordinate exists: " + str(path))


AUTH_KEYS = {"admission_id", "date", "status", "launch_now", "release",
             "runner", "source_static_hammer", "upstream", "canonical",
             "claim_boundary"}


def verify_authorization(shell_path, python_runner_path):
    require(AUTH == HW / "contracts/m608_m606_m593_parent_scratch_energy_true_launch_admission_r1_20260828.json",
            "authorization coordinate drift")
    plain_chain(AUTH); plain(Path(str(AUTH) + ".sha256")); plain(Path(str(AUTH) + ".sha256.seal.sha256"))
    side, outer = Path(str(AUTH) + ".sha256"), Path(str(AUTH) + ".sha256.seal.sha256")
    auth_sha = sha(AUTH)
    require(side.read_text().strip().split() == [auth_sha, AUTH.name], "auth sidecar drift")
    require(outer.read_text().strip().split() == [sha(side), side.name], "auth outer drift")
    value = strict(AUTH)
    require(set(value) == AUTH_KEYS and
            value["admission_id"] == "m608_m606_m593_parent_scratch_energy_true_launch_admission_r1_20260828" and
            value["status"] == "TRUE_LAUNCH_ADMISSION__FRESH_M607_P0_P1_ZERO_REQUIRED" and
            value["launch_now"] is True and value["release"] is True,
            "authorization predicate drift")
    expected_runner = {
        "shell_path": str(Path(shell_path).relative_to(REPO)), "shell_sha256": sha(shell_path),
        "python_path": str(Path(python_runner_path).relative_to(REPO)), "python_sha256": sha(python_runner_path),
        "adapter_path": ADAPTER_REL, "adapter_sha256": ADAPTER_SHA}
    require(value["runner"] == expected_runner, "authorization runner drift")
    require(value["canonical"] == {
        "result_dir": str(RESULT.relative_to(HW)),
        "attempt_dir": str(ATTEMPT.relative_to(HW)),
        "consumed_attempt_dir": str(CONSUMED.relative_to(HW))},
        "authorization coordinates drift")
    require(value["upstream"] == {
        "m597_contract_sha256": CONTRACT_SHA, "m597_analyzer_sha256": UPSTREAM_SHA,
        "m606_adapter_sha256": ADAPTER_SHA, "m604_failed_review_sha256": M604_REVIEW_SHA},
        "authorization upstream drift")
    hammer = value["source_static_hammer"]
    require(set(hammer) == {"path", "sha256", "manifest_sha256", "outer_seal_file_sha256"} and
            REPO / hammer["path"] == M607_REVIEW, "M607 hammer coordinate drift")
    review_dir = M607_REVIEW.parent
    manifest_sha, outer_sha = verify_seal(review_dir, {"review.json", "review.md"})
    require(sha(M607_REVIEW) == hammer["sha256"] and
            manifest_sha == hammer["manifest_sha256"] and outer_sha == hammer["outer_seal_file_sha256"],
            "M607 hammer seal drift")
    review = strict(M607_REVIEW)
    require(set(review) >= {"schema", "status", "score_0_to_100", "p0_count", "p1_count", "authorization"} and
            review["schema"] == "m607_m606_m593_parent_scratch_energy_exact_runner_static_hammer_v1" and
            review["status"] == "PASS_RUNNER_STATIC__TRUE_LAUNCH_ADMISSION_AUTHORING_ONLY__NO_EXECUTION" and
            review["score_0_to_100"] == 100 and
            (review["p0_count"], review["p1_count"]) == (0, 0) and
            review["authorization"].get("true_launch_admission_authoring_allowed") is True,
            "M607 hammer predicate drift")
    require(value["claim_boundary"] == {"component_only": True, "paper_data": False,
            "system_energy": False, "result_hammer_pending": True},
            "authorization claim drift")
    return auth_sha


ROW_KEYS = {"design", "cycle_source", "traffic_source", "cycles_s10",
    "latency_ms_per_frozen_sampled_inference_at_3ns", "macro_reads_per_output_block",
    "raw_forwards_per_output_block", "macro_writes_per_output_block",
    "parent_edges_per_output_block", "active_rows_per_output_block", "output_block_banks",
    "logical_word_bytes", "read_accesses_s10", "write_accesses_s10", "read_bytes_s10",
    "write_bytes_s10", "raw_forward_macro_read_energy_charged",
    "read_plus_forward_equals_parent_edges", "writes_do_not_exceed_active_rows",
    "dynamic_energy_mj_per_frozen_sampled_inference",
    "leakage_energy_mj_per_frozen_sampled_inference",
    "modeled_parent_scratch_energy_mj_per_frozen_sampled_inference"}


def verify_result(directory, final=False, shell_path=None, python_runner_path=None, auth_sha=None):
    directory = Path(directory)
    base = {RESULT_JSON, CSV_NAME, COMPLETE}
    final_extra = {"production_stdout.log", "production_stderr.log", "m606_terminal_rehash_receipt.json"}
    expected = base | (final_extra if final else set())
    verify_seal(directory, expected)
    require((directory / COMPLETE).read_text(encoding="utf-8") == COMPLETE_TOKEN,
            "RUN_COMPLETE token drift")
    result = strict(directory / RESULT_JSON)
    require(set(result) == {"schema", "date", "status", "identity", "scope", "macro",
                           "rows", "conservation", "ablation", "claim_boundary"},
            "result top-level key set drift")
    require(result["schema"] == "m597_m593_m528_parent_scratch_generated_macro_energy_result_v2" and
            result["date"] == "2026-08-28" and
            result["status"] == "PASS_BOUNDED_GENERATED_MACRO_COMPONENT_MODEL__PENDING_FRESH_INDEPENDENT_RESULT_HAMMER",
            "result identity drift")
    identity = result["identity"]
    require(set(identity) == {"source_contract", "frozen_inputs"} and
            identity["source_contract"] == {"path": CONTRACT_REL, "sha256": CONTRACT_SHA,
                                             "exact_key_set_pass": True},
            "result source identity drift")
    contract = strict(REPO / CONTRACT_REL)
    require(set(identity["frozen_inputs"]) == set(contract["frozen_inputs"]),
            "result frozen input population drift")
    for name, expected_input in contract["frozen_inputs"].items():
        observed = identity["frozen_inputs"][name]
        require(observed.get("path") == expected_input["path"] and
                observed.get("sha256") == expected_input["sha256"],
                "result frozen input drift: " + name)
        if "directory" in expected_input:
            require(observed == dict(expected_input, double_seal_pass="true"),
                    "result sealed input receipt drift: " + name)
        else:
            require(observed == {"path": expected_input["path"], "sha256": expected_input["sha256"]},
                    "result file receipt drift: " + name)
    require(result["scope"] == {
        "checkpoint": "H67 ep35",
        "sequence_count": 1,
        "frozen_sampled_inference_count": 10,
        "sample_is_camera_frame": False,
        "operators": "four bottleneck Conv3x3 only",
        "component": "nine generated 128x128-bit 1RW parent-scratch macros only",
        "corner": "ssg0p9v125c at 0.9 V",
        "clock_period_ns": 3.0,
        "leakage_assumption": "all nine macros remain powered for the complete modeled four-Conv sample schedule; no power gating credited",
    }, "result scope drift")
    read_energy_pj = 9.0 * 11.6754 * 0.9
    write_energy_pj = 9.0 * 11.1923 * 0.9
    leakage_power_mw = 9.0 * 66.6783 * 0.9 / 1000.0
    require(result["macro"] == {
        "cell": "TS1N28HPCPHVTB128X128M4S",
        "count": 9,
        "area_um2": 9.0 * 8758.3606,
        "cycle_ns": 0.616,
        "access_ns": 0.4679,
        "full_1152b_read_energy_pj_per_physical_macro_access": read_energy_pj,
        "full_1152b_write_energy_pj_per_physical_macro_access": write_energy_pj,
        "leakage_power_mw": leakage_power_mw,
        "model_note": "generated-macro slow-corner datasheet current; all nine 128-bit slices activated per physical 1152-bit access",
    }, "result macro schema/value drift")
    rows = result["rows"]
    require(isinstance(rows, list) and len(rows) == 2 and
            [row.get("design") for row in rows] ==
            ["m504_all_write_1rw_parent_scratch", "m528_dead_write_only_1rw_parent_scratch"] and
            all(set(row) == ROW_KEYS for row in rows), "result row schema/order drift")
    anchors = [
        (456016645, 16490761, 1714628, 27305568, 31456014336,
         "sealed M504 result: cycle_comparison.deadline_lookahead_single_port_cycles",
         "sealed M504 result: aggregate_one_output_block.deadline_macro_reads/deadline_forwarded_reads/deadline_macro_writes"),
        (435293339, 16490761, 1714628, 9947701, 11459751552,
         "sealed M528 result: aggregate_cycles.m505_dead_write_only_1rw_cycles",
         "sealed M528 result: traffic row m505_dead_write_only_1rw; forward split cross-checked by sealed M528 hammer"),
    ]
    for row, anchor in zip(rows, anchors):
        require((row["cycles_s10"], row["macro_reads_per_output_block"],
                 row["raw_forwards_per_output_block"], row["macro_writes_per_output_block"],
                 row["write_bytes_s10"]) == anchor[:5] and
                row["parent_edges_per_output_block"] == 18205389 and
                row["active_rows_per_output_block"] == 27305568 and
                row["output_block_banks"] == 8 and row["logical_word_bytes"] == 144 and
                row["read_accesses_s10"] == 16490761 * 8 and
                row["write_accesses_s10"] == anchor[3] * 8 and
                row["read_bytes_s10"] == 18997356672 and
                row["raw_forward_macro_read_energy_charged"] is False and
                row["read_plus_forward_equals_parent_edges"] is True and
                row["writes_do_not_exceed_active_rows"] is True,
                "row anchor/conservation drift")
        require(row["cycle_source"] == anchor[5] and row["traffic_source"] == anchor[6],
                "row source identity drift")
        latency_ms = row["cycles_s10"] * 3.0 / 10.0 / 1.0e6
        dynamic_mj = (row["read_accesses_s10"] * read_energy_pj +
                      row["write_accesses_s10"] * write_energy_pj) / 10.0 / 1.0e9
        leakage_mj = leakage_power_mw * latency_ms / 1000.0
        require(math.isclose(row["latency_ms_per_frozen_sampled_inference_at_3ns"],
                             latency_ms, rel_tol=0.0, abs_tol=1e-15) and
                math.isclose(row["dynamic_energy_mj_per_frozen_sampled_inference"],
                             dynamic_mj, rel_tol=0.0, abs_tol=1e-15) and
                math.isclose(row["leakage_energy_mj_per_frozen_sampled_inference"],
                             leakage_mj, rel_tol=0.0, abs_tol=1e-15) and
                math.isclose(row["modeled_parent_scratch_energy_mj_per_frozen_sampled_inference"],
                             dynamic_mj + leakage_mj, rel_tol=0.0, abs_tol=1e-15),
                "row physical energy equation drift")
    conservation = result["conservation"]
    require(set(conservation) == {"m504_macro_reads_plus_raw_forwards_equal_parent_edges",
        "m504_macro_writes_equal_active_rows", "m504_and_dead_only_macro_reads_equal",
        "dead_macro_writes_plus_dead_elisions_equal_active_rows",
        "raw_forwards_charged_as_macro_reads", "all_byte_counts_are_accesses_times_144",
        "all_equalities_pass"} and all(conservation[key] is True for key in conservation
        if key != "raw_forwards_charged_as_macro_reads") and
        conservation["raw_forwards_charged_as_macro_reads"] is False,
        "conservation schema/value drift")
    ablation = result["ablation"]
    require(set(ablation) == {"dead_write_only_cycle_speedup_vs_m504_all_write",
        "dead_write_only_parent_scratch_component_energy_reduction_fraction",
        "dead_write_only_parent_scratch_component_energy_reduction_percent",
        "dead_write_only_parent_scratch_component_energy_saved_mj_per_frozen_sampled_inference",
        "label"} and
        math.isclose(ablation["dead_write_only_cycle_speedup_vs_m504_all_write"],
                     456016645.0 / 435293339.0, rel_tol=0, abs_tol=1e-15) and
        math.isclose(ablation["dead_write_only_parent_scratch_component_energy_reduction_fraction"],
                     0.38228307918921945, rel_tol=0, abs_tol=1e-15) and
        math.isclose(ablation["dead_write_only_parent_scratch_component_energy_reduction_percent"],
        38.228307918921945, rel_tol=0, abs_tol=1e-12) and
        math.isclose(ablation["dead_write_only_parent_scratch_component_energy_saved_mj_per_frozen_sampled_inference"],
        1.2622562286593053, rel_tol=0, abs_tol=1e-12) and
        ablation["label"] == "generated-macro datasheet component ablation on ten frozen sampled inferences; pending independent result hammer",
        "ablation drift")
    claim = result["claim_boundary"]
    require(set(claim) == {"allowed_label_after_independent_result_hammer", "component_energy_model",
        "sealed_trace_physical_macro_access_counts", "per_frozen_sampled_inference", "sample_is_camera_frame",
        "rtl_integrated_macro_ppa", "interconnect_or_clock_tree_energy", "logic_energy", "other_sram_energy",
        "dram_energy", "c1_total_energy", "full_network_energy", "energy_per_system_frame", "system_energy",
        "silicon_measurement", "system_speedup", "date_headline", "result_hammer_pending"} and
        claim["allowed_label_after_independent_result_hammer"] ==
            "generated-macro datasheet component model for M528 parent scratch" and
        claim["component_energy_model"] is True and claim["sealed_trace_physical_macro_access_counts"] is True and
        claim["per_frozen_sampled_inference"] is True and claim["result_hammer_pending"] is True and
        all(claim[key] is False for key in ("sample_is_camera_frame", "rtl_integrated_macro_ppa",
            "interconnect_or_clock_tree_energy", "logic_energy", "other_sram_energy", "dram_energy",
            "c1_total_energy", "full_network_energy", "energy_per_system_frame", "system_energy",
            "silicon_measurement", "system_speedup", "date_headline")), "claim boundary drift")
    with (directory / CSV_NAME).open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == 2 and set(csv_rows[0]) == ROW_KEYS,
            "CSV schema/population drift")
    for jrow, crow in zip(rows, csv_rows):
        require(all(crow[key] == str(jrow[key]) for key in ROW_KEYS),
                "CSV/JSON row mismatch")
    if final:
        receipt = strict(directory / "m606_terminal_rehash_receipt.json")
        five = {RESULT_JSON, CSV_NAME, COMPLETE, "production_stdout.log", "production_stderr.log"}
        require(set(receipt) == {"schema", "status", "runner", "adapter", "upstream_analyzer",
            "source_contract", "authorization", "output_schema", "output_status",
            "output_members_preseal", "claim"} and
            receipt["schema"] == "m606_m593_energy_terminal_rehash_receipt_v1" and
            receipt["status"] == "PASS_M606_TERMINAL_IDENTITY_AND_OUTPUT_REHASH" and
            receipt["runner"] == {"shell_path": str(Path(shell_path).relative_to(REPO)),
                "shell_sha256": sha(shell_path), "python_path": str(Path(python_runner_path).relative_to(REPO)),
                "python_sha256": sha(python_runner_path)} and
            receipt["adapter"] == {"path": ADAPTER_REL, "sha256": ADAPTER_SHA} and
            receipt["upstream_analyzer"] == {"path": UPSTREAM_REL, "sha256": UPSTREAM_SHA} and
            receipt["source_contract"] == {"path": CONTRACT_REL, "sha256": CONTRACT_SHA} and
            receipt["authorization"] == {"path": str(AUTH.relative_to(REPO)), "sha256": auth_sha} and
            receipt["output_schema"] == result["schema"] and receipt["output_status"] == result["status"] and
            set(receipt["output_members_preseal"]) == five and
            all(receipt["output_members_preseal"][name] == sha(directory / name) for name in five) and
            receipt["claim"] == "component-only per-frozen-sampled-inference; pending independent result hammer; not paper data",
            "terminal receipt exactness drift")


def remove_top_seal(directory):
    for name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
        path = Path(directory) / name; plain(path); os.unlink(str(path))


def quarantine_failure(staging, shell_path, python_runner_path, error, stage, auth_sha):
    parent = HW / "results"
    stamp = "%d.%d" % (int(time.time() * 1000000), os.getpid())
    qstage = parent / (".m606_energy.failed_quarantine.staging." + stamp)
    qfinal = parent / ("m606_energy.failed_or_incomplete." + stamp)
    require(not lexists(qstage) and not lexists(qfinal), "quarantine collision")
    os.mkdir(str(qstage), 0o700)
    coordinates = [("canonical_result", RESULT), ("attempt", ATTEMPT),
                   ("consumed_attempt", CONSUMED), ("runner_staging", staging)]
    for name, path in coordinates:
        if lexists(path): rename_noreplace(path, qstage / name)
    prefix = "." + staging.name + ".m606_staging_"
    index = 0
    for entry in sorted(os.scandir(str(parent)), key=lambda item: item.name):
        if entry.name.startswith(prefix):
            index += 1
            rename_noreplace(Path(entry.path), qstage / ("adapter_internal_staging_%d" % index))
    receipt = {"schema": "m606_m593_energy_failed_attempt_quarantine_v1",
        "status": "FAILED_OR_INTERRUPTED_ALL_COORDINATES_QUARANTINED",
        "failure_stage": stage, "exception_type": type(error).__name__, "message": str(error),
        "runner": {"shell_path": str(Path(shell_path)), "shell_sha256": sha(shell_path),
                   "python_path": str(Path(python_runner_path)), "python_sha256": sha(python_runner_path)},
        "authorization_sha256_start": auth_sha,
        "canonical_coordinates_absent_after_move": all(not lexists(path) for _, path in coordinates)}
    write_exclusive(qstage / "failure_receipt.json",
                    json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    seal_tree(qstage); verify_seal(qstage)
    rename_noreplace(qstage, qfinal); verify_seal(qfinal)
    require(all(not lexists(path) for _, path in coordinates),
            "canonical coordinate survived failure")


def execute(shell_path, python_runner_path, staging):
    stage = "pre_attempt"
    started = False
    auth_sha = None
    def caught(signum, frame):
        raise Failure("signal " + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, caught)
    try:
        verify_static(shell_path, python_runner_path)
        verify_coordinates(staging)
        auth_sha = verify_authorization(shell_path, python_runner_path)
        verify_static(shell_path, python_runner_path)
        verify_coordinates(staging)
        stage = "attempt_mkdir"
        os.mkdir(str(ATTEMPT), 0o700); started = True
        write_exclusive(ATTEMPT / "ATTEMPT_CONSUMED.json", json.dumps({
            "schema": "m606_m593_energy_attempt_v1", "status": "ATTEMPT_CONSUMED",
            "runner_shell_sha256": sha(shell_path), "runner_python_sha256": sha(python_runner_path),
            "authorization_sha256_start": auth_sha}, sort_keys=True, indent=2) + "\n")
        stage = "formal_analyzer"
        with (ATTEMPT / "production_stdout.log").open("x") as out, \
             (ATTEMPT / "production_stderr.log").open("x") as err:
            cp = subprocess.run([str(PYTHON), str(REPO / ADAPTER_REL), "--source-contract",
                str(REPO / CONTRACT_REL), "--output-dir", str(staging)], stdout=out, stderr=err)
        require(cp.returncode == 0, "formal adapter failed")
        stage = "terminal_verify"
        verify_result(staging)
        for name in ("production_stdout.log", "production_stderr.log"):
            source = ATTEMPT / name
            with source.open("rb") as handle, (staging / name).open("xb") as target:
                target.write(handle.read()); target.flush(); os.fsync(target.fileno())
        remove_top_seal(staging)
        receipt = {"schema": "m606_m593_energy_terminal_rehash_receipt_v1",
            "status": "PASS_M606_TERMINAL_IDENTITY_AND_OUTPUT_REHASH",
            "runner": {"shell_path": str(Path(shell_path).relative_to(REPO)), "shell_sha256": sha(shell_path),
                       "python_path": str(Path(python_runner_path).relative_to(REPO)), "python_sha256": sha(python_runner_path)},
            "adapter": {"path": ADAPTER_REL, "sha256": ADAPTER_SHA},
            "upstream_analyzer": {"path": UPSTREAM_REL, "sha256": UPSTREAM_SHA},
            "source_contract": {"path": CONTRACT_REL, "sha256": CONTRACT_SHA},
            "authorization": {"path": str(AUTH.relative_to(REPO)), "sha256": auth_sha},
            "output_schema": "m597_m593_m528_parent_scratch_generated_macro_energy_result_v2",
            "output_status": "PASS_BOUNDED_GENERATED_MACRO_COMPONENT_MODEL__PENDING_FRESH_INDEPENDENT_RESULT_HAMMER",
            "output_members_preseal": {name: sha(staging / name) for name in
                (RESULT_JSON, CSV_NAME, COMPLETE, "production_stdout.log", "production_stderr.log")},
            "claim": "component-only per-frozen-sampled-inference; pending independent result hammer; not paper data"}
        write_exclusive(staging / "m606_terminal_rehash_receipt.json",
                        json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        seal_tree(staging)
        verify_result(staging, True, shell_path, python_runner_path, auth_sha)
        stage = "pre_publish_rehash"
        require(verify_authorization(shell_path, python_runner_path) == auth_sha,
                "authorization changed prepublish")
        verify_static(shell_path, python_runner_path)
        verify_result(staging, True, shell_path, python_runner_path, auth_sha)
        stage = "publish_result_noreplace"
        rename_noreplace(staging, RESULT)
        stage = "post_publish_rehash"
        require(verify_authorization(shell_path, python_runner_path) == auth_sha,
                "authorization changed postpublish")
        verify_static(shell_path, python_runner_path)
        verify_result(RESULT, True, shell_path, python_runner_path, auth_sha)
        stage = "seal_attempt"
        write_exclusive(ATTEMPT / "ATTEMPT_COMPLETION.json", json.dumps({
            "schema": "m606_m593_energy_attempt_completion_v1",
            "status": "RESULT_PUBLISHED_AND_ALL_REHASH_PASS",
            "authorization_sha256_start": auth_sha}, sort_keys=True, indent=2) + "\n")
        seal_tree(ATTEMPT); verify_seal(ATTEMPT)
        stage = "consume_attempt_noreplace"
        rename_noreplace(ATTEMPT, CONSUMED)
        stage = "post_consume_terminal_rehash"
        verify_seal(CONSUMED, {"ATTEMPT_CONSUMED.json", "ATTEMPT_COMPLETION.json",
                               "production_stdout.log", "production_stderr.log"})
        require(verify_authorization(shell_path, python_runner_path) == auth_sha,
                "authorization changed postconsume")
        verify_static(shell_path, python_runner_path)
        verify_result(RESULT, True, shell_path, python_runner_path, auth_sha)
        return
    except BaseException as error:
        if started:
            quarantine_failure(staging, shell_path, python_runner_path,
                               error, stage, auth_sha)
        raise


def main(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization")
    parser.add_argument("--shell-path", required=True, type=Path)
    args = parser.parse_args(argv)
    shell_path = plain_chain(args.shell_path)
    python_runner = Path(__file__).resolve()
    staging = RESULT.parent / (RESULT.name + ".staging." + str(os.getpid()))
    verify_static(shell_path, python_runner)
    verify_coordinates(staging)
    if args.preflight_only:
        require(args.authorization is None, "preflight received authorization")
        cp = subprocess.run([str(PYTHON), str(REPO / ADAPTER_REL), "--source-contract",
            str(REPO / CONTRACT_REL), "--self-test"], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True)
        require(cp.returncode == 0 and cp.stdout.splitlines()[-1] ==
                "PASS_M606_NOREPLACE_ADAPTER_STATIC_SELF_TEST", "adapter self-test failed")
        verify_static(shell_path, python_runner); verify_coordinates(staging)
        print("PASS_M606_M593_SOURCE_PREFLIGHT_ONLY__NO_RESULT_ATTEMPT_OR_LAUNCH")
        return 0
    require(args.authorization is not None, "--execute requires future authorization")
    require(Path(args.authorization).resolve() == AUTH.resolve(), "authorization path drift")
    execute(shell_path, python_runner, staging)
    print("PASS_M606_M593_ATOMIC_COMPONENT_RESULT_PENDING_INDEPENDENT_RESULT_HAMMER " + str(RESULT))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as error:
        print("M606_FAIL_CLOSED: " + str(error), file=sys.stderr)
        sys.exit(70)
