#!/opt/anaconda3/bin/python3.12
"""Read-only M2089 checker for a successful M2088 matched two-axis DC run.

The checker never invokes EDA, a license utility, or a GPU program.  Static
mode validates only frozen source/review identities and intentionally does not
inspect any R9 runtime namespace.  Production mode requires explicit result,
failure, attempt, and output paths; it writes one unsealed mechanical-audit
JSON and does not create or authorize a final review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
CHECKER = Path(__file__).resolve()
RUNNER = HW / (
    "dc_handoff/scripts/run_m2088_m2087_m2086_m2067_fc2_continuation_"
    "tsbg_matched_two_axis_logic_only_dc_one_shot.py")
CONTRACT = HW / (
    "contracts/m2086_m2067_fc2_continuation_tsbg_matched_dc_source_"
    "contract_r1_20260904.json")
FILELIST = HW / (
    "dc_handoff/filelists/iscas_m2086_m2067_fc2_continuation_tsbg_"
    "matched_two_axis_logic_only_dc.f")
TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
RTL_M803 = HW / (
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv")
RTL_M2018 = HW / (
    "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv")
RTL_M2067 = HW / "rtl_m2067/m2067_fc2_exact_continuation_wrapper.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2085 = HW / (
    "reviews/m2085_m2067_ep34_fc2_exact_continuation_vcs_r9_result_"
    "hammer_r1_20260904")
M2087 = HW / (
    "reviews/m2087_m2086_m2067_fc2_continuation_tsbg_matched_dc_source_"
    "hammer_r1_20260904")
R9_RESULT = HW / (
    "results/m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_"
    "20260904")

RUNNER_SHA256 = "0933de895ad10972c8b8e4556a8c26d0a7b9dec4bd26dc1fcf60f89833e3db34"
CONTRACT_SHA256 = "6302946283b4f3e1dc59e7a8eff92741d8de05ee77c6dc81f1abc7a8d44bae88"
FILELIST_SHA256 = "f5f661eb98e011c9e5f9922bf298eb91083e014869e714fdf1c1d8971d1b490d"
TCL_SHA256 = "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe"
SDC_SHA256 = "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
RTL_M803_SHA256 = "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156"
RTL_M2018_SHA256 = "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21"
RTL_M2067_SHA256 = "755027453b9fc91264f44918cc16e31b278cf70e1b13821666ca2be602022c92"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M2087_REVIEW_SHA256 = "7a744894bfb01d2bcac67f769c6f4f76fc21d00bbeda3331361f24c2d246c47d"
M2087_MANIFEST_SHA256 = "99efda16f51611d04e512ce08a4035fcb5bbefef1554d64fb53372efca0004bd"
M2087_OUTER_SHA256 = "cda3c279114a67c7f9fc7147968153a8474d7c9175b6a841768b25a5c7c7d424"

DESIGN = "m2067_fc2_exact_continuation_wrapper"
RESULT_NAME = (
    "m2088_m2067_fc2_continuation_tsbg_matched_two_axis_logic_only_dc_"
    "r1_20260904")
FAILURE_NAME = RESULT_NAME + ".failed_or_incomplete.quarantine"
ATTEMPT_NAME = ".m2088_m2067_fc2_continuation_tsbg_dc_attempt_consumed"
BOOTSTRAP_SHA256 = "3f0791c8c38447275968806360703faa95ef6a45ae53bd3502d09a6c535049e1"
SLOW_DB_BASENAME = "tcbn28hpcplusbwp35p140ssg0p9v125c.db"
FAST_DB_BASENAME = "tcbn28hpcplusbwp35p140ffg1p05vm40c.db"


class Failure(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path: Path, digest: str) -> None:
    need(path.is_file() and not path.is_symlink(), "missing/symlink " + str(path))
    need(sha256(path) == digest, "identity drift " + str(path))


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key " + key)
            value[key] = item
        return value
    value = json.loads(
        path.read_text(errors="strict"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            Failure("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root " + str(path))
    return value


def sealed_directory(root: Path) -> dict[str, str]:
    need(root.is_dir() and not root.is_symlink(), "sealed root missing/symlink")
    files: set[str] = set()
    for current, dirs, names in os.walk(root, followlinks=False):
        base = Path(current)
        for name in dirs:
            need(not (base / name).is_symlink(), "directory symlink in seal")
        for name in names:
            path = base / name
            need(path.is_file() and not path.is_symlink(), "non-regular seal member")
            files.add(path.relative_to(root).as_posix())
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need({"SHA256SUMS", "SHA256SUMS.seal.sha256"} <= files,
         "double seal absent")
    need(outer.read_text(errors="strict").split() ==
         [sha256(manifest), "SHA256SUMS"], "outer seal")
    mapping: dict[str, str] = {}
    for line in manifest.read_text(errors="strict").splitlines():
        fields = line.split(maxsplit=1)
        need(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
             "manifest syntax")
        relative = Path(fields[1].lstrip("*"))
        name = relative.as_posix()
        need(not relative.is_absolute() and ".." not in relative.parts,
             "unsafe manifest member")
        need(name not in mapping and name not in {
            "SHA256SUMS", "SHA256SUMS.seal.sha256"}, "recursive/duplicate seal")
        exact(root / relative, fields[0])
        mapping[name] = fields[0]
    need(set(mapping) == files - {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
         "non-exhaustive sealed directory")
    return mapping


def validate_static() -> dict:
    for path, digest in (
        (RUNNER, RUNNER_SHA256), (CONTRACT, CONTRACT_SHA256),
        (FILELIST, FILELIST_SHA256), (TCL, TCL_SHA256), (SDC, SDC_SHA256),
        (RTL_M803, RTL_M803_SHA256), (RTL_M2018, RTL_M2018_SHA256),
        (RTL_M2067, RTL_M2067_SHA256), (DOCS359, DOCS359_SHA256),
    ):
        exact(path, digest)
    mapping = sealed_directory(M2087)
    exact(M2087 / "review.json", M2087_REVIEW_SHA256)
    exact(M2087 / "SHA256SUMS", M2087_MANIFEST_SHA256)
    exact(M2087 / "SHA256SUMS.seal.sha256", M2087_OUTER_SHA256)
    need(mapping.get("review.json") == M2087_REVIEW_SHA256,
         "M2087 review not sealed")
    review = strict_json(M2087 / "review.json")
    need(review.get("status", "").startswith("PASS_M2087_"), "M2087 status")
    need(review.get("severity_counts", {}).get("P0") == 0 and
         review.get("severity_counts", {}).get("P1") == 0, "M2087 P0/P1")
    need(review.get("reviewed_source_identity") == {
        "contract_sha256": CONTRACT_SHA256,
        "filelist_sha256": FILELIST_SHA256,
        "runner_sha256": RUNNER_SHA256,
    }, "M2087 source identity")
    need(review.get("authorization", {}).get("execute_once") is True and
         review.get("authorization", {}).get("automatic_retry") is False,
         "M2087 authorization")
    contract = strict_json(CONTRACT)
    need(contract.get("physical_flow", {}).get("clock_period_ns") == 3.0,
         "contract clock")
    need(contract.get("physical_flow", {}).get("axes") == [
        {"name": "ordinary", "elaboration_parameter": "SCHEDULE_MODE=0"},
        {"name": "tsbg_b4", "elaboration_parameter": "SCHEDULE_MODE=1"},
    ], "contract axes")
    return {
        "status": "PASS_M2089_CHECKER_STATIC_ONLY__NO_R9_RUNTIME_NAMESPACE_READ",
        "checker_sha256": sha256(CHECKER),
        "runner_sha256": RUNNER_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "m2087_review_sha256": M2087_REVIEW_SHA256,
        "production_result_read": False,
        "r9_runtime_namespace_read": False,
        "eda_or_license_or_gpu_executed": False,
    }


def validate_attempt(attempt_dir: Path) -> dict:
    mapping = sealed_directory(attempt_dir)
    need(set(mapping) == {"attempt.json", "owner.json"}, "attempt inventory")
    attempt = strict_json(attempt_dir / "attempt.json")
    owner = strict_json(attempt_dir / "owner.json")
    need(attempt.get("schema") ==
         "m2088_m2067_fc2_continuation_tsbg_dc_attempt_v1", "attempt schema")
    need(attempt.get("runner_sha256") == RUNNER_SHA256 and
         attempt.get("contract_sha256") == CONTRACT_SHA256, "attempt identity")
    need(attempt.get("dc_shell_runs_budget") == 2 and
         attempt.get("automatic_retry") is False, "attempt budget")
    need(owner.get("schema") == "m2088_attempt_owner_v1", "owner schema")
    need(type(owner.get("pid")) is int and owner["pid"] > 1, "owner pid")
    need(re.fullmatch(r"[0-9a-f]{32}", str(owner.get("nonce", ""))) is not None,
         "owner nonce")
    need(owner.get("runner_sha256") == RUNNER_SHA256, "owner runner")
    return {
        "owner_pid": owner["pid"], "owner_nonce": owner["nonce"],
        "attempt_json_sha256": sha256(attempt_dir / "attempt.json"),
        "owner_json_sha256": sha256(attempt_dir / "owner.json"),
        "manifest_sha256": sha256(attempt_dir / "SHA256SUMS"),
        "outer_file_sha256": sha256(attempt_dir / "SHA256SUMS.seal.sha256"),
        "dc_shell_runs_budget": 2, "automatic_retry": False,
    }


def minimum_slack(path: Path) -> float:
    values = re.findall(
        r"slack \((?:MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)",
        path.read_text(errors="replace"))
    need(values, "missing timing slack " + str(path))
    return min(float(value) for value in values)


def normalized_ports(path: Path) -> str:
    lines = path.read_text(errors="strict").splitlines()
    return "\n".join(line for line in lines
                     if not line.startswith("Design :") and
                     not line.startswith("Date   :"))


def validate_dc_log(path: Path, mode: int) -> dict:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    bootstrap = (
        "Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/"
        "auxx/gui/dv/.synopsys_dv.tcl")
    hits = [index for index, line in enumerate(lines) if line == bootstrap]
    need(len(hits) == 1, "bootstrap cardinality " + str(path))
    start, end = hits[0], hits[0] + 15
    need(start >= 1 and end + 1 < len(lines) and
         lines[start - 1] == "Initializing..." and
         lines[end + 1].startswith("Current time:"), "bootstrap placement")
    block = "\n".join(lines[start:end + 1]) + "\n"
    need(hashlib.sha256(block.encode()).hexdigest() == BOOTSTRAP_SHA256,
         "bootstrap identity")
    filtered = lines[:start] + lines[end + 1:]
    need(not any(re.search(r"(?i)(error:|fatal:)", line) for line in filtered),
         "non-whitelisted DC error/fatal")
    need(not any(re.search(r"(?:Warning|Information|Error):.*"
                           r"\((?:TIM-209|OPT-150)\)", line)
                 for line in filtered), "TIM-209/OPT-150 diagnostic")
    elaborated = DESIGN + f"_SCHEDULE_MODE{mode}"
    need(f"Presto compilation completed successfully. ({elaborated})" in text and
         f"Current design is now '{elaborated}'." in text,
         "schedule-mode elaboration evidence")
    need(text.count("\ncompile_ultra\n") == 1, "compile_ultra log cardinality")
    for token in (SLOW_DB_BASENAME, FAST_DB_BASENAME,
                  SDC.as_posix(), "Using operating conditions 'ssg0p9v125c'",
                  "set_wire_load_model -name ZeroWireload"):
        need(token in text, "DC flow token " + token)
    return {"dc_log_sha256": sha256(path), "elaborated_design": elaborated}


def validate_axis(root: Path, name: str, mode: int, producer: dict) -> dict:
    reports = root / "reports"
    netlist = root / "netlist"
    required = [
        root / "dc.log", root / "TCL_PASS_TERMINAL.txt",
        reports / "area.rpt", reports / "qor.rpt",
        reports / "timing_setup.rpt", reports / "timing_hold_diagnostic.rpt",
        reports / "precompile_loop_gate.rpt", reports / "flow_contract.rpt",
        reports / "compile_receipt.rpt", reports / "ports.rpt",
        reports / "port_count.txt", reports / "clocks.rpt",
        reports / "constraint_max_capacitance.rpt",
        reports / "constraint_max_transition.rpt",
        reports / "constraint_max_fanout.rpt",
        netlist / f"{DESIGN}_mapped.v", netlist / f"{DESIGN}_mapped.sdc",
        netlist / f"{DESIGN}.ddc", netlist / f"{DESIGN}.svf",
    ]
    need(all(path.is_file() and not path.is_symlink() and path.stat().st_size > 0
             for path in required), "axis artifact missing/empty " + name)
    log = validate_dc_log(root / "dc.log", mode)
    need((reports / "precompile_loop_gate.rpt").read_text().splitlines() == [
        "TIM-209=0", "OPT-150=0", "status=PASS_PRECOMPILE_LOOP_GATE"],
        "precompile loop gate " + name)
    terminal = (root / "TCL_PASS_TERMINAL.txt").read_text(errors="strict")
    for token in (
        "status=PASS_M519_R8_SETUP_AREA_DC_TCL_TERMINAL",
        f"design={DESIGN}", "TIM-209=0", "OPT-150=0",
        "compile_ultra_count=1", "incremental_compile_count=0",
        "hold_optimization_count=0", "hold_not_closed_at_dc=true",
    ):
        need(token in terminal, "terminal token " + token + " " + name)
    flow = (reports / "flow_contract.rpt").read_text(errors="strict")
    receipt = (reports / "compile_receipt.rpt").read_text(errors="strict")
    for token in ("compile_ultra_count=1", "incremental_compile_count=0",
                  "hold_optimization_count=0"):
        need(token in flow or token in receipt, "compile receipt " + token)
    need(receipt.count("compile_ultra_count=1") == 1,
         "compile receipt cardinality")
    for report_name in ("constraint_max_capacitance.rpt",
                        "constraint_max_transition.rpt",
                        "constraint_max_fanout.rpt"):
        report = (reports / report_name).read_text(errors="replace")
        need(report.count("This design has no violated constraints.") == 1,
             "electrical constraint violation " + name + "/" + report_name)
    area_hits = re.findall(r"Total cell area:\s*([0-9.]+)",
                           (reports / "area.rpt").read_text(errors="replace"))
    need(len(area_hits) == 1, "area cardinality " + name)
    area = float(area_hits[0])
    setup = minimum_slack(reports / "timing_setup.rpt")
    hold = minimum_slack(reports / "timing_hold_diagnostic.rpt")
    ports = int((reports / "port_count.txt").read_text(errors="strict").strip())
    mapped = netlist / f"{DESIGN}_mapped.v"
    mapped_text = mapped.read_text(errors="replace")
    need(re.search(r"\bmodule\s+" + re.escape(
        DESIGN + f"_SCHEDULE_MODE{mode}") + r"\s*\(", mapped_text) is not None,
        "mapped mode evidence " + name)
    mapped_sdc = (netlist / f"{DESIGN}_mapped.sdc").read_text(errors="replace")
    need(re.search(r"create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)",
                   mapped_sdc) is not None, "mapped 3 ns clock " + name)
    need("set_operating_conditions ssg0p9v125c" in mapped_sdc and
         "set_wire_load_model -name ZeroWireload" in mapped_sdc,
         "mapped SDC corner/wireload " + name)
    observed = {
        "schedule_mode": mode, "area_um2": area,
        "setup_wns_ns": setup, "hold_diagnostic_wns_ns": hold,
        "public_port_count": ports,
        "mapped_netlist_sha256": sha256(mapped),
        "dc_log_sha256": log["dc_log_sha256"],
    }
    need(set(producer) == set(observed), "producer axis field inventory " + name)
    for key, value in observed.items():
        if isinstance(value, float):
            need(math.isclose(producer.get(key, math.nan), value,
                              rel_tol=0.0, abs_tol=1e-12),
                 "producer axis mismatch " + name + "/" + key)
        else:
            need(producer.get(key) == value,
                 "producer axis mismatch " + name + "/" + key)
    return {**observed, "ports_normalized": normalized_ports(reports / "ports.rpt"),
            "mapped_sdc_sha256": sha256(netlist / f"{DESIGN}_mapped.sdc"),
            "elaborated_design": log["elaborated_design"]}


def validate_m2085_and_r9(result: dict) -> dict:
    m2085_mapping = sealed_directory(M2085)
    review_path = M2085 / "review.json"
    need(m2085_mapping.get("review.json") == sha256(review_path),
         "M2085 review not sealed")
    review = strict_json(review_path)
    need(review.get("status", "").startswith("PASS_M2085_"), "M2085 status")
    need(review.get("severity_counts", {}).get("P0", 0) == 0 and
         review.get("severity_counts", {}).get("P1", 0) == 0, "M2085 P0/P1")
    need(review.get("authorization", {}).get("m2088_two_axis_dc") is True,
         "M2085 does not authorize M2088")
    need(review.get("observed", {}).get("workloads") == 960,
         "M2085 workload identity")
    r9_mapping = sealed_directory(R9_RESULT)
    r9_json = R9_RESULT / "result.json"
    need(r9_mapping.get("result.json") == sha256(r9_json), "R9 result not sealed")
    identity = review.get("reviewed_result_identity", {})
    need(identity.get("result_json_sha256") == sha256(r9_json) and
         identity.get("manifest_sha256") == sha256(R9_RESULT / "SHA256SUMS") and
         identity.get("outer_file_sha256") ==
         sha256(R9_RESULT / "SHA256SUMS.seal.sha256"), "M2085/R9 identity")
    source = result.get("source_and_authority_identity", {})
    need(source.get("m2085_review_sha256") == sha256(review_path) and
         source.get("m2087_review_sha256") == M2087_REVIEW_SHA256 and
         source.get("r9_result_json_sha256") == sha256(r9_json),
         "M2088 authority identity")
    ratio = review.get("observed", {}).get("rtl_cycle_ratio_observed")
    need(type(ratio) in (int, float) and math.isfinite(ratio) and ratio > 0,
         "M2085 cycle ratio")
    return {
        "m2085_review_sha256": sha256(review_path),
        "m2085_manifest_sha256": sha256(M2085 / "SHA256SUMS"),
        "m2085_outer_file_sha256": sha256(M2085 / "SHA256SUMS.seal.sha256"),
        "r9_result_json_sha256": sha256(r9_json),
        "r9_result_manifest_sha256": sha256(R9_RESULT / "SHA256SUMS"),
        "r9_result_outer_file_sha256":
            sha256(R9_RESULT / "SHA256SUMS.seal.sha256"),
        "rtl_cycle_ratio_observed": float(ratio),
    }


def production_check(result_dir: Path, failure_path: Path, attempt_dir: Path,
                     output: Path) -> dict:
    need(result_dir.name == RESULT_NAME and failure_path.name == FAILURE_NAME and
         attempt_dir.name == ATTEMPT_NAME, "M2088 namespace basename identity")
    parent = result_dir.parent.resolve(strict=False)
    need(attempt_dir.parent.resolve(strict=False) == parent and
         failure_path.parent.resolve(strict=False) == parent,
         "M2088 namespaces must share one parent")
    need(result_dir.is_dir() and not result_dir.is_symlink(), "result absent")
    need(not os.path.lexists(failure_path), "failure namespace exists")
    need(not os.path.lexists(output), "output already exists")
    need(output.parent.is_dir() and not output.parent.is_symlink(),
         "output parent must preexist")
    output_abs = output.resolve(strict=False)
    for sealed_input in (result_dir.resolve(), attempt_dir.resolve()):
        need(output_abs != sealed_input and sealed_input not in output_abs.parents,
             "output inside sealed input")
    static = validate_static()
    result_mapping = sealed_directory(result_dir)
    need({"result.json", "source_review_snapshot.json", "lmstat.log"} <=
         set(result_mapping), "result root inventory")
    need(any(name.startswith("ordinary/") for name in result_mapping) and
         any(name.startswith("tsbg_b4/") for name in result_mapping),
         "axis inventory")
    result = strict_json(result_dir / "result.json")
    need(result.get("schema") ==
         "m2088_m2067_fc2_continuation_tsbg_matched_dc_result_v1",
         "result schema")
    need(result.get("status") ==
         "PENDING_INDEPENDENT_RESULT_HAMMER_DO_NOT_CITE", "result status")
    source = result.get("source_and_authority_identity", {})
    need(source.get("runner_sha256") == RUNNER_SHA256 and
         source.get("contract_sha256") == CONTRACT_SHA256 and
         source.get("filelist_sha256") == FILELIST_SHA256,
         "result source identity")
    claim = result.get("claim_boundary", {})
    need(claim == {
        "same_public_ports_library_clock_constraints": True,
        "logic_only_pre_macro": True, "ideal_clock": True,
        "wireload": "ZeroWireload", "macro_count": 0,
        "hold_diagnostic_not_closed": True, "power": False,
        "energy": False, "full_fc_wall_time": False,
        "system_speedup": False, "paper_ppa_ready": False,
        "paper_admitted": False,
    }, "producer claim boundary")
    need(result.get("execution") == {
        "license_queries": 1, "dc_shell_runs": 2, "automatic_retry": False},
         "execution budget")
    attempt = validate_attempt(attempt_dir)
    authority = validate_m2085_and_r9(result)
    axes = result.get("axes")
    need(type(axes) is dict and set(axes) == {"ordinary", "tsbg_b4"},
         "two unique axes")
    ordinary = validate_axis(result_dir / "ordinary", "ordinary", 0,
                             axes["ordinary"])
    tsbg = validate_axis(result_dir / "tsbg_b4", "tsbg_b4", 1,
                         axes["tsbg_b4"])
    need(ordinary["ports_normalized"] == tsbg["ports_normalized"],
         "public port signature mismatch")
    area_ratio = tsbg["area_um2"] / ordinary["area_um2"]
    cycle_ratio = authority["rtl_cycle_ratio_observed"]
    throughput_per_area = cycle_ratio / area_ratio
    recomputed = {
        "ordinary_over_tsbg_rtl_cycle_ratio": cycle_ratio,
        "tsbg_over_ordinary_logic_area_ratio": area_ratio,
        "tsbg_logic_area_overhead_fraction": area_ratio - 1.0,
        "tsbg_over_ordinary_throughput_per_logic_area_ratio":
            throughput_per_area,
        "both_setup_met": ordinary["setup_wns_ns"] >= 0 and
            tsbg["setup_wns_ns"] >= 0,
        "public_ports_equal": True,
    }
    comparison = result.get("comparison", {})
    for key, value in recomputed.items():
        if isinstance(value, float):
            need(math.isclose(comparison.get(key, math.nan), value,
                              rel_tol=0.0, abs_tol=1e-12),
                 "comparison mismatch " + key)
        else:
            need(comparison.get(key) is value, "comparison mismatch " + key)
    candidate_gate = {
        "both_setup_met": recomputed["both_setup_met"],
        "public_ports_equal": True,
        "logic_area_tax_at_most_2pct": area_ratio <= 1.02,
        "throughput_per_logic_area_at_least_1p15x":
            throughput_per_area >= 1.15,
    }
    need(comparison.get("candidate_gate") == candidate_gate,
         "candidate gate mismatch")
    snapshot = strict_json(result_dir / "source_review_snapshot.json")
    need(snapshot == {
        "m2087_status": strict_json(M2087 / "review.json")["status"],
        "m2087_review_sha256": M2087_REVIEW_SHA256,
    }, "M2087 snapshot")
    output_value = {
        "schema": "m2089_m2088_m2067_fc2_continuation_tsbg_matched_dc_"
                  "independent_result_hammer_mechanical_v1",
        "status": "PASS_M2089_M2088_SUCCESS_RESULT_MECHANICALLY_REPARSED__"
                  "FINAL_REVIEW_AND_PAPER_ADMISSION_SEPARATE",
        "decision": "GO_COMPONENT_THROUGHPUT_PER_LOGIC_AREA_CANDIDATE" if
            all(candidate_gate.values()) else "NO_GO_CANDIDATE_GATE",
        "input_identity": {
            "result_json_sha256": sha256(result_dir / "result.json"),
            "result_manifest_sha256": sha256(result_dir / "SHA256SUMS"),
            "result_outer_file_sha256":
                sha256(result_dir / "SHA256SUMS.seal.sha256"),
            "attempt": attempt, "authority": authority,
            "runner_sha256": RUNNER_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "m2087_review_sha256": M2087_REVIEW_SHA256,
            "docs359_sha256": DOCS359_SHA256,
        },
        "axes": {
            "ordinary": {key: value for key, value in ordinary.items()
                         if key != "ports_normalized"},
            "tsbg_b4": {key: value for key, value in tsbg.items()
                        if key != "ports_normalized"},
        },
        "comparison_recomputed": {**recomputed, "candidate_gate": candidate_gate},
        "flow_evidence": {
            "result_double_sealed_exhaustively": True,
            "attempt_double_sealed_exhaustively": True,
            "failure_namespace_absent": True,
            "owner_attempt_verified_against_frozen_runner": True,
            "axes_unique": ["ordinary:SCHEDULE_MODE=0", "tsbg_b4:SCHEDULE_MODE=1"],
            "one_compile_ultra_per_axis": True,
            "same_3ns_sdc_libraries_ports": True,
            "tim209_opt150_diagnostics": 0,
            "electrical_constraint_violations": 0,
        },
        "claim_boundary": {
            "component_throughput_per_logic_area_candidate":
                all(candidate_gate.values()),
            "logic_only_pre_macro": True, "ideal_clock": True,
            "hold_closed": False, "power": False, "energy": False,
            "macro_inclusive": False, "full_fc_wall_time": False,
            "system_speedup": False, "paper_ppa_ready": False,
            "paper_admitted": False, "final_human_review_completed": False,
        },
        "static_identity": static,
        "eda_or_license_or_gpu_executed_by_checker": False,
    }
    with output.open("x") as stream:
        json.dump(output_value, stream, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
    return output_value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--failure-path", type=Path)
    parser.add_argument("--attempt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    supplied = [args.result_dir, args.failure_path, args.attempt_dir, args.output]
    if args.static:
        need(not any(supplied), "--static accepts no production paths")
        value = validate_static()
    else:
        need(all(item is not None for item in supplied),
             "production mode requires result/failure/attempt/output")
        value = production_check(args.result_dir, args.failure_path,
                                 args.attempt_dir, args.output)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(json.dumps({"status": "FAIL_M2089", "error": str(exc)},
                         sort_keys=True, allow_nan=False))
        raise SystemExit(1)
