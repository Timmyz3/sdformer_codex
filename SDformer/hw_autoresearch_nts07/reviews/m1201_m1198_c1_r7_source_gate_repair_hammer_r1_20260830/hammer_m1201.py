#!/usr/bin/env python3
"""Fresh independent source-only hammer for M1198/R7. Never invokes EDA."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
TB = HW / "verif_m1193r6_c1_common_charge_protocol/tb_m1193r6_m1162_common_charge_protocol_unit_delay_r6.sv"
CHECKER = HW / "verif_m1198r7_c1_common_charge_protocol/static_check_m1198r7_m1162_vcs_source.py"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1198r7_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
CONTRACT = HW / "contracts/m1198_m1194_m1193_m1162_c1_r7_source_gate_repair_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1198_m1194_c1_r7_source_gate_repair_author_receipt_r1_20260830"
M1194 = HW / "reviews/m1194_m1193_c1_r6_service_call_closure_source_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    CHECKER: "b1cfb957d5c4fc518d46980040afa61288eb7dcaa79fa5e6c45e25b097094795",
    CONTRACT: "44c5a3add48ef74ef0698f81f20fef417989c17b74df3e1d366cf404b7ce5488",
    AUTHOR / "SHA256SUMS.seal.sha256": "7286441a67b9cb1196dec9356e5bf1b33ca5a6e90522ff4b404137c6fc76768b",
    TB: "0fcc2138ef5d716735eea01dee25a148a5223b1d6adf1e3b2fa464341fbf1345",
    FILELIST: "444ff65d575c6e897f9d459689f323290f16eb89c962c91b395964c7850fcbfa",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    M1194 / "hammer_m1194.py": "d9456931be035ff020146750ca678b626d29b424c011dabbee4c94cf4b93190c",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

ALLOW = {
    "dut.issue_request_valid", "dut.issue_request_epoch", "dut.issue_request_row_id",
    "dut.issue_request_first", "dut.issue_request_last",
    "dut.issue_request_source_valid", "dut.issue_request_source_index",
    "dut.issue_request_parent_valid", "dut.issue_request_parent_id",
}
checks = 0


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_sealed_dir(directory: Path) -> None:
    require(directory.is_dir() and not directory.is_symlink(), f"sealed dir {directory}")
    sums = directory / "SHA256SUMS"
    seal = directory / "SHA256SUMS.seal.sha256"
    require(sums.is_file() and not sums.is_symlink(), f"manifest {directory}")
    require(seal.is_file() and not seal.is_symlink(), f"outer seal {directory}")
    require(seal.read_text().split() == [sha(sums), "SHA256SUMS"], f"outer seal value {directory}")
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute()
                and ".." not in Path(name).parts, f"unsafe manifest member {name}")
        listed[name] = digest
    actual: set[str] = set()
    for root, dirs, files in os.walk(directory, followlinks=False):
        base = Path(root)
        dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            member = base / name
            rel = member.relative_to(directory).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or member.is_symlink():
                continue
            if stat.S_ISREG(member.lstat().st_mode):
                actual.add(rel)
    require(actual == set(listed), f"manifest membership {directory}")
    for name, digest in listed.items():
        require(sha(directory / name) == digest, f"member drift {directory}/{name}")


def strip(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text)


def tasks(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for match in re.finditer(r"\btask\s+automatic\s+([A-Za-z_]\w*)\b(.*?)\bendtask\b", text, re.S):
        require(match.group(1) not in found, "duplicate task")
        found[match.group(1)] = match.group(2)
    return found


def reachable(text: str) -> set[str]:
    defs = tasks(text)
    require("service_assumption_attacks" in defs, "service root")
    seen: set[str] = set()
    pending = ["service_assumption_attacks"]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        body = strip(defs[name])
        for candidate in defs:
            if candidate == name:
                continue
            # Both legal task-call forms at every statement position.
            if re.search(r"(?<![A-Za-z0-9_.$])" + re.escape(candidate)
                         + r"\s*(?:\(|;)", body):
                pending.append(candidate)
    return seen


def independent_validate(tb: str, sva: str) -> None:
    defs = tasks(tb)
    closure = reachable(tb)
    expected_closure = {"service_assumption_attacks", "force_request_no_core_ready",
                        "reset_dut", "release_request", "clear_public_drivers"}
    require(closure == expected_closure, f"closure {sorted(closure)}")
    require("force_request" not in closure, "generic force helper reachable")
    aggregate: list[str] = []
    for name in closure:
        body = strip(defs[name])
        require(not re.search(r"\balias\b", body), f"reachable alias {name}")
        targets = re.findall(r"\bforce\s+([A-Za-z_][A-Za-z0-9_.$\[\]]*)\s*=", body)
        require(set(targets) <= ALLOW, f"unexpected reachable force {name}:{targets}")
        aggregate.extend(targets)
    helper = strip(defs["force_request_no_core_ready"])
    helper_targets = re.findall(r"\bforce\s+([A-Za-z_][A-Za-z0-9_.$\[\]]*)\s*=", helper)
    require(len(helper_targets) == 9 and set(helper_targets) == ALLOW,
            "helper exact nine-force multiset")
    require(len(aggregate) == 9 and set(aggregate) == ALLOW,
            "closure exact nine-force multiset")
    service = strip(defs["service_assumption_attacks"])
    require(service.count("force_request_no_core_ready(") == 2, "two service helper calls")
    require("weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b0;" in service,
            "complete weight-only skew")
    require("weight_rsp_valid = 1'b0; psum_rsp_valid = 1'b1;" in service,
            "complete psum-only skew")
    require(service.count("weight_service_attack_mode = 1'b1;") == 1,
            "one weight service-attack window")
    require(service.count("psum_service_attack_mode = 1'b1;") == 1,
            "one psum service-attack window")
    weight = re.compile(
        r"if\s*\(\s*!weight_service_fault\s*\|\|\s*psum_service_fault\s*\|\|\s*protocol_error"
        r"\s*\|\|\s*dut\.boundary_fault_q\s*\|\|\s*dut\.core_protocol_error\s*\)", re.S)
    psum = re.compile(
        r"if\s*\(\s*!psum_service_fault\s*\|\|\s*weight_service_fault\s*\|\|\s*protocol_error"
        r"\s*\|\|\s*dut\.boundary_fault_q\s*\|\|\s*dut\.core_protocol_error\s*\)", re.S)
    require(len(weight.findall(service)) == 1, "exact weight oracle own1 peer0 composed0")
    require(len(psum.findall(service)) == 1, "exact psum oracle own1 peer0 composed0")
    require(sva.count("assert property") == 16, "16 assertions")
    require(sva.count("cover property") == 6, "6 covers")
    for token in (
        "directed_random=24", "protocol_attacks=7", "service_assumption_attacks=2",
        "request_attack_windows=2", "legal_masks_clear=29", "reset_states=3", "ii=2",
        "normal_m935_rows=1", "normal_m935_tasks=1", "service_skew_isolated=1",
        "reachable_core_ready_force=0", "boundary_fault=0", "core_fault=0",
        "functional_vcs_only=true", "timing_verified=false", "cycles_measured=false",
        "speedup=false", "ppa=false", "energy=false", "system_speedup=false",
        "headline=false"):
        require(tb.count(token) >= 1, "regression/claim token " + token)
    for gate in (
        "cov_random_transactions != 24", "cov_legal_masks_clear != 29",
        "cov_request_attack_windows != 2", "cov_weight_service_attack_windows != 1",
        "cov_psum_service_attack_windows != 1", "cov_normal_issue != 2",
        "cov_normal_row != 1", "cov_normal_task != 1"):
        require(gate in tb, "executable coverage gate " + gate)


def load_checker():
    spec = importlib.util.spec_from_file_location("m1198_author_checker", CHECKER)
    require(spec is not None and spec.loader is not None, "load author checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def author_rejects(module, tb: str, sva: str) -> bool:
    try:
        module.validate(tb, sva)
    except AssertionError:
        return True
    return False


def main() -> None:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift " + str(path))
    verify_sealed_dir(AUTHOR)
    verify_sealed_dir(M1194)
    contract = json.loads(CONTRACT.read_text())
    receipt = json.loads((AUTHOR / "review.json").read_text())
    require(contract["source_identity"]["r7_static_checker"]["sha256"] == sha(CHECKER),
            "contract checker pin")
    require(contract["source_identity"]["clean_r6_tb"]["sha256"] == sha(TB),
            "contract TB pin")
    require(contract["source_identity"]["r7_filelist"]["sha256"] == sha(FILELIST),
            "contract filelist pin")
    require(receipt["contract_sha256"] == sha(CONTRACT), "author receipt contract pin")
    require(receipt["r7_repair"]["new_static_checker_sha256"] == sha(CHECKER),
            "author receipt checker pin")
    require(receipt["status"] ==
            "PASS_R7_SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_VCS_NO_EDA",
            "author source-only boundary")
    lines = [line.strip() for line in FILELIST.read_text().splitlines() if line.strip()]
    require(len(lines) == 6 and len(set(lines)) == 6, "filelist cardinality")
    require(lines[-1] == str(TB), "filelist exact R6 TB tail")
    for member in lines:
        require(Path(member).is_file() and not Path(member).is_symlink(), "filelist member " + member)
    tb, sva = TB.read_text(), SVA.read_text()
    independent_validate(tb, sva)
    module = load_checker()
    require(not author_rejects(module, tb, sva), "author checker rejects pristine source")

    marker = "// R6_SERVICE_NO_CORE_READY_FORCE_BOUNDARY"
    service_anchor = "    task automatic service_assumption_attacks;"
    require(marker in tb and service_anchor in tb, "mutation anchors")
    prefix, suffix = tb.split(marker, 1)
    force_removed = prefix + marker + suffix.replace(
        "            force dut.issue_request_parent_id = 6'b0;\n", "", 1)
    helper_task = (
        "    task automatic m1201_hidden_core_ready_force;\n"
        "        begin force dut.core_issue_data_ready = 1'b1; end\n"
        "    endtask\n\n" + service_anchor)
    bare_task = (
        "    task automatic m1201_bare_core_ready_force;\n"
        "        begin force dut.core_issue_data_ready = 1'b1; end\n"
        "    endtask\n\n" + service_anchor)
    same_task = (
        "    task automatic m1201_same_line_core_ready_force;\n"
        "        begin force dut.core_issue_data_ready = 1'b1; end\n"
        "    endtask\n\n" + service_anchor)
    mutations: dict[str, tuple[str, str]] = {
        # All six bypasses that caused M1194 P1.
        "m1194_bare_helper_call": (tb.replace(service_anchor, bare_task, 1).replace(
            "force_request_no_core_ready(1'b1, 1'b0, 16'h7301",
            "m1201_bare_core_ready_force;\n            force_request_no_core_ready(1'b1, 1'b0, 16'h7301", 1), sva),
        "m1194_same_line_helper_call": (tb.replace(service_anchor, same_task, 1).replace(
            "reset_dut();\n            @(negedge clk_core);\n            force_request_no_core_ready(1'b1, 1'b0, 16'h7301",
            "reset_dut(); m1201_same_line_core_ready_force();\n            @(negedge clk_core);\n"
            "            force_request_no_core_ready(1'b1, 1'b0, 16'h7301", 1), sva),
        "m1194_weight_peer_oracle_relaxed": (tb.replace(
            "!weight_service_fault || psum_service_fault || protocol_error",
            "!weight_service_fault || (1'b0 && psum_service_fault) || protocol_error", 1), sva),
        "m1194_psum_peer_oracle_relaxed": (tb.replace(
            "!psum_service_fault || weight_service_fault || protocol_error",
            "!psum_service_fault || (1'b0 && weight_service_fault) || protocol_error", 1), sva),
        "m1194_protocol_oracle_relaxed": (tb.replace(
            "!weight_service_fault || psum_service_fault || protocol_error",
            "!weight_service_fault || psum_service_fault || (1'b0 && protocol_error)", 1), sva),
        "m1194_one_force_removed": (force_removed, sva),
        # Prior R5/R4 attack families and preserved-regression attacks.
        "r5_nested_generic_helper": (tb.replace(
            "force_request_no_core_ready(1'b1, 1'b0, 16'h7301",
            "force_request(1'b1, 1'b0, 16'h7301", 1), sva),
        "r5_indirect_core_ready_helper": (tb.replace(service_anchor, helper_task, 1).replace(
            "reset_dut();\n            @(negedge clk_core);\n            force_request_no_core_ready(1'b1, 1'b0, 16'h7301",
            "reset_dut();\n            m1201_hidden_core_ready_force();\n            @(negedge clk_core);\n"
            "            force_request_no_core_ready(1'b1, 1'b0, 16'h7301", 1), sva),
        "r5_alias_force": (tb.replace(marker,
            "alias m1201_ready_alias = dut.core_issue_data_ready;\n"
            "            force m1201_ready_alias = 1'b1;", 1), sva),
        "r4_weight_peer_present": (tb.replace(
            "weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b0;",
            "weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b1;", 1), sva),
        "r4_psum_peer_present": (tb.replace(
            "weight_rsp_valid = 1'b0; psum_rsp_valid = 1'b1;",
            "weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b1;", 1), sva),
        "weight_attack_mask_removed": (tb.replace(
            "weight_service_attack_mode = 1'b1;", "weight_service_attack_mode = 1'b0;", 1), sva),
        "psum_attack_mask_removed": (tb.replace(
            "psum_service_attack_mode = 1'b1;", "psum_service_attack_mode = 1'b0;", 1), sva),
        "normal_m935_removed": (tb.replace("normal_m935_tasks=1", "normal_m935_tasks=0", 1), sva),
        "assertion_removed": (tb, sva.replace("ap_psum_request_hold: assert property",
                                              "ap_psum_request_hold: assume property", 1)),
        "cover_removed": (tb, sva.replace("cp_ii2: cover property",
                                          "cp_ii2: assert property", 1)),
    }
    results: dict[str, dict[str, bool]] = {}
    for label, (mut_tb, mut_sva) in mutations.items():
        independent_rejected = False
        try:
            independent_validate(mut_tb, mut_sva)
        except AssertionError:
            independent_rejected = True
        checker_rejected = author_rejects(module, mut_tb, mut_sva)
        require(independent_rejected, "independent validator accepted " + label)
        require(checker_rejected, "M1198 checker accepted " + label)
        results[label] = {
            "independent_rejected": independent_rejected,
            "m1198_checker_rejected": checker_rejected,
        }
    require(len(results) == 16, "all sixteen independent mutations")
    print(json.dumps({
        "schema": "m1201_m1198_c1_r7_source_gate_repair_hammer_v1",
        "status": "PASS_SOURCE_HAMMER__SUCCESSOR_RELEASE_AUTHORING_ONLY__NO_VCS_NO_EDA",
        "checks_passed": checks,
        "reviewer_independent_of_source_author": True,
        "m1194_p1_closed": True,
        "pristine_r6_source_clean": True,
        "task_call_forms": ["helper(...) every statement position", "helper; every statement position"],
        "service_closure": sorted(reachable(tb)),
        "generic_helper_reachable": False,
        "service_force_multiset_exact_nine": True,
        "reachable_alias_or_core_ready_force": False,
        "weight_oracle": "own1_peer0_protocol0_boundary0_core0",
        "psum_oracle": "own1_peer0_protocol0_boundary0_core0",
        "mutations_rejected": results,
        "mutation_count": len(results),
        "assertions": 16,
        "covers": 6,
        "protocol_attacks": 7,
        "service_assumption_attacks": 2,
        "deterministic_legal_transactions": 24,
        "legal_masks_clear": 29,
        "reset_states": 3,
        "minimum_completed_issue_ii": 2,
        "normal_m935_rows": 1,
        "normal_m935_tasks": 1,
        "docs359_sha256": sha(DOCS359),
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "gpu_runs": 0,
        "network_runs": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
