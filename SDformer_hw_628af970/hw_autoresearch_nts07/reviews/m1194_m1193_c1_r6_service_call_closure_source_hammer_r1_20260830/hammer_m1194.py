#!/usr/bin/env python3
"""Independent source-only hammer for M1193/R6. Never invokes EDA."""
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
CHECKER = HW / "verif_m1193r6_c1_common_charge_protocol/static_check_m1193r6_m1162_vcs_source.py"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1193r6_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
CONTRACT = HW / "contracts/m1193_m1192_m1191_m1187_m1162_c1_service_attack_call_closure_repair_source_contract_r1_20260830.json"
CONTRACT_SUM = Path(str(CONTRACT) + ".sha256")
CONTRACT_SEAL = Path(str(CONTRACT_SUM) + ".seal.sha256")
M1192 = HW / "reviews/m1192_m1191_c1_r5_service_attack_oracle_source_hammer_r1_20260830"
AUTHOR = HW / "reviews/m1193_m1192_c1_r6_service_call_closure_source_author_receipt_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    TB: "0fcc2138ef5d716735eea01dee25a148a5223b1d6adf1e3b2fa464341fbf1345",
    CHECKER: "c7260f648e1240d277e527015d72e89c2cc807b7cab70fa1ba38b43168733fc0",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    FILELIST: "444ff65d575c6e897f9d459689f323290f16eb89c962c91b395964c7850fcbfa",
    CONTRACT: "a4af7ab1c78739a8cf9ae92265737262be17456c2e8f4ada3abde58328270974",
    CONTRACT_SUM: "7a6540ae89eae8b6c24ace3bb233562a7a2df604395aed67c37d9e69eda0e23e",
    CONTRACT_SEAL: "47a17747231a91902e843048bcca2706df3dd649ce6357f30b586544dc6593dd",
    M1192 / "review.md": "d6126d64e3b68c18156127fa72ecd5c3246c39b1514a6f45715f7309557bee33",
    M1192 / "review.json": "09e4bc0d54e573ddd48496e558a70c178dfc411028939156f3d8f56f876b844a",
    M1192 / "SHA256SUMS": "6fa77fc55280bcc8e710bfcb38722cd061eb9dd42a0c85b84333c4946c5e3e4c",
    M1192 / "SHA256SUMS.seal.sha256": "c1f8044aeae4a42205c88d397e6f5956620e7b850c0dce78a9628be48aaf3dea",
    AUTHOR / "review.md": "1c55a72d14741ddfc4fec92fdc392a9d6d3658c832071688e20f34ce6303e41a",
    AUTHOR / "review.json": "7d4584021a72a104ff3daa2326ee6097dd0598c9aa98609a5f862d851095eb65",
    AUTHOR / "mechanical_checks.json": "c043056dbeb8c6599f84cd7459d0b04de0bbb1b849011d5ac2544078bcefb13e",
    AUTHOR / "SHA256SUMS": "89b9fd6b51e8e0546ee10112aa7a041eabb36b8f99b9d78850eb88ce40e70d7f",
    AUTHOR / "SHA256SUMS.seal.sha256": "b5ce26789c817c30a54df8b6aeca91a9fd20053232d9287b85d57167c3c6c562",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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
    require(directory.is_dir() and not directory.is_symlink(), f"missing sealed dir {directory}")
    sums = directory / "SHA256SUMS"
    seal = directory / "SHA256SUMS.seal.sha256"
    require(seal.read_text().split() == [sha(sums), "SHA256SUMS"], f"outer seal {directory}")
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute()
                and ".." not in Path(name).parts, f"unsafe member {name}")
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
    require(actual == set(listed), f"membership drift {directory}")
    for name, digest in listed.items():
        require(sha(directory / name) == digest, f"member drift {directory}/{name}")


def strip_comments_and_strings(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text)


def tasks(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for match in re.finditer(r"\btask\s+automatic\s+([A-Za-z_]\w*)\b(.*?)\bendtask\b", text, re.S):
        require(match.group(1) not in found, "duplicate task")
        found[match.group(1)] = match.group(2)
    return found


def closure(text: str) -> set[str]:
    task_defs = tasks(text)
    require("service_assumption_attacks" in task_defs, "service root")
    seen: set[str] = set()
    pending = ["service_assumption_attacks"]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        body = strip_comments_and_strings(task_defs[name])
        for candidate in task_defs:
            if candidate == name:
                continue
            # Accept both SystemVerilog task call spellings: foo(...) and foo;
            # at any statement position, not only at the beginning of a line.
            called = re.search(r"(?<![A-Za-z0-9_.$])" + re.escape(candidate)
                               + r"\s*(?:\(|;)", body)
            if called and candidate not in seen:
                pending.append(candidate)
    return seen


REQUEST_FORCE_ALLOWLIST = {
    "dut.issue_request_valid", "dut.issue_request_epoch", "dut.issue_request_row_id",
    "dut.issue_request_first", "dut.issue_request_last",
    "dut.issue_request_source_valid", "dut.issue_request_source_index",
    "dut.issue_request_parent_valid", "dut.issue_request_parent_id",
}


def independent_validate(tb: str, sva: str) -> None:
    task_defs = tasks(tb)
    reached = closure(tb)
    require(reached == {"service_assumption_attacks", "force_request_no_core_ready",
                        "reset_dut", "release_request", "clear_public_drivers"},
            f"unexpected closure {sorted(reached)}")
    require("force_request" not in reached, "generic helper reachable")
    aggregate_forces: list[str] = []
    for name in reached:
        body = strip_comments_and_strings(task_defs[name])
        require(not re.search(r"\balias\b", body), f"alias reachable in {name}")
        force_targets = re.findall(r"\bforce\s+([A-Za-z_][A-Za-z0-9_.$\[\]]*)\s*=", body)
        require(set(force_targets) <= REQUEST_FORCE_ALLOWLIST,
                f"non-request force reachable in {name}: {force_targets}")
        aggregate_forces.extend(force_targets)
    helper = strip_comments_and_strings(task_defs["force_request_no_core_ready"])
    helper_forces = re.findall(r"\bforce\s+([A-Za-z_][A-Za-z0-9_.$\[\]]*)\s*=", helper)
    require(len(helper_forces) == 9 and set(helper_forces) == REQUEST_FORCE_ALLOWLIST,
            "service helper does not force exactly nine request fields")
    require(len(aggregate_forces) == 9 and set(aggregate_forces) == REQUEST_FORCE_ALLOWLIST,
            "reachable force aggregate differs from nine request fields")
    service = strip_comments_and_strings(task_defs["service_assumption_attacks"])
    require(service.count("force_request_no_core_ready(") == 2, "two service helper calls")
    require("weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b0;" in service, "weight-only skew")
    require("weight_rsp_valid = 1'b0; psum_rsp_valid = 1'b1;" in service, "psum-only skew")
    weight_oracle = re.compile(
        r"if\s*\(\s*!weight_service_fault\s*\|\|\s*psum_service_fault\s*\|\|\s*protocol_error"
        r"\s*\|\|\s*dut\.boundary_fault_q\s*\|\|\s*dut\.core_protocol_error\s*\)", re.S)
    psum_oracle = re.compile(
        r"if\s*\(\s*!psum_service_fault\s*\|\|\s*weight_service_fault\s*\|\|\s*protocol_error"
        r"\s*\|\|\s*dut\.boundary_fault_q\s*\|\|\s*dut\.core_protocol_error\s*\)", re.S)
    require(len(weight_oracle.findall(service)) == 1, "exact weight oracle")
    require(len(psum_oracle.findall(service)) == 1, "exact psum oracle")
    require(sva.count("assert property") == 16 and sva.count("cover property") == 6,
            "16 assert / 6 cover")
    for token in ("protocol_attacks=7", "service_assumption_attacks=2",
                  "directed_random=24", "legal_masks_clear=29", "reset_states=3",
                  "ii=2", "normal_m935_rows=1", "normal_m935_tasks=1"):
        require(token in tb, f"regression token {token}")


def author_accepts(checker_module, tb: str, sva: str) -> bool:
    try:
        checker_module.validate(tb, sva)
        return True
    except AssertionError:
        return False


def main() -> None:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                f"identity drift {path}")
    verify_sealed_dir(M1192)
    verify_sealed_dir(AUTHOR)
    require(CONTRACT_SUM.read_text().split() == [sha(CONTRACT), CONTRACT.name], "contract checksum")
    require(CONTRACT_SEAL.read_text().split() == [sha(CONTRACT_SUM), CONTRACT_SUM.name], "contract outer seal")
    contract = json.loads(CONTRACT.read_text())
    receipt = json.loads((AUTHOR / "review.json").read_text())
    require(contract["authority"]["m1192_review_sha256"] == sha(M1192 / "review.md"), "M1192 authority")
    require(receipt["contract_sha256"] == sha(CONTRACT), "author contract pin")
    require(contract["source_identity"]["r6_tb"]["sha256"] == sha(TB), "TB pin")
    require(contract["source_identity"]["static_checker"]["sha256"] == sha(CHECKER), "checker pin")
    require(contract["source_identity"]["filelist"]["sha256"] == sha(FILELIST), "filelist pin")
    tb, sva = TB.read_text(), SVA.read_text()
    independent_validate(tb, sva)

    spec = importlib.util.spec_from_file_location("m1193_author_checker", CHECKER)
    require(spec is not None and spec.loader is not None, "load author checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(author_accepts(module, tb, sva), "author checker rejects pristine source")

    marker = "// R6_SERVICE_NO_CORE_READY_FORCE_BOUNDARY"
    require(marker in tb, "mutation marker")
    before_marker, after_marker = tb.split(marker, 1)
    removed_service_force = before_marker + marker + after_marker.replace(
        "            force dut.issue_request_parent_id = 6'b0;\n", "", 1)
    service_task_anchor = "    task automatic service_assumption_attacks;"
    require(service_task_anchor in tb, "service task anchor")
    indirect_task = (
        "    task automatic m1194_hidden_core_force;\n"
        "        begin force dut.core_issue_data_ready = 1'b1; end\n"
        "    endtask\n\n" + service_task_anchor)
    direct_mutations = {
        "helper_alias_to_generic": tb.replace(
            "force_request_no_core_ready(1'b1, 1'b0, 16'h7301",
            "force_request(1'b1, 1'b0, 16'h7301", 1),
        "direct_indirect_helper": tb.replace(service_task_anchor, indirect_task, 1).replace(
            "reset_dut();\n            @(negedge clk_core);\n            force_request_no_core_ready(1'b1, 1'b0, 16'h7301",
            "reset_dut();\n            m1194_hidden_core_force();\n            @(negedge clk_core);\n"
            "            force_request_no_core_ready(1'b1, 1'b0, 16'h7301", 1),
        "aliased_force": tb.replace(marker,
            "alias m1194_ready_alias = dut.core_issue_data_ready;\n"
            "            force m1194_ready_alias = 1'b1;", 1),
        "one_request_force_removed": removed_service_force,
        "weight_peer_oracle_relaxed": tb.replace(
            "!weight_service_fault || psum_service_fault || protocol_error",
            "!weight_service_fault || (1'b0 && psum_service_fault) || protocol_error", 1),
        "psum_peer_oracle_relaxed": tb.replace(
            "!psum_service_fault || weight_service_fault || protocol_error",
            "!psum_service_fault || (1'b0 && weight_service_fault) || protocol_error", 1),
        "composed_protocol_oracle_relaxed": tb.replace(
            "!weight_service_fault || psum_service_fault || protocol_error",
            "!weight_service_fault || psum_service_fault || (1'b0 && protocol_error)", 1),
    }
    # Parser bypass: a legal no-formal task invocation without parentheses.
    bypass_insert = (
        "    task automatic m1194_bare_core_force;\n"
        "        begin force dut.core_issue_data_ready = 1'b1; end\n"
        "    endtask\n\n" + service_task_anchor)
    bypass = tb.replace(service_task_anchor, bypass_insert, 1).replace(
        "force_request_no_core_ready(1'b1, 1'b0, 16'h7301",
        "m1194_bare_core_force;\n            force_request_no_core_ready(1'b1, 1'b0, 16'h7301", 1)
    direct_mutations["bare_task_call_parser_bypass"] = bypass
    # Same-line calls also bypass an author parser anchored at beginning-of-line.
    same_line_insert = (
        "    task automatic m1194_same_line_core_force;\n"
        "        begin force dut.core_issue_data_ready = 1'b1; end\n"
        "    endtask\n\n" + service_task_anchor)
    same_line = tb.replace(service_task_anchor, same_line_insert, 1).replace(
        "reset_dut();\n            @(negedge clk_core);\n            force_request_no_core_ready(1'b1, 1'b0, 16'h7301",
        "reset_dut(); m1194_same_line_core_force();\n            @(negedge clk_core);\n"
        "            force_request_no_core_ready(1'b1, 1'b0, 16'h7301", 1)
    direct_mutations["same_line_call_parser_bypass"] = same_line

    mutation_results: dict[str, dict[str, bool]] = {}
    for label, changed in direct_mutations.items():
        independent_rejected = False
        try:
            independent_validate(changed, sva)
        except AssertionError:
            independent_rejected = True
        mutation_results[label] = {
            "independent_hammer_rejected": independent_rejected,
            "author_checker_rejected": not author_accepts(module, changed, sva),
        }
        require(independent_rejected, f"independent checker accepted {label}")

    # Identity-layer attacks are checked directly rather than by source validate().
    identity_attacks = {
        "reuse_r5_tb": sha(HW / "verif_m1191r5_c1_common_charge_protocol/tb_m1191r5_m1162_common_charge_protocol_unit_delay_r5.sv") != EXPECTED[TB],
        "reuse_r5_filelist": sha(HW / "dc_handoff/filelists/date_m1191r5_m1162_c1_common_charge_protocol_unit_delay_vcs.f") != EXPECTED[FILELIST],
        "reuse_failed_r4_namespace": (
            (HW / "results/.m1187_m1168r3_m1162_c1_common_charge_protocol_vcs_r4_attempt_consumed/identity.txt").is_file()
            and (HW / "results/m1187_m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r4_20260830.failed_or_incomplete.3580131.quarantine/RUN_FAILED_OR_INCOMPLETE.txt").is_file()),
        "docs359_mutation": hashlib.sha256(DOCS359.read_bytes() + b"M1194_MUTATION").hexdigest() != EXPECTED[DOCS359],
    }
    for label, rejected in identity_attacks.items():
        require(rejected, f"identity attack accepted {label}")

    author_blind_spots = sorted(label for label, result in mutation_results.items()
                                if not result["author_checker_rejected"])
    p0 = []
    p1 = []
    if author_blind_spots:
        p1.append({
            "id": "M1194-P1-AUTHOR-CHECKER-BYPASS",
            "accepted_mutations": author_blind_spots,
            "impact": "The sealed R6 source is clean, but its mandatory source checker accepts call-graph/oracle relaxations; source milestone is not release-ready.",
        })
    print(json.dumps({
        "schema": "m1194_m1193_r6_source_hammer_v1",
        "status": ("PASS_SOURCE_HAMMER__RELEASE_SOURCE_MAY_BE_AUTHORED__NO_EDA"
                   if not p0 and not p1 else
                   "FAIL_SOURCE_CONTRACT__DO_NOT_AUTHOR_RELEASE__NO_VCS_NO_EDA"),
        "checks_passed": checks,
        "actual_r6_source_clean": True,
        "m1192_p0_closed_in_actual_source": True,
        "service_closure": sorted(closure(tb)),
        "service_force_targets": sorted(REQUEST_FORCE_ALLOWLIST),
        "mutation_results": mutation_results,
        "identity_attacks_rejected": identity_attacks,
        "author_checker_blind_spots": author_blind_spots,
        "p0": p0,
        "p1": p1,
        "docs359_sha256": sha(DOCS359),
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
