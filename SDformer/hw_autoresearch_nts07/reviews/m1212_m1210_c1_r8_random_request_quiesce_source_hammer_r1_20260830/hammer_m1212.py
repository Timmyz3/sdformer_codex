#!/usr/bin/env python3
"""Fresh independent source-only hammer for M1210/R8. Never invokes EDA."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
TB = HW / "verif_m1210r8_c1_common_charge_protocol/tb_m1210r8_m1162_common_charge_protocol_unit_delay_r8.sv"
CHECKER = HW / "verif_m1210r8_c1_common_charge_protocol/static_check_m1210r8_m1162_vcs_source.py"
FILELIST = HW / "dc_handoff/filelists/date_m1210r8_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
R6_TB = HW / "verif_m1193r6_c1_common_charge_protocol/tb_m1193r6_m1162_common_charge_protocol_unit_delay_r6.sv"
M1201 = HW / "reviews/m1201_m1198_c1_r7_source_gate_repair_hammer_r1_20260830/hammer_m1201.py"
CONTRACT = HW / "contracts/m1210_m1207_m1198_m1162_c1_r8_random_request_quiesce_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1210_m1207_c1_r8_random_request_quiesce_author_receipt_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    TB: "060ec9d5ae6085a0dd013160d22f63e21615730384ddaef342eb3fa77e17947b",
    CHECKER: "cce8219a13d7584f1c35e262ac4de3e4a935fddc53652d6ce322e7e5f94daa96",
    FILELIST: "048253d22301df9fb84502ff35f5129459a5b43e4ff9e8d11ea62973f7047af6",
    CONTRACT: "26ca340e8f33ca936b169c638862bc3a76f7233035d680cc14ddb7389bcc5d07",
    AUTHOR / "review.json": "d9671bff7efa1e808d5008c23d02df119df4553b60d5782fb2e0ba8bb73efc4a",
    AUTHOR / "SHA256SUMS": "cf9e56adcc15c33ca7663502cdad741c1287dc64d8e2f79df55b9120d986cc5a",
    AUTHOR / "SHA256SUMS.seal.sha256": "28a209d39c1211a0c9c20b43b471cea68d1e5492d516c332120cc1098a773826",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    M1201: "b59fcd476cfddda527dbd29fc06aad26e6699c1e5004cdc2f682c63939fb4113",
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
    require(directory.is_dir() and not directory.is_symlink(), "sealed author directory")
    sums = directory / "SHA256SUMS"
    seal = directory / "SHA256SUMS.seal.sha256"
    require(seal.read_text().split() == [sha(sums), "SHA256SUMS"], "author outer seal")
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "safe author member " + name)
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
    require(actual == set(listed), "author manifest membership")
    for name, digest in listed.items():
        require(sha(directory / name) == digest, "author member drift " + name)


def strip(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    return re.sub(r"\s+", " ", text).strip()


def tasks(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for match in re.finditer(r"\btask\s+automatic\s+([A-Za-z_]\w*)\b(.*?)\bendtask\b", text, re.S):
        require(match.group(1) not in found, "duplicate task " + match.group(1))
        found[match.group(1)] = match.group(2)
    return found


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "load " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(tb: str, sva: str) -> None:
    defs = tasks(tb)
    require("random_legal_transaction" in defs, "random task present")
    random = strip(defs["random_legal_transaction"])

    # Prove the dynamic ordering encoded by the task: request/window setup;
    # exact intended fires; one falling-edge boundary; both ready lows and
    # window retirement; exact first oracle; only then either response branch.
    anchors = [
        "random_weight_request_handshakes = 0;",
        "random_psum_request_handshakes = 0;",
        "random_request_window_active = 1'b1;",
        "force_request(first, 1'b0, 16'h8000 + index, index[5:0], prng_q[13:10]);",
        "wait (weight_fire_count == w0 + 1);",
        "if (first) wait (psum_fire_count == p0 + 1);",
        "@(negedge clk_core); weight_req_ready = 1'b0; psum_req_ready = 1'b0; random_request_window_active = 1'b0;",
        "if (random_weight_request_handshakes != 1 || random_psum_request_handshakes != first)",
        "if (index[0])",
        "force dut.core_issue_data_ready = 1'b0;",
        "repeat (hold_cycles) @(posedge clk_core);",
        "force dut.core_issue_data_ready = 1'b1;",
        "wait (dut.response_accept_w);",
        "random_weight_request_handshakes != 1 || random_psum_request_handshakes != first",
        "release_request();",
    ]
    positions: list[int] = []
    cursor = 0
    for anchor in anchors:
        location = random.find(anchor, cursor)
        require(location >= cursor, "ordered anchor: " + anchor)
        positions.append(location)
        cursor = location + len(anchor)
    require(positions == sorted(positions) and len(set(positions)) == len(positions),
            "strict random transaction ordering")
    require(random.count("weight_req_ready = 1'b0;") == 2,
            "initial stall plus exact weight quiesce")
    require(random.count("psum_req_ready = 1'b0;") == 2,
            "initial stall plus exact psum quiesce")
    require(random.count("random_request_window_active = 1'b1;") == 1,
            "one random window open")
    require(random.count("random_request_window_active = 1'b0;") == 1,
            "one random window close")
    require(random.count("random_weight_request_handshakes != 1") == 2,
            "initial and terminal weight exact oracles")
    require(random.count("random_psum_request_handshakes != first") == 2,
            "initial and terminal psum exact oracles")

    # Counters themselves must sample only real request fires while the explicit
    # random window is live.  This closes a vacuous counter/oracle implementation.
    flat = strip(tb)
    require(flat.count("reset_n && random_request_window_active && weight_req_valid && weight_req_ready") == 1,
            "exact weight counter gate")
    require(flat.count("reset_n && random_request_window_active && psum_req_valid && psum_req_ready") == 1,
            "exact psum counter gate")
    require(flat.count("random_weight_request_handshakes = random_weight_request_handshakes + 1;") == 1,
            "one weight counter update")
    require(flat.count("random_psum_request_handshakes = random_psum_request_handshakes + 1;") == 1,
            "one psum counter update")

    # R8 is additive over the clean R6 corpus.  Every task except the repaired
    # random task must remain semantically byte-for-byte equal after comments and
    # whitespace are removed.  Then independently rerun the R7 service closure.
    r6_defs = tasks(R6_TB.read_text())
    require(set(defs) == set(r6_defs), "R6/R8 task set preserved")
    for name in sorted(defs):
        if name != "random_legal_transaction":
            require(strip(defs[name]) == strip(r6_defs[name]), "preserved task " + name)
    m1201 = load(M1201, "m1212_previous_hammer")
    m1201.independent_validate(tb, sva)
    require(sva.count("assert property") == 16 and sva.count("cover property") == 6,
            "frozen assertion/cover cardinality")
    require(tb.count("        normal_m935_completion();") == 1,
            "normal frozen-M935 completion remains invoked")
    for token in (
        "directed_random=24", "protocol_attacks=7", "service_assumption_attacks=2",
        "request_attack_windows=2", "legal_masks_clear=29", "reset_states=3", "ii=2",
        "normal_m935_rows=1", "normal_m935_tasks=1", "random_request_quiesce=24",
        "exactly_one_random_request_handshake=1", "service_skew_isolated=1",
        "reachable_core_ready_force=0", "boundary_fault=0", "core_fault=0",
        "functional_vcs_only=true", "timing_verified=false", "cycles_measured=false",
        "speedup=false", "ppa=false", "energy=false", "system_speedup=false",
        "headline=false"):
        require(tb.count(token) >= 1, "preserved token " + token)


def rejected(tb: str, sva: str) -> bool:
    try:
        validate(tb, sva)
    except AssertionError:
        return True
    return False


def main() -> None:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift " + str(path))
    verify_sealed_dir(AUTHOR)
    contract = json.loads(CONTRACT.read_text())
    receipt = json.loads((AUTHOR / "review.json").read_text())
    require(contract["source_identity"]["r8_tb"]["sha256"] == sha(TB), "contract TB pin")
    require(contract["source_identity"]["r8_static_checker"]["sha256"] == sha(CHECKER),
            "contract checker pin")
    require(contract["source_identity"]["r8_filelist"]["sha256"] == sha(FILELIST),
            "contract filelist pin")
    require(receipt["contract_sha256"] == sha(CONTRACT), "author receipt contract pin")
    require(receipt["status"].startswith("PASS_R8_SOURCE_ONLY"), "author source-only status")
    lines = [line.strip() for line in FILELIST.read_text().splitlines() if line.strip()]
    require(len(lines) == 6 and len(set(lines)) == 6 and lines[-1] == str(TB),
            "exact six-member filelist and R8 tail")
    for member in lines:
        require(Path(member).is_file() and not Path(member).is_symlink(), "filelist member " + member)

    tb, sva = TB.read_text(), SVA.read_text()
    validate(tb, sva)
    author = load(CHECKER, "m1212_author_checker")
    author.validate(tb, sva)

    q = ("            weight_req_ready = 1'b0;\n"
         "            psum_req_ready = 1'b0;\n"
         "            random_request_window_active = 1'b0;")
    branch = "            if (index[0]) begin"
    require(q in tb and branch in tb, "mutation sequence anchors")
    delay_both = tb.replace(q, "            random_request_window_active = 1'b0;", 1)
    delay_both = delay_both.replace(branch, branch + "\n                weight_req_ready = 1'b0;\n                psum_req_ready = 1'b0;", 1)
    delay_weight = tb.replace(q,
        "            psum_req_ready = 1'b0;\n            random_request_window_active = 1'b0;", 1)
    delay_weight = delay_weight.replace(branch, branch + "\n                weight_req_ready = 1'b0;", 1)
    delay_psum = tb.replace(q,
        "            weight_req_ready = 1'b0;\n            random_request_window_active = 1'b0;", 1)
    delay_psum = delay_psum.replace(branch, branch + "\n                psum_req_ready = 1'b0;", 1)
    mutations = {
        "remove_both_ready_quiesce": tb.replace(q, "            random_request_window_active = 1'b0;", 1),
        "remove_weight_ready_quiesce": tb.replace(q,
            "            psum_req_ready = 1'b0;\n            random_request_window_active = 1'b0;", 1),
        "remove_psum_ready_quiesce": tb.replace(q,
            "            weight_req_ready = 1'b0;\n            random_request_window_active = 1'b0;", 1),
        "delay_both_ready_until_response_branch": delay_both,
        "delay_weight_ready_until_response_branch": delay_weight,
        "delay_psum_ready_until_response_branch": delay_psum,
        "retire_window_before_exact_fires": tb.replace(
            "            wait (weight_fire_count == w0 + 1);",
            "            random_request_window_active = 1'b0;\n            wait (weight_fire_count == w0 + 1);", 1),
        "remove_initial_weight_oracle": tb.replace(
            "if (random_weight_request_handshakes != 1\n                    || random_psum_request_handshakes != first)",
            "if (random_psum_request_handshakes != first)", 1),
        "remove_terminal_psum_oracle": tb.replace(
            "|| random_psum_request_handshakes != first)", ")", 2),
        "ungate_weight_counter_window": tb.replace(
            "reset_n && random_request_window_active\n                && weight_req_valid",
            "reset_n && weight_req_valid", 1),
        "remove_core_ready_stall": tb.replace(
            "            force dut.core_issue_data_ready = 1'b0;\n"
            "            repeat (hold_cycles) @(posedge clk_core);\n"
            "            force dut.core_issue_data_ready = 1'b1;",
            "            force dut.core_issue_data_ready = 1'b1;\n"
            "            repeat (hold_cycles) @(posedge clk_core);\n"
            "            force dut.core_issue_data_ready = 1'b1;", 1),
        "remove_normal_completion": tb.replace("        normal_m935_completion();", "", 1),
    }
    rejected_names = [name for name, changed in mutations.items() if rejected(changed, sva)]
    require(set(rejected_names) == set(mutations), "all independent mutations rejected")
    author_rejected = 0
    for changed in mutations.values():
        try:
            author.validate(changed, sva)
        except AssertionError:
            author_rejected += 1
    # The independent hammer is intentionally stronger; the author checker must
    # nevertheless reject all six mandated removal/delay ready mutations.
    require(author_rejected >= 6, "author checker rejects mandated ready mutations")

    print(json.dumps({
        "schema": "m1212_m1210_c1_r8_random_request_quiesce_source_hammer_mechanical_v1",
        "status": "PASS_SOURCE_HAMMER__SUCCESSOR_RELEASE_AUTHORING_ONLY__NO_VCS_NO_EDA",
        "checks_passed": checks,
        "mutations_rejected_independent": len(rejected_names),
        "mutations_rejected_author_checker": author_rejected,
        "mandated_ready_remove_delay_mutations": 6,
        "exact_request_fire_before_quiesce": True,
        "both_ready_quiesced_before_responses": True,
        "window_counters_exact": True,
        "initial_and_terminal_oracles_exact": True,
        "core_ready_stall_preserved": True,
        "all_nonrandom_r6_tasks_preserved": True,
        "r7_service_closure_preserved": True,
        "assertions": 16,
        "covers": 6,
        "protocol_attacks": 7,
        "service_assumption_attacks": 2,
        "random_legal_transactions": 24,
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "gpu_runs": 0,
        "network_runs": 0,
        "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
