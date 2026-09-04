#!/usr/bin/env python3
"""Independent, read-only M1222 hammer for the M1221 one-shot release."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1221_m1219r9_m1162_c1_common_charge_protocol_exact_sha_r9.sh"
FILELIST = HW / "dc_handoff/filelists/date_m1221_m1219r9_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
CHECKER = HW / "verif_m1221_c1_r9_vcs_release/static_check_m1221_c1_r9_vcs_release_source.py"
TESTS = HW / "verif_m1221_c1_r9_vcs_release/test_m1221_c1_r9_vcs_release_source.py"
SOURCE_CONTRACT = HW / "contracts/m1221_m1220_m1219_c1_r9_vcs_launcher_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1221_m1220_m1219_c1_r9_vcs_launch_release_r1_20260830.json"
TB = HW / "verif_m1219r9_c1_common_charge_protocol/tb_m1219r9_m1162_common_charge_protocol_unit_delay_r9.sv"
M1218 = HW / "reviews/m1218_m1213_c1_r8_vcs_timeout_failure_review_r1_20260830"
M1219 = HW / "reviews/m1219_m1218_c1_r9_observability_source_author_r1_20260830"
M1220 = HW / "reviews/m1220_m1219_c1_r9_observability_source_hammer_r1_20260830"
M1221 = HW / "reviews/m1221_m1220_m1219_c1_r9_vcs_release_author_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    RUNNER: "e018fa988cdf5f1a60033884b3ea5e95e4c4985ac799ead140aaba61e52df1d1",
    FILELIST: "e3819d608f7245ba0145164e5f8d6d3d8bc35fba22f48427aec1f3e9b2b70fc5",
    CHECKER: "777e2e9574e9ca646cefc6f04f34c7f510d1d86cfb2b1b47b524a399766eec4b",
    TESTS: "25b86102ddd958f4509e5342049ad7a3bb18d15f308efc71c684eb8ba080f04f",
    SOURCE_CONTRACT: "d47bbf42ec274f6514f0a4273ec65c2c8303fc6e3adaaee74381edf5f15c91ab",
    RELEASE: "9a1b09c7270c83f795a3fb7a2493941eae24e43601eccc4ee9802f7c3bcc28c0",
    TB: "9666e086c69ecda4670622e063e9d54c89f94f2c77cd5eb012da54ca23492a75",
    M1218 / "review.json": "fa397c84821ecd36c66eca0f598f6345ba7463943ad701e7b90acca119a7c460",
    M1218 / "SHA256SUMS": "63cbc4084a883e0e83dd1ce6b1036db89aab3fbe86c37ecc6a43abc159a71261",
    M1218 / "SHA256SUMS.seal.sha256": "73817b962f5fd76525f9f984184441811f1c2da00aa785a9a9dc4252cbb0e370",
    M1219 / "author_review.md": "0aa4b4515d0e507eee42d8d01dba07cffd24946c0bcad3e35764b0cef5c8d966",
    M1219 / "SHA256SUMS": "3924a2b4dc976de6e4c121c0e2a7254722078a2328891fee0f547e85d66b9647",
    M1219 / "SHA256SUMS.seal.sha256": "5a7007cecaaffa76cc5951951965dfa237e8fb42e4e06cb0ab2444380881f01e",
    M1220 / "review.json": "7004b6f30793971b3d297502d587edb99222e4a90f683a301b1e23dc84356572",
    M1220 / "SHA256SUMS": "d3a064202c2bfa8b257c190d898855451726b72b8ec9dce5736208c8f1daaa04",
    M1220 / "SHA256SUMS.seal.sha256": "fc05610ec4ea83059f8f61bd9263de7130e24c63cb88800fc1979f30ab91ba4d",
    M1221 / "review.json": "17c616dc63135248d9666b01437278c743d6e27c8e7d9ec845590247fb1c8685",
    M1221 / "SHA256SUMS": "90cb4c61b02efbb120ec7cce51862df576598ffc451c170e21094038e85771bb",
    M1221 / "SHA256SUMS.seal.sha256": "5baf4aa57b35c246bb5f7e06fe8cfe9c702c3a53715747814fe6e8f170ee7433",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sidecar(path: Path) -> List[str]:
    errors: List[str] = []
    sums = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    if sums.read_text().split() != [sha(path), path.name]:
        errors.append("sidecar " + path.name)
    if outer.read_text().split() != [sha(sums), sums.name]:
        errors.append("outer sidecar " + path.name)
    return errors


def sealed(root: Path) -> List[str]:
    errors: List[str] = []
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    if not root.is_dir() or root.is_symlink() or outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        return ["outer seal " + root.name]
    listed: Dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1); name = name.lstrip("*")
        if name in listed or Path(name).is_absolute() or ".." in Path(name).parts:
            errors.append("unsafe seal member " + root.name); continue
        listed[name] = digest
    actual = set()
    for base_text, dirs, files in os.walk(root, followlinks=False):
        base = Path(base_text); dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            path = base / name; rel = path.relative_to(root).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or path.is_symlink():
                continue
            if stat.S_ISREG(path.lstat().st_mode): actual.add(rel)
    if actual != set(listed): errors.append("seal population " + root.name)
    for name, digest in listed.items():
        if sha(root / name) != digest: errors.append("seal member drift " + root.name + "/" + name)
    return errors


def audit_runner(text: str) -> List[str]:
    errors: List[str] = []
    need = lambda value, message: errors.append(message) if not value else None
    need(text.count('"${VCS_BIN}" -full64') == 1, "compile count")
    need(text.count('./simv -no_save') == 1, "sim count")
    need(text.count("/usr/bin/timeout --signal=TERM --kill-after=30s") == 1 and
         '"${SIM_TIMEOUT_SECONDS}s" ./simv -no_save' in text and
         "SIM_TIMEOUT_SECONDS=1800" in text, "bounded simulation")
    need(text.count("automatic_retry=false") == 2 and
         text.count("'automatic_retry':False") == 2 and
         "automatic_retry=true" not in text and "'automatic_retry':True" not in text,
         "retry surface")
    try:
        seal_gate = text.index('for sealed in "${M1218_FAILURE}"')
        review_gate = text.index('sha_exact "${M1221_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}"')
        checker_gate = text.index('"${PYTHON_BIN}" -I "${RELEASE_CHECKER}"')
        freshness = text.index('[[ ! -e "${ATTEMPT}"')
        collision = text.index("blocked={'vcs','vcs1','simv'")
        memory = text.index('mem_kib="$(awk')
        attempt = text.index('/bin/mkdir -- "${ATTEMPT}"')
        work = text.index('/bin/mkdir -- "${WORK}"')
        active = text.index("WORK_ACTIVE=1")
        compile_at = text.index('"${VCS_BIN}" -full64')
        sim_at = text.index('./simv -no_save')
        need(seal_gate < review_gate < checker_gate < freshness < collision < memory < attempt < work < active < compile_at < sim_at,
             "one-shot gate order")
    except ValueError:
        errors.append("one-shot gate anchor")
    need(text.count("trap on_exit EXIT") == 1 and
         "if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d \"${WORK}\" ]]" in text and
         'phase_dump "phase_watchdog_timeout_dump.txt"' in text and
         'seal_dir "${WORK}"' in text and 'mv -- "${WORK}" "${QUARANTINE}"' in text,
         "failure quarantine")
    need("RUN_FAILED_OR_INCOMPLETE.txt" in text and "TIMEOUT_M1219R9" in text and
         "tail -n 200 sim.log" in text, "failure dump")
    phases = re.search(r"for phase in ([A-Z0-9_ ]+); do", text)
    expected = ["DIRECTED", "RESET_PENDING", "STICKY_ATTACKS", "SERVICE_ATTACKS",
                "RANDOM", "NORMAL_M935", "CLEAN_RESET_PREP"]
    need(phases is not None and phases.group(1).split() == expected,
         "seven phase pairs")
    need("for index in $(seq 0 23); do" in text and
         "PHASE_M1219R9_RANDOM_TRANSACTION_ENTER index=${index}" in text and
         "PHASE_M1219R9_RANDOM_TRANSACTION_COMPLETE index=${index}" in text,
         "24 random pairs")
    need("if rg -q '^TIMEOUT_M1219R9 ' sim.log; then exit 34; fi" in text,
         "internal timeout rejection")
    need(text.count("verify_recursive_seal \"${sealed}\"") == 1 and
         "${M1218_FAILURE}" in text and "${M1219_AUTHOR}" in text and
         "${M1220_HAMMER}" in text and "${AUTHOR_DIR}" in text and
         "${RELEASE_HAMMER}" in text, "evidence recursive seals")
    need("rm -" not in text and "os.replace" not in text, "destructive overwrite")
    need('mv -- "${WORK}" "${RESULT}"' in text and "seal_dir \"${WORK}\"" in text,
         "success recursive seal")
    return errors


def audit_filelist(text: str) -> List[str]:
    expected = [
        "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v",
        str(HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"),
        str(HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"),
        str(HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"),
        str(HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"),
        str(TB),
    ]
    return [] if text.splitlines() == expected else ["filelist ordered population"]


def audit_contracts(source: dict, release: dict) -> List[str]:
    errors: List[str] = []
    if source.get("status") != "M1221_C1_R9_ONE_SHOT_RELEASE_SOURCE_READY__FRESH_M1222_REQUIRED__NO_VCS_NO_EDA":
        errors.append("source status")
    if release.get("status") != "AUTHORIZE_ONE_M1221_R9_UNIT_DELAY_VCS_ATTEMPT_AFTER_FRESH_M1222_HAMMER":
        errors.append("release status")
    policy = source.get("execution_policy", {})
    if policy.get("vcs_compiles") != 1 or policy.get("bounded_simv_runs") != 1 or \
            policy.get("simv_timeout_seconds") != 1800 or policy.get("automatic_retry") is not False:
        errors.append("source execution policy")
    auth = release.get("authorization", {})
    if auth != {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0,
                "automatic_retry": False}:
        errors.append("release authorization")
    hammer = release.get("fresh_release_hammer", {})
    if hammer.get("schema") != "m1222_m1221_m1219_c1_r9_vcs_release_hammer_review_r1_v1" or \
            hammer.get("status") != "PASS_M1222_M1221_C1_R9_ONE_SHOT_UNIT_DELAY_VCS_RELEASE__AUTHORIZE_ONE_LAUNCH" or \
            hammer.get("minimum_score") != 95 or hammer.get("maximum_p0") != 0 or hammer.get("maximum_p1") != 0:
        errors.append("future hammer admission")
    for data in (source, release):
        for key, value in data.get("claim_boundary", {}).items():
            if value is not False: errors.append("claim boundary " + key)
    return errors


def main() -> int:
    errors: List[str] = []
    hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path)
              for path in EXPECTED}
    for path, digest in EXPECTED.items():
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            errors.append("identity drift " + str(path))
    for root in (M1218, M1219, M1220, M1221): errors.extend(sealed(root))
    for artifact in (HW / "contracts/m1219_m1218_m1213_c1_r9_observability_source_contract_r1_20260830.json",
                     SOURCE_CONTRACT, RELEASE): errors.extend(sidecar(artifact))
    runner = RUNNER.read_text(); filelist = FILELIST.read_text()
    errors.extend(audit_runner(runner)); errors.extend(audit_filelist(filelist))
    errors.extend(audit_contracts(json.loads(SOURCE_CONTRACT.read_text()),
                                  json.loads(RELEASE.read_text())))
    syntax = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)], check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if syntax.returncode != 0: errors.append("bash syntax " + syntax.stderr.strip())

    mutations = {
        "second_compile": runner + '\n"${VCS_BIN}" -full64\n',
        "second_sim": runner + "\n./simv -no_save\n",
        "remove_timeout": runner.replace("/usr/bin/timeout --signal=TERM --kill-after=30s", "/usr/bin/env", 1),
        "enable_retry": runner.replace("automatic_retry=false", "automatic_retry=true", 1),
        "remove_phase": runner.replace(" RANDOM NORMAL_M935 CLEAN_RESET_PREP", " RANDOM CLEAN_RESET_PREP", 1),
        "remove_random_pair": runner.replace("for index in $(seq 0 23); do", "for index in $(seq 0 22); do", 1),
        "remove_timeout_dump": runner.replace("phase_watchdog_timeout_dump.txt", "phase_dump_removed.txt", 1),
        "remove_quarantine": runner.replace('mv -- "${WORK}" "${QUARANTINE}"', ": # quarantine removed", 1),
    }
    mutation_results = {}
    for name, mutant in mutations.items():
        rejected = bool(audit_runner(mutant))
        mutation_results[name] = "REJECTED" if rejected else "ACCEPTED_IN_ERROR"
        if not rejected: errors.append("mutation accepted " + name)

    attempt = HW / "results/.m1221_m1219r9_m1162_c1_common_charge_protocol_vcs_r9_attempt_consumed"
    result = HW / "results/m1221_m1219r9_m1162_c1_common_charge_protocol_unit_delay_vcs_r9_20260830"
    fresh = (not os.path.lexists(attempt) and not os.path.lexists(result) and
             not list((HW / "results").glob(".m1221_m1219r9_m1162_c1_common_charge_protocol_vcs_r9_work.*")) and
             not list((HW / "results").glob(
                 "m1221_m1219r9_m1162_c1_common_charge_protocol_unit_delay_vcs_r9_20260830.failed_or_incomplete.*")))
    if not fresh: errors.append("M1221 namespace not fresh")
    output = {
        "schema": "m1222_m1221_c1_r9_vcs_release_hammer_mechanical_r1_v1",
        "status": "PASS" if not errors else "FAIL",
        "score": 99 if not errors else 0,
        "p0_count": len(errors), "p1_count": 0, "p2_count": 0,
        "hashes": hashes,
        "fresh_namespace": fresh,
        "one_compile": True,
        "one_bounded_sim": True,
        "sim_timeout_seconds": 1800,
        "automatic_retry": False,
        "phase_pairs": 7,
        "random_transaction_pairs": 24,
        "timeout_dump": True,
        "recursive_failure_quarantine": True,
        "independent_mutations": mutation_results,
        "authorization": {"vcs_compiles": 1, "simv_runs": 1,
                          "all_other_eda_runs": 0, "automatic_retry": False},
        "vcs_invoked_by_hammer": False,
        "eda_invoked_by_hammer": False,
        "errors": errors,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
