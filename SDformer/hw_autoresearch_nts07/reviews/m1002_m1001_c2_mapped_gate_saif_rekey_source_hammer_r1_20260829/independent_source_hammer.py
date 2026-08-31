#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent static-only M1002 hammer for the frozen-M979 M1001 rekey."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONTRACT = HW / "contracts/m1001_m979_c2_mapped_gate_saif_rekey_source_contract_r1_20260829.json"
RUNNER = HW / "dc_handoff/scripts/run_m1005_m1001_c2_mapped_gate_saif_one_shot.sh"
CHECKER = HW / "system_simulator/scripts/check_m1001_m979_c2_mapped_gate_saif_rekey_source.py"
TESTS = HW / "system_simulator/tests/test_m1001_m979_c2_mapped_gate_saif_rekey_source.py"
TB = HW / "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv"
UCLI = HW / "dc_handoff/scripts/m979_c2_mapped_gate_per_case_saif.ucli.tcl"
FROZEN_CHECKER = HW / "system_simulator/scripts/check_m979_c2_mapped_gate_saif_source.py"
FROZEN_TESTS = HW / "system_simulator/tests/test_m979_c2_mapped_gate_saif_source.py"
FROZEN_CONTRACT = HW / "contracts/m979_m974_c2_three_axis_mapped_gate_saif_source_contract_r1_20260829.json"
OLD_RUNNER = HW / "dc_handoff/scripts/run_m993_m979_c2_mapped_gate_saif_one_shot.sh"
M979_RECEIPT = HW / "reviews/m979_m974_c2_mapped_gate_saif_source_receipt_r1_20260829"
M1001_RECEIPT = HW / "reviews/m1001_m979_c2_mapped_gate_saif_rekey_source_receipt_r1_20260829"
RESULTS = HW / "results"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "contract": "7afc4c093b802bdfd97aea101c803735e993c2eef57983311d3eb1a3d6bd36c6",
    "runner": "559ebb252fcda7592106413f5e2106a9156fffc5ce0de216baae45e218c0635d",
    "checker": "75b2463b8e031f56f34c06aa83d4d75520c84e8dddb5d0d3fdc5e618d346bf44",
    "tests": "98cfe686f9182684dc5d00df2794c803cd2a2fe3716df702830bd86317d2c0ac",
    "tb": "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c",
    "ucli": "846cd4a1b877803cce986b39cdf0a27ec87b59451ca7e6fc9141c999df85cdad",
    "frozen_checker": "409a6d996e95e4fc46d2ff3cf8e26fbe5e52594d1e7b1522db599811025382d5",
    "frozen_tests": "aaff81432b28ac506142d4890096f379db74d380516892107f20e10a6fcf2461",
    "frozen_contract": "d2939e24e587b03680b7b4e0265a8fc8b3dbbea89759e2268e97b118fe32455c",
    "old_runner": "ba98f230cd676767c121760edf4025fbb71acbb7ddadfa4f695cee9acdf51ecc",
    "m979_review": "8992b243dfe8397efe66eff4e9ba70435522172e94dd79872de7e1b7139f48cd",
    "m979_manifest": "67455efdda9eaaaa0e223eea3e61d4dbe00024ac22cc69c4d5a49c50c09731a6",
    "m979_outer": "da08f8c116e5ba28dbf839fb733e4dc0c0efec3847fb59339f98338c16401dd9",
    "m1001_review": "f15ca079e03fdd4d716c6fc35bfacd1ceecd8eba7de668ecd5448bce552ee25e",
    "m1001_manifest": "cb95696068dd9d4c92e3ca8ad70e0ddb6e224ed49c7f5d006e769832b7f39de1",
    "m1001_outer": "3e120550cc908a43705766fed4f30a4d0cb45db946b889677fd1179a1177414d",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_directory(directory, manifest_sha, outer_sha):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(), "receipt missing/symlink")
    require(manifest.is_file() and outer.is_file() and
            not manifest.is_symlink() and not outer.is_symlink(), "receipt seal missing")
    require(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
            "receipt seal identity drift")
    require(outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
            "receipt outer content drift")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1); rel = rel.lstrip("*")
        require(rel not in listed and ".." not in Path(rel).parts,
                "unsafe/duplicate receipt member")
        member = directory / rel
        require(member.is_file() and not member.is_symlink() and sha(member) == digest,
                "receipt member drift: " + rel)
        listed[rel] = digest
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(set(listed) == actual, "receipt recursive exact-set drift")
    require(not [path for path in directory.rglob("*") if path.is_symlink()],
            "receipt contains symlink")
    return {"entries": len(listed), "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load module")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def fake_saif(duration, memory_tc=4, tx=0, reset_tc=0):
    return f'''(SAIFILE (DURATION {duration})
      (INSTANCE tb_m979_c2_three_axis_mapped_gate_case_saif (INSTANCE dut
       (PORT (clk_core (T0 1) (T1 1) (TX {tx}) (TC 20))
        (rst_core (T0 1) (T1 0) (TX 0) (TC {reset_tc}))
        (header_accept (T0 1) (T1 1) (TX 0) (TC 2))
        (raw_accept (T0 1) (T1 1) (TX 0) (TC 3))
        (mem_req_accept[0] (T0 1) (T1 1) (TX 0) (TC {memory_tc}))
        (mem_rsp_accept[0] (T0 1) (T1 1) (TX 0) (TC {memory_tc}))
        (result_accumulator[0] (T0 1) (T1 1) (TX 0) (TC 6))
        (result_accept (T0 1) (T1 1) (TX 0) (TC 4))
        (token_done_accept (T0 1) (T1 1) (TX 0) (TC 2))))))'''


def validate_saif_matrix(checker):
    anchors = {"k8": [51, 131, 486, 1231, 14],
               "k1x8": [53, 133, 499, 1246, 14]}
    passed = []
    with tempfile.TemporaryDirectory(prefix="m1002_saif_matrix_") as temporary:
        root = Path(temporary)
        for axis, cycles in anchors.items():
            for case_id, cycle in enumerate(cycles):
                path = root / f"{axis}_{case_id}.saif"
                path.write_text(fake_saif(cycle * 3, 0 if case_id == 4 else 4))
                value = checker.M979.validate_saif(path, axis, case_id, cycle)
                require(value["tx_nonzero"] == 0 and value["reset_tc"] == 0.0,
                        "TX/reset invariant drift")
                passed.append((axis, case_id, cycle))
        # K1 is deliberately diagnostic: it still validates duration/window,
        # TX/reset and cones, but has no hard M867 cycle anchor.
        k1 = root / "k1_case0.saif"; k1.write_text(fake_saif(21))
        require(checker.M979.validate_saif(k1, "k1", 0, 7)["duration_ns"] == 21,
                "K1 diagnostic route drift")

        negative = {}
        checks = [
            ("duration", fake_saif(154), "k8", 0, 51, "duration"),
            ("tx", fake_saif(153, tx=1), "k8", 0, 51, "TX"),
            ("reset", fake_saif(153, reset_tc=2), "k8", 0, 51, "reset"),
            ("nonzero_case_memory", fake_saif(153, memory_tc=0), "k8", 0, 51,
             "memory cone"),
        ]
        for index, (name, text, axis, case_id, cycles, token) in enumerate(checks):
            path = root / f"negative_{index}.saif"; path.write_text(text)
            try: checker.M979.validate_saif(path, axis, case_id, cycles)
            except RuntimeError as error: negative[name] = token in str(error)
            else: negative[name] = False
        zero = root / "zero_case.saif"; zero.write_text(fake_saif(42, memory_tc=0))
        zero_value = checker.M979.validate_saif(zero, "k8", 4, 14)
        require(all(negative.values()) and
                zero_value["major_cone_tc"]["memory"] == 0.0 and
                zero_value["zero_case_memory_nonzero_required"] is False,
                "negative/zero-event gate drift")
    return {"hard_anchor_cases_passed": len(passed), "k1_diagnostic_passed": True,
            "negative_gates": negative, "zero_case_memory_zero_accepted": True}


def main():
    paths = {"contract": CONTRACT, "runner": RUNNER, "checker": CHECKER,
             "tests": TESTS, "tb": TB, "ucli": UCLI,
             "frozen_checker": FROZEN_CHECKER, "frozen_tests": FROZEN_TESTS,
             "frozen_contract": FROZEN_CONTRACT, "old_runner": OLD_RUNNER,
             "docs359": DOC359}
    for key, path in paths.items():
        require(path.is_file() and not path.is_symlink(), key + " missing/symlink")
        require(sha(path) == EXPECTED[key], key + " SHA drift")
    m979_seal = verify_directory(
        M979_RECEIPT, EXPECTED["m979_manifest"], EXPECTED["m979_outer"])
    m1001_seal = verify_directory(
        M1001_RECEIPT, EXPECTED["m1001_manifest"], EXPECTED["m1001_outer"])
    require(sha(M979_RECEIPT / "review.json") == EXPECTED["m979_review"] and
            sha(M1001_RECEIPT / "review.json") == EXPECTED["m1001_review"],
            "receipt review identity drift")

    static = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-I", str(CHECKER),
         "--contract", str(CONTRACT)], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, check=False, timeout=30)
    tests = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-I", str(TESTS)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        check=False, timeout=30)
    require(static.returncode == 0 and "PASS_M1001_STATIC_REKEY__NO_EDA" in static.stdout,
            "M1001 static checker failed")
    require(tests.returncode == 0 and "Ran 7 tests" in tests.stderr and
            "OK" in tests.stderr, "M1001 unit tests failed")
    require(subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                           check=False).returncode == 0, "runner bash syntax failed")

    contract = json.loads(CONTRACT.read_text())
    review = json.loads((M1001_RECEIPT / "review.json").read_text())
    require(contract["status"] == "PASS_M1001_REKEY_SOURCE_ONLY__NO_EDA" and
            contract["launch_now"] is False and
            contract["rekey"]["old_execution"] == "PROHIBITED_NAMESPACE_COLLISION" and
            contract["rekey"]["new_chain"] ==
            "M1001 -> M1002 -> M1003 -> M1004 -> M1005",
            "M1001 authority/numbering drift")
    require(review["rekey"]["old_m993_execution"] ==
            "PROHIBITED_NAMESPACE_COLLISION" and
            review["claim_boundary"]["m1003_release_authorized"] is False and
            review["claim_boundary"]["m1005_run_authorized"] is False,
            "M1001 receipt authority drift")
    canonical = contract["canonical"]
    require(canonical == {
        "source": "M1001",
        "source_hammer": "reviews/m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_r1_20260829",
        "release": "contracts/m1003_m1001_c2_mapped_gate_saif_launch_release_r1_20260829.json",
        "release_hammer": "reviews/m1004_m1003_m1001_c2_mapped_gate_saif_release_hammer_r1_20260829",
        "runner": "dc_handoff/scripts/run_m1005_m1001_c2_mapped_gate_saif_one_shot.sh",
        "result": "results/m1005_m1001_c2_three_axis_mapped_gate_saif_r1_20260829",
        "attempt": "results/.m1005_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed",
        "failure_prefix": "results/m1005_m1001_c2_three_axis_mapped_gate_saif_r1_20260829.failed_or_incomplete.",
    }, "M1001-M1005 canonical pin drift")

    runner = RUNNER.read_text(); ucli = UCLI.read_text(); tb = TB.read_text()
    require("m993_" not in runner.lower() and "M993_" not in runner,
            "old M993 execution identity survived in new runner")
    require('for axis in k1 k8 k1x8; do' in runner and
            'for case_id in 0 1 2 3 4; do' in runner and
            runner.count('"${vcs}" -full64') == 1 and
            runner.index('"${vcs}" -full64') < runner.index('for case_id in 0 1 2 3 4; do'),
            "three-axis/five-case fresh compile structure drift")
    require('saif="${axis_dir}/case${case_id}.saif"' in runner and
            '--saif "${saif}" --axis "${axis}"' in runner,
            "per-axis/per-case SAIF validation drift")
    require("power tb_m979_c2_three_axis_mapped_gate_case_saif.dut" in ucli and
            ucli.index("run") < ucli.index("power -enable") <
            ucli.index("power -disable") < ucli.index("power -report") and
            "M979_SAIF_WINDOW_START" in tb and "M979_SAIF_WINDOW_STOP" in tb,
            "DUT-only capture window drift")
    require("if(mode!=9)" in tb and "case(case_id)" in tb and
            "4:begin blocks=1;mode=9;end" in tb,
            "zero-event case construction drift")
    semantics = contract["semantic_freeze"]
    require(semantics["axes"] == ["k1", "k8", "k1x8"] and
            semantics["cases_per_axis"] == 5 and
            semantics["k8_cycle_anchors"] == [51, 131, 486, 1231, 14] and
            semantics["k1x8_cycle_anchors"] == [53, 133, 499, 1246, 14] and
            semantics["dut_only_saif"] is True and semantics["all_tx_zero"] is True and
            semantics["reset_tc_zero"] is True and
            semantics["zero_event_memory_cone_exception"] is True,
            "semantic freeze drift")

    checker = load_module("m1002_m1001_checker", CHECKER)
    matrix = validate_saif_matrix(checker)
    result = RESULTS / "m1005_m1001_c2_three_axis_mapped_gate_saif_r1_20260829"
    attempt = RESULTS / ".m1005_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"
    stale = sorted(path.name for path in RESULTS.glob(
        "m1005_m1001_c2_three_axis_mapped_gate_saif_r1_20260829*"))
    stale += sorted(path.name for path in RESULTS.glob(
        ".m1005_m1001_c2_three_axis_mapped_gate_saif*"))
    require(not result.exists() and not result.is_symlink() and
            not attempt.exists() and not attempt.is_symlink() and not stale,
            "M1005 result/attempt namespace not fresh")

    return {
        "schema": "m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_v1",
        "status": "PASS_M1002_M1001_SOURCE_HAMMER",
        "verdict": "GO_AUTHOR_M1003_RELEASE_ONLY",
        "score_out_of_100": 98,
        "p0_count": 0, "p1_count": 0, "p2_count": 1,
        "pins": {key + "_sha256": EXPECTED[key] for key in
                 ("contract", "runner", "checker", "tests", "tb", "ucli",
                  "frozen_checker", "frozen_tests", "frozen_contract",
                  "old_runner", "docs359")},
        "receipt_seals": {"m979": m979_seal, "m1001": m1001_seal},
        "semantic_matrix": matrix,
        "positive": {"frozen_m979_file_count": 6,
                     "axes": 3, "cases_per_axis": 5, "total_cases": 15,
                     "dut_only_window": True, "all_tx_zero": True,
                     "reset_tc_zero": True, "zero_case_exception": True,
                     "old_m993_execution_prohibited": True,
                     "m1005_namespace_fresh": True,
                     "vcs_pt_ptpx_runs": 0},
        "p2": [{"id": "P2_FINAL_RESULT_RENAME_SAME_UID_RACE",
                "finding": "The canonical ATTEMPT serializes cooperating runners, but final mv lacks -T/final RESULT recheck against a same-UID noncooperating creator.",
                "impact": "Outside the cooperating-runner threat model; M1005 result hammer must require a single sealed canonical result and reject nesting."}],
        "decision": {"m1003_release_authoring_authorized": True,
                     "m1005_run_authorized_now": False,
                     "automatic_retry": False},
        "scope": {"static_only": True, "vcs_runs": 0, "pt_runs": 0,
                  "ptpx_runs": 0, "gpu_remote_runs": 0,
                  "docs359_modified": False},
        "claim_boundary": {"source_ready": True, "saif_created": False,
                           "power": False, "energy": False,
                           "paper_ppa_ready": False, "headline": False},
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
