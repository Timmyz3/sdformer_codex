#!/usr/bin/env python3
"""Independently hammer and validate the M31-r5 evidence hardening.

This reviewer does not rerun DC or Formality and does not modify their evidence.
It rehashes the frozen evidence, reruns the exact 40-test regression, rebuilds the
r5 receipt twice in memory, and executes synthetic attacks against the former r4
P1 path, ledger, RTL-binding, VCS-ledger, and clock weaknesses.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


WORK = Path("/home/zhumd/work")
REPO = WORK / "sdformer_codex/SDformer"
HW = REPO / "hw_autoresearch_nts07"
RUNS = WORK / "synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs"
R5_SOURCE = HW / "dc_handoff/scripts/build_m31_r5_synopsys_receipt.py"
R5_SEALER = HW / "dc_handoff/scripts/seal_m31_r5_evidence_reanchor.py"
R5_TEST = HW / "tests/test_build_m31_r5_synopsys_receipt.py"
R5_RECEIPT = HW / (
    "contracts/m31_synopsys_receipt_r5_evidence_hardened_20260822.json")
REVIEW_OUTPUT = HW / (
    "results/m31_r5_independent_hammer_review_20260822/"
    "m31_r5_independent_hammer_review.json")

EXPECTED = {
    "r5_builder": "16ac8369c39402024a47a57a95901b0118791a773be6fb495a5760f836a4cd0a",
    "r5_reanchor_sealer": "56e416cd7370cd370b3d235c8a075a5e386776aff4dc7466f3fba1c39b5413d3",
    "r5_test": "314072d027a8ddde29bc55aa101a1624987660c21de0de972b4fe487bd1c4e0b",
    "r5_receipt": "5135fb099e3bd434f928d5345c74e01db13c89604ae3956d1117782938b071b7",
    "r5_reanchor": "9e132540a206a0f6747d733ff14b3beee23ad5a7716c82f3ee8fb437f1deac07",
    "r4_vcs_receipt": "bae2f05e74ffa8863195bda9f222c22fc06364ade872e9cf83d3cd4106e5b77d",
    "r4_vcs_admission": "e8bd1b6452280396a5c8fc83ce79f34d1ae08256f97b469613207418dcfd0ff6",
    "r4_vcs_snapshot_ledger": "41009ec9ec86d4e19489bd49816634ca148340a0f19f784bd2d18bf2d3d0f22d",
    "r4_synopsys_receipt": "5cf35c5ef92e174e04d4169c2c924a5ac6962ab19dcf0fe4fa48c2f5e0e5c561",
    "r4_dc_audit": "4e4d24d25651d6fe37c018eb34bd7e9472a4869fd4c0da669a3db716e472f095",
    "r4_dc_live_ledger": "be3619946ddc22e2b05d24931eaaa1fe17d42cf60c433253b3422725d8054ef8",
    "r4_dc_sealed_ledger": "919f6e54ebb0841855306870e8ab3aa402ebc4ddfc00cefc9a1e14e0be9bc4fe",
    "r4_dc_log": "7294185b993e542064fd916d1470d3f007b25df8de13b842ccc7a661335088c1",
    "r4_fm_audit": "aff84cef659e40930156d569a8f10d95d95e9bf2b8bdd2faa757ce7ab94f8f88",
    "r4_fm_live_ledger": "1b8a4dc99097e9f1641ad8a87cb8b86ee2423b07c0a124e8dd1def1ce6b1fa0f",
    "r4_fm_snapshot_ledger": "0b9586ae19bafbbb6d968067271c39b6c960446590b96a710cacccfd98ed0528",
    "r4_fm_log": "976d75661f1c5a8cda094f2704a75f77ed38dbcc321ea007c88682212738ebc0",
}

TEST_MODULES = (
    "hw_autoresearch_nts07/system_simulator/tests/"
    "test_m31_r4_static_phase_vcs_validator.py",
    "hw_autoresearch_nts07/tests/test_audit_m31_r4_dc_reports.py",
    "hw_autoresearch_nts07/tests/test_audit_m31_r4_formality.py",
    "hw_autoresearch_nts07/tests/test_build_m31_r4_synopsys_receipt.py",
    "hw_autoresearch_nts07/tests/test_build_m31_r5_synopsys_receipt.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_no_duplicates(path):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate independent-review JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=hook)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_r5():
    spec = importlib.util.spec_from_file_location("m31_r5_hammer_target",
                                                  str(R5_SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_paths(r5):
    run = RUNS / r5.RUN_NAME
    return run, {
        "r5_builder": R5_SOURCE,
        "r5_reanchor_sealer": R5_SEALER,
        "r5_test": R5_TEST,
        "r5_receipt": R5_RECEIPT,
        "r5_reanchor": WORK / r5.REANCHOR_RELATIVE,
        "r4_vcs_receipt": HW / (
            "contracts/m31_output_receipt_r4_static_phase_20260822.json"),
        "r4_vcs_admission": HW / (
            "results/m31_r4_static_phase_vcs_machine_admission_20260822/"
            "m31_r4_static_phase_vcs_machine_admission.json"),
        "r4_vcs_snapshot_ledger": HW / (
            "system_simulator/evidence/"
            "m31_r4_vcs_inputs_c094849e_20260822.sha256"),
        "r4_synopsys_receipt": HW / (
            "contracts/m31_synopsys_receipt_r2_static_phase_20260822.json"),
        "r4_dc_audit": run / "reports/m31_r4_dc_machine_audit.json",
        "r4_dc_live_ledger": run / "evidence.sha256",
        "r4_dc_sealed_ledger": run / "sealed_dc_evidence.sha256",
        "r4_dc_log": run / "dc.log",
        "r4_fm_audit": run / (
            "formality_machine_audit_{}.json".format(r5.ATTEMPT)),
        "r4_fm_live_ledger": run / (
            "formality_evidence_{}.sha256".format(r5.ATTEMPT)),
        "r4_fm_snapshot_ledger": run / (
            "sealed_formality_evidence_{}.sha256".format(r5.FM_SNAPSHOT_TAG)),
        "r4_fm_log": run / "formality_{}.log".format(r5.ATTEMPT),
    }


def validate_hashes(r5):
    run, paths = exact_paths(r5)
    observed = {}
    for label in sorted(EXPECTED):
        path = paths[label]
        require(path.is_file(), "missing independent-review anchor {}".format(label))
        digest = sha256(path)
        require(digest == EXPECTED[label],
                "independent-review anchor drift: {}".format(label))
        observed[label] = {"path": str(path), "sha256": digest}
    return run, paths, observed


def run_40_tests():
    command = [sys.executable, "-B", "-m", "unittest", "-v"] + list(TEST_MODULES)
    completed = subprocess.run(
        command, cwd=str(REPO), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, universal_newlines=True)
    output = completed.stdout
    require(completed.returncode == 0, "M31-r5 independent 40-test regression failed")
    require(re.search(r"^Ran 40 tests in [0-9.]+s$", output, re.MULTILINE),
            "M31-r5 regression did not run exactly 40 tests")
    require(output.rstrip().endswith("OK"), "M31-r5 regression lacks terminal OK")
    require(output.count(" ... ok") == 40,
            "M31-r5 regression does not contain exactly 40 passing rows")
    return {"command_python": sys.executable, "tests_run": 40,
            "tests_passed": 40, "exit_status": completed.returncode,
            "terminal_status": "OK"}


def validate_reanchor(r5, run, paths):
    expected = r5.expected_reanchor_files(WORK, REPO, run)
    ledger, rows = r5.parse_exact_relative_ledger(
        paths["r5_reanchor"], WORK, expected,
        "independent r5 reanchor", EXPECTED["r5_reanchor"])
    require(len(rows) == 10, "M31-r5 reanchor is not exactly 10 entries")
    return {"ledger_path": str(ledger), "ledger_sha256": sha256(ledger),
            "entries_checked": len(rows), "entries_passed": len(rows),
            "absolute_entries": 0, "path_escape_entries": 0,
            "extra_entries": 0, "missing_entries": 0}


def deterministic_build(r5, run, paths):
    args = argparse.Namespace(
        work_root=WORK, repo_root=REPO, runs_root=RUNS, run_dir=run,
        reanchor_ledger=paths["r5_reanchor"], date="2026-08-22", output=None)
    first = (json.dumps(r5.build(args), indent=2, sort_keys=True) + "\n").encode()
    second = (json.dumps(r5.build(args), indent=2, sort_keys=True) + "\n").encode()
    receipt = R5_RECEIPT.read_bytes()
    require(first == second == receipt,
            "M31-r5 repeated build is not byte deterministic")
    digest = hashlib.sha256(first).hexdigest()
    require(digest == EXPECTED["r5_receipt"],
            "M31-r5 deterministic build SHA drift")
    return {"build_count": 2, "builds_byte_equal": True,
            "builds_equal_live_receipt": True, "receipt_bytes": len(receipt),
            "receipt_sha256": digest}


def write_ledger(path, rows):
    Path(path).write_text("".join(
        "{}  {}\n".format(sha256(target), relative)
        for relative, target in rows), encoding="utf-8")


def expect_reject(results, name, callback):
    try:
        callback()
    except (ValueError, OSError):
        results.append(name)
        return
    raise ValueError("M31-r5 independent attack was accepted: {}".format(name))


def make_cross_binding(r5, base):
    fm, vcs, dc = base / "fm", base / "vcs", base / "dc"
    bindings = {
        "rtl_m31/qfit_signed_int8_mul96_pool.sv": HW / (
            "rtl_m31/qfit_signed_int8_mul96_pool.sv"),
        "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv": HW / (
            "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv"),
    }
    for relative, source in bindings.items():
        for root, prefix in ((fm, "inputs/hw_root"),
                             (vcs, "inputs/hw_root"), (dc, "inputs")):
            target = root / prefix / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(source), str(target))
    filelist = fm / (
        "inputs/hw_root/dc_handoff/filelists/date_m31_unified_t10_t2_dc.f")
    filelist.parent.mkdir(parents=True, exist_ok=True)
    filelist.write_text(
        "rtl_m31/qfit_signed_int8_mul96_pool.sv\n"
        "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv\n",
        encoding="utf-8")
    other = {
        "dc_handoff/filelists/date_m31_unified_t10_t2_vcs.f": "filelist\n",
        "dc_handoff/scripts/run_vcs_m31_unified_t10_t2_sva.sh": "runner\n",
        "tb_m31/tb_qfit_atlif_unified_t10_t2_stream_core.sv": "tb\n",
        "verif_m31/qfit_atlif_unified_t10_t2_stream_assertions.sv": "sva\n",
    }
    manifest = []
    for relative in sorted(set(bindings) | set(other)):
        target = vcs / "inputs/hw_root" / relative
        if relative in other:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(other[relative], encoding="utf-8")
        manifest.append("{}  {}\n".format(sha256(target), relative))
    (vcs / "input_sha256.txt").write_text("".join(manifest), encoding="utf-8")
    return fm, vcs, dc


def replace_manifest_hash(manifest, relative, digest):
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        old, name = line.split("  ", 1)
        rows.append("{}  {}\n".format(digest if name == relative else old, name))
    manifest.write_text("".join(rows), encoding="utf-8")


def run_attacks(r5, run, paths):
    rejected = []
    safe_aliases = []
    with tempfile.TemporaryDirectory(prefix="m31_r5_hammer_") as directory:
        base = Path(directory)
        root = base / "root"
        root.mkdir()
        (root / "a").write_text("a\n", encoding="utf-8")
        (root / "b").write_text("b\n", encoding="utf-8")
        outside = base / "outside"
        outside.write_text("a\n", encoding="utf-8")

        expect_reject(rejected, "repo_root_escape", lambda: r5.validate_roots(
            WORK, WORK / "synopsys_date_dual", RUNS, run))
        expect_reject(rejected, "runs_root_escape", lambda: r5.validate_roots(
            WORK, REPO, REPO / "hw_autoresearch_nts07", run))
        expect_reject(rejected, "run_root_escape", lambda: r5.validate_roots(
            WORK, REPO, RUNS, REPO))
        expect_reject(rejected, "copied_run_outside_canonical_root",
                      lambda: r5.validate_roots(WORK, REPO, RUNS, base))
        expect_reject(rejected, "canonical_audit_out_of_root",
                      lambda: r5.canonical_file(outside, root, "audit"))

        link = root / "audit_link"
        link.symlink_to(outside)
        expect_reject(rejected, "audit_symlink_same_bytes",
                      lambda: r5.canonical_file(link, root, "audit"))
        external_snapshot = base / "external_snapshot"
        external_snapshot.mkdir()
        (external_snapshot / "a").write_text("a\n", encoding="utf-8")
        snapshot_link = root / "snapshot"
        snapshot_link.symlink_to(external_snapshot, target_is_directory=True)
        snapshot_ledger = root / "snapshot.sha256"
        snapshot_ledger.write_text(
            "{}  snapshot/a\n".format(sha256(external_snapshot / "a")),
            encoding="utf-8")
        expect_reject(rejected, "snapshot_root_symlink_replacement", lambda:
                      r5.parse_exact_relative_ledger(
                          snapshot_ledger, root, {"snapshot/a"}, "snapshot"))

        absolute = root / "absolute.sha256"
        absolute.write_text("{}  {}\n".format(sha256(root / "a"), root / "a"),
                            encoding="utf-8")
        expect_reject(rejected, "absolute_ledger_entry", lambda:
                      r5.parse_exact_relative_ledger(
                          absolute, root, {"a"}, "absolute"))
        dotdot = root / "dotdot.sha256"
        dotdot.write_text("{}  ../outside\n".format(sha256(outside)),
                          encoding="utf-8")
        expect_reject(rejected, "dotdot_ledger_escape", lambda:
                      r5.parse_exact_relative_ledger(
                          dotdot, root, {"../outside"}, "dotdot"))
        normalized = root / "normalized.sha256"
        normalized.write_text("{}  ./a\n".format(sha256(root / "a")),
                              encoding="utf-8")
        expect_reject(rejected, "normalized_ledger_alias", lambda:
                      r5.parse_exact_relative_ledger(
                          normalized, root, {"a"}, "normalized"))
        extra = root / "extra.sha256"
        write_ledger(extra, (("a", root / "a"), ("b", root / "b")))
        expect_reject(rejected, "rehashed_extra_ledger_entry", lambda:
                      r5.parse_exact_relative_ledger(extra, root, {"a"}, "extra"))
        missing = root / "missing.sha256"
        write_ledger(missing, (("a", root / "a"),))
        expect_reject(rejected, "rehashed_missing_ledger_entry", lambda:
                      r5.parse_exact_relative_ledger(
                          missing, root, {"a", "b"}, "missing"))
        expect_reject(rejected, "snapshot_extra_file_closure", lambda:
                      r5.assert_exact_directory_files(root, {root / "a"}, "closure"))

        for label, relative in (
                ("alternate_fm_core_rtl",
                 "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv"),
                ("alternate_fm_pool_rtl",
                 "rtl_m31/qfit_signed_int8_mul96_pool.sv")):
            fixture = base / label
            fm, vcs, dc = make_cross_binding(r5, fixture)
            (fm / "inputs/hw_root" / relative).write_text(
                "module forged; endmodule\n", encoding="utf-8")
            expect_reject(rejected, label, lambda fm=fm, vcs=vcs, dc=dc:
                          r5.validate_exact_rtl_cross_binding(fm, vcs, dc))

        for label, relative in (
                ("coherently_rehashed_all_core_rtl",
                 "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv"),
                ("coherently_rehashed_all_pool_rtl",
                 "rtl_m31/qfit_signed_int8_mul96_pool.sv")):
            fixture = base / label
            fm, vcs, dc = make_cross_binding(r5, fixture)
            forged = b"module coherent_forgery; endmodule\n"
            for target in (fm / "inputs/hw_root" / relative,
                           vcs / "inputs/hw_root" / relative,
                           dc / "inputs" / relative):
                target.write_bytes(forged)
            replace_manifest_hash(vcs / "input_sha256.txt", relative,
                                  hashlib.sha256(forged).hexdigest())
            expect_reject(rejected, label, lambda fm=fm, vcs=vcs, dc=dc:
                          r5.validate_exact_rtl_cross_binding(fm, vcs, dc))

        forged_vcs_ledger = root / "forged_vcs.sha256"
        forged_vcs_ledger.write_text(
            paths["r4_vcs_snapshot_ledger"].read_text(encoding="utf-8") +
            "{}  forged\n".format("0" * 64), encoding="utf-8")
        expect_reject(rejected, "vcs_ledger_identity_drift", lambda:
                      r5.parse_exact_relative_ledger(
                          forged_vcs_ledger, root, r5.vcs_snapshot_expected(),
                          "VCS", r5.R4_VCS_SNAPSHOT_LEDGER_SHA256))

        clock = root / "clocks.rpt"
        separator = "-" * 80
        clock.write_text(
            "Clock Period Waveform Attrs Sources\n" + separator + "\n"
            "core_clk 3.00 {0 1.5} f {clk_core}\n"
            "forged_clk 3.00 {0 1.5} f {forged}\n" + separator + "\n2\n",
            encoding="utf-8")
        expect_reject(rejected, "rehashed_second_clock",
                      lambda: r5.parse_unique_clock_report(clock))

        coherent_target = root / "coherent_target"
        coherent_target.write_text("same bytes\n", encoding="utf-8")
        coherent_link = root / "coherent_link"
        coherent_link.symlink_to(coherent_target)
        coherent_ledger = root / "coherent.sha256"
        coherent_ledger.write_text(
            "{}  coherent_link\n".format(sha256(coherent_target)),
            encoding="utf-8")
        expect_reject(rejected, "coherent_rehash_symlink_path_replacement",
                      lambda: r5.parse_exact_relative_ledger(
                          coherent_ledger, root, {"coherent_link"}, "coherent"))

    canonical = r5.validate_roots(
        Path(str(WORK) + "/x/../"), Path(str(REPO) + "/./"),
        Path(str(RUNS) + "/x/../"), Path(str(run) + "/./"))
    require(canonical == (WORK, REPO, RUNS, run),
            "normalized root alias changed the canonical target")
    safe_aliases.extend(("work_root_x_dotdot", "repo_root_dot",
                         "runs_root_x_dotdot", "run_root_dot"))
    return {
        "rejected_attack_count": len(rejected),
        "rejected_attacks": sorted(rejected),
        "safe_same_target_canonicalization_count": len(safe_aliases),
        "safe_same_target_canonicalizations": sorted(safe_aliases),
        "accepted_different_target_or_identity_count": 0,
    }


def validate_claim_boundary(receipt):
    require(receipt.get("schema") ==
            "m31_synopsys_receipt_r5_frozen_evidence_hardened_v1",
            "M31-r5 receipt schema drift")
    require(receipt.get("headline_admitted") is False,
            "M31-r5 headline admission drift")
    require(receipt.get("independent_review_required") is True,
            "M31-r5 independent review flag drift")
    require(receipt.get("advances", {}).get("dc_or_formality_rerun") is False,
            "M31-r5 falsely claims a tool rerun")
    require(receipt.get("frozen_dc_sta", {}).get("paper_ppa_ready") is False,
            "M31-r5 falsely admits paper PPA")
    forbidden = receipt.get("claim_boundary", {}).get("forbidden", "")
    for token in ("post-layout", "SAIF/PTPX", "full-network", "DATE headline"):
        require(token in forbidden, "M31-r5 forbidden claim boundary drift")
    permitted = receipt.get("claim_boundary", {}).get("permitted", "")
    for token in ("zero-wire", "ideal-clock", "logic-only", "3.000ns"):
        require(token in permitted, "M31-r5 permitted claim boundary drift")
    return receipt["claim_boundary"]


def build_review():
    r5 = load_r5()
    run, paths, anchors = validate_hashes(r5)
    regression = run_40_tests()
    reanchor = validate_reanchor(r5, run, paths)
    determinism = deterministic_build(r5, run, paths)
    attacks = run_attacks(r5, run, paths)
    receipt = load_json_no_duplicates(R5_RECEIPT)
    claim = validate_claim_boundary(receipt)
    return {
        "schema": "m31_r5_independent_hammer_review_v1",
        "status": "GO_M31_R5_EVIDENCE_HARDENING_LOGIC_ONLY_NO_TOOL_RERUN",
        "date": "2026-08-22",
        "reviewer_role": "INDEPENDENT_R4_HAMMER_DID_NOT_IMPLEMENT_R5",
        "verdict": {
            "score_0_to_100": 96,
            "scoped_go": True,
            "go_scope": "r5 evidence hardening over frozen r4 VCS/DC/Formality evidence",
            "date_ppa_power_system_no_go": True,
            "p0": [],
            "p1": [],
            "p2": [
                "CLI root spellings containing ./ or x/../ canonicalize to the exact pinned realpath instead of being lexically rejected; no alternate target is admitted",
                "0444 receipts and reanchor ledgers remain removable by an owner because their parent directories are writable; identity is hash-anchored and tamper-evident, not OS-immutable",
            ],
        },
        "implementation_anchors": anchors,
        "regression": regression,
        "relative_reanchor": reanchor,
        "determinism": determinism,
        "former_p1_attacks": attacks,
        "r4_tool_evidence": {
            "dc_or_formality_rerun_performed_by_review": False,
            "all_expected_frozen_and_live_hashes_unchanged": True,
            "dc_logic_only": True,
            "formality_strict_equivalence": True,
            "vcs_exact_six_input_snapshot": True,
        },
        "claim_boundary": claim,
        "review_validator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "python36_compatible": True,
            "output_policy": "CREATE_ONLY_OR_EXACT_CHECK",
        },
    }


def write_output(path, result):
    path = Path(os.path.abspath(str(path)))
    require(path == REVIEW_OUTPUT, "independent-review output path drift")
    require(not path.exists() and not path.is_symlink(),
            "refusing to overwrite M31-r5 independent review")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(path), flags, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o444)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--check-existing", type=Path)
    args = parser.parse_args()
    result = build_review()
    if args.output is not None:
        write_output(args.output, result)
        print(args.output)
    else:
        require(Path(args.check_existing).resolve() == REVIEW_OUTPUT.resolve(),
                "independent-review check path drift")
        recorded = load_json_no_duplicates(args.check_existing)
        require(recorded == result,
                "M31-r5 independent review is not an exact live rebuild")
        print("PASS {}".format(sha256(args.check_existing)))


if __name__ == "__main__":
    sys.exit(main())
