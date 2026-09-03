#!/usr/bin/env python3
"""Read-only independent result hammer for the sole M2037 successor run."""
from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
RESULTS = HW / "results"
RESULT = RESULTS / "m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_r1_20260902"
ATTEMPT = RESULTS / ".m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_attempt_consumed"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m2033_m2031_ep34_c1_first64_model_rtl_calibration_one_shot.sh"
TOP = HW / "rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv"
MACRO = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
FOUNDRY = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v")
TB = HW / "tb_m528_dw1rw/tb_m2031_ep34_c1_first64_model_rtl_calibration.sv"
FIXTURE = HW / "tb_m528_dw1rw/fixtures/m2031_ep34_c1_first64_support16.memh"
AUDIT = HW / "system_simulator/scripts/check_m2031_ep34_c1_first64_model_rtl_calibration_source.py"
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PYTHON = Path("/opt/anaconda3/bin/python3.12")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

UPSTREAM = {
    "source_review": HW / "reviews/m2032_m2031_ep34_c1_first64_model_rtl_calibration_source_hammer_r1_20260902",
    "old_runner_review": HW / "reviews/m2034_m2033_ep34_c1_first64_model_rtl_calibration_runner_source_hammer_r1_20260902",
    "failure_review": HW / "reviews/m2035_m2033_ep34_c1_first64_vcs_seal_failure_hammer_r1_20260902",
    "successor_review": HW / "reviews/m2036_m2037_ep34_c1_first64_model_rtl_calibration_successor_runner_source_hammer_r1_20260902",
}

EXPECTED_PASS = (
    "PASS_M2031_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION rows=64 active=64 "
    "input_nnz=565 residual_nnz=192 exact_parent_rows=4 issue=196 "
    "parent_edges=58 dead_elisions=31 macro_reads=54 macro_writes=33 "
    "forwards=4 deadline_holds=6 stalls=14 psum_commits=64 "
    "row_completions=64 numeric_commits=64 rtl_cycle_speedup=false "
    "full_network=false system_speedup=false"
)

EXPECTED_SHA = {
    "runner": "9ecfea0331368385421c2b7bfbf84d00fe9bf6f4d793f8fc07bfa2b25fc047b3",
    "top": "726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1",
    "macro": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "foundry": "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    "tb": "8cac9b384ce6812336d6961bc9ae50ca5a46e636ee8e74d2d49de40c0b4d74f1",
    "fixture": "4601182ca0dbba23d444de7d65cd2d7969159aa8564fd54a516a1934bf8112b3",
    "audit": "c3937a5d069f56cee3bd641eda0b78777acda8c15aae54e8650360e1105c485a",
    "source_review": "f0b6ce291ec25b52815db25c0bc8e76d87162c9b3821fa9d3b7eb3577bfa238a",
    "old_runner_review": "3eb091f8385e73745deea40e82cb4a04711b22f3b91e619692c5d0156b027544",
    "failure_review": "e3b8bffe5b9c0d33d326b5431ba79c9bcacec67527c4f996786cb5dd5f634654",
    "successor_review": "738e79dfd5f9880f2fa9983d895a69a085c460868f276ca8d36280a52d5890b1",
    "release": "c51bb1e520d61bf558410c5725fced81c17488aef6f8406366d6cc39642b9e1a",
    "vcs": "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    "python": "873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "compile_log": "eddbe0c653d511628c3f03a409e53a1f13b15f6201ba2a9ede9530c3dd7e40bf",
    "sim_log": "635b7d6227419d0952d9bf9a3bf08729bea559cfddde7616ee0d55c674b68f6e",
    "removal": "92bd21875017a32fc0216388f429dcbdf065b30346f3ee37d60f604ca92c69da",
    "receipt": "46ee304d62004ffeb1719f7cbb450ff7e44d65461a859358c530a68f4481e162",
    "run_complete": "01aa8874ebdc3e96edd9ea84c502c27947deebd2a7624b90df72857644544e9c",
    "result_manifest": "1f9f43bbadf503e6e874a803490e437f5e522e7d630dbb3b21926766dacef27e",
    "result_outer": "895c143c473169403216eb4426e9d64c2fca29f4c6b8a22e7a068c9a0a9c1dca",
    "attempt_text": "aaadb95ff343f2f2cb92861fa6ed80986fefadbd232e6211f4a3024a84a4dd88",
    "attempt_manifest": "fae4249e780082555d6d988e9bf40842d7e608c7d7a600285311dcf9165aa5d7",
    "attempt_outer": "765e63801be07ccbdb328b890927c4a13f58fb49b224599ef1b45dfa77abb6e0",
    "archive_target": "83632f8b4f001e977ce3ed4b263a672e7834caa02e9910ca48fb0324da64a144",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_members(directory):
    members = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2, "bad manifest grammar")
        expected, name = fields
        name = name.lstrip("* ")
        require(name and name not in members, "duplicate/empty manifest member")
        members[name] = expected
    return members


def verify_exact_double_seal(directory, manifest_sha=None, outer_sha=None):
    require(directory.is_dir() and not directory.is_symlink(), "bad sealed directory")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "missing manifest")
    require(outer.is_file() and not outer.is_symlink(), "missing outer seal")
    if manifest_sha is not None:
        require(sha(manifest) == manifest_sha, "manifest identity drift")
    if outer_sha is not None:
        require(sha(outer) == outer_sha, "outer-seal file identity drift")
    fields = outer.read_text().split()
    require(len(fields) == 2 and fields[1] == "SHA256SUMS", "outer seal grammar drift")
    require(fields[0] == sha(manifest), "outer seal digest drift")
    listed = manifest_members(directory)
    actual = {}
    for path in directory.rglob("*"):
        require(not path.is_symlink(), "symlink in sealed tree: " + str(path))
        if path.is_file() and path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            actual[path.relative_to(directory).as_posix()] = sha(path)
        else:
            require(path.is_file() or path.is_dir(), "non-file/directory member")
    require(set(listed) == set(actual), "manifest topology differs from actual files")
    for name in listed:
        require(listed[name] == actual[name], "sealed member SHA drift: " + name)


def main():
    require(RESULT.is_dir() and not RESULT.is_symlink(), "canonical result missing")
    require(ATTEMPT.is_dir() and not ATTEMPT.is_symlink(), "attempt marker missing")

    prefix = "m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor"
    public = sorted(p.name for p in RESULTS.iterdir() if p.name.startswith(prefix))
    require(public == [RESULT.name], "retry/quarantine/public-result cardinality drift: " + repr(public))
    hidden_prefix = ".m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor"
    hidden = sorted(p.name for p in RESULTS.iterdir() if p.name.startswith(hidden_prefix))
    require(hidden == [ATTEMPT.name], "stage/hidden-attempt cardinality drift: " + repr(hidden))

    verify_exact_double_seal(RESULT, EXPECTED_SHA["result_manifest"], EXPECTED_SHA["result_outer"])
    verify_exact_double_seal(ATTEMPT, EXPECTED_SHA["attempt_manifest"], EXPECTED_SHA["attempt_outer"])
    for directory in UPSTREAM.values():
        verify_exact_double_seal(directory)

    root_entries = sorted(("d" if p.is_dir() else "f", p.name) for p in RESULT.iterdir())
    require(root_entries == sorted([
        ("d", "csrc"), ("d", "simv.daidir"),
        ("f", "RUN_COMPLETE.txt"), ("f", "SHA256SUMS"),
        ("f", "SHA256SUMS.seal.sha256"), ("f", "compile.log"),
        ("f", "compile.rc"), ("f", "generated_symlink_removal.json"),
        ("f", "receipt.json"), ("f", "sim.log"), ("f", "sim.rc"),
        ("f", "simv"), ("f", "ucli.key")]), "root topology drift")
    directories = sorted(p.relative_to(RESULT).as_posix() for p in RESULT.rglob("*") if p.is_dir())
    require(directories == sorted([
        "csrc", "csrc/archive.0", "csrc/diag", "csrc/hsim", "csrc/objs",
        "simv.daidir", "simv.daidir/cc", "simv.daidir/debug_dump",
        "simv.daidir/debug_dump/fsearch"]), "directory topology drift")
    require(len(manifest_members(RESULT)) == 96, "result manifest member count drift")

    for key, path in {
        "runner": RUNNER, "top": TOP, "macro": MACRO, "foundry": FOUNDRY,
        "tb": TB, "fixture": FIXTURE, "audit": AUDIT,
        "source_review": UPSTREAM["source_review"] / "review.json",
        "old_runner_review": UPSTREAM["old_runner_review"] / "review.json",
        "failure_review": UPSTREAM["failure_review"] / "review.json",
        "successor_review": UPSTREAM["successor_review"] / "review.json",
        "release": UPSTREAM["successor_review"] / "launch_release.json",
        "vcs": VCS, "python": PYTHON, "docs359": DOCS359,
        "compile_log": RESULT / "compile.log", "sim_log": RESULT / "sim.log",
        "removal": RESULT / "generated_symlink_removal.json",
        "receipt": RESULT / "receipt.json", "run_complete": RESULT / "RUN_COMPLETE.txt",
        "attempt_text": ATTEMPT / "ATTEMPT_CONSUMED.txt",
    }.items():
        require(path.is_file() and not path.is_symlink(), "bad identity member: " + key)
        require(sha(path) == EXPECTED_SHA[key], "identity SHA drift: " + key)

    require((RESULT / "compile.rc").read_text() == "0\n", "compile rc drift")
    require((RESULT / "sim.rc").read_text() == "0\n", "sim rc drift")
    require((ATTEMPT / "ATTEMPT_CONSUMED.txt").read_text() ==
            "status=M2037_SUCCESSOR_ATTEMPT_CONSUMED\nvcs_compile_runs=1\nsimv_runs=1\nretry=false\n",
            "attempt ledger drift")
    require((RESULT / "RUN_COMPLETE.txt").read_text() ==
            "PASS_M2037_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION_VCS_SUCCESSOR_PENDING_INDEPENDENT_REVIEW\n",
            "run-complete token drift")

    compile_text = (RESULT / "compile.log").read_text(errors="replace")
    sim_text = (RESULT / "sim.log").read_text(errors="replace")
    require(sim_text.splitlines().count(EXPECTED_PASS) == 1, "exact terminal cardinality drift")
    bad = re.compile(r"(^|[^A-Za-z])(Error|Fatal|Assertion.*failed)|\$fatal|global watchdog expired|counter mismatch|numeric mismatch|protocol_error", re.M)
    require(not bad.search(compile_text) and not bad.search(sim_text), "functional error token found")
    require("Compiler version V-2023.12-SP1_Full64; Runtime version V-2023.12-SP1_Full64" in sim_text,
            "VCS version banner drift")

    receipt = json.loads((RESULT / "receipt.json").read_text())
    require(receipt["status"] == "PASS_M2037_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION_VCS_SUCCESSOR_PENDING_INDEPENDENT_REVIEW",
            "receipt status drift")
    expected_identity = {
        "runner_sha256": EXPECTED_SHA["runner"],
        "top_rtl_sha256": EXPECTED_SHA["top"],
        "macro_wrapper_sha256": EXPECTED_SHA["macro"],
        "foundry_model_sha256": EXPECTED_SHA["foundry"],
        "tb_sha256": EXPECTED_SHA["tb"],
        "fixture_sha256": EXPECTED_SHA["fixture"],
        "source_audit_sha256": EXPECTED_SHA["audit"],
        "source_review_sha256": EXPECTED_SHA["source_review"],
        "old_runner_review_sha256": EXPECTED_SHA["old_runner_review"],
        "failure_review_sha256": EXPECTED_SHA["failure_review"],
        "successor_runner_review_sha256": EXPECTED_SHA["successor_review"],
        "launch_release_sha256": EXPECTED_SHA["release"],
        "vcs_sha256": EXPECTED_SHA["vcs"],
        "python_sha256": EXPECTED_SHA["python"],
        "docs359_sha256": EXPECTED_SHA["docs359"],
        "compile_log_sha256": EXPECTED_SHA["compile_log"],
        "sim_log_sha256": EXPECTED_SHA["sim_log"],
        "generated_symlink_removal_sha256": EXPECTED_SHA["removal"],
    }
    require(receipt["identity"] == expected_identity, "receipt identity map drift")
    require(receipt["model_to_rtl_counts"] == {
        "issue_accepts": 196, "parent_edges": 58, "dead_write_elisions": 31,
        "macro_reads": 54, "macro_writes": 33, "forwards": 4,
        "deadline_holds": 6, "issue_stalls": 14, "psum_commits": 64,
        "row_completions": 64, "numeric_commits": 64}, "receipt count drift")
    require(receipt["execution"] == {
        "automatic_retry": False, "macro_model": "foundry UNIT_DELAY functional",
        "simv_runs": 1, "vcs_compile_runs": 1}, "receipt execution drift")
    require(receipt["payload_boundary"] == {
        "masks": "real ep34 sealed-ledger prefix", "psum_prior": "all zero",
        "real_weight_or_real_psum_numeric_calibration": False,
        "signed12_values": "synthetic deterministic function of source index and lane"},
        "receipt payload boundary drift")
    require(receipt["claim_boundary"]["single_real_tile_event_and_synthetic_numeric_calibration"] is True,
            "positive calibration boundary missing")
    require(receipt["claim_boundary"]["functional_vcs"] is True, "functional VCS boundary missing")
    for forbidden in ("cpu_model_1p694510x_upgraded_to_rtl", "rtl_cycle_speedup", "same_area", "timing", "power", "energy", "full_network", "system_speedup", "headline"):
        require(receipt["claim_boundary"][forbidden] is False, "forbidden receipt promotion: " + forbidden)

    removal = json.loads((RESULT / "generated_symlink_removal.json").read_text())
    require(removal == {
        "link_path": "csrc/_2545240_archive_1.so",
        "raw_target": ".//../simv.daidir//_2545240_archive_1.so",
        "remaining_symlinks_after_unlink": 0,
        "resolved_target_path": "simv.daidir/_2545240_archive_1.so",
        "schema": "m2037_expected_vcs_archive_symlink_removal_r1_v1",
        "status": "RECORDED_AND_UNLINKED_EXPECTED_VCS_ARCHIVE_SYMLINK",
        "target_sha256": EXPECTED_SHA["archive_target"],
        "target_size_bytes": 573992}, "generated-symlink removal record drift")
    target = RESULT / removal["resolved_target_path"]
    require(target.is_file() and not target.is_symlink(), "recorded target not regular")
    require(target.stat().st_size == removal["target_size_bytes"], "recorded target size drift")
    require(sha(target) == removal["target_sha256"], "recorded target SHA drift")
    require(not (RESULT / removal["link_path"]).exists() and not (RESULT / removal["link_path"]).is_symlink(),
            "removed link remains")
    require(not any(path.is_symlink() for path in RESULT.rglob("*")), "result contains symlink")

    old_canonical = RESULTS / "m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs_r1_20260902"
    old_attempt = RESULTS / ".m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs_attempt_consumed"
    old_quarantines = list(RESULTS.glob("m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs_r1_20260902.failed_or_incomplete.*.quarantine"))
    require(not old_canonical.exists() and not old_canonical.is_symlink(), "old M2033 canonical result appeared")
    require(old_attempt.is_dir() and not old_attempt.is_symlink(), "old M2033 attempt missing")
    require(len(old_quarantines) == 1 and (old_quarantines[0] / "FAILED_DO_NOT_CITE").is_file(),
            "old M2033 failure boundary drift")
    failure_review = json.loads((UPSTREAM["failure_review"] / "review.json").read_text())
    require(failure_review["failure_classification"]["old_attempt_permanently_do_not_cite"] is True,
            "old-attempt no-cite boundary drift")

    print(json.dumps({
        "status": "PASS_M2038_M2037_EP34_C1_CALIBRATION_VCS_SUCCESSOR_RESULT_HAMMER",
        "score": 100,
        "severity_counts": {"P0": 0, "P1": 0, "P2": 0},
        "unique_attempt_and_result": True,
        "retry_quarantine_stage_count": 0,
        "compile_rc": 0,
        "sim_rc": 0,
        "exact_terminal_count": 1,
        "functional_error_tokens": 0,
        "sealed_manifest_members": 96,
        "result_symlinks": 0,
        "functional_vcs_admitted": True,
        "cpu_model_1p694510x_upgraded_to_rtl": False,
        "eda_launched_by_reviewer": False,
        "license_query_launched_by_reviewer": False,
        "gpu_launched_by_reviewer": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
