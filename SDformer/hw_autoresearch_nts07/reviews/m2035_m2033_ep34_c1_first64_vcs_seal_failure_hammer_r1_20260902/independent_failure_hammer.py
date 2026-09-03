#!/usr/bin/env python3
"""Receipt-blind, read-only forensic checks for the sole failed M2033 attempt.

This program launches no EDA, simulation, GPU work, or license query and writes
nothing.  It authenticates only the failure classification and the boundary for
a separately reviewed successor.
"""
from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
RESULTS = HW / "results"
Q = RESULTS / "m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs_r1_20260902.failed_or_incomplete.2361455.quarantine"
ATTEMPT = RESULTS / ".m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs_attempt_consumed"
M2034 = HW / "reviews/m2034_m2033_ep34_c1_first64_model_rtl_calibration_runner_source_hammer_r1_20260902"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m2033_m2031_ep34_c1_first64_model_rtl_calibration_one_shot.sh"
CANONICAL = RESULTS / "m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_PASS = (
    "PASS_M2031_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION rows=64 active=64 "
    "input_nnz=565 residual_nnz=192 exact_parent_rows=4 issue=196 "
    "parent_edges=58 dead_elisions=31 macro_reads=54 macro_writes=33 "
    "forwards=4 deadline_holds=6 stalls=14 psum_commits=64 "
    "row_completions=64 numeric_commits=64 rtl_cycle_speedup=false "
    "full_network=false system_speedup=false"
)

EXPECTED = {
    "runner": "7a3f7340955edcdb5eb68e28c1b92a6fbf3f2fe2baeba8037f254978322ea41d",
    "m2034_review": "3eb091f8385e73745deea40e82cb4a04711b22f3b91e619692c5d0156b027544",
    "m2034_release": "d4fed2666d983dcde8cd04c78751c1b4a847d1e757b9a4c2508be62e24f368f2",
    "attempt_text": "c67ed9fd378ebef60d1bc1acadc70fd21925ed14617440fa3a6e53004d9c2973",
    "attempt_manifest": "846fa53a9cf1a09f75832cd8a001e44df0d6262bb91f5d5fef62eb236d55dbda",
    "attempt_outer": "b166bfe203dab6e5bc5b8e444d52b82b4e65146c92e29025d6e4c7134ab82c5a",
    "compile_log": "e3515b220a4f9449b3c6f4cc83a41263702d553d112a33b4113cb74eb9f263b8",
    "sim_log": "f8705e3714a7349dfdddbdf7072d72e4f9a5dfc45e22a25c25780080d269ad57",
    "receipt": "3369cc34a9be7fe0bb0f3f7a208a355a1d9529392b37f4e0e87f1e06485edf21",
    "run_complete": "c924f71874a7db84a3cf846c10d5ef45915c0260876395c23c5e7ec12b7f5d33",
    "failed": "233f67bee9baadec7a2c2a7788bfaa44031451e49ac69b257fc679358637900f",
    "archive_so": "6e63b0e29cf867d67d6eb68fbfd434cbed4b26a6bbf6176d3a20ec22995924c8",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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


def verify_double_seal(directory):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "missing manifest: " + str(directory))
    require(outer.is_file() and not outer.is_symlink(), "missing outer seal: " + str(directory))
    for line in manifest.read_text().splitlines():
        expected, name = line.split(None, 1)
        name = name.lstrip("* ")
        member = directory / name
        require(member.is_file() and not member.is_symlink(), "bad sealed member: " + name)
        require(sha(member) == expected, "member SHA drift: " + name)
    fields = outer.read_text().split()
    require(len(fields) >= 2 and fields[1] == "SHA256SUMS", "outer seal grammar")
    require(fields[0] == sha(manifest), "outer seal SHA drift")


def main():
    require(Q.is_dir() and not Q.is_symlink(), "missing quarantine")
    require(ATTEMPT.is_dir() and not ATTEMPT.is_symlink(), "missing attempt")
    require(not CANONICAL.exists() and not CANONICAL.is_symlink(), "canonical result must remain absent")

    prefix = "m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs_r1_20260902"
    public = sorted(p.name for p in RESULTS.iterdir() if p.name.startswith(prefix))
    require(public == [Q.name], "M2033 public result/quarantine cardinality drift: " + repr(public))
    hidden_prefix = ".m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs"
    hidden = sorted(p.name for p in RESULTS.iterdir() if p.name.startswith(hidden_prefix))
    require(hidden == [ATTEMPT.name], "M2033 hidden namespace cardinality drift: " + repr(hidden))

    verify_double_seal(ATTEMPT)
    verify_double_seal(M2034)
    require(sha(RUNNER) == EXPECTED["runner"], "runner SHA drift")
    require(sha(M2034 / "review.json") == EXPECTED["m2034_review"], "M2034 review SHA drift")
    require(sha(M2034 / "launch_release.json") == EXPECTED["m2034_release"], "M2034 release SHA drift")
    require(sha(DOCS359) == EXPECTED["docs359"], "docs359 SHA drift")
    require(sha(ATTEMPT / "ATTEMPT_CONSUMED.txt") == EXPECTED["attempt_text"], "attempt text SHA drift")
    require(sha(ATTEMPT / "SHA256SUMS") == EXPECTED["attempt_manifest"], "attempt manifest SHA drift")
    require(sha(ATTEMPT / "SHA256SUMS.seal.sha256") == EXPECTED["attempt_outer"], "attempt outer SHA drift")

    review = json.loads((M2034 / "review.json").read_text())
    release = json.loads((M2034 / "launch_release.json").read_text())
    require(review["status"] == "PASS_M2034_M2033_RUNNER_SOURCE_HAMMER", "M2034 status drift")
    require(review["runner_sha256"] == EXPECTED["runner"], "M2034 runner binding drift")
    require(release["status"] == "AUTHORIZED_EXACTLY_ONE_M2033_VCS_COMPILE_AND_SIM", "release status drift")
    require(release["execution_budget"] == {"vcs_compile_runs": 1, "simv_runs": 1, "automatic_retry": False}, "release budget drift")
    require(release["runner_sha256"] == EXPECTED["runner"], "release runner binding drift")
    require(release["review_sha256"] == EXPECTED["m2034_review"], "release review binding drift")

    expected_files = {
        "compile.log": "compile_log",
        "sim.log": "sim_log",
        "receipt.json": "receipt",
        "RUN_COMPLETE.txt": "run_complete",
        "FAILED_DO_NOT_CITE": "failed",
    }
    for name, key in expected_files.items():
        path = Q / name
        require(path.is_file() and not path.is_symlink(), "missing forensic member: " + name)
        require(sha(path) == EXPECTED[key], "forensic member SHA drift: " + name)
    require((Q / "compile.rc").read_text() == "0\n", "compile rc is not zero")
    require((Q / "sim.rc").read_text() == "0\n", "sim rc is not zero")
    require((Q / "FAILED_DO_NOT_CITE").read_text() ==
            "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=1\nretry=false\n",
            "failure marker drift")
    require((Q / "RUN_COMPLETE.txt").read_text() ==
            "PASS_M2033_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION_VCS_PENDING_INDEPENDENT_REVIEW\n",
            "pre-seal run-complete token drift")

    compile_text = (Q / "compile.log").read_text(errors="replace")
    sim_text = (Q / "sim.log").read_text(errors="replace")
    require(sim_text.splitlines().count(EXPECTED_PASS) == 1, "exact PASS cardinality drift")
    bad = re.compile(r"(^|[^A-Za-z])(Error|Fatal|Assertion.*failed)|\$fatal|global watchdog expired|counter mismatch|numeric mismatch|protocol_error", re.M)
    require(not bad.search(compile_text) and not bad.search(sim_text), "functional error token found")
    require("Compiler version V-2023.12-SP1_Full64; Runtime version V-2023.12-SP1_Full64" in sim_text,
            "VCS version banner drift")

    receipt = json.loads((Q / "receipt.json").read_text())
    require(receipt["status"] == "PASS_M2033_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION_VCS_PENDING_INDEPENDENT_REVIEW",
            "pending receipt status drift")
    require(receipt["identity"]["runner_sha256"] == EXPECTED["runner"], "receipt runner SHA drift")
    require(receipt["identity"]["runner_review_sha256"] == EXPECTED["m2034_review"], "receipt review SHA drift")
    require(receipt["identity"]["launch_release_sha256"] == EXPECTED["m2034_release"], "receipt release SHA drift")
    require(receipt["identity"]["compile_log_sha256"] == EXPECTED["compile_log"], "receipt compile-log SHA drift")
    require(receipt["identity"]["sim_log_sha256"] == EXPECTED["sim_log"], "receipt sim-log SHA drift")
    require(receipt["execution"] == {"automatic_retry": False, "macro_model": "foundry UNIT_DELAY functional", "simv_runs": 1, "vcs_compile_runs": 1},
            "receipt execution drift")
    require(receipt["payload_boundary"]["real_weight_or_real_psum_numeric_calibration"] is False,
            "payload boundary drift")
    for forbidden in ("cpu_model_1p694510x_upgraded_to_rtl", "rtl_cycle_speedup", "same_area", "timing", "power", "energy", "full_network", "system_speedup", "headline"):
        require(receipt["claim_boundary"][forbidden] is False, "forbidden receipt promotion: " + forbidden)

    links = sorted(p for p in Q.rglob("*") if p.is_symlink())
    require(len(links) == 1, "symlink cardinality drift: " + repr([str(p) for p in links]))
    link = links[0]
    require(link.relative_to(Q).as_posix() == "csrc/_2362104_archive_1.so", "unexpected symlink path")
    require(os.readlink(str(link)) == ".//../simv.daidir//_2362104_archive_1.so", "unexpected raw symlink target")
    target = Path(os.path.realpath(str(link)))
    require(target == Q / "simv.daidir/_2362104_archive_1.so", "symlink resolves outside expected target")
    require(target.is_file() and not target.is_symlink(), "symlink target not regular")
    require(sha(target) == EXPECTED["archive_so"] and sha(link) == EXPECTED["archive_so"], "archive target SHA drift")
    require(not (Q / "SHA256SUMS").exists() and not (Q / "SHA256SUMS.seal.sha256").exists(),
            "failed quarantine unexpectedly sealed")

    source = RUNNER.read_text()
    receipt_pos = source.index("(stage/'receipt.json').write_text")
    run_complete_pos = source.index("printf 'PASS_M2033_EP34")
    canonical_seal_pos = source.rindex('seal_dir "${STAGE}"')
    require(receipt_pos < run_complete_pos,
            "receipt/run-complete order drift")
    require(run_complete_pos < canonical_seal_pos,
            "run-complete/seal order drift")
    require(source.index('[[ -z "$(find "${directory}" -type l -print -quit)" ]]') < canonical_seal_pos,
            "symlink gate/seal order drift")
    require(source.index('>"${STAGE}/FAILED_DO_NOT_CITE"') < source.index('mv -T -- "${STAGE}" "${FAILED}"'),
            "failure marker/quarantine order drift")

    print(json.dumps({
        "status": "PASS_M2035_M2033_SEAL_FAILURE_FORENSIC_NO_EDA",
        "score": 100,
        "severity_counts": {"P0": 0, "P1": 0, "P2": 0},
        "unique_attempt": True,
        "canonical_result_absent": True,
        "compile_rc": 0,
        "sim_rc": 0,
        "exact_pass_lines": 1,
        "functional_error_tokens": 0,
        "failure_phase": "CANONICAL_RESULT_SEAL",
        "root_cause": "NORMAL_VCS_GENERATED_CSRC_ARCHIVE_SYMLINK_REJECTED_BY_ZERO_SYMLINK_SEAL_POLICY",
        "symlink_count": 1,
        "old_attempt_paper_citable": False,
        "successor_source_authoring_permitted": True,
        "successor_execution_authorized": False,
        "eda_launched_by_reviewer": False,
        "license_query_launched_by_reviewer": False,
        "gpu_launched_by_reviewer": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
