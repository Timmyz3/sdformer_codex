#!/usr/libexec/platform-python3.6
"""M617 r5 immutable one-shot energy runner; no M612 file is modified."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time


class Failure(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Failure(message)


def lexists(path):
    return os.path.lexists(str(path))


def lexical_absolute(path):
    return Path(os.path.abspath(os.fspath(path)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain_chain(path, directory=False):
    lexical = lexical_absolute(path)
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current = current / part
        require(lexists(current), "missing lexical path: " + str(current))
        mode = os.lstat(str(current)).st_mode
        require(not stat.S_ISLNK(mode), "symlink lexical path: " + str(current))
        final = current == lexical
        require(stat.S_ISDIR(mode) if (not final or directory) else stat.S_ISREG(mode),
                "wrong lexical path type: " + str(current))
    require(Path(os.path.realpath(str(lexical))) == lexical,
            "lexical/real path drift after lstat walk")
    return lexical


def strict_json(path):
    def pairs(items):
        value = {}
        for key, child in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = child
        return value
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=pairs,
                          parse_constant=lambda token: (_ for _ in ()).throw(Failure(token)))
    require(isinstance(value, dict), "top-level JSON is not object")
    def finite(node):
        if isinstance(node, float):
            require(math.isfinite(node), "non-finite JSON")
        elif isinstance(node, dict):
            for child in node.values():
                finite(child)
        elif isinstance(node, list):
            for child in node:
                finite(child)
    finite(value)
    return value


SELF = plain_chain(__file__)
REPO = SELF.parents[3]
HW = REPO / "hw_autoresearch_nts07"
PYTHON = Path("/usr/libexec/platform-python3.6")
PYTHON_SHA = "9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f"

M612_PYTHON = HW / "system_simulator/scripts/execute_m612_m593_parent_scratch_energy_r4.py"
M612_PYTHON_SHA = "82cf5a6d7d33a78246b9c88fa5a4db50be4821b4a30c8ffb198f114a59b76727"
M612_SHELL = HW / "system_simulator/scripts/run_m612_m593_parent_scratch_energy_r4_exact_sha.sh"
M612_SHELL_SHA = "b6082e1492b8d4885addb0343970917b79073e8b9cad1414ffac01ecff55f98f"
ADAPTER_REL = "hw_autoresearch_nts07/system_simulator/scripts/analyze_m612_m597_m593_parent_scratch_generated_macro_energy_r4.py"
ADAPTER_SHA = "65f6f006c62a5e7732eefc62106af14b76eb708567da995a3b45ad9a9d78daba"
UPSTREAM_REL = "hw_autoresearch_nts07/system_simulator/scripts/analyze_m597_m593_m528_parent_scratch_generated_macro_energy_r2.py"
UPSTREAM_SHA = "6896c8a406dc3274926e6c7d958136aca47b9df9afa3522d6c2539a142ea9cf9"
SOURCE_REL = "hw_autoresearch_nts07/contracts/m597_m593_m528_parent_scratch_generated_macro_energy_source_contract_r2_20260828.json"
SOURCE_SHA = "90399b6c932e28f6eac38f3408af0374b23beb369e1fd4e57e3b98d92d28b1bf"
DOCS = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

CONTRACT = HW / "contracts/m617_m616_m615_parent_scratch_energy_r5_execution_contract_r1_20260828.json"
CONTRACT_SHA = "404b789129b51469ebe81f8620e22c0e551bf60425d3bb56deb8b8191ac60509"
CONTRACT_SIDE_SHA = "09146a938b7817be23e50776787e63abe18e1509b0f156b1df0755670c7ecd4e"
RELEASE = HW / "contracts/m614_m612_m593_parent_scratch_energy_true_release_r1_20260828.json"
RELEASE_SHA = "9f465b9a091ded283bdddb2a37dc596b2cbfed83e48b4f0567ba9297819e8fa2"
RELEASE_SIDE_SHA = "e27d90b2cd937d462b34951f12676cb6988a15fe55245899f5ab8a7f12e059f5"
RELEASE_OUTER_SHA = "a474c48cad9650d994de25f6fc9e016ed21df8764ab342f0f7593973511225ee"
M616_DIR = HW / "reviews/m616_m615_m614_m612_m593_parent_scratch_energy_true_launch_hammer_r1_20260828"
M616_REVIEW = M616_DIR / "review.json"
M616_REVIEW_SHA = "94e4735566783d938c95efdc744dcda6f3db9ea6861bac4bf9891391608058b7"
M616_MANIFEST_SHA = "ed00600b902baebbed279d5909d2a7b0072667ad6ecd20c969aca840736ac723"
M616_OUTER_SHA = "6e76fd6476511bbe51975380ed7bea4f422d49c8e5eda131ebf61814616a504a"

CANDIDATE = HW / "contracts/m617_m616_m615_parent_scratch_energy_r5_execution_candidate_r1_20260828.json"
M620_ID = "m620_m617_m616_m615_parent_scratch_energy_r5_runner_static_hammer_r1_20260828"
M620_DIR = HW / ("reviews/" + M620_ID)
M620_REVIEW = M620_DIR / "review.json"
AUTH_ID = "m621_m620_m617_m616_parent_scratch_energy_r5_true_launch_admission_r1_20260828"
AUTH = HW / ("contracts/" + AUTH_ID + ".json")

RESULT = HW / "results/m617_m597_m593_parent_scratch_generated_macro_energy_r5_20260828"
ATTEMPT = HW / "results/m617_m597_m593_parent_scratch_generated_macro_energy_r5_20260828.attempt"
CONSUMED = HW / "results/m617_m597_m593_parent_scratch_generated_macro_energy_r5_20260828.attempt.consumed"
RESULT_STAGING_PREFIX = RESULT.name + ".staging."
RUNTIME_PREFIX = ".m617_energy.runtime."
QRAW_PREFIX = ".m617_energy.failed_raw."
QSTAGE_PREFIX = ".m617_energy.failed_quarantine.staging."
QFINAL_PREFIX = "m617_energy.failed_or_incomplete."


plain_chain(M612_PYTHON)
require(sha(M612_PYTHON) == M612_PYTHON_SHA, "frozen M612 Python SHA drift")
spec = importlib.util.spec_from_file_location("m617_frozen_m612", str(M612_PYTHON))
m612 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m612)
core = m612.core


def verify_file_double_seal(path, expected_file_sha, expected_side_sha, expected_outer_sha=None):
    path = plain_chain(path)
    side = plain_chain(Path(str(path) + ".sha256"))
    outer = plain_chain(Path(str(path) + ".sha256.seal.sha256"))
    require(sha(path) == expected_file_sha, "fixed file SHA drift: " + str(path))
    require(side.read_text(encoding="utf-8").strip().split() == [expected_file_sha, path.name],
            "member sidecar drift: " + str(path))
    require(sha(side) == expected_side_sha, "member sidecar file SHA drift: " + str(path))
    require(outer.read_text(encoding="utf-8").strip().split() == [expected_side_sha, side.name],
            "outer sidecar content drift: " + str(path))
    if expected_outer_sha is not None:
        require(sha(outer) == expected_outer_sha, "outer sidecar file SHA drift: " + str(path))


def verify_fixed_lineage():
    identities = {
        PYTHON: PYTHON_SHA,
        M612_PYTHON: M612_PYTHON_SHA,
        M612_SHELL: M612_SHELL_SHA,
        REPO / ADAPTER_REL: ADAPTER_SHA,
        REPO / UPSTREAM_REL: UPSTREAM_SHA,
        REPO / SOURCE_REL: SOURCE_SHA,
        DOCS: DOCS_SHA,
    }
    for path, expected in identities.items():
        plain_chain(path)
        require(sha(path) == expected, "frozen identity drift: " + str(path))
    verify_file_double_seal(CONTRACT, CONTRACT_SHA, CONTRACT_SIDE_SHA)
    contract = strict_json(CONTRACT)
    require(contract.get("contract_id") ==
            "m617_m616_m615_parent_scratch_energy_r5_execution_contract_r1_20260828" and
            contract.get("status") == "IMMUTABLE_R5_SOURCE_CONTRACT__NOT_AUTHORIZATION__NOT_EXECUTED",
            "r5 source contract predicate drift")
    verify_file_double_seal(RELEASE, RELEASE_SHA, RELEASE_SIDE_SHA, RELEASE_OUTER_SHA)
    release = strict_json(RELEASE)
    require(RELEASE.name == "m614_m612_m593_parent_scratch_energy_true_release_r1_20260828.json" and
            release.get("schema") == "m614_m612_m593_parent_scratch_energy_true_release_v1" and
            release.get("status") == "TRUE_RELEASE_AUTHORED__STILL_NOT_EXECUTED__FRESH_M616_REVIEW_REQUIRED" and
            release.get("authorization", {}).get("max_attempts") == 1 and
            release.get("authorization", {}).get("still_not_executed") is True,
            "M615 true-release predicate drift")
    plain_chain(M616_DIR, directory=True)
    manifest_sha, outer_sha = core.verify_seal(M616_DIR, {"review.json", "review.md"})
    require(manifest_sha == M616_MANIFEST_SHA and outer_sha == M616_OUTER_SHA and
            sha(M616_REVIEW) == M616_REVIEW_SHA, "M616 FAIL evidence seal drift")
    review = strict_json(M616_REVIEW)
    p0_ids = [item.get("id") for item in review.get("findings", {}).get("p0", [])]
    require(review.get("schema") ==
            "m616_m615_m614_m612_m593_parent_scratch_energy_true_launch_hammer_v1" and
            review.get("status") ==
            "FAIL_TRUE_LAUNCH__FRESH_REVIEW_AND_ONE_SHOT_NOT_MECHANICALLY_BOUND__NO_EXECUTION" and
            review.get("p0_count") == 2 and p0_ids == ["M616-P0-01", "M616-P0-02"] and
            review.get("authorization", {}).get("root_unique_formal_execution_allowed") is False,
            "M616 FAIL semantic evidence drift")


def verify_candidate(shell_path, python_runner_path):
    plain_chain(CANDIDATE)
    value = strict_json(CANDIDATE)
    require(set(value) == {"candidate_id", "date", "status", "launch_now", "release",
            "objective", "source_contract", "runner", "frozen_lineage", "one_shot",
            "canonical", "future_review", "future_authorization", "resource_policy",
            "claim_boundary"}, "candidate exact key set drift")
    require(value["candidate_id"] ==
            "m617_m616_m615_parent_scratch_energy_r5_execution_candidate_r1_20260828" and
            value["status"] == "R5_CANDIDATE_ONLY__M620_FRESH_PASS_REQUIRED__NO_EXECUTION" and
            value["launch_now"] is False and value["release"] is False,
            "candidate predicate drift")
    require(value["source_contract"] == {"path": str(CONTRACT.relative_to(REPO)),
            "sha256": CONTRACT_SHA, "member_sidecar_file_sha256": CONTRACT_SIDE_SHA},
            "candidate source-contract binding drift")
    require(value["runner"] == {
            "shell_path": str(Path(shell_path).relative_to(REPO)), "shell_sha256": sha(shell_path),
            "python_path": str(Path(python_runner_path).relative_to(REPO)), "python_sha256": sha(python_runner_path),
            "adapter_path": ADAPTER_REL, "adapter_sha256": ADAPTER_SHA,
            "python_binary": str(PYTHON), "python_binary_sha256": PYTHON_SHA},
            "candidate runner identity drift")
    require(value["future_review"]["full_id"] == M620_ID and
            REPO / value["future_review"]["path"] == M620_REVIEW and
            value["future_authorization"] == {"full_id": AUTH_ID,
                "path": str(AUTH.relative_to(REPO)), "must_bind_m620_exact_seals": True},
            "candidate future gate drift")
    return sha(CANDIDATE)


def blocker_entries(parent):
    exact = {RESULT.name, ATTEMPT.name, CONSUMED.name}
    prefixes = (RESULT_STAGING_PREFIX, "." + RESULT_STAGING_PREFIX,
                RUNTIME_PREFIX, QRAW_PREFIX, QSTAGE_PREFIX, QFINAL_PREFIX)
    return sorted(entry.name for entry in os.scandir(str(parent))
                  if entry.name in exact or entry.name.startswith(prefixes))


def verify_coordinates():
    parent = plain_chain(HW / "results", directory=True)
    for path in (RESULT, ATTEMPT, CONSUMED):
        require(path.parent == parent, "coordinate parent drift")
        require(not lexists(path), "coordinate exists: " + str(path))
    require(not blocker_entries(parent), "r5 canonical/staging/quarantine blocker exists")


def verify_fresh_review(shell_path, python_runner_path, review_binding, candidate_sha):
    require(set(review_binding) == {"full_id", "path", "sha256", "manifest_sha256",
            "outer_seal_file_sha256"} and review_binding["full_id"] == M620_ID and
            REPO / review_binding["path"] == M620_REVIEW, "M620 review binding drift")
    plain_chain(M620_DIR, directory=True)
    manifest_sha, outer_sha = core.verify_seal(M620_DIR, {"review.json", "review.md"})
    require(sha(M620_REVIEW) == review_binding["sha256"] and
            manifest_sha == review_binding["manifest_sha256"] and
            outer_sha == review_binding["outer_seal_file_sha256"],
            "M620 review exact SHA/seal drift")
    review = strict_json(M620_REVIEW)
    require(review.get("schema") ==
            "m620_m617_m616_m615_parent_scratch_energy_r5_runner_static_hammer_v1" and
            review.get("status") == "PASS_M620_M617_R5_RUNNER_STATIC_AND_ONE_SHOT_HAMMER" and
            review.get("score_0_to_100", 0) >= 95 and
            (review.get("p0_count"), review.get("p1_count")) == (0, 0) and
            review.get("authorization", {}).get("r5_true_launch_admission_authoring_allowed") is True and
            review.get("authorization", {}).get("formal_execution_performed") is False,
            "M620 PASS predicate drift")
    reviewed = review.get("reviewed", {})
    require(reviewed.get("r5_shell") == {"path": str(Path(shell_path).relative_to(REPO)),
            "sha256": sha(shell_path)} and
            reviewed.get("r5_python") == {"path": str(Path(python_runner_path).relative_to(REPO)),
            "sha256": sha(python_runner_path)} and
            reviewed.get("candidate") == {"path": str(CANDIDATE.relative_to(REPO)),
            "sha256": candidate_sha} and
            reviewed.get("source_contract") == {"path": str(CONTRACT.relative_to(REPO)),
            "sha256": CONTRACT_SHA} and
            reviewed.get("m615_true_release", {}).get("sha256") == RELEASE_SHA and
            reviewed.get("m616_failed_review", {}).get("sha256") == M616_REVIEW_SHA,
            "M620 reviewed identity drift")
    one = review.get("one_shot", {})
    require(one.get("attempt_consumed_before_analyzer") is True and
            one.get("consumed_survives_success_failure_signal") is True and
            one.get("qfinal_is_prelaunch_blocker") is True and
            one.get("all_coordinates_lexists_fail_closed") is True,
            "M620 one-shot verdict drift")


AUTH_KEYS = {"admission_id", "date", "status", "launch_now", "release", "max_attempts",
             "runner", "source_contract", "frozen_lineage", "fresh_launch_hammer",
             "canonical", "resource_policy", "claim_boundary"}


def verify_authorization(supplied, shell_path, python_runner_path):
    supplied = lexical_absolute(supplied)
    require(supplied == AUTH, "authorization full path drift")
    plain_chain(AUTH)
    side = plain_chain(Path(str(AUTH) + ".sha256"))
    outer = plain_chain(Path(str(AUTH) + ".sha256.seal.sha256"))
    auth_sha = sha(AUTH)
    require(side.read_text(encoding="utf-8").strip().split() == [auth_sha, AUTH.name],
            "authorization member seal drift")
    require(outer.read_text(encoding="utf-8").strip().split() == [sha(side), side.name],
            "authorization outer seal drift")
    value = strict_json(AUTH)
    require(set(value) == AUTH_KEYS and value["admission_id"] == AUTH_ID and
            value["status"] == "TRUE_LAUNCH_ADMISSION__FRESH_M620_PASS__ONE_SHOT_R5" and
            value["launch_now"] is True and value["release"] is True and
            value["max_attempts"] == 1, "authorization predicate drift")
    candidate_sha = verify_candidate(shell_path, python_runner_path)
    require(value["runner"] == {
            "shell_path": str(Path(shell_path).relative_to(REPO)), "shell_sha256": sha(shell_path),
            "python_path": str(Path(python_runner_path).relative_to(REPO)), "python_sha256": sha(python_runner_path),
            "adapter_path": ADAPTER_REL, "adapter_sha256": ADAPTER_SHA},
            "authorization runner drift")
    require(value["source_contract"] == {"path": str(CONTRACT.relative_to(REPO)),
            "sha256": CONTRACT_SHA, "candidate_path": str(CANDIDATE.relative_to(REPO)),
            "candidate_sha256": candidate_sha}, "authorization source/candidate drift")
    require(value["frozen_lineage"] == {
            "m615_true_release_full_id": "m614_m612_m593_parent_scratch_energy_true_release_r1_20260828",
            "m615_true_release_sha256": RELEASE_SHA,
            "m615_true_release_outer_seal_file_sha256": RELEASE_OUTER_SHA,
            "m616_failed_review_full_id":
                "m616_m615_m614_m612_m593_parent_scratch_energy_true_launch_hammer_r1_20260828",
            "m616_failed_review_sha256": M616_REVIEW_SHA,
            "m616_failed_review_manifest_sha256": M616_MANIFEST_SHA,
            "m616_failed_review_outer_seal_file_sha256": M616_OUTER_SHA},
            "authorization frozen lineage drift")
    require(value["canonical"] == {"result": str(RESULT.relative_to(HW)),
            "attempt": str(ATTEMPT.relative_to(HW)), "consumed": str(CONSUMED.relative_to(HW)),
            "qfinal_prefix": "results/" + QFINAL_PREFIX}, "authorization coordinates drift")
    require(value["resource_policy"] == {
            "fresh_root_live_recheck_immediately_before_invocation_required": True,
            "shared_host_admission_not_claimed_by_runner": True}, "resource policy drift")
    require(value["claim_boundary"] == {"component_only": True,
            "per_frozen_sampled_inference": True, "sample_is_camera_frame": False,
            "paper_data": False, "system_energy": False, "system_speedup": False,
            "result_hammer_pending": True}, "authorization claim boundary drift")
    verify_fresh_review(shell_path, python_runner_path,
                        value["fresh_launch_hammer"], candidate_sha)
    return auth_sha, value["fresh_launch_hammer"]


def fsync_directory(path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def consume_unique_attempt(shell_path, python_runner_path, auth_sha, review_binding):
    require(not lexists(ATTEMPT) and not lexists(CONSUMED), "attempt already consumed")
    os.mkdir(str(ATTEMPT), 0o700)
    fsync_directory(ATTEMPT.parent)
    core.write_exclusive(ATTEMPT / "ATTEMPT_CONSUMED.json", json.dumps({
        "schema": "m617_m593_energy_irreversible_attempt_v1",
        "status": "PERMANENT_ATTEMPT_CONSUMED_BEFORE_FORMAL_ANALYZER",
        "authorization_full_id": AUTH_ID,
        "authorization_sha256": auth_sha,
        "fresh_review_full_id": M620_ID,
        "fresh_review_sha256": review_binding["sha256"],
        "m615_true_release_sha256": RELEASE_SHA,
        "m616_failed_review_sha256": M616_REVIEW_SHA,
        "runner_shell_sha256": sha(shell_path),
        "runner_python_sha256": sha(python_runner_path),
        "retry_allowed": False
    }, sort_keys=True, indent=2) + "\n")
    core.seal_tree(ATTEMPT)
    core.verify_seal(ATTEMPT, {"ATTEMPT_CONSUMED.json"})
    core.rename_noreplace(ATTEMPT, CONSUMED)
    fsync_directory(CONSUMED.parent)
    core.verify_seal(CONSUMED, {"ATTEMPT_CONSUMED.json"})
    require(not lexists(ATTEMPT) and lexists(CONSUMED),
            "irreversible consumed publication invariant failed")


def bind_r5_core():
    core.ADAPTER_REL = ADAPTER_REL
    core.ADAPTER_SHA = ADAPTER_SHA
    core.UPSTREAM_REL = UPSTREAM_REL
    core.UPSTREAM_SHA = UPSTREAM_SHA
    core.CONTRACT_REL = SOURCE_REL
    core.CONTRACT_SHA = SOURCE_SHA
    core.RESULT = RESULT
    core.ATTEMPT = ATTEMPT
    core.CONSUMED = CONSUMED
    core.AUTH = AUTH


def adapter_internal_entries(staging):
    prefix = "." + staging.name + ".m606_staging_"
    return [Path(entry.path) for entry in os.scandir(str(RESULT.parent))
            if entry.name.startswith(prefix)]


def quarantine_failure(staging, runtime, shell_path, python_runner_path, error, stage, auth_sha):
    parent = RESULT.parent
    stamp = "%d.%d" % (int(time.time() * 1000000), os.getpid())
    qstage = parent / (QSTAGE_PREFIX + stamp)
    qfinal = parent / (QFINAL_PREFIX + stamp)
    raw_prefix = QRAW_PREFIX + stamp + "."
    require(lexists(CONSUMED), "permanent consumed token missing during failure")
    core.verify_seal(CONSUMED, {"ATTEMPT_CONSUMED.json"})
    require(not lexists(qstage) and not lexists(qfinal), "failure quarantine collision")
    coordinates = [("canonical_result", RESULT), ("runner_staging", staging),
                   ("runtime_staging", runtime)]
    coordinates.extend(("adapter_internal_staging_%d" % index, path)
                       for index, path in enumerate(adapter_internal_entries(staging), 1))
    evidence = []
    for index, (name, coordinate) in enumerate(coordinates):
        if not lexists(coordinate):
            evidence.append({"name": name, "original_path": str(coordinate), "present": False})
            continue
        raw = parent / (raw_prefix + str(index))
        require(not lexists(raw), "raw quarantine collision")
        core.rename_noreplace(coordinate, raw)
        item = m612.snapshot_entry(raw)
        evidence.append({"name": name, "original_path": str(coordinate),
                         "present": True, "filesystem_evidence": item})
        m612.remove_entry_nofollow(raw)
    os.mkdir(str(qstage), 0o700)
    core.write_exclusive(qstage / "filesystem_evidence.json", json.dumps({
        "schema": "m617_arbitrary_filesystem_evidence_v1", "entries": evidence
    }, sort_keys=True, indent=2, ensure_ascii=True) + "\n")
    core.write_exclusive(qstage / "failure_receipt.json", json.dumps({
        "schema": "m617_m593_energy_failed_attempt_quarantine_v1",
        "status": "FAILED_AFTER_PERMANENT_ATTEMPT_CONSUMPTION__RETRY_FORBIDDEN",
        "failure_stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error),
        "authorization_sha256_start": auth_sha,
        "consumed_path": str(CONSUMED),
        "consumed_manifest_sha256": sha(CONSUMED / "SHA256SUMS"),
        "consumed_outer_seal_file_sha256": sha(CONSUMED / "SHA256SUMS.seal.sha256"),
        "runner_shell_sha256": sha(shell_path),
        "runner_python_sha256": sha(python_runner_path),
        "same_authorization_retry_allowed": False
    }, sort_keys=True, indent=2) + "\n")
    core.seal_tree(qstage)
    core.verify_seal(qstage, {"filesystem_evidence.json", "failure_receipt.json"})
    core.rename_noreplace(qstage, qfinal)
    fsync_directory(parent)
    core.verify_seal(qfinal, {"filesystem_evidence.json", "failure_receipt.json"})
    require(lexists(CONSUMED) and lexists(qfinal) and not lexists(qstage),
            "terminal failure one-shot invariant failed")


def remove_runtime_after_copy(runtime):
    plain_chain(runtime, directory=True)
    for name in ("production_stdout.log", "production_stderr.log"):
        path = plain_chain(runtime / name)
        os.unlink(str(path))
    os.rmdir(str(runtime))


def execute(shell_path, python_runner_path, supplied_auth):
    stage = "pre_attempt"
    consumed = False
    auth_sha = None
    review_binding = None
    staging = RESULT.parent / (RESULT_STAGING_PREFIX + str(os.getpid()))
    runtime = RESULT.parent / (RUNTIME_PREFIX + str(os.getpid()))
    def caught(signum, frame):
        raise Failure("signal " + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, caught)
    try:
        verify_fixed_lineage()
        verify_coordinates()
        auth_sha, review_binding = verify_authorization(supplied_auth, shell_path, python_runner_path)
        verify_fixed_lineage()
        verify_coordinates()
        require(verify_authorization(supplied_auth, shell_path, python_runner_path)[0] == auth_sha,
                "authorization changed before attempt consumption")
        stage = "consume_attempt_before_analyzer"
        consume_unique_attempt(shell_path, python_runner_path, auth_sha, review_binding)
        consumed = True
        stage = "formal_analyzer"
        os.mkdir(str(runtime), 0o700)
        bind_r5_core()
        with (runtime / "production_stdout.log").open("x") as out, \
             (runtime / "production_stderr.log").open("x") as err:
            cp = subprocess.run([str(PYTHON), str(REPO / ADAPTER_REL), "--source-contract",
                str(REPO / SOURCE_REL), "--output-dir", str(staging)], stdout=out, stderr=err)
        require(cp.returncode == 0, "formal adapter failed")
        stage = "terminal_verify"
        core.verify_result(staging)
        core.remove_top_seal(staging)
        for name in ("production_stdout.log", "production_stderr.log"):
            source = runtime / name
            with source.open("rb") as handle, (staging / name).open("xb") as target:
                target.write(handle.read())
                target.flush()
                os.fsync(target.fileno())
        receipt = {
            "schema": "m606_m593_energy_terminal_rehash_receipt_v1",
            "status": "PASS_M606_TERMINAL_IDENTITY_AND_OUTPUT_REHASH",
            "runner": {"shell_path": str(Path(shell_path).relative_to(REPO)),
                "shell_sha256": sha(shell_path), "python_path": str(Path(python_runner_path).relative_to(REPO)),
                "python_sha256": sha(python_runner_path)},
            "adapter": {"path": ADAPTER_REL, "sha256": ADAPTER_SHA},
            "upstream_analyzer": {"path": UPSTREAM_REL, "sha256": UPSTREAM_SHA},
            "source_contract": {"path": SOURCE_REL, "sha256": SOURCE_SHA},
            "authorization": {"path": str(AUTH.relative_to(REPO)), "sha256": auth_sha},
            "output_schema": "m597_m593_m528_parent_scratch_generated_macro_energy_result_v2",
            "output_status": "PASS_BOUNDED_GENERATED_MACRO_COMPONENT_MODEL__PENDING_FRESH_INDEPENDENT_RESULT_HAMMER",
            "output_members_preseal": {name: sha(staging / name) for name in
                (core.RESULT_JSON, core.CSV_NAME, core.COMPLETE,
                 "production_stdout.log", "production_stderr.log")},
            "claim": "component-only per-frozen-sampled-inference; pending independent result hammer; not paper data"
        }
        core.write_exclusive(staging / "m606_terminal_rehash_receipt.json",
                             json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        core.seal_tree(staging)
        core.verify_result(staging, True, shell_path, python_runner_path, auth_sha)
        stage = "pre_publish_rehash"
        require(verify_authorization(supplied_auth, shell_path, python_runner_path)[0] == auth_sha,
                "authorization changed prepublish")
        verify_fixed_lineage()
        core.verify_result(staging, True, shell_path, python_runner_path, auth_sha)
        stage = "publish_result_noreplace"
        core.rename_noreplace(staging, RESULT)
        fsync_directory(RESULT.parent)
        stage = "post_publish_rehash"
        core.verify_seal(CONSUMED, {"ATTEMPT_CONSUMED.json"})
        require(verify_authorization(supplied_auth, shell_path, python_runner_path)[0] == auth_sha,
                "authorization changed postpublish")
        verify_fixed_lineage()
        core.verify_result(RESULT, True, shell_path, python_runner_path, auth_sha)
        remove_runtime_after_copy(runtime)
        require(lexists(CONSUMED) and lexists(RESULT) and not lexists(runtime),
                "successful permanent one-shot invariant failed")
    except BaseException as error:
        if consumed:
            quarantine_failure(staging, runtime, shell_path, python_runner_path,
                               error, stage, auth_sha)
        raise


def synthetic_self_test():
    verify_fixed_lineage()
    with tempfile.TemporaryDirectory(prefix="m617_r5_synthetic_") as root_text:
        root = Path(root_text)
        attempt = root / "only.attempt"
        consumed = root / "only.attempt.consumed"
        os.mkdir(str(attempt), 0o700)
        core.write_exclusive(attempt / "ATTEMPT_CONSUMED.json", "{\"synthetic\":true}\n")
        core.seal_tree(attempt)
        core.rename_noreplace(attempt, consumed)
        core.verify_seal(consumed, {"ATTEMPT_CONSUMED.json"})
        require(not lexists(attempt) and lexists(consumed), "synthetic consume failed")
        retry_rejected = False
        try:
            require(not lexists(consumed), "synthetic second attempt blocked")
        except Failure:
            retry_rejected = True
        require(retry_rejected, "synthetic second attempt was accepted")
        collision_source = root / "collision.source"
        collision_target = root / "collision.target"
        collision_source.write_text("source\n", encoding="utf-8")
        collision_target.write_text("target\n", encoding="utf-8")
        noreplace_rejected = False
        try:
            core.rename_noreplace(collision_source, collision_target)
        except OSError:
            noreplace_rejected = True
        require(noreplace_rejected and lexists(collision_source) and lexists(collision_target),
                "synthetic RENAME_NOREPLACE collision was not closed")
        qfinal = root / (QFINAL_PREFIX + "synthetic")
        qfinal.write_text("failure\n", encoding="utf-8")
        require(any(entry.name.startswith(QFINAL_PREFIX) for entry in os.scandir(str(root))),
                "synthetic qfinal blocker missed")
        dangling = root / "dangling"
        os.symlink("missing-target", str(dangling))
        require(lexists(dangling), "dangling symlink not visible under lexists")
        symlink_rejected = False
        try:
            plain_chain(dangling)
        except Failure:
            symlink_rejected = True
        require(symlink_rejected, "synthetic lexical symlink was accepted")
    print("PASS_M617_R5_SYNTHETIC_ONE_SHOT_FAULT_INJECTION__NO_FORMAL_ANALYZER")


def main(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lineage-preflight-only", action="store_true")
    group.add_argument("--synthetic-self-test", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization")
    parser.add_argument("--shell-path", required=True, type=Path)
    args = parser.parse_args(argv)
    shell_path = plain_chain(args.shell_path)
    if args.lineage_preflight_only:
        require(args.authorization is None, "lineage preflight received authorization")
        verify_fixed_lineage()
        verify_coordinates()
        print("PASS_M617_R5_FIXED_LINEAGE_PREFLIGHT__M620_AND_AUTH_PENDING__NO_EXECUTION")
        return 0
    if args.synthetic_self_test:
        require(args.authorization is None, "synthetic test received authorization")
        synthetic_self_test()
        return 0
    require(args.authorization is not None, "--execute requires future M621 authorization")
    execute(shell_path, SELF, args.authorization)
    print("PASS_M617_R5_ATOMIC_COMPONENT_RESULT_PENDING_INDEPENDENT_RESULT_HAMMER " + str(RESULT))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as error:
        print("M617_R5_FAIL_CLOSED: " + str(error), file=sys.stderr)
        sys.exit(70)
