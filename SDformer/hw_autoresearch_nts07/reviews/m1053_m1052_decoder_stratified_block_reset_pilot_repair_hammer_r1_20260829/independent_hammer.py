#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind, source-only adversarial hammer for M1052/M1054.

This hammer does not invoke the M1054 runner on its authorized path, does not
open any M699 call payload, and does not execute a real decoder window.  All
runtime/quarantine checks use synthetic temporary namespaces.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
DRIVER = HW / "system_simulator/scripts/execute_m1052_decoder_stratified_block_reset_pilot_repair.py"
RUNNER = HW / "system_simulator/scripts/run_m1054_m1052_decoder_stratified_block_reset_pilot_one_shot.sh"
CONTRACT = HW / "contracts/m1052_decoder_stratified_block_reset_pilot_repair_contract_r1_20260829.json"
TESTS = HW / "system_simulator/tests/test_m1052_decoder_stratified_block_reset_pilot_repair.py"
SOURCE_RECEIPT = HW / "reviews/m1052_decoder_stratified_block_reset_pilot_repair_source_receipt_r1_20260829"
M1049 = HW / "reviews/m1049_m1048_decoder_stratified_block_reset_pilot_release_hammer_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1054_m1052_decoder_stratified_block_reset_pilot_attempt_consumed"
RESULT = HW / "results/m1054_m1052_decoder_stratified_block_reset_pilot_r1_20260829"

EXPECTED = {
    "driver": "756bf90d52505a68f089dd42296244b94b9c9a50cf013efc0dbc02cd6bb25cec",
    "runner": "582797f28e439b37083f95f8dc5b94c5348a26ab657355964163b90559b94bd9",
    "contract": "c6915033370a236e26959de80bc8ef71efef85a3aabb36fd915015169eed2922",
    "tests": "2e6c9837d51ab95aa69c737e1a3ec73ab7bfb8c29deb80b59d53faa9fdcece23",
    "source_review": "ce28f537613bd03d5f04fe5dfb1766706d533ac2f321595a5484178ad8ca8652",
    "source_manifest": "3f43c5d556c14191c0f3d34d3f6e03b3c5db19699ee2017075debb02e1cccf72",
    "source_outer": "f8a47181ff2702e2b101e8b0f9d2f7a6fdc1772f700bac874ce6bcbc15cb6ba6",
    "m1049_review": "62338edb351b49e7a25b1d81d4b930bc7b7663501a614ae0400a9447cad27670",
    "m1049_manifest": "a4bb8a91b1a0ecffbfc8b2a40ecb05b4b3f203c126c200362ee25528d6317785",
    "m1049_outer": "792b629eada6a06f4e8256168551d01bba0a574cf67cca47061f99171fd4855c",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def verify_flat(directory: Path, review_sha: str, manifest_sha: str,
                outer_sha: str) -> None:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha(review), sha(manifest), sha(outer)) ==
            (review_sha, manifest_sha, outer_sha),
            "sealed identity drift: " + directory.name)
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2, "malformed manifest")
        expected, name = fields
        target = directory / name
        require(name not in listed and target.is_file() and
                not target.is_symlink() and sha(target) == expected,
                "sealed member drift: " + name)
        listed.add(name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(listed == actual, "sealed exact-set drift")
    require(outer.read_text(encoding="utf-8") ==
            manifest_sha + "  SHA256SUMS\n", "outer drift")


def load_driver():
    require(sha(DRIVER) == EXPECTED["driver"], "driver identity drift")
    spec = importlib.util.spec_from_file_location("m1053_driver_under_hammer", DRIVER)
    require(spec is not None and spec.loader is not None, "cannot load driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rejected(callable_):
    try:
        callable_()
    except (RuntimeError, TypeError, ValueError, OSError):
        return True
    return False


def forged_payload(module):
    return {
        "schema": module.PAYLOAD_SCHEMA,
        "status": "PASS_M1054_POSTATTEMPT_FULL_PAYLOAD_IDENTITY",
        "attempt_receipt_sha256": "0" * 64,
        "m699_manifest_sha256": "1" * 64,
        "m699_root_manifest_sha256": "2" * 64,
        "m699_outer_seal_file_sha256": "3" * 64,
        "selected_records": [{
            "layer": layer, "sequence": module.M1048.SEQUENCE,
            "sample_id": 0,
            "module_index": module.M1048.MODULE_BY_LAYER[layer],
            "route": "EXACT_BINARY_BITPACK",
            "relative_path": "calls/FORGED_DOES_NOT_EXIST.npz",
            "packed_sha256": "4" * 64,
        } for layer in module.LAYERS],
        "payload_members_verified": True, "post_attempt": True,
        "d1_scheduled": False, "paper_citable": False,
    }


def envelope(module):
    return {
        "schema": module.ENVELOPE_SCHEMA,
        "status": "CYCLE_CI_AT_MOST_5_PERCENT_NO_DERIVED_VALUES",
        "state": "CANDIDATE_AT_MOST_5_PERCENT",
        "bounds": {"candidate_total_cycles_ci95": [1.0, 2.0],
                   "baseline_total_cycles_ci95": [1.0, 2.0]},
        "uncertainty": {"candidate_cycles_relative_halfwidth": 0.04,
            "baseline_cycles_relative_halfwidth": 0.04,
            "maximum_relative_halfwidth": 0.04, "t_critical": 2.365},
        "coverage": {"strata": [{"stratum": name,
            "population_blocks": 8, "sample_blocks": 8,
            "finite_population_fraction": 1.0} for name in module.NONCENSUS]},
        "identity": {"metric": "serial block-reset executable schedule raw cycles"},
        "admission": {"derived_values_emitted": False, "paper_citable": False},
    }


def seal_temp(directory: Path):
    members = sorted(path for path in directory.iterdir()
                     if path.is_file() and path.name not in
                     ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(sha(path) + "  " + path.name + "\n"
                                for path in members), encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(sha(manifest) + "  SHA256SUMS\n", encoding="utf-8")
    return sha(directory / "review.json"), sha(manifest), sha(outer)


def main():
    for path, key in ((DRIVER, "driver"), (RUNNER, "runner"),
                      (CONTRACT, "contract"), (TESTS, "tests"),
                      (DOC359, "docs359")):
        require(path.is_file() and not path.is_symlink() and
                sha(path) == EXPECTED[key], "identity drift: " + key)
    verify_flat(SOURCE_RECEIPT, EXPECTED["source_review"],
                EXPECTED["source_manifest"], EXPECTED["source_outer"])
    verify_flat(M1049, EXPECTED["m1049_review"], EXPECTED["m1049_manifest"],
                EXPECTED["m1049_outer"])
    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink() and
            not RESULT.exists() and not RESULT.is_symlink(),
            "canonical M1054 namespace already changed")

    module = load_driver()
    contract = module.contract_value()
    require(contract["status"] == "REPAIR_SOURCE_ONLY__M1053_HAMMER_REQUIRED" and
            contract["launch_now"] is False and
            contract["d1"] == {"status": "DIAGNOSTIC_ONLY",
                "generator_allowed": False, "scheduler_allowed": False,
                "numeric_equivalence_admitted": False},
            "contract source-only/D1 boundary drift")

    # Independently prove that pre-attempt validation never invokes the full
    # payload verifier and never opens/stats a calls/* member.
    full_verifier_calls = []
    payload_member_access = []
    original_verify = module.M785.verify_sealed_directory
    original_open = builtins.open
    original_stat = os.stat

    def trip_verify(path):
        full_verifier_calls.append(str(path))
        raise RuntimeError("full payload verifier forbidden before attempt")

    def watched_open(file, *args, **kwargs):
        if "/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828/calls/" in str(file):
            payload_member_access.append("open:" + str(file))
            raise RuntimeError("payload member open before attempt")
        return original_open(file, *args, **kwargs)

    def watched_stat(path, *args, **kwargs):
        if "/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828/calls/" in str(path):
            payload_member_access.append("stat:" + str(path))
            raise RuntimeError("payload member stat before attempt")
        return original_stat(path, *args, **kwargs)

    module.M785.verify_sealed_directory = trip_verify
    builtins.open = watched_open
    os.stat = watched_stat
    try:
        pre = module.validate_pre_attempt_source(CONTRACT, RUNNER)
    finally:
        module.M785.verify_sealed_directory = original_verify
        builtins.open = original_open
        os.stat = original_stat
    require(full_verifier_calls == [] and payload_member_access == [] and
            pre["payload_members_opened"] is False and
            pre["payload_members_statted"] is False and
            pre["payload_members_hashed"] is False,
            "pre-attempt payload access detected")

    # Re-run M1049 semantic/D1 attacks and extend to case/camel/plural plus
    # nonfinite/bool/extra-schema shapes.
    semantic_aliases = ("candidate_mean_cycles", "candidateMeanCycles",
        "point_speedup", "pointSpeedups", "normalizedCycles",
        "runtimeEstimate", "throughput", "FPS", "averageLatency")
    semantic_rejected = []
    for alias in semantic_aliases:
        attack = copy.deepcopy(envelope(module))
        attack["coverage"]["strata"][0]["nested"] = {"deeper": {alias: 1.0}}
        require(rejected(lambda a=attack: module.validate_envelope(a)),
                "semantic alias survived: " + alias)
        semantic_rejected.append(alias)
    shape_attacks = []
    for value in ([1.0], [2.0, 1.0], [False, 2.0], [1.0, math.nan],
                  [1.0, math.inf]):
        attack = copy.deepcopy(envelope(module))
        attack["bounds"]["candidate_total_cycles_ci95"] = value
        require(rejected(lambda a=attack: module.validate_envelope(a)),
                "bounds shape attack survived")
        shape_attacks.append("bounds")
    for value in (False, [0.04], math.nan, math.inf, -0.1):
        attack = copy.deepcopy(envelope(module))
        attack["uncertainty"]["maximum_relative_halfwidth"] = value
        require(rejected(lambda a=attack: module.validate_envelope(a)),
                "uncertainty shape attack survived")
        shape_attacks.append("uncertainty")
    for key, value in (("extra", 1), ("paper_citable", True)):
        attack = copy.deepcopy(envelope(module))
        if key == "extra":
            attack[key] = value
        else:
            attack["admission"][key] = value
        require(rejected(lambda a=attack: module.validate_envelope(a)),
                "extra/boolean attack survived")
        shape_attacks.append(key)

    contract_attacks = []
    for mutate in (
        lambda x: x["d1"].__setitem__("scheduler_allowed", True),
        lambda x: x["d1"].__setitem__("extra", False),
        lambda x: x.__setitem__("candidate_mean_cycles", 1.0),
        lambda x: x["sampling"].__setitem__("nested", {"pointSpeedups": 2.0}),
    ):
        attack = copy.deepcopy(contract); mutate(attack)
        require(rejected(lambda a=attack: module.validate_contract(a)),
                "contract/D1 attack survived")
        contract_attacks.append(True)

    # Review SHA / manifest / outer seal and status are independently pinned.
    authority_attacks = []
    with tempfile.TemporaryDirectory(prefix="m1053_authority_") as td:
        authority_dir = Path(td)
        review_path = authority_dir / "review.json"
        good_review = {"status":
            "PASS_M1053_M1052_REPAIR_HAMMER__GO_ONE_M1054_ATTEMPT",
            "authorization": {"one_m1054_attempt": True,
                "real_payload_after_attempt_only": True,
                "eda_gpu_remote": False}}
        review_path.write_text(json.dumps(good_review, sort_keys=True) + "\n",
                               encoding="utf-8")
        good_pins = seal_temp(authority_dir)
        old_m1053_dir = module.M1053_DIR
        module.M1053_DIR = authority_dir
        try:
            module.validate_m1053(*good_pins)
            for index in range(3):
                bad = list(good_pins); bad[index] = "0" * 64
                require(rejected(lambda p=tuple(bad): module.validate_m1053(*p)),
                        "wrong M1053 pin survived")
                authority_attacks.append("wrong_pin_" + str(index))
            for name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                (authority_dir / name).unlink()
            bad_review = copy.deepcopy(good_review)
            bad_review["status"] = "PASS_BUT_WRONG_STATUS"
            review_path.write_text(json.dumps(bad_review, sort_keys=True) + "\n",
                                   encoding="utf-8")
            bad_pins = seal_temp(authority_dir)
            require(rejected(lambda: module.validate_m1053(*bad_pins)),
                    "wrong M1053 status survived")
            authority_attacks.append("wrong_status")
        finally:
            module.M1053_DIR = old_m1053_dir

    # Blocking reproduction: a wholly forged attempt hash, root identities,
    # nonexistent payload paths and packed hashes pass the sealed-output
    # payload validator.  Thus assemble can accept identity relabeling after a
    # real run when raw/result hashes are correspondingly refreshed.
    forged = forged_payload(module)
    forged_payload_accepted = not rejected(
        lambda: module.validate_payload_receipt(forged))
    require(forged_payload_accepted,
            "expected identity-binding escape was not reproduced")
    require(forged["m699_manifest_sha256"] !=
            contract["source_identity"]["m699_root_seal"]["payload_manifest_sha256"] and
            forged["m699_root_manifest_sha256"] !=
            contract["source_identity"]["m699_root_seal"]["root_manifest_sha256"],
            "forged hashes accidentally canonical")

    # A result summary is exact-derived, so arbitrary deep/extra fields and
    # semantic aliases are rejected even without a real raw replay.
    raw_stub = {"layers": [{"layer": layer,
        "selection_identity_sha256": "1" * 64,
        "block_population_index_sha256": "2" * 64,
        "transaction_assignment_census_sha256": "3" * 64,
        "generated_compressed_transactions": 1,
        "assigned_compressed_transactions": 1,
        "coverage": [], "source_census_cycles": {"candidate": 1, "baseline": 1},
        "cycle_ci_envelope": envelope(module), "windows": [],
        "exact_mismatch_count": 0} for layer in module.LAYERS]}
    canonical_result = module.make_result(raw_stub, "a" * 64, "b" * 64)
    result_attacks = []
    for alias in ("candidate_mean_cycles", "pointSpeedups", "normalized",
                  "derived", "throughput"):
        attack = copy.deepcopy(canonical_result)
        attack["layers"][0]["coverage"] = [{"deep": {alias: 2.0}}]
        require(rejected(lambda a=attack: module.validate_result(
            a, raw_stub, "a" * 64, "b" * 64)),
            "result injection survived: " + alias)
        result_attacks.append(alias)

    # Missing/wrong caller pins fail before canonical state changes.  These
    # subprocesses never have a valid M1053 authority and therefore cannot
    # reach attempt consumption or payload/cycle execution.
    clean_env = {"PATH": "/usr/bin:/bin"}
    missing = subprocess.run(["/bin/bash", str(RUNNER)], env=clean_env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wrong = subprocess.run(["/bin/bash", str(RUNNER)], env={**clean_env,
        "M1054_EXPECTED_CONTRACT_SHA": "0" * 64,
        "M1054_EXPECTED_M1053_REVIEW_SHA": "0" * 64,
        "M1054_EXPECTED_M1053_MANIFEST_SHA": "0" * 64,
        "M1054_EXPECTED_M1053_OUTER_SHA": "0" * 64},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(missing.returncode != 0 and wrong.returncode != 0 and
            not ATTEMPT.exists() and not RESULT.exists(),
            "missing/wrong pin attack changed canonical state")

    # Direct runtime/namespace bypasses are rejected synthetically.
    namespace_rejections = 0
    for path, role in ((module.RESULTS / ".wrong-attempt", "attempt"),
                       (module.RESULTS / "wrong-result", "result"),
                       (module.RESULTS / ".wrong-work", "work"),
                       (module.RESULTS / "wrong-quarantine", "quarantine"),
                       (Path("/tmp/m1053-outside"), "work")):
        require(rejected(lambda p=path, r=role: module.safe_path(p, r)),
                "namespace bypass survived")
        namespace_rejections += 1
    require(rejected(lambda: module.run_pilot(ATTEMPT,
        module.RESULTS / ("." + module.RESULT_NAME + ".work.direct"), {})),
        "direct run-pilot bypass survived")

    # Exercise the intended post-attempt failure isolation with synthetic
    # namespaces only: a forced identity verifier failure occurs after attempt
    # creation, and the work tree moves into quarantine while attempt remains.
    synthetic_quarantine = False
    with tempfile.TemporaryDirectory(prefix="m1053_postattempt_") as td:
        old_results = module.RESULTS
        module.RESULTS = Path(td)
        try:
            attempt = module.RESULTS / module.ATTEMPT_NAME
            authority = {"review_sha256": "5" * 64,
                         "manifest_sha256": "6" * 64,
                         "outer_seal_file_sha256": "7" * 64}
            module.consume_attempt(attempt, RUNNER, EXPECTED["contract"], authority)
            work = module.RESULTS / ("." + module.RESULT_NAME + ".work.synthetic")
            work.mkdir(mode=0o700)
            old_verify = module.M785.verify_sealed_directory
            module.M785.verify_sealed_directory = lambda path: (_ for _ in ()).throw(
                RuntimeError("synthetic post-attempt identity failure"))
            try:
                require(rejected(lambda: module.validate_payload_after_attempt(
                    attempt, work, authority)), "post-attempt identity failure survived")
            finally:
                module.M785.verify_sealed_directory = old_verify
            quarantine = module.RESULTS / (
                module.RESULT_NAME + ".failed_or_incomplete.synthetic")
            module.quarantine(work, quarantine, 99)
            synthetic_quarantine = (attempt.is_dir() and quarantine.is_dir() and
                (quarantine / "FAILURE.json").is_file() and not work.exists())
        finally:
            module.RESULTS = old_results
    require(synthetic_quarantine, "post-attempt synthetic quarantine failed")

    # Static runner order is security-relevant because dynamic resource/EDA
    # attacks cannot be authorized before this independent receipt exists.
    text = RUNNER.read_text(encoding="utf-8")
    ordered = [text.index('m1054_flock}" -n 9'), text.index('/usr/bin/pgrep'),
               text.index('MemAvailable:'), text.index('CommitLimit:'),
               text.index('--consume-attempt'), text.index('/usr/bin/mkdir -m 700'),
               text.index('--validate-payload-after-attempt'),
               text.index('--run-pilot')]
    require(ordered == sorted(ordered), "runner gate order drift")
    require('for m1054_process in dc_shell vcs simv fm_shell pt_shell' in text and
            'm1054_mem}" -ge 16777216' in text and
            'm1054_limit-m1054_used' in text and
            'exec 9>"/tmp/m1054_decoder_stratified_block_reset_pilot.lock"' in text,
            "runner flock/EDA/resource gate absent")
    syntax = subprocess.run(["/bin/bash", "-n", str(RUNNER)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(syntax.returncode == 0, "runner shell syntax invalid")

    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink() and
            not RESULT.exists() and not RESULT.is_symlink() and
            sha(DOC359) == EXPECTED["docs359"],
            "canonical state/docs359 changed")

    output = {
        "status": "FAIL_M1053_M1052_POSTRUN_PAYLOAD_IDENTITY_REBINDING_ESCAPE__STOP_M1054",
        "verdict": "STOP_M1054__ADDITIVE_M1056_IDENTITY_BINDING_REPAIR_REQUIRED",
        "positive": {
            "source_identities_and_seals": "PASS",
            "author_regression": "9/9 PASS (independently rerun outside this script)",
            "pre_attempt_payload_member_access": "0 stat / 0 open / 0 hash",
            "m1049_semantic_aliases_rejected": len(semantic_rejected),
            "schema_type_range_attacks_rejected": len(shape_attacks),
            "contract_d1_attacks_rejected": len(contract_attacks),
            "review_sha_seal_status_attacks_rejected": len(authority_attacks),
            "result_injections_rejected": len(result_attacks),
            "wrong_runtime_namespaces_rejected": namespace_rejections,
            "missing_wrong_caller_pins": "rejected without attempt",
            "synthetic_postattempt_identity_failure": "quarantined; attempt retained",
            "runner_gate_order": "flock -> EDA -> memory/commit -> attempt -> work -> payload -> run",
        },
        "blocking": {
            "id": "POSTRUN_PAYLOAD_IDENTITY_REBINDING_ESCAPE",
            "severity": "P0",
            "reproduction": "validate_payload_receipt accepted wholly forged attempt/root/manifest/packed SHA values and nonexistent selected-record paths",
            "impact": "assemble validates only SHA syntax for payload identity and does not bind the receipt to the frozen M699 identities, canonical attempt receipt, or raw layer record identities; a post-run relabel plus refreshed raw/result SHA can be double-sealed without replay",
            "required_repair": "In assemble, bind payload root identities exactly to the frozen contract, bind attempt_receipt_sha256 to the canonical attempt, bind each selected record to the frozen manifest (path/module/packed SHA), and cross-bind each raw layer record_identity to its selected payload record before sealing. Rehammer identity mutation and quarantine paths.",
        },
        "execution": {"m1054_executed": False,
            "canonical_attempt_consumed": False, "real_payload_members_opened": False,
            "real_window_cycles_executed": False, "eda_gpu_remote_used": False,
            "docs359_sha256": sha(DOC359)},
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
