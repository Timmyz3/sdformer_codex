#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind source hammer for the M1060 decoder identity repair.

The hammer never runs M1062, never opens/stats/hashes a calls/* payload member,
and never executes decoder cycles.  It independently checks the frozen M699
root/manifest/member-list anchors, adversarially exercises the post-attempt
identity validators, and uses temporary namespaces for failure isolation.
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
DRIVER = HW / "system_simulator/scripts/execute_m1060_decoder_identity_binding_repair.py"
RUNNER = HW / "system_simulator/scripts/run_m1062_m1060_decoder_identity_binding_pilot_one_shot.sh"
CONTRACT = HW / "contracts/m1060_decoder_identity_binding_repair_contract_r1_20260830.json"
TESTS = HW / "system_simulator/tests/test_m1060_decoder_identity_binding_repair.py"
SOURCE_RECEIPT = HW / "reviews/m1060_decoder_identity_binding_repair_source_receipt_r1_20260830"
M1053 = HW / "reviews/m1053_m1052_decoder_stratified_block_reset_pilot_repair_hammer_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1062_m1060_decoder_identity_binding_pilot_attempt_consumed"
RESULT = HW / "results/m1062_m1060_decoder_identity_binding_pilot_r1_20260830"

EXPECTED = {
    "driver": "440d6a12e19ac5561627ae9181d9b6f8ae1be23b1e988c139816a5261c760eb1",
    "runner": "85618d79502d0d2026532da29f0a5475e27a132abc6a243b36ad6f939bd41ac0",
    "contract": "7539ed98a6ebf672757789a973c05abc4af41ddd2e2011cbf3bcbbfeae7384e5",
    "tests": "8f67d17d67739077e83b2d91f80050d7186f58419f249f292567e99e05d6d40f",
    "source_review": "34231fe9dc6c3f1c011529e0e259aefe9910fc684fdd074794b053da882ea2ac",
    "source_manifest": "acbf5f35001397aabc9aef7d2bdb82e34fad53d4d1aa2fa78ea785fa1155fa9b",
    "source_outer": "934dc4c3aa561b6767fb813dd8923f2d9f71ed306aa2682f8e40d3077e175935",
    "m1053_review": "a0c544a0fd081e0589da6a91d9d7c9a694d4d5bb8c7a8e6fca48fbbb327e3e05",
    "m1053_manifest": "18fa9a077ba835afd6bf518fd04fb32aa756375019745e4c051197ce42352cf3",
    "m1053_outer": "3c13b12faaf8956e947191d832b2f75439a8bdd01e421327c288cfc5876f02ea",
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


def rejected(callable_):
    try:
        callable_()
    except (RuntimeError, TypeError, ValueError, OSError, KeyError):
        return True
    return False


def verify_flat(directory: Path, review_sha: str, manifest_sha: str,
                outer_sha: str) -> None:
    review, manifest = directory / "review.json", directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha(review), sha(manifest), sha(outer)) ==
            (review_sha, manifest_sha, outer_sha),
            "sealed authority identity drift: " + directory.name)
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and fields[1] not in listed,
                "malformed/duplicate flat seal")
        target = directory / fields[1]
        require(target.is_file() and not target.is_symlink() and
                sha(target) == fields[0], "sealed member drift: " + fields[1])
        listed.add(fields[1])
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(listed == actual and outer.read_text(encoding="utf-8") ==
            manifest_sha + "  SHA256SUMS\n", "flat seal exact-set/outer drift")


def load_driver():
    require(sha(DRIVER) == EXPECTED["driver"], "driver identity drift")
    spec = importlib.util.spec_from_file_location("m1061_driver_under_hammer", DRIVER)
    require(spec is not None and spec.loader is not None, "cannot load driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def envelope(module):
    return {
        "schema": module.BASE.ENVELOPE_SCHEMA,
        "status": "CYCLE_CI_AT_MOST_5_PERCENT_NO_DERIVED_VALUES",
        "state": "CANDIDATE_AT_MOST_5_PERCENT",
        "bounds": {"candidate_total_cycles_ci95": [1.0, 2.0],
                   "baseline_total_cycles_ci95": [1.0, 2.0]},
        "uncertainty": {"candidate_cycles_relative_halfwidth": 0.04,
            "baseline_cycles_relative_halfwidth": 0.04,
            "maximum_relative_halfwidth": 0.04, "t_critical": 2.365},
        "coverage": {"strata": [{"stratum": name,
            "population_blocks": 8, "sample_blocks": 8,
            "finite_population_fraction": 1.0}
            for name in module.BASE.NONCENSUS]},
        "identity": {"metric": "serial block-reset executable schedule raw cycles"},
        "admission": {"derived_values_emitted": False, "paper_citable": False},
    }


def seal_synthetic(directory: Path) -> None:
    names = sorted(path.name for path in directory.iterdir()
                   if path.is_file() and path.name not in
                   ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(sha(directory / name) + "  " + name + "\n"
                                for name in names), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        sha(manifest) + "  SHA256SUMS\n", encoding="utf-8")


def main():
    for path, key in ((DRIVER, "driver"), (RUNNER, "runner"),
                      (CONTRACT, "contract"), (TESTS, "tests"),
                      (DOC359, "docs359")):
        require(path.is_file() and not path.is_symlink() and
                sha(path) == EXPECTED[key], "identity drift: " + key)
    verify_flat(SOURCE_RECEIPT, EXPECTED["source_review"],
                EXPECTED["source_manifest"], EXPECTED["source_outer"])
    verify_flat(M1053, EXPECTED["m1053_review"], EXPECTED["m1053_manifest"],
                EXPECTED["m1053_outer"])
    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink() and
            not RESULT.exists() and not RESULT.is_symlink(),
            "canonical M1062 namespace is not fresh")

    module = load_driver()
    contract = module.contract_value()
    require(contract["status"] ==
            "IDENTITY_BINDING_REPAIR_SOURCE_ONLY__M1061_HAMMER_REQUIRED" and
            contract["launch_now"] is False and
            all(value is False for value in contract["claim_boundary"].values()),
            "source-only/claim boundary drift")

    # Pre-attempt source validation must never even stat/open/hash calls/*.
    member_access = []
    full_verifier_calls = []
    original_open = builtins.open
    original_path_open, original_path_stat = Path.open, Path.stat
    original_os_stat = os.stat
    original_verify = module.M785.verify_sealed_directory

    def member(path) -> bool:
        return "/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828/calls/" in str(path)

    def watched_open(file, *args, **kwargs):
        if member(file): member_access.append("open:" + str(file)); raise RuntimeError("calls open")
        return original_open(file, *args, **kwargs)

    def watched_path_open(self, *args, **kwargs):
        if member(self): member_access.append("path_open:" + str(self)); raise RuntimeError("calls path open")
        return original_path_open(self, *args, **kwargs)

    def watched_path_stat(self, *args, **kwargs):
        if member(self): member_access.append("path_stat:" + str(self)); raise RuntimeError("calls path stat")
        return original_path_stat(self, *args, **kwargs)

    def watched_os_stat(path, *args, **kwargs):
        if member(path): member_access.append("os_stat:" + str(path)); raise RuntimeError("calls os stat")
        return original_os_stat(path, *args, **kwargs)

    def trip_full(path):
        full_verifier_calls.append(str(path))
        raise RuntimeError("full verifier before attempt")

    builtins.open, Path.open, Path.stat, os.stat = (
        watched_open, watched_path_open, watched_path_stat, watched_os_stat)
    module.M785.verify_sealed_directory = trip_full
    try:
        pre = module.validate_pre_attempt_source(CONTRACT, RUNNER)
    finally:
        builtins.open, Path.open, Path.stat, os.stat = (
            original_open, original_path_open, original_path_stat, original_os_stat)
        module.M785.verify_sealed_directory = original_verify
    require(member_access == [] and full_verifier_calls == [] and
            pre["payload_members_opened"] is False and
            pre["payload_members_statted"] is False and
            pre["payload_members_hashed"] is False,
            "pre-attempt calls/* access detected")

    # Independently verify the root seal, manifest identity, and the two
    # independent path/SHA anchors (manifest record and sealed member list)
    # without touching the selected calls/* files.
    frozen = contract["frozen_payload"]
    payload_root = HW / frozen["directory"]
    sums = payload_root / "SHA256SUMS"
    outer = payload_root / "SHA256SUMS.seal.sha256"
    require(sha(sums) == frozen["m699_root_manifest_sha256"] and
            sha(outer) == frozen["m699_outer_seal_file_sha256"] and
            outer.read_text(encoding="utf-8") ==
            frozen["m699_root_manifest_sha256"] + "  SHA256SUMS\n" and
            sha(payload_root / "manifest.json") == frozen["m699_manifest_sha256"],
            "M699 root/manifest/outer anchor drift")
    sealed_members = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in sealed_members, "M699 duplicate member")
        sealed_members[name] = digest
    manifest = module.M785.strict_json(payload_root / "manifest.json")
    records = module.M785.normalized_population_records(manifest,
                                                        module.M1048.POPULATION_ID)
    member_double_anchors = []
    for layer, expected in zip(module.LAYERS, frozen["selected_records"]):
        actual = module.normalized_selected(module.M1048.select_record(records, layer), layer)
        require(actual == expected and
                sealed_members.get(expected["relative_path"]) == expected["packed_sha256"],
                "selected manifest/sealed-list double anchor drift: " + layer)
        member_double_anchors.append(layer)
    require(member_double_anchors == ["D0", "D2", "D3"], "selected layers drift")

    context = module.synthetic_context()
    canonical_receipt = module.make_payload_receipt(context)
    module.validate_payload_receipt(canonical_receipt, context)
    identity_attacks = []
    identity_escapes = []
    mutations = {
        "all_fake_attempt_root_manifest_packed_sha": lambda x: (
            x["attempt"].update({"attempt_json_sha256": "f" * 64,
                "runner_sha256": "e" * 64, "contract_sha256": "d" * 64,
                "m1061_authority": {"review_sha256": "c" * 64,
                    "manifest_sha256": "b" * 64, "outer_seal_file_sha256": "a" * 64}}),
            x["payload"].update({"m699_manifest_sha256": "9" * 64,
                "m699_root_manifest_sha256": "8" * 64,
                "m699_outer_seal_file_sha256": "7" * 64}),
            [row.update({"packed_sha256": "6" * 64,
                         "payload_member_sha256": "6" * 64})
             for row in x["payload"]["selected_records"]]),
        "nonexistent_selected_path": lambda x: x["payload"]["selected_records"][0].update(
            {"relative_path": "calls/FORGED_DOES_NOT_EXIST.bitpack"}),
        "selected_relabel_and_rehash": lambda x: x["payload"]["selected_records"][1].update(
            {"relative_path": "calls/renamed.bitpack", "packed_sha256": "5" * 64,
             "payload_member_sha256": "5" * 64}),
        "canonical_context_sha_refresh": lambda x: x.update(
            {"canonical_context_sha256": module.canonical_sha({"attacker": x["payload"]})}),
        "payload_members_verified_bool_int": lambda x: x.update(
            {"payload_members_verified": 1}),
        "post_attempt_bool_int": lambda x: x.update({"post_attempt": 1}),
        "paper_citable_bool_int": lambda x: x.update({"paper_citable": 0}),
        "payload_extra_schema": lambda x: x.update({"attacker": False}),
        "payload_nan": lambda x: x["payload"].update({"m699_manifest_sha256": math.nan}),
        "d1_scheduled": lambda x: x.update({"d1_scheduled": True}),
    }
    for name, mutate in mutations.items():
        attacked = copy.deepcopy(canonical_receipt)
        mutate(attacked)
        if rejected(lambda a=attacked: module.validate_payload_receipt(a, context)):
            identity_attacks.append(name)
        else:
            identity_escapes.append(name)

    # Raw record identity is bound independently to selected path, packed SHA,
    # verified member SHA, and layer position.  Refreshing attacker hashes does
    # not turn a relabel into a canonical record.
    raw_stub = {"layers": [{"layer": row["layer"],
        "record_identity": module.expected_raw_record(row),
        "verified_payload_member_sha256": row["payload_member_sha256"]}
        for row in context["payload"]["selected_records"]]}
    module.bind_raw_records(raw_stub, context)
    raw_attacks = []
    raw_escapes = []
    for name, mutate in (
        ("raw_relative_path_relabel", lambda x: x["layers"][0]["record_identity"].update(
            {"relative_path": "calls/relabel.bitpack"})),
        ("raw_packed_sha_rehash", lambda x: x["layers"][1]["record_identity"].update(
            {"packed_sha256": "4" * 64})),
        ("raw_verified_member_rehash", lambda x: x["layers"][2].update(
            {"verified_payload_member_sha256": "f" * 64})),
        ("raw_layer_relabel", lambda x: x["layers"][0].update({"layer": "D2"})),
        ("raw_sample_id_bool_int", lambda x: x["layers"][0]["record_identity"].update(
            {"sample_id": False})),
        ("raw_timestep_bool_int", lambda x: x["layers"][1]["record_identity"].update(
            {"timestep": False})),
    ):
        attacked = copy.deepcopy(raw_stub); mutate(attacked)
        if rejected(lambda a=attacked: module.bind_raw_records(a, context)):
            raw_attacks.append(name)
        else:
            raw_escapes.append(name)

    # Re-run M1049 semantic, finite-number, boolean and exact-schema attacks.
    semantic_aliases = ("candidate_mean_cycles", "candidateMeanCycles",
        "point_speedup", "pointSpeedups", "normalizedCycles",
        "runtimeEstimate", "throughput", "FPS", "averageLatency")
    semantic_rejected = []
    for alias in semantic_aliases:
        attack = copy.deepcopy(envelope(module))
        attack["coverage"]["strata"][0]["nested"] = {"deeper": {alias: 1.0}}
        require(rejected(lambda a=attack: module.BASE.validate_envelope(a)),
                "M1049 semantic alias survived: " + alias)
        semantic_rejected.append(alias)
    shape_rejected = []
    for value in ([1.0], [2.0, 1.0], [False, 2.0], [1.0, math.nan],
                  [1.0, math.inf]):
        attack = copy.deepcopy(envelope(module))
        attack["bounds"]["candidate_total_cycles_ci95"] = value
        require(rejected(lambda a=attack: module.BASE.validate_envelope(a)),
                "CI shape/nonfinite/bool attack survived")
        shape_rejected.append("bounds")
    for value in (False, [0.04], math.nan, math.inf, -0.1):
        attack = copy.deepcopy(envelope(module))
        attack["uncertainty"]["maximum_relative_halfwidth"] = value
        require(rejected(lambda a=attack: module.BASE.validate_envelope(a)),
                "uncertainty shape attack survived")
        shape_rejected.append("uncertainty")
    contract_attacks = []
    for mutate in (
        lambda x: x["d1"].__setitem__("scheduler_allowed", True),
        lambda x: x["d1"].__setitem__("extra", False),
        lambda x: x.__setitem__("candidate_mean_cycles", 1.0),
        lambda x: x["sampling"].__setitem__("nested", {"pointSpeedups": 2.0}),
        lambda x: x["claim_boundary"].__setitem__("paper_citable", 0),
        lambda x: x["frozen_payload"].__setitem__("m699_manifest_sha256", math.nan),
    ):
        attack = copy.deepcopy(contract); mutate(attack)
        require(rejected(lambda a=attack: module.validate_contract(a)),
                "contract D1/schema/nonfinite/bool attack survived")
        contract_attacks.append(True)

    # Assemble and publish both explicitly rederive canonical context and
    # revalidate payload/raw/result.  Exercise a double-sealed relabel attack
    # in a temporary results root; no real payload member is consulted.
    source = DRIVER.read_text(encoding="utf-8")
    assemble_body = source[source.index("def assemble("):source.index("def publish(")]
    publish_body = source[source.index("def publish("):source.index("def quarantine(")]
    for body in (assemble_body, publish_body):
        require("build_canonical_context" in body and
                "load_work_context" in body and
                "validate_raw(raw, context)" in body and
                "validate_result(" in body,
                "assemble/publish canonical cross-binding missing")
    double_seal_rejected = False
    with tempfile.TemporaryDirectory(prefix="m1061_double_seal_") as td:
        old_results = module.RESULTS
        module.RESULTS = Path(td)
        work = module.RESULTS / ("." + module.RESULT_NAME + ".work.attack")
        work.mkdir(mode=0o700)
        attacked_receipt = copy.deepcopy(canonical_receipt)
        attacked_context = copy.deepcopy(context)
        attacked_context["payload"]["selected_records"][0].update(
            {"relative_path": "calls/relabeled.bitpack",
             "packed_sha256": "2" * 64, "payload_member_sha256": "2" * 64})
        attacked_receipt = module.make_payload_receipt(attacked_context)
        (work / "canonical_context.json").write_text(
            json.dumps(attacked_context, sort_keys=True) + "\n", encoding="utf-8")
        (work / "payload_validation.json").write_text(
            json.dumps(attacked_receipt, sort_keys=True) + "\n", encoding="utf-8")
        (work / "raw_windows.json").write_text(json.dumps({"layers": []}) + "\n",
                                                encoding="utf-8")
        (work / "result.json").write_text(json.dumps({"attacker": "rehash"}) + "\n",
                                           encoding="utf-8")
        (work / "RUN_COMPLETE.txt").write_text("attacker refreshed\n", encoding="utf-8")
        seal_synthetic(work)
        old_builder = module.build_canonical_context
        module.build_canonical_context = lambda *args, **kwargs: context
        try:
            double_seal_rejected = rejected(lambda: module.publish(
                work, module.RESULTS / module.RESULT_NAME,
                module.RESULTS / module.ATTEMPT_NAME, RUNNER,
                EXPECTED["contract"], context["attempt"]["m1061_authority"]))
        finally:
            module.build_canonical_context = old_builder
            module.RESULTS = old_results
    require(double_seal_rejected, "relabel+rehash double seal survived publish")

    # Wrong/missing caller pins and direct namespace/runtime bypasses cannot
    # change the canonical namespace.  We intentionally never provide GO pins.
    clean_env = {"PATH": "/usr/bin:/bin"}
    missing = subprocess.run(["/bin/bash", str(RUNNER)], env=clean_env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wrong = subprocess.run(["/bin/bash", str(RUNNER)], env={**clean_env,
        "M1062_EXPECTED_CONTRACT_SHA": "0" * 64,
        "M1062_EXPECTED_M1061_REVIEW_SHA": "0" * 64,
        "M1062_EXPECTED_M1061_MANIFEST_SHA": "0" * 64,
        "M1062_EXPECTED_M1061_OUTER_SHA": "0" * 64},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(missing.returncode != 0 and wrong.returncode != 0 and
            not ATTEMPT.exists() and not RESULT.exists(),
            "missing/wrong caller pins changed canonical state")
    namespace_rejections = 0
    for path, role in ((module.RESULTS / ".wrong-attempt", "attempt"),
                       (module.RESULTS / "wrong-result", "result"),
                       (module.RESULTS / ".wrong-work", "work"),
                       (module.RESULTS / "wrong-quarantine", "quarantine"),
                       (Path("/tmp/m1061-outside"), "work")):
        require(rejected(lambda p=path, r=role: module.safe_path(p, r)),
                "runtime namespace bypass survived")
        namespace_rejections += 1
    require(rejected(lambda: module.run_pilot(ATTEMPT,
        module.RESULTS / ("." + module.RESULT_NAME + ".work.direct"), RUNNER,
        EXPECTED["contract"], {"synthetic": "authority"})),
        "direct run-pilot bypass survived")

    # After a synthetic canonical attempt, a forced full-verifier failure is
    # isolated in quarantine and the attempt remains immutable.
    synthetic_quarantine = False
    with tempfile.TemporaryDirectory(prefix="m1061_postattempt_") as td:
        old_results = module.RESULTS
        module.RESULTS = Path(td)
        authority = {"review_sha256": "7" * 64,
                     "manifest_sha256": "8" * 64,
                     "outer_seal_file_sha256": "9" * 64}
        attempt = module.RESULTS / module.ATTEMPT_NAME
        module.consume_attempt(attempt, RUNNER, EXPECTED["contract"], authority)
        work = module.RESULTS / ("." + module.RESULT_NAME + ".work.synthetic")
        work.mkdir(mode=0o700)
        old_verify = module.M785.verify_sealed_directory
        module.M785.verify_sealed_directory = lambda path: (_ for _ in ()).throw(
            RuntimeError("synthetic post-attempt identity failure"))
        try:
            require(rejected(lambda: module.validate_payload_after_attempt(
                attempt, work, RUNNER, EXPECTED["contract"], authority)),
                "post-attempt identity failure survived")
        finally:
            module.M785.verify_sealed_directory = old_verify
        quarantine = module.RESULTS / (
            module.RESULT_NAME + ".failed_or_incomplete.synthetic")
        module.quarantine(work, quarantine, 99)
        synthetic_quarantine = (attempt.is_dir() and quarantine.is_dir() and
            (quarantine / "FAILURE.json").is_file() and not work.exists())
        module.RESULTS = old_results
    require(synthetic_quarantine, "post-attempt quarantine/attempt retention failed")

    # Resource and state-change ordering is frozen in the actual runner.
    text = RUNNER.read_text(encoding="utf-8")
    ordered = [text.index('m1062_flock}" -n 9'), text.index('/usr/bin/pgrep'),
               text.index('MemAvailable:'), text.index('CommitLimit:'),
               text.index('--consume-attempt'), text.index('/usr/bin/mkdir -m 700'),
               text.index('--validate-payload-after-attempt'),
               text.index('--run-pilot'), text.index('--assemble'),
               text.index('--publish')]
    require(ordered == sorted(ordered) and
            'for m1062_process in dc_shell vcs simv fm_shell pt_shell' in text and
            'm1062_mem}" -ge 16777216' in text and
            'm1062_limit-m1062_used' in text and
            'exec 9>"/tmp/m1062_decoder_identity_binding_pilot.lock"' in text,
            "runner flock/EDA/resource/attempt order drift")
    syntax = subprocess.run(["/bin/bash", "-n", str(RUNNER)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tests = subprocess.run(["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(TESTS)], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    require(syntax.returncode == 0 and tests.returncode == 0,
            "runner syntax or author regression failed")

    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink() and
            not RESULT.exists() and not RESULT.is_symlink() and
            sha(DOC359) == EXPECTED["docs359"],
            "canonical state/docs359 changed")
    require(set(identity_escapes) == {"payload_members_verified_bool_int",
            "post_attempt_bool_int", "paper_citable_bool_int"},
            "unexpected payload bool/int escape set")
    require(set(raw_escapes) == {"raw_sample_id_bool_int",
            "raw_timestep_bool_int"}, "unexpected raw equality escape set: " + repr(raw_escapes))
    output = {
        "status": "FAIL_M1061_M1060_BOOL_INTEGER_EXACT_SCHEMA_ESCAPE__STOP_M1062",
        "verdict": "STOP_M1062__ADDITIVE_EXACT_BOOL_REPAIR_REQUIRED",
        "score": 88,
        "positive": {
            "source_identities_and_double_seals": "PASS",
            "author_regression": "11/11 PASS",
            "pre_attempt_calls_member_access": "0 open / 0 stat / 0 hash",
            "m699_root_manifest_outer": "exact",
            "selected_manifest_plus_sealed_list_double_anchors": member_double_anchors,
            "payload_identity_schema_attacks_rejected": identity_attacks,
            "payload_identity_attacks_escaped": identity_escapes,
            "raw_selected_member_attacks_rejected": raw_attacks,
            "raw_bind_helper_attacks_escaped_but_full_layer_validator_rejects": raw_escapes,
            "m1049_semantic_aliases_rejected": len(semantic_rejected),
            "schema_nonfinite_bool_attacks_rejected": len(shape_rejected),
            "contract_d1_boundary_attacks_rejected": len(contract_attacks),
            "relabel_rehash_double_seal_publish": "REJECTED",
            "wrong_runtime_namespaces_rejected": namespace_rejections,
            "missing_wrong_caller_pins": "REJECTED_WITHOUT_ATTEMPT",
            "postattempt_failure": "QUARANTINED_WITH_ATTEMPT_RETAINED",
            "runner_gate_order": "flock -> EDA -> memory/commit -> attempt -> work -> payload -> cycles -> assemble -> publish",
        },
        "blocking": {"id": "PYTHON_BOOL_INT_EQUALITY_BYPASSES_EXACT_PAYLOAD_SCHEMA",
            "severity": "P1_FAIL_CLOSED",
            "reproduction": "validate_payload_receipt accepted integer 1 for payload_members_verified/post_attempt and integer 0 for paper_citable because dict equality treats bool as int",
            "impact": "The all-fake SHA, nonexistent path, relabel+rehash and double-seal P0 attacks are repaired, but the advertised recursive exact JSON schema is not exact and the same values would survive assemble/publish.",
            "required_repair": "Add explicit type(value) is bool checks for every canonical-context/payload/result boolean before equality; add exact type checks for record integer fields in bind_raw_records or rely only on validate_layer; rerun this hammer."},
        "authorization": {"one_m1062_attempt": False,
            "real_payload_after_attempt_only": True, "eda_gpu_remote": False},
        "claim_boundary": {"paper_citable": False, "decoder_complete": False,
            "table_a_row": False, "system_performance_claim": False,
            "local_performance_claim": False, "continuous_row_cycles": False},
        "execution": {"m1062_executed": False,
            "canonical_attempt_consumed": False,
            "real_payload_members_opened": False,
            "real_window_cycles_executed": False,
            "eda_gpu_remote_used": False, "docs359_sha256": sha(DOC359)},
        "next_gate": "Additive M1064 exact-bool repair and independent rehammer; M1062 remains inert.",
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
