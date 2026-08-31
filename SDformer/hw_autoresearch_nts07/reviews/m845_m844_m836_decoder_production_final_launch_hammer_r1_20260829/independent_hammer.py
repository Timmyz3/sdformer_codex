#!/usr/bin/env python3
"""Receipt-blind M845 final-launch hammer for the frozen M836 decoder chain.

This program is source/release only.  It never invokes the one-shot runner and
never calls the production, VCS, EDA, licence, GPU, or remote paths.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent

DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
RELEASE_SHA = "32ada02b95c3b845d604cf3d902cda105bddd386dd5499e7f73ad8cfb40f445e"
DRIVER_SHA = "4ffb51eddc4991e2d80688cdd9398bb49256128c2e18409c5332f6cbd43f2642"
RUNNER_SHA = "be666f23002f390a28b1e59d32a62513326ad202a9320103b16e8062eb8e7c9b"
CONTRACT_SHA = "4479f537d4f42f65d4f8aa2d4d7d26500ad98c9ea3300b68b2d845c72858528e"
CANDIDATE_SHA = "bcdaa5769a42211ea7206888287713036de3601d18e43cdfa06627439d6693f8"
TESTS_SHA = "2ac8562d06e19268ed887d07ab2be9b80c51ce4cf8cd78551a9ec527b363d1a1"
M839_REVIEW_SHA = "44a994feecfd7933bb4ecf3b2eb307af0d0dd3aedf4858a97b9ab488e7fd17a5"
M839_MANIFEST_SHA = "366cae8c2521cedf50f41b073253838d7da8ee5e4af3b073651541fb5b9c596e"
M839_OUTER_FILE_SHA = "2da70628d4cbcc9b50ccb9f2c2fa77e4ff0c902b27e44daeaf730cbf2740d4a3"
HANDOFF_JSON_SHA = "d7a3bbfa4e44beeb3e6844c17c77954d6275a38a03c6df344d4291c6359bed9f"
HANDOFF_MANIFEST_SHA = "fac56e3412b446faa80e5b3c5ec420abb7516617f4d5dfb6a1063a592b0a9094"
HANDOFF_OUTER_FILE_SHA = "e724b8bb163b583be81a63b7d6c1fba38c9a8daa27c5e4431a86d0426d96697a"

REQUEST_DIR = HW / "reviews/m845_m844_m836_decoder_production_final_launch_hammer_REQUEST_r1_20260829"
HANDOFF_DIR = HW / "reviews/m844_m836_decoder_true_release_author_handoff_r1_20260829"
M839_DIR = HW / "reviews/m839_m836_m785_decoder_publication_boundary_source_fresh_hammer_r1_20260829"
M835_DIR = HW / "reviews/m835_m832_m785_decoder_directory_bound_consumption_source_fresh_hammer_r1_20260829"
RELEASE = HW / "contracts/m836_m785_decoder_physical_residency_production_true_release_r1_20260829.json"
CANDIDATE = HW / "contracts/m836_m785_decoder_publication_boundary_repair_candidate_r1_20260829.json"
CONTRACT = HW / "contracts/m836_m832_decoder_publication_boundary_repair_contract_r1_20260829.json"
DRIVER = HW / "system_simulator/scripts/execute_m836_m832_decoder_publication_boundary_repair.py"
RUNNER = HW / "system_simulator/scripts/run_m836_m785_decoder_physical_residency_one_shot.sh"
TESTS = HW / "system_simulator/tests/test_m836_m832_decoder_publication_boundary_repair.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PY310 = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PY36 = Path("/usr/bin/python3.6")
DATACLASSES = Path("/opt/synopsys/syn/V-2023.12-SP3/icc2/etc/Python/lib/python3.6/site-packages")

EXPECTED_AUTH = {
    "launch_now": True,
    "production_replay": True,
    "max_attempts": 1,
    "run_vcs": False,
    "query_license": False,
    "run_eda": False,
    "run_cpu_training": False,
    "run_gpu": False,
    "run_remote": False,
    "network_or_remote_jobs": 0,
    "raw_result_requires_result_hammer": True,
    "table_a_before_result_hammer": False,
    "release_reuse": False,
}


class HammerFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HammerFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def reject_duplicate(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HammerFailure("duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise HammerFailure("nonfinite JSON value: " + value)

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=reject_duplicate,
                          parse_constant=reject_constant)

    def walk(item: Any) -> None:
        if isinstance(item, float):
            require(math.isfinite(item), "nonfinite JSON float")
        elif isinstance(item, list):
            for child in item:
                walk(child)
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)

    walk(value)
    return value


def regular_exact(path: Path, expected: str, label: str) -> None:
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and not path.is_symlink(),
            label + " is not a regular nonsymlink file")
    require(sha256(path) == expected, label + " SHA drift")


def verify_sidecar(path: Path) -> Dict[str, str]:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    require(sidecar.is_file() and not sidecar.is_symlink(),
            str(sidecar) + " absent/nonregular")
    require(outer.is_file() and not outer.is_symlink(),
            str(outer) + " absent/nonregular")
    expected_sidecar = sha256(path) + "  " + path.name + "\n"
    expected_outer = sha256(sidecar) + "  " + sidecar.name + "\n"
    require(sidecar.read_text(encoding="utf-8") == expected_sidecar,
            str(sidecar) + " content drift")
    require(outer.read_text(encoding="utf-8") == expected_outer,
            str(outer) + " content drift")
    return {
        "file_sha256": sha256(path),
        "manifest_file_sha256": sha256(sidecar),
        "outer_seal_file_sha256": sha256(outer),
    }


def verify_directory_seal(directory: Path) -> Dict[str, Any]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(),
            str(directory) + " absent/non-directory")
    require(manifest.is_file() and not manifest.is_symlink(),
            str(manifest) + " absent/nonregular")
    require(outer.is_file() and not outer.is_symlink(),
            str(outer) + " absent/nonregular")
    listed: List[str] = []
    pattern = re.compile(r"^([0-9a-f]{64})  ([^/]+)$")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = pattern.fullmatch(line)
        require(match is not None, "malformed manifest line in " + str(directory))
        expected, name = match.groups()
        require(name not in listed, "duplicate manifest member " + name)
        member = directory / name
        require(member.is_file() and not member.is_symlink(),
                "manifest member absent/nonregular: " + str(member))
        require(sha256(member) == expected, "manifest member SHA drift: " + name)
        listed.append(name)
    actual = sorted(p.name for p in directory.iterdir()
                    if p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    require(sorted(listed) == actual, "sealed directory population drift: " + str(directory))
    require(outer.read_text(encoding="utf-8") ==
            sha256(manifest) + "  SHA256SUMS\n",
            "outer seal content drift: " + str(directory))
    return {
        "manifest_sha256": sha256(manifest),
        "outer_seal_file_sha256": sha256(outer),
        "members": len(listed),
    }


def exact_typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (set(left) == set(right) and
                all(exact_typed_equal(left[k], right[k]) for k in left))
    if isinstance(left, list):
        return (len(left) == len(right) and
                all(exact_typed_equal(a, b) for a, b in zip(left, right)))
    return left == right


def authorization_negative_matrix() -> Dict[str, str]:
    require(len(EXPECTED_AUTH) == 13, "authorization key count drift")
    require(exact_typed_equal(copy.deepcopy(EXPECTED_AUTH), EXPECTED_AUTH),
            "positive authorization rejected")
    attacks: Dict[str, Any] = {}
    missing = copy.deepcopy(EXPECTED_AUTH)
    missing.pop("release_reuse")
    attacks["missing_key"] = missing
    extra = copy.deepcopy(EXPECTED_AUTH)
    extra["extra"] = False
    attacks["extra_key"] = extra
    confused_bool = copy.deepcopy(EXPECTED_AUTH)
    confused_bool["launch_now"] = 1
    attacks["bool_as_int"] = confused_bool
    confused_int = copy.deepcopy(EXPECTED_AUTH)
    confused_int["max_attempts"] = True
    attacks["int_as_bool"] = confused_int
    wrong = copy.deepcopy(EXPECTED_AUTH)
    wrong["run_vcs"] = True
    attacks["wrong_value"] = wrong
    for name, value in attacks.items():
        require(not exact_typed_equal(value, EXPECTED_AUTH),
                "authorization attack accepted: " + name)
    with tempfile.TemporaryDirectory(prefix="m845_json_negative_") as temp:
        root = Path(temp)
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"launch_now":true,"launch_now":false}\n',
                             encoding="utf-8")
        nonfinite = root / "nonfinite.json"
        nonfinite.write_text('{"max_attempts":NaN}\n', encoding="utf-8")
        for name, path in (("duplicate_key", duplicate),
                           ("nonfinite", nonfinite)):
            rejected = False
            try:
                strict_json(path)
            except HammerFailure:
                rejected = True
            require(rejected, "strict JSON attack accepted: " + name)
            attacks[name] = "REJECTED"
    return {name: "REJECTED" for name in attacks}


def formal_population() -> List[str]:
    results = HW / "results"
    names: List[str] = []
    exact_prefixes = (
        ".m836_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed",
        "m836_m785_h67_decoder_physical_residency_cycles_r1_20260829",
    )
    for entry in results.iterdir():
        if any(entry.name == prefix or entry.name.startswith(prefix + ".")
               for prefix in exact_prefixes):
            names.append(entry.name)
        elif ("m836" in entry.name and
              ("driver_stdout" in entry.name or "driver_stderr" in entry.name or
               ".stage." in entry.name)):
            names.append(entry.name)
    return sorted(set(names))


def run_checked(command: List[str], cwd: Path, env: Mapping[str, str]) -> str:
    completed = subprocess.run(command, cwd=str(cwd), env=dict(env),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="strict", check=False)
    require(completed.returncode == 0,
            "command failed rc={} command={}\n{}".format(
                completed.returncode, command, completed.stdout))
    return completed.stdout


def clean_env(py36: bool = False) -> Dict[str, str]:
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    if py36:
        env["PYTHONPATH"] = str(DATACLASSES)
    return env


def actual_release_preflight() -> Dict[str, Any]:
    env = clean_env()
    env["M836_EXPECTED_RELEASE_SHA256"] = RELEASE_SHA
    output = run_checked([
        str(PY310), str(DRIVER), "--validate-release-preflight",
        "--candidate", str(CANDIDATE), "--release", str(RELEASE),
    ], REPO, env)
    value = json.loads(output)
    require(value == {
        "candidate_sha256": CANDIDATE_SHA,
        "production_cycles": None,
        "release_sha256": RELEASE_SHA,
        "status": "PASS_M836_TRUE_RELEASE_PREFLIGHT__UNCONSUMED",
    }, "actual release preflight drift")
    return value


def make_mirror(root: Path) -> Path:
    mirror_repo = root / "SDformer"
    mirror_hw = mirror_repo / "hw_autoresearch_nts07"
    mirror_hw.mkdir(parents=True)
    for name in ("system_simulator", "contracts", "reviews", "docs"):
        subprocess.run(["/bin/cp", "-a", "--reflink=auto",
                        str(HW / name), str(mirror_hw / name)], check=True)
    outgoing = mirror_hw / "system_handoff/outgoing"
    outgoing.mkdir(parents=True)
    for name in (
        "m686r6_h67_ep35_layer_static_decoder_payload_s10_r1_20260828",
        "m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828",
    ):
        subprocess.run(["/bin/cp", "-a", "--reflink=auto",
                        str(HW / "system_handoff/outgoing" / name),
                        str(outgoing / name)], check=True)
    (mirror_hw / "results").mkdir()
    old_attempt = HW / "results/.m798_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed"
    require(old_attempt.is_dir() and not old_attempt.is_symlink(),
            "frozen M798 parent attempt absent")
    subprocess.run(["/bin/cp", "-a", "--reflink=auto", str(old_attempt),
                    str(mirror_hw / "results" / old_attempt.name)], check=True)
    # Only the future M836 release is omitted.  Frozen M798/M819 parent
    # releases remain because the delegated source validators bind them.
    mirror_release = mirror_hw / RELEASE.relative_to(HW)
    for suffix in ("", ".sha256", ".sha256.seal.sha256"):
        target = Path(str(mirror_release) + suffix)
        require(target.is_file() and not target.is_symlink(),
                "mirror release member absent before omission")
        target.unlink()
    require(not mirror_release.exists(), "future M836 release remained in mirror")
    return mirror_hw


def mirror_regression() -> Dict[str, Any]:
    suites = [
        ("m836", "test_m836_m832_decoder_publication_boundary_repair.py", 12),
        ("m832", "test_m832_m828_decoder_directory_bound_consumption.py", 12),
        ("m828", "test_m828_m819_decoder_failure_prefix_guard.py", 12),
        ("m809", "test_m809_m785_decoder_production_recovery.py", 9),
        ("m815", "test_m815_m809_decoder_runner_recovery.py", 10),
    ]
    compile_rel = [
        "system_simulator/scripts/execute_m836_m832_decoder_publication_boundary_repair.py",
        "system_simulator/scripts/execute_m832_m828_decoder_directory_bound_consumption.py",
        "system_simulator/scripts/execute_m828_m819_decoder_failure_prefix_guard.py",
        "system_simulator/scripts/execute_m819_m809_decoder_production_delegation_compat.py",
        "system_simulator/scripts/execute_m809_m785_decoder_physical_residency_production.py",
    ] + ["system_simulator/tests/" + name for _, name, _ in suites]
    outcome: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="m845_m836_hermetic_") as temp:
        mirror_hw = make_mirror(Path(temp))
        mirror_repo = mirror_hw.parent
        mirror_driver = mirror_hw / DRIVER.relative_to(HW)
        mirror_candidate = mirror_hw / CANDIDATE.relative_to(HW)
        require(not (mirror_hw / RELEASE.relative_to(HW)).exists(),
                "future release present in source-only mirror")
        for label, python, py36 in (("python310", PY310, False),
                                    ("python36", PY36, True)):
            env = clean_env(py36)
            compile_output = run_checked(
                [str(python), "-m", "py_compile"] +
                [str(mirror_hw / rel) for rel in compile_rel],
                mirror_repo, env)
            require(compile_output == "", label + " compile emitted output")
            suite_results: Dict[str, str] = {}
            for suite_label, filename, count in suites:
                output = run_checked([
                    str(python), "-m", "unittest", "-v",
                    str(mirror_hw / "system_simulator/tests" / filename),
                ], mirror_repo, env)
                require("Ran {} tests".format(count) in output and
                        re.search(r"\nOK\s*$", output) is not None,
                        label + " " + suite_label + " count/status drift\n" + output)
                suite_results[suite_label] = "{}_PASS_0_FAIL".format(count)
            self_output = run_checked([str(python), str(mirror_driver), "--self-test"],
                                      mirror_repo, env)
            self_value = json.loads(self_output)
            require(self_value["status"] ==
                    "PASS_M836_PUBLICATION_BOUNDARY_REPAIR_SYNTHETIC_SELF_TEST" and
                    self_value["scheduled_rows"] == 0 and
                    self_value["formal_attempt_created"] is False and
                    self_value["production_cycles"] is None,
                    label + " self-test drift")
            require(self_value["prepublish_content_change"]["rejected"] is True and
                    self_value["after_final_rebind_swap"]["rejected"] is True and
                    self_value["after_final_rebind_swap"]["self_publication_rolled_back"] is True and
                    self_value["after_final_rebind_swap"]["replacement_unchanged"] is True and
                    self_value["postpublish_swap"]["rejected"] is True and
                    self_value["postpublish_swap"]["self_publication_rolled_back"] is True and
                    self_value["postpublish_swap"]["replacement_unchanged"] is True,
                    label + " publication race evidence drift")
            candidate_output = run_checked([
                str(python), str(mirror_driver), "--validate-candidate",
                "--candidate", str(mirror_candidate),
            ], mirror_repo, env)
            candidate_value = json.loads(candidate_output)
            require(candidate_value["status"] ==
                    "PASS_M836_PUBLICATION_BOUNDARY_REPAIR_SOURCE_CANDIDATE__NO_PRODUCTION_RUN" and
                    candidate_value["candidate_sha256"] == CANDIDATE_SHA and
                    candidate_value["production_cycles"] is None,
                    label + " source candidate validation drift")
            formal = [p.name for p in (mirror_hw / "results").iterdir()
                      if p.name.startswith(".m836_") or p.name.startswith("m836_")]
            require(formal == [], label + " created formal M836 artifacts in mirror")
            outcome[label] = {
                "compile": "PASS_10_SOURCES",
                "unit_tests": suite_results,
                "self_test": self_value,
                "candidate_validation": candidate_value["status"],
                "future_release_absent": True,
                "formal_m836_population": 0,
            }
    return outcome


def runner_order() -> Dict[str, Any]:
    text = RUNNER.read_text(encoding="utf-8")
    release = text.index("--validate-release-preflight")
    resource = text.index("m836_free_kib=")
    consume = text.index("--guard-and-consume-attempt")
    started = text.index("m836_started=1", text.index("m836_started=0") + 1)
    production = text.index("--run-production")
    require(release < resource < consume < started < production,
            "runner gate/consume/start/production order drift")
    require(text.count("--guard-and-consume-attempt") == 1,
            "consume helper count drift")
    return {
        "release_preflight_before_resource_gate": True,
        "resource_gate_before_sole_consume": True,
        "sole_consume_before_started_latch": True,
        "started_latch_before_production": True,
        "consume_helper_count": 1,
    }


def main() -> int:
    before_population = formal_population()
    require(before_population == [], "preexisting formal M836 population: " + repr(before_population))
    frozen_before = {
        "docs359": sha256(DOCS359), "release": sha256(RELEASE),
        "driver": sha256(DRIVER), "runner": sha256(RUNNER),
        "contract": sha256(CONTRACT), "candidate": sha256(CANDIDATE),
        "tests": sha256(TESTS),
    }
    require(frozen_before == {
        "docs359": DOCS359_SHA, "release": RELEASE_SHA,
        "driver": DRIVER_SHA, "runner": RUNNER_SHA,
        "contract": CONTRACT_SHA, "candidate": CANDIDATE_SHA,
        "tests": TESTS_SHA,
    }, "frozen primary identity drift")

    request_seal = verify_directory_seal(REQUEST_DIR)
    handoff_seal = verify_directory_seal(HANDOFF_DIR)
    m839_seal = verify_directory_seal(M839_DIR)
    m835_seal = verify_directory_seal(M835_DIR)
    require(handoff_seal["manifest_sha256"] == HANDOFF_MANIFEST_SHA and
            handoff_seal["outer_seal_file_sha256"] == HANDOFF_OUTER_FILE_SHA,
            "M844 handoff pinned seal drift")
    require(m839_seal["manifest_sha256"] == M839_MANIFEST_SHA and
            m839_seal["outer_seal_file_sha256"] == M839_OUTER_FILE_SHA,
            "M839 pinned seal drift")
    regular_exact(HANDOFF_DIR / "handoff.json", HANDOFF_JSON_SHA, "M844 handoff")
    regular_exact(M839_DIR / "review.json", M839_REVIEW_SHA, "M839 review")

    request = strict_json(REQUEST_DIR / "request.json")
    handoff = strict_json(HANDOFF_DIR / "handoff.json")
    release = strict_json(RELEASE)
    candidate = strict_json(CANDIDATE)
    contract = strict_json(CONTRACT)
    m839 = strict_json(M839_DIR / "review.json")
    for path in (RELEASE, DRIVER, RUNNER, CONTRACT, CANDIDATE, TESTS):
        verify_sidecar(path)
    release_sidecar = verify_sidecar(RELEASE)
    require(release_sidecar["manifest_file_sha256"] ==
            "6605c3db35e6fd06c624d35a094d2c027fda362ef1d30caecfeb2d90d05c4d9a" and
            release_sidecar["outer_seal_file_sha256"] ==
            "fb21ce8b551d995f4d963c3df0ec262569a6b0637ef63619a61014ba5e09bab5",
            "release double-seal pin drift")

    require(request["required_final_authorization"]["authorization"] == EXPECTED_AUTH,
            "request authorization values drift")
    require(request["required_final_authorization"]["key_count"] == 13 and
            request["required_final_authorization"]["comparison"] ==
            "EXACT_KEY_SET_VALUE_AND_PYTHON_TYPE_EQUALITY",
            "request authorization comparison drift")
    require(handoff["status"] ==
            "PASS_AUTHOR_M836_ONE_WAY_TRUE_RELEASE__PENDING_FRESH_M845_FINAL_HAMMER__NO_LAUNCH" and
            handoff["release"]["sha256"] == RELEASE_SHA and
            handoff["formal_identity"]["current_population_count"] == 0,
            "M844 handoff authority/state drift")
    require(m839["status"] ==
            "PASS100_M836_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY" and
            m839["score"] == 100 and
            m839["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0} and
            m839["true_release_authorized"] is True and
            m839["production_launch_authorized"] is False,
            "M839 authority drift")
    require(release["schema"] == "m836_m785_decoder_production_true_release_v1" and
            release["status"] ==
            "TRUE_RELEASE_AFTER_FRESH_M836_SOURCE_HAMMER__EXACTLY_ONE_PRODUCTION_REPLAY" and
            release["launch_now"] is True and release["release"] is True and
            type(release["max_attempts"]) is int and release["max_attempts"] == 1,
            "true release authority drift")
    require(release["candidate_binding"] == {
        "path": "contracts/m836_m785_decoder_publication_boundary_repair_candidate_r1_20260829.json",
        "sha256": CANDIDATE_SHA,
    }, "release candidate binding drift")
    require(release["source_identity"] == candidate["source_identity"] and
            release["canonical"] == candidate["canonical"] and
            release["publication_boundary_repair"] == candidate["publication_boundary_repair"],
            "release/candidate identity drift")
    for name, entry in candidate["source_identity"].items():
        regular_exact(HW / entry["path"], entry["sha256"], "source_identity." + name)

    runtime = release["runtime_semantics"]
    require(runtime == {
        "populations": "M686_40_AND_M699_120_SEPARATE",
        "configs": ["A1_OSG", "EQUAL_SERVICE_K1X8", "TYPED_SIGNED_K8"],
        "schedule": "RECORD_TIMESTEP_SEQUENTIAL_NO_CROSS_RECORD_OR_POPULATION_OVERLAP",
        "resource": "96_LANES_245760B_ACC24_3NS_192B_PER_CYCLE",
        "headline_ratio": "TYPED_SIGNED_K8_VS_EQUAL_SERVICE_K1X8_ONLY",
        "headline_excludes_module_indices": [1],
        "all_module_total_cycles_retained": True,
        "d1": "COMMON_CHARGED_DIAGNOSTIC_NONHEADLINE",
        "delegated_schedule_body": "FROZEN_M832_M828_M819_M809_EXACT_SHA",
        "attempt_status": "CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY",
    }, "frozen runtime semantics drift")
    require(candidate["production_semantics"]["records"] == "40_PLUS_120" and
            candidate["production_semantics"]["timesteps"] == 10 and
            candidate["common_resource"] == {
                "lanes": 96,
                "onchip_sram_bytes_macro_rounded": 245760,
                "accumulator_bits": 24,
                "clock_ns": 3.0,
                "external_bytes_per_cycle": 192,
                "weight_bytes": 13824,
                "psum_bytes": 221184,
                "descriptor_control_bytes": 8192,
                "reserved_unallocated_bytes": 2560,
                "reserved_borrow_allowed": False,
                "resource_manifest_sha256": "a7400bddb174a00875298cd9bd8d2692e636727ff27b22ae580803383fdea0f3",
            }, "frozen records/resource drift")
    require(contract["claim_boundary"]["production_cycles"] is False and
            contract["claim_boundary"]["production_speedup"] is False,
            "source contract claim boundary drift")

    auth_negatives = authorization_negative_matrix()
    order = runner_order()
    preflight_before = actual_release_preflight()
    mirror = mirror_regression()
    preflight_after = actual_release_preflight()

    after_population = formal_population()
    frozen_after = {
        "docs359": sha256(DOCS359), "release": sha256(RELEASE),
        "driver": sha256(DRIVER), "runner": sha256(RUNNER),
        "contract": sha256(CONTRACT), "candidate": sha256(CANDIDATE),
        "tests": sha256(TESTS),
    }
    require(after_population == [], "hammer created formal M836 population")
    require(frozen_after == frozen_before, "hammer modified frozen sources")
    require(stat.S_IMODE(RUNNER.lstat().st_mode) == 0o664,
            "runner mode drift")

    evidence = {
        "schema": "m845_m844_m836_decoder_production_final_launch_independent_hammer_evidence_v1",
        "date": "2026-08-29",
        "status": "PASS_M845_INDEPENDENT_FINAL_LAUNCH_HAMMER_EVIDENCE",
        "receipt_blind": True,
        "source_and_release_only": True,
        "request_seal": request_seal,
        "handoff_seal": handoff_seal,
        "m839_seal": m839_seal,
        "m835_seal": m835_seal,
        "release_sidecar": release_sidecar,
        "source_identity_members_recomputed": len(candidate["source_identity"]),
        "authorization_positive": "ACCEPTED_EXACT_13_KEY_TYPED_MAP",
        "authorization_negatives": auth_negatives,
        "actual_release_preflight_before": preflight_before,
        "actual_release_preflight_after": preflight_after,
        "mirror_regression": mirror,
        "publication_boundary_attacks": {
            "prepublish_same_inode_content_mutation": "REJECTED_EXACT_RECORDED_STAGE_ROLLED_BACK",
            "results_replace_after_final_prepublication_rebind": "REJECTED_SELF_PUBLICATION_ROLLED_BACK_REPLACEMENT_UNCHANGED",
            "results_replace_after_publication": "REJECTED_SELF_PUBLICATION_ROLLED_BACK_REPLACEMENT_UNCHANGED",
            "stage_fd_held_through_publication_and_final_check": True,
            "rollback_exact_self_owned_inode_only": True,
        },
        "runner_order": order,
        "frozen_runtime": {
            "records": "40_PLUS_120", "total_rows": 160,
            "timesteps": 10,
            "configs": ["A1_OSG", "EQUAL_SERVICE_K1X8", "TYPED_SIGNED_K8"],
            "lanes": 96, "onchip_sram_bytes": 245760,
            "accumulator_bits": 24, "clock_ns": 3.0,
            "external_bytes_per_cycle": 192,
            "d1_charged_nonheadline": True,
            "only_legal_headline_ratio": "TYPED_SIGNED_K8_VS_EQUAL_SERVICE_K1X8",
        },
        "formal_population_before": before_population,
        "formal_population_after": after_population,
        "frozen_sha_before": frozen_before,
        "frozen_sha_after": frozen_after,
        "runner_mode": "REGULAR_0664",
        "forbidden_invocations": {
            "one_shot_runner": 0, "production": 0, "vcs": 0,
            "license_query": 0, "eda": 0, "cpu_training": 0,
            "gpu": 0, "remote": 0, "network_jobs": 0,
        },
        "production_cycles": None,
        "speedup_citable": False,
        "raw_result_requires_fresh_result_hammer": True,
        "docs359_sha256": frozen_after["docs359"],
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
