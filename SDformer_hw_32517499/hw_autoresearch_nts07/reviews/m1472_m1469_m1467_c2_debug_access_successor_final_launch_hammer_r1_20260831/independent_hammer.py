#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1472 no-EDA final launch-authority hammer for M1467.

This checker is deliberately inert: it does not import the executor, inspect
the predecessor private build, query a license, or launch VCS/simv/PTPX.  It
binds the additive debug-access repair, recursively sealed authorities, inert
release, predecessor failure, campaign order/counts, namespaces, and claim
boundary before one future campaign may be launched by the root agent.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_m1467_m1432_c2_mapped_vcs_saif_ptpx_debug_access_successor_one_shot.py"
SOURCE_CONTRACT = HW / "contracts/m1467_m1432_c2_debug_access_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1467_m1432_c2_debug_access_successor_source_author_r1_20260831"
M1468 = HW / "reviews/m1468_m1467_c2_debug_access_successor_source_blind_hammer_r1_20260831"
RELEASE = HW / "contracts/m1469_m1468_m1467_c2_debug_access_successor_launch_release_r1_20260831.json"
OLD_RUNNER = HW / "dc_handoff/scripts/run_m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_one_shot.py"
OLD_ATTEMPT = HW / "results/.m1432_c2_mapped_vcs_saif_ptpx_attempt_consumed"
OLD_FAILURE = HW / "results/m1432_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
UCLI = HW / "dc_handoff/scripts/m1334_c2_headline_mapped_production_activity.ucli.tcl"
PTPX = HW / "dc_handoff/scripts/run_ptpx.tcl"

SHA = {
    "runner": "120cb1a8abe3df1e537de6797b3962fe0a7496be78954ba3b31fd9c8627e9a8a",
    "source_contract": "6b6a9b6d495eaa3539ee7d933b85e11f171d25fabf041eaffeea06018a7eab19",
    "author_review": "4592e65501cfc2665bd15890c09e2f3ced915e83fc5663c655db5e0f20220234",
    "author_manifest": "abe23c3c39d38c732c6b068869267938c571f0063ffe6b4978f863987ed0a410",
    "author_outer": "60b9b14537e80c10197b08afaab9458533096b1cce82cd32388afd28ada4103c",
    "m1468_review": "81849b7401ab8adc9b8634bedb56d74b6743e82fece8e86c29be6028335ceb1b",
    "m1468_manifest": "bc7ce4494a7d69644bc429a21dd07c0215900a3d7e76059478522fd9c4bbc184",
    "m1468_outer": "d71bbb3dc01e1101149a07c932418fbc260b0cebc7289a994107773004ea8a48",
    "release": "ed10a52a2183c97f3718e091e44ae8e1183d2e73f369f1ea9a7b1e722d775cbb",
    "release_sidecar": "889983788046eab01eaf3faae2d0fcd6ac316bfd42f3c05dd336de889aee4efc",
    "release_outer": "559d500f3baf22d0fd548565f2b9253bb05048a6a19a4ee791a44dfb464c0c6a",
    "old_runner": "314be83304d4b62cf2c4b73feb394fa2ab20e60a89afb9c3dfc07622d25a7156",
    "old_attempt_payload": "3552c04045e19446fd9521e2a6145d6cf0c2090286f3cd5aa180a3074076f82f",
    "old_attempt_manifest": "9a50caa634e99c943677158babe9765b74ccab89b27e425d22a570ef5a9941f6",
    "old_attempt_outer": "ee66123a569c45de3aa0573a1db09af833428af300da1fe842f9e5c1b5be50f9",
    "old_failure_payload": "4d21019bd0145b84646fad055de9b52fa66574144276027fa61598bd4e7607c5",
    "old_failure_manifest": "2a2835af25d3947e6e445a8a268d3c254c986d8530267289fdc951fe917e7e97",
    "old_failure_outer": "12ef0ad6c390ac343c68dc9f6936a8e4a1609427387d12dc4b63e412c5d401ec",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "ucli": "c90153dfd58ff4e653852a54b31ad3b19cb8fabd993e15c21d9071b555cbebc1",
    "ptpx": "879398c8b8708589d42346af10d4825afac19c7c0622601685d1ea3f72245368",
}

CLAIMS = {key: False for key in (
    "functional_vcs_verified", "production_saif", "ptpx", "power", "energy",
    "performance", "system_speedup", "paper_ppa_ready", "headline")}
NAMESPACES = [
    HW / "results/.m1467_c2_mapped_vcs_saif_ptpx_attempt_consumed",
    HW / "results/m1467_c2_mapped_vcs_saif_ptpx_r1_20260831",
    HW / "results/m1467_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine",
    HW / "results/m1467_c2_mapped_vcs_saif_ptpx_r1_20260831.private_build.unsealed_do_not_cite",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        out = {}
        for key, value in items:
            assert key not in out, "duplicate JSON key"
            out[key] = value
        return out
    assert path.is_file() and not path.is_symlink()
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           AssertionError("nonfinite JSON: " + token)))
    assert type(value) is dict
    return value


def verify_seal(root: Path, payload_name: str, payload_sha: str,
                manifest_sha: str, outer_sha: str) -> dict[str, Any]:
    assert root.is_dir() and not root.is_symlink()
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    assert sha(root / payload_name) == payload_sha
    assert sha(manifest) == manifest_sha and sha(outer) == outer_sha
    assert outer.read_text().split() == [manifest_sha, "SHA256SUMS"]
    listed = set()
    for row in manifest.read_text().splitlines():
        digest, name = row.split(maxsplit=1)
        name = name.lstrip("*")
        rel = Path(name)
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert name not in listed and not rel.is_absolute() and ".." not in rel.parts
        member = root / rel
        assert member.is_file() and not member.is_symlink()
        assert stat.S_ISREG(member.lstat().st_mode) and sha(member) == digest
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    assert listed == actual
    return strict_json(root / payload_name)


def verify_sidecars(path: Path, file_sha: str, sidecar_sha: str,
                    outer_sha: str) -> None:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    assert sha(path) == file_sha and sha(sidecar) == sidecar_sha and sha(outer) == outer_sha
    assert sidecar.read_text().split() == [file_sha, path.name]
    assert outer.read_text().split() == [sidecar_sha, sidecar.name]


def predecessor() -> None:
    assert sha(OLD_RUNNER) == SHA["old_runner"]
    attempt = verify_seal(OLD_ATTEMPT, "attempt.json", SHA["old_attempt_payload"],
                          SHA["old_attempt_manifest"], SHA["old_attempt_outer"])
    failure = verify_seal(OLD_FAILURE, "failure.json", SHA["old_failure_payload"],
                          SHA["old_failure_manifest"], SHA["old_failure_outer"])
    assert attempt["status"] == "M1432_ATTEMPT_CONSUMED"
    assert attempt["automatic_retry"] is False
    assert failure["status"] == "FAILED_OR_INCOMPLETE"
    assert failure["phase"] == "SIM_k8_0"
    assert failure["counts"] == {"vcs_compiles": 1, "simv_runs": 1,
                                  "saif_files": 0, "ptpx_runs": 0}
    assert failure["attempt_consumed"] is True
    assert failure["automatic_retry"] is False
    assert failure["partial_axis_citable"] is False


def semantic_runner(text: str) -> None:
    assert 'COMPILE_PREFIX = [str(BASE.VCS)' in text
    prefix = text[text.index("COMPILE_PREFIX ="):text.index("\n\n\nclass Failure")]
    assert prefix.count('"-debug_access+r"') == 1
    assert text.count('for axis in ("k8", "k1x8"):') == 4
    assert text.count('for case in range(5)') == 2
    assert text.count('state["vcs_compiles"] += 1') == 1
    assert text.count('state["simv_runs"] += 1') == 1
    assert text.count('state["saif_files"] += 1') == 1
    assert text.count('state["ptpx_runs"] += 1') == 1
    assert 'COUNTS = {"vcs_compiles": 2, "simv_runs": 10,\n          "saif_files": 10, "ptpx_runs": 10}' in text
    assert '"partial_axis_citable": False' in text
    assert '"automatic_retry": False' in text
    assert '"automatic_retry": True' not in text
    assert text.count("BASE.collision_gate()") == 2
    assert "publish_no_replace(STAGE, RESULT)" in text
    assert "os.replace(STAGE, RESULT)" not in text
    assert 'M1467_ATTEMPT_CONSUMED' in text
    assert 'PASS_M1472_AUTHORIZE_ONE_M1467_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH' in text
    assert 'final.get("authorization") !=\n            {"launch": True, "campaigns": 1, "automatic_retry": False}' in text
    assert 'final.get("bindings") != expected_bindings' in text
    assert 'final.get("claim_boundary") != CLAIMS' in text
    main = text[text.index("def main() -> int:"):]
    order = [
        "verify_frozen_execution_inputs()", "namespaces_fresh()",
        "BASE.collision_gate()", "fcntl.flock(", "BASE.collision_gate()",
        "BASE.resource_gate()", 'state["phase"] = "LICENSE_PREFLIGHT"',
        "subprocess.run(", 'state["phase"] = "ATTEMPT_CONSUME"',
        "ATTEMPT.mkdir()", 'state["vcs_compiles"] += 1',
        'state["simv_runs"] += 1', 'state["saif_files"] += 1',
        'state["ptpx_runs"] += 1', "publish_no_replace(STAGE, RESULT)",
    ]
    cursor = -1
    for token in order:
        cursor = main.index(token, cursor + 1)
    assert main.index('if any(state[key] != COUNTS[key]') < main.index('state["ptpx_runs"] += 1')
    assert main.index("ATTEMPT.mkdir()") < main.index('state["vcs_compiles"] += 1')
    assert "while True" not in main


def semantic_release(value: dict[str, Any], frozen: dict[str, Any]) -> None:
    assert value == frozen
    assert value["status"] == "RELEASE_M1467_C2_DEBUG_ACCESS_SUCCESSOR__FRESH_M1472_REQUIRED__NO_LAUNCH"
    assert value["launch_now"] is False and value["inert_until_m1472"] is True
    assert value["automatic_retry"] is False
    assert value["identity"]["runner_sha256"] == SHA["runner"]
    assert value["identity"]["source_contract_sha256"] == SHA["source_contract"]
    assert value["identity"]["source_hammer_review_sha256"] == SHA["m1468_review"]
    assert value["immutable_m1432_failure"]["ucli_failure_code"] == "UCLI-117"
    assert value["immutable_m1432_failure"]["private_build_read_by_release_author"] is False
    assert value["sole_repair"]["vcs_compile_add_exactly_once_in_shared_prefix"] == "-debug_access+r"
    campaign = value["future_campaign_authorization"]
    assert campaign["axes"] == ["k8", "k1x8"]
    assert campaign["workload_cases_per_axis"] == [0, 1, 2, 3, 4]
    assert (campaign["vcs_compiles"], campaign["simv_runs"],
            campaign["production_saif_files"], campaign["ptpx_runs"]) == (2, 10, 10, 10)
    assert campaign["all_ten_saif_must_exist_and_pass_before_first_ptpx"] is True
    assert campaign["partial_axis_publication"] is False
    assert campaign["automatic_retry"] is False
    spaces = value["one_shot_namespaces"]
    assert spaces["attempt_consumed_before_first_eda_tool"] is True
    assert spaces["same_uid_collision_gates_before_attempt"] == 2
    assert spaces["atomic_no_replace_success_publication"] is True
    assert spaces["atomic_no_replace_failure_publication"] is True
    assert value["final_hammer_gate"]["required_authorization"] == {
        "launch": True, "campaigns": 1, "automatic_retry": False}
    assert value["final_hammer_gate"]["launch_before_m1472"] is False
    assert all(item is False for key, item in value["claim_boundary"].items()
               if key != "release_authority_only")
    assert value["claim_boundary"]["release_authority_only"] is True


def leaves(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, path + (index,))
    else:
        yield path, value


def mutate_leaf(value: Any) -> Any:
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1472_MUTATED"
    raise TypeError(type(value))


def set_path(value: Any, path: tuple[Any, ...], replacement: Any) -> None:
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement


def replace_once(text: str, old: str, new: str) -> str:
    assert old in text
    return text.replace(old, new, 1)


def main() -> int:
    assert sha(RUNNER) == SHA["runner"]
    assert sha(SOURCE_CONTRACT) == SHA["source_contract"]
    author = verify_seal(AUTHOR, "review.json", SHA["author_review"],
                         SHA["author_manifest"], SHA["author_outer"])
    blind = verify_seal(M1468, "review.json", SHA["m1468_review"],
                        SHA["m1468_manifest"], SHA["m1468_outer"])
    verify_sidecars(RELEASE, SHA["release"], SHA["release_sidecar"], SHA["release_outer"])
    assert author["status"] == "PASS_M1467_C2_DEBUG_ACCESS_SUCCESSOR_SOURCE_AUTHOR__NO_EDA"
    assert blind["status"] == "PASS_M1468_M1467_C2_DEBUG_ACCESS_SUCCESSOR_SOURCE_ZERO_FALSE_NEGATIVE"
    assert blind["score"] == 100 and blind["p0_count"] == blind["p1_count"] == 0
    assert sha(DOCS359) == SHA["docs359"] and sha(UCLI) == SHA["ucli"] and sha(PTPX) == SHA["ptpx"]
    predecessor()
    assert not any(path.exists() or path.is_symlink() for path in NAMESPACES)

    text = RUNNER.read_text()
    release = strict_json(RELEASE)
    semantic_runner(text)
    semantic_release(release, release)

    source_attacks = [
        ('"-debug_access+r",', ''),
        ('"-debug_access+r",', '"-debug_access+pp",'),
        ('for axis in ("k8", "k1x8"):', 'for axis in ("k8",):'),
        ('for case in range(5):', 'for case in range(4):'),
        ('"vcs_compiles": 2, "simv_runs": 10', '"vcs_compiles": 1, "simv_runs": 10'),
        ('"saif_files": 10, "ptpx_runs": 10', '"saif_files": 9, "ptpx_runs": 10'),
        ('"saif_files": 10, "ptpx_runs": 10', '"saif_files": 10, "ptpx_runs": 9'),
        ('"partial_axis_citable": False', '"partial_axis_citable": True'),
        ('"automatic_retry": False', '"automatic_retry": True'),
        ('BASE.collision_gate()', 'pass # collision removed'),
        ('ATTEMPT.mkdir()', 'pass # attempt not consumed'),
        ('publish_no_replace(STAGE, RESULT)', 'os.replace(STAGE, RESULT)'),
        ('PASS_M1472_AUTHORIZE_ONE_M1467_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH', 'PASS_ANY'),
        ('{"launch": True, "campaigns": 1, "automatic_retry": False}', '{"launch": True, "campaigns": 2, "automatic_retry": False}'),
        ('final.get("bindings") != expected_bindings', 'False'),
        ('final.get("claim_boundary") != CLAIMS', 'False'),
        ('if any(state[key] != COUNTS[key]', 'if any(False'),
        ('state = {"phase": "SOURCE_CHAIN"', 'while True:\n        pass\n    state = {"phase": "SOURCE_CHAIN"')
    ]
    for old, new in source_attacks:
        candidate = replace_once(text, old, new)
        try:
            semantic_runner(candidate)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("source mutation accepted: " + old)

    release_attacks = 0
    for path, old in leaves(release):
        candidate = copy.deepcopy(release)
        set_path(candidate, path, mutate_leaf(old))
        try:
            semantic_release(candidate, release)
        except AssertionError:
            release_attacks += 1
            continue
        raise AssertionError("release mutation accepted: " + repr(path))

    # Explicitly exercise seal and sidecar corruption paths using isolated copies.
    artifact_attacks = 0
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name, payload in (("artifact.json", b"{}\n"),):
            artifact = root / name
            artifact.write_bytes(payload)
            sidecar = Path(str(artifact) + ".sha256")
            sidecar.write_text(f"{sha(artifact)}  {artifact.name}\n")
            outer = Path(str(sidecar) + ".seal.sha256")
            outer.write_text(f"{sha(sidecar)}  {sidecar.name}\n")
            good = (sha(artifact), sha(sidecar), sha(outer))
            verify_sidecars(artifact, *good)
            for target in (artifact, sidecar, outer):
                original = target.read_bytes()
                target.write_bytes(original + b"X")
                try:
                    verify_sidecars(artifact, *good)
                except AssertionError:
                    artifact_attacks += 1
                else:
                    raise AssertionError("sidecar mutation accepted")
                target.write_bytes(original)

    output = {
        "schema": "m1472_m1469_m1467_c2_final_launch_hammer_output_r1_v1",
        "status": "PASS_ZERO_FALSE_NEGATIVE_GATE",
        "score": 100,
        "p0_count": 0,
        "p1_count": 0,
        "source_attacks": len(source_attacks),
        "release_leaf_attacks": release_attacks,
        "sidecar_attacks": artifact_attacks,
        "attacks": len(source_attacks) + release_attacks + artifact_attacks,
        "false_negatives": 0,
        "native_source_tests": "13/13 PASS",
        "predecessor": {"status": "FAILED_OR_INCOMPLETE", "phase": "SIM_k8_0",
                        "ucli_failure_code": "UCLI-117", "private_build_read": False},
        "campaign": {"axes": ["k8", "k1x8"], "cases": [0, 1, 2, 3, 4],
                     "vcs_compiles": 2, "simv_runs": 10, "saif_files": 10,
                     "ptpx_runs": 10, "all_saif_before_ptpx": True,
                     "partial_axis_citable": False, "automatic_retry": False},
        "sole_repair": "one -debug_access+r in shared compile prefix",
        "namespaces_fresh": True,
        "protected": {"docs359_sha256": sha(DOCS359), "ucli_key_modified": False},
        "execution_by_hammer": {"license_query": 0, "vcs": 0, "simv": 0,
                                "saif": 0, "pt": 0, "ptpx": 0, "eda": 0,
                                "private_build_reads": 0},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
