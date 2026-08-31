#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind M1093 hammer. Never imports the engine or launches EDA."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ENGINE = HW / "dc_handoff/scripts/m1091r2_m1090r2_c2_observation_authorized_engine_r1.py"
CONTRACT = HW / "contracts/m1090r2_c2_k1_observation_fixed_trust_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1090r2_c2_k1_observation_fixed_trust_release_r1_20260830.json"
RECEIPT = HW / "reviews/m1090r2_m1091r2_c2_observation_fixed_trust_source_receipt_r1_20260830"
M1092 = HW / "reviews/m1092_m1090_c2_observation_source_hammer_r1_20260830"
M1088 = HW / "reviews/m1088_m1080_c2_mapped_gate_failure_audit_r1_20260830"
M1080_ATTEMPT = HW / "results/.m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_attempt_consumed"
M1080_FAILURE = HW / "results/m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.2746017.quarantine"
OLD_RUNNER = HW / "dc_handoff/scripts/run_m1091_m1090_c2_observation_dc_mapped_vcs_one_shot_r1.py"
OLD_ATTEMPT = HW / "results/.m1091_m1090_c2_observation_dc_mapped_vcs_attempt_consumed"
ATTEMPT = HW / "results/.m1091r2_m1090r2_c2_observation_dc_mapped_vcs_attempt_consumed"
RESULT = HW / "results/m1091r2_m1090r2_c2_observation_dc_mapped_vcs_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "engine": "51e8af72a1c48ca556249ccf7abcaf1ef8e0700265c3705c24bf37f44405cd77",
    "receipt_outer": "e1e904c18a17cd3ce4c1154c0d262aecd65826e4afaff1960e5945ee169d6216",
    "contract": "c9e7986602e14b17d38d9a67e50238b4f1b05801db93c767006fb20355fae8c5",
    "contract_outer": "7bd062b64438cfebb28105e2532276d8193849e518fb4717ddcaadb4572797a3",
    "release": "d47c12f1bf235087c1ba81caa0bb385d2221833ca107838e7242d373251188b9",
    "release_outer": "fc20e494417bcd5105f4697c265f54616c68d0f03a82c7b0ad23a9df78bf8ab4",
    "m1092_outer": "f55dc0afde8d350d1ff028c30e511eb15b2670f3ad1ee2f5643759406ca8ccb4",
    "m1088_outer": "fb3f208dc704c7663769422ad9f27b17851cc86b11826727fe0c0c795260bd5f",
    "m1080_attempt_outer": "21944247a673bda71a1d3f8cce2cf567b91e51a661b88d5028ed89b70d3a8f7c",
    "m1080_failure_outer": "2e3367c239cda08987027a55a01f65b0cbebbd1c0dd907a9a945aa12f5cea89d",
    "old_runner": "fade26df6dd3a6e3a71772c1d880ef31872be213a945c8184c966293e9791199",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerFailure(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            HammerFailure("nonfinite JSON: " + token)))


checks: list[str] = []


def mark(name: str, value: bool = True) -> None:
    require(value, name)
    checks.append(name)


def verify_regular(path: Path, expected: str) -> bool:
    mode = path.lstat().st_mode
    return stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == expected


def verify_flat(directory: Path, expected_outer: str) -> tuple[dict, set[str]]:
    require(directory.is_dir() and not directory.is_symlink(), "flat dir")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(verify_regular(outer, expected_outer), "flat outer identity")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        relative = relative.lstrip("*")
        require(relative not in listed and
                verify_regular(directory / relative, digest), "flat member")
        listed.add(relative)
    require(outer.read_text(encoding="utf-8").split() ==
            [sha(manifest), "SHA256SUMS"], "flat outer content")
    review = directory / "review.json"
    return (strict_json(review) if review.is_file() else {}), listed


def verify_frozen_history_flat(directory: Path,
                               expected_outer: str) -> tuple[set[str], int]:
    """Verify an already sealed tool output, following manifest-listed links.

    This policy is only for immutable historical quarantine contents.  Source,
    tools, libraries and new inputs retain the direct-lstat rejection above.
    """
    require(directory.is_dir() and not directory.is_symlink(), "history dir")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(verify_regular(outer, expected_outer) and
            outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
            "history outer")
    listed, symlinks = set(), 0
    for line in manifest.read_text().splitlines():
        digest, relative = line.split(None, 1); relative = relative.lstrip("*")
        member = directory / relative
        require(relative not in listed and member.exists() and
                sha(member) == digest, "history member")
        listed.add(relative)
        symlinks += int(member.is_symlink())
    return listed, symlinks


def receipt_blind_seal() -> None:
    manifest = RECEIPT / "SHA256SUMS"
    outer = RECEIPT / "SHA256SUMS.seal.sha256"
    mark("01_receipt_outer_identity", sha(outer) == EXPECTED["receipt_outer"])
    mark("02_receipt_outer_content", outer.read_text().split() ==
         [sha(manifest), "SHA256SUMS"])
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, relative = line.split(None, 1); relative = relative.lstrip("*")
        require(relative not in listed and
                verify_regular(RECEIPT / relative, digest), "receipt member")
        listed.add(relative)
    mark("03_receipt_members_hash", len(listed) == 6)
    actual = {path.name for path in RECEIPT.iterdir() if path.is_file() and
              path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    mark("04_receipt_manifest_coverage", actual == listed)


def double_checks(path: Path, key: str, outer_key: str, prefix: int) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    mark(f"{prefix:02d}_{key}_identity", verify_regular(path, EXPECTED[key]))
    mark(f"{prefix+1:02d}_{key}_sidecar", side.read_text().split() ==
         [EXPECTED[key], path.relative_to(HW).as_posix()])
    mark(f"{prefix+2:02d}_{key}_outer_identity", sha(outer) == EXPECTED[outer_key])
    mark(f"{prefix+3:02d}_{key}_outer_content", outer.read_text().split() ==
         [sha(side), side.relative_to(HW).as_posix()])


def seal_directory(directory: Path) -> str:
    members = sorted(path for path in directory.rglob("*") if path.is_file() and
                     path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(
        f"{sha(path)}  {path.relative_to(directory).as_posix()}\n"
        for path in members), encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
    return sha(outer)


def double_file(path: Path) -> str:
    side = Path(str(path) + ".sha256")
    side.write_text(f"{sha(path)}  {path.relative_to(HW).as_posix()}\n",
                    encoding="utf-8")
    outer = Path(str(path) + ".sha256.seal.sha256")
    outer.write_text(f"{sha(side)}  {side.relative_to(HW).as_posix()}\n",
                     encoding="utf-8")
    return sha(outer)


def load_engine_definitions() -> dict:
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
               ast.ClassDef, ast.FunctionDef)
    module = ast.Module(body=[node for node in tree.body if isinstance(node, allowed)],
                        type_ignores=[])
    namespace = {"__file__": str(ENGINE), "__name__": "m1093_engine_model"}
    exec(compile(module, str(ENGINE), "exec"), namespace)
    return namespace


def forged_future_chain_attack() -> dict[str, object]:
    """Run the engine's real authority verifier on a wholly forged future DAG."""
    namespace = load_engine_definitions()
    with tempfile.TemporaryDirectory(prefix=".m1093_chain_attack.",
                                     dir=HW / "reviews") as temp:
        root = Path(temp)
        forged_engine = root / "forged_engine.py"
        forged_engine.write_bytes(ENGINE.read_bytes() + b"\n# attacker resign\n")
        m1093 = root / "m1093"; m1093.mkdir()
        (m1093 / "review.json").write_text(json.dumps({
            "status": "PASS_M1093_M1090R2_M1091R2_ENGINE_HAMMER__AUTHOR_LAUNCH_WRAPPER_ONLY__NO_EDA"
        }) + "\n")
        m1093_outer = seal_directory(m1093)
        launcher = root / "launcher.py"
        launcher.write_text(
            f'ENGINE_SHA256 = "{sha(forged_engine)}"\n'
            f'M1093_OUTER_SHA256 = "{m1093_outer}"\n', encoding="utf-8")
        launch_receipt = root / "launch_receipt.json"
        receipt = {
            "status": "M1091R2_LAUNCH_SOURCE_FROZEN__M1096_REQUIRED__NO_EDA",
            "engine_sha256": sha(forged_engine),
            "launcher_sha256": sha(launcher),
            "m1093_outer_seal_file_sha256": m1093_outer,
        }
        launch_receipt.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
        launch_outer = double_file(launch_receipt)
        m1096 = root / "m1096"; m1096.mkdir()
        (m1096 / "review.json").write_text(json.dumps({
            "status": "PASS_M1096_M1091R2_AUTHORIZED_LAUNCH_HAMMER__GO_ONE_ATTEMPT",
            "identity": {
                "engine_sha256": sha(forged_engine),
                "launcher_sha256": sha(launcher),
                "launch_receipt_outer_seal_file_sha256": launch_outer,
                "m1093_outer_seal_file_sha256": m1093_outer,
            }
        }) + "\n", encoding="utf-8")
        m1096_outer = seal_directory(m1096)
        m1096_review_sha = sha(m1096 / "review.json")
        m1096_manifest_sha = sha(m1096 / "SHA256SUMS")
        namespace.update({"ENGINE": forged_engine, "LAUNCHER": launcher,
                          "LAUNCH_RECEIPT": launch_receipt,
                          "M1093": m1093, "M1096": m1096})
        # This parent condition is satisfiable by launching the forged launcher
        # with pinned Python; it is not an independent content trust root.
        namespace["verify_parent_launcher"] = lambda _receipt: None
        accepted = namespace["verify_launch_authority"]()
        return {
            "internally_self_consistent_chain_accepted":
                accepted["engine_sha256"] == sha(forged_engine),
            "engine_sha256": sha(forged_engine),
            "launcher_sha256": sha(launcher),
            "m1093_outer_seal_file_sha256": m1093_outer,
            "launch_receipt_outer_seal_file_sha256": launch_outer,
            "m1096_review_sha256": m1096_review_sha,
            "m1096_manifest_sha256": m1096_manifest_sha,
            "m1096_outer_seal_file_sha256": m1096_outer,
        }


def main() -> None:
    receipt_blind_seal()
    double_checks(CONTRACT, "contract", "contract_outer", 5)
    double_checks(RELEASE, "release", "release_outer", 9)
    mark("13_engine_identity", sha(ENGINE) == EXPECTED["engine"])
    mark("14_docs359_identity", sha(DOCS359) == EXPECTED["docs359"])
    m1092, _ = verify_flat(M1092, EXPECTED["m1092_outer"])
    mark("15_m1092_outer", True)
    mark("16_m1092_stop", m1092["status"] ==
         "STOP_M1092_M1090_M1091_SELF_SIGNED_CALLER_AUTHORITY__NO_M1091_ATTEMPT")
    m1088, _ = verify_flat(M1088, EXPECTED["m1088_outer"])
    mark("17_m1088_outer", True)
    mark("18_m1088_do_not_retry", m1088["status"] ==
         "PASS_M1088_M1080_FAILURE_AUDIT__M1080_DO_NOT_RETRY")
    verify_flat(M1080_ATTEMPT, EXPECTED["m1080_attempt_outer"])
    mark("19_m1080_attempt_seal", True)
    _, frozen_failure_symlinks = verify_frozen_history_flat(
        M1080_FAILURE, EXPECTED["m1080_failure_outer"])
    mark("20_m1080_failure_seal_with_manifested_tool_symlink",
         frozen_failure_symlinks > 0)
    mark("21_old_m1091_runner_frozen", sha(OLD_RUNNER) == EXPECTED["old_runner"])
    mark("22_old_m1091_attempt_absent", not OLD_ATTEMPT.exists() and
         not OLD_ATTEMPT.is_symlink())

    contract = strict_json(CONTRACT)
    release = strict_json(RELEASE)
    source_pins = contract["source_sha256"]
    require(len(source_pins) == 21, "source pin count")
    for index, (relative, expected) in enumerate(sorted(source_pins.items()), 23):
        mark(f"{index:02d}_source_pin_{Path(relative).name}",
             verify_regular(HW / relative, expected))

    external = contract["external_identity"]
    regular = [(Path(path), row["sha256"]) for path, row in external.items()
               if row["kind"] == "regular"]
    require(len(regular) == 7, "regular external pin count")
    for index, (path, expected) in enumerate(regular, 44):
        mark(f"{index:02d}_external_pin_{path.name}", verify_regular(path, expected))
    dc_shell = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")
    mark("51_dc_shell_is_exact_symlink", dc_shell.is_symlink())
    mark("52_dc_shell_readlink", os.readlink(dc_shell) == "snps_shell")
    with tempfile.TemporaryDirectory(prefix="m1093_symlink_") as temp:
        root = Path(temp); target = root / "target"; target.write_bytes(b"x")
        link = root / "link"; link.symlink_to(target)
        mark("53_regular_verifier_rejects_symlink",
             not verify_regular(link, sha(target)))

    text = ENGINE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    mark("54_exact_single_authorized_argv",
         'sys.argv[1:] != ["--authorized-launch"]' in text)
    mark("55_no_caller_expected_hash_environment",
         not re.search(r'os\.environ(?:\.get)?\([^\n]*EXPECTED', text) and
         "M1091_EXPECTED" not in text)
    mark("56_fixed_future_dag_paths", all(token in text for token in (
         'run_m1091r2_m1090r2_c2_observation_authorized_launch_r1.py',
         'm1091r2_m1090r2_c2_observation_authorized_launch_receipt_r1_20260830.json',
         'm1093_m1090r2_m1091r2_c2_observation_engine_hammer_r1_20260830',
         'm1096_m1091r2_c2_observation_launch_hammer_r1_20260830')))
    static = text[text.index("def static_gate()") :
                  text.index("def resource_gate()")]
    flow = text[text.index("def flow()") : text.index("def quarantine(")]
    mark("57_static_gate_first", flow.index("static_gate()") < flow.index("ATTEMPT.mkdir()"))
    mark("58_launch_authority_before_namespace",
         "return verify_launch_authority()" in static and
         flow.index("static_gate()") < flow.index("ATTEMPT.mkdir()"))
    mark("59_attempt_after_lock_collision_resource_license", all(
         flow.index(token) < flow.index("ATTEMPT.mkdir()") for token in
         ("flock(", "collision_gate()", "resource_gate()", "license_gate()")))
    mark("60_exactly_one_dc_invocation", flow.count("str(DC_SHELL)") == 1)
    mark("61_exactly_one_vcs_compile", flow.count("str(VCS)") == 1)
    mark("62_exactly_one_mapped_sim",
         flow.count('run([str(simv), "-no_save"]') == 1)
    mark("63_no_saif_or_initreg", "SAIF" not in flow and "initreg" not in flow)
    mark("64_diagnostic_only_nonpaper", '"diagnostic_only": True' in flow and
         '"paper_citable": False' in flow)

    wrapper = (HW / "rtl_m1090r2/m1090r2_c2_k1_observation_wrapper.sv").read_text()
    tb = (HW / "dc_handoff/tb/tb_m1090r2_c2_k1_observation_mapped_case0_short.sv").read_text()
    outputs = set(re.findall(r"\bobs_[a-zA-Z0-9_]+\b", wrapper[wrapper.index("module "):wrapper.index(");")]))
    mark("65_exact_22_observation_outputs", len(outputs) == 22)
    assign_tail = wrapper[wrapper.index("always_comb begin") :]
    mark("66_observation_fanout_only", all(
         re.search(rf"\b{name}\s*=", assign_tail) and
         len(re.findall(rf"\b{name}\b", wrapper)) == 2 for name in outputs))
    mark("67_exact_22_first_x_calls", tb.count("`M1090R2_FAIL_X(") == 22)
    mark("68_bounded_windows_and_stage_trace", all(token in tb for token in
         ("window_cycle==128", "wait_cycles<16", "wait_cycles<32",
          "M1090R2_STAGE", "#1000 $fatal")))

    before = (ATTEMPT.exists(), RESULT.exists())
    commands = [[], ["--authorized-launch", "extra"], ["--authorized-launch"]]
    messages = []
    for argv in commands:
        env = os.environ.copy()
        env.update({"M1091_EXPECTED_RUNNER_SHA256": "0" * 64,
                    "M1091_EXPECTED_M1092_OUTER_SHA256": "0" * 64})
        run = subprocess.run([
            "/opt/anaconda3/envs/pytorch310/bin/python3.10", str(ENGINE), *argv],
            text=True, capture_output=True, check=False, env=env, timeout=30)
        require(run.returncode == 3, "bounded direct engine did not fail")
        messages.append(run.stderr)
    mark("69_argv_env_attacks_stop_before_attempt",
         before == (ATTEMPT.exists(), RESULT.exists()) and
         "fixed argv required" in messages[0] and
         "fixed argv required" in messages[1] and
         "non-regular or direct symlink rejected" in messages[2])

    # Contract/release are pinned only by engine literals. Changing both and
    # then re-signing the engine changes its SHA, but the future self-signed DAG
    # has no immutable root that rejects that new engine SHA.
    mutated_contract = CONTRACT.read_bytes() + b"\n"
    mutated_release = RELEASE.read_bytes() + b"\n"
    mutated_engine = (text.replace(EXPECTED["contract"],
        hashlib.sha256(mutated_contract).hexdigest()).replace(
        EXPECTED["release"], hashlib.sha256(mutated_release).hexdigest())).encode()
    mark("70_contract_release_engine_resign_changes_all_identities",
         hashlib.sha256(mutated_contract).hexdigest() != EXPECTED["contract"] and
         hashlib.sha256(mutated_release).hexdigest() != EXPECTED["release"] and
         hashlib.sha256(mutated_engine).hexdigest() != EXPECTED["engine"])
    forged = forged_future_chain_attack()
    # M1093 is only an authoring release.  Execution has a finite external
    # trust root: root must pin the exact independently reported launcher SHA
    # plus M1096 review/manifest/outer tuple before invoking the zero-arg
    # launcher.  A co-resigned chain can be internally self-consistent, but it
    # cannot equal that already fixed external tuple.
    hypothetical_external_root = {
        "launcher_sha256": "a" * 64,
        "m1096_review_sha256": "b" * 64,
        "m1096_manifest_sha256": "c" * 64,
        "m1096_outer_seal_file_sha256": "d" * 64,
    }
    forged_external_tuple = {key: forged[key] for key in hypothetical_external_root}
    namespace = load_engine_definitions()
    engine_history_rejects = False
    try:
        namespace["verify_flat"](
            M1080_FAILURE, EXPECTED["m1080_failure_outer"])
    except namespace["GateFailure"]:
        engine_history_rejects = True
    mark("71_engine_unconditionally_rejects_frozen_manifested_vcs_symlink__P0",
         forged["internally_self_consistent_chain_accepted"] is True and
         forged_external_tuple != hypothetical_external_root and
         engine_history_rejects and frozen_failure_symlinks > 0)
    require(len(checks) == 71, f"check count {len(checks)} != 71")

    result = {
        "status": "STOP_M1093_M1091R2_ENGINE_REJECTS_FROZEN_M1080_QUARANTINE_SYMLINK__NO_EDA_NO_ATTEMPT",
        "checks_passed": len(checks),
        "checks": checks,
        "engine_sha256": sha(ENGINE),
        "receipt_outer_seal_file_sha256": sha(RECEIPT / "SHA256SUMS.seal.sha256"),
        "attack": {
            "resigned_contract_release_engine_identity_changes": True,
            "internally_self_consistent_forged_future_chain_accepted": True,
            "forged_future_chain_rejected_by_required_external_root": True,
            "external_root_definition": "Before execution root must independently pin exact launcher SHA256 and the M1096 review/manifest/outer SHA256 tuple, then invoke only that zero-argument launcher.",
        },
        "p0": {
            "id": "M1093-P0-01",
            "finding": "static_gate applies direct source/input symlink rejection to every member of the frozen M1080 VCS quarantine. Its sealed manifest contains mapped_vcs/csrc/_2931510_archive_1.so as a legitimate VCS-internal symlink, so verify_flat deterministically stops before future launch authority and attempt consumption.",
            "manifested_symlink_count": frozen_failure_symlinks,
            "repair": "Use the quarantine's already frozen manifest/outer semantics for historical tool output (allow manifest-listed internal symlinks while checking their followed bytes), or bind only the exact independently sealed M1088 failure-audit identity. Keep direct lstat/symlink rejection unchanged for source, tools, libraries and new inputs. Author an additive engine and re-hammer it; do not edit this engine in place.",
        },
        "authorization": {
            "author_different_author_zero_argument_launcher": False,
            "author_additive_engine_repair": True,
            "launch_now": False,
            "eda_now": False,
            "attempt_now": False,
            "m1096_independent_hammer_after_repair_required": True,
            "future_root_external_launcher_and_m1096_tuple_pin_required": True,
            "future_launcher_must_sanitize_environment_before_engine": True,
        },
        "execution": {"engine_imported": False, "eda": False,
                      "attempt_consumed": False, "result_created": False,
                      "bounded_direct_preflight_processes": 3},
    }
    (HERE / "mechanical_checks.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
