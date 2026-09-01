#!/usr/bin/env python3
"""Different-author, compile-free hammer for the exact M1621 runner repair."""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
OLD = HW / "dc_handoff/scripts/run_vcs_m1613_c2_m1609_registered_fault_directed_exact_sha_r1.sh"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1621_m1613_c2_m1609_registered_fault_directed_exact_sha_r1.sh"
CONTRACT = HW / "contracts/m1621_m1613_python36_regular_path_runner_successor_source_contract_r1_20260901.json"
AUTHOR_TEST = HW / "system_simulator/tests/test_m1621_m1613_python36_regular_path_runner_successor_source.py"
AUTHOR = HW / "reviews/m1621_m1613_python36_regular_path_runner_successor_author_receipt_r1_20260901"
RTL = HW / "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv"
TB = HW / "dc_handoff/tb/tb_m1613_c2_m1609_registered_fault_directed.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1613_c2_m1609_registered_fault_directed_vcs.f"
M1613_CONTRACT = HW / "contracts/m1613_c2_m1609_registered_fault_directed_source_contract_r1_20260901.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "results/m1613_c2_m1609_registered_fault_directed_vcs_r1_20260901"
ATTEMPT = HW / "results/.m1613_c2_m1609_registered_fault_directed_vcs_attempt_consumed"
RELEASE = HW / "contracts/m1623_m1622_m1621_m1613_c2_registered_fault_directed_vcs_launch_release_r1_20260901.json"
PY36_OLD = Path("/usr/bin/python3.6")
PY36_NEW = Path("/usr/libexec/platform-python3.6")

PINS = {
    OLD: "f2b3888879cb5a6af4396eb8b4971510453a47622299e17dd6702925587c0b29",
    RUNNER: "11da68ff4eb9da70c83b56ae7dd2dbff26f125833224beb08f165fe97a0ea30b",
    CONTRACT: "ccda9594402154f3aeb3105da014cdb2134015017b2c113aa604ed4e385a9fdb",
    AUTHOR_TEST: "2349a3f8380568f9c31eb747422fc370719a75aaec47357f1f61296bce07094b",
    AUTHOR / "review.json": "49cfbc36692065ece3bb6efbdbc515d8ccbba47122f65c38f6c45739e48c8db8",
    AUTHOR / "SHA256SUMS": "17e15fb8977be691b0cc5473e7486ec052d10c315e01e962d854d01dbffdfcf4",
    AUTHOR / "SHA256SUMS.seal.sha256": "946f33169cf3859dea89e2993d0b61835cecd87a8838dd5b042498561a4b0d84",
    RTL: "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    TB: "096f32095a81abbedf4c0bda59a2a146df764d18bdc2136c31e0c7a7319a57a4",
    FILELIST: "071e37d731988a12c3b0adc6380e179de55260c34325fce944daabba6c58671d",
    M1613_CONTRACT: "248c9065d81608a8fc2aacdd8539a3287462653e411ee545a8f320a98a8a5f8d",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    PY36_NEW: "9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f",
}

OLD_REL = "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"
NEW_REL = "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv"
TB_REL = "dc_handoff/tb/tb_m1613_c2_m1609_registered_fault_directed.sv"
RESULT_REL = "results/m1613_c2_m1609_registered_fault_directed_vcs_r1_20260901"
ATTEMPT_REL = "results/.m1613_c2_m1609_registered_fault_directed_vcs_attempt_consumed"
REVIEW_REL = "reviews/m1622_m1621_m1613_c2_registered_fault_directed_runner_source_hammer_r1_20260901"
RELEASE_REL = "contracts/m1623_m1622_m1621_m1613_c2_registered_fault_directed_vcs_launch_release_r1_20260901.json"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(
                             AssertionError("nonfinite JSON: " + token)))


def verify_tree(root):
    """Verify both hashes and exact flat/nested topology; reject unlisted extras."""
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(root.is_dir() and not root.is_symlink(), "tree root nonregular")
    require(manifest.is_file() and not manifest.is_symlink(), "manifest absent")
    require(outer.is_file() and not outer.is_symlink(), "outer seal absent")
    listed = {}
    for row in manifest.read_text(encoding="ascii").splitlines():
        if not row.strip():
            continue
        pieces = row.split("  ", 1)
        require(len(pieces) == 2, "malformed manifest row")
        digest, name = pieces
        rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                "malformed digest")
        require(not rel.is_absolute() and ".." not in rel.parts,
                "unsafe manifest path")
        require(name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"),
                "recursive manifest member")
        require(name not in listed, "duplicate manifest member")
        listed[name] = digest
    require(listed, "empty manifest")
    require(outer.read_text(encoding="ascii") ==
            "%s  SHA256SUMS\n" % sha256(manifest), "outer seal mismatch")
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        base_path = Path(base)
        for name in list(dirs):
            require(not (base_path / name).is_symlink(), "symlink directory")
        for name in files:
            path = base_path / name
            require(stat.S_ISREG(path.lstat().st_mode), "nonregular member")
            rel = path.relative_to(root).as_posix()
            if rel not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                actual.add(rel)
    require(actual == set(listed), "manifest topology mismatch")
    for name, digest in listed.items():
        path = root / name
        require(path.is_file() and not path.is_symlink(), "listed member nonregular")
        require(sha256(path) == digest, "listed member hash mismatch")


def verify_file_seal(path):
    path = Path(path)
    manifest = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(path.is_file() and not path.is_symlink(), "sealed file nonregular")
    require(manifest.is_file() and not manifest.is_symlink(), "file manifest nonregular")
    require(outer.is_file() and not outer.is_symlink(), "file outer nonregular")
    require(manifest.read_text(encoding="ascii") ==
            "%s  %s\n" % (sha256(path), path.name), "file manifest mismatch")
    require(outer.read_text(encoding="ascii") ==
            "%s  %s\n" % (sha256(manifest), manifest.name), "file outer mismatch")


def changed(text, old, new):
    require(text.count(old) >= 1, "mutation anchor absent: " + old)
    return text.replace(old, new, 1)


def normalize_successor(text):
    substitutions = (
        ("# M1621 additive M1613 one-shot directed VCS runner source.",
         "# M1613 one-shot directed VCS runner source."),
        ("python36=/usr/libexec/platform-python3.6", "python36=/usr/bin/python3.6"),
        (REVIEW_REL,
         "reviews/m1617_m1613_c2_m1609_registered_fault_directed_source_hammer_r1_20260901"),
        (RELEASE_REL,
         "contracts/m1618_m1617_m1613_c2_registered_fault_directed_vcs_launch_release_r1_20260901.json"),
        ("M1621_EXPECTED_RUNNER_SHA256", "M1613_EXPECTED_RUNNER_SHA256"),
        ("PASS_M1622_M1621_M1613_C2_REGISTERED_FAULT_RUNNER_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT",
         "PASS_M1617_M1613_C2_REGISTERED_FAULT_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT"),
        ("AUTHORIZE_ONE_M1621_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_ATTEMPT",
         "AUTHORIZE_ONE_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_ATTEMPT"),
        ("M1621_EXPECTED_RELEASE_SHA256", "M1613_EXPECTED_RELEASE_SHA256"),
    )
    for before, after in substitutions:
        require(before in text, "normalization anchor absent: " + before)
        text = text.replace(before, after)
    return text


def before(text, left, right):
    require(left in text and right in text and text.index(left) < text.index(right),
            "ordering failure: %s before %s" % (left, right))


def audit_runner(text, old_text):
    require(normalize_successor(text) == old_text,
            "successor delta exceeds path/control-plane repair")
    exact_once = (
        "python36=/usr/libexec/platform-python3.6",
        'result="${hw_root}/' + RESULT_REL + '"',
        'attempt="${hw_root}/' + ATTEMPT_REL + '"',
        'hammer_dir="${hw_root}/' + REVIEW_REL + '"',
        'release="${hw_root}/' + RELEASE_REL + '"',
        "PASS_M1622_M1621_M1613_C2_REGISTERED_FAULT_RUNNER_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT",
        "AUTHORIZE_ONE_M1621_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_ATTEMPT",
        'r["authorization"] == {"vcs_compiles": 1, "simv_runs": 1,',
        'r["identity"]["runner_sha256"] == sha(runner)',
        'r["identity"]["source_contract_sha256"] == sha(contract)',
        'r["identity"]["hammer_review_sha256"] == sha(hammer)',
        'mkdir "${attempt}"',
        'mkdir "${work}"',
        '"${vcs}" -full64 -sverilog -assert svaext -timescale=1ns/1ps',
        '"${simv}" +ntb_random_seed=1613 -no_save -cm assert',
        "source_only=false performance=false",
        '"performance":false,"dc":false,"power":false',
        'mv -T -n "${work}" "${result}"',
    )
    for token in exact_once:
        require(text.count(token) == 1, "runner token count drift: " + token)
    require(text.count("M1621_EXPECTED_RUNNER_SHA256") == 2,
            "runner pin count drift")
    require(text.count("M1621_EXPECTED_RELEASE_SHA256") == 2,
            "release pin count drift")
    require("python36=/usr/bin/python3.6" not in text, "symlink path restored")
    require(text.count('expect_sha "${python36}" 9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f') == 1,
            "python SHA gate drift")
    require(text.count('expect_sha "${successor}" 7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931') == 1,
            "RTL pin drift")
    require(text.count('expect_sha "${testbench}" 096f32095a81abbedf4c0bda59a2a146df764d18bdc2136c31e0c7a7319a57a4') == 1,
            "TB pin drift")
    require(text.count('expect_sha "${filelist}" 071e37d731988a12c3b0adc6380e179de55260c34325fce944daabba6c58671d') == 1,
            "filelist pin drift")
    require(text.count("+ntb_random_seed=1613") == 1, "seed drift")
    require(text.count("VCS_COMPILE vcs_compiles=1") == 1, "compile budget drift")
    require(text.count("SIMV_RUN simv_runs=1 seed=1613") == 1, "sim budget drift")
    require("automatic_retry=false" in text and "rm -rf" not in text,
            "retry/destructive policy drift")
    require('blocked = {"vcs", "vcs1", "vlogan", "simv"}' in text,
            "collision executable population drift")
    require("p.stat().st_uid != os.getuid()" in text, "same-UID gate absent")
    require("int(p.name) in ancestry" in text, "ancestry exclusion absent")
    require("pgrep -x simv" not in text, "global UID-agnostic collision gate")
    require("dc_shell" not in text.lower() and "pt_shell" not in text.lower()
            and "fm_shell" not in text.lower() and "ptpx" not in text.lower(),
            "non-VCS EDA inserted")
    require(text.count(OLD_REL) == 1, "predecessor guard drift")
    require('-f "${filelist}" -top "${top}"' in text, "top/filelist drift")
    before(text, "M1621_EXPECTED_RUNNER_SHA256", 'verify_dir_seal "${hammer_dir}"')
    before(text, 'verify_dir_seal "${hammer_dir}"', "M1621_EXPECTED_RELEASE_SHA256")
    before(text, "M1621_EXPECTED_RELEASE_SHA256", "result/attempt/work namespace is not fresh")
    before(text, "result/attempt/work namespace is not fresh", "same-UID VCS collision:")
    before(text, "same-UID VCS collision:", 'mkdir "${attempt}"')
    before(text, "VCS environment mismatch", 'mkdir "${attempt}"')
    before(text, 'mkdir "${attempt}"', '\n"${vcs}" -full64')
    before(text, '\n"${vcs}" -full64', '\n"${simv}" +ntb_random_seed=1613')
    before(text, '\n"${simv}" +ntb_random_seed=1613', 'mv -T -n "${work}" "${result}"')


def audit_contract(value):
    require(value["status"] ==
            "SOURCE_ONLY__ADDITIVE_RUNNER_PATH_REPAIR__NO_VCS_NO_ATTEMPT",
            "contract status drift")
    identity = value["identity"]
    require(identity["predecessor_runner_sha256"] == PINS[OLD], "old binding")
    require(identity["successor_runner_sha256"] == PINS[RUNNER], "new binding")
    require(identity["author_static_test_sha256"] == PINS[AUTHOR_TEST],
            "author-test binding")
    delta = value["unique_semantic_delta"]
    require(delta == {
        "field": "python36 executable path",
        "before": "/usr/bin/python3.6",
        "before_lstat": "symbolic link to /usr/libexec/platform-python3.6",
        "after": "/usr/libexec/platform-python3.6",
        "after_lstat": "regular non-symlink executable",
        "before_and_after_resolved_sha256": PINS[PY36_NEW],
        "python_semantics_changed": False,
    }, "semantic delta drift")
    frozen = value["frozen_hardware_and_test"]
    require(frozen["successor_rtl_sha256"] == PINS[RTL], "RTL freeze")
    require(frozen["testbench_sha256"] == PINS[TB], "TB freeze")
    require(frozen["filelist_sha256"] == PINS[FILELIST], "filelist freeze")
    require(frozen["top"] == "tb_m1613_c2_m1609_registered_fault_directed",
            "top freeze")
    require(frozen["seed"] == 1613, "seed freeze")
    require(frozen["vcs_compile_budget_after_future_release"] == 1 and
            frozen["simv_run_budget_after_future_release"] == 1 and
            frozen["all_other_eda_budget"] == 0 and
            frozen["automatic_retry"] is False, "execution budget drift")
    ns = value["namespace_frozen"]
    require(ns["result"] == RESULT_REL and ns["attempt"] == ATTEMPT_REL,
            "namespace drift")
    require(ns["result_exists_at_m1621_authoring"] is False and
            ns["attempt_exists_at_m1621_authoring"] is False,
            "namespace authoring fact drift")
    auth = value["authorization"]
    require(auth == {
        "different_author_m1622_source_hammer": True,
        "future_m1623_release_authoring_after_m1622_pass": True,
        "vcs_now": False, "simv_now": False, "attempt_now": False,
        "all_other_eda": False,
    }, "current authority drift")
    claim = value["claim_boundary"]
    require(claim["source_only"] is True, "source-only boundary")
    for key in ("rtl_behavior_proven", "cycle_performance", "speedup", "area",
                "timing", "power", "energy", "paper_result"):
        require(claim[key] is False, "claim boundary: " + key)


def reject(label, fn, *args):
    try:
        fn(*args)
    except (AssertionError, KeyError, TypeError, ValueError):
        return label
    raise AssertionError("mutation survived: " + label)


def topology_attacks():
    rejected = []
    temp = Path(tempfile.mkdtemp(prefix="m1622_topology."))
    try:
        root = temp / "tree"
        root.mkdir()
        (root / "review.json").write_text("{}\n", encoding="ascii")
        (root / "RUN_COMPLETE.txt").write_text("PASS\n", encoding="ascii")
        rows = []
        for name in ("RUN_COMPLETE.txt", "review.json"):
            rows.append("%s  %s\n" % (sha256(root / name), name))
        (root / "SHA256SUMS").write_text("".join(rows), encoding="ascii")
        (root / "SHA256SUMS.seal.sha256").write_text(
            "%s  SHA256SUMS\n" % sha256(root / "SHA256SUMS"), encoding="ascii")
        verify_tree(root)

        (root / "extra.txt").write_text("hidden\n", encoding="ascii")
        rejected.append(reject("topology_extra_flat", verify_tree, root))
        (root / "extra.txt").unlink()

        (root / "nested").mkdir()
        (root / "nested/payload.txt").write_text("hidden\n", encoding="ascii")
        rejected.append(reject("topology_extra_nested", verify_tree, root))
        shutil.rmtree(str(root / "nested"))

        (root / "link").symlink_to("review.json")
        rejected.append(reject("topology_unlisted_symlink", verify_tree, root))
        (root / "link").unlink()

        original = (root / "SHA256SUMS").read_text(encoding="ascii")
        (root / "SHA256SUMS").write_text(original + original.splitlines(True)[0],
                                           encoding="ascii")
        (root / "SHA256SUMS.seal.sha256").write_text(
            "%s  SHA256SUMS\n" % sha256(root / "SHA256SUMS"), encoding="ascii")
        rejected.append(reject("topology_duplicate_manifest", verify_tree, root))
    finally:
        shutil.rmtree(str(temp))
    return rejected


def build():
    for path, digest in PINS.items():
        require(path.is_file(), "identity path absent: " + str(path))
        if path != PY36_NEW:
            require(not path.is_symlink(), "unexpected identity symlink: " + str(path))
        require(sha256(path) == digest, "identity SHA drift: " + str(path))
    require(PY36_OLD.is_symlink(), "predecessor Python path no longer symlink")
    require(PY36_OLD.resolve() == PY36_NEW, "Python symlink target drift")
    require(sha256(PY36_OLD) == sha256(PY36_NEW), "Python content differs")
    require(stat.S_ISREG(PY36_NEW.lstat().st_mode), "successor Python nonregular")
    require(stat.S_IMODE(RUNNER.stat().st_mode) & stat.S_IXUSR,
            "successor runner not executable")
    verify_file_seal(CONTRACT)
    verify_tree(AUTHOR)
    require(not RESULT.exists() and not RESULT.is_symlink(), "result consumed")
    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink(), "attempt consumed")
    require(not RELEASE.exists() and not RELEASE.is_symlink(), "release premature")
    rows = [row for row in FILELIST.read_text(encoding="utf-8").splitlines()
            if row.strip()]
    require(rows == [NEW_REL, TB_REL], "filelist no longer exact successor-only pair")

    old_text = OLD.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    contract = strict_json(CONTRACT)
    audit_runner(runner, old_text)
    audit_contract(contract)

    attacks = topology_attacks()
    runner_mutations = (
        ("python_symlink_restored", "python36=/usr/libexec/platform-python3.6",
         "python36=/usr/bin/python3.6"),
        ("python_sha_drift", PINS[PY36_NEW], "0" * 64),
        ("rtl_sha_drift", PINS[RTL], "1" * 64),
        ("tb_sha_drift", PINS[TB], "2" * 64),
        ("filelist_sha_drift", PINS[FILELIST], "3" * 64),
        ("review_path_bypass", REVIEW_REL, "reviews/unreviewed"),
        ("release_path_bypass", RELEASE_REL, "contracts/unsealed.json"),
        ("review_status_weaken", "PASS_M1622_M1621_M1613_C2_REGISTERED_FAULT_RUNNER_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT", "PASS"),
        ("release_status_weaken", "AUTHORIZE_ONE_M1621_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_ATTEMPT", "PASS"),
        ("runner_pin_rename", "M1621_EXPECTED_RUNNER_SHA256", "IGNORED_RUNNER_SHA256"),
        ("release_pin_rename", "M1621_EXPECTED_RELEASE_SHA256", "IGNORED_RELEASE_SHA256"),
        ("result_namespace", RESULT_REL, RESULT_REL + "_other"),
        ("attempt_namespace", ATTEMPT_REL, ATTEMPT_REL + "_other"),
        ("compile_budget_two", "VCS_COMPILE vcs_compiles=1", "VCS_COMPILE vcs_compiles=2"),
        ("sim_budget_two", "SIMV_RUN simv_runs=1 seed=1613", "SIMV_RUN simv_runs=2 seed=1613"),
        ("seed_drift", "+ntb_random_seed=1613", "+ntb_random_seed=7"),
        ("pass_token_drift", "legal_descriptor_accepts=1", "legal_descriptor_accepts=0"),
        ("performance_claim", "performance=false", "performance=true"),
        ("predecessor_guard_removed", OLD_REL, "rtl_m214/removed.sv"),
        ("same_uid_gate_removed", "p.stat().st_uid != os.getuid()", "False"),
        ("ancestry_removed", "int(p.name) in ancestry", "False"),
        ("simv_collision_removed", 'blocked = {"vcs", "vcs1", "vlogan", "simv"}',
         'blocked = {"vcs", "vcs1", "vlogan"}'),
        ("atomic_publish_weaken", 'mv -T -n "${work}" "${result}"',
         'mv -T "${work}" "${result}"'),
        ("release_runner_binding_removed", 'r["identity"]["runner_sha256"] == sha(runner)', "True"),
        ("release_contract_binding_removed", 'r["identity"]["source_contract_sha256"] == sha(contract)', "True"),
        ("release_review_binding_removed", 'r["identity"]["hammer_review_sha256"] == sha(hammer)', "True"),
    )
    for label, old, new in runner_mutations:
        attacks.append(reject(label, audit_runner, changed(runner, old, new), old_text))

    insertions = (
        ("second_vcs", '\n"${vcs}" -full64 -sverilog\n'),
        ("second_simv", '\n"${simv}" +ntb_random_seed=1613\n'),
        ("retry_cleanup", '\nrm -rf "${attempt}"\n'),
        ("other_eda", '\n/opt/synopsys/syn/bin/dc_shell\n'),
    )
    for label, suffix in insertions:
        attacks.append(reject(label, audit_runner, runner + suffix, old_text))

    moved = changed(runner, 'mkdir "${attempt}"', 'true # attempt moved')
    moved += '\nmkdir "${attempt}"\n'
    attacks.append(reject("attempt_after_tool", audit_runner, moved, old_text))

    contract_mutations = (
        ("contract_python_changed", ("unique_semantic_delta", "python_semantics_changed"), True),
        ("contract_vcs_now", ("authorization", "vcs_now"), True),
        ("contract_attempt_now", ("authorization", "attempt_now"), True),
        ("contract_retry", ("frozen_hardware_and_test", "automatic_retry"), True),
        ("contract_seed", ("frozen_hardware_and_test", "seed"), 7),
        ("contract_compile_two", ("frozen_hardware_and_test", "vcs_compile_budget_after_future_release"), 2),
        ("contract_speedup", ("claim_boundary", "speedup"), True),
        ("contract_result_namespace", ("namespace_frozen", "result"), "results/other"),
    )
    for label, keys, value in contract_mutations:
        mutant = copy.deepcopy(contract)
        mutant[keys[0]][keys[1]] = value
        attacks.append(reject(label, audit_contract, mutant))

    require(len(attacks) >= 40, "mutation population too small")
    return {
        "schema": "m1622_m1621_m1613_c2_registered_fault_runner_source_hammer_audit_r1_v1",
        "status": "PASS_M1622_M1621_M1613_C2_REGISTERED_FAULT_RUNNER_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT",
        "date": "2026-09-01",
        "score": 98,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 1,
        "identity": {
            "predecessor_runner_sha256": PINS[OLD],
            "successor_runner_sha256": PINS[RUNNER],
            "source_contract_sha256": PINS[CONTRACT],
            "author_test_sha256": PINS[AUTHOR_TEST],
            "author_receipt_review_sha256": PINS[AUTHOR / "review.json"],
            "author_receipt_manifest_sha256": PINS[AUTHOR / "SHA256SUMS"],
            "author_receipt_outer_seal_file_sha256": PINS[AUTHOR / "SHA256SUMS.seal.sha256"],
            "rtl_sha256": PINS[RTL], "testbench_sha256": PINS[TB],
            "filelist_sha256": PINS[FILELIST], "docs359_sha256": PINS[DOCS359],
        },
        "static_audit": {
            "unique_semantic_delta": "PYTHON36_SYMLINK_TO_IDENTICAL_SHA_REGULAR_FILE_ONLY",
            "resolved_python_sha_equal": True,
            "normalized_runner_byte_equal": True,
            "mutation_attacks": len(attacks), "mutation_rejections": len(attacks),
            "topology_extra_flat_rejected": True,
            "topology_extra_nested_rejected": True,
            "topology_unlisted_symlink_rejected": True,
            "topology_duplicate_manifest_rejected": True,
            "vcs_compiles": 0, "simv_runs": 0, "all_other_eda_runs": 0,
        },
        "authorization": {
            "m1623_release_authoring": True,
            "future_vcs_compiles_after_exact_sealed_release": 1,
            "future_simv_runs_after_exact_sealed_release": 1,
            "seed": 1613, "automatic_retry": False,
            "vcs_now": 0, "simv_now": 0, "all_eda_now": 0,
        },
        "p2": {
            "id": "P2_RUNNER_SEAL_CHECK_DOES_NOT_REJECT_UNLISTED_REVIEW_MEMBERS",
            "finding": "The runner validates review manifest members and its outer seal but does not compare the canonical review tree topology with the manifest; an unlisted flat or nested member would not itself block launch.",
            "impact": "Non-blocking because release binds the exact review.json SHA and no unlisted member can change the checked verdict. This M1622 package is exact-topology sealed; the future result hammer must independently reject extra, nested or symlink members before admission.",
        },
        "claim_boundary": {
            "source_hammer": True, "release_authored": False,
            "vcs_executed": False, "simv_executed": False,
            "rtl_behavior_proven": False, "performance": False,
            "speedup": False, "dc": False, "ptpx": False,
            "paper_result": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-audit")
    args = parser.parse_args()
    result = build()
    if args.write_audit:
        path = Path(args.write_audit)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print("PASS M1622 different-author source hammer attacks=%d rejected=%d "
          "python_regular_same_sha=1 attempt=0 vcs=0 simv=0" %
          (result["static_audit"]["mutation_attacks"],
           result["static_audit"]["mutation_rejections"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
