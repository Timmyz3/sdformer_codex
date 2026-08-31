#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Non-launching static and negative self-test for additive M1080 source."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_one_shot_r1.py"
OLD_RUNNER = HW / "dc_handoff/scripts/run_m1070_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_one_shot_r1.py"
CONTRACT = HW / "contracts/m1080_c2_k1_reset_hygiene_dc_mapped_vcs_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1080_c2_k1_reset_hygiene_dc_mapped_vcs_one_shot_release_r1_20260830.json"
M1071 = HW / "reviews/m1071_m1070_c2_k1_reset_hygiene_one_shot_release_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DC_SHELL = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")
DC_TARGET = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")

RUNNER_SHA = "6ca208ed899337b8acf0433e8d28af7da1afe9855f779bebb24e0b5da1735836"
CONTRACT_SHA = "b77cbbd958ef737291dffa7861e1411377a96a5d51a5612461366bddcf3cbf67"
CONTRACT_OUTER_SHA = "a568aaadc0021839c54977c1b9dccc055fd32c3aab733874418b02ad85900fea"
RELEASE_SHA = "478097925d21093dfd66e5fac5d799daaac415f0229e7c8013ebfcc06b1028ee"
RELEASE_OUTER_SHA = "831023d2f5e96cbd50a1f51e1167da3fa717b33cda2e6971b262450df8cf612e"
M1071_OUTER_SHA = "812a1543dc9c198ca504768cf7e4bfd5ef3941094438a1ebc8cc32cd709f3725"
DC_PAYLOAD_SHA = "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rejected(call: Callable[[], Any]) -> bool:
    try:
        call()
    except (RuntimeError, OSError, ValueError):
        return True
    return False


def load_selected_production_definitions() -> dict[str, Any]:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    names = {"GateFailure", "fail", "sha", "expect_exact_symlink_payload",
             "verify_sidecar"}
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in names]
    require({node.name for node in selected} == names,
            "production helper definitions missing")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "Path": Path, "os": os, "stat": stat, "hashlib": hashlib,
        "HW_ROOT": HW,
    }
    exec(compile(module, str(RUNNER), "exec"), namespace)
    return namespace


def sidecar_identity(path: Path, expected_primary: str,
                     expected_outer_file: str) -> dict[str, Any]:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    primary = side.read_text(encoding="utf-8").strip().split()
    outer_tokens = outer.read_text(encoding="utf-8").strip().split()
    expected_name = path.relative_to(HW).as_posix()
    require(primary == [sha(path), expected_name], "primary sidecar token drift")
    require(outer_tokens == [sha(side), expected_name + ".sha256"],
            "outer sidecar token drift")
    require(sha(path) == expected_primary and sha(outer) == expected_outer_file,
            "sidecar identity drift")
    return {
        "primary_sha256": sha(path),
        "sidecar_sha256": sha(side),
        "outer_seal_file_sha256": sha(outer),
        "exact_primary_token": expected_name,
        "exact_outer_token": expected_name + ".sha256",
    }


def function_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(text, node)
            require(segment is not None, "cannot extract function " + name)
            return segment
    raise RuntimeError("function absent: " + name)


def normalize_m1080(text: str) -> str:
    return text.replace("m1080", "m1070").replace("M1080", "M1070").replace(
        "m1081", "m1071").replace("M1081", "M1071")


def main() -> dict[str, Any]:
    identity = {
        "runner_sha256": sha(RUNNER),
        "contract_sha256": sha(CONTRACT),
        "release_sha256": sha(RELEASE),
        "m1071_outer_seal_file_sha256": sha(M1071 / "SHA256SUMS.seal.sha256"),
        "docs359_sha256": sha(DOCS359),
    }
    require(identity == {
        "runner_sha256": RUNNER_SHA,
        "contract_sha256": CONTRACT_SHA,
        "release_sha256": RELEASE_SHA,
        "m1071_outer_seal_file_sha256": M1071_OUTER_SHA,
        "docs359_sha256": DOCS359_SHA,
    }, "top-level source identity drift")
    contract_sidecar = sidecar_identity(CONTRACT, CONTRACT_SHA, CONTRACT_OUTER_SHA)
    release_sidecar = sidecar_identity(RELEASE, RELEASE_SHA, RELEASE_OUTER_SHA)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    m1071 = json.loads((M1071 / "review.json").read_text(encoding="utf-8"))
    require(contract["launch_now"] is False and contract["max_attempts_now"] == 0
            and release["launch_now"] is False
            and m1071["status"] ==
            "STOP_M1071_M1070_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER"
            and m1071["authorization"][
                "one_m1070_dc_then_mapped_vcs_attempt"] is False,
            "nonlaunching/M1071 STOP boundary drift")

    ns = load_selected_production_definitions()
    helper = ns["expect_exact_symlink_payload"]
    helper(DC_SHELL, "snps_shell", DC_TARGET, DC_PAYLOAD_SHA)
    actual_info = os.lstat(DC_SHELL)
    actual_symlink = {
        "lstat_is_symlink": stat.S_ISLNK(actual_info.st_mode),
        "readlink": os.readlink(DC_SHELL),
        "resolved_target": str(DC_SHELL.resolve(strict=True)),
        "target_regular_nonsymlink": (
            stat.S_ISREG(os.lstat(DC_TARGET).st_mode) and not DC_TARGET.is_symlink()
        ),
        "payload_sha256": sha(DC_TARGET),
    }
    require(actual_symlink == {
        "lstat_is_symlink": True,
        "readlink": "snps_shell",
        "resolved_target": str(DC_TARGET),
        "target_regular_nonsymlink": True,
        "payload_sha256": DC_PAYLOAD_SHA,
    }, "actual DC launcher identity drift")

    symlink_attacks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="m1080_symlink_attacks.") as raw:
        root = Path(raw)
        payload = root / "snps_shell"
        payload.write_bytes(b"frozen-payload")
        payload_sha = sha(payload)
        link = root / "dc_shell"
        link.symlink_to("snps_shell")
        helper(link, "snps_shell", payload, payload_sha)

        regular = root / "regular_dc_shell"
        regular.write_bytes(b"frozen-payload")
        symlink_attacks["regular_launcher_rejected"] = rejected(
            lambda: helper(regular, "snps_shell", payload, payload_sha)
        )
        wrong_target = root / "wrong_shell"
        wrong_target.write_bytes(b"frozen-payload")
        wrong_link = root / "wrong_link"
        wrong_link.symlink_to("wrong_shell")
        symlink_attacks["wrong_readlink_rejected"] = rejected(
            lambda: helper(wrong_link, "snps_shell", wrong_target, payload_sha)
        )
        other = root / "other"
        other.write_bytes(b"frozen-payload")
        symlink_attacks["wrong_resolved_target_rejected"] = rejected(
            lambda: helper(link, "snps_shell", other, payload_sha)
        )
        symlink_attacks["wrong_payload_sha_rejected"] = rejected(
            lambda: helper(link, "snps_shell", payload, "0" * 64)
        )
        dangling = root / "dangling"
        dangling.symlink_to("snps_shell_missing")
        symlink_attacks["dangling_target_rejected"] = rejected(
            lambda: helper(dangling, "snps_shell", root / "snps_shell_missing",
                           payload_sha)
        )
    require(all(symlink_attacks.values()), "symlink identity attack escaped")

    # Exercise the unchanged production sidecar validator with ephemeral files
    # under HW_ROOT, so relative-token behavior is exact.
    verify_sidecar = ns["verify_sidecar"]
    sidecar_attacks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix=".m1080_sidecar_attacks.",
                                     dir=str(HW / "contracts")) as raw:
        root = Path(raw)
        item = root / "x.json"
        side = Path(str(item) + ".sha256")
        outer = Path(str(item) + ".sha256.seal.sha256")
        item.write_text("{}\n", encoding="utf-8")
        exact_name = item.relative_to(HW).as_posix()

        def install(primary_name: str, outer_name: str,
                    primary_extra: str = "", outer_extra: str = "") -> str:
            side.write_text(f"{sha(item)}  {primary_name}{primary_extra}\n",
                            encoding="utf-8")
            outer.write_text(f"{sha(side)}  {outer_name}{outer_extra}\n",
                             encoding="utf-8")
            return sha(outer)

        good_outer = install(exact_name, exact_name + ".sha256")
        verify_sidecar(item, good_outer)
        install(item.name, exact_name + ".sha256")
        sidecar_attacks["basename_only_rejected"] = rejected(
            lambda: verify_sidecar(item, sha(outer))
        )
        install(exact_name + ".suffix", exact_name + ".sha256")
        sidecar_attacks["arbitrary_suffix_rejected"] = rejected(
            lambda: verify_sidecar(item, sha(outer))
        )
        install("contracts/../" + exact_name, exact_name + ".sha256")
        sidecar_attacks["path_traversal_rejected"] = rejected(
            lambda: verify_sidecar(item, sha(outer))
        )
        install(exact_name, exact_name + ".sha256", primary_extra="  extra")
        sidecar_attacks["extra_primary_token_rejected"] = rejected(
            lambda: verify_sidecar(item, sha(outer))
        )
        install(exact_name, exact_name + ".sha256", outer_extra="  extra")
        sidecar_attacks["extra_outer_token_rejected"] = rejected(
            lambda: verify_sidecar(item, sha(outer))
        )
    require(all(sidecar_attacks.values()), "sidecar attack escaped")

    # Exact M1070 flow preservation: the execution, release-chain and failure
    # functions are byte-identical after milestone namespace normalization.
    preserved_functions = {}
    for name in ("release_chain_gate", "run_flow", "quarantine_failure"):
        preserved_functions[name] = (
            normalize_m1080(function_source(RUNNER, name))
            == function_source(OLD_RUNNER, name)
        )
    require(all(preserved_functions.values()), "M1070 production flow changed")
    runner_text = RUNNER.read_text(encoding="utf-8")
    preserved_static = {
        "anchors_exact": "ANCHORS = [259, 737, 3153, 7569, 14]" in runner_text,
        "attempt_before_first_run_logged": (
            runner_text.index("ATTEMPT.mkdir()") <
            runner_text.index("dc_rc = run_logged") <
            runner_text.index("vcs_rc = run_logged")
        ),
        "forbidden_initreg_literal_absent": "initreg" not in runner_text.lower(),
        "fresh_arch_mode0": '"ELAB_PARAMETERS": "ARCH_MODE=0"' in runner_text,
        "five_case_loop": "for case_id, anchor in enumerate(ANCHORS):" in runner_text,
        "saif_ptpx_false": '"saif_files": 0' in runner_text and '"ptpx_runs": 0' in runner_text,
        "m1071_stop_outer_pinned": M1071_OUTER_SHA in runner_text,
        "m1081_pass_required":
            "PASS_M1081_M1080_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER" in runner_text,
    }
    require(all(preserved_static.values()), "preserved static flow invariant drift")

    forbidden_paths = [
        HW / "results/m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830",
        HW / "results/.m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_attempt_consumed",
    ]
    require(not any(path.exists() for path in forbidden_paths),
            "M1080 result/attempt already exists")
    prelaunch_env = os.environ.copy()
    prelaunch_env.pop("M1080_EXPECTED_RUNNER_SHA256", None)
    prelaunch_env.pop("M1080_EXPECTED_M1081_OUTER_SHA256", None)
    prelaunch = subprocess.run(
        [str(PYTHON), str(RUNNER)], text=True, capture_output=True,
        timeout=120, check=False, env=prelaunch_env,
    )
    (HERE / "prelaunch.stdout.txt").write_text(prelaunch.stdout, encoding="utf-8")
    (HERE / "prelaunch.stderr.txt").write_text(prelaunch.stderr, encoding="utf-8")
    prelaunch_gate = {
        "return_code": prelaunch.returncode,
        "reached_missing_runner_pin":
            "caller must pin exact M1080 runner SHA" in prelaunch.stderr,
        "attempt_absent_after": not forbidden_paths[1].exists(),
        "result_absent_after": not forbidden_paths[0].exists(),
        "no_failure_quarantine": not any(
            (HW / "results").glob(
                "m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*.quarantine"
            )
        ),
    }
    require(prelaunch_gate == {
        "return_code": 3, "reached_missing_runner_pin": True,
        "attempt_absent_after": True, "result_absent_after": True,
        "no_failure_quarantine": True,
    }, "nonlaunching source preflight boundary drift")
    require(sha(DOCS359) == DOCS359_SHA, "docs359 changed")

    return {
        "schema": "m1080_m1071_c2_k1_reset_hygiene_additive_source_selftest_v1",
        "status": "PASS_M1080_ADDITIVE_SOURCE_SELFTEST__M1081_REQUIRED_NO_EDA",
        "identity": identity,
        "contract_sidecar": contract_sidecar,
        "release_sidecar": release_sidecar,
        "actual_dc_shell_identity": actual_symlink,
        "symlink_negative_attacks_rejected": symlink_attacks,
        "sidecar_negative_attacks_rejected": sidecar_attacks,
        "m1070_flow_preserved": preserved_functions,
        "preserved_static_invariants": preserved_static,
        "nonlaunching_preflight": prelaunch_gate,
        "claim_boundary": {
            "source_ready_for_independent_m1081": True,
            "launch_now": False,
            "attempt_consumed": False,
            "real_eda_launched": False,
            "dc": False,
            "mapped_vcs": False,
            "power": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
        },
    }


if __name__ == "__main__":
    result = main()
    temporary = HERE / ".mechanical_checks.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(HERE / "mechanical_checks.json")
    print(result["status"])
