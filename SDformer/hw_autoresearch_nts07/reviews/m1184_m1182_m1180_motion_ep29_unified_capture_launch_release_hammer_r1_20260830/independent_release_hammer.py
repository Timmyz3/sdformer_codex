#!/usr/bin/env python3
"""Read-only M1184 release hammer.  Never contacts remote/GPU or runs capture/EDA."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
LAUNCHER = HW / "scripts/run_m1182_m1180_motion_ep29_unified_capture_remote_one_shot_source.py"
CONTRACT = HW / "contracts/m1182_m1180_motion_ep29_unified_capture_launch_release_r1_20260830.json"
TEST = HW / "tests/test_run_m1182_m1180_motion_ep29_unified_capture_remote_one_shot_source.py"
INVENTORY = HW / "contracts/m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json"
TRANSFER = HW / "contracts/m1182_m1180_motion_ep29_unified_capture_remote_transfer_files_r1_20260830.txt"
AUTHOR = HW / "reviews/m1182_m1180_motion_ep29_unified_capture_launch_release_author_r1_20260830"
SOURCE_AUTHOR = HW / "reviews/m1180_motion_checkpoint_parametric_unified_capture_r2_author_r1_20260830"
SOURCE_HAMMER = HW / "reviews/m1181_m1180_motion_checkpoint_parametric_unified_capture_r2_source_hammer_r1_20260830"
M1175 = HW / "reviews/m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    LAUNCHER: "04267c86171937a1e12ef3282319193ad9f42aaba316a55e125c8e0b06e80e17",
    CONTRACT: "46450015bcdb3b8c0a32ccd7aaba68a78abf923705a133147202283e7bc7220f",
    TEST: "946f1a9ba5b98ba20294d4ba2f4ccd3c500169ad000796f7b3594b1b131f8731",
    INVENTORY: "de6ff2b13719580b77674b44f7414a7798cffd3f7cde5e80e88ff3ea8f0d97ae",
    TRANSFER: "ec53838c3f6961a9b1143ba96d6d4452980bc3c569948f16ca7842a0a08cbc1b",
    AUTHOR / "SHA256SUMS": "33dcd8b36be9b30e6159a3eb41ee55c4d301f0d0cc891c8bf76b23a4b991e9ee",
    AUTHOR / "SHA256SUMS.seal.sha256": "26fc2bf725ab2278ef7cef11fc8681dae63c469badeb638fbb33f2f839756907",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def strict_json(path: Path):
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    def reject(value):
        raise AssertionError("non-standard JSON token: " + value)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=reject)


def verify_double_seal(directory: Path) -> dict[str, str]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "missing manifest")
    require(outer.is_file() and not outer.is_symlink(), "missing outer seal")
    outer_match = re.fullmatch(r"([0-9a-f]{64})  SHA256SUMS\n", outer.read_text())
    require(outer_match is not None and outer_match.group(1) == sha(manifest), "outer seal mismatch")
    rows = {}
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/][^\n]*)", line)
        require(match is not None, "malformed manifest line")
        name = match.group(2)
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and ".." not in rel.parts and name not in rows,
                "unsafe/duplicate manifest member")
        member = directory / name
        require(member.is_file() and not member.is_symlink() and sha(member) == match.group(1),
                "manifest member mismatch: " + name)
        rows[name] = match.group(1)
    actual = sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*")
                    if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    require(sorted(rows) == actual, "manifest population mismatch")
    return rows


def main() -> None:
    checks = []
    for path, expected in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(), "missing/non-regular exact artifact")
        require(sha(path) == expected, "exact SHA mismatch: " + str(path))
    checks.append("exact_launcher_contract_test_inventory_transfer_author_docs359")

    verify_double_seal(AUTHOR)
    source_author = verify_double_seal(SOURCE_AUTHOR)
    source_hammer = verify_double_seal(SOURCE_HAMMER)
    m1175 = verify_double_seal(M1175)
    require(sha(SOURCE_AUTHOR / "SHA256SUMS") == "1363a7256655b8b64874099b6de7d4ac87a93ffe5712afa5fdfcb94371393547", "source author manifest")
    require(sha(SOURCE_AUTHOR / "SHA256SUMS.seal.sha256") == "d7bc3196af16c8f97fbc07bd11ac477f8b942b222042b372e459843a6cfe7e36", "source author outer")
    require(source_hammer.get("review.json") == "2dc8f5b39c990d67fd73d9f5fc8ff5167b17c6759d93781ee8dbdad128d05330", "source hammer review")
    require(sha(SOURCE_HAMMER / "SHA256SUMS") == "8c483b73ee4623f1a1876f55b710e4292e3f21530907b7084f41efa71398c837", "source hammer manifest")
    require(sha(SOURCE_HAMMER / "SHA256SUMS.seal.sha256") == "9b85611c24595f70d4e08b12522294c1a98c53a0a5981cce91873af4d1c1499b", "source hammer outer")
    require(m1175.get("review.json") == "8b83690b8b1130d2335bb118d35645ae4d172740966ab69c6fcea9bc8b5d307b", "M1175 review")
    checks.append("recursive_author_source_hammer_m1175_seals")

    contract = strict_json(CONTRACT)
    inventory = strict_json(INVENTORY)
    require(contract["schema"] == "m1180_motion_checkpoint_parametric_unified_capture_launch_r1_v1", "contract schema")
    require(contract["status"] == "M1175_AND_M1181_BOUND__ONE_M1180_GPU_RUN_AUTHORIZED", "contract status")
    require(contract["release_hammer_gate"] == {
        "exact_remote_launch_authorized_now": False,
        "path": "hw_autoresearch_nts07/reviews/m1184_m1182_m1180_motion_ep29_unified_capture_launch_release_hammer_r1_20260830",
        "present_now": False,
        "required_schema": "m1184_m1182_m1180_motion_unified_capture_launch_release_hammer_r1_v1",
        "required_status": "PASS",
    }, "release hammer preregistration")
    require(contract["claim_boundary"]["remote_execution_authorized_now"] is False, "inert contract")
    checks.append("contract_schema_status_inert_gate")

    rows = inventory["dependencies"]
    require(inventory["schema"] == "m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_v1", "inventory schema")
    require(inventory["status"] == "COMPLETE_EXACT_REMOTE_PREFLIGHT_INVENTORY", "inventory status")
    require(len(rows) == 95 and Counter(r["disposition"] for r in rows) == {"transfer_required": 40, "remote_existing_hash_verify": 55}, "inventory counts")
    require(len(inventory["required_labels"]) == 95 and set(inventory["required_labels"]) == {r["label"] for r in rows}, "inventory labels")
    paths = []
    for row in rows:
        rel = PurePosixPath(row["path"])
        require(not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() == row["path"], "unsafe inventory path")
        require(isinstance(row["size_bytes"], int) and not isinstance(row["size_bytes"], bool) and row["size_bytes"] > 0, "bad size")
        require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "bad SHA")
        paths.append(row["path"])
        local = ROOT / row["path"]
        if row["disposition"] == "transfer_required" or local.exists():
            require(local.is_file() and not local.is_symlink(), "local dependency regularity")
            require(local.stat().st_size == row["size_bytes"] and sha(local) == row["sha256"], "dependency path/size/SHA")
    require(len(paths) == len(set(paths)), "duplicate dependency path")
    require(Counter(r["label"].startswith("frozen_data_") for r in rows)[True] == 40, "frozen row count")
    checks.append("inventory_95_40_transfer_55_verify_path_size_sha")

    cohort = contract["cohort"]["samples"]
    frozen = {r["path"]: r for r in rows if r["label"].startswith("frozen_data_")}
    require(len(cohort) == len(frozen) == 40, "forty data binding")
    require([r["global_sample_id"] for r in cohort] == list(range(40)), "cohort order")
    for row in cohort:
        dep = frozen[row["path"]]
        require(dep["disposition"] == "remote_existing_hash_verify" and
                dep["size_bytes"] == row["bytes"] and dep["sha256"] == row["sha256"], "frozen cross-bind")
    checks.append("forty_data_frozen_order_and_cross_binding")

    transfer_lines = TRANSFER.read_text(encoding="utf-8").splitlines()
    require(len(transfer_lines) == 42 and len(set(transfer_lines)) == 42 and transfer_lines == sorted(transfer_lines), "transfer line count/order")
    expected_transfer = sorted([r["path"] for r in rows if r["disposition"] == "transfer_required"] + [INVENTORY.relative_to(ROOT).as_posix(), TRANSFER.relative_to(ROOT).as_posix()])
    require(transfer_lines == expected_transfer, "files-from population")
    for line in transfer_lines:
        rel = PurePosixPath(line)
        require(rel.as_posix() == line and not rel.is_absolute() and ".." not in rel.parts,
                "files-from traversal")
        require(not any(ch.isspace() for ch in line) and "\\" not in line and not line.startswith("-") and "//" not in line,
                "files-from escaping/option ambiguity")
    exact_rsync = "rsync -ah --partial --append-verify --relative --files-from=hw_autoresearch_nts07/contracts/m1182_m1180_motion_ep29_unified_capture_remote_transfer_files_r1_20260830.txt -e 'ssh -p 10037' ./ root@ssh.sd5ai.scnet.cn:/root/private_data/work/sdformer_codex/SDformer/"
    require(inventory["exact_transfer_command"] == exact_rsync and contract["remote_dependency_closure"]["transfer_command"] == exact_rsync, "exact rsync")
    checks.append("files_from_42_relative_no_escape_or_traversal")

    absent = {r["label"]: r for r in rows if not (ROOT / r["path"]).exists()}
    require(set(absent) == {"m1171_result_RUN_COMPLETE.txt", "m1171_result_SHA256SUMS", "m1171_result_SHA256SUMS.seal.sha256", "m1171_result_e0_e8_rebind_targets.json", "m1171_result_final_checkpoint_selection.json", "m1171_result_five_checkpoint_metrics.csv", "ep29_config", "ep29_checkpoint"}, "unexpected local-absent remote set")
    remote_readback = strict_json(M1175 / "remote_readback.json")
    m1175_review = strict_json(M1175 / "review.json")
    require(m1175_review["status"] == "PASS" and m1175_review["selection"]["epoch"] == 29, "M1175 ep29 admission")
    require(absent["ep29_checkpoint"]["sha256"] == remote_readback["epochs"]["29"]["checkpoint_sha256"] and absent["ep29_checkpoint"]["size_bytes"] == remote_readback["epochs"]["29"]["checkpoint_size_bytes"], "remote checkpoint evidence")
    require(absent["ep29_config"]["sha256"] == remote_readback["configuration"]["sha256"] and absent["ep29_config"]["size_bytes"] == remote_readback["configuration"]["size_bytes"], "remote config evidence")
    require(absent["m1171_result_SHA256SUMS"]["sha256"] == m1175_review["remote_result_manifest_sha256"] and absent["m1171_result_SHA256SUMS.seal.sha256"]["sha256"] == m1175_review["remote_result_outer_file_sha256"], "remote result seal evidence")
    checks.append("eight_remote_only_rows_backed_by_m1175_and_reverified_pre_attempt")

    source = LAUNCHER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    preflight = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preflight")
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute_once")
    pf = ast.get_source_segment(source, preflight) or ""
    ex = ast.get_source_segment(source, execute) or ""
    for token in ("remote interpreter path mismatch", "remote interpreter version mismatch", "remote repository cwd mismatch", "DOCS359_REL", "validate_release_hammer", "validate_dependency_inventory", "not os.path.lexists(policy.repo / ATTEMPT_REL)", "gpu_compute_pids() == []", "running_legacy_watchers() == []", "prove_lease_available"):
        require(token in pf, "preflight missing: " + token)
    require(ex.count("runner(command, policy.repo)") == 1 and "while " not in ex and "for " not in ex, "one child/no retry")
    require(ex.index("canonical_verify_double_seal(policy.repo / RESULT_REL)") < ex.index("write_production_log"), "result seal before log")
    require("fresh_result_hammer_required" in CONTRACT.read_text() and "shell=True" not in source, "result hammer/no shell")
    checks.append("remote_preflight_identity_namespace_lease_gpu_legacy_one_shot_result_seal")

    output = {
        "schema": "m1184_m1182_m1180_motion_unified_capture_release_hammer_output_r1_v1",
        "status": "PASS",
        "checks": checks,
        "counts": {"check_groups": len(checks), "inventory_rows": 95,
                   "transfer_required": 40, "remote_existing_hash_verify": 55,
                   "frozen_data": 40, "files_from_lines": 42,
                   "controlled_tests": 10},
        "no_actions": {"remote": True, "gpu": True, "capture": True, "eda": True},
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
