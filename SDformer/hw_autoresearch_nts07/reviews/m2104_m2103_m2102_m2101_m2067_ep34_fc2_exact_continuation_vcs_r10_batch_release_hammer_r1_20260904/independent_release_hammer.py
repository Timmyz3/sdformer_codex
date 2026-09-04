#!/opt/anaconda3/bin/python3.12
"""Offline M2104 hammer for the M2103 R10 batch launch release.

This program performs only filesystem, JSON, SHA-256, AST, /proc-presence,
and read-only git checks.  It must never import or execute the production
runner, query a license server, invoke VCS/simv/EDA, or use a GPU.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
RELEASE = HW / (
    "contracts/m2103_m2102_m2101_m2067_ep34_fc2_exact_continuation_vcs_"
    "r10_batch_launch_release_r1_20260904.json")
CONTRACT = HW / (
    "contracts/m2101_m2067_ep34_fc2_exact_continuation_vcs_source_contract_"
    "r10_codex_batch_20260904.json")
RUNNER = HW / (
    "dc_handoff/scripts/run_m2067_ep34_fc2_exact_continuation_vcs_one_shot_"
    "codex_batch_r10_20260904.py")
PARSER = HW / (
    "system_simulator/scripts/parse_m2067_ep34_fc2_exact_continuation_vcs_"
    "codex_batch_r10_20260904.py")
FILELIST = HW / (
    "dc_handoff/filelists/iscas_m2067_ep34_fc2_exact_continuation_vcs_"
    "codex_batch_r10_20260904.f")
M2102 = HW / (
    "reviews/m2102_m2101_m2067_ep34_fc2_exact_continuation_vcs_source_r10_"
    "batch_hammer_r1_20260904")
R9_ATTEMPT = HW / (
    "results/.m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_"
    "attempt_consumed")
R9_RESULT = HW / (
    "results/m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_"
    "20260904")
R9_FAILURE = HW / (
    "results/m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_"
    "20260904.failed_or_incomplete.quarantine")
R10_ATTEMPT = HW / (
    "results/.m2067_ep34_fc2_exact_continuation_vcs_r10_codex_batch_"
    "attempt_consumed")
R10_RESULT = HW / (
    "results/m2067_ep34_fc2_exact_continuation_vcs_r10_codex_batch_"
    "20260904")
R10_FAILURE = HW / (
    "results/m2067_ep34_fc2_exact_continuation_vcs_r10_codex_batch_"
    "20260904.failed_or_incomplete.quarantine")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_RELEASE_SHA256 = (
    "dd417b8f9e35ad315af70ee6f9b839168c35af040ee9889fb9d620ff9e8237a8")
EXPECTED_DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
EXPECTED_AUTHORIZATION = {
    "license_preflight_lmstat": 1,
    "vcs_compiles": 1,
    "simv_runs": 1,
    "workloads_inside_single_simv": 960,
    "automatic_retry": False,
    "all_other_eda_runs": 0,
}
EXPECTED_CLAIM_BOUNDARY = {
    "source_only": False,
    "directed_weights": True,
    "real_ep34_activity_and_sign_descriptors": True,
    "component_workloads": True,
    "full_fc_wall_time": False,
    "same_area": False,
    "power": False,
    "energy": False,
    "system_speedup": False,
    "paper_admitted": False,
}


class Failure(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def exact_regular(path: Path, digest: str | None = None) -> str:
    require(path.is_file() and not path.is_symlink(), "not regular: " + str(path))
    actual = sha256(path)
    if digest is not None:
        require(actual == digest, "identity drift: " + str(path))
    return actual


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    value = json.loads(
        path.read_text(), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            Failure("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root is not an object: " + str(path))
    return value


def same_json(actual, expected, label: str) -> None:
    require(type(actual) is type(expected), label + " JSON type drift")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), label + " key-set drift")
        for key in expected:
            same_json(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), label + " list length drift")
        for index, (left, right) in enumerate(zip(actual, expected)):
            same_json(left, right, label + "[" + str(index) + "]")
    else:
        require(actual == expected, label + " value drift")


def verify_double_sealed_file(path: Path, digest: str) -> dict:
    exact_regular(path, digest)
    inner = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    exact_regular(inner)
    exact_regular(outer)
    require(
        inner.read_text() == digest + "  " + path.name + "\n",
        "inner sidecar content drift: " + str(path))
    require(
        outer.read_text() == sha256(inner) + "  " + inner.name + "\n",
        "outer sidecar content drift: " + str(path))
    return strict_json(path)


def verify_sealed_directory(root: Path, manifest_digest: str,
                            outer_digest: str) -> dict[str, str]:
    require(root.is_dir() and not root.is_symlink(), "bad sealed dir: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact_regular(manifest, manifest_digest)
    exact_regular(outer, outer_digest)
    require(
        outer.read_text() == manifest_digest + "  SHA256SUMS\n",
        "outer directory seal content drift: " + str(root))
    mapping = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, "manifest syntax: " + str(root))
        digest, name = fields
        relative = Path(name.lstrip("*"))
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "unsafe manifest path: " + str(relative))
        key = relative.as_posix()
        require(key not in mapping, "duplicate manifest member: " + key)
        exact_regular(root / relative, digest)
        mapping[key] = digest
    members = list(root.rglob("*"))
    require(not any(path.is_symlink() for path in members),
            "symlink in sealed directory: " + str(root))
    actual = {
        path.relative_to(root).as_posix() for path in members
        if path.is_file() and path.name not in {
            "SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    require(actual == set(mapping), "non-exhaustive sealed directory: " + str(root))
    return mapping


def verify_self_sealed_directory(root: Path) -> dict[str, str]:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact_regular(manifest)
    exact_regular(outer)
    return verify_sealed_directory(root, sha256(manifest), sha256(outer))


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    hits = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(hits) == 1, "function cardinality drift: " + name)
    return hits[0]


def named_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [item for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name) and item.func.id == name]


def attribute_calls(node: ast.AST, owner: str, name: str) -> list[ast.Call]:
    return [item for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and isinstance(item.func.value, ast.Name)
            and item.func.value.id == owner and item.func.attr == name]


def tool_call_is_outside_loop(root: ast.AST, target: ast.Call) -> bool:
    for node in ast.walk(root):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            if target in list(ast.walk(node)):
                return False
    return True


def git_head_docs359() -> tuple[str, bool]:
    prefix = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--show-prefix"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()
    object_path = prefix.strip("\n") + DOCS359.relative_to(REPO).as_posix()
    head = subprocess.run(
        ["git", "-C", str(REPO), "show", "HEAD:" + object_path],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    diff = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--",
         str(DOCS359)], check=False)
    return sha256_bytes(head), diff.returncode == 0


def main() -> int:
    release = verify_double_sealed_file(RELEASE, EXPECTED_RELEASE_SHA256)
    contract_digest = exact_regular(CONTRACT)
    contract = verify_double_sealed_file(CONTRACT, contract_digest)
    runner_digest = exact_regular(RUNNER)
    parser_digest = exact_regular(PARSER)
    filelist_digest = exact_regular(FILELIST)

    same_json(set(release), {
        "schema", "milestone", "date", "status", "purpose", "identity",
        "authorization", "claim_boundary"}, "release top-level fields")
    same_json(
        release["schema"],
        "m2103_m2102_m2101_m2067_ep34_fc2_exact_continuation_vcs_"
        "r10_batch_launch_release_r1_v1", "release.schema")
    same_json(release["milestone"], "M2103", "release.milestone")
    same_json(release["date"], "2026-09-04", "release.date")
    same_json(
        release["status"],
        "AUTHORIZE_ONE_M2101_R10_BATCH_VCS_COMPILE_AND_SIMV",
        "release.status")
    same_json(
        release["purpose"],
        "Authorize one exact R10 single-simv batch after the independent M2102 "
        "source hammer and the sealed no-retry R9 failure. This release does not "
        "authorize retries or any other EDA run.", "release.purpose")

    m2102_manifest_digest = sha256(M2102 / "SHA256SUMS")
    m2102_outer_digest = sha256(M2102 / "SHA256SUMS.seal.sha256")
    m2102_mapping = verify_sealed_directory(
        M2102, m2102_manifest_digest, m2102_outer_digest)
    m2102_review_digest = exact_regular(M2102 / "review.json")
    require(m2102_mapping.get("review.json") == m2102_review_digest,
            "M2102 review is not sealed")
    m2102 = strict_json(M2102 / "review.json")
    same_json(
        m2102["status"],
        "PASS_M2102_M2101_R10_BATCH_SOURCE_HAMMER__AUTHORIZE_M2103_RELEASE_ONLY",
        "M2102.status")
    same_json(m2102["authorization"], {
        "m2103_release_authoring": 1,
        "vcs_execution": 0,
        "license_queries": 0,
        "automatic_retry": False,
    }, "M2102.authorization")

    r9_manifest_digest = sha256(R9_FAILURE / "SHA256SUMS")
    r9_outer_digest = sha256(R9_FAILURE / "SHA256SUMS.seal.sha256")
    r9_mapping = verify_sealed_directory(
        R9_FAILURE, r9_manifest_digest, r9_outer_digest)
    r9_failure_digest = exact_regular(R9_FAILURE / "failure.json")
    require(r9_mapping.get("failure.json") == r9_failure_digest,
            "R9 failure JSON is not sealed")
    r9_failure = strict_json(R9_FAILURE / "failure.json")
    same_json(r9_failure["status"], "FAILED_DO_NOT_CITE_NO_RETRY",
              "R9 failure.status")
    same_json(r9_failure["automatic_retry"], False,
              "R9 failure.automatic_retry")
    same_json(r9_failure["completed_slots"], 163,
              "R9 failure.completed_slots")
    same_json(r9_failure["current_slot"], 163, "R9 failure.current_slot")
    same_json(r9_failure["simv_runs"], 164, "R9 failure.simv_runs")
    require(not os.path.lexists(R9_RESULT), "R9 success namespace exists")

    verify_self_sealed_directory(R9_ATTEMPT)
    r9_owner = strict_json(R9_ATTEMPT / "owner.json")
    require(type(r9_owner.get("pid")) is int and r9_owner["pid"] > 1,
            "R9 owner PID invalid")
    require(not (Path("/proc") / str(r9_owner["pid"])).exists(),
            "R9 owner PID remains live")
    for key in ("nonce", "runner_sha256"):
        same_json(r9_failure[key if key != "nonce" else "owner_nonce"],
                  r9_owner[key], "R9 owner binding." + key)

    predecessor = m2102["reviewed_predecessor_failure_identity"]
    same_json(predecessor["failure_json_sha256"], r9_failure_digest,
              "M2102 predecessor.failure_json_sha256")
    same_json(predecessor["manifest_sha256"], r9_manifest_digest,
              "M2102 predecessor.manifest_sha256")
    same_json(predecessor["outer_file_sha256"], r9_outer_digest,
              "M2102 predecessor.outer_file_sha256")
    same_json(predecessor["attempt_owner_json_sha256"],
              sha256(R9_ATTEMPT / "owner.json"),
              "M2102 predecessor.attempt_owner_json_sha256")
    same_json(predecessor["owner_pid"], r9_owner["pid"],
              "M2102 predecessor.owner_pid")
    same_json(predecessor["owner_pid_dead"], True,
              "M2102 predecessor.owner_pid_dead")
    same_json(predecessor["completed_slots"], 163,
              "M2102 predecessor.completed_slots")
    same_json(predecessor["failed_slot"], 163,
              "M2102 predecessor.failed_slot")
    same_json(predecessor["simv_runs"], 164,
              "M2102 predecessor.simv_runs")
    same_json(predecessor["automatic_retry"], False,
              "M2102 predecessor.automatic_retry")

    expected_identity = {
        "runner_sha256": runner_digest,
        "parser_sha256": parser_digest,
        "filelist_sha256": filelist_digest,
        "contract_sha256": contract_digest,
        "m2102_review_sha256": m2102_review_digest,
        "m2102_manifest_sha256": m2102_manifest_digest,
        "m2102_outer_file_sha256": m2102_outer_digest,
        "r9_failure_json_sha256": r9_failure_digest,
        "r9_failure_manifest_sha256": r9_manifest_digest,
        "r9_failure_outer_file_sha256": r9_outer_digest,
    }
    same_json(release["identity"], expected_identity, "release.identity")
    same_json(release["authorization"], EXPECTED_AUTHORIZATION,
              "release.authorization")
    same_json(release["claim_boundary"], EXPECTED_CLAIM_BOUNDARY,
              "release.claim_boundary")

    source_identity = m2102["reviewed_source_identity"]
    for key in ("runner_sha256", "parser_sha256", "filelist_sha256",
                "contract_sha256"):
        same_json(source_identity[key], expected_identity[key],
                  "M2102 source identity." + key)

    inventory = contract["frozen_source_inventory"]
    require(type(inventory) is list and inventory, "empty source inventory")
    inventory_map = {}
    for index, row in enumerate(inventory):
        require(type(row) is dict and set(row) == {"path", "sha256"},
                "source inventory row shape " + str(index))
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                "unsafe source inventory path")
        key = relative.as_posix()
        require(key not in inventory_map, "duplicate source inventory path")
        exact_regular(HW / relative, row["sha256"])
        inventory_map[key] = row["sha256"]
    for required in (RUNNER, PARSER, FILELIST, DOCS359):
        require(required.relative_to(HW).as_posix() in inventory_map,
                "required source absent from inventory: " + str(required))
    require(inventory_map[RUNNER.relative_to(HW).as_posix()] == runner_digest,
            "runner inventory drift")
    require(inventory_map[PARSER.relative_to(HW).as_posix()] == parser_digest,
            "parser inventory drift")
    require(inventory_map[FILELIST.relative_to(HW).as_posix()] == filelist_digest,
            "filelist inventory drift")

    docs359_digest = exact_regular(DOCS359, EXPECTED_DOCS359_SHA256)
    head_docs359_digest, docs359_clean = git_head_docs359()
    require(head_docs359_digest == docs359_digest,
            "docs/359 differs from HEAD bytes")
    require(docs359_clean, "docs/359 has git diff")
    same_json(source_identity["docs359_sha256"], docs359_digest,
              "M2102 source identity.docs359_sha256")

    runner_text = RUNNER.read_text()
    tree = ast.parse(runner_text)
    verify = function(tree, "verify_authority")
    main_node = function(tree, "main")
    run_checked_node = function(tree, "run_checked")
    require(len(named_calls(verify, "verify_review")) == 1,
            "verify_authority M2102 verifier call drift")
    require(len(named_calls(verify, "verify_r9_failed_and_inactive")) == 1,
            "verify_authority R9 verifier call drift")
    require(len(named_calls(verify, "verify_double_sealed_file")) == 1,
            "verify_authority release seal verifier call drift")
    require(len(attribute_calls(verify, "PARSER", "validate_source")) == 1,
            "verify_authority parser validation call drift")
    run_checked_calls = named_calls(main_node, "run_checked")
    direct_main_runs = attribute_calls(main_node, "subprocess", "run")
    require(len(run_checked_calls) == 2,
            "compile/simv run_checked call cardinality drift")
    require(len(direct_main_runs) == 1, "lmstat call cardinality drift")
    require(len(attribute_calls(run_checked_node, "subprocess", "run")) == 1,
            "run_checked subprocess cardinality drift")
    for call in run_checked_calls + direct_main_runs:
        require(tool_call_is_outside_loop(main_node, call),
                "tool call appears inside a retry-capable loop")
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {
                    "Popen", "call", "check_call", "check_output", "system",
                    "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv",
                    "spawnve", "spawnvp", "spawnvpe"}:
                forbidden.append(node.func.attr)
    require(not forbidden, "additional process launch API found")
    required_fragments = (
        'lmstat_command = [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER]',
        "run_checked(compile_command, WORK, compile_log, 21600)",
        '"simv_runs": 1, "command": ["./simv", "-lca"]',
        'run_checked(RUN_STATE["command"], WORK, batch_log, 86400)',
        '"workloads_inside_single_simv": 960',
        '"automatic_retry": False',
    )
    for fragment in required_fragments:
        require(runner_text.count(fragment) >= 1,
                "runner static fragment absent: " + fragment)

    r10_namespaces = (R10_ATTEMPT, R10_RESULT, R10_FAILURE)
    require(not any(os.path.lexists(path) for path in r10_namespaces),
            "R10 public/attempt namespace is not fresh")
    private_patterns = (
        ".m2067_ep34_fc2_exact_continuation_r10_codex_batch_work.*",
        ".m2067_ep34_fc2_exact_continuation_r10_codex_batch_stage.*",
        ".m2067_ep34_fc2_exact_continuation_r10_codex_batch_failure.*",
    )
    private_residue = [
        path.as_posix() for pattern in private_patterns
        for path in (HW / "results").glob(pattern)
    ]
    require(not private_residue, "R10 private namespace residue")

    report = {
        "status": (
            "PASS_M2104_M2103_R10_BATCH_RELEASE_HAMMER__"
            "AUTHORIZE_ROOT_ONE_R10_ATTEMPT_ONLY"),
        "release": {
            "sha256": sha256(RELEASE),
            "inner_sidecar_sha256": sha256(Path(str(RELEASE) + ".sha256")),
            "outer_sidecar_file_sha256": sha256(
                Path(str(RELEASE) + ".sha256.seal.sha256")),
            "double_sealed_regular_non_symlink": True,
            "all_fields_exact": True,
        },
        "runner_verify_authority": {
            "runner_sha256": runner_digest,
            "schema_status_exact": True,
            "identity_fields_exact": len(expected_identity),
            "authorization_fields_exact": len(EXPECTED_AUTHORIZATION),
            "claim_boundary_fields_exact": len(EXPECTED_CLAIM_BOUNDARY),
            "m2101_source_inventory_members_verified": len(inventory_map),
            "m2102_exhaustive_double_seal_verified": True,
            "m2102_sealed_members": len(m2102_mapping),
            "r9_failure_exhaustive_double_seal_verified": True,
            "r9_failure_sealed_members": len(r9_mapping),
            "authority_call_sites": len(named_calls(tree, "verify_authority")),
        },
        "authorization": EXPECTED_AUTHORIZATION,
        "runner_tool_static_cardinality": {
            "lmstat": 1,
            "vcs_compile": 1,
            "simv": 1,
            "workloads_inside_single_simv": 960,
            "tool_calls_inside_loops": 0,
            "additional_process_launch_apis": 0,
            "automatic_retry": False,
        },
        "r9_failure": {
            "failure_json_sha256": r9_failure_digest,
            "manifest_sha256": r9_manifest_digest,
            "outer_file_sha256": r9_outer_digest,
            "owner_pid": r9_owner["pid"],
            "owner_pid_dead": True,
            "completed_slots": 163,
            "failed_slot": 163,
            "simv_runs": 164,
            "automatic_retry": False,
            "success_namespace_absent": True,
        },
        "docs359": {
            "sha256": docs359_digest,
            "head_sha256": head_docs359_digest,
            "git_diff_clean": True,
        },
        "point_in_time_r10_freshness": {
            "attempt_absent": True,
            "result_absent": True,
            "failure_absent": True,
            "private_namespace_residue_count": 0,
        },
        "reviewer_execution": {
            "lmstat": 0,
            "vcs": 0,
            "simv": 0,
            "other_eda": 0,
            "gpu": 0,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
