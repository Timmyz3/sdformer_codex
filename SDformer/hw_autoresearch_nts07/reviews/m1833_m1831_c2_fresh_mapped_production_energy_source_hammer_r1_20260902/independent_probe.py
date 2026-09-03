#!/usr/bin/env python3
"""Independent M1833 source probe; CPU-only and never launches EDA.

The author mutation suite changes a source without updating the contract's
source_inventory digest.  Such a test proves identity binding, not semantic
coverage.  This probe updates the affected inventory entry before invoking
validate_source_texts, so rejection must come from an actual semantic guard.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "system_simulator/scripts/check_m1831_c2_fresh_mapped_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1831_checker_for_m1833", str(CHECKER))
MODULE = importlib.util.module_from_spec(SPEC)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1831 checker unavailable")
SPEC.loader.exec_module(MODULE)


def digest_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def source_texts():
    paths = (MODULE.RUNNER, MODULE.CHECKER, MODULE.TEST, MODULE.CONTRACT,
             MODULE.CORE, MODULE.FAULT, MODULE.TOP_TB, MODULE.MEM,
             MODULE.PROD_ASSERT, MODULE.M979, MODULE.UCLI, MODULE.PT_TCL,
             MODULE.FILELISTS["k8"], MODULE.FILELISTS["k1x8"])
    return dict((path, path.read_text()) for path in paths)


def update_inventory(texts, changed_path):
    contract = json.loads(texts[MODULE.CONTRACT])
    relative = str(changed_path.relative_to(MODULE.HW))
    hits = 0
    for row in contract["source_inventory"]:
        if row["path"] == relative:
            row["sha256"] = digest_text(texts[changed_path])
            hits += 1
    if hits != 1:
        raise RuntimeError("inventory path missing/duplicate " + relative)
    texts[MODULE.CONTRACT] = json.dumps(contract, indent=2, sort_keys=False) + "\n"


def rejected_with_synchronized_inventory(base, path, old, new):
    if old not in base[path]:
        raise RuntimeError("attack anchor absent: " + old[:80])
    mutated = dict(base)
    mutated[path] = base[path].replace(old, new, 1)
    update_inventory(mutated, path)
    try:
        MODULE.validate_source_texts(mutated)
    except RuntimeError:
        return True
    return False


ATTACKS = (
    ("tool_nonzero_return_ignored", MODULE.RUNNER,
     '    if completed.returncode != 0: raise Failure("tool failure " + Path(command[0]).name)',
     '    if False and completed.returncode != 0: raise Failure("tool failure " + Path(command[0]).name)'),
    ("runtime_log_validation_bypassed", MODULE.RUNNER,
     '                checked["runtime"] = CHECK.validate_runtime_log(log, axis, case_id)',
     '                checked["runtime"] = {"bypassed": True}'),
    ("source_review_admission_guard_noop", MODULE.RUNNER,
     '    if (type(source_severity) is not dict\n            or source_severity.get("p0") != 0',
     '    if False and (type(source_severity) is not dict\n            or source_severity.get("p0") != 0'),
    ("release_identity_guard_noop", MODULE.RUNNER,
     '    if release.get("identity") != expected_release_identity:\n        raise Failure("M1835 transitive identity")',
     '    if False and release.get("identity") != expected_release_identity:\n        raise Failure("M1835 transitive identity")'),
    ("release_budget_guard_noop", MODULE.RUNNER,
     '    if release.get("fresh_execution_budget") != dict(\n            COUNTS, automatic_retry=False, reuse_prior_simv=False):',
     '    if False and release.get("fresh_execution_budget") != dict(\n            COUNTS, automatic_retry=False, reuse_prior_simv=False):'),
    ("release_authorization_guard_noop", MODULE.RUNNER,
     '    if release.get("authorization") != {',
     '    if False and release.get("authorization") != {'),
    ("attempt_latch_reusable", MODULE.RUNNER,
     '        ATTEMPT.mkdir(); state["attempt"] = True',
     '        ATTEMPT.mkdir(exist_ok=True); state["attempt"] = True'),
    ("collision_detection_body_noop", MODULE.RUNNER,
     '        if comm in blocked: hits.append((item.name, comm))',
     '        if comm in blocked: pass'),
    ("resource_gate_early_return", MODULE.RUNNER,
     'def resource_gate():\n    values = {}',
     'def resource_gate():\n    return\n    values = {}'),
    ("exact_identity_primitive_early_return", MODULE.RUNNER,
     'def exact(path, digest):\n    path = Path(path)',
     'def exact(path, digest):\n    return\n    path = Path(path)'),
    ("directory_seal_mapping_forged", MODULE.RUNNER,
     'def verify_directory_seal(root, manifest_sha, outer_sha):\n    root = Path(root); manifest = root / "SHA256SUMS"',
     'def verify_directory_seal(root, manifest_sha, outer_sha):\n    if Path(root) == M1811: return {"receipt.json": M1811_RECEIPT_SHA256}\n    if Path(root) == M1830: return {"review.json": M1830_REVIEW_SHA256}\n    return {"review.json": sha(Path(root) / "review.json")}\n    root = Path(root); manifest = root / "SHA256SUMS"'),
    ("file_double_seal_primitive_early_return", MODULE.RUNNER,
     'def verify_file_double_seal(path, file_sha, sidecar_sha, outer_sha):\n    sidecar = Path(str(path) + ".sha256")',
     'def verify_file_double_seal(path, file_sha, sidecar_sha, outer_sha):\n    return\n    sidecar = Path(str(path) + ".sha256")'),
    ("no_replace_publication_overwrite", MODULE.RUNNER,
     'def publish_no_replace(source, destination):\n    libc = ctypes.CDLL(None, use_errno=True); renameat2 = getattr(libc, "renameat2")',
     'def publish_no_replace(source, destination):\n    os.replace(str(source), str(destination)); return\n    libc = ctypes.CDLL(None, use_errno=True); renameat2 = getattr(libc, "renameat2")'),
    ("saif_completeness_guard_noop", MODULE.RUNNER,
     '        if state["vcs_compiles"] != 2 or state["simv_runs"] != 10 or state["saif_files"] != 10:',
     '        if False and (state["vcs_compiles"] != 2 or state["simv_runs"] != 10 or state["saif_files"] != 10):'),
    ("ptpx_marker_guard_noop", MODULE.RUNNER,
     '                if not marker.is_file() or "PASS_M1831_C2_FRESH_MAPPED_PRODUCTION_PTPX_PENDING_RESULT_HAMMER" not in marker.read_text():',
     '                if False and (not marker.is_file() or "PASS_M1831_C2_FRESH_MAPPED_PRODUCTION_PTPX_PENDING_RESULT_HAMMER" not in marker.read_text()):'),
    ("final_execution_count_guard_noop", MODULE.RUNNER,
     '        if any(state[key] != value for key, value in COUNTS.items()):',
     '        if False and any(state[key] != value for key, value in COUNTS.items()):'),
    ("per_tool_source_revalidation_removed", MODULE.RUNNER,
     '    CHECK.validate_sources(); collision_gate()\n    with Path(output).open("wb") as stream:',
     '    collision_gate()\n    with Path(output).open("wb") as stream:'),
    ("ptpx_exact_annotation_guard_noop", MODULE.PT_TCL,
     'if {$total_nets <= 0 || $annotated_nets != $total_nets',
     'if {0 && ($total_nets <= 0 || $annotated_nets != $total_nets'),
)


def main():
    positive = MODULE.validate_sources()
    base = source_texts()
    rows = []
    for name, path, old, new in ATTACKS:
        rows.append({"name": name,
                     "rejected": rejected_with_synchronized_inventory(
                         base, path, old, new)})
    print(json.dumps({
        "schema": "m1833_m1831_c2_energy_source_independent_probe_r1_v1",
        "status": "FAIL_CLOSED_IF_ANY_SYNCHRONIZED_SEMANTIC_ATTACK_ESCAPES",
        "python": sys.version.split()[0],
        "positive_status": positive["status"],
        "attacks": len(rows),
        "rejected": sum(1 for row in rows if row["rejected"]),
        "escaped": sum(1 for row in rows if not row["rejected"]),
        "escaped_names": [row["name"] for row in rows if not row["rejected"]],
        "rows": rows,
        "inventory_updated_with_each_mutation": True,
        "eda_runs": 0,
        "license_queries": 0,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
