#!/usr/bin/env python3
"""Synchronized-inventory semantic tests for the M1845 successor source."""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1845_c2_fresh_mapped_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1845_checker_test", str(CHECKER))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1845 checker unavailable")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def texts():
    paths = (MODULE.RUNNER, MODULE.CHECKER, MODULE.TEST, MODULE.CONTRACT,
             MODULE.CORE, MODULE.FAULT, MODULE.TOP_TB, MODULE.MEM,
             MODULE.PROD_ASSERT, MODULE.M979, MODULE.UCLI, MODULE.PT_TCL,
             MODULE.FILELISTS["k8"], MODULE.FILELISTS["k1x8"])
    return dict((path, path.read_text()) for path in paths)


def update_inventory(values, changed_path):
    contract = json.loads(values[MODULE.CONTRACT])
    relative = str(changed_path.relative_to(MODULE.HW))
    hits = 0
    for row in contract["source_inventory"]:
        if row["path"] == relative:
            row["sha256"] = digest_text(values[changed_path])
            hits += 1
    if hits != 1:
        raise RuntimeError("inventory path missing/duplicate " + relative)
    values[MODULE.CONTRACT] = json.dumps(
        contract, indent=2, sort_keys=False, allow_nan=False) + "\n"


def rejected_with_synchronized_inventory(base, path, old, new):
    if old not in base[path]:
        raise RuntimeError("attack anchor absent: " + old[:100])
    mutated = dict(base)
    mutated[path] = base[path].replace(old, new, 1)
    update_inventory(mutated, path)
    try:
        MODULE.validate_source_texts(mutated)
    except (RuntimeError, SyntaxError):
        return True
    return False


# The first 18 rows reproduce the exact semantic escapes reported by M1833.
# Every mutation updates its source_inventory digest before validation.
ATTACKS = (
    ("tool_nonzero_return_ignored", MODULE.RUNNER,
     '    if completed.returncode != 0: raise Failure("tool failure " + Path(command[0]).name)',
     '    if False and completed.returncode != 0: raise Failure("tool failure " + Path(command[0]).name)'),
    ("runtime_log_validation_bypassed", MODULE.RUNNER,
     '                checked["runtime"] = CHECK.validate_runtime_log(log, axis, case_id)',
     '                checked["runtime"] = {"bypassed": True}'),
    ("source_review_admission_guard_noop", MODULE.RUNNER,
     '    if (source_review.get("schema") !=',
     '    if False and (source_review.get("schema") !='),
    ("release_identity_guard_noop", MODULE.RUNNER,
     '    if release.get("identity") != expected_release_identity:\n        raise Failure("M1849 transitive identity")',
     '    if False and release.get("identity") != expected_release_identity:\n        raise Failure("M1849 transitive identity")'),
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
     'def verify_directory_seal(root, manifest_sha, outer_sha):\n    return {}\n    root = Path(root); manifest = root / "SHA256SUMS"'),
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
    ("compile_command_record_disabled", MODULE.RUNNER,
     '                build / "compile.log", record_command=True)',
     '                build / "compile.log", record_command=False)'),
    ("sealed_compile_log_copy_removed", MODULE.RUNNER,
     '            shutil.copy2(str(source_log), str(sealed_log))',
     '            pass  # compile log copy removed'),
    ("sealed_result_validation_removed", MODULE.RUNNER,
     '        CHECK.validate_sealed_result_stage(STAGE)',
     '        pass  # result-stage validation removed'),
    ("review_schema_relaxed", MODULE.RUNNER,
     '            "m1848_m1845_c2_fresh_mapped_production_energy_source_hammer_review_r1_v1"',
     '            source_review.get("schema")'),
    ("review_status_relaxed", MODULE.RUNNER,
     '            "PASS_M1848_M1845_C2_FRESH_MAPPED_PRODUCTION_ENERGY_SOURCE_HAMMER__P0_0_P1_0_P2_0__AUTHORIZED_FOR_M1849_RELEASE"',
     '            source_review.get("status")'),
    ("review_severity_p2_relaxed", MODULE.RUNNER,
     '{"p0": 0, "p1": 0, "p2": 0}',
     '{"p0": 0, "p1": 0, "p2": source_review.get("severity_counts", {}).get("p2")}'),
    ("review_authorization_relaxed", MODULE.RUNNER,
     '            or source_review.get("authorization") != {',
     '            or False and source_review.get("authorization") != {'),
)


def compile_log_unit_checks():
    rows = []
    with tempfile.TemporaryDirectory(prefix="m1845_compile_log_test_") as temp:
        temp = Path(temp)
        for axis in ("k8", "k1x8"):
            command = MODULE.expected_compile_command(axis)
            good = temp / (axis + ".good.log")
            good.write_text("M1845_COMMAND_JSON=" + json.dumps(
                command, separators=(",", ":")) + "\nChronologic VCS compiler version\n")
            MODULE.validate_compile_log(good, axis, command)
            bad = temp / (axis + ".bad.log")
            bad.write_text(good.read_text() + "Unresolved module foo\n")
            try:
                MODULE.validate_compile_log(bad, axis, command)
                rejected = False
            except RuntimeError:
                rejected = True
            if not rejected:
                raise RuntimeError("compile diagnostic escaped " + axis)
            wrong = temp / (axis + ".wrong.log")
            changed = list(command); changed[-1] = "other_simv"
            wrong.write_text("M1845_COMMAND_JSON=" + json.dumps(
                changed, separators=(",", ":")) + "\nclean\n")
            try:
                MODULE.validate_compile_log(wrong, axis, command)
                rejected_wrong = False
            except RuntimeError:
                rejected_wrong = True
            if not rejected_wrong:
                raise RuntimeError("compile command drift escaped " + axis)
            rows.append({"axis": axis, "fatal_rejected": rejected,
                         "command_drift_rejected": rejected_wrong})
    return rows


def main():
    positive = MODULE.validate_sources()
    base = texts(); rows = []
    for name, path, old, new in ATTACKS:
        rows.append({"name": name,
                     "rejected": rejected_with_synchronized_inventory(
                         base, path, old, new)})
    if not all(row["rejected"] for row in rows):
        escaped = [row["name"] for row in rows if not row["rejected"]]
        raise RuntimeError("M1845 synchronized semantic mutation escaped "
                           + repr(escaped))
    compile_rows = compile_log_unit_checks()
    print(json.dumps({
        "schema": "m1845_c2_fresh_mapped_energy_synchronized_semantic_tests_r1_v1",
        "status": "PASS_M1845_SYNCHRONIZED_SEMANTIC_MUTATIONS",
        "positive": positive,
        "m1833_escapes_reproduced": 18,
        "attacks": len(rows),
        "rejected": sum(1 for row in rows if row["rejected"]),
        "inventory_updated_with_each_mutation": True,
        "rows": rows,
        "compile_log_unit_checks": compile_rows,
        "eda_runs": 0,
        "license_queries": 0,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
