#!/usr/bin/env python3
"""Independent in-memory M1827 governance probe; never launches tools."""
from __future__ import print_function

import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "system_simulator/scripts/check_m1826_m1794_c2_tsbg_production_campaign_source.py"
SPEC = importlib.util.spec_from_file_location("m1826_checker_for_m1827", str(CHECKER))
MODULE = importlib.util.module_from_spec(SPEC)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1826 checker unavailable")
SPEC.loader.exec_module(MODULE)

M1824_NAMES = (
    "equivalent_queue_flock_downgrade",
    "equivalent_local_flock_downgrade",
    "equivalent_local_flock_wrong_handle",
    "equivalent_atomic_result_publish_unreachable",
    "equivalent_source_contract_authority_misbound",
    "equivalent_namespace_attempt_omitted",
    "equivalent_collision_set_emptied",
    "equivalent_resource_mem_gate_zeroed",
    "equivalent_m1812_release_identity_renamed",
    "equivalent_m1813_release_identity_renamed",
    "equivalent_m1813_manifest_identity_renamed",
    "equivalent_m1813_outer_identity_renamed",
)

NOVEL = (
    ("queue_path_misbound",
     'SHARED_QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")',
     'SHARED_QUEUE = Path("/tmp/isolated") / "date_dual_synopsys_same_uid_eda_queue.lock"'),
    ("local_lock_alias_shared",
     'LOCK = Path("/tmp/m1826_m1794_tsbg_directed_vcs.lock")',
     "LOCK = SHARED_QUEUE"),
    ("queue_handle_origin_wrong",
     'queue_handle = SHARED_QUEUE.open("a+")',
     'queue_handle = LOCK.open("a+")'),
    ("local_handle_origin_wrong",
     'lock_handle = LOCK.open("a+")',
     'lock_handle = SHARED_QUEUE.open("a+")'),
    ("queue_open_mode_wrong",
     'queue_handle = SHARED_QUEUE.open("a+")',
     'queue_handle = SHARED_QUEUE.open("r+")'),
    ("attempt_mkdir_reusable",
     '        ATTEMPT.mkdir()\n        state["attempt"] = True',
     '        ATTEMPT.mkdir(exist_ok=True)\n        state["attempt"] = True'),
    ("attempt_path_conditional_misbound",
     'ATTEMPT = HW / "results/.m1826_m1794_tsbg_directed_vcs_attempt_consumed"',
     'ATTEMPT = HW / ("results/.m1826_m1794_tsbg_directed_vcs_attempt_consumed" if False else "results/.m1826_bypass_attempt")'),
    ("publish_primitive_constant_false",
     '    if renameat2(-100, os.fsencode(source), -100,\n                 os.fsencode(destination), 1) != 0:',
     '    if False and renameat2(-100, os.fsencode(source), -100,\n                 os.fsencode(destination), 1) != 0:'),
    ("collision_membership_body_noop",
     '        if comm in blocked:\n            hits.append((item.name, comm))',
     '        if comm in blocked:\n            pass'),
    ("resource_values_forged",
     '    if values.get("MemAvailable", 0) < 16 * 1024 * 1024:',
     '    values = {"MemAvailable": 1 << 60, "SwapFree": 1 << 60, "CommitLimit": 1 << 60, "Committed_AS": 0}\n    if values.get("MemAvailable", 0) < 16 * 1024 * 1024:'),
    ("resource_raise_noop",
     '        raise Failure("MemAvailable below 16 GiB")',
     "        pass"),
    ("exact_primitive_early_return",
     'def exact(path, digest):\n    path = Path(path)',
     'def exact(path, digest):\n    return\n    path = Path(path)'),
    ("release_identity_guard_noop",
     '    if release.get("identity") != expected_identity:\n        raise Failure("M1828 transitive identity")',
     '    if release.get("identity") != expected_identity:\n        pass'),
    ("failure_quarantine_creation_noop",
     "                FAIL_STAGE.mkdir(exist_ok=False)",
     "                pass"),
    ("review_seal_primitive_early_return",
     'def verify_directory_seal(root, manifest_sha, outer_sha):\n    root = Path(root)',
     'def verify_directory_seal(root, manifest_sha, outer_sha):\n    return\n    root = Path(root)'),
)


def source_texts():
    return {
        MODULE.RTL: MODULE.RTL.read_text(),
        MODULE.TB: MODULE.TB.read_text(),
        MODULE.SVA: MODULE.SVA.read_text(),
        MODULE.RUNNER: MODULE.RUNNER.read_text(),
    }


def rejected(texts):
    try:
        MODULE.validate_semantics(texts)
    except RuntimeError:
        return True
    return False


def main():
    positive = MODULE.validate_sources()
    base = source_texts()
    declared = dict((row[0], row) for row in MODULE.MUTATION_SPECS)
    m1824_rows = []
    for name in M1824_NAMES:
        row = declared[name]
        path = {"tb": MODULE.TB, "sva": MODULE.SVA,
                "runner": MODULE.RUNNER}[row[1]]
        if row[2] not in base[path]:
            raise RuntimeError("M1824 anchor absent " + name)
        mutated = dict(base)
        mutated[path] = mutated[path].replace(row[2], row[3], 1)
        m1824_rows.append({"name": name, "rejected": rejected(mutated)})

    novel_rows = []
    for name, old, new in NOVEL:
        if old not in base[MODULE.RUNNER]:
            raise RuntimeError("novel anchor absent " + name)
        mutated = dict(base)
        mutated[MODULE.RUNNER] = base[MODULE.RUNNER].replace(old, new, 1)
        novel_rows.append({"name": name, "rejected": rejected(mutated)})

    print(json.dumps({
        "schema": "m1827_independent_governance_probe_r1_v1",
        "status": "FAIL_CLOSED_NOVEL_EQUIVALENT_BYPASSES_ESCAPED",
        "python": sys.version.split()[0],
        "positive_status": positive["status"],
        "m1824_declared": {
            "attacks": len(m1824_rows),
            "rejected": sum(1 for row in m1824_rows if row["rejected"]),
            "escaped": sum(1 for row in m1824_rows if not row["rejected"]),
            "rows": m1824_rows,
        },
        "novel_equivalent": {
            "attacks": len(novel_rows),
            "rejected": sum(1 for row in novel_rows if row["rejected"]),
            "escaped": sum(1 for row in novel_rows if not row["rejected"]),
            "rows": novel_rows,
        },
        "contract_sha_rejection_counted_as_semantic": False,
        "source_files_modified": False,
        "eda_runs": 0,
        "license_queries": 0,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
