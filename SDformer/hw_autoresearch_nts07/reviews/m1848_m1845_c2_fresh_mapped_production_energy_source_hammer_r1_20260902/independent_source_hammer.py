#!/usr/bin/env python3
"""Independent, CPU-only source/mutation hammer for M1845."""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
CHECKER = HW / "system_simulator/scripts/check_m1845_c2_fresh_mapped_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1848_independent_checker", str(CHECKER))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1845 checker unavailable")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

AUTHOR = HW / "reviews/m1845_m1833_m1831_c2_fresh_mapped_production_energy_successor_source_author_receipt_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "contract": "b6aa9ee8763a79711135184bb8e2dfeee359436674fe659e2feed000a048aa94",
    "contract_sidecar": "8fb2dc35b9e624e19ef4af47793d59be15f754e4c88639a845198f1add4c2e07",
    "contract_outer": "de8a9d3beb5b8204fef74a601d48db601c48fab0e9bbb65c67608af8511b3ede",
    "runner": "e5f9d6d68b0acde2966c7140ca9d070dc40f362bccada827aad00d77688acc81",
    "checker": "8517a0bc8887ce272d5d1ae72f5ba9f738a0c702c3d7f1d48381df6c1828cb48",
    "test": "a9613236f41fa57b8f12e1fbae5af215b3898e977bf8a60fa474524ce90b39d6",
    "author_receipt": "8fa17213f2cae24b07b6628498a80ed1c7adbdd27791b9aff301eaeab581959e",
    "author_manifest": "a2407a1b2e56b9accf154eadc73469cdb14f86f3db3b7e6dbb0889de26d52192",
    "author_outer": "e48890fb0691f4ce8c6ad6e2c577353bc7c615e468f491c21ffdfb527c31102b",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def verify_author_seal():
    manifest = AUTHOR / "SHA256SUMS"
    outer = AUTHOR / "SHA256SUMS.seal.sha256"
    need(sha(AUTHOR / "receipt.json") == EXPECTED["author_receipt"], "author receipt")
    need(sha(manifest) == EXPECTED["author_manifest"], "author manifest")
    need(sha(outer) == EXPECTED["author_outer"], "author outer")
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "author outer content")
    mapping = {}
    for row in manifest.read_text().splitlines():
        digest, name = row.split(maxsplit=1)
        name = name.lstrip("*")
        need(name not in mapping and sha(AUTHOR / name) == digest,
             "author manifest member " + name)
        mapping[name] = digest
    actual = set(path.relative_to(AUTHOR).as_posix() for path in AUTHOR.rglob("*")
                 if path.is_file() and path.name not in
                 ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    need(actual == set(mapping), "author manifest exhaustive")


def source_texts():
    paths = (MODULE.RUNNER, MODULE.CHECKER, MODULE.TEST, MODULE.CONTRACT,
             MODULE.CORE, MODULE.FAULT, MODULE.TOP_TB, MODULE.MEM,
             MODULE.PROD_ASSERT, MODULE.M979, MODULE.UCLI, MODULE.PT_TCL,
             MODULE.FILELISTS["k8"], MODULE.FILELISTS["k1x8"])
    return dict((path, path.read_text()) for path in paths)


def sync_inventory(values, path):
    contract = json.loads(values[MODULE.CONTRACT])
    relative = str(path.relative_to(MODULE.HW))
    hits = 0
    for row in contract["source_inventory"]:
        if row["path"] == relative:
            row["sha256"] = digest_text(values[path]); hits += 1
    need(hits == 1, "inventory synchronization " + relative)
    values[MODULE.CONTRACT] = json.dumps(
        contract, indent=2, sort_keys=False, allow_nan=False) + "\n"


def independent_policy(values):
    runner = values[MODULE.RUNNER]
    checker = values[MODULE.CHECKER]
    pt = values[MODULE.PT_TCL]
    ucli = values[MODULE.UCLI]
    contract = json.loads(values[MODULE.CONTRACT])
    required_runner = (
        'source_review.get("schema") !=',
        'm1848_m1845_c2_fresh_mapped_production_energy_source_hammer_review_r1_v1',
        'PASS_M1848_M1845_C2_FRESH_MAPPED_PRODUCTION_ENERGY_SOURCE_HAMMER__P0_0_P1_0_P2_0__AUTHORIZED_FOR_M1849_RELEASE',
        'source_review.get("severity_counts") !=',
        '{"p0": 0, "p1": 0, "p2": 0}',
        'source_review.get("reviewer_identity") ==',
        'source_review.get("authorization") !=',
        'M1845_EXPECTED_M1848_REVIEW_SHA256',
        'M1845_EXPECTED_M1848_MANIFEST_SHA256',
        'M1845_EXPECTED_M1848_OUTER_FILE_SHA256',
        'M1845_EXPECTED_M1849_RELEASE_SHA256',
        'M1845_EXPECTED_M1849_SIDECAR_SHA256',
        'M1845_EXPECTED_M1849_OUTER_FILE_SHA256',
        '"source_review_json_sha256"',
        '"source_review_manifest_sha256"',
        '"source_review_outer_file_sha256"',
        'verify_file_double_seal(\n        M1849_RELEASE',
        'compile_evidence = STAGE / "compile_logs"',
        'shutil.copy2(str(source_log), str(sealed_log))',
        'CHECK.validate_compile_log(\n                sealed_log',
        'seal_dir(STAGE)',
        'CHECK.validate_sealed_result_stage(STAGE)',
        'publish_no_replace(STAGE, RESULT)',
        'all ten mapped SAIF coordinates required before PTPX',
        'if any(state[key] != value for key, value in COUNTS.items()):',
    )
    for token in required_runner:
        need(token in runner, "independent runner token " + token[:60])
    need(runner.index('seal_dir(STAGE)') <
         runner.index('CHECK.validate_sealed_result_stage(STAGE)') <
         runner.index('publish_no_replace(STAGE, RESULT)'),
         "seal/validate/publish order")
    need(runner.index("all ten mapped SAIF coordinates required before PTPX") <
         runner.index('state["phase"] = "PTPX_"'), "SAIF/PTPX order")
    required_checker = (
        'need(actual == set(mapping), "result manifest not exhaustive")',
        'required = {"compile_logs/k8.compile.log",',
        '"compile_logs/k1x8.compile.log", "compile_log_rows.json"}',
        'need(required.issubset(set(mapping)), "sealed compile evidence absent")',
        'checked = [validate_compile_log(root / "compile_logs" /',
        'need(rows == checked, "compile evidence row drift")',
        'need(lines and lines[0].startswith("M1845_COMMAND_JSON=")',
        'need(type(command) is list and command == expected,',
        'forbidden = ("error-", "fatal", "unresolved", "undefined module",',
        '"black box", "black-box", "compile error")',
        'need(not any(token in lowered for token in forbidden),',
        'need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"]',
    )
    for token in required_checker:
        need(token in checker, "independent checker token " + token[:60])
    normalized_pt = " ".join(pt.split())
    need('read_saif -strip_path $::env(M1831_SAIF_INSTANCE)' in normalized_pt,
         "PTPX strip path")
    need('if {$total_nets <= 0 || $annotated_nets != $total_nets || $annotated_percent != 100.0 || $total_leaf_cells <= 0 || $annotated_leaf_cells != $total_leaf_cells || $annotated_leaf_percent != 100.0} { error "M1831_FAIL_EXACT_ANNOTATION_GATE" }' in normalized_pt,
         "PTPX exact annotation")
    need('power tb_m1831_c2_fresh_mapped_production_energy.core.dut.implementation' in ucli and
         'power -report $::env(M1831_SAIF_FILE) 1e-9 tb_m1831_c2_fresh_mapped_production_energy.core.dut.implementation' in ucli,
         "DUT-only SAIF scope")
    future = contract.get("future_authority", {})
    need(future.get("m1848_required_schema") ==
         "m1848_m1845_c2_fresh_mapped_production_energy_source_hammer_review_r1_v1",
         "contract future review schema")
    need(future.get("m1848_required_status") ==
         "PASS_M1848_M1845_C2_FRESH_MAPPED_PRODUCTION_ENERGY_SOURCE_HAMMER__P0_0_P1_0_P2_0__AUTHORIZED_FOR_M1849_RELEASE",
         "contract future review status")
    need(future.get("m1848_required_severity_counts") ==
         {"p0": 0, "p1": 0, "p2": 0}, "contract future severity")
    need(future.get("m1849_launch_release_sha256") == "PENDING_EXTERNAL_PIN" and
         future.get("m1849_launch_release_sidecar_sha256") == "PENDING_EXTERNAL_PIN" and
         future.get("m1849_launch_release_outer_file_sha256") == "PENDING_EXTERNAL_PIN",
         "contract release triple pins")


def reject_mutation(base, path, old, new):
    need(old in base[path], "attack anchor absent " + old[:80])
    values = dict(base)
    values[path] = base[path].replace(old, new, 1)
    if path in MODULE.SOURCE_PATHS:
        sync_inventory(values, path)
    rejected_by_checker = False
    rejected_by_independent = False
    try:
        MODULE.validate_source_texts(values)
    except (RuntimeError, SyntaxError, ValueError):
        rejected_by_checker = True
    try:
        independent_policy(values)
    except (RuntimeError, SyntaxError, ValueError):
        rejected_by_independent = True
    need(rejected_by_checker or rejected_by_independent,
         "semantic mutation escaped checker and independent policy")
    return {"checker": rejected_by_checker, "independent": rejected_by_independent}


ATTACKS = (
    ("tool_return_ignored", MODULE.RUNNER,
     '    if completed.returncode != 0: raise Failure("tool failure " + Path(command[0]).name)',
     '    if False and completed.returncode != 0: raise Failure("tool failure " + Path(command[0]).name)'),
    ("runtime_validation_bypassed", MODULE.RUNNER,
     '                checked["runtime"] = CHECK.validate_runtime_log(log, axis, case_id)',
     '                checked["runtime"] = {"bypassed": True}'),
    ("review_gate_noop", MODULE.RUNNER,
     '    if (source_review.get("schema") !=',
     '    if False and (source_review.get("schema") !='),
    ("review_different_author_removed", MODULE.RUNNER,
     '            or source_review.get("reviewer_identity") ==',
     '            or False and source_review.get("reviewer_identity") =='),
    ("review_manifest_pin_self_fulfilled", MODULE.RUNNER,
     '        authority_pin("M1845_EXPECTED_M1848_MANIFEST_SHA256"),',
     '        sha(M1848_SOURCE_REVIEW / "SHA256SUMS"),'),
    ("review_outer_pin_self_fulfilled", MODULE.RUNNER,
     '        authority_pin("M1845_EXPECTED_M1848_OUTER_FILE_SHA256"))',
     '        sha(M1848_SOURCE_REVIEW / "SHA256SUMS.seal.sha256"))'),
    ("release_identity_noop", MODULE.RUNNER,
     '    if release.get("identity") != expected_release_identity:',
     '    if False and release.get("identity") != expected_release_identity:'),
    ("release_budget_noop", MODULE.RUNNER,
     '    if release.get("fresh_execution_budget") != dict(',
     '    if False and release.get("fresh_execution_budget") != dict('),
    ("release_authorization_noop", MODULE.RUNNER,
     '    if release.get("authorization") != {',
     '    if False and release.get("authorization") != {'),
    ("release_sidecar_pin_self_fulfilled", MODULE.RUNNER,
     'authority_pin("M1845_EXPECTED_M1849_SIDECAR_SHA256")',
     'sha(M1849_RELEASE_SIDECAR)'),
    ("review_manifest_release_binding_removed", MODULE.RUNNER,
     '        "source_review_manifest_sha256": sha(M1848_SOURCE_REVIEW / "SHA256SUMS"),',
     '        "source_review_manifest_sha256": "unbound",'),
    ("attempt_reusable", MODULE.RUNNER,
     '        ATTEMPT.mkdir(); state["attempt"] = True',
     '        ATTEMPT.mkdir(exist_ok=True); state["attempt"] = True'),
    ("collision_noop", MODULE.RUNNER,
     '        if comm in blocked: hits.append((item.name, comm))',
     '        if comm in blocked: pass'),
    ("resource_early_return", MODULE.RUNNER,
     'def resource_gate():\n    values = {}',
     'def resource_gate():\n    return\n    values = {}'),
    ("exact_early_return", MODULE.RUNNER,
     'def exact(path, digest):\n    path = Path(path)',
     'def exact(path, digest):\n    return\n    path = Path(path)'),
    ("directory_seal_forged", MODULE.RUNNER,
     'def verify_directory_seal(root, manifest_sha, outer_sha):\n    root = Path(root); manifest = root / "SHA256SUMS"',
     'def verify_directory_seal(root, manifest_sha, outer_sha):\n    return {}\n    root = Path(root); manifest = root / "SHA256SUMS"'),
    ("file_seal_early_return", MODULE.RUNNER,
     'def verify_file_double_seal(path, file_sha, sidecar_sha, outer_sha):\n    sidecar = Path(str(path) + ".sha256")',
     'def verify_file_double_seal(path, file_sha, sidecar_sha, outer_sha):\n    return\n    sidecar = Path(str(path) + ".sha256")'),
    ("publication_overwrite", MODULE.RUNNER,
     'def publish_no_replace(source, destination):\n    libc = ctypes.CDLL(None, use_errno=True); renameat2 = getattr(libc, "renameat2")',
     'def publish_no_replace(source, destination):\n    os.replace(str(source), str(destination)); return\n    libc = ctypes.CDLL(None, use_errno=True); renameat2 = getattr(libc, "renameat2")'),
    ("saif_completeness_noop", MODULE.RUNNER,
     '        if state["vcs_compiles"] != 2 or state["simv_runs"] != 10 or state["saif_files"] != 10:',
     '        if False and (state["vcs_compiles"] != 2 or state["simv_runs"] != 10 or state["saif_files"] != 10):'),
    ("ptpx_marker_noop", MODULE.RUNNER,
     '                if not marker.is_file() or "PASS_M1831_C2_FRESH_MAPPED_PRODUCTION_PTPX_PENDING_RESULT_HAMMER" not in marker.read_text():',
     '                if False and (not marker.is_file() or "PASS_M1831_C2_FRESH_MAPPED_PRODUCTION_PTPX_PENDING_RESULT_HAMMER" not in marker.read_text()):'),
    ("final_count_noop", MODULE.RUNNER,
     '        if any(state[key] != value for key, value in COUNTS.items()):',
     '        if False and any(state[key] != value for key, value in COUNTS.items()):'),
    ("source_revalidation_removed", MODULE.RUNNER,
     '    CHECK.validate_sources(); collision_gate()\n    with Path(output).open("wb") as stream:',
     '    collision_gate()\n    with Path(output).open("wb") as stream:'),
    ("compile_command_record_disabled", MODULE.RUNNER,
     '                build / "compile.log", record_command=True)',
     '                build / "compile.log", record_command=False)'),
    ("sealed_compile_copy_removed", MODULE.RUNNER,
     '            shutil.copy2(str(source_log), str(sealed_log))',
     '            pass  # removed'),
    ("stage_validation_removed", MODULE.RUNNER,
     '        CHECK.validate_sealed_result_stage(STAGE)',
     '        pass  # validation removed'),
    ("publish_before_validation", MODULE.RUNNER,
     '        CHECK.validate_sealed_result_stage(STAGE)\n        publish_no_replace(WORK, PRIVATE)',
     '        publish_no_replace(WORK, PRIVATE)\n        CHECK.validate_sealed_result_stage(STAGE)'),
    ("ptpx_annotation_noop", MODULE.PT_TCL,
     'if {$total_nets <= 0 || $annotated_nets != $total_nets',
     'if {0 && ($total_nets <= 0 || $annotated_nets != $total_nets'),
    ("saif_scope_widened", MODULE.UCLI,
     'power tb_m1831_c2_fresh_mapped_production_energy.core.dut.implementation',
     'power tb_m1831_c2_fresh_mapped_production_energy.core'),
    ("result_manifest_exhaustiveness_removed", MODULE.CHECKER,
     '    need(actual == set(mapping), "result manifest not exhaustive")',
     '    need(True, "result manifest not exhaustive")'),
    ("compile_fatal_diagnostic_removed", MODULE.CHECKER,
     'forbidden = ("error-", "fatal", "unresolved", "undefined module",',
     'forbidden = ("error-", "unresolved", "undefined module",'),
    ("compile_command_identity_removed", MODULE.CHECKER,
     '    need(type(command) is list and command == expected,',
     '    need(type(command) is list,'),
    ("k1x8_compile_member_removed", MODULE.CHECKER,
     '                "compile_logs/k1x8.compile.log", "compile_log_rows.json"}',
     '                "compile_log_rows.json"}'),
    ("compile_rows_revalidation_removed", MODULE.CHECKER,
     '    need(rows == checked, "compile evidence row drift")',
     '    need(True, "compile evidence row drift")'),
    ("contract_release_outer_pin_removed", MODULE.CONTRACT,
     '"m1849_launch_release_outer_file_sha256": "PENDING_EXTERNAL_PIN"',
     '"m1849_launch_release_outer_file_sha256": "UNBOUND"'),
    ("contract_budget_sim_reduced", MODULE.CONTRACT,
     '"simv_runs": 10,',
     '"simv_runs": 9,'),
)


def compile_log_tests():
    results = []
    with tempfile.TemporaryDirectory(prefix="m1848_compile_") as temp_name:
        temp = Path(temp_name)
        for axis in ("k8", "k1x8"):
            command = MODULE.expected_compile_command(axis)
            good = temp / (axis + ".good.log")
            good.write_text("M1845_COMMAND_JSON=" + json.dumps(
                command, separators=(",", ":")) + "\nChronologic VCS compiler\nCPU time: 1\n")
            MODULE.validate_compile_log(good, axis, command)
            rejected = 0
            for index, bad_token in enumerate((
                    "Error-[X]", "Fatal", "unresolved", "undefined module",
                    "black box", "black-box", "compile error")):
                bad = temp / (axis + ".bad" + str(index) + ".log")
                bad.write_text(good.read_text() + bad_token + "\n")
                try:
                    MODULE.validate_compile_log(bad, axis, command)
                except RuntimeError:
                    rejected += 1
            wrong = temp / (axis + ".wrong.log")
            changed = list(command); changed[-1] = "wrong_simv"
            wrong.write_text("M1845_COMMAND_JSON=" + json.dumps(
                changed, separators=(",", ":")) + "\nclean\n")
            try:
                MODULE.validate_compile_log(wrong, axis, command)
            except RuntimeError:
                rejected += 1
            need(rejected == 8, "compile negatives " + axis)
            results.append({"axis": axis, "positive": True,
                            "negative_checks": 8, "negative_rejected": 8})
    return results


def main():
    need(sha(MODULE.CONTRACT) == EXPECTED["contract"], "contract identity")
    need(sha(MODULE.CONTRACT_SIDECAR) == EXPECTED["contract_sidecar"],
         "contract sidecar identity")
    need(sha(MODULE.CONTRACT_OUTER) == EXPECTED["contract_outer"],
         "contract outer identity")
    need(sha(MODULE.RUNNER) == EXPECTED["runner"], "runner identity")
    need(sha(MODULE.CHECKER) == EXPECTED["checker"], "checker identity")
    need(sha(MODULE.TEST) == EXPECTED["test"], "test identity")
    need(sha(DOCS359) == EXPECTED["docs359"], "docs359 identity")
    verify_author_seal()
    positive = MODULE.validate_sources()
    base = source_texts(); independent_policy(base)
    rows = []
    for name, path, old, new in ATTACKS:
        rejected = reject_mutation(base, path, old, new)
        rows.append({"name": name, "rejected": True,
                     "checker": rejected["checker"],
                     "independent": rejected["independent"]})
    compile_rows = compile_log_tests()
    results = HW / "results"
    namespaces = [
        results / ".m1845_c2_fresh_mapped_production_energy_attempt_consumed",
        results / "m1845_c2_fresh_mapped_production_energy_r1_20260902",
        results / "m1845_c2_fresh_mapped_production_energy_r1_20260902.failed_or_incomplete.quarantine",
        results / "m1845_c2_fresh_mapped_production_energy_r1_20260902.private_build.unsealed_do_not_cite",
    ]
    need(not any(path.exists() for path in namespaces), "M1845 namespace not fresh")
    release = HW / "contracts/m1849_m1848_m1845_c2_fresh_mapped_production_energy_launch_release_r1_20260902.json"
    need(not release.exists() and not Path(str(release) + ".sha256").exists()
         and not Path(str(release) + ".sha256.seal.sha256").exists(),
         "M1849 release created before review")
    print(json.dumps({
        "schema": "m1848_m1845_c2_energy_independent_source_hammer_r1_v1",
        "status": "PASS_M1848_INDEPENDENT_SOURCE_AND_MUTATION_HAMMER",
        "positive": positive,
        "semantic_attacks": len(rows),
        "semantic_rejected": len(rows),
        "inventory_synchronized": True,
        "rows": rows,
        "compile_log_checks": compile_rows,
        "compile_negative_checks": 16,
        "compile_negative_rejected": 16,
        "m1845_namespaces_fresh": True,
        "m1849_release_absent": True,
        "eda_runs": 0,
        "license_queries": 0,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
