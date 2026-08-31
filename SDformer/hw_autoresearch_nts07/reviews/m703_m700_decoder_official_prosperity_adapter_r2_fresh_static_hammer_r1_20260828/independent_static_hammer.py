#!/usr/bin/env python3
"""Fresh static hammer for M700; never run the official replay, GPU, or EDA."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
AUTHOR = HW / "reviews/m700_decoder_official_prosperity_adapter_r2_author_handoff_r1_20260828"
RUNNER = HW / "scripts/run_m700_h67_ep35_decoder_official_prosperity_iso_workload_r2.py"
CONTRACT = HW / "contracts/m700_h67_ep35_decoder_official_prosperity_iso_workload_contract_r2_20260828.json"
TESTS = HW / "system_simulator/tests/test_m700_decoder_official_prosperity_adapter_r2.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
CANONICAL = HW / "results/m700_h67_ep35_decoder_official_prosperity_dev_r2_20260828"
LOCK = HW / "results/.m700_h67_ep35_decoder_official_prosperity_dev_r2.single_writer.lock"

EXPECTED = {
    "runner": "a5e7113b3c56354bbcbd8196837ab444ed1830ab66c55f0d3610dd78cf713098",
    "contract": "c340a167cc3641a468327697b57d43197ddb6699d3ca744d0a9d9f7f26c1bb65",
    "tests": "0f587d0ca691ff0d5f53eeaefd56f6f9e8069d16d05931b633e9561ee63093ba",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "author_manifest": "a1babfd1c57fe06138d68f1886f3f675889d039bea20f025a38b801e1582d426",
    "author_outer": "e97f7913f3136a2450bab068a3b3e65f0213f4e7e2d9b03d9b320ea51b780147",
    "m697_review": "f5fd5a172cd011654224aa0591df30518c0753d9b563f88535ff42ad39188dd1",
    "m686_manifest": "c06de650b50db92dd0c374b57f0ce3ea72cfb3dcd18a369aea7d552341e5bb33",
    "m692_review": "5088e36fa935536766f51f4e58c198d16f49ac3fe415b2f3d6432b184a36f49f",
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict(path: Path):
    def reject(value: str):
        raise ValueError("nonfinite " + value)

    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate key " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def safe_member(value: str) -> PurePosixPath:
    member = PurePosixPath(value)
    need(member.parts and not member.is_absolute() and
         member.parts[0] not in ("", ".") and ".." not in member.parts,
         "unsafe member")
    return member


def trusted(root: Path, member: str) -> Path:
    member_path = safe_member(member)
    cursor = root
    for index, part in enumerate(member_path.parts):
        cursor /= part
        observed = os.lstat(cursor)
        need(not stat.S_ISLNK(observed.st_mode), "symlink component")
        need(stat.S_ISREG(observed.st_mode) if index + 1 == len(member_path.parts)
             else stat.S_ISDIR(observed.st_mode), "member kind")
    need(cursor.resolve(strict=True).is_relative_to(root.resolve(strict=True)),
         "resolved escape")
    return cursor


def verify_seal(root: Path, expected_manifest: str | None = None,
                expected_outer: str | None = None) -> dict:
    manifest = trusted(root, "SHA256SUMS")
    outer = trusted(root, "SHA256SUMS.seal.sha256")
    if expected_manifest:
        need(sha(manifest) == expected_manifest, "manifest file SHA")
    if expected_outer:
        need(sha(outer) == expected_outer, "outer file SHA")
    need(outer.read_text(encoding="utf-8").strip().split("  ", 1) ==
         [sha(manifest), "SHA256SUMS"], "outer content")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, member = line.split("  ", 1)
        need(member not in listed, "duplicate sealed member")
        need(sha(trusted(root, member)) == digest, "member SHA " + member)
        listed.add(member)
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in
        {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    need(actual == listed, "sealed population")
    return {"members": len(listed), "manifest_file_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def import_runner():
    spec = importlib.util.spec_from_file_location("m703_target", RUNNER)
    need(spec is not None and spec.loader is not None, "runner spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_result(module, sample, bit_seed, product_seed, m):
    phases = []
    for bank in m.PHASE_ORDER:
        bit_raw = {field: bit_seed + bank for field in m.COUNTER_FIELDS}
        product_raw = {field: product_seed + 2 * bank for field in m.COUNTER_FIELDS}
        bit_raw["total_cycles"] = 1000 + 100 * sample + 10 * module + bank
        product_raw["total_cycles"] = 400 + 20 * sample + 3 * module + bank
        bit_raw["num_ops"] = 96 * (200 + sample + module + bank)
        product_raw["num_ops"] = 96 * (130 + sample + module + bank)
        bit = m.derived(bit_raw, 96)
        product = m.derived(product_raw, 96)
        phases.append({
            "phase_bank": bank,
            "modes": {"bit": bit, "product": product},
            "support_accounting": {
                "active_tap_events": 11 + sample + module + bank,
                "valid_tap_slots_per_time": 101 + bank,
                "valid_tap_slots_all_time": 1001 + 10 * sample + bank,
                "active_products": 96 * (11 + sample + module + bank),
                "structural_boundary_zeros_are_not_data_sparsity": True,
            },
            "product_vs_bit_speedup": bit["total_cycles"] /
                product["total_cycles"],
        })
    return {"sample_id": sample, "module_index": module, "phases": phases}


def verify_phase_population(rows, m):
    result = m.aggregate_breakdowns(rows)
    need(all(f"phase:{bank}" in result for bank in m.PHASE_ORDER),
         "missing phase aggregate")
    overall = result["overall"]
    for bank in m.PHASE_ORDER:
        phase = result[f"phase:{bank}"]
        need(phase["phase_bank"] == bank, "phase identity")
        need(phase["aggregate_cycle_ratio_speedup"] ==
             phase["bit"]["total_cycles"] /
             phase["product"]["total_cycles"], "phase ratio-of-sums")
    for mode in ("bit", "product"):
        for field, expected in overall[mode].items():
            need(sum(result[f"phase:{bank}"][mode][field]
                     for bank in m.PHASE_ORDER) == expected,
                 "counter conservation " + mode + "/" + field)
    for field, expected in overall["mapped_support_accounting"].items():
        need(sum(result[f"phase:{bank}"]["mapped_support_accounting"][field]
                 for bank in m.PHASE_ORDER) == expected,
             "support conservation " + field)
    need(sum(result[f"phase:{bank}"]["support_calls_per_mode"]
             for bank in m.PHASE_ORDER) == overall["support_calls_per_mode"],
         "call conservation")
    return result


def expect_failure_receipt(m, stage: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="m703_failure_") as raw:
        directory = Path(raw) / "receipt"
        directory.mkdir()
        marker = RuntimeError("injected " + stage)
        try:
            m.run_with_failure_receipt(
                directory, {"name": stage},
                lambda: (_ for _ in ()).throw(marker),
            )
        except RuntimeError as error:
            need(str(error) == str(marker), "injected failure identity")
        else:
            raise AssertionError("fault injection did not fail")
        seal = m.verify_double_seal(directory)
        receipt = m.strict_json(directory / "FAILED.json")
        need(receipt["stage"] == stage and receipt["workers"] == 3 and
             receipt["canonical_output_admitted"] is False and
             receipt["cycles_admitted"] is False and
             receipt["speedup_admitted"] is False,
             "failure receipt boundary")
        return {"stage": stage, **seal}


def main() -> None:
    checks = []
    need(sha(RUNNER) == EXPECTED["runner"], "runner SHA")
    need(sha(CONTRACT) == EXPECTED["contract"], "contract SHA")
    need(sha(TESTS) == EXPECTED["tests"], "tests SHA")
    need(sha(DOCS359) == EXPECTED["docs359"], "docs359 SHA")
    checks.append("runner_contract_tests_docs359_exact_sha")

    author_seal = verify_seal(AUTHOR, EXPECTED["author_manifest"],
                              EXPECTED["author_outer"])
    author = strict(AUTHOR / "author_handoff.json")
    need(author["status"] ==
         "STATIC_AUTHOR_R2_READY__FRESH_HAMMER_REQUIRED_BEFORE_OFFICIAL_REPLAY",
         "author status")
    need(author["claim_boundary"]["official_simulator_run"] is False and
         author["claim_boundary"]["cycles"] is False and
         author["claim_boundary"]["speedup"] is False,
         "author claim boundary")
    checks.append("author_double_seal_exact_population_and_static_boundary")

    contract = strict(CONTRACT)
    need(contract["execution_gate"]["required_status"] ==
         "GO_M700_FULL_OFFICIAL_CPU_REPLAY__P0_0_P1_0" and
         contract["execution_gate"]["workers_fixed"] == 3,
         "execution gate")
    need(contract["mapping"]["phase_order"] == [3, 2, 1, 0], "phase order")
    need(contract["mapping"]["phase_taps"] == {
        "3": [[0, 0], [0, 2], [2, 0], [2, 2]],
        "2": [[0, 1], [2, 1]], "1": [[1, 0], [1, 2]],
        "0": [[1, 1]],
    }, "phase taps")
    need(contract["d1_policy"]["official_exact_subset"] == [0, 2, 3] and
         contract["d1_policy"]["exact_decoder_complete_cycles"] is None and
         contract["d1_policy"]["exact_decoder_complete_speedup"] is None,
         "D1 boundary")
    checks.append("phase_mapping_worker3_and_d1_exact_population_contract")

    expected_mkn = {
        0: ((3000, 12, 72), (6144, 3072, 3072, 1536), (384, 3, 0)),
        1: ((12000, 47, 32), (3080, 1540, 1540, 770), (192, 2, 64)),
        2: ((48000, 188, 128), (1544, 772, 772, 386), (96, 1, 32)),
        3: ((192000, 750, 0), (776, 388, 388, 194), (96, 1, 32)),
    }
    for row in contract["decoder_modules"]:
        module = row["module_index"]
        t, _b, c, h, w = row["input_shape"]
        n = row["output_channels"]
        m_dim = t * h * w
        observed = (
            (m_dim, math.ceil(m_dim / 256), math.ceil(m_dim / 256) * 256 - m_dim),
            tuple(taps * c for taps in (4, 2, 2, 1)),
            (n, math.ceil(n / 128), math.ceil(n / 128) * 128 - n),
        )
        need(observed == expected_mkn[module], "MKN arithmetic")
    checks.append("independent_partial_mkn_arithmetic")

    official = Path(contract["frozen_inputs"]["official_prosperity_repo"]["path"])
    entry = contract["frozen_inputs"]["official_prosperity_repo"]
    commit = subprocess.check_output(
        ["git", "-C", str(official), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(official), "status", "--porcelain",
         "--untracked-files=all"], text=True
    ).strip()
    need(commit == entry["commit"] and not dirty, "official repo identity")
    for member, digest in entry["files"].items():
        need(sha(trusted(official, member)) == digest,
             "official source " + member)
    checks.append("official_commit_clean_and_source_sha")

    m = import_runner()
    identity, exact_records, diagnostic_records = m.preflight(contract)
    need(len(exact_records) == 30 and len(diagnostic_records) == 10,
         "preflight populations")
    need({row["module_index"] for row in exact_records} == {0, 2, 3} and
         {row["module_index"] for row in diagnostic_records} == {1},
         "exact/diagnostic separation")
    need(all(row["admission_role"] == "EXACT_BINARY_OFFICIAL_SUBSET"
             for row in exact_records) and
         all(row["admission_role"] ==
             "SCALED_BINARY_OPPORTUNITY_DIAGNOSTIC_ONLY"
             for row in diagnostic_records), "admission roles")
    need(identity["m697_review"]["sha256"] == EXPECTED["m697_review"] and
         identity["m686_package"]["manifest_sha256"] == EXPECTED["m686_manifest"] and
         identity["m692"]["review_sha256"] == EXPECTED["m692_review"],
         "reverse lineage")
    need(m._MAPPER is None and m._FC is None and m._SIMULATOR is None,
         "static preflight imported engine")
    checks.append("full_frozen_preflight_no_mapper_or_official_engine")

    exact_rows = [fake_result(module, sample, 20 + sample, 7 + module, m)
                  for sample in range(3) for module in (0, 2, 3)]
    diag_rows = [fake_result(1, sample, 31 + sample, 9, m)
                 for sample in range(3)]
    exact_aggregates = verify_phase_population(exact_rows, m)
    diagnostic_aggregates = verify_phase_population(diag_rows, m)
    report = m.build_report(contract, {}, exact_rows, diag_rows, {}, workers=3)
    need(report["official_binary_support_subset"]["aggregates"] == exact_aggregates,
         "exact aggregate placement")
    need(report["d1_scaled_binary_opportunity_diagnostic"]["aggregates"] ==
         diagnostic_aggregates, "diagnostic aggregate placement")
    need(report["exact_decoder_complete"]["admitted"] is False and
         report["exact_decoder_complete"]["total_cycles"] is None and
         report["exact_decoder_complete"]["product_vs_bit_speedup"] is None and
         report["d1_scaled_binary_opportunity_diagnostic"][
             "folded_weight_deployment_admitted"] is False,
         "report boundary")
    need(exact_aggregates["phase:3"]["bit"]["total_cycles"] !=
         diagnostic_aggregates["phase:3"]["bit"]["total_cycles"],
         "population cross-contamination")
    checks.append("separate_phase_3_2_1_0_ratio_of_sums_and_full_conservation")

    malformed = copy.deepcopy(exact_rows)
    malformed[0]["phases"].pop()
    try:
        m.aggregate_breakdowns(malformed)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing phase accepted")
    for workers in (0, 1, 2, 4):
        try:
            m.execute_records([], workers=workers)
        except RuntimeError:
            pass
        else:
            raise AssertionError("worker drift accepted")
    checks.append("missing_phase_and_worker_drift_attacks_blocked")

    base = {field: 3 for field in m.COUNTER_FIELDS}
    base["dram_reads"] = 100000
    base["dram_writes"] = 0
    expanded = m.expand_exact_n128(base, m_dim=3000, k_dim=6144, n_dim=384)
    need(expanded["compute_cycles"] == base["compute_cycles"] * 3,
         "D0 expansion")
    for partial in (96, 192):
        try:
            m.expand_exact_n128(base, m_dim=3000, k_dim=1544, n_dim=partial)
        except RuntimeError:
            pass
        else:
            raise AssertionError("partial N expansion accepted")
    source = RUNNER.read_text(encoding="utf-8")
    for fragment in (
        "direct = run_official(", "if module == 0:",
        "M700 D0 direct-vs-N128x3 miter failed",
        'name = f"h67_decoder_d{module_index}_phase{bank}_polyphase"',
        'require(not name.endswith("_fc")',
    ):
        need(fragment in source, "source boundary " + fragment)
    checks.append("direct_full_n_partial_n_rejection_d0_miter_and_fc_name_guard")

    failure_receipts = [expect_failure_receipt(m, stage) for stage in (
        "ARGUMENT_AND_AUTHORIZATION_PREFLIGHT",
        "EXECUTE_EXACT_D0_D2_D3_POPULATION",
        "EXECUTE_D1_DIAGNOSTIC_POPULATION",
        "POST_EXECUTION_IDENTITY_RECHECK",
        "ATOMIC_PUBLICATION_AND_POST_VERIFY",
    )]
    checks.append("all_five_failure_stages_emit_double_sealed_receipts")

    with tempfile.TemporaryDirectory(prefix="m703_publish_") as raw:
        tmp = Path(raw)
        lock = m.acquire_single_writer_lock(tmp / m.SINGLE_WRITER_LOCK.name)
        try:
            try:
                m.acquire_single_writer_lock(tmp / m.SINGLE_WRITER_LOCK.name)
            except RuntimeError:
                pass
            else:
                raise AssertionError("second writer acquired lock")
            m.validate_single_writer_lock(lock)

            output = tmp / "canonical"
            output.symlink_to(tmp / "missing")
            try:
                m.atomic_publish(output, {"schema": "x"}, {"schema": "r"}, lock)
            except RuntimeError:
                pass
            else:
                raise AssertionError("dangling canonical leaf accepted")
            output.unlink()

            original_write = m.write_double_seal
            armed = {"value": True}

            def concurrent_leaf(directory):
                original_write(directory)
                if armed["value"]:
                    armed["value"] = False
                    output.symlink_to(tmp / "missing_after_first_check")

            m.write_double_seal = concurrent_leaf
            try:
                m.atomic_publish(output, {"schema": "x"}, {"schema": "r"}, lock)
            except RuntimeError:
                pass
            else:
                raise AssertionError("pre-rename lexists recheck missing")
            finally:
                m.write_double_seal = original_write
            need(os.path.lexists(output), "injected leaf disappeared")
            output.unlink()
        finally:
            m.release_single_writer_lock(lock)
        need(not os.path.lexists(tmp / m.SINGLE_WRITER_LOCK.name),
             "lock release")
    checks.append("o_excl_singlewriter_inode_and_pre_rename_lexists_recheck")

    need("M700_EXPECTED_STATIC_REVIEW_OUTER_SEAL_FILE_SHA256" in source and
         "--allow-full-official-replay" in source and
         "target.get(\"runner_sha256\")" in source and
         "target.get(\"contract_sha256\")" in source and
         "target.get(\"test_sha256\")" in source and
         "authorization_after == authorization" in source and
         "identity_after == identity" in source,
         "authorization/reverse binding source")
    boundary = contract["claim_boundary"]
    for key in ("monolithic_convtranspose_latency", "ours", "same_resource_local",
                "full_decoder_latency", "full_network", "system_speedup",
                "energy", "ppa", "accuracy_change", "date_headline"):
        need(boundary[key] is False, "claim upgrade " + key)
    checks.append("cli_env_reverse_sha_postcheck_and_claim_boundaries")

    need(not os.path.lexists(CANONICAL) and not os.path.lexists(LOCK),
         "canonical or lock appeared during static hammer")
    result = {
        "schema": "m703_m700_decoder_adapter_r2_fresh_static_independent_audit_v1",
        "status": "PASS_STATIC_ATTACKS__NO_OFFICIAL_REPLAY_GPU_OR_EDA",
        "checks": checks,
        "author_seal": author_seal,
        "phase_fixture": {
            "exact_records": len(exact_rows),
            "diagnostic_records": len(diag_rows),
            "exact_support_calls_per_mode": exact_aggregates["overall"]["support_calls_per_mode"],
            "diagnostic_support_calls_per_mode": diagnostic_aggregates["overall"]["support_calls_per_mode"],
            "phase_order": list(m.PHASE_ORDER),
        },
        "fault_injection": failure_receipts,
        "prelaunch_state": {"canonical_absent": True, "lock_absent": True},
        "prohibited_execution": {
            "production_mapper_run": False,
            "official_simulator_imported": False,
            "official_run_fc_called": False,
            "official_replay": False, "gpu": False, "eda": False,
            "cycles_generated": False, "speedup_generated": False,
        },
    }
    (REVIEW / "independent_audit_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
