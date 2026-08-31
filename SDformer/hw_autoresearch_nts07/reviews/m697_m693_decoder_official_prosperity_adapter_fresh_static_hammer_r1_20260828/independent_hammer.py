#!/usr/bin/env python3
"""Receipt-blind M697 static hammer for the M693 decoder adapter.

This program deliberately never calls the production mapper or the official
Prosperity Simulator.  It independently checks the frozen receipts and input
population, then attacks only static/unit surfaces.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
AUTHOR = HW / "reviews/m693_decoder_official_prosperity_adapter_author_handoff_r1_20260828"
RUNNER = HW / "scripts/run_m693_h67_ep35_decoder_official_prosperity_iso_workload.py"
CONTRACT = HW / "contracts/m693_h67_ep35_decoder_official_prosperity_iso_workload_contract_r1_20260828.json"
TEST = HW / "system_simulator/tests/test_m693_decoder_official_prosperity_adapter.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "runner": "9b1f0adf72db9ddf496a753f10574e3983feb5e452abb8a81f7ba79c172fe64e",
    "contract": "e675a1a8d156d270ee729e1671d4a5b606626ee2957eacf89f478b3e937c4e98",
    "test": "d7a1c7ccc37690d75d2ef22901dced8c0ed00e9b21036ed8c7ac3bacf3057ad8",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "author_manifest": "07212204579634eab6c688fc81812622de027680b4b2204918d760bcdad12adf",
    "author_outer": "da75adaec4209affd285760c72158877aef438425b67647011ba4b534bbd2a73",
    "m692_review": "5088e36fa935536766f51f4e58c198d16f49ac3fe415b2f3d6432b184a36f49f",
    "m686_manifest": "c06de650b50db92dd0c374b57f0ce3ea72cfb3dcd18a369aea7d552341e5bb33",
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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

    return json.loads(path.read_text(encoding="utf-8"),
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
        if index + 1 == len(member_path.parts):
            need(stat.S_ISREG(observed.st_mode), "non-regular leaf")
        else:
            need(stat.S_ISDIR(observed.st_mode), "non-directory parent")
    need(cursor.resolve(strict=True).is_relative_to(root.resolve(strict=True)),
         "resolved escape")
    return cursor


def verify_seal(root: Path, expected_manifest: str | None = None,
                expected_outer: str | None = None) -> dict[str, str]:
    manifest = trusted(root, "SHA256SUMS")
    outer = trusted(root, "SHA256SUMS.seal.sha256")
    if expected_manifest:
        need(sha(manifest) == expected_manifest, "manifest file SHA")
    if expected_outer:
        need(sha(outer) == expected_outer, "outer file SHA")
    outer_tokens = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    need(outer_tokens == [sha(manifest), "SHA256SUMS"], "outer content")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, member = line.split("  ", 1)
        need(member not in listed, "duplicate sealed member")
        need(sha(trusted(root, member)) == digest, "sealed member SHA")
        listed.add(member)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
        and path.relative_to(root).as_posix() not in
        {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    need(actual == listed, "sealed population")
    return {"manifest_file_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def import_runner():
    spec = importlib.util.spec_from_file_location("m697_target", RUNNER)
    need(spec is not None and spec.loader is not None, "runner import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_payload_lattice(contract: dict) -> dict:
    package = HW / contract["frozen_inputs"]["m686_package"]["path"]
    package_entry = contract["frozen_inputs"]["m686_package"]
    top = verify_seal(package, package_entry["manifest_file_sha256"],
                      package_entry["outer_seal_file_sha256"])
    for name in ("runtime_receipt", "weights"):
        expected = package_entry["nested_seals"][name]
        verify_seal(package / name, expected["manifest_file_sha256"],
                    expected["outer_seal_file_sha256"])
    manifest_path = trusted(package, "manifest.json")
    need(sha(manifest_path) == EXPECTED["m686_manifest"], "M686 manifest")
    manifest = strict(manifest_path)
    records = manifest["d0_d2_d3_binary_records"] + manifest["d1_records"]
    need(len(records) == 40, "40-cell record count")
    seen = set()
    module_counts = {index: 0 for index in range(4)}
    popcounts = {index: 0 for index in range(4)}
    for row in records:
        sample = row["sample_id"]
        module = row["module_index"]
        need((sample, module) not in seen, "duplicate cell")
        seen.add((sample, module))
        need(row["name"] == contract["decoder_modules"][module]["name"],
             "module name")
        need(row["input_shape"] == contract["decoder_modules"][module]["input_shape"],
             "module shape")
        if module == 1:
            identity = row["theta_binary_candidate"]
            need(row["route"] == "EXACT_SCALED_BINARY_BITPACK", "D1 route")
            need(identity["theta_gate_pass"] is True and
                 identity["other_finite_count"] == 0 and
                 identity["nonfinite_count"] == 0, "D1 mask gate")
            need(row["folded_weight_miter"]["bit_exact"] is False,
                 "D1 nonexact boundary")
            expected_ones = identity["theta_count"]
        else:
            identity = row["input"]
            need(row["route"] == "EXACT_BINARY_BITPACK", "exact route")
            expected_ones = identity["one_count"]
        path = trusted(package, row["relative_path"])
        raw = np.fromfile(path, dtype=np.uint8)
        elements = math.prod(row["input_shape"])
        need(raw.size == (elements + 7) // 8, "packed length")
        if elements & 7:
            need(int(raw[-1]) >> (elements & 7) == 0, "packed tail")
        ones = int(np.unpackbits(raw, bitorder="little")[:elements].sum(dtype=np.uint64))
        need(ones == expected_ones, "payload popcount")
        need(sha(path) == identity["packed_sha256"], "payload SHA")
        module_counts[module] += 1
        popcounts[module] += ones
    need(seen == {(sample, module) for sample in range(10) for module in range(4)},
         "40-cell lattice")
    need(module_counts == {0: 10, 1: 10, 2: 10, 3: 10}, "module count")
    return {"top_seal": top, "module_counts": module_counts,
            "popcounts": popcounts}


def main() -> None:
    checks = []

    author_seal = verify_seal(AUTHOR, EXPECTED["author_manifest"],
                              EXPECTED["author_outer"])
    checks.append("author_double_seal_and_exact_population")
    author = strict(AUTHOR / "author_handoff.json")
    need(author["status"] ==
         "STATIC_AUTHOR_READY__FRESH_HAMMER_REQUIRED_BEFORE_OFFICIAL_REPLAY",
         "author status")
    need(author["claim_boundary"]["official_simulator_run"] is False and
         author["claim_boundary"]["cycles"] is False and
         author["claim_boundary"]["speedup"] is False,
         "author static boundary")

    need(sha(RUNNER) == EXPECTED["runner"], "runner SHA")
    need(sha(CONTRACT) == EXPECTED["contract"], "contract SHA")
    need(sha(TEST) == EXPECTED["test"], "test SHA")
    need(sha(DOCS359) == EXPECTED["docs359"], "docs359 SHA")
    checks.append("runner_contract_test_docs359_exact_sha")

    contract = strict(CONTRACT)
    need(contract["mapping"]["phase_order"] == [3, 2, 1, 0], "phase order")
    need(contract["mapping"]["phase_taps"] == {
        "3": [[0, 0], [0, 2], [2, 0], [2, 2]],
        "2": [[0, 1], [2, 1]], "1": [[1, 0], [1, 2]],
        "0": [[1, 1]],
    }, "tap order")
    need(contract["mapping"]["k_order"] ==
         "tap-major then source-channel: k=tap_index*Cin+cin", "K order")
    expected_tiles = {
        0: ((3000, 12, 72), (384, 3, 0), (6144, 3072, 3072, 1536)),
        1: ((12000, 47, 32), (192, 2, 64), (3080, 1540, 1540, 770)),
        2: ((48000, 188, 128), (96, 1, 32), (1544, 772, 772, 386)),
        3: ((192000, 750, 0), (96, 1, 32), (776, 388, 388, 194)),
    }
    for row in contract["decoder_modules"]:
        module = row["module_index"]
        t_dim, _batch, channels, height, width = row["input_shape"]
        output = row["output_channels"]
        m_dim = t_dim * height * width
        m_tuple = (m_dim, math.ceil(m_dim / 256),
                   math.ceil(m_dim / 256) * 256 - m_dim)
        n_tuple = (output, math.ceil(output / 128),
                   math.ceil(output / 128) * 128 - output)
        k_tuple = tuple(taps * channels for taps in (4, 2, 2, 1))
        need((m_tuple, n_tuple, k_tuple) == expected_tiles[module],
             "partial M/K/N ledger")
    checks.append("phase_tap_k_and_partial_mkn_independent_arithmetic")

    payload = independent_payload_lattice(contract)
    checks.append("m686_top_nested_seals_40_cell_popcount_d1_nonexact")

    m692_entry = contract["frozen_inputs"]["m692_review_directory"]
    m692_root = HW / m692_entry["path"]
    verify_seal(m692_root, m692_entry["manifest_file_sha256"],
                m692_entry["outer_seal_file_sha256"])
    m692 = strict(m692_root / "review.json")
    need(sha(m692_root / "review.json") == EXPECTED["m692_review"], "M692 review")
    need(m692["status"] == "GO_M672_STATIC_ADAPTER_INPUT_ONLY" and
         m692["severity"] == {"p0": 0, "p1": 0, "p2": 0} and
         m692["go"] is True, "M692 admission")
    checks.append("m692_external_root_seal_and_scope")

    official = Path(contract["frozen_inputs"]["official_prosperity_repo"]["path"])
    entry = contract["frozen_inputs"]["official_prosperity_repo"]
    commit = subprocess.check_output(
        ["git", "-C", str(official), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(official), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    need(commit == entry["commit"] and not dirty, "official commit/clean")
    for member, digest in entry["files"].items():
        need(sha(trusted(official, member)) == digest, "official source " + member)
    checks.append("official_repo_commit_clean_source_sha")

    module = import_runner()
    identity, exact_records, diagnostic_records = module.preflight(contract)
    need(len(exact_records) == 30 and len(diagnostic_records) == 10,
         "frozen preflight population")
    need(module._MAPPER is None and module._SIMULATOR is None and module._FC is None,
         "preflight imported prohibited engines")
    checks.append("frozen_preflight_no_mapper_or_official_import")

    with tempfile.TemporaryDirectory(prefix="m697_attacks_") as raw_tmp:
        tmp = Path(raw_tmp)
        duplicate = tmp / "dup.json"
        duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
        nonfinite = tmp / "nan.json"
        nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
        for path in (duplicate, nonfinite):
            try:
                module.strict_json(path)
            except RuntimeError:
                pass
            else:
                raise AssertionError("strict JSON attack accepted")
        for value in ("../x", "/tmp/x", "a/../../b", ".", ""):
            try:
                module.safe_member(value)
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsafe member accepted")
        sealed = tmp / "sealed"
        sealed.mkdir()
        (sealed / "member").write_text("x\n", encoding="utf-8")
        module.write_double_seal(sealed)
        (sealed / "extra").write_text("extra\n", encoding="utf-8")
        try:
            module.verify_double_seal(sealed)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsealed extra accepted")
        symlink_root = tmp / "symlink_root"
        symlink_root.mkdir()
        outside = tmp / "outside"
        outside.write_text("outside\n", encoding="utf-8")
        (symlink_root / "link").symlink_to(outside)
        try:
            module.trusted_file(symlink_root, "link", "attack")
        except RuntimeError:
            pass
        else:
            raise AssertionError("symlink accepted")
    checks.append("duplicate_nonfinite_path_symlink_extra_population_attacks")

    source = RUNNER.read_text(encoding="utf-8")
    need('name = f"h67_decoder_d{module_index}_phase{bank}_polyphase"' in source and
         'require(not name.endswith("_fc")' in source and
         "op = _FC(" in source, "fresh FC/name guard")
    need("direct = run_official(" in source and "if module == 0:" in source and
         "M693 D0 direct-vs-N128x3 miter failed" in source,
         "D0 direct/N128 miter source")
    try:
        module.expand_exact_n128(
            {field: 1 for field in module.COUNTER_FIELDS},
            m_dim=12000, k_dim=1540, n_dim=192,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("partial-N expansion accepted")
    checks.append("fresh_fc_name_guard_direct_full_n_and_partial_n_reject")

    # Static worker reasoning: executor.map preserves input order; the runner
    # additionally sorts by (sample,module), and worker-local official objects
    # are initialized once.  No official execution is performed here.
    need("executor.map(worker_run, records, chunksize=1)" in source and
         "return sorted(completed" in source and
         "1 <= workers <= 3" in source, "worker determinism guards")
    checks.append("worker_range_order_and_aggregation_static_guards")

    # Authorization needs all three gates: exact status/booleans, outer file
    # SHA supplied out-of-band, and reverse binding to runner/contract/test.
    need("M693_EXPECTED_STATIC_REVIEW_OUTER_SEAL_FILE_SHA256" in source and
         'args.allow_full_official_replay' in source and
         'target.get("runner_sha256")' in source and
         'target.get("contract_sha256")' in source and
         'target.get("test_sha256")' in source,
         "authorization reverse binding")
    checks.append("cli_outer_env_review_reverse_binding")

    # Atomic publication is covered by the unit test, and its implementation
    # uses fresh staging + seal + rename + post-publish quarantine.
    need("tempfile.mkdtemp" in source and "os.rename(staging, output)" in source and
         ".quarantine.post_publish_verification_failed" in source,
         "atomic publication")
    checks.append("atomic_publish_nonoverwrite_quarantine_static")

    # Contract/report boundary invariants.
    boundary = contract["claim_boundary"]
    for field in ("exact_decoder_complete", "monolithic_convtranspose_latency",
                  "ours", "same_resource_local", "full_decoder_latency",
                  "full_network", "system_speedup", "energy", "ppa",
                  "accuracy_change", "date_headline"):
        need(boundary[field] is False, "claim boundary " + field)
    fake_exact = [{"sample_id": 0, "module_index": 0, "phases": []}]
    fake_diag = [{"sample_id": 0, "module_index": 1, "phases": []}]
    # Build representative complete phase objects without official calls.
    def fake_row(sample: int, decoder: int):
        phases = []
        for bank in module.PHASE_ORDER:
            base = {field: 1 for field in module.COUNTER_FIELDS}
            base["total_cycles"] = 2
            base["compute_cycles"] = 1
            base["memory_stall_cycles"] = 1
            bit = module.derived(base, 1)
            product_mode = module.derived(dict(base), 1)
            phases.append({"phase_bank": bank,
                           "modes": {"bit": bit, "product": product_mode},
                           "product_vs_bit_speedup": 1.0})
        return {"sample_id": sample, "module_index": decoder,
                "phases": phases}
    report = module.build_report(contract, {}, [fake_row(0, 0)],
                                 [fake_row(0, 1)], {})
    need(report["exact_decoder_complete"]["total_cycles"] is None and
         report["exact_decoder_complete"]["product_vs_bit_speedup"] is None and
         report["claim_boundary"]["ours"] is False and
         report["claim_boundary"]["system_speedup"] is False and
         report["claim_boundary"]["date_headline"] is False,
         "report claim/null boundary")
    checks.append("exact_subset_d1_diagnostic_complete_null_claim_boundary")

    # Findings intentionally remain assertions about the frozen target; they
    # are not repairs and do not execute the workload.
    aggregate_keys = set(report["official_binary_support_subset"]["aggregates"])
    missing_phase_aggregate = not any(key.startswith("phase:") for key in aggregate_keys)
    need(missing_phase_aggregate, "expected frozen-target phase aggregate gap absent")
    need("exact_results = execute_records" in source and
         source.index("exact_results = execute_records") < source.index("atomic_publish(args.output"),
         "expected pre-publish execution ordering changed")
    no_outer_failure_receipt = (
        source.count("m693_failure_receipt_v1") == 1 and
        source.index("m693_failure_receipt_v1") > source.index("def atomic_publish")
    )
    need(no_outer_failure_receipt, "expected outer failure-receipt gap absent")

    output = {
        "schema": "m697_m693_static_independent_audit_v1",
        "status": "PASS_STATIC_ATTACKS__NO_OFFICIAL_EXECUTION__TWO_CONTRACT_GAPS_CONFIRMED",
        "author_seal": author_seal,
        "checks": checks,
        "payload": payload,
        "findings": {
            "missing_required_phase_aggregate": missing_phase_aggregate,
            "no_sealed_failure_receipt_before_atomic_publish": no_outer_failure_receipt,
            "worker_equivalence_evidence": "STATIC_ORDER_GUARDS_ONLY__NO_1_2_3_WORKER_OFFICIAL_RUN",
        },
        "prohibited_execution": {
            "production_mapper_run": False,
            "official_simulator_imported": False,
            "official_run_fc_called": False,
            "performance_generated": False,
            "gpu": False,
            "eda": False,
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
