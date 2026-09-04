#!/usr/bin/env python3
"""Independent static/synthetic engineering QA for M1579.

No production record is decoded and no 51.84-million-row replay is executed.
Production helpers are replaced with tiny in-memory fixtures before execute()
is exercised for staging, sealing, and release-boundary attacks.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/run_m1579_ep34_c1_same_ledger_cycle_model.py"
TEST = HW / "system_simulator/tests/test_m1579_ep34_c1_same_ledger_cycle_model.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINNED = {
    "source": "741e532b58e8b558fb2399fb1e407004ee247eca3a9c1716edcc30eea7336988",
    "test": "ca1541d2ef6aae4f7a170e143d84b72d51aca4e514468f1f2a669a835a5305aa",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

FINAL_EP34 = {
    "checkpoint_sha256": "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    "capture_manifest_sha256": "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d",
    "ordered_records_sha256": "5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def load_source() -> Any:
    spec = importlib.util.spec_from_file_location("m1581_bound_m1579", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M1579")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expect_reject(function, label: str) -> None:
    try:
        function()
    except (RuntimeError, FileExistsError, ValueError, TypeError):
        return
    raise RuntimeError(label + " did not fail closed")


def verify_result_seal(directory: Path) -> dict[str, str]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(), "missing result double seal")
    manifest_sha = sha256(manifest)
    require(outer.read_text(encoding="ascii").split() ==
            [manifest_sha, "SHA256SUMS"], "result outer seal drift")
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and fields[1] not in entries and
                "/" not in fields[1] and ".." not in fields[1],
                "result manifest malformed")
        entries[fields[1]] = fields[0]
        regular_exact(directory / fields[1], fields[0],
                      "result member " + fields[1])
    actual = set(path.name for path in directory.iterdir() if path.is_file() and
                 path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(actual == set(entries), "result seal coverage drift")
    return entries


def release_value(module: Any, output: Path, ledger: Path) -> dict[str, Any]:
    return {
        "schema": module.RELEASE_SCHEMA,
        "status": module.RELEASE_STATUS,
        "source_sha256": sha256(SOURCE),
        "output": str(output),
        "ledger": str(ledger),
        "cpu_runs": 1,
        "gpu_runs": 0,
        "eda_runs": 0,
        "maximum_workers": 3,
        "frozen_inputs": {
            "m1524": module.M1524_SHA256,
            "m528": module.M528_SHA256,
            "m505": module.M505_SHA256,
            "m504": module.M504_SHA256,
            "docs359": module.DOCS359_SHA256,
        },
    }


class Support:
    def __init__(self, sample: int, operator: int, active: int):
        self.sample = sample
        self.operator = operator
        self.active = active

    def sum(self) -> int:
        return self.active


class TinyM1524:
    MODULES = ("op0", "op1")

    def __init__(self, samples: int, operators: int, partitions: int,
                 rows: int):
        self.samples = samples
        self.operators = operators
        self.partitions = partitions
        self.rows = rows
        self.records = []
        for sample in range(samples):
            for operator in range(operators):
                active = sample + operator + 1
                self.records.append({"name": self.MODULES[operator],
                                     "sample": sample, "operator": operator,
                                     "input": {"active": active}})

    def collect_records(self):
        return list(self.records), {}

    def decode_support(self, record):
        return Support(record["sample"], record["operator"],
                       record["input"]["active"])

    def phase_masks(self, support, partition):
        base = ((support.sample + 1) * 0x1000 +
                (support.operator + 1) * 0x100 + partition * 0x10)
        return np.asarray([base + row for row in range(self.rows)],
                          dtype=np.uint16)

    @staticmethod
    def m528_compatible_lines(masks):
        return "".join("0000{:04x}\n".format(int(value))
                       for value in masks).encode("ascii")


def synthetic_ledger_order(module: Any) -> dict[str, Any]:
    original = {name: getattr(module, name) for name in
                ("SAMPLES", "OPERATORS", "PARTITIONS", "ROWS_PER_PHASE",
                 "PHASES", "SOURCE_ROWS", "LEDGER_BYTES", "M1524",
                 "load_modules")}
    samples, operators, partitions, rows = 2, 2, 3, 4
    tiny = TinyM1524(samples, operators, partitions, rows)
    try:
        module.SAMPLES = samples
        module.OPERATORS = operators
        module.PARTITIONS = partitions
        module.ROWS_PER_PHASE = rows
        module.PHASES = samples * operators * partitions
        module.SOURCE_ROWS = module.PHASES * rows
        module.LEDGER_BYTES = module.SOURCE_ROWS * module.ROW_BYTES
        module.M1524 = tiny
        module.load_modules = lambda: (tiny, SimpleNamespace())
        with tempfile.TemporaryDirectory(prefix="m1581_ledger_order.") as temp:
            path = Path(temp) / "rows.memh"
            identity = module.materialize_ledger(path)
            lines = path.read_text(encoding="ascii").splitlines()
            expected = []
            for sample in range(samples):
                for operator in range(operators):
                    for partition in range(partitions):
                        base = ((sample + 1) * 0x1000 +
                                (operator + 1) * 0x100 + partition * 0x10)
                        expected.extend("0000{:04x}".format(base + row)
                                        for row in range(rows))
            require(lines == expected, "support-to-ledger order drift")
            require(identity["rows"] == len(expected) and
                    identity["phase_order"] == "sample,operator,partition" and
                    identity["row_order"] == "timestep,output_y,output_x",
                    "ledger identity/order receipt drift")

            bad = TinyM1524(samples, operators, partitions, rows)
            bad.records[0], bad.records[1] = bad.records[1], bad.records[0]
            module.M1524 = bad
            module.load_modules = lambda: (bad, SimpleNamespace())
            expect_reject(lambda: module.materialize_ledger(
                Path(temp) / "bad_order.memh"), "record order mutation")
            return {"synthetic_rows": len(expected),
                    "synthetic_phases": samples * operators * partitions,
                    "exact_order": True, "record_order_mutation_rejected": True}
    finally:
        for name, value in original.items():
            setattr(module, name, value)


class FakePool:
    fields: tuple[str, ...] = ()

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def map(self, _function, phases, chunksize=1):
        del chunksize
        output = []
        for phase in phases:
            values = {name: np.zeros(1, dtype=np.int32) for name in self.fields}
            values.update({
                "row_count": np.asarray([4], dtype=np.int32),
                "input_nnz": np.asarray([5], dtype=np.int32),
                "active_rows": np.asarray([3], dtype=np.int32),
                "residual_nnz": np.asarray([2], dtype=np.int32),
                "exact_parent_rows": np.asarray([1], dtype=np.int32),
                "ideal_issue_cycles": np.asarray([3], dtype=np.int32),
                "parent_edges": np.asarray([2], dtype=np.int32),
                "dead_reads": np.asarray([1], dtype=np.int32),
                "dead_forwards": np.asarray([1], dtype=np.int32),
                "dead_writes": np.asarray([1], dtype=np.int32),
                "dead_elisions": np.asarray([2], dtype=np.int32),
            })
            output.append((int(phase), values))
        return output


class TinyM528:
    FIELD_NAMES = (
        "row_count", "input_nnz", "active_rows", "residual_nnz",
        "exact_parent_rows", "ideal_issue_cycles", "parent_edges",
        "dead_reads", "dead_forwards", "dead_writes", "dead_elisions")

    def __init__(self):
        self.array_ids: list[int] = []

    def cycle_row(self, arrays, sample, operator):
        self.array_ids.append(id(arrays))
        if operator is None:
            rows = (
                (100, 90, 50, 70, 60, 55),
                (300, 200, 100, 150, 120, 110),
            )
            values = rows[sample]
        else:
            base = 40 + sample * 20 + operator * 5
            values = (base * 2, base + 20, base, base + 15,
                      base + 10, base + 8)
        keys = (
            "m468_strong_zero_cycles",
            "m473_same_coordinate_bit_cycles",
            "m473_fused_concurrent_1r1w_ceiling_cycles",
            "m504_all_write_1rw_cycles",
            "m505_dead_write_only_1rw_cycles",
            "m505_combined_pvrf_1rw_cycles",
        )
        return dict(zip(keys, values))

    @staticmethod
    def ratio_fields(row):
        candidate = row["m505_dead_write_only_1rw_cycles"]
        ceiling = row["m473_fused_concurrent_1r1w_ceiling_cycles"]
        return {
            "speedup_vs_m468_strong_zero":
                row["m468_strong_zero_cycles"] / candidate,
            "speedup_vs_m473_same_coordinate_bit":
                row["m473_same_coordinate_bit_cycles"] / candidate,
            "port_tax_vs_m473_ceiling": candidate / ceiling - 1.0,
            "m504_to_dead_write_speedup":
                row["m504_all_write_1rw_cycles"] / candidate,
            "dead_to_combined_speedup":
                candidate / row["m505_combined_pvrf_1rw_cycles"],
        }


def synthetic_same_ledger_statistics(module: Any) -> dict[str, Any]:
    names = ("SAMPLES", "OPERATORS", "PARTITIONS", "ROWS_PER_PHASE",
             "PHASES", "SOURCE_ROWS", "LEDGER_BYTES", "CHUNKS", "M528",
             "load_modules", "ProcessPoolExecutor")
    original = {name: getattr(module, name) for name in names}
    tiny = TinyM528()
    try:
        module.SAMPLES = 2
        module.OPERATORS = 2
        module.PARTITIONS = 2
        module.ROWS_PER_PHASE = 4
        module.PHASES = 8
        module.SOURCE_ROWS = 32
        module.LEDGER_BYTES = 32 * module.ROW_BYTES
        module.CHUNKS = 1
        module.M528 = tiny
        module.load_modules = lambda: (SimpleNamespace(MODULES=("op0", "op1")), tiny)
        FakePool.fields = tiny.FIELD_NAMES
        module.ProcessPoolExecutor = FakePool
        with tempfile.TemporaryDirectory(prefix="m1581_replay.") as temp:
            ledger = Path(temp) / "rows.memh"
            ledger.write_bytes(b"0" * module.LEDGER_BYTES)
            summary, samples, operators = module.replay(ledger, 2)
        expected_ratio = (100 + 300) / (60 + 120)
        actual_ratio = summary["aggregate_cycles"][
            "speedup_vs_m468_strong_zero"]
        arithmetic_mean = summary["distribution"]["sample_major"]["ratios"][
            "speedup_vs_m468_strong_zero"]["arithmetic_mean"]
        require(math.isclose(actual_ratio, expected_ratio) and
                not math.isclose(actual_ratio, arithmetic_mean),
                "ratio-of-sums collapsed into mean-of-ratios")
        require(summary["ratio_semantics"] ==
                "ratio_of_sums_over_ten_ep34_samples" and
                summary["distribution"]["sample_major"]["cycles"][
                    "m468_strong_zero_cycles"]["count"] == 2 and
                summary["distribution"]["operator_isolated"]["cycles"][
                    "m468_strong_zero_cycles"]["count"] == 4,
                "distribution population drift")
        require(len(set(tiny.array_ids)) == 1 and len(samples) == 2 and
                len(operators) == 4,
                "baselines did not share one ledger-derived array set")
        return {"ratio_of_sums": actual_ratio,
                "mean_of_sample_ratios": arithmetic_mean,
                "same_array_identity_for_all_baselines": True,
                "sample_distribution_count": len(samples),
                "operator_distribution_count": len(operators)}
    finally:
        for name, value in original.items():
            setattr(module, name, value)


def synthetic_publication_and_release(module: Any) -> dict[str, Any]:
    names = ("M1524", "materialize_ledger", "replay", "os")
    original = {name: getattr(module, name) for name in names}
    fake_m1524 = SimpleNamespace(
        MODULES=("op0", "op1", "op2", "op3"),
        CHECKPOINT_SHA256=FINAL_EP34["checkpoint_sha256"],
        CAPTURE_MANIFEST_SHA256=FINAL_EP34["capture_manifest_sha256"],
        ORDERED_SHA256=FINAL_EP34["ordered_records_sha256"])
    replace_calls = []

    def fake_materialize(path):
        path.write_bytes(b"tiny-ledger\n")
        return {"path": path.name, "sha256": sha256(path),
                "bytes": path.stat().st_size, "rows": 1,
                "line_format": "synthetic", "phase_order":
                "sample,operator,partition", "row_order":
                "timestep,output_y,output_x",
                "captured_input_active_values": 1,
                "captured_input_active_values_by_operator": [1, 0, 0, 0]}

    def fake_replay(_ledger, _workers):
        return ({"aggregate_cycles": {
                    "m468_strong_zero_cycles": 20,
                    "m505_dead_write_only_1rw_cycles": 10,
                    "speedup_vs_m468_strong_zero": 2.0},
                 "ratio_semantics": "ratio_of_sums_over_ten_ep34_samples",
                 "distribution": {}, "conservation": {}, "traffic": {},
                 "capacity": {}},
                [{"sample": 0, "cycles": 10}],
                [{"sample": 0, "operator": 0, "cycles": 10}])

    try:
        module.M1524 = fake_m1524
        module.materialize_ledger = fake_materialize
        module.replay = fake_replay
        original_replace = module.os.replace

        def checked_replace(source, target):
            source_path = Path(source)
            target_path = Path(target)
            require(not target_path.exists() and source_path.parent == target_path.parent,
                    "publication is not fresh same-parent rename")
            require((source_path / "SHA256SUMS").is_file() and
                    (source_path / "SHA256SUMS.seal.sha256").is_file(),
                    "publication attempted before double seal")
            replace_calls.append((source_path.name, target_path.name))
            original_replace(source, target)

        module.os = SimpleNamespace(replace=checked_replace, fsync=module.os.fsync)
        with tempfile.TemporaryDirectory(prefix="m1581_publish.") as temp:
            base = Path(temp)
            output = base / "result"
            ledger = output / "rows.memh"
            release = base / "release.json"
            release.write_text(json.dumps(release_value(module, output, ledger)),
                               encoding="utf-8")
            first = module.execute(release, output, ledger, 3)
            entries = verify_result_seal(output)
            require(set(entries) == {"rows.memh",
                    "m1579_ep34_c1_same_ledger_cycle_model_result_r1.json",
                    "sample_major_cycles.csv", "operator_isolated_cycles.csv",
                    "RUN_COMPLETE.txt"}, "published result member set drift")
            archived = base / "first_publication"
            output.rename(archived)
            second = module.execute(release, output, ledger, 3)
            verify_result_seal(output)
            reusable = (first["aggregate_cycles"] == second["aggregate_cycles"] and
                        len(replace_calls) == 2)

            wrong_output = base / "wrong"
            expect_reject(lambda: module.verify_release(
                release, wrong_output, wrong_output / "rows.memh", 3),
                "release output mutation")
            outside_ledger = base / "outside.memh"
            expect_reject(lambda: module.execute(
                release, output, outside_ledger, 3),
                "ledger outside canonical output")
            expect_reject(lambda: module.verify_release(
                release, output, ledger, 4), "worker count above three")

            tampered = base / "tampered"
            shutil.copytree(output, tampered)
            member = tampered / "sample_major_cycles.csv"
            member.write_text(member.read_text(encoding="utf-8") + "tamper\n",
                              encoding="utf-8")
            expect_reject(lambda: verify_result_seal(tampered),
                          "sealed member mutation")

            return {"same_parent_atomic_replace_after_seal": True,
                    "published_members": sorted(entries),
                    "output_path_mutation_rejected": True,
                    "ledger_outside_output_rejected": True,
                    "workers_above_three_rejected": True,
                    "sealed_member_mutation_rejected": True,
                    "same_release_accepted_twice": reusable}
    finally:
        for name, value in original.items():
            setattr(module, name, value)


def release_toctou(module: Any) -> bool:
    names = ("M1524", "materialize_ledger", "replay")
    original = {name: getattr(module, name) for name in names}
    fake_m1524 = SimpleNamespace(
        MODULES=("op0", "op1", "op2", "op3"),
        CHECKPOINT_SHA256=FINAL_EP34["checkpoint_sha256"],
        CAPTURE_MANIFEST_SHA256=FINAL_EP34["capture_manifest_sha256"],
        ORDERED_SHA256=FINAL_EP34["ordered_records_sha256"])
    try:
        module.M1524 = fake_m1524

        def fake_materialize(path):
            path.write_bytes(b"tiny-ledger\n")
            return {"path": path.name, "sha256": sha256(path), "bytes": 12,
                    "rows": 1, "line_format": "synthetic",
                    "phase_order": "sample,operator,partition",
                    "row_order": "timestep,output_y,output_x",
                    "captured_input_active_values": 1,
                    "captured_input_active_values_by_operator": [1, 0, 0, 0]}

        module.materialize_ledger = fake_materialize
        with tempfile.TemporaryDirectory(prefix="m1581_toctou.") as temp:
            base = Path(temp)
            output = base / "result"
            ledger = output / "rows.memh"
            release = base / "release.json"
            value = release_value(module, output, ledger)
            release.write_text(json.dumps(value), encoding="utf-8")

            def mutate_then_replay(_ledger, _workers):
                changed = dict(value)
                changed["cpu_runs"] = 99
                release.write_text(json.dumps(changed), encoding="utf-8")
                return ({"aggregate_cycles": {"x_cycles": 1},
                         "ratio_semantics": "ratio_of_sums_over_ten_ep34_samples",
                         "distribution": {}, "conservation": {}, "traffic": {},
                         "capacity": {}},
                        [{"sample": 0}], [{"operator": 0}])

            module.replay = mutate_then_replay
            result = module.execute(release, output, ledger, 1)
            require(json.loads(release.read_text())["cpu_runs"] == 99 and
                    result["identity"]["release_sha256"] == sha256(release),
                    "release TOCTOU witness drift")
            return True
    finally:
        for name, value in original.items():
            setattr(module, name, value)


def main() -> int:
    regular_exact(SOURCE, PINNED["source"], "M1579 source")
    regular_exact(TEST, PINNED["test"], "M1579 test")
    regular_exact(DOC359, PINNED["docs359"], "docs/359")
    module = load_source()
    source_text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    imports = {alias.name.split(".")[0]
               for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom))
               for alias in node.names}
    require({"torch", "socket", "requests", "paramiko"}.isdisjoint(imports) and
            all(token not in source_text for token in
                ("dc_shell", "vcs ", "ssh ", "time.time(", "perf_counter(")),
            "forbidden execution/timing dependency present")

    audit = module.source_audit()
    m1524, m528 = module.load_modules()
    require(audit["checkpoint_sha256"] == FINAL_EP34["checkpoint_sha256"] and
            audit["capture_manifest_sha256"] ==
                FINAL_EP34["capture_manifest_sha256"] and
            m1524.ORDERED_SHA256 == FINAL_EP34["ordered_records_sha256"] and
            audit["geometry"] == {"samples": 10, "operators": 4,
                "partitions": 432, "rows_per_phase": 3000,
                "source_rows": 51_840_000, "ledger_bytes": 466_560_000} and
            audit["old_ep35_cycles_reusable"] is False and
            audit["production"] is False,
            "final ep34 identity/geometry drift")
    require(sha256(module.M528_PATH) == module.M528_SHA256 and
            sha256(module.M505_PATH) == module.M505_SHA256 and
            sha256(module.M504_PATH) == module.M504_SHA256 and
            tuple(m528.FIELD_NAMES) and
            "m528.worker_phase(index)" in source_text and
            "m528.cycle_row(arrays, sample, None)" in source_text and
            "m528.ratio_fields(totals)" in source_text,
            "M528 recurrence reuse drift")

    ledger = synthetic_ledger_order(module)
    statistics_receipt = synthetic_same_ledger_statistics(module)
    publication = synthetic_publication_and_release(module)
    toctou = release_toctou(module)
    require(publication["same_release_accepted_twice"] and toctou,
            "release bypass proof drift")

    result = {
        "schema": "m1581_m1579_ep34_c1_same_ledger_cycle_model_independent_engineering_qa_r1_v1",
        "status": "NO_GO_M1581_M1579_PRODUCTION_RELEASE__CORE_SAME_LEDGER_MODEL_ENGINEERING_PASS__RELEASE_REUSABLE_AND_TOCTOU",
        "pinned_inputs": {"source_sha256": sha256(SOURCE),
                          "test_sha256": sha256(TEST),
                          "docs359_sha256": sha256(DOC359),
                          "m1524_sha256": module.M1524_SHA256,
                          "m528_sha256": module.M528_SHA256,
                          "m505_sha256": module.M505_SHA256,
                          "m504_sha256": module.M504_SHA256},
        "passed": {
            "final_ep34_binding": FINAL_EP34,
            "production_geometry_static_only": audit["geometry"],
            "support_to_ledger_order_synthetic": ledger,
            "same_ledger_ratio_and_distribution_synthetic": statistics_receipt,
            "m528_recurrence_exact_sha_and_api_reused": True,
            "zero_bit_product_share_one_ledger_derived_arrays": True,
            "atomic_staging_and_double_seal": publication,
            "claim_boundary_cycle_model_not_rtl_not_full_network": True,
            "no_network_gpu_eda_wallclock_dependency": True},
        "p0_findings": {
            "release_is_declarative_not_consumable": True,
            "same_release_accepted_twice_after_first_output_is_archived": True,
            "no_attempt_marker_or_lock": True,
            "release_bytes_can_change_after_verification": True,
            "result_hashes_post_verification_mutated_release": True,
            "verified_release_bytes_bound_to_execution": False,
            "actual_51m_production_authorized": False},
        "required_fix": [
            "Add a fresh canonical attempt namespace and atomic O_EXCL/flock consumption before any ledger materialization; a failed or completed first attempt must make the release permanently unusable.",
            "Read and hash one immutable regular non-symlink release byte snapshot, validate that snapshot, and bind the exact pre-execution digest into the result.",
            "Keep maximum_workers as exact int in [1,3], exact output/ledger binding, same-parent staging, atomic publish, and the current double seal.",
            "Run a fresh independent source hammer before the single CPU production release; independently hammer the result before any paper citation."],
        "authorization": {"release_successor_authoring": True,
                          "actual_51m_execution": False,
                          "production": False, "gpu": False,
                          "rtl": False, "eda": False},
        "claim_boundary": {"source_audit_and_synthetic_only": True,
                           "cycle_model": True, "rtl_cycle": False,
                           "wall_clock": False, "full_network": False,
                           "system_speedup": False, "energy": False,
                           "paper_citable": False}}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
