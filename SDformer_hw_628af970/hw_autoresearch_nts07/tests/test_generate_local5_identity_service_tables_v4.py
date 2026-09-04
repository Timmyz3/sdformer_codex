from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.generate_local5_identity_service_tables_v4 import (
    OUT_DIM,
    TOKENS,
    build_tables,
    canonical_sha,
    load_plan,
    relation_transaction,
    sample_identity,
    write_tables,
)
from scripts.verify_local5_identity_service_tables_v4 import (
    verify_core,
    verify_package,
    write_verification_receipt,
)


def fixture_plan() -> dict[str, object]:
    heads = 3
    tasks = [
        {"input_group_index": 276 + head, "output_tile": tile}
        for tile in range(heads)
        for head in range(heads)
    ]
    return {
        "schema": "local5_projection_task_plan_v1",
        "scope": "fixture",
        "sample": 2,
        "stage": 0,
        "block": 0,
        "window": 249,
        "heads": heads,
        "out_dim": OUT_DIM,
        "tasks": tasks,
        "task_sha256": canonical_sha(tasks),
        "source_manifest_sha256": "1" * 64,
        "source_payload_sha256": "2" * 64,
        "projection_contract_sha256": "3" * 64,
        "projection_payload_sha256": "4" * 64,
    }


class IdentityServiceTableGeneratorTest(unittest.TestCase):
    def test_h3_counts_delays_and_repeated_relation_identity(self) -> None:
        plan = fixture_plan()
        tables = build_tables(plan, seed=20260810)
        self.assertEqual(tables["relation_delay"].shape, (3 * TOKENS,))
        self.assertEqual(tables["weight_delay"].shape, (3 * 3 * 32 * 32,))
        self.assertEqual(tables["final_delay"].shape, (3 * TOKENS * 32,))
        for name in ("relation_delay", "weight_delay", "final_delay"):
            self.assertEqual(tables[name].dtype, np.uint8)
            self.assertTrue(np.all(tables[name] <= 3))
        first = relation_transaction(plan, 0, 0, 20260810)
        self.assertEqual(int(tables["relation_delay"][0]), first.delay)
        runtime = tables["ledger_summary"]["relation_runtime"]
        self.assertEqual(runtime["transaction_count"], 3 * 3 * TOKENS)
        self.assertEqual(runtime["identity_count"], 3 * TOKENS)
        self.assertEqual(runtime["multiplicity_histogram"], {"3": 3 * TOKENS})

    def test_sample_string_binds_manifest_and_numeric_sample(self) -> None:
        value = sample_identity(fixture_plan())
        self.assertEqual(
            value,
            "local5/profile100/" + "1" * 64 + "/sample/002",
        )

    def test_bad_task_order_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan = fixture_plan()
            plan["tasks"] = list(reversed(plan["tasks"]))
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tile-major"):
                load_plan(path)

    def test_foreign_input_groups_on_later_tile_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan = fixture_plan()
            plan["tasks"][3]["input_group_index"] = 999
            plan["task_sha256"] = canonical_sha(plan["tasks"])
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tile-major"):
                load_plan(path)

    def test_written_artifacts_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(fixture_plan()), encoding="utf-8")
            report = write_tables(plan_path, root / "out", 20260810)
            self.assertEqual(report["formal_g0"], "DENY")
            self.assertEqual(report["runtime_counts"]["relation_lookup"], 4050)
            with np.load(root / "out/identity_service_tables.npz", allow_pickle=False) as archive:
                self.assertEqual(int(archive["schema_version"][0]), 4)
                self.assertEqual(archive["relation_digest"].shape, (1350, 32))
            self.assertEqual(
                len((root / "out/relation_delay.memh").read_text().splitlines()),
                1350,
            )
            self.assertEqual(report["npz_members"]["final_digest"]["shape"], [43200, 32])
            self.assertEqual(len(report["source_bindings"]), 3)
            self.assertTrue((root / "out/producer_complete.json").is_file())
            self.assertFalse((root / "out/verification_receipt.json").exists())
            verify_core(root / "out")
            write_verification_receipt(root / "out")
            verified = verify_package(root / "out")
            self.assertEqual(
                verified["status"],
                "PASS_IDENTITY_SERVICE_TABLES_VERIFIED_NOT_G0",
            )

    def test_nondefault_seed_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen DEFAULT_SEED"):
            build_tables(fixture_plan(), seed=20260811)

    def test_existing_output_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(fixture_plan()), encoding="utf-8")
            output = root / "out"
            write_tables(plan_path, output, 20260810)
            with self.assertRaisesRegex(ValueError, "already exists"):
                write_tables(plan_path, output, 20260810)

    def test_memh_tamper_fails_independent_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(fixture_plan()), encoding="utf-8")
            output = root / "out"
            write_tables(plan_path, output, 20260810)
            write_verification_receipt(output)
            path = output / "relation_delay.memh"
            rows = path.read_text(encoding="ascii").splitlines()
            rows[0] = str((int(rows[0]) + 1) % 4)
            path.write_text("\n".join(rows) + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "artifact"):
                verify_package(output)

    def test_contract_metadata_tamper_matrix_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(fixture_plan()), encoding="utf-8")
            base = root / "base"
            write_tables(plan_path, base, 20260810)
            mutations = {
                "empty_artifacts": lambda value: value.update(artifacts={}),
                "runtime_order": lambda value: value["runtime_order"].update(
                    relation="forged"
                ),
                "runtime_counts": lambda value: value["runtime_counts"].update(
                    relation_lookup=1
                ),
                "npz_members": lambda value: value["npz_members"][
                    "schema_version"
                ].update(dtype="float64"),
                "source_role": lambda value: value["source_bindings"][0].update(
                    role="identity_oracle"
                ),
                "handshake": lambda value: value["handshake_contract"].update(
                    response_latency="forged"
                ),
            }
            for name, mutate in mutations.items():
                target = root / name
                shutil.copytree(base, target)
                manifest_path = target / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                producer_path = target / "producer_complete.json"
                producer = json.loads(producer_path.read_text(encoding="utf-8"))
                producer["manifest_sha256"] = hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest()
                producer_path.write_text(json.dumps(producer), encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(ValueError):
                    verify_core(target)

    def test_npz_dtype_tamper_fails_after_sha_rebind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(fixture_plan()), encoding="utf-8")
            output = root / "out"
            write_tables(plan_path, output, 20260810)
            npz_path = output / "identity_service_tables.npz"
            with np.load(npz_path, allow_pickle=False) as archive:
                values = {name: archive[name].copy() for name in archive.files}
            values["schema_version"] = values["schema_version"].astype(np.float64)
            np.savez(npz_path, **values)
            digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["identity_service_tables"]["sha256"] = digest
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            producer_path = output / "producer_complete.json"
            producer = json.loads(producer_path.read_text(encoding="utf-8"))
            producer["manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            producer["artifact_sha256"]["identity_service_tables"] = digest
            producer_path.write_text(json.dumps(producer), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                verify_core(output)

    def test_receipt_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(fixture_plan()), encoding="utf-8")
            output = root / "out"
            write_tables(plan_path, output, 20260810)
            write_verification_receipt(output)
            receipt_path = output / "verification_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["status"] = "FORGED"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "receipt"):
                verify_package(output)

    def test_unregistered_package_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(fixture_plan()), encoding="utf-8")
            output = root / "out"
            write_tables(plan_path, output, 20260810)
            (output / "extra.bin").write_bytes(b"unregistered")
            with self.assertRaisesRegex(ValueError, "file set"):
                verify_core(output)

    def test_verifier_has_no_project_oracle_import(self) -> None:
        source = Path(
            "scripts/verify_local5_identity_service_tables_v4.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("from scripts.local5_erep_identity_service_v4", source)
        self.assertNotIn("from local5_erep_identity_service_v4", source)

    def test_boolean_and_uppercase_sha_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "plan.json"
            plan = fixture_plan()
            plan["heads"] = True
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "heads"):
                load_plan(path)
            plan = fixture_plan()
            plan["source_manifest_sha256"] = "A" * 64
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_plan(path)


if __name__ == "__main__":
    unittest.main()
