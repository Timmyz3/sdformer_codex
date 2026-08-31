#!/usr/bin/env python3
"""Synthetic and current-evidence tests for the M520 fail-closed registry."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/build_m520_h67_paper_metric_registry.py"
)
SPEC = importlib.util.spec_from_file_location("m520_registry", str(SCRIPT))
M520 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M520)
ROOT = Path(__file__).resolve().parents[3]
CURRENT_CONFIG = (
    ROOT / "hw_autoresearch_nts07/system_simulator/config/"
    "m520_h67_paper_metric_registry_v1_20260827.json"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_config(directory):
    source = directory / "source.json"
    source.write_text(json.dumps({"value": 40}), encoding="utf-8")
    blockers = [
        "decoder exact cycles missing",
        "Phi adapter missing",
        "system schedule missing",
        "macro area missing",
        "multi-sequence coverage missing",
    ]
    rows = []
    for row_id in M520.ROW_IDS:
        rows.append({
            "row_id": row_id,
            "display_name": row_id,
            "workload": "synthetic workload",
            "checkpoint": "synthetic checkpoint",
            "sequence": "synthetic sequence",
            "operator_scope": "synthetic operator scope",
            "resource_scope": "synthetic resource scope",
            "default_blocking_reason": "synthetic evidence is absent",
        })
    return {
        "schema": M520.CONFIG_SCHEMA,
        "date": "2026-08-27",
        "status": M520.STATUS,
        "system_speedup_generated": False,
        "system_table_blockers": blockers,
        "sources": {
            "synthetic": {
                "path": str(source.relative_to(ROOT)),
                "sha256": digest(source),
                "format": "json",
                "role": "synthetic unit-test source",
            }
        },
        "rows": rows,
        "populated_metrics": [{
            "row_id": "fixed_dense",
            "metric": "compute_cycles",
            "source_id": "synthetic",
            "json_pointers": ["/value"],
            "aggregation": "single",
            "divisor": 2,
            "numerator_unit": "cycle/two-window population",
            "denominator_unit": "selected_window/population",
            "denominator_definition": "two synthetic selected windows",
            "evidence_class": "SYNTHETIC_TEST",
            "claim_boundary": "synthetic test only",
        }],
        "blocked_metric_evidence": [],
    }


class M520RegistryTests(unittest.TestCase):
    def test_synthetic_registry_is_complete_and_blocked(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            config = synthetic_config(Path(tmp))
            registry = M520.build_registry(config)
        self.assertEqual(registry["status"], M520.STATUS)
        self.assertFalse(registry["system_speedup_generated"])
        self.assertEqual(len(registry["rows"]), 8)
        cells = [cell for row in registry["rows"] for cell in row["metrics"]]
        self.assertEqual(len(cells), 120)
        self.assertEqual(len({cell["metric_id"] for cell in cells}), 120)
        value = registry["rows"][0]["metrics"][0]
        self.assertEqual(value["value"], 20)
        self.assertFalse(value["admission"]["system_table_eligible"])
        nulls = [cell for cell in cells if cell["value"] is None]
        self.assertEqual(len(nulls), 119)
        self.assertTrue(all(cell["blocking_reason"] for cell in nulls))

    def test_source_sha_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            config = synthetic_config(Path(tmp))
            config["sources"]["synthetic"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(M520.RegistryError, "SHA mismatch"):
                M520.build_registry(config)

    def test_row_population_and_order_are_fixed(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            config = synthetic_config(Path(tmp))
            config["rows"] = config["rows"][:-1]
            with self.assertRaisesRegex(M520.RegistryError, "row order/population"):
                M520.build_registry(config)

    def test_null_without_reason_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            config = synthetic_config(Path(tmp))
            config["rows"][1]["default_blocking_reason"] = ""
            with self.assertRaisesRegex(M520.RegistryError, "default_blocking_reason"):
                M520.build_registry(config)

    def test_duplicate_metric_spec_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            config = synthetic_config(Path(tmp))
            config["populated_metrics"].append(
                copy.deepcopy(config["populated_metrics"][0])
            )
            with self.assertRaisesRegex(M520.RegistryError, "duplicate populated"):
                M520.build_registry(config)

    def test_missing_pointer_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            config = synthetic_config(Path(tmp))
            config["populated_metrics"][0]["json_pointers"] = ["/missing"]
            with self.assertRaisesRegex(M520.RegistryError, "missing JSON pointer"):
                M520.build_registry(config)

    def test_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            directory = Path(tmp)
            config = synthetic_config(directory)
            source = directory / "source.json"
            source.write_text('{"value": NaN}', encoding="utf-8")
            config["sources"]["synthetic"]["sha256"] = digest(source)
            with self.assertRaisesRegex(M520.RegistryError, "non-standard JSON token"):
                M520.build_registry(config)

    def test_current_inventory_preserves_claim_boundaries(self):
        registry = M520.build_registry(M520.strict_json(CURRENT_CONFIG))
        by_row = {row["row_id"]: row for row in registry["rows"]}
        prosperity = {
            cell["metric_id"].split(".", 1)[1]: cell
            for cell in by_row["prosperity_official_external_iso_workload"]["metrics"]
        }
        self.assertEqual(prosperity["total_cycles"]["value"], 22614000.6)
        self.assertEqual(prosperity["dram_write_bytes"]["value"], 0)
        self.assertFalse(prosperity["total_cycles"]["admission"]["system_table_eligible"])
        self.assertIn("support-tile", prosperity["total_cycles"]["admission"]["claim_boundary"])
        self.assertTrue(all(cell["value"] is None for cell in by_row["phi_like"]["metrics"]))
        serialized = json.dumps(registry).lower()
        self.assertNotIn("2.459487", serialized)
        self.assertNotIn('"speedup"', serialized)

    def test_validator_rejects_numeric_without_provenance(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            registry = M520.build_registry(synthetic_config(Path(tmp)))
        registry["rows"][0]["metrics"][0]["source"] = None
        with self.assertRaisesRegex(M520.RegistryError, "lacks derivation/provenance"):
            M520.validate_registry(registry)


if __name__ == "__main__":
    unittest.main()
