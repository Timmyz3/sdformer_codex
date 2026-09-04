import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_m1340_ep34_table_a_common_charge_compiler.py"
SPEC = importlib.util.spec_from_file_location("m1340_common_charge", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)
REPO = Path(__file__).resolve().parents[3]
DOCS359 = REPO / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CommonChargeCompilerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1340_test_")
        self.root = Path(self.temp.name)
        (self.root / "inputs").mkdir()
        (self.root / "docs").mkdir()
        protected = self.root / "docs/359_DATE终局冻结_20260813.md"
        protected.write_bytes(DOCS359.read_bytes())
        protected.chmod(0o444)
        self.population = []
        sequence_rows = (
            ("interlaken_01_a", "low"),
            ("thun_01_b", "medium"),
            ("zurich_city_12_a", "high"),
        )
        for sequence, stratum in sequence_rows:
            for sample in range(10):
                self.population.append({"sequence_id": sequence, "sample_id": sample,
                                        "density_stratum": stratum, "weight": 1.0 / 30.0})
        self.identity = {}
        identity_media = {
            "checkpoint": "application/octet-stream",
            "config": "application/yaml",
            "profile": "application/json",
            "capture_result": "application/json",
            "capture_result_hammer": "application/json",
        }
        for name, media in identity_media.items():
            self.identity[name] = self.write_spec("identity_%s.dat" % name,
                                                  ("sealed-%s" % name).encode(), media)
        self.common = {}
        for category in M.COMMON_CATEGORIES:
            self.common[category] = self.charge_spec("common", category, 10, 1)
        self.branches = {}
        row_cycles = {"B0": 100, "B1": 92, "B2": 88, "B3": 84, "C2": 82, "Ours": 70}
        for branch_index, branch in enumerate(M.DIRECT_BRANCHES):
            self.branches[branch] = {}
            for row, cycles in row_cycles.items():
                self.branches[branch][row] = self.charge_spec(
                    "direct", "%s.%s" % (branch, row), cycles + branch_index, 1)
        energy = {
            "schema": M.ENERGY_SCHEMA,
            "identity": "Motion-C12-ep34-final",
            "native_mapped_activity_coverage": 0.96,
            "logic_pj_per_cycle": {row: 2.0 + i * 0.1 for i, row in enumerate(M.ROWS)},
            "dram_pj_per_byte": {"read": 3.7, "write": 3.9},
            "sram_pj_per_byte": {
                macro: {"read": 0.2 + i * 0.001, "write": 0.3 + i * 0.001}
                for i, macro in enumerate(M.SRAM_MACROS)
            },
        }
        self.energy = self.write_json_spec("energy.json", energy)
        self.config = {
            "schema": M.SCHEMA,
            "status": "SOURCE_FIXTURE",
            "identity": {"name": "Motion-C12-ep34-final", **self.identity},
            "resource": copy.deepcopy(M.RESOURCE),
            "population": self.population,
            "rows": list(M.ROWS),
            "common_operators": self.common,
            "direct_branches": self.branches,
            "energy_authority": self.energy,
            "claim_boundary": {
                "same_denominator": True,
                "common_charge_identical_all_rows": True,
                "component_speedups_not_multiplied": True,
                "external_prosperity_not_ours": True,
                "independent_hammer_required": True,
                "paper_headline_admitted": False,
            },
            "protected_file": {"path": "docs/359_DATE终局冻结_20260813.md",
                               "sha256": M.PROTECTED_SHA256,
                               "media_type": "text/markdown"},
        }

    def tearDown(self):
        self.temp.cleanup()

    def write_spec(self, name, content, media="application/json"):
        path = self.root / "inputs" / name
        path.write_bytes(content)
        path.chmod(0o444)
        return {"path": path.relative_to(self.root).as_posix(),
                "sha256": digest(path), "media_type": media}

    def write_json_spec(self, name, payload):
        return self.write_spec(name, (json.dumps(payload, sort_keys=True,
                                                 separators=(",", ":")) + "\n").encode())

    def charge(self, cycles, numerator):
        return {
            "cycles": cycles,
            "fixed_numerator": numerator,
            "dram_read_bytes": cycles * 2,
            "dram_write_bytes": cycles,
            "sram_bytes": {
                macro: {"read_bytes": cycles + i, "write_bytes": i}
                for i, macro in enumerate(M.SRAM_MACROS)
            },
        }

    def charge_spec(self, kind, name, cycles, numerator):
        payload = {"schema": M.CHARGE_SCHEMA, "kind": kind, "name": name,
                   "identity": "Motion-C12-ep34-final", "population": {}}
        for row in self.population:
            key = M.population_key(row["sequence_id"], row["sample_id"])
            payload["population"][key] = self.charge(cycles + row["sample_id"], numerator)
        return self.write_json_spec("charge_%s_%s.json" % (kind, name.replace(".", "_")),
                                    payload)

    def write_config(self, config=None, raw=None):
        path = self.root / "config.json"
        if path.exists():
            path.chmod(0o644)
            path.unlink()
        path.write_text(raw if raw is not None else json.dumps(config or self.config),
                        encoding="utf-8")
        path.chmod(0o444)
        return path

    def test_build_has_six_same_denominator_rows_and_common_charge(self):
        result = M.build(self.write_config(), self.root)
        self.assertEqual(result["status"], "PASS_SOURCE_FIXTURE_NOT_PRODUCTION")
        self.assertEqual(len(result["rows"]), 6)
        self.assertEqual(result["population"]["points"], 30)
        self.assertEqual(result["energy_authority"]["sram_macro_count"], 17)
        numerators = {row["weighted_fixed_numerator"] for row in result["rows"]}
        self.assertEqual(len(numerators), 1)
        self.assertGreater(result["rows"][-1]["speedup_vs_B0"], 1.0)
        # Nine common records contribute 9*10 cycles before direct branches.
        ours_first = result["rows"][-1]["per_population"][0]["charge"]["cycles"]
        self.assertEqual(ours_first, 90 + 70 + 71 + 72)
        self.assertFalse(result["claim_boundary"]["paper_headline_admitted"])

    def test_missing_common_operator_rejected(self):
        config = copy.deepcopy(self.config)
        del config["common_operators"]["fc1"]
        with self.assertRaises(M.CompileError):
            M.build(self.write_config(config), self.root)

    def test_missing_population_point_rejected(self):
        config = copy.deepcopy(self.config)
        config["population"] = config["population"][:-1]
        with self.assertRaisesRegex(M.CompileError, "30 points"):
            M.build(self.write_config(config), self.root)

    def test_weight_sum_rejected(self):
        config = copy.deepcopy(self.config)
        config["population"][0]["weight"] = 0.5
        with self.assertRaisesRegex(M.CompileError, "sum to one"):
            M.build(self.write_config(config), self.root)

    def test_missing_sram_macro_rejected(self):
        spec = self.common["fc1"]
        path = self.root / spec["path"]
        path.chmod(0o644)
        payload = json.loads(path.read_text())
        first = next(iter(payload["population"].values()))
        del first["sram_bytes"]["parent_00"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o444)
        spec["sha256"] = digest(path)
        with self.assertRaisesRegex(M.CompileError, "17 SRAM"):
            M.build(self.write_config(), self.root)

    def test_row_numerator_mismatch_rejected(self):
        spec = self.branches["decoder"]["Ours"]
        path = self.root / spec["path"]
        path.chmod(0o644)
        payload = json.loads(path.read_text())
        for charge in payload["population"].values():
            charge["fixed_numerator"] = 2
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o444)
        spec["sha256"] = digest(path)
        with self.assertRaisesRegex(M.CompileError, "fixed numerator differs"):
            M.build(self.write_config(), self.root)

    def test_energy_coverage_below_95_rejected(self):
        path = self.root / self.energy["path"]
        path.chmod(0o644)
        payload = json.loads(path.read_text())
        payload["native_mapped_activity_coverage"] = 0.949
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o444)
        self.energy["sha256"] = digest(path)
        with self.assertRaisesRegex(M.CompileError, "\[0.95,1\]"):
            M.build(self.write_config(), self.root)

    def test_duplicate_json_key_rejected(self):
        with self.assertRaisesRegex(M.CompileError, "duplicate JSON key"):
            M.load_json(self.write_config(raw='{"schema":1,"schema":2}'))

    def test_writable_or_symlink_input_rejected(self):
        path = self.root / self.identity["profile"]["path"]
        path.chmod(0o644)
        with self.assertRaisesRegex(M.CompileError, "read-only"):
            M.build(self.write_config(), self.root)
        path.chmod(0o444)
        target = self.root / "inputs/profile_target.dat"
        shutil.copyfile(path, target)
        target.chmod(0o444)
        path.unlink()
        path.symlink_to(target.name)
        self.identity["profile"]["sha256"] = digest(target)
        with self.assertRaisesRegex(M.CompileError, "single-link regular"):
            M.build(self.write_config(), self.root)

    def test_cli_no_replace(self):
        config = self.write_config()
        output = self.root / "out.json"
        command = [sys.executable, str(SCRIPT), "--config", str(config),
                   "--workspace-root", str(self.root), "--output", str(output)]
        first = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(output.stat().st_mode & 0o777, 0o444)
        second = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(second.returncode, 2)
        self.assertIn("File exists", second.stdout)


if __name__ == "__main__":
    unittest.main()
