from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1233_motion_final_checkpoint_unified_hardware_selection_interface_r2.py"
)
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1233_motion_final_checkpoint_unified_capture_selection_interface_"
    "successor_source_contract_r1_20260830.json"
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


M = load("m1233_selection_interface_under_test", SOURCE)


class M1233SelectionInterfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        results = M.HW / "results"
        reviews = M.HW / "reviews"
        results.mkdir(parents=True, exist_ok=True)
        reviews.mkdir(parents=True, exist_ok=True)
        self.asset_tmp = tempfile.TemporaryDirectory(prefix=".m1233_assets_", dir=results)
        self.selection_tmp = tempfile.TemporaryDirectory(
            prefix=".m1233_selection_", dir=results
        )
        self.hammer_tmp = tempfile.TemporaryDirectory(prefix=".m1233_hammer_", dir=reviews)
        self.assets = Path(self.asset_tmp.name)
        self.selection_root = Path(self.selection_tmp.name)
        self.hammer_root = Path(self.hammer_tmp.name)
        self.checkpoint = self.assets / "checkpoint_epoch32.pth"
        self.configuration = self.assets / "motion_resume.yml"
        self.profile = self.assets / "spike_profile.json"
        self.checkpoint.write_bytes(b"m1233-checkpoint-epoch32\n" * 16)
        self.configuration.write_text("experiment: motion_resume\n", encoding="utf-8")
        self.profile.write_text('{"samples":825,"AEE":0.99}\n', encoding="utf-8")
        self.selection = self._selection()
        self.selection_entry = self._write_selection(self.selection)
        self.hammer_entry = self._write_hammer(
            self._hammer(self.selection_entry, self.selection["selected"])
        )

    def tearDown(self) -> None:
        self.hammer_tmp.cleanup()
        self.selection_tmp.cleanup()
        self.asset_tmp.cleanup()

    @staticmethod
    def _identity(path: Path) -> dict[str, object]:
        record = path.stat()
        return {
            "absolute_path": str(path.resolve()),
            "size_bytes": record.st_size,
            "mtime_ns": record.st_mtime_ns,
            "sha256": sha(path),
        }

    def _selection(self) -> dict[str, object]:
        profile = {
            **self._identity(self.profile),
            "samples": 825,
            "artifact_identity_exact": True,
            "load_audit_exact_zero": True,
            "module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "immutable_single_read": True,
            "hash_and_parse_same_bytes": True,
        }
        selected = {
            "candidate_id": "resume_ep32",
            "epoch": 32,
            "run_directory": str(self.assets.resolve()),
            "checkpoint": self._identity(self.checkpoint),
            "configuration": self._identity(self.configuration),
            "profile": profile,
            "accuracy_metrics": {"AEE": "0.99"},
            "activity": {"total_spikes": 1234},
        }
        return {
            "schema": M.ALLOWED_SELECTION_SCHEMA,
            "status": M.ALLOWED_SELECTION_STATUS,
            "new_run_manifest": {"evaluation_epochs": [30, 32, 34]},
            "candidate_population": [],
            "selection_rule": {"primary": "minimum finite nonnegative standard-valid825 AEE"},
            "selected": selected,
            "e0_e8_activation_dependent_invalidation_and_rebind_targets": [],
            "claim_boundary": {
                "fresh_result_hammer_required": True,
                "hardware_rebind_authorized": False,
            },
        }

    @staticmethod
    def _clear_seal(root: Path) -> None:
        for name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            path = root / name
            if path.exists():
                path.unlink()

    def _write_selection(self, value: dict[str, object]) -> dict[str, str]:
        self._clear_seal(self.selection_root)
        member = self.selection_root / "final_checkpoint_selection.json"
        member.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        M.R1.write_double_seal(self.selection_root)
        return {
            "result_path": str(self.selection_root.relative_to(M.ROOT)),
            "manifest_sha256": sha(self.selection_root / "SHA256SUMS"),
            "outer_file_sha256": sha(self.selection_root / "SHA256SUMS.seal.sha256"),
            "selection_member": member.name,
            "selection_sha256": sha(member),
        }

    def _hammer(
        self, selection_entry: dict[str, str], selected: dict[str, object]
    ) -> dict[str, object]:
        return {
            "schema": M.SELECTION_RESULT_HAMMER_SCHEMA,
            "status": M.SELECTION_RESULT_HAMMER_STATUS,
            "selection_authority": {
                "result_path": selection_entry["result_path"],
                "selection_member": selection_entry["selection_member"],
                "selection_sha256": selection_entry["selection_sha256"],
                "selection_manifest_sha256": selection_entry["manifest_sha256"],
                "selection_outer_file_sha256": selection_entry["outer_file_sha256"],
                "selection_schema": M.ALLOWED_SELECTION_SCHEMA,
                "selection_status": M.ALLOWED_SELECTION_STATUS,
                "selected_candidate_id": selected["candidate_id"],
                "selected_epoch": selected["epoch"],
                "selected_profile_sha256": selected["profile"]["sha256"],
                "selected_checkpoint_sha256": selected["checkpoint"]["sha256"],
                "selected_config_sha256": selected["configuration"]["sha256"],
            },
            "independence": {"different_author": True},
            "authorization": {
                "hardware_rebind_release_authoring": True,
                "production_capture": False,
            },
        }

    def _write_hammer(self, value: dict[str, object]) -> dict[str, str]:
        self._clear_seal(self.hammer_root)
        review = self.hammer_root / "review.json"
        review.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        M.R1.write_double_seal(self.hammer_root)
        return {
            "path": str(self.hammer_root.relative_to(M.ROOT)),
            "manifest_sha256": sha(self.hammer_root / "SHA256SUMS"),
            "outer_file_sha256": sha(self.hammer_root / "SHA256SUMS.seal.sha256"),
            "review_sha256": sha(review),
        }

    def assert_rejected(self, selection_entry=None, hammer_entry=None) -> None:
        with self.assertRaises(M.M1233Error):
            M.validate_final_selection(
                self.selection_entry if selection_entry is None else selection_entry,
                self.hammer_entry if hammer_entry is None else hammer_entry,
            )

    def test_01_predecessor_and_capture_logic_are_exact_aliases(self) -> None:
        self.assertEqual(sha(M.PREDECESSOR), M.PREDECESSOR_SHA256)
        for name in (
            "EXPECTED_STATIC_COUNTS", "EXPECTED_LIVE_COUNTS", "DEAD_SN_V",
            "audit_call_matrix", "audit_attention_population",
            "validate_payload_population", "atomic_sample_snapshot",
            "final_validate_and_seal",
        ):
            self.assertIs(getattr(M, name), getattr(M.R1, name))
        self.assertEqual(M.EXPECTED_STATIC_COUNTS["atlif"], 105)
        self.assertEqual(M.EXPECTED_LIVE_COUNTS["atlif"], 93)
        self.assertEqual(len(M.DEAD_SN_V), 12)

    def test_02_import_is_lazy_for_gpu_and_model_stack(self) -> None:
        code = (
            "import importlib.util,sys;"
            "s=importlib.util.spec_from_file_location('isolated_m1233',{!r});"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "print(int('torch' in sys.modules),int('numpy' in sys.modules),"
            "int('m1227_sealed_m1174' in sys.modules))"
        ).format(str(SOURCE))
        self.assertEqual(subprocess.check_output(
            [sys.executable, "-c", code]).decode().strip(), "0 0 0")

    def test_03_exact_m1234_shape_passes_and_keyerror_is_regressed(self) -> None:
        binding = M.validate_final_selection(self.selection_entry, self.hammer_entry)
        self.assertEqual(binding["checkpoint_path"], self.checkpoint)
        self.assertEqual(binding["config_path"], self.configuration)
        self.assertEqual(binding["profile_path"], self.profile)
        self.assertEqual(binding["identity"]["candidate_id"], "resume_ep32")
        self.assertEqual(binding["identity"]["epoch"], 32)
        self.assertEqual(binding["identity"]["config_sha256"], sha(self.configuration))
        self.assertNotIn("configuration", self.selection)

    def test_04_top_level_configuration_is_always_rejected(self) -> None:
        for value in (
            copy.deepcopy(self.selection["selected"]["configuration"]),
            {**self._identity(self.configuration), "sha256": "0" * 64},
        ):
            attack = copy.deepcopy(self.selection)
            attack["configuration"] = value
            entry = self._write_selection(attack)
            with self.subTest(value=value["sha256"]):
                self.assert_rejected(selection_entry=entry)

    def test_05_selection_schema_and_status_are_fixed(self) -> None:
        for key, value in (
            ("schema", "m1228_motion_cross_run_final_checkpoint_rebind_binder_source_r1_v1"),
            ("status", "READY_CROSS_RUN_SELECTION"),
        ):
            attack = copy.deepcopy(self.selection)
            attack[key] = value
            entry = self._write_selection(attack)
            with self.subTest(key=key):
                self.assert_rejected(selection_entry=entry)

    def test_06_selected_shape_candidate_and_epoch_are_exact(self) -> None:
        attacks = []
        for key in M.SELECTED_KEYS:
            row = copy.deepcopy(self.selection)
            del row["selected"][key]
            attacks.append(("missing_" + key, row))
        extra = copy.deepcopy(self.selection)
        extra["selected"]["top_level_config_fallback"] = True
        attacks.append(("extra", extra))
        for value in (True, 32.0, 34):
            row = copy.deepcopy(self.selection)
            row["selected"]["epoch"] = value
            attacks.append(("epoch_" + repr(value), row))
        row = copy.deepcopy(self.selection)
        row["selected"]["candidate_id"] = "resume_ep34"
        attacks.append(("candidate_pair", row))
        for label, attack in attacks:
            entry = self._write_selection(attack)
            with self.subTest(label=label):
                self.assert_rejected(selection_entry=entry)

    def test_07_checkpoint_config_and_profile_identity_types_are_exact(self) -> None:
        attacks = []
        for member in ("checkpoint", "configuration", "profile"):
            for field, value in (("size_bytes", True), ("mtime_ns", 1.0),
                                 ("sha256", "A" * 64)):
                row = copy.deepcopy(self.selection)
                row["selected"][member][field] = value
                attacks.append((member + "_" + field, row))
        for label, attack in attacks:
            entry = self._write_selection(attack)
            with self.subTest(label=label):
                self.assert_rejected(selection_entry=entry)

    def test_08_checkpoint_config_and_profile_content_drift_is_rejected(self) -> None:
        for path in (self.checkpoint, self.configuration, self.profile):
            original = path.read_bytes()
            path.write_bytes(original + b"drift")
            with self.subTest(path=path.name):
                self.assert_rejected()
            path.write_bytes(original)

    def test_09_profile_semantics_are_fixed(self) -> None:
        attacks = []
        for field, value in (
            ("samples", True), ("samples", 825.0), ("samples", 824),
            ("artifact_identity_exact", False), ("load_audit_exact_zero", False),
        ):
            row = copy.deepcopy(self.selection)
            row["selected"]["profile"][field] = value
            attacks.append((field + repr(value), row))
        row = copy.deepcopy(self.selection)
        row["selected"]["profile"]["module_counts"]["ATLIFTernaryPSN"] = 104
        attacks.append(("module_count", row))
        for label, attack in attacks:
            entry = self._write_selection(attack)
            with self.subTest(label=label):
                self.assert_rejected(selection_entry=entry)

    def test_10_selection_and_hammer_must_each_be_double_sealed(self) -> None:
        attacks = (
            {**self.selection_entry, "manifest_sha256": "0" * 64},
            {**self.selection_entry, "outer_file_sha256": "0" * 64},
            {**self.selection_entry, "selection_sha256": "0" * 64},
        )
        for entry in attacks:
            self.assert_rejected(selection_entry=entry)
        for key in ("manifest_sha256", "outer_file_sha256", "review_sha256"):
            entry = {**self.hammer_entry, key: "0" * 64}
            self.assert_rejected(hammer_entry=entry)

    def test_11_hammer_schema_status_independence_and_authorization_are_fixed(self) -> None:
        canonical = self._hammer(self.selection_entry, self.selection["selected"])
        mutations = (
            ("schema", lambda row: row.__setitem__("schema", "wrong")),
            ("status", lambda row: row.__setitem__("status", "PASS")),
            ("independence", lambda row: row.__setitem__(
                "independence", {"different_author": False})),
            ("authorization", lambda row: row["authorization"].__setitem__(
                "production_capture", True)),
        )
        for label, mutate in mutations:
            attack = copy.deepcopy(canonical)
            mutate(attack)
            entry = self._write_hammer(attack)
            with self.subTest(label=label):
                self.assert_rejected(hammer_entry=entry)

    def test_12_every_hammer_cross_sha_and_selected_pair_field_is_bound(self) -> None:
        canonical = self._hammer(self.selection_entry, self.selection["selected"])
        authority = canonical["selection_authority"]
        for key in sorted(M.HAMMER_AUTHORITY_KEYS):
            attack = copy.deepcopy(canonical)
            value = authority[key]
            attack["selection_authority"][key] = (
                value + 1 if type(value) is int else str(value) + ".drift"
            )
            entry = self._write_hammer(attack)
            with self.subTest(key=key):
                self.assert_rejected(hammer_entry=entry)

    def test_13_result_hammer_entry_shape_and_review_population_are_exact(self) -> None:
        self.assert_rejected(hammer_entry={**self.hammer_entry, "extra": True})
        attack = self._hammer(self.selection_entry, self.selection["selected"])
        attack["selection_authority"]["extra"] = True
        self.assert_rejected(hammer_entry=self._write_hammer(attack))

    def test_14_delegate_changes_only_result_namespace_and_passes_substrate(self) -> None:
        seen = {}

        def run_capture(contract, binding, r1=None):
            seen.update(contract=contract, binding=binding, substrate=r1,
                        result=predecessor.CANONICAL_RESULT)
            return predecessor.CANONICAL_RESULT

        old_result = Path("old-result")
        predecessor = types.SimpleNamespace(
            CANONICAL_RESULT=old_result, run_capture=run_capture
        )
        substrate = object()
        output = M.run_capture(
            {"contract": 1}, {"binding": 2}, predecessor=predecessor,
            substrate=substrate,
        )
        self.assertEqual(output, M.CANONICAL_RESULT)
        self.assertEqual(predecessor.CANONICAL_RESULT, old_result)
        self.assertIs(seen["substrate"], substrate)
        self.assertEqual(seen["contract"], {"contract": 1})
        self.assertEqual(seen["binding"], {"binding": 2})

    def test_15_source_only_cannot_launch_and_namespaces_are_fresh(self) -> None:
        with self.assertRaises(M.M1233Error):
            M.validate_launch_contract({}, CONTRACT)
        self.assertNotEqual(M.CANONICAL_RESULT, M.R1.CANONICAL_RESULT)
        self.assertFalse(M.CANONICAL_RESULT.exists())
        self.assertFalse(M.CANONICAL_ATTEMPT.exists())
        self.assertFalse(M.CANONICAL_LOG.exists())

    def test_16_source_contract_hashes_and_claim_boundary(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["source"]["sha256"], sha(SOURCE))
        self.assertEqual(contract["test"]["sha256"], sha(Path(__file__).resolve()))
        self.assertEqual(contract["status"], M.SOURCE_STATUS)
        self.assertFalse(contract["claim_boundary"]["production_capture"])
        self.assertFalse(contract["claim_boundary"]["hardware_rebind_authorized"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
