from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from hw_autoresearch_nts07.tests.test_build_m1166_motion_final_checkpoint_selection_rebind_binder_r2 import (
    M1166TypedZeroTest,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_m1167_motion_final_checkpoint_selection_rebind_binder_r3.py"
SPEC = importlib.util.spec_from_file_location("m1167_binder_r3", SCRIPT)
assert SPEC and SPEC.loader
M3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M3
SPEC.loader.exec_module(M3)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M1167CanonicalEpochNameTest(M1166TypedZeroTest):
    def test_r3_canonical_build_and_namespace(self) -> None:
        result = M3.build(self.policy)
        self.assertEqual(result["selected"]["epoch"], 29)
        self.assertEqual(result["source_hardening"]["revision"], "r3")
        self.assertEqual(
            set(result["source_hardening"]["canonical_epoch_entry_names"]),
            {"epoch9", "epoch14", "epoch19", "epoch24", "epoch29"},
        )
        output = self.root / "r3_receipt"
        M3.write_receipt(output, result)
        self.assertEqual(
            (output / "RUN_COMPLETE.txt").read_text(),
            "PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
            "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n",
        )

    def test_alias_case_and_extra_file_attacks(self) -> None:
        standard = self.run / "standard_valid825"
        attacks = (
            ("epoch09", "directory"),
            ("epoch009", "directory"),
            ("epoch+9", "directory"),
            ("Epoch9", "directory"),
            ("EXTRA.txt", "file"),
        )
        for name, kind in attacks:
            with self.subTest(name=name):
                path = standard / name
                if kind == "directory":
                    path.mkdir()
                else:
                    path.write_text("extra\n", encoding="utf-8")
                with self.assertRaises(M3.R1.BinderError):
                    M3.build(self.policy)
                if kind == "directory":
                    path.rmdir()
                else:
                    path.unlink()

    def test_r2_dependency_identity_is_pinned_by_r3(self) -> None:
        self.assertEqual(sha(M3.R2_SOURCE), M3.R2_SOURCE_SHA256)

    def test_integral_float_bool_and_identity_schema_attacks(self) -> None:
        path = self._profile(24)
        canonical = path.read_text(encoding="utf-8")

        def samples_float(value):
            value["samples"] = 825.0

        def checkpoint_overlay_float(value):
            value["checkpoint_load_audit"]["checkpoint_overlay_keys"] = 210.0

        def model_overlay_float(value):
            value["checkpoint_load_audit"]["model_overlay_keys"] = 210.0

        def atlif_float(value):
            value["module_counts"]["ATLIFTernaryPSN"] = 105.0

        def attention_float(value):
            value["module_counts"]["ShiftmaxAttention"] = 12.0

        def checkpoint_size_float(value):
            original = value["artifact_identity"]["checkpoint_size"]
            value["artifact_identity"]["checkpoint_size"] = float(original)

        def checkpoint_mtime_bool(value):
            value["artifact_identity"]["checkpoint_mtime_ns"] = True

        for name, mutation in (
            ("samples_825_float", samples_float),
            ("checkpoint_overlay_210_float", checkpoint_overlay_float),
            ("model_overlay_210_float", model_overlay_float),
            ("atlif_105_float", atlif_float),
            ("attention_12_float", attention_float),
            ("checkpoint_size_float", checkpoint_size_float),
            ("checkpoint_mtime_bool", checkpoint_mtime_bool),
        ):
            with self.subTest(name=name):
                value = json.loads(canonical)
                mutation(value)
                path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
                with self.assertRaises(M3.R1.BinderError):
                    M3.build(self.policy)
        path.write_text(canonical, encoding="utf-8")


if __name__ == "__main__":
    import unittest
    unittest.main()
