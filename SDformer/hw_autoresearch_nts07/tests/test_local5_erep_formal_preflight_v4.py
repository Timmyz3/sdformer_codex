from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import local5_erep_formal_preflight_v4 as preflight


class Local5ErepFormalPreflightV4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(preflight.SELECTION_PLAN.read_text(encoding="utf-8"))
        cls.contract = json.loads(preflight.PROJECTION_CONTRACT.read_text(encoding="utf-8"))
        cls.old_manifest = json.loads(
            (
                preflight.ROOT
                / "results/local5_fullres_bb1e4_postg0_profile100_20260805/ordered_term_manifest.json"
            ).read_text(encoding="utf-8")
        )
        payload_path = preflight.PROFILE_DIR / cls.contract["payload_file"]
        with np.load(payload_path, allow_pickle=False) as payload_file:
            cls.payload = {name: payload_file[name] for name in payload_file.files}

    def test_fixed_inputs_expand_to_exact_hxh_counts(self) -> None:
        self.assertEqual(
            preflight.sha256_file(preflight.PROJECTION_CONTRACT),
            preflight.PROJECTION_CONTRACT_SHA256,
        )
        self.assertEqual(
            self.contract["payload_sha256"], preflight.PROJECTION_PAYLOAD_SHA256
        )
        windows = preflight.validate_selection_plan(self.plan)
        blocks = preflight.validate_projection_contract(self.contract, self.payload)
        tasks = preflight.enumerate_hxh_tasks(windows, blocks)
        self.assertEqual(len(windows), 1200)
        self.assertEqual(len(preflight.expected_head_keys(windows)), 13_800)
        self.assertEqual(len(tasks), 210_600)
        by_stage = {stage: 0 for stage in range(4)}
        for _, stage, _, _, _, _ in tasks:
            by_stage[stage] += 1
        self.assertEqual(by_stage, {0: 1800, 1: 7200, 2: 86_400, 3: 115_200})

    def test_selection_rejects_missing_duplicate_and_wrong_head_count(self) -> None:
        variants = []
        missing = copy.deepcopy(self.plan)
        missing["records"].pop()
        variants.append(missing)
        duplicate = copy.deepcopy(self.plan)
        duplicate["records"][1] = copy.deepcopy(duplicate["records"][0])
        variants.append(duplicate)
        wrong_heads = copy.deepcopy(self.plan)
        wrong_heads["records"][0]["heads"] = 2
        variants.append(wrong_heads)
        for index, value in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                preflight.validate_selection_plan(value)

    def test_projection_rejects_non_square_hxh_shape_and_dtype(self) -> None:
        wrong_contract = copy.deepcopy(self.contract)
        wrong_contract["blocks"][0]["weight_shape"] = [96, 64]
        with self.assertRaisesRegex(ValueError, r"H\*32 square"):
            preflight.validate_projection_contract(wrong_contract, self.payload)

        wrong_payload = dict(self.payload)
        wrong_payload["s0_b0_weight_int8"] = wrong_payload["s0_b0_weight_int8"].astype(
            np.int16
        )
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            preflight.validate_projection_contract(self.contract, wrong_payload)

        float_shape = copy.deepcopy(self.contract)
        float_shape["blocks"][0]["weight_shape"] = [96.0, 96.0]
        with self.assertRaisesRegex(ValueError, "shape type"):
            preflight.validate_projection_contract(float_shape, self.payload)

    def test_manifest_group_coverage_is_order_independent_but_exact(self) -> None:
        windows = preflight.validate_selection_plan(self.plan)
        groups = [
            {
                "tag": tag,
                "empty": False,
                "sample": sample,
                "stage": stage,
                "block": block,
                "window": window,
                "head": head,
                "heads": preflight.STAGE_HEADS[stage],
                "lanes": 32,
                "tokens": 450,
                "time_planes": 2,
                "plane_tokens": 225,
                "spatial_side": 15,
                "flat_group": window * preflight.STAGE_HEADS[stage] + head,
                "batch_windows": preflight.STAGE_WINDOWS[stage],
                "plane_execution": "plane_serial_drain",
                "module": (
                    "sttmultires_unet.encoders.swin3d.layers."
                    f"{stage}.swin_blocks.{block}.attn"
                ),
                "selection": "uniform_plan_window_all_heads_v1",
                "ordered_item_sha256": f"{tag:064x}",
            }
            for tag, (sample, stage, block, window, head) in enumerate(
                reversed(preflight.expected_head_keys(windows))
            )
        ]
        manifest = copy.deepcopy(self.old_manifest)
        manifest["groups"] = groups
        manifest["qualification"]["qualified"] = True
        result = preflight.validate_manifest_groups(manifest, windows)
        self.assertEqual(result["head_group_count"], 13_800)
        self.assertEqual(len(result["group_order_key_sha256"]), 64)
        self.assertEqual(len(result["group_order_identity_sha256"]), 64)
        self.assertEqual(len(result["canonical_sorted_key_sha256"]), 64)
        for mutation in ("missing", "duplicate", "wrong_head"):
            modified = copy.deepcopy(manifest)
            if mutation == "missing":
                modified["groups"].pop()
            elif mutation == "duplicate":
                modified["groups"][-1] = copy.deepcopy(modified["groups"][0])
            else:
                modified["groups"][0]["head"] = preflight.STAGE_HEADS[
                    modified["groups"][0]["stage"]
                ]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                preflight.validate_manifest_groups(modified, windows)

        shadow = copy.deepcopy(manifest)
        shadow["groups"][0]["input_head"] = 99
        with self.assertRaisesRegex(ValueError, "field set"):
            preflight.validate_manifest_groups(shadow, windows)
        float_lanes = copy.deepcopy(manifest)
        float_lanes["groups"][0]["lanes"] = 32.0
        with self.assertRaises(ValueError):
            preflight.validate_manifest_groups(float_lanes, windows)

    def test_task_enumeration_rejects_noncanonical_order(self) -> None:
        windows = preflight.validate_selection_plan(self.plan)
        blocks = preflight.validate_projection_contract(self.contract, self.payload)
        with self.assertRaisesRegex(ValueError, "canonical window order"):
            preflight.enumerate_hxh_tasks(tuple(reversed(windows)), blocks)

    def test_float_topology_fields_are_rejected(self) -> None:
        modified = copy.deepcopy(self.plan)
        modified["records"][0]["heads"] = 3.0
        with self.assertRaises(ValueError):
            preflight.validate_selection_plan(modified)

        float_seed = copy.deepcopy(self.plan)
        float_seed["seed"] = 20260809.0
        with self.assertRaises(ValueError):
            preflight.validate_selection_plan(float_seed)

    def test_top_level_field_sets_are_frozen(self) -> None:
        selection = copy.deepcopy(self.plan)
        selection["shadow"] = 1
        with self.assertRaisesRegex(ValueError, "header"):
            preflight.validate_selection_plan(selection)

        projection = copy.deepcopy(self.contract)
        projection["shadow"] = 1
        with self.assertRaisesRegex(ValueError, "header"):
            preflight.validate_projection_contract(projection, self.payload)

        windows = preflight.validate_selection_plan(self.plan)
        manifest = copy.deepcopy(self.old_manifest)
        manifest["shadow"] = 1
        with self.assertRaisesRegex(ValueError, "header"):
            preflight.validate_manifest_groups(manifest, windows)

    def test_task_helper_rejects_direct_float_topology(self) -> None:
        windows = list(preflight.validate_selection_plan(self.plan))
        blocks = preflight.validate_projection_contract(self.contract, self.payload)
        windows[0] = dict(windows[0])
        windows[0]["sample"] = 0.0
        with self.assertRaises(ValueError):
            preflight.enumerate_hxh_tasks(tuple(windows), blocks)

        windows = preflight.validate_selection_plan(self.plan)
        float_blocks = copy.deepcopy(blocks)
        float_blocks[(0, 0)]["input_head_count"] = 3.0
        with self.assertRaises(ValueError):
            preflight.enumerate_hxh_tasks(windows, float_blocks)

    def test_formal_artifact_bindings_rehash_every_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=preflight.ROOT) as temporary:
            root = Path(temporary)
            profile = root / "profile"
            profile.mkdir()
            files = {
                "formal_manifest": profile / "ordered_term_manifest.json",
                "ordered_payload": profile / "ordered_term_items.npz",
                "cohort": profile / "ordered_cohort.json",
                "projection_contract": profile / "checkpoint_projection_contract.json",
                "projection_payload": profile / "checkpoint_projection_contract.npz",
            }
            for index, path in enumerate(files.values()):
                path.write_bytes(f"fixture-{index}".encode("ascii"))
            contract = {
                "payload_file": files["projection_payload"].name,
                "payload_sha256": preflight.sha256_file(files["projection_payload"]),
            }
            manifest = {
                "payload_file": files["ordered_payload"].name,
                "payload_sha256": preflight.sha256_file(files["ordered_payload"]),
                "cohort_file": files["cohort"].name,
                "cohort_file_sha256": preflight.sha256_file(files["cohort"]),
                "projection_contract_file": files["projection_contract"].name,
                "projection_contract_file_sha256": preflight.sha256_file(
                    files["projection_contract"]
                ),
                "projection_contract_payload": files["projection_payload"].name,
                "projection_contract_payload_sha256": preflight.sha256_file(
                    files["projection_payload"]
                ),
            }
            bindings = preflight.validate_manifest_artifact_bindings(
                manifest,
                contract,
                profile_dir=profile,
                formal_manifest=files["formal_manifest"],
                projection_contract=files["projection_contract"],
                root=root,
            )
            self.assertEqual(
                set(bindings),
                {
                    "formal_manifest", "ordered_payload", "cohort",
                    "projection_contract", "projection_payload",
                },
            )
            corrupted = copy.deepcopy(manifest)
            corrupted["payload_file"] = "../ordered_term_items.npz"
            with self.assertRaisesRegex(ValueError, "basename"):
                preflight.validate_manifest_artifact_bindings(
                    corrupted,
                    contract,
                    profile_dir=profile,
                    formal_manifest=files["formal_manifest"],
                    projection_contract=files["projection_contract"],
                    root=root,
                )

    def test_fixed_entry_is_branch_complete_and_never_admits(self) -> None:
        report = preflight.evaluate_fixed_preflight()
        self.assertFalse(report["admission_generated"])
        self.assertEqual(report["expected_head_groups"], 13_800)
        self.assertEqual(report["hxh_projection_tasks"], 210_600)
        self.assertEqual(report["hxh_task_sha256"], preflight.EXPECTED_HXH_TASK_SHA256)
        if preflight.FORMAL_MANIFEST.is_file():
            self.assertEqual(report["status"], "PREFLIGHT_PASS_NOT_G0")
            self.assertTrue(report["formal_manifest_present"])
            self.assertEqual(len(report["formal_artifact_bindings"]), 5)
            self.assertTrue(
                report["formal_group_contract"]["canonical_key_coverage_exact"]
            )
        else:
            self.assertEqual(report["status"], "DENY_FORMAL_MANIFEST_ABSENT")
            self.assertFalse(report["formal_manifest_present"])
            self.assertIsNone(report["formal_artifact_bindings"])
        self.assertEqual(preflight.validate_report_for_packaging(report), report)

        forged = copy.deepcopy(report)
        forged["status"] = (
            "DENY_FORMAL_MANIFEST_ABSENT"
            if report["status"] == "PREFLIGHT_PASS_NOT_G0"
            else "PREFLIGHT_PASS_NOT_G0"
        )
        with self.assertRaisesRegex(ValueError, "independently replayed"):
            preflight.validate_report_for_packaging(forged)


if __name__ == "__main__":
    unittest.main()
