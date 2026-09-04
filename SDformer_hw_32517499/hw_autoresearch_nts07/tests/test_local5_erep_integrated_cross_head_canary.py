from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.local5_erep_integrated_cross_head_actual import (
    parse_acc32,
    parse_unique_identity,
    parse_unique_terminal,
    validate_vector_files,
)
from scripts.local5_erep_integrated_cross_head_merge import (
    validate_execution_binding,
)
from scripts.local5_erep_integrated_cross_head_vectors import validate_plan


class IntegratedCrossHeadCanaryTest(unittest.TestCase):
    def make_inputs(self):
        groups = [
            {
                "sample": 0,
                "stage": 0,
                "block": 0,
                "window": 94,
                "head": head,
            }
            for head in range(3)
        ]
        plan = {
            "schema": "local5_projection_task_plan_v1",
            "heads": 3,
            "tasks": [
                {"input_group_index": head, "output_tile": tile}
                for tile in range(3)
                for head in range(3)
            ],
        }
        return plan, {"groups": groups}

    def test_validate_complete_hxh(self):
        plan, manifest = self.make_inputs()
        groups, heads = validate_plan(plan, manifest)
        self.assertEqual(heads, 3)
        self.assertEqual([row[0] for row in groups], [0, 1, 2])

    def test_reject_noncanonical_order(self):
        plan, manifest = self.make_inputs()
        plan["tasks"][0], plan["tasks"][1] = (
            plan["tasks"][1],
            plan["tasks"][0],
        )
        with self.assertRaisesRegex(ValueError, "完整HxH顺序"):
            validate_plan(plan, manifest)

    def test_parse_signed_acc32(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actual.memh"
            path.write_text("00000001\nffffffff\n80000000\n", encoding="ascii")
            self.assertEqual(parse_acc32(path), [1, -1, -(1 << 31)])

    def test_terminal_must_be_unique(self):
        line = "PASS Local5 multi-tile cycles=123 final=43200"
        self.assertEqual(parse_unique_terminal(line), (123, 43200))
        with self.assertRaisesRegex(ValueError, "唯一PASS"):
            parse_unique_terminal(f"{line}\n{line}\n")

    def test_terminal_identity_must_be_unique(self):
        line = (
            "PASS Local5 multi-tile memo=0 stage=2 block=5 "
            "window=317 cycles=123 final=172800"
        )
        self.assertEqual(parse_unique_identity(line), (2, 5, 317))
        with self.assertRaisesRegex(ValueError, "唯一stage"):
            parse_unique_identity(f"{line}\n{line}\n")

    def test_vector_helper_and_numpy_binding_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vector = root / "input.txt"
            vector.write_text("00\n", encoding="ascii")
            generator = Path(
                "scripts/local5_erep_integrated_cross_head_vectors.py"
            ).resolve()
            helpers = [
                Path("scripts/generate_local5_checkpoint_score_vectors.py").resolve(),
                Path("scripts/generate_local5_masked_integer_vectors.py").resolve(),
            ]
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {
                "generator_binding": {
                    "file": str(generator),
                    "sha256": digest(generator),
                    "helpers": [
                        {"file": str(path), "sha256": digest(path)}
                        for path in helpers
                    ],
                    "numpy_version": np.__version__,
                },
                "files": {
                    "input": {
                        "file": vector.name,
                        "entries": 1,
                        "sha256": digest(vector),
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(len(validate_vector_files(manifest_path, manifest)), 1)
            manifest["generator_binding"]["helpers"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "helper SHA"):
                validate_vector_files(manifest_path, manifest)
            manifest["generator_binding"]["helpers"][0]["sha256"] = digest(
                helpers[0]
            )
            manifest["generator_binding"]["numpy_version"] = "0.0.0"
            with self.assertRaisesRegex(ValueError, "helper绑定"):
                validate_vector_files(manifest_path, manifest)

    def test_execution_binding_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "simv"
            tools = root / "tools.txt"
            executable.write_bytes(b"binary")
            tools.write_text("tool versions\n", encoding="utf-8")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            receipt = {
                "executable": str(executable),
                "executable_sha256": digest(executable),
                "tool_versions": str(tools),
                "tool_versions_sha256": digest(tools),
                "run_command": (
                    "vvp simv +NO_ACC_CHECK +WEIGHTS=w "
                    "+ACTUAL_ACC_FILE=a +STAGE_ID=0 "
                    "+BLOCK_ID=0 +WINDOW_ID=94"
                ),
                "compile_command": (
                    "tb_qfit_local5_memo_multitile_cross_head "
                    "USE_MEMO=0 USE_INPLACE=0 "
                    "TRANSACTION_INDEXED_SERVICE=1 "
                    "STAGE_ID=0 BLOCK_ID=0 WINDOW_ID=94"
                ),
            }
            validate_execution_binding(receipt, "icarus")
            receipt["run_command"] = receipt["run_command"].replace(
                "+NO_ACC_CHECK ", ""
            )
            with self.assertRaisesRegex(ValueError, "命令合同"):
                validate_execution_binding(receipt, "icarus")
            receipt["run_command"] = (
                "vvp simv +NO_ACC_CHECK +WEIGHTS=w +ACTUAL_ACC_FILE=a "
                "+STAGE_ID=0 +BLOCK_ID=0 +WINDOW_ID=94"
            )
            receipt["compile_command"] = receipt["compile_command"].replace(
                "WINDOW_ID=94", "WINDOW=94"
            )
            with self.assertRaisesRegex(ValueError, "命令合同"):
                validate_execution_binding(receipt, "icarus")
            receipt["compile_command"] = receipt["compile_command"].replace(
                "WINDOW=94", "WINDOW_ID=94"
            )
            receipt["compile_command"] = receipt["compile_command"].replace(
                "USE_MEMO=0", "USE_MEMO=1"
            )
            with self.assertRaisesRegex(ValueError, "命令合同"):
                validate_execution_binding(receipt, "icarus")
            validate_execution_binding(receipt, "icarus", use_memo=1)
            receipt["compile_command"] = receipt["compile_command"].replace(
                "USE_MEMO=1", "USE_MEMO=0"
            )
            tools.write_text("mutated tools\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "executable/tool"):
                validate_execution_binding(receipt, "icarus")
            tools.write_text("tool versions\n", encoding="utf-8")
            executable.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "executable/tool"):
                validate_execution_binding(receipt, "icarus")


if __name__ == "__main__":
    unittest.main()
