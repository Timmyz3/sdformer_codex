from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.local5_erep_integrated_cross_head_merge import validate_execution_binding
from scripts.local5_erep_numeric_release import (
    HEADS,
    REQUIRED_TOOLS,
    TOOL_SCHEMA,
    seal_release,
    sha256,
    verify_release,
)


class NumericReleaseTest(unittest.TestCase):
    def test_release_runner_seals_archive_replay_transitive_dependencies(self) -> None:
        runner = Path("sim_qfit/run_local5_erep_numeric_sample_shard.sh").read_text(
            encoding="utf-8"
        )
        for dependency in (
            "scripts/local5_erep_ledger_replay_v4.py",
            "scripts/local5_erep_capacity_baselines_v4.py",
            "scripts/local5_erep_command_schedule_v4.py",
            "scripts/local5_erep_identity_service_v4.py",
            "scripts/generate_local5_identity_service_tables_v4.py",
            "scripts/verify_local5_identity_service_tables_v4.py",
            "scripts/verify_local5_identity_service_rtl_trace_v1.py",
        ):
            self.assertIn(dependency, runner)
        self.assertIn("RELEASE_SERVICE_MODE", runner)

    def make_release(self, root: Path) -> dict[str, Path]:
        source = root / "source/source.sv"
        source.parent.mkdir(parents=True)
        source.write_text("module source; endmodule\n", encoding="ascii")
        (root / "source_sha256.txt").write_text(
            f"{sha256(source)}  source.sv\n", encoding="utf-8"
        )
        with tarfile.open(root / "source_bundle.tar", "w") as archive:
            archive.add(source, arcname="source.sv")
        (root / "tool_versions.txt").write_text("tools\n", encoding="ascii")
        tool_root = root / "tools"
        tool_root.mkdir()
        tool_rows = []
        for name in sorted(REQUIRED_TOOLS):
            tool = tool_root / name
            tool.write_bytes(f"tool-{name}".encode("ascii"))
            tool_rows.append(
                {
                    "name": name,
                    "path": str(tool.resolve()),
                    "sha256": sha256(tool),
                    "version": f"{name}-test",
                }
            )
        (root / "tool_bindings.json").write_text(
            json.dumps({"schema": TOOL_SCHEMA, "tools": tool_rows}), encoding="utf-8"
        )
        verilator = next(row["path"] for row in tool_rows if row["name"] == "verilator")
        for heads in HEADS:
            build = root / "build" / f"h{heads}"
            executable = build / "obj/Vtb_qfit_local5_memo_multitile_cross_head"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(f"binary-{heads}".encode("ascii"))
            (build / "compile.log").write_text("PASS\n", encoding="ascii")
            argv = [
                verilator,
                "--binary",
                "--timing",
                "--assert",
                "--top-module",
                "tb_qfit_local5_memo_multitile_cross_head",
                "--Mdir",
                f"build/h{heads}/obj",
                "-GUSE_MEMO=0",
                "-GUSE_INPLACE=0",
                "-GTRANSACTION_INDEXED_SERVICE=1",
                f"-GHEADS={heads}",
                f"-GOUTPUT_TILES={heads}",
                "-GTIMEOUT_CYCLES=100000000",
                "source/source.sv",
                "source/tb_qfit/tb_qfit_local5_memo_multitile_cross_head.sv",
            ]
            # The frozen TB path is part of the compile contract and source closure.
            if not (root / "source/tb_qfit/tb_qfit_local5_memo_multitile_cross_head.sv").exists():
                tb = root / "source/tb_qfit/tb_qfit_local5_memo_multitile_cross_head.sv"
                tb.parent.mkdir(parents=True)
                tb.write_text("module tb_qfit_local5_memo_multitile_cross_head; endmodule\n")
            (build / "compile_argv.json").write_text(json.dumps(argv), encoding="utf-8")
        tb = root / "source/tb_qfit/tb_qfit_local5_memo_multitile_cross_head.sv"
        helper = root / "source/scripts/helper.py"
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text("print('helper-pass')\n", encoding="ascii")
        bindings = [
            ("source.sv", source),
            ("tb_qfit/tb_qfit_local5_memo_multitile_cross_head.sv", tb),
            ("scripts/helper.py", helper),
        ]
        (root / "source_sha256.txt").write_text(
            "".join(f"{sha256(path)}  {name}\n" for name, path in bindings),
            encoding="utf-8",
        )
        with tarfile.open(root / "source_bundle.tar", "w") as archive:
            for name, path in bindings:
                archive.add(path, arcname=name)
        seal_release(root)
        return {"source": source, "tb": tb, "tool_root": tool_root}

    def make_receipt(self, root: Path, heads: int = 3) -> tuple[dict[str, object], Path]:
        executable = root / f"build/h{heads}/obj/Vtb_qfit_local5_memo_multitile_cross_head"
        compile_path = root / f"build/h{heads}/compile_argv.json"
        actual = root / "actual.memh"
        inputs = root / "inputs.txt"
        weights = root / "weights.txt"
        actual.write_text("00000000\n", encoding="ascii")
        inputs.write_text("0\n", encoding="ascii")
        weights.write_text("0\n", encoding="ascii")
        identity = {"sample": 0, "stage": 0, "block": 0, "window": 94, "heads": heads}
        service_seed = 17717
        vector_bindings = [
            {"name": "combined_head_inputs", "path": str(inputs.resolve()), "entries": 1, "sha256": sha256(inputs)},
            {"name": "projection_weights", "path": str(weights.resolve()), "entries": 1, "sha256": sha256(weights)},
        ]
        run_argv = [
            str(executable.resolve()),
            f"+INPUTS={inputs.resolve()}",
            f"+WEIGHTS={weights.resolve()}",
            "+STAGE_ID=0",
            "+BLOCK_ID=0",
            "+WINDOW_ID=94",
            "+NO_ACC_CHECK",
            f"+SERVICE_SEED={service_seed}",
            f"+ACTUAL_ACC_FILE={actual.resolve()}",
        ]
        run_path = root / "run_argv.json"
        run_path.write_text(json.dumps(run_argv), encoding="utf-8")
        receipt: dict[str, object] = {
            "provenance_level": "exact_argv_sealed_release",
            "identity": identity,
            "service_seed": service_seed,
            "actual_acc32": str(actual.resolve()),
            "vector_file_bindings": vector_bindings,
            "executable": str(executable.resolve()),
            "executable_sha256": sha256(executable),
            "tool_versions": str((root / "tool_versions.txt").resolve()),
            "tool_versions_sha256": sha256(root / "tool_versions.txt"),
            "run_argv": run_argv,
            "run_argv_file": str(run_path.resolve()),
            "run_argv_file_sha256": sha256(run_path),
            "compile_argv": json.loads(compile_path.read_text(encoding="utf-8")),
            "compile_argv_file": str(compile_path.resolve()),
            "compile_argv_file_sha256": sha256(compile_path),
            "release_manifest": str((root / "release_manifest.json").resolve()),
            "release_manifest_sha256": sha256(root / "release_manifest.json"),
        }
        return receipt, run_path

    def test_seal_and_verify_exact_four_builds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_release(root)
            manifest = verify_release(root)
            self.assertEqual(set(manifest["builds"]), {"3", "6", "12", "24"})
            self.assertEqual(manifest["formal_g0"], "DENY")
            self.assertEqual(
                {row["service_mode"] for row in manifest["builds"].values()},
                {"transaction_indexed"},
            )

    def test_verify_is_independent_of_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as elsewhere:
            root = Path(temporary)
            self.make_release(root)
            previous = Path.cwd()
            try:
                os.chdir(elsewhere)
                self.assertEqual(verify_release(root)["formal_g0"], "DENY")
            finally:
                os.chdir(previous)

    def test_source_tree_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_release(root)
            paths["source"].write_text("module changed; endmodule\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "source tree"):
                verify_release(root)

    def test_extra_bytecode_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_release(root)
            cache = root / "source/scripts/__pycache__"
            cache.mkdir()
            (cache / "helper.pyc").write_bytes(b"cache")
            with self.assertRaisesRegex(ValueError, "source tree"):
                verify_release(root)

    def test_consumer_with_bytecode_disabled_keeps_release_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_release(root)
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run(
                [sys.executable, str(root / "source/scripts/helper.py")],
                check=True,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertFalse((root / "source/scripts/__pycache__").exists())
            self.assertEqual(verify_release(root)["formal_g0"], "DENY")

    def test_source_bundle_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_release(root)
            with (root / "source_bundle.tar").open("ab") as handle:
                handle.write(b"mutated")
            with self.assertRaisesRegex(ValueError, "source_bundle_path"):
                verify_release(root)

    def test_tool_binary_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_release(root)
            (paths["tool_root"] / "verilator").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "tool binding"):
                verify_release(root)

    def test_executable_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_release(root)
            executable = root / "build/h3/obj/Vtb_qfit_local5_memo_multitile_cross_head"
            executable.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "H3 build binding"):
                verify_release(root)

    def test_build_argv_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_release(root)
            argv_path = root / "build/h3/compile_argv.json"
            argv = json.loads(argv_path.read_text(encoding="utf-8"))
            argv.append("--trace")
            argv_path.write_text(json.dumps(argv), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "H3 build binding"):
                verify_release(root)

    def test_seal_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_release(root)
            with self.assertRaisesRegex(ValueError, "不允许覆盖"):
                seal_release(root)

    def test_exact_argv_receipt_binds_release_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_release(root)
            receipt, run_path = self.make_receipt(root)
            validate_execution_binding(receipt, "verilator")
            run_argv = json.loads(run_path.read_text(encoding="utf-8"))
            run_argv[3] = "+STAGE_ID=1"
            run_path.write_text(json.dumps(run_argv), encoding="utf-8")
            receipt["run_argv"] = run_argv
            receipt["run_argv_file_sha256"] = sha256(run_path)
            with self.assertRaisesRegex(ValueError, "task/vector/output"):
                validate_execution_binding(receipt, "verilator")

    def test_wrong_h_binary_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_release(root)
            receipt, run_path = self.make_receipt(root, heads=3)
            wrong = root / "build/h6/obj/Vtb_qfit_local5_memo_multitile_cross_head"
            receipt["executable"] = str(wrong.resolve())
            receipt["executable_sha256"] = sha256(wrong)
            run_argv = list(receipt["run_argv"])
            run_argv[0] = str(wrong.resolve())
            run_path.write_text(json.dumps(run_argv), encoding="utf-8")
            receipt["run_argv"] = run_argv
            receipt["run_argv_file_sha256"] = sha256(run_path)
            with self.assertRaisesRegex(ValueError, "H-class"):
                validate_execution_binding(receipt, "verilator")


if __name__ == "__main__":
    unittest.main()
