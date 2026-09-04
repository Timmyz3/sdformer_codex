import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "hw_autoresearch_nts07/system_simulator/scripts/"
    "validate_m31_r4_static_phase_vcs.py"
)
SPEC = importlib.util.spec_from_file_location("m31r4validator", str(SCRIPT))
M31 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M31)
RESULT = ROOT / (
    "hw_autoresearch_nts07/results/"
    "m31_r4_static_phase_vcs_machine_admission_20260822/"
    "m31_r4_static_phase_vcs_machine_admission.json"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class M31R4StaticPhaseValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads(
            M31.DEFAULT_RECEIPT.read_text(encoding="utf-8")
        )

    def make_rehashed_run(self, directory, mutate_sim=None,
                          mutate_input_manifest=None):
        directory = Path(directory)
        run = directory / "m31_unified_t10_t2_vcs_r4_static_phase_20260822"
        run.mkdir()
        source_run = Path(self.receipt["vcs_run"]["directory"])
        for name in ("input_sha256.txt", "compile.log", "sim.log"):
            shutil.copy2(str(source_run / name), str(run / name))
        if mutate_sim is not None:
            path = run / "sim.log"
            path.write_text(mutate_sim(path.read_text(encoding="utf-8")),
                            encoding="utf-8")
        if mutate_input_manifest is not None:
            path = run / "input_sha256.txt"
            path.write_text(
                mutate_input_manifest(path.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
        output = run / "output_sha256.txt"
        output.write_text(
            "{}  {}\n{}  {}\n".format(
                digest(run / "compile.log"), run / "compile.log",
                digest(run / "sim.log"), run / "sim.log",
            ),
            encoding="utf-8",
        )
        receipt = copy.deepcopy(self.receipt)
        receipt["vcs_run"].update({
            "directory": str(run),
            "input_sha256_manifest": digest(run / "input_sha256.txt"),
            "output_sha256_manifest": digest(output),
            "compile_log": digest(run / "compile.log"),
            "sim_log": digest(run / "sim.log"),
        })
        fake = directory / "receipt.json"
        fake.write_text(json.dumps(receipt), encoding="utf-8")
        return fake

    def test_positive_machine_admission(self):
        result = M31.build()
        self.assertEqual(
            result["identity"]["receipt_sha256"],
            M31.EXPECTED_RECEIPT_SHA256,
        )
        self.assertEqual(result["manifest_audit"]["input_count"], 6)
        self.assertEqual(result["manifest_audit"]["output_count"], 2)
        self.assertEqual(result["log_audit"]["assert_property_count"], 24)
        self.assertEqual(result["log_audit"]["cover_property_count"], 4)
        self.assertEqual(
            result["source_audit"]["dynamic_phase_indexed_t10_arrays"], 0
        )
        self.assertFalse(result["admission"]["dc_sta_admitted"])
        self.assertFalse(result["admission"]["formality_admitted"])
        self.assertFalse(result["admission"]["headline_admitted"])
        self.assertEqual(
            result, json.loads(RESULT.read_text(encoding="utf-8"))
        )

    def test_extra_receipt_key_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = copy.deepcopy(self.receipt)
            receipt["unreviewed_extra"] = True
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "key population drift"):
                M31.build(path, enforce_receipt_sha=False)

    def test_forged_summary_with_all_run_hashes_refreshed_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            def mutate(text):
                return text.replace("t10_tiles=28", "t10_tiles=29", 1)
            receipt = self.make_rehashed_run(directory, mutate_sim=mutate)
            with self.assertRaisesRegex(ValueError, "observed receipt drift"):
                M31.build(receipt, enforce_receipt_sha=False)

    def test_forged_cover_with_all_run_hashes_refreshed_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            def mutate(text):
                return text.replace("527 attempts, 26 match",
                                    "527 attempts, 0 match", 1)
            receipt = self.make_rehashed_run(directory, mutate_sim=mutate)
            with self.assertRaisesRegex(ValueError, "cover population drift"):
                M31.build(receipt, enforce_receipt_sha=False)

    def test_forged_warning_with_all_run_hashes_refreshed_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            def mutate(text):
                return text + "Warning: forged clean-log bypass\n"
            receipt = self.make_rehashed_run(directory, mutate_sim=mutate)
            with self.assertRaisesRegex(ValueError, "warning population"):
                M31.build(receipt, enforce_receipt_sha=False)

    def test_rehashed_extra_manifest_entry_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            core = ROOT / M31.EXPECTED_FILES["unified_core_rtl"][0]
            def mutate(text):
                return text + "{}  {}\n".format(digest(core), core)
            receipt = self.make_rehashed_run(
                directory, mutate_input_manifest=mutate
            )
            with self.assertRaisesRegex(ValueError, "manifest population drift"):
                M31.build(receipt, enforce_receipt_sha=False)

    def test_rehashed_source_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = copy.deepcopy(self.receipt)
            receipt["files"]["unified_core_rtl"][1] = "0" * 64
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact source identity drift"):
                M31.build(path, enforce_receipt_sha=False)

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = M31.DEFAULT_RECEIPT.read_text(encoding="utf-8")
            raw = raw.replace('{\n  "schema"',
                              '{\n  "schema": "forged",\n  "schema"', 1)
            path = Path(directory) / "receipt.json"
            path.write_text(raw, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                M31.build(path, enforce_receipt_sha=False)

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "admission.json"
            output.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                M31.write_output(output, {"bad": True})
            self.assertEqual(output.read_text(encoding="utf-8"), "occupied")


if __name__ == "__main__":
    unittest.main()
