import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_m22_ordered_system_transactions.py"
SPEC = importlib.util.spec_from_file_location("m22_ordered_system_transactions", str(SCRIPT))
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class M22OrderedSystemTransactionsTest(unittest.TestCase):
    def write_csv(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def refresh_contract_hashes(self, directory, contract):
        contract["files_sha256"] = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in MODULE.REQUIRED_FILES
        }

    def refresh_receipt_binding(self, directory, name):
        receipt_path = directory / "dual_line_trace.sha256"
        receipt = receipt_path.read_text(encoding="utf-8")
        old = MODULE.parse_sha_receipt(receipt_path)[name]
        receipt_path.write_text(
            receipt.replace(old, MODULE.sha256(directory / name)), encoding="utf-8"
        )

    def make_identity(self, root):
        directory = root / "trace"
        directory.mkdir()
        execution = [
            {
                "call_index": 0, "dense_macs": 64, "input_active": 6,
                "input_elements": 32, "input_shape": "[2,2,8]", "kind": "operator",
                "name": "net.op", "operator": "Linear", "output_elements": 16,
                "output_shape": "[2,2,4]", "pair_total": "", "sample_id": 0,
                "sample_key": "sample.npy", "scope": "encoder", "sequence_key": "seq",
                "stage": "", "temporal_steps": "", "token_total": "", "windows": "",
            },
            {
                "call_index": 1, "dense_macs": 16, "input_active": "",
                "input_elements": 16, "input_shape": "[2,2,4]", "kind": "atlif",
                "name": "net.sn", "operator": "", "output_elements": 16,
                "output_shape": "[2,2,4]", "pair_total": "", "sample_id": 0,
                "sample_key": "sample.npy", "scope": "", "sequence_key": "seq",
                "stage": "", "temporal_steps": 2, "token_total": "", "windows": "",
            },
            {
                "call_index": 2, "dense_macs": "", "input_active": "",
                "input_elements": "", "input_shape": "", "kind": "attention",
                "name": "S0.B0.attn", "operator": "", "output_elements": "",
                "output_shape": "", "pair_total": 8, "sample_id": 0,
                "sample_key": "sample.npy", "scope": "", "sequence_key": "seq",
                "stage": 0, "temporal_steps": "", "token_total": 16, "windows": 1,
            },
        ]
        dual = []
        for timestep, current, positive, local, motion, selected, local_rows, motion_rows in (
            (0, 3, 3, 12, 12, 12, 2, 0),
            (1, 3, 1, 12, 4, 4, 0, 2),
        ):
            dual.append({
                "current_source_count": current, "input_shape": "[2,2,8]",
                "local_selected_rows": local_rows, "local_work": local,
                "motion_selected_rows": motion_rows, "motion_work": motion,
                "name": "net.op", "negative_transition_source_count": 0,
                "operator": "Linear", "operator_call_index": 0,
                "output_channel_fanout": 4, "positive_transition_source_count": positive,
                "sample_id": 0, "sample_key": "sample.npy", "scope": "encoder",
                "selected_work": selected, "selector_rows": 2,
                "selector_saved_work": local - selected, "sequence_key": "seq",
                "state_valid": timestep > 0, "status": "PASS_EXACT_SOURCE_WORK",
                "temporal_step": timestep, "valid_source_work": 64,
            })
        operators = [{
            "activity_weighted_macs_proxy": 24, "calls": 1, "dense_macs": 64,
            "input_elements": 32, "input_sample_elements": 32,
            "input_sample_binary01_ratio": 1.0,
            "name": "net.op", "operator": "Linear", "output_elements": 16,
            "scope": "encoder", "weight_elements": 16,
        }]
        atlifs = [{
            "active": 4, "activity": 0.25, "calls": 1, "deployment_dead_result": False,
            "elements": 16, "name": "net.sn", "parameter_entries": 4,
            "temporal_steps": 2,
        }]
        self.write_csv(directory / "execution_trace.csv", execution)
        self.write_csv(directory / "dual_line_operator_trace.csv", dual)
        self.write_csv(directory / "operator_runtime.csv", operators)
        self.write_csv(directory / "atlif_activity.csv", atlifs)
        eval_protocol = {"resolution": [2, 2], "tokens_per_window": 8}
        profile_identity = {
            "experiment": "fixture", "samples": 1,
            "checkpoint_basename": "checkpoint.pth", "checkpoint_sha256": "c" * 64,
            "config_basename": "config.yml", "config_sha256": "d" * 64,
            "module_counts": {"ATLIFTernaryPSN": 1, "ShiftmaxAttention": 1},
            "eval_protocol": eval_protocol,
        }
        profile = {
            "experiment": "fixture", "config": "/remote/config.yml",
            "checkpoint": "/remote/checkpoint.pth", "samples": 1,
            "ordered_trace": True, "dual_line_trace": True,
            "module_counts": profile_identity["module_counts"],
            "checkpoint_load_audit": {
                "checkpoint": "/remote/checkpoint.pth", "missing_count": 0,
                "unexpected_count": 0, "overlay_missing_count": 0,
                "overlay_unexpected_count": 0, "missing_sample": [], "unexpected_sample": [],
            },
            "eval_protocol": eval_protocol,
            "artifact_identity": {
                "checkpoint_path": "/remote/checkpoint.pth", "checkpoint_sha256": "c" * 64,
                "config_path": "/remote/config.yml", "config_sha256": "d" * 64,
            },
            "summary": {
                "execution_records": 3, "dual_line_records": 2,
                "operator_rows": 1, "atlif_rows": 1,
            },
        }
        (directory / "nts11_hardware_p0_profile.json").write_text(
            json.dumps(profile), encoding="utf-8"
        )
        receipt = []
        for name in ("execution_trace.csv", "dual_line_operator_trace.csv", "nts11_hardware_p0_profile.json"):
            digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            receipt.append("{}  /remote/producer/path/{}\n".format(digest, name))
        (directory / "dual_line_trace.sha256").write_text("".join(receipt), encoding="utf-8")
        contract = {"directory": "trace", "profile_identity": profile_identity}
        self.refresh_contract_hashes(directory, contract)
        return directory, contract

    def config(self):
        return {
            "dram_bytes_per_cycle": 16, "sram_bytes_per_cycle": 8,
            "activation_bits": 8, "weight_bits": 8, "accumulator_bits": 32,
            "selector_bits_per_row": 1,
            "boundary_policy": "OPERATOR_AND_ATLIF_BOUNDARIES_MATERIALIZED_ATTENTION_EXCLUDED",
            "motion_policy": "AGGREGATE_ROW_SELECTOR_WITH_SHARED_AND_EXPLICIT_COPY_STATE_TRANSPORT",
        }

    def manifest_meta(self):
        return {"path": "/trusted/manifest.json", "sha256": "e" * 64,
                "expected_sha256_from_cli": "e" * 64}

    def test_trusted_manifest_and_remote_receipt_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, contract = self.make_identity(root)
            validated = MODULE.validate_identity(directory, contract)
            self.assertEqual(validated["sample_count"], 1)
            producer = root / "producer.py"
            producer.write_text("# producer\n", encoding="utf-8")
            manifest = {
                "schema": "m22_ordered_trace_input_manifest_v2",
                "status": "FROZEN_EXPECTED_INPUT_IDENTITY",
                "producer_source": {"path": "producer.py", "sha256": MODULE.sha256(producer)},
                "identities": {"fixture": contract},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            admitted = MODULE.load_input_manifest(
                manifest_path, MODULE.sha256(manifest_path), root
            )
            self.assertEqual(admitted[2][0][0], "fixture")
            with self.assertRaisesRegex(ValueError, "manifest SHA mismatch"):
                MODULE.load_input_manifest(manifest_path, "0" * 64, root)

            operator_path = directory / "operator_runtime.csv"
            text = operator_path.read_text(encoding="utf-8").replace(",16\n", ",999\n")
            operator_path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trusted input manifest mismatch"):
                MODULE.validate_identity(directory, contract)

    def test_local_shared_and_copy_state_traffic_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, contract = self.make_identity(Path(temporary))
            validated = MODULE.validate_identity(directory, contract)
            result = {}
            for variant in (
                "local_line", "motion_selector_shared_state", "motion_selector_explicit_copy"
            ):
                result[variant] = MODULE.schedule_variant(validated, "fixture", variant, self.config())
        local, local_totals = result["local_line"][:2]
        shared, shared_totals = result["motion_selector_shared_state"][:2]
        copy, copy_totals = result["motion_selector_explicit_copy"][:2]
        self.assertEqual(sum(row["byte_count"] for row in local if row["phase"] == "coefficient_term_read"), 24)
        self.assertEqual(sum(row["byte_count"] for row in shared if row["phase"] == "coefficient_term_read"), 16)
        self.assertEqual(shared_totals["motion_previous_bitmap_read_bytes"], 2)
        self.assertEqual(shared_totals["motion_previous_acc_read_bytes"], 32)
        self.assertEqual(shared_totals["motion_selector_decision_read_bytes"], 1)
        self.assertEqual(shared_totals["motion_selector_decision_write_bytes"], 1)
        self.assertEqual(shared_totals["motion_incremental_state_peak_bytes"], 1)
        self.assertEqual(copy_totals["motion_state_bitmap_copy_write_bytes"], 2)
        self.assertEqual(copy_totals["motion_state_acc_copy_write_bytes"], 32)
        self.assertEqual(copy_totals["motion_incremental_state_peak_bytes"], 35)
        self.assertFalse(any(row["phase"].startswith("motion_") for row in local))
        self.assertEqual(local_totals["dram_read_bytes"], shared_totals["dram_read_bytes"])
        self.assertEqual(local_totals["dram_read_bytes"], copy_totals["dram_read_bytes"])
        self.assertEqual([row["request_issue_order"] for row in shared], list(range(len(shared))))
        self.assertEqual(
            [row["previous_in_trace_order"] for row in shared],
            [-1] + list(range(len(shared) - 1)),
        )
        self.assertTrue(all(
            row["serialized_service_end_exclusive"] > row["serialized_service_start"]
            for row in shared
        ))

    def test_determinism_claims_and_output_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, contract = self.make_identity(root)
            identities = [("fixture", directory, contract)]
            first, rows_a = MODULE.build(identities, self.config(), self.manifest_meta())
            second, rows_b = MODULE.build(identities, self.config(), self.manifest_meta())
            self.assertEqual(rows_a, rows_b)
            self.assertEqual(first, second)
            self.assertEqual([row["transaction_id"] for row in rows_a], list(range(len(rows_a))))
            comparisons = first["identities"]["fixture"]["motion_models_vs_local"]
            self.assertEqual(set(comparisons), {
                "motion_selector_shared_state", "motion_selector_explicit_copy"
            })
            self.assertTrue(all(row["address"].startswith("0x") for row in rows_a))
            self.assertIn("PARTIAL", first["status"])
            forbidden = " ".join(first["claim_boundary"]["forbidden"])
            self.assertIn("speedup", forbidden)
            self.assertIn("selector/popcount", forbidden)
            input_manifest = root / "trusted.json"
            input_manifest.write_text("{}", encoding="utf-8")
            first["input_manifest"]["sha256"] = MODULE.sha256(input_manifest)
            output = root / "out"
            MODULE.write_outputs(output, first, rows_a, input_manifest)
            manifest = json.loads((output / "m22_output_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["transaction_records"], len(rows_a))
            for name, receipt in manifest["artifacts"].items():
                self.assertEqual(receipt["sha256"], MODULE.sha256(output / name))

    def test_strengthened_dual_profile_and_order_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, contract = self.make_identity(Path(temporary))
            dual_path = directory / "dual_line_operator_trace.csv"
            rows, _ = MODULE.read_csv(dual_path)
            rows[1]["positive_transition_source_count"] = "0"
            rows[1]["motion_work"] = "0"
            self.write_csv(dual_path, rows)
            self.refresh_receipt_binding(directory, dual_path.name)
            self.refresh_contract_hashes(directory, contract)
            with self.assertRaisesRegex(ValueError, "exceeds Motion"):
                MODULE.validate_identity(directory, contract)

        with tempfile.TemporaryDirectory() as temporary:
            directory, contract = self.make_identity(Path(temporary))
            profile_path = directory / "nts11_hardware_p0_profile.json"
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["artifact_identity"]["checkpoint_sha256"] = "f" * 64
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            self.refresh_receipt_binding(directory, profile_path.name)
            self.refresh_contract_hashes(directory, contract)
            with self.assertRaisesRegex(ValueError, "profile checkpoint"):
                MODULE.validate_identity(directory, contract)

    def test_negative_counts_identity_attention_and_atlif_fail_closed(self):
        cases = (
            ("negative counts", "dual", "dual-line exact counts"),
            ("identity mismatch", "identity", "owner/call/identity"),
            ("attention extent", "attention", "attention summary extent"),
            ("ATLIF extent", "atlif", "ATLIF runtime parameter"),
            ("binary NaN", "binary_nan", "packed-binary"),
            ("binary Inf", "binary_inf", "packed-binary"),
            ("binary 0.999", "binary_999", "packed-binary"),
        )
        for label, mutation, expected in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                directory, contract = self.make_identity(Path(temporary))
                if mutation in ("dual", "identity"):
                    path = directory / "dual_line_operator_trace.csv"
                    rows, _ = MODULE.read_csv(path)
                    if mutation == "dual":
                        rows[1]["current_source_count"] = "-1"
                        rows[1]["local_work"] = "-4"
                    else:
                        rows[1]["sample_key"] = "wrong.npy"
                    self.write_csv(path, rows)
                    self.refresh_receipt_binding(directory, path.name)
                elif mutation == "attention":
                    path = directory / "execution_trace.csv"
                    rows, _ = MODULE.read_csv(path)
                    rows[2]["windows"] = "0"
                    self.write_csv(path, rows)
                    self.refresh_receipt_binding(directory, path.name)
                elif mutation == "atlif":
                    path = directory / "atlif_activity.csv"
                    rows, _ = MODULE.read_csv(path)
                    rows[0]["parameter_entries"] = "0"
                    self.write_csv(path, rows)
                else:
                    path = directory / "operator_runtime.csv"
                    rows, _ = MODULE.read_csv(path)
                    rows[0]["input_sample_binary01_ratio"] = {
                        "binary_nan": "NaN", "binary_inf": "Infinity", "binary_999": "0.999"
                    }[mutation]
                    self.write_csv(path, rows)
                self.refresh_contract_hashes(directory, contract)
                with self.assertRaisesRegex(ValueError, expected):
                    MODULE.validate_identity(directory, contract)

    def test_frozen_real_manifest_and_cardinalities(self):
        repo = Path(__file__).resolve().parents[3]
        path = repo / "hw_autoresearch_nts07/contracts/m22_ordered_trace_input_manifest_r2_20260822.json"
        expected_sha = "82178cfed470d188f18e6917c161b979f60469e724730511f823d149ed7523e4"
        _manifest, actual, identities = MODULE.load_input_manifest(path, expected_sha, repo)
        self.assertEqual(actual, expected_sha)
        expected = {
            "h67_ep35": (1840, 3580, "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158"),
            "local_ep44": (1720, 4030, "19820bec07cc3bf3da7e9e2e31e2af0b36bda89e636b0d273c0257b368c34f57"),
        }
        for label, directory, contract in identities:
            validated = MODULE.validate_identity(directory, contract)
            execution_count, dual_count, checkpoint_sha = expected[label]
            self.assertEqual(len(validated["execution"]), execution_count)
            self.assertEqual(sum(validated["status_counts"].values()), dual_count)
            self.assertEqual(validated["profile_identity"]["checkpoint_sha256"], checkpoint_sha)

    def test_frozen_r2_output_receipt(self):
        repo = Path(__file__).resolve().parents[3]
        receipt_path = repo / "hw_autoresearch_nts07/contracts/m22_output_receipt_r2_final_20260822.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["schema"], "m22_frozen_output_receipt_v2")
        artifact = repo / receipt["artifact_directory"]
        for name, expected_sha in receipt["files_sha256"].items():
            self.assertEqual(MODULE.sha256(artifact / name), expected_sha)
        summary = json.loads((artifact / "m22_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["transaction_records"], receipt["transaction_records"])
        self.assertEqual(summary["transactions_sha256"], receipt["files_sha256"]["m22_ordered_transactions.csv"])
        for identity, expected in receipt["core_transport"].items():
            actual = summary["identities"][identity]
            for variant, totals in expected["variants"].items():
                for key, value in totals.items():
                    self.assertEqual(actual["variants"][variant]["totals"][key], value)


if __name__ == "__main__":
    unittest.main()
