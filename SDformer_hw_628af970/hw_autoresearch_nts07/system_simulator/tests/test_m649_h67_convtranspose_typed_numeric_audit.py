#!/usr/bin/env python3
"""Static and CPU-only unit tests for the M649 numeric-audit source."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "audit_m649_h67_convtranspose_typed_numeric_inputs.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m649_h67_ep35_convtranspose_typed_numeric_audit_contract_r1_20260828.json")
M511_CANONICAL = ROOT / (
    "hw_autoresearch_nts07/system_handoff/outgoing/"
    "m511_h67_ep35_convtranspose_binary_inputs_s10_r1_20260827")
M649_CANONICAL = ROOT / (
    "hw_autoresearch_nts07/results/"
    "m649_h67_ep35_convtranspose_typed_numeric_audit_s10_r1_20260828")


def load_module():
    spec = importlib.util.spec_from_file_location("m649_test_target", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M649 = load_module()


def decision_record(sample, module, pass_first2=True, dtype="torch.float32"):
    if module == 0:
        partition = {"gate_pass": pass_first2}
    else:
        partition = {
            "first2_flow_hypothesis": {
                "gate_pass": pass_first2,
                "flow_summary": {
                    "nonbinary_finite_count": 7 if pass_first2 else 0,
                },
            },
            "last2_flow_hypothesis": {
                "gate_pass": False,
                "flow_summary": {"nonbinary_finite_count": 0},
            },
        }
    return {
        "sample_id": sample,
        "module_index": module,
        "input_numeric": {"dtype": dtype, "typed_partition": partition},
    }


class M649Tests(unittest.TestCase):
    def test_contract_and_launcher_identity(self):
        contract = M649.strict_json(CONTRACT)
        self.assertEqual(
            contract["schema"],
            "m649_h67_ep35_convtranspose_typed_numeric_audit_contract_v1")
        self.assertEqual(M649.sha256(SCRIPT),
                         contract["inputs"]["launcher"]["sha256"])
        self.assertFalse(contract["resource_policy"][
            "gpu_execution_currently_authorized"])
        self.assertFalse(contract["claim_boundary"]["cycles"])
        self.assertFalse(contract["claim_boundary"]["speedup"])

    def test_all_contract_inputs_and_failed_state_are_exact(self):
        contract = M649.strict_json(CONTRACT)
        identities = M649.verify_contract_inputs(contract, SCRIPT)
        self.assertEqual(set(identities), set(contract["required_input_names"]))
        failed = M649.verify_failed_m511_state(contract)
        self.assertTrue(failed["original_m511_canonical_absent"])
        self.assertEqual(len(failed["failed_staging_population"]), 2)
        prior = M649.verify_prior_failed_m649_state(contract)
        self.assertEqual(prior["completed_records_before_prefetch_failure"], 40)
        self.assertTrue(prior["canonical_result_absent"])

    def test_no_production_outputs_exist_during_static_handoff(self):
        self.assertFalse(os.path.lexists(str(M511_CANONICAL)))
        self.assertFalse(os.path.lexists(str(M649_CANONICAL)))

    def test_d0_exact_binary(self):
        tensor = torch.tensor([[[[[0.0, 1.0]], [[1.0, 0.0]],
                                  [[0.0, 0.0]], [[1.0, 1.0]]]]])
        expected = {"module_index": 0, "name": "d0",
                    "input_shape": [1, 1, 4, 1, 2]}
        result = M649.audit_decoder_input(tensor, expected, 2)
        self.assertTrue(result["typed_partition"]["gate_pass"])
        self.assertEqual(result["full_tensor"]["zero_count"], 4)
        self.assertEqual(result["full_tensor"]["one_count"], 4)
        self.assertTrue(all(row["all_exact_binary"]
                            for row in result["per_channel_exactness"]))

    def test_first2_flow_suffix_binary_and_last2_hypothesis_rejected(self):
        tensor = torch.tensor([[[[[0.5, 1.5]], [[-0.5, 0.25]],
                                  [[0.0, 1.0]], [[1.0, 0.0]]]]])
        expected = {"module_index": 1, "name": "d1",
                    "input_shape": [1, 1, 4, 1, 2]}
        result = M649.audit_decoder_input(tensor, expected, 2)
        split = result["typed_partition"]
        self.assertTrue(split["first2_flow_hypothesis"]["gate_pass"])
        self.assertFalse(split["last2_flow_hypothesis"]["gate_pass"])
        self.assertEqual(split["first2_flow_hypothesis"][
            "binary_suffix_summary"]["exact_binary_count"], 4)
        self.assertEqual(split["last2_flow_hypothesis"][
            "binary_prefix_summary"]["nonbinary_finite_count"], 4)

    def test_nonbinary_suffix_is_no_go(self):
        tensor = torch.tensor([[[[[0.5]], [[-0.5]], [[0.25]], [[1.0]]]]])
        expected = {"module_index": 2, "name": "d2",
                    "input_shape": [1, 1, 4, 1, 1]}
        result = M649.audit_decoder_input(tensor, expected, 4)
        self.assertFalse(result["typed_partition"][
            "first2_flow_hypothesis"]["gate_pass"])

    def test_nonfinite_flow_is_counted_and_no_go(self):
        tensor = torch.tensor([[[[[float("nan")]], [[float("inf")]],
                                  [[0.0]], [[1.0]]]]])
        expected = {"module_index": 3, "name": "d3",
                    "input_shape": [1, 1, 4, 1, 1]}
        result = M649.audit_decoder_input(tensor, expected, 4)
        first = result["typed_partition"]["first2_flow_hypothesis"]
        self.assertFalse(first["gate_pass"])
        self.assertEqual(first["flow_summary"]["nonfinite_count"], 2)
        json.dumps(result, allow_nan=False)

    def test_full_decision_go_and_fail_closed_variants(self):
        records = [decision_record(sample, module)
                   for sample in range(10) for module in range(4)]
        decision = M649.typed_split_decision(records)
        self.assertTrue(decision["typed_split_authorized"])
        self.assertTrue(decision["first2_flow_hypothesis_all_modules_pass"])
        self.assertFalse(decision["last2_flow_hypothesis_all_modules_pass"])
        records[7] = decision_record(1, 3, pass_first2=False)
        self.assertFalse(M649.typed_split_decision(records)[
            "typed_split_authorized"])
        records[7] = decision_record(1, 3, dtype="torch.float16")
        self.assertFalse(M649.typed_split_decision(records)[
            "typed_split_authorized"])

    def test_source_order_is_frozen_first2_not_last2(self):
        model = (ROOT / "third_party/SDformerFlow/models/STSwinNet_SNN/"
                 "Spiking_STSwinNet.py").read_text(encoding="utf-8")
        util = (ROOT / "third_party/SDformerFlow/models/model_util.py").read_text(
            encoding="utf-8")
        self.assertIn("self.skip_ftn(predictions[-1], x,dim=2)", model)
        self.assertIn("torch.cat([x1, x2], dim=dim)", util)
        contract = M649.strict_json(CONTRACT)
        self.assertEqual(contract["numeric_audit"]["source_order_fact"][
            "therefore_expected_flow_channel_indices"], [0, 1])

    def test_double_seal_detects_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.json").write_text("{}\n", encoding="utf-8")
            M649.write_double_seal(root)
            M649.verify_own_double_seal(root)
            (root / "a.json").write_text("{\"x\":1}\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                M649.verify_own_double_seal(root)

    def test_symlink_output_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                M649.reject_symlink_chain(link)

    def test_dangling_canonical_output_is_rejected_before_resolve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            canonical.symlink_to(root / "missing", target_is_directory=True)
            with self.assertRaises(RuntimeError):
                M649.checked_path_match(
                    canonical, canonical, allow_missing_leaf=True,
                    label="output")

    def test_runtime_input_symlink_alias_is_rejected_before_equality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "input.json"
            expected.write_text("{}\n", encoding="utf-8")
            alias = root / "alias.json"
            alias.symlink_to(expected)
            with self.assertRaises(RuntimeError):
                M649.checked_path_match(alias, expected, label="contract")

    def test_parent_traversal_is_rejected_before_abspath_collapse(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "child" / ".." / "payload"
            with self.assertRaises(RuntimeError):
                M649.checked_path(raw, allow_missing_leaf=True,
                                  label="attack")

    def test_take_exact_never_fetches_item_eleven(self):
        class GuardedIterator(object):
            def __init__(self):
                self.calls = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.calls += 1
                if self.calls > 10:
                    raise AssertionError("item eleven must never be fetched")
                return self.calls

        guarded = GuardedIterator()
        self.assertEqual(list(M649.take_exact(guarded, 10)),
                         list(range(1, 11)))
        self.assertEqual(guarded.calls, 10)


if __name__ == "__main__":
    unittest.main()
