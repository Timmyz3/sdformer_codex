#!/usr/bin/env python3
"""Small CPU-only M1514 tests; no real checkpoint or weight export."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch


SOURCE = (Path(__file__).resolve().parent.parent / "scripts" /
          "build_m1514_ep34_decoder_weight_identity_export_source.py")
SPEC = importlib.util.spec_from_file_location("test_m1514_source", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


TINY_SHAPES = ((2, 2), (2, 3), (3, 2), (1, 5))


def tiny_state() -> OrderedDict:
    state = OrderedDict()
    for index in range(M.EXPECTED_STATE_KEYS - 4):
        state["filler.%04d" % index] = torch.tensor([index], dtype=torch.int64)
    for ordinal, (key, shape) in enumerate(zip(M.WEIGHT_KEYS, TINY_SHAPES)):
        elements = 1
        for extent in shape:
            elements *= extent
        state[key] = (torch.arange(elements, dtype=torch.float32)
                      .reshape(shape).contiguous() + ordinal)
    return state


def content_hashes(state: OrderedDict) -> tuple[str, ...]:
    return tuple(hashlib.sha256(
        state[key].numpy().tobytes(order="C")).hexdigest()
                 for key in M.WEIGHT_KEYS)


@contextmanager
def tiny_contract(state: OrderedDict):
    total = sum(tensor.numel() for key, tensor in state.items()
                if key in M.WEIGHT_KEYS)
    with mock.patch.object(M, "WEIGHT_SHAPES", TINY_SHAPES), \
            mock.patch.object(M, "EXPECTED_CONTENT_SHA256", content_hashes(state)), \
            mock.patch.object(M, "EXPECTED_TOTAL_ELEMENTS", total), \
            mock.patch.object(M, "EXPECTED_TOTAL_BYTES", total * 4):
        yield


class CheckpointObjectTests(unittest.TestCase):
    def test_01_exact_model_state_dict_and_weights_pass(self):
        state = tiny_state()
        with tiny_contract(state):
            rows = M.validate_checkpoint_object({"model_state_dict": state})
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(row["content_bytes"] for row in rows),
                         sum(t.numel() * 4 for k, t in state.items()
                             if k in M.WEIGHT_KEYS))
        self.assertTrue(all(row["bias"] is None for row in rows))

    def test_02_extra_or_wrong_checkpoint_root_rejected(self):
        state = tiny_state()
        for value in (
                {"model_state_dict": state, "optimizer_state_dict": {}},
                {"state_dict": state}, state):
            with self.subTest(keys=list(value)[:2]), self.assertRaisesRegex(
                    M.M1514Error, "model_state_dict-only"):
                M.validate_checkpoint_object(value)

    def test_03_missing_or_renamed_target_key_rejected(self):
        reference = tiny_state()
        state = tiny_state()
        tensor = state.pop(M.WEIGHT_KEYS[2])
        state["renamed.decoder.weight"] = tensor
        with tiny_contract(reference), self.assertRaisesRegex(
                M.M1514Error, "missing or duplicate alias"):
            M.validate_checkpoint_object({"model_state_dict": state})

    def test_04_shape_and_dtype_rejected(self):
        for attack in ("shape", "dtype"):
            state = tiny_state()
            if attack == "shape":
                state[M.WEIGHT_KEYS[0]] = torch.zeros((1, 4), dtype=torch.float32)
            else:
                state[M.WEIGHT_KEYS[0]] = torch.zeros(TINY_SHAPES[0], dtype=torch.float64)
            with self.subTest(attack=attack), tiny_contract(tiny_state()), \
                    self.assertRaisesRegex(M.M1514Error, attack):
                M.validate_checkpoint_object({"model_state_dict": state})

    def test_05_any_decoder_bias_rejected(self):
        state = tiny_state()
        state.pop("filler.0000")
        state[M.BIAS_KEYS[1]] = torch.zeros(1, dtype=torch.float32)
        with tiny_contract(tiny_state()), self.assertRaisesRegex(
                M.M1514Error, "bias"):
            M.validate_checkpoint_object({"model_state_dict": state})

    def test_06_content_drift_rejected(self):
        reference = tiny_state()
        attacked = tiny_state()
        attacked[M.WEIGHT_KEYS[3]] = attacked[M.WEIGHT_KEYS[3]].clone()
        attacked[M.WEIGHT_KEYS[3]][0, 0] += 1
        with tiny_contract(reference), self.assertRaisesRegex(
                M.M1514Error, "content SHA"):
            M.validate_checkpoint_object({"model_state_dict": attacked})

    def test_07_duplicate_key_alias_and_storage_alias_rejected(self):
        state = tiny_state()
        state.pop("filler.0000")
        state["module." + M.WEIGHT_KEYS[0]] = state[M.WEIGHT_KEYS[0]]
        with tiny_contract(tiny_state()), self.assertRaisesRegex(
                M.M1514Error, "duplicate alias"):
            M.validate_checkpoint_object({"model_state_dict": state})

        state = tiny_state()
        # Make shapes equal only for this storage-alias attack.
        shapes = (TINY_SHAPES[0],) * 4
        for ordinal, key in enumerate(M.WEIGHT_KEYS):
            state[key] = torch.full(shapes[ordinal], float(ordinal))
        state[M.WEIGHT_KEYS[1]] = state[M.WEIGHT_KEYS[0]]
        hashes = content_hashes(state)
        total = sum(state[key].numel() for key in M.WEIGHT_KEYS)
        with mock.patch.object(M, "WEIGHT_SHAPES", shapes), \
                mock.patch.object(M, "EXPECTED_CONTENT_SHA256", hashes), \
                mock.patch.object(M, "EXPECTED_TOTAL_ELEMENTS", total), \
                mock.patch.object(M, "EXPECTED_TOTAL_BYTES", total * 4), \
                self.assertRaisesRegex(M.M1514Error, "storage duplicate alias"):
            M.validate_checkpoint_object({"model_state_dict": state})

    def test_08_checkpoint_sha_and_cpu_load_contract(self):
        state = tiny_state()
        with tempfile.TemporaryDirectory(prefix="m1514_") as temporary:
            checkpoint = Path(temporary) / "checkpoint.pth"
            checkpoint.write_bytes(b"synthetic frozen checkpoint identity")
            checkpoint_sha = M.sha256(checkpoint)
            with tiny_contract(state), \
                    mock.patch.object(torch, "load",
                                      return_value={"model_state_dict": state}) as load, \
                    mock.patch.object(M, "verify_capture_authorities",
                                      return_value={"m1512_status": "PASS",
                                                    "m1513_status": "PASS"}):
                result = M.audit_checkpoint(checkpoint, checkpoint_sha)
            self.assertEqual(result["checkpoint"]["sha256"], checkpoint_sha)
            self.assertEqual(load.call_args.kwargs["map_location"], torch.device("cpu"))
            self.assertFalse(result["future_export"]["production"])
            with self.assertRaisesRegex(M.M1514Error, "SHA drift"):
                M.audit_checkpoint(checkpoint, "0" * 64)


class AuthorityAndPolicyTests(unittest.TestCase):
    def test_09_exact_m1510_m1512_m1513_authority_chain(self):
        self.assertEqual(M.sha256(M.M1510_SOURCE), M.M1510_SOURCE_SHA256)
        self.assertEqual(M.sha256(M.M1510_CONTRACT), M.M1510_CONTRACT_SHA256)
        authority = M.verify_capture_authorities()
        self.assertIn("M1512", authority["m1512_status"])
        self.assertIn("M1513", authority["m1513_status"])

    def test_10_no_export_gpu_remote_or_eda_action(self):
        text = SOURCE.read_text(encoding="utf-8")
        for token in ("subprocess", "paramiko", "torch.cuda", "os.kill",
                      "ssh ", "vcs", "dc_shell", "pt_shell", ".to(\"cuda\")"):
            self.assertNotIn(token, text)
        self.assertNotIn('add_argument("--export"', text)
        self.assertFalse(M.FUTURE_EXPORT["production"])
        self.assertFalse(M.CLAIM_BOUNDARY["weight_payload_written"])
        self.assertFalse(M.CLAIM_BOUNDARY["cycles"])
        self.assertFalse(M.CLAIM_BOUNDARY["speedup"])


if __name__ == "__main__":
    unittest.main()
