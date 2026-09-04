#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1188R2 strict M1184 semantic-gate tests; no remote/transfer/GPU/capture."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1188r2_m1182_m1180_capture_tar_transport_adapter_source.py"
SPEC = importlib.util.spec_from_file_location("m1188r2_transport", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1188R2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = M.load_contract()
        self.review = M.strict_json(ROOT / M.M1184_REL / "review.json")

    def reject(self, value, pattern: str) -> None:
        with self.assertRaisesRegex(M.R2Error, pattern):
            M.validate_m1184_review(value, self.contract)

    def test_01_exact51_requires_strict_m1184(self) -> None:
        rows = M.exact_members(self.contract)
        self.assertEqual((len(rows), sum(r["class"] == "ORIGINAL_EXACT42" for r in rows),
                          sum(r["class"] == "M1184_EXACT_SEAL" for r in rows)),
                         (51, 42, 9))
        admitted = M.strict_m1184_admission(self.contract)
        self.assertEqual(admitted["status"], "PASS")

    def test_02_schema_mutation_fails(self) -> None:
        value = copy.deepcopy(self.review); value["schema"] += "_mutated"
        self.reject(value, "schema")

    def test_03_status_mutations_fail(self) -> None:
        for status in ("PASS_EXACT_TRANSFER_AND_ONE_REMOTE_GPU_LAUNCH_AUTHORIZED__NO_AUTOMATIC_RETRY__FRESH_RESULT_HAMMER_REQUIRED",
                       "pass", True, 1, "PASS "):
            value = copy.deepcopy(self.review); value["status"] = status
            self.reject(value, "status")

    def test_04_verdict_mutations_fail(self) -> None:
        for verdict in ("PASS", self.review["verdict"].replace("NO_AUTOMATIC_RETRY", "RETRY"),
                        self.review["verdict"] + "_EXTRA"):
            value = copy.deepcopy(self.review); value["verdict"] = verdict
            self.reject(value, "verdict")

    def test_05_every_binding_mutation_fails(self) -> None:
        for key in sorted(self.review["bindings"]):
            value = copy.deepcopy(self.review)
            value["bindings"][key] = "0" * 64
            self.reject(value, "bindings")
        value = copy.deepcopy(self.review); value["bindings"]["extra"] = "0" * 64
        self.reject(value, "bindings")
        value = copy.deepcopy(self.review); del value["bindings"][next(iter(value["bindings"]))]
        self.reject(value, "bindings")

    def test_06_authorization_mutations_fail(self) -> None:
        mutations = [
            ("exact_remote_launch", False), ("automatic_retry", True),
            ("one_gpu_attempt", False), ("fresh_result_hammer_after_capture", False),
            ("remote_interpreter", "/usr/bin/python"),
            ("exact_transfer_list", "changed"),
        ]
        for key, changed in mutations:
            value = copy.deepcopy(self.review); value["authorization"][key] = changed
            self.reject(value, "authorization")
        value = copy.deepcopy(self.review); value["authorization"]["extra"] = True
        self.reject(value, "authorization")

    def test_07_duplicate_and_topology_mutations_fail(self) -> None:
        value = copy.deepcopy(self.review); value["extra"] = 1
        self.reject(value, "top-level")
        raw = (ROOT / M.M1184_REL / "review.json").read_text()
        mutated = raw.replace('"status": "PASS",', '"status": "PASS", "status": "PASS",', 1)
        with self.assertRaisesRegex(M.R2Error, "duplicate JSON key"):
            M.strict_json_bytes(mutated.encode())

    def test_08_seals_and_review_manifest_binding(self) -> None:
        expected = self.contract["m1184_exact_semantics"]
        manifest = ROOT / M.M1184_REL / "SHA256SUMS"
        outer = ROOT / M.M1184_REL / "SHA256SUMS.seal.sha256"
        self.assertEqual(M.sha256(manifest), expected["manifest_sha256"])
        self.assertEqual(M.sha256(outer), expected["outer_sha256"])
        rows = dict(M.R1.parse_sha_manifest(manifest))
        self.assertEqual(rows["review.json"], expected["review_sha256"])

    def test_09_fixed_transport_is_preserved(self) -> None:
        self.assertEqual(M.R1.fixed_ssh_argv()[0], "/usr/bin/ssh")
        scp = M.fixed_scp_argv(Path("/fixed/archive.tar"))
        self.assertEqual(scp[0], "/usr/bin/scp")
        self.assertEqual(scp[-1], M.R1.REMOTE_HOST + ":" + str(M.REMOTE_ARCHIVE))
        source = SOURCE.read_text()
        self.assertIn("shell=False", source)
        self.assertNotIn("shell=True", source)
        self.assertIn("strict_m1184_admission(contract)", source)

    def test_10_docs359_and_no_runtime_namespaces(self) -> None:
        self.assertEqual(M.sha256(ROOT / M.DOCS359_REL), M.DOCS359_SHA256)
        self.assertFalse(M.ATTEMPT.exists())
        self.assertFalse(M.RESULT.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
