from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import report_local5_active_projection_postg0_rtl as report


class Local5ProjectionReportBindingTest(unittest.TestCase):
    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _fixture(self, root: Path) -> Path:
        source_dir = root / "source"
        vector_dir = root / "vectors"
        source_dir.mkdir()
        vector_dir.mkdir()
        payload = source_dir / "ordered_term_items.npz"
        payload.write_bytes(b"payload")
        group = {
            "sample": 4,
            "stage": 2,
            "block": 3,
            "window": 7,
            "head": 5,
            "ordered_item_sha256": "ordered-item",
        }
        source_manifest = source_dir / "ordered_term_manifest.json"
        source_manifest.write_text(
            json.dumps(
                {
                    "schema": "et3_ordered_term_trace_v2",
                    "qualification": {"qualified": True, "processed_samples": 100},
                    "payload_sha256": self._sha(payload),
                    "groups": [group],
                }
            ),
            encoding="utf-8",
        )
        artifact = vector_dir / "input.memh"
        artifact.write_text("00\n", encoding="ascii")
        vector_manifest = vector_dir / "manifest.json"
        vector_manifest.write_text(
            json.dumps(
                {
                    "schema": "local5_active_projection_postg0_vectors_v1",
                    "source_manifest": str(source_manifest),
                    "source_manifest_sha256": self._sha(source_manifest),
                    "source_payload": str(payload),
                    "source_payload_sha256": self._sha(payload),
                    "selection": {
                        "rows": [{**group, "input_group_index": 0}],
                    },
                    "artifacts": {
                        "input": {
                            "file": artifact.name,
                            "sha256": self._sha(artifact),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return vector_dir

    def test_accepts_fully_bound_vector_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vector_dir = self._fixture(Path(directory))
            manifest, source, _, _ = report.load_bound_inputs(vector_dir)
            self.assertEqual(len(manifest["selection"]["rows"]), 1)
            self.assertEqual(source["qualification"]["processed_samples"], 100)

    def test_rejects_vector_artifact_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vector_dir = self._fixture(Path(directory))
            (vector_dir / "input.memh").write_text("ff\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "artifact input SHA"):
                report.load_bound_inputs(vector_dir)

    def test_rejects_selection_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vector_dir = self._fixture(Path(directory))
            path = vector_dir / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["selection"]["rows"][0]["head"] = 6
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "身份不一致"):
                report.load_bound_inputs(vector_dir)

    def test_rejects_source_receipt_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bound.sv"
            source.write_text("module bound; endmodule\n", encoding="ascii")
            receipt = root / "source_sha256.txt"
            receipt.write_text(f"{self._sha(source)}  {source}\n", encoding="utf-8")
            source.write_text("module changed; endmodule\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "source receipt SHA失配"):
                report.verify_source_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
