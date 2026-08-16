#!/usr/bin/env python3
"""Unit tests for the H67 multisample vector generator."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from scripts import generate_h67_multisample_checkpoint_row_vectors as gen
except ModuleNotFoundError:
    import generate_h67_multisample_checkpoint_row_vectors as gen


def fake_manifest(sample_count: int = 2) -> dict[str, Any]:
    records = []
    for sample_id in range(sample_count):
        for name in gen.expected_names():
            records.append(
                {
                    "sample_id": sample_id,
                    "sample_key": f"sample-{sample_id}",
                    "name": name,
                    "file": f"sample{sample_id}_{name}.npz",
                    "sha256": f"sha-{sample_id}-{name}",
                    "windows_captured": 1,
                }
            )
    return {
        "sample_limit": sample_count,
        "windows_per_call": 1,
        "first_block_only": False,
        "run_context": {"test": True},
        "records": records,
    }


def fake_parser(record: dict[str, Any], tokens: int):
    match = gen.expected_names().index(record["name"])
    stage = next(
        stage for stage in gen.EXPECTED_BLOCKS
        if record["name"].startswith(f"S{stage}.")
    )
    block = int(record["name"].split(".")[1][1:])
    rows = []
    for head in range(gen.EXPECTED_HEADS[stage]):
        rows.append(
            {
                "stage": stage,
                "block": block,
                "head": head,
                "expected_outputs": 1,
                "expected_folded": tokens - 1,
                "vectors": [
                    {"q": match, "current_k": index == 0,
                     "peer_k": 0, "gate": 128}
                    for index in range(tokens)
                ],
            }
        )
    return (
        {
            "name": record["name"],
            "source": record["file"],
            "source_sha256": record["sha256"],
            "stage": stage,
            "block": block,
            "heads": len(rows),
            "rows": len(rows),
        },
        rows,
    )


class GeneratorTests(unittest.TestCase):
    def test_positive_two_sample_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(json.dumps(fake_manifest()), encoding="utf-8")
            result = gen.generate_vectors(
                source,
                root / "vectors",
                expected_tokens=2,
                context_validator=lambda manifest, tokens: manifest["run_context"],
                record_parser=fake_parser,
            )
            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(result["row_count"], 276)
            vector = Path(result["artifacts"]["vector_file"])
            self.assertEqual(vector.read_text(encoding="ascii").splitlines()[0], "276 2")
            index_rows = Path(result["artifacts"]["row_index"]).read_text(
                encoding="ascii"
            ).splitlines()
            self.assertEqual(len(index_rows), 276)
            first = json.loads(index_rows[0])
            last = json.loads(index_rows[-1])
            self.assertEqual((first["row_tag"], first["sample_id"]), (0, 0))
            self.assertEqual((last["row_tag"], last["sample_id"]), (275, 1))

    def test_rejects_missing_or_reordered_record(self) -> None:
        manifest = fake_manifest()
        manifest["records"].pop(3)
        with self.assertRaises(ValueError):
            gen.validate_record_sequence(manifest)
        manifest = fake_manifest()
        manifest["records"][0], manifest["records"][1] = (
            manifest["records"][1], manifest["records"][0]
        )
        with self.assertRaises(ValueError):
            gen.validate_record_sequence(manifest)

    def test_rejects_sample_identity_and_window_violations(self) -> None:
        cases = []
        manifest = fake_manifest()
        manifest["records"][12]["sample_id"] = 0
        cases.append(manifest)
        manifest = fake_manifest()
        manifest["records"][12]["sample_key"] = "sample-0"
        for record in manifest["records"][13:24]:
            record["sample_key"] = "sample-0"
        cases.append(manifest)
        manifest = fake_manifest()
        manifest["records"][5]["sample_key"] = "changed"
        cases.append(manifest)
        manifest = fake_manifest()
        manifest["records"][7]["windows_captured"] = 2
        cases.append(manifest)
        for case in cases:
            with self.subTest(case=cases.index(case)):
                with self.assertRaises(ValueError):
                    gen.validate_record_sequence(deepcopy(case))


if __name__ == "__main__":
    unittest.main()
