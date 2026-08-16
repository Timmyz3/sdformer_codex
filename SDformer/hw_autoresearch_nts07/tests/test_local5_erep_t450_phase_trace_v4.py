from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim_qfit"))

import local5_erep_t450_phase_trace_v4 as phase


PHASE = (
    "EREP_PHASE_V4 schema=local5_erep_t450_phase_v4 group=0 "
    "first_relation_accept_cycle=1 last_relation_accept_cycle=450 "
    "execute_begin_cycle=452 execute_end_cycle=462 done_cycle=465 "
    "prepare=0 relation_fill=450 relation_commit=2 execute=10 compute_drain=3 "
    "total=465 active=2 terms=3 updates=4 "
    "term_stall=1 sram_reads=2 sram_writes=4"
)
GROUP = (
    "GROUP backend=0 new1rw=1 mode=0 latency=1 group=0 cycles=465 active=2 "
    "avoided=448 memory_wait=4 terms=3 updates=4 term_stall=1 sram_reads=2 "
    "sram_writes=4"
)
PASS = (
    "PASS post-G0 active projection backend=0 latency=1 groups=1 "
    "total_cycles=465 descriptors=2"
)
FINISH = "- tb.sv:1: Verilog $finish"


def parse_text(value: str) -> list[dict[str, int]]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.log"
        path.write_text(value, encoding="utf-8")
        return phase.parse_log(path)


class Local5ErepT450PhaseTraceV4Test(unittest.TestCase):
    def write_manifest(self, root: Path) -> Path:
        source_manifest = root / "source_manifest.json"
        source_manifest.write_text(
            json.dumps(
                {
                    "schema": "et3_ordered_term_trace_v2",
                    "qualification": {"qualified": True},
                }
            ),
            encoding="utf-8",
        )
        source_payload = root / "source_payload.npz"
        source_payload.write_bytes(b"fixture")
        contracts = {
            "input_valid": (450, 5),
            "input_active": (450, 5),
            "input_k": (450, 32),
            "input_gates": (450, 45),
            "input_weights": (64, 8),
            "expected_acc": (900, 32),
            "expected_active": (1, 16),
            "expected_terms": (1, 32),
            "expected_updates": (1, 32),
        }
        artifacts = {}
        for name, (entries, width) in contracts.items():
            artifact = root / f"{name}.memh"
            artifact.write_text("0\n" * entries, encoding="ascii")
            artifacts[name] = {
                "file": artifact.name,
                "entries": entries,
                "width": width,
                "sha256": phase.sha256(artifact),
            }
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "local5_active_projection_postg0_vectors_v1",
                    "shape": phase.SHAPE,
                    "source_manifest": str(source_manifest),
                    "source_manifest_sha256": phase.sha256(source_manifest),
                    "source_payload": str(source_payload),
                    "source_payload_sha256": phase.sha256(source_payload),
                    "selection": {
                        "method": "manifest_order_all_groups",
                        "groups": 1,
                        "rows": [
                            {
                                "vector_group_index": 0,
                                "active_sources": 2,
                                "terms": 3,
                                "updates": 4,
                            }
                        ],
                    },
                    "artifacts": artifacts,
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_exact_pair_and_terminal_are_accepted(self) -> None:
        rows = parse_text(f"{PHASE}\n{GROUP}\n{PASS}\n{FINISH}\n")
        self.assertEqual(rows[0]["execute"], 10)
        self.assertEqual(rows[0]["memory_wait"], 4)

    def test_exact_schema_and_order_fail_closed(self) -> None:
        variants = (
            f"noise\n{PHASE}\n{GROUP}\n{PASS}\n{FINISH}\n",
            f"{PHASE.replace(' term_stall=1', '')}\n{GROUP}\n{PASS}\n{FINISH}\n",
            f"{GROUP}\n{PHASE}\n{PASS}\n{FINISH}\n",
            f"{PHASE}\n{PHASE}\n{GROUP}\n{PASS}\n{FINISH}\n",
            f"{PHASE}\n{GROUP}\n{PASS}\n{PASS}\n{FINISH}\n",
        )
        for value in variants:
            with self.subTest(value=value[:60]):
                with self.assertRaises(ValueError):
                    parse_text(value)

    def test_phase_group_and_terminal_ledgers_fail_closed(self) -> None:
        variants = (
            f"{PHASE.replace('total=465', 'total=466')}\n{GROUP}\n{PASS}\n{FINISH}\n",
            f"{PHASE}\n{GROUP.replace('cycles=465', 'cycles=466')}\n{PASS}\n{FINISH}\n",
            f"{PHASE}\n{GROUP.replace('updates=4', 'updates=5')}\n{PASS}\n{FINISH}\n",
            f"{PHASE}\n{GROUP}\n{PASS.replace('descriptors=2', 'descriptors=3')}\n{FINISH}\n",
            f"{PHASE.replace('relation_fill=450', 'relation_fill=449')}\n{GROUP}\n{PASS}\n{FINISH}\n",
            f"{PHASE.replace('execute=10 compute_drain=3', 'execute=9 compute_drain=4')}\n{GROUP}\n{PASS}\n{FINISH}\n",
            f"{PHASE}\n{GROUP.replace('memory_wait=4', 'memory_wait=-1')}\n{PASS}\n{FINISH}\n",
        )
        for value in variants:
            with self.subTest(value=value[:80]):
                with self.assertRaises(ValueError):
                    parse_text(value)

    def test_finish_must_follow_terminal_and_be_unique(self) -> None:
        self.assertEqual(
            len(parse_text(f"{PHASE}\n{GROUP}\n{PASS}\n{FINISH}\n")),
            1,
        )
        for value in (
            f"{FINISH}\n{PHASE}\n{GROUP}\n{PASS}\n",
            f"{PHASE}\n{GROUP}\n{PASS}\n{FINISH}\n{FINISH}\n",
            f"{PHASE}\n{GROUP}\n{PASS}\n",
        ):
            with self.assertRaises(ValueError):
                parse_text(value)

    def test_manifest_artifacts_and_formal_boundary_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.log"
            trace_path.write_text(
                f"{PHASE}\n{GROUP}\n{PASS}\n{FINISH}\n", encoding="utf-8"
            )
            manifest = self.write_manifest(root)
            evidence = phase.build_evidence(trace_path, manifest)
            self.assertEqual(evidence["status"], "PASS_RTL_CALIBRATION_ONLY")
            self.assertEqual(evidence["formal_adapter_status"], "DENY")
            (root / "input_k.memh").write_text("1\n" * 450, encoding="ascii")
            with self.assertRaises(ValueError):
                phase.build_evidence(trace_path, manifest)


if __name__ == "__main__":
    unittest.main()
