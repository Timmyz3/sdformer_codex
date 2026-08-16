from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hw_autoresearch_nts07.scripts.evidence_provenance import (
    LOCAL5_PROJECTION_SOURCE_SUFFIXES,
    validate_local5_projection_provenance,
)
from hw_autoresearch_nts07.scripts import run_local5_bb1e4_checkpoint_bound_rtl as runner


def binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


class Local5ProjectionProvenanceTest(unittest.TestCase):
    def test_shell_source_manifest_skips_separately_bound_vector_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vector_manifest = root / "vectors/manifest.json"
            vector_manifest.parent.mkdir(parents=True)
            vector_manifest.write_text("{}\n", encoding="utf-8")
            paths = []
            for index, suffix in enumerate(sorted(LOCAL5_PROJECTION_SOURCE_SUFFIXES)):
                path = root / suffix
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"module source_{index}; endmodule\n", encoding="utf-8")
                paths.append(path)
            source_manifest = root / "source_sha256.txt"
            source_manifest.write_text(
                "".join(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
                    for path in [*paths, vector_manifest]
                ),
                encoding="utf-8",
            )
            with patch.object(runner, "HW_ROOT", root):
                result = runner.projection_source_bindings(
                    source_manifest, vector_manifest
                )
            self.assertEqual(len(result), 20)

    def test_scope_binds_payloads_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vectors = root / "vectors"
            vectors.mkdir()
            source_trace_manifest = root / "ordered_manifest.json"
            source_trace_payload = root / "ordered_payload.npz"
            source_trace_manifest.write_text('{"trace": true}\n', encoding="utf-8")
            source_trace_payload.write_bytes(b"trace-payload")

            artifacts = {}
            for index in range(8):
                payload = vectors / f"payload_{index}.memh"
                payload.write_text(f"{index:08x}\n", encoding="utf-8")
                artifacts[f"payload_{index}"] = {
                    "file": payload.name,
                    "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                }
            vector_manifest = vectors / "manifest.json"
            vector_manifest.write_text(
                json.dumps(
                    {
                        "source_manifest": str(source_trace_manifest),
                        "source_manifest_sha256": hashlib.sha256(
                            source_trace_manifest.read_bytes()
                        ).hexdigest(),
                        "source_payload": str(source_trace_payload),
                        "source_payload_sha256": hashlib.sha256(
                            source_trace_payload.read_bytes()
                        ).hexdigest(),
                        "artifacts": artifacts,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            named = {}
            named_suffixes = {
                "projection_report": "projection_report.txt",
                "projection_source_manifest": "projection_source_manifest.txt",
                "runner": "scripts/run_local5_bb1e4_checkpoint_bound_rtl.py",
                "projection_shell": "sim_new_arch/run_local5_qgasr2c_fivebank_checks.sh",
                "projection_generator": "scripts/generate_local5_active_projection_postg0_vectors.py",
                "projection_summarizer": "scripts/summarize_local5_gasr2c_fivebank_rtl.py",
                "score_shell": "sim_local5/run_local5_checkpoint_score_trace_checks.sh",
                "score_generator": "scripts/generate_local5_checkpoint_score_vectors.py",
                "score_reporter": "scripts/report_local5_checkpoint_score_rtl.py",
                "provenance_validator": "scripts/evidence_provenance.py",
            }
            for name, suffix in named_suffixes.items():
                path = root / suffix
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name + "\n", encoding="utf-8")
                named[name] = binding(path)

            score_dir = root / "score"
            score_dir.mkdir()
            score_source_manifest = score_dir / "ordered_manifest.json"
            score_source_payload = score_dir / "ordered_payload.npz"
            score_vectors = score_dir / "local5_checkpoint_score_vectors.txt"
            score_source_manifest.write_text("{}\n", encoding="utf-8")
            score_source_payload.write_bytes(b"score-source")
            score_vectors.write_text("fixture\n", encoding="utf-8")
            score_manifest = score_dir / "manifest.json"
            score_manifest.write_text(
                json.dumps(
                    {
                        "source_manifest": str(score_source_manifest),
                        "source_manifest_sha256": hashlib.sha256(
                            score_source_manifest.read_bytes()
                        ).hexdigest(),
                        "source_payload": str(score_source_payload),
                        "source_payload_sha256": hashlib.sha256(
                            score_source_payload.read_bytes()
                        ).hexdigest(),
                        "vector_sha256": hashlib.sha256(score_vectors.read_bytes()).hexdigest(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            score_rtl = []
            for index in range(3):
                path = score_dir / f"score_{index}.sv"
                path.write_text(f"module score_{index}; endmodule\n", encoding="utf-8")
                score_rtl.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            score_logs = {}
            for name in ("iverilog", "verilator", "yosys"):
                path = score_dir / f"{name}.log"
                path.write_text("PASS\n", encoding="utf-8")
                score_logs[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            score_report = score_dir / "report.json"
            score_report.write_text(
                json.dumps(
                    {
                        "schema": "local5_checkpoint_score_rtl_report_v1",
                        "status": "PASS",
                        "checks": {"fixture": True},
                        "vector_manifest": str(score_manifest),
                        "vector_manifest_sha256": hashlib.sha256(score_manifest.read_bytes()).hexdigest(),
                        "rtl_bindings": score_rtl,
                        "logs": score_logs,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            named["score_report"] = binding(score_report)

            atlif_dir = root / "atlif"
            atlif_dir.mkdir()
            atlif_vector_sources = []
            for index in range(9):
                path = atlif_dir / f"vector_source_{index}.mem"
                path.write_text(f"{index}\n", encoding="utf-8")
                atlif_vector_sources.append(path)
            atlif_manifest = atlif_dir / "manifest.json"
            atlif_manifest.write_text(
                json.dumps(
                    {
                        "source_sha256": {
                            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in atlif_vector_sources
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            legacy_atlif_source_suffixes = (
                "tb_hitflow/tb_checkpoint_atlif_dptme.sv",
                "tb_hitflow/tb_hitflow_dptme_array.sv",
                "sim_hitflow/run_checkpoint_atlif_dptme_checks.sh",
                "verif_hitflow/hitflow_dptme_assertions.sv",
                "verif_hitflow/bind_hitflow_dptme_assertions.sv",
                "scripts/generate_checkpoint_atlif_dptme_vectors.py",
                "scripts/report_checkpoint_atlif_dptme_rtl.py",
            )
            atlif_report_sources = []
            for index, suffix in enumerate(legacy_atlif_source_suffixes):
                path = root / suffix
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"module atlif_{index}; endmodule\n", encoding="utf-8")
                atlif_report_sources.append(path)
            atlif_report = atlif_dir / "report.json"
            atlif_report.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "evidence_scope": "checkpoint_bound_atlif_component_rtl_exact_not_full_network",
                        "source_sha256": {
                            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [*atlif_report_sources, atlif_manifest]
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            named["atlif_report"] = binding(atlif_report)
            sources = []
            for index, suffix in enumerate(sorted(LOCAL5_PROJECTION_SOURCE_SUFFIXES)):
                path = root / suffix
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"module source_{index}; endmodule\n", encoding="utf-8")
                sources.append(binding(path))

            scope = {
                "schema": "local5_checkpoint_bound_component_rtl_exact_v2",
                "source_artifacts": {
                    **named,
                    "projection_vector_manifest": binding(vector_manifest),
                    "projection_source_trace_manifest": binding(source_trace_manifest),
                    "projection_source_trace_payload": binding(source_trace_payload),
                    "projection_payloads": [
                        binding(vectors / row["file"]) for row in artifacts.values()
                    ],
                    "projection_sources": sources,
                },
            }
            validate_local5_projection_provenance(scope)
            (vectors / "payload_0.memh").write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA drift|size drift"):
                validate_local5_projection_provenance(scope)

            (vectors / "payload_0.memh").write_text("00000000\n", encoding="utf-8")
            scope["source_artifacts"]["projection_payloads"][0] = binding(
                vectors / "payload_0.memh"
            )
            score_source_payload.write_bytes(b"stale-score-source")
            with self.assertRaisesRegex(RuntimeError, "score source_payload source SHA drift"):
                validate_local5_projection_provenance(scope)
            score_source_payload.write_bytes(b"score-source")

            atlif_vector_sources[0].write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "ATLIF vector source source SHA drift"):
                validate_local5_projection_provenance(scope)
            atlif_vector_sources[0].write_text("0\n", encoding="utf-8")

            wrong_runner = root / "wrong_runner.py"
            wrong_runner.write_text("runner\n", encoding="utf-8")
            scope["source_artifacts"]["runner"] = binding(wrong_runner)
            with self.assertRaisesRegex(RuntimeError, "runner path mismatch"):
                validate_local5_projection_provenance(scope)
            scope["source_artifacts"]["runner"] = named["runner"]

            scope["source_artifacts"]["projection_sources"][-1] = sources[0]
            with self.assertRaisesRegex(RuntimeError, "incomplete or duplicated"):
                validate_local5_projection_provenance(scope)


if __name__ == "__main__":
    unittest.main()
