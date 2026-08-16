from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HW_ROOT = Path(__file__).resolve().parents[1]
RUNNER = HW_ROOT / "sim_hitflow/run_gatestack_dctf96_real_trace_checks.sh"
SCRIPTS = HW_ROOT / "scripts"
SOURCE_SET = [
    RUNNER,
    HW_ROOT / "tb_hitflow/tb_gatestack_dctf96_banklocal_projection_real_trace.sv",
    HW_ROOT / "verif_hitflow/gatestack_dctf96_banklocal_projection_top_assertions.sv",
    HW_ROOT / "verif_hitflow/bind_gatestack_dctf96_banklocal_projection_top_assertions.sv",
    HW_ROOT / "rtl_hitflow/gatestack_decoupled_product_engine.sv",
    HW_ROOT / "rtl_hitflow/gatestack_dctf32_bank_executor.sv",
    HW_ROOT / "rtl_hitflow/gatestack_dctf_term_event_adapter.sv",
    HW_ROOT / "rtl_hitflow/gatestack_dctf_term_fabric.sv",
    HW_ROOT / "rtl_hitflow/gatestack_dctf96_term_datapath_top.sv",
    HW_ROOT / "rtl_hitflow/hitflow_banked_accumulator.sv",
    HW_ROOT / "rtl_hitflow/gatestack_dctf96_banklocal_projection_top.sv",
]
sys.path.insert(0, str(SCRIPTS))

from evidence_provenance import EXPECTED_ALL12_NAMES, validate_projection_provenance


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_set_sha256() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_SET:
        resolved = path.resolve()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(resolved).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


class ProjectionReportProvenanceTest(unittest.TestCase):
    def _report_block(self) -> str:
        source = RUNNER.read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
        self.assertGreaterEqual(len(blocks), 2)
        return blocks[-1]

    def test_report_block_executes_and_binds_all_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "result"
            result.mkdir()
            (result / "logs").mkdir()
            (result / "tool_versions.txt").write_text(
                "iverilog fixture\nverilator fixture\npython fixture\n",
                encoding="utf-8",
            )
            (result / "logs/generator_unittest.log").write_text(
                "OK\n", encoding="utf-8"
            )
            (result / "logs/vector_generation.log").write_text(
                "PASS\n", encoding="utf-8"
            )
            source_manifest = result / "source_manifest.json"
            source_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
            source_sha = source_set_sha256()

            records = []
            for name in sorted(EXPECTED_ALL12_NAMES):
                stage = int(name[1])
                block = int(name.split(".B", 1)[1].split(".", 1)[0])
                heads = (3, 6, 12, 24)[stage]
                vector_dir = result / "vectors" / f"s{stage}_b{block}"
                vector_dir.mkdir(parents=True)
                payload = vector_dir / "metadata.memh"
                payload.write_text("00000001\n", encoding="utf-8")
                record_manifest = vector_dir / "manifest.json"
                record_manifest.write_text(
                    json.dumps({"name": name, "files": [payload.name]}) + "\n",
                    encoding="utf-8",
                )
                record = {
                    "name": name,
                    "stage": stage,
                    "heads": heads,
                    "tokens": 450,
                    "token_id_width": 9,
                    "vector_dir": str(vector_dir),
                    "expected_issued_terms": 1,
                    "expected_physical_weight_requests": 2,
                    "expected_bias_requests": 3,
                    "expected_final_checks": 4,
                    "files": {
                        payload.name: {
                            "bytes": payload.stat().st_size,
                            "sha256": sha256(payload),
                        }
                    },
                }
                records.append(record)
                vector_sha = hashlib.sha256(
                    json.dumps(
                        record["files"], sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
                receipt_common = (
                    f"vector_name={name} vector_id={vector_dir.name} "
                    f"vector_aggregate_sha256={vector_sha} "
                    f"source_set_sha256={source_sha} stage={stage} "
                    f"heads={heads} tokens=450\n"
                )
                pass_line = (
                    f"PASS DCTF96 REAL TRACE stage=S{stage} heads={heads} cycles=5 "
                    "terms=1 physical_weight_req=2 bias_req=3 final_checks=4\n"
                )
                record_id = vector_dir.name
                (result / "logs" / f"icarus_{record_id}.log").write_text(
                    "RUN_RECEIPT simulator=icarus assertions=none "
                    + receipt_common
                    + pass_line,
                    encoding="utf-8",
                )
                (result / "logs" / f"verilator_{record_id}.log").write_text(
                    "RUN_RECEIPT simulator=verilator assertions=enabled "
                    + receipt_common
                    + pass_line,
                    encoding="utf-8",
                )

            vector_manifest = result / "vectors_manifest.json"
            vector_manifest.write_text(
                json.dumps(
                    {
                        "source_manifest": str(source_manifest),
                        "source_manifest_sha256": sha256(source_manifest),
                        "source_run_context": {
                            "artifact_identity": {
                                "checkpoint_path": "/fixture/checkpoint.pth",
                                "checkpoint_sha256": "checkpoint-sha",
                                "config_path": "/fixture/config.yml",
                                "config_sha256": "config-sha",
                            }
                        },
                        "temporal_tokens": 450,
                        "token_id_width": 9,
                        "records": records,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    self._report_block(),
                    str(result),
                    str(HW_ROOT),
                    str(RUNNER),
                ],
                cwd=HW_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((result / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["schema"], "h67_checkpoint_projection_rtl_exact_v4")
            validate_projection_provenance(report)

            self.assertEqual(len(report["source_artifacts"]["simulation_logs"]), 12)
            self.assertEqual(len(report["run_receipts"]), 24)
            log_path = Path(
                report["source_artifacts"]["simulation_logs"][0]["icarus"]["path"]
            )
            original_log = log_path.read_text(encoding="utf-8")
            log_path.write_text("stale log\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA drift|size drift"):
                validate_projection_provenance(report)
            log_path.write_text(original_log, encoding="utf-8")

            Path(report["vector_payloads"][0]["files"][0]["path"]).write_text(
                "stale\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "SHA drift|size drift"):
                validate_projection_provenance(report)


if __name__ == "__main__":
    unittest.main()
