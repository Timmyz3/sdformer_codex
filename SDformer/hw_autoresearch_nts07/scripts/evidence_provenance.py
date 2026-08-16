"""Fail-closed provenance checks shared by H67 hardware evidence watchers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ALL12_NAMES = {
    *(f"S0.B{block}.attn" for block in range(2)),
    *(f"S1.B{block}.attn" for block in range(2)),
    *(f"S2.B{block}.attn" for block in range(6)),
    *(f"S3.B{block}.attn" for block in range(2)),
}

LOCAL5_PROJECTION_SOURCE_SUFFIXES = {
    "tb_qfit/tb_qfit_local5_active_projection_postg0.sv",
    "rtl_qfit/qfit_local5_1rw_active_projection_tile.sv",
    "rtl_qfit/qfit_dual_color_relation_frontier_sync.sv",
    "rtl_qfit/qfit_dual_color_word_skipper_index.sv",
    "rtl_qfit/qfit_sync_relation_bank.sv",
    "rtl_qfit/qfit_fakeram45_relation_bank_450.sv",
    "rtl_qfit/qfit_source_multicast_term_builder_fifo2.sv",
    "rtl_qfit/qfit_source_multicast_term_builder.sv",
    "rtl_qfit/qfit_local5_1rw_projection_backend.sv",
    "rtl_qfit/qfit_local5_color_map.sv",
    "rtl_qfit/qfit_direct_1rw_acc_bank.sv",
    "rtl_qfit/qfit_gasr2c_acc_bank.sv",
    "rtl_qfit/qfit_single_port_acc_memory.sv",
    "verif_qfit/qfit_local5_1rw_active_projection_assertions.sv",
    "verif_qfit/qfit_gasr2c_acc_bank_assertions.sv",
    "verif_qfit/qfit_direct_1rw_acc_bank_assertions.sv",
    "verif_qfit/qfit_single_port_acc_memory_assertions.sv",
    "verif_qfit/qfit_dual_color_word_skipper_assertions.sv",
    "verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv",
    "verif_qfit/qfit_source_multicast_assertions.sv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_set_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(resolved).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_binding(binding: dict[str, Any], label: str) -> Path:
    path = Path(str(binding.get("path", "")))
    if not path.is_file():
        raise RuntimeError(f"{label} source missing: {path}")
    if binding.get("sha256") != sha256_file(path):
        raise RuntimeError(f"{label} source SHA drift: {path}")
    if int(binding.get("bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"{label} source size drift: {path}")
    return path


def _validate_path_sha(path_value: Any, sha_value: Any, label: str) -> Path:
    path = Path(str(path_value))
    if not path.is_file():
        raise RuntimeError(f"{label} source missing: {path}")
    if sha_value != sha256_file(path):
        raise RuntimeError(f"{label} source SHA drift: {path}")
    return path


def _require_suffix(path: Path, suffix: str, label: str) -> None:
    if not path.as_posix().endswith("/" + suffix):
        raise RuntimeError(f"{label} path mismatch: {path}")


def validate_local5_score_provenance(report: dict[str, Any]) -> None:
    if report.get("schema") != "local5_checkpoint_score_rtl_report_v1":
        raise RuntimeError("unexpected Local5 score RTL report schema")
    if report.get("status") != "PASS" or not all((report.get("checks") or {}).values()):
        raise RuntimeError("Local5 score RTL report did not pass")
    manifest_path = _validate_path_sha(
        report.get("vector_manifest"),
        report.get("vector_manifest_sha256"),
        "Local5 score vector manifest",
    )
    for index, row in enumerate(report.get("rtl_bindings") or []):
        _validate_path_sha(row.get("path"), row.get("sha256"), f"Local5 score RTL[{index}]")
    if len(report.get("rtl_bindings") or []) != 3:
        raise RuntimeError("Local5 score report must bind two RTL files and one testbench")
    logs = report.get("logs") or {}
    if set(logs) != {"iverilog", "verilator", "yosys"}:
        raise RuntimeError("Local5 score report log set mismatch")
    for name, row in logs.items():
        _validate_path_sha(row.get("path"), row.get("sha256"), f"Local5 score {name} log")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("source_manifest", "source_payload"):
        path = _validate_path_sha(
            manifest.get(key), manifest.get(f"{key}_sha256"), f"Local5 score {key}"
        )
        if not path.is_file():
            raise RuntimeError(f"Local5 score {key} missing")
    vector_path = manifest_path.parent / "local5_checkpoint_score_vectors.txt"
    _validate_path_sha(vector_path, manifest.get("vector_sha256"), "Local5 score vectors")


def validate_local5_atlif_provenance(report: dict[str, Any]) -> None:
    if report.get("status") != "PASS" or "component_rtl_exact" not in str(
        report.get("evidence_scope", "")
    ):
        raise RuntimeError("Local5 ATLIF RTL report did not pass")
    sources = report.get("source_sha256") or {}
    # Current reporter binds 9 paths (8 RTL/sim/script files + vector manifest).
    # Older reports used 8; accept both so rescope does not fail-closed on growth.
    if len(sources) not in (8, 9):
        raise RuntimeError(
            "Local5 ATLIF report must bind eight or nine source artifacts "
            f"(got {len(sources)})"
        )
    manifest_path = None
    for path_value, digest in sources.items():
        path = _validate_path_sha(path_value, digest, "Local5 ATLIF report source")
        if path.name == "manifest.json":
            manifest_path = path
    if manifest_path is None:
        raise RuntimeError("Local5 ATLIF report does not bind its vector manifest")
    # Required producer/runtime bindings for checkpoint-bound ATLIF.
    required_suffixes = (
        "tb_hitflow/tb_checkpoint_atlif_dptme.sv",
        "sim_hitflow/run_checkpoint_atlif_dptme_checks.sh",
        "scripts/generate_checkpoint_atlif_dptme_vectors.py",
        "scripts/report_checkpoint_atlif_dptme_rtl.py",
    )
    joined = "\n".join(str(Path(p).as_posix()) for p in sources)
    for suffix in required_suffixes:
        if suffix not in joined:
            raise RuntimeError(f"Local5 ATLIF report missing required source: {suffix}")
    if len(sources) == 9 and "rtl_hitflow/hitflow_dptme_array.sv" not in joined:
        raise RuntimeError(
            "Local5 ATLIF nine-source report missing required source: "
            "rtl_hitflow/hitflow_dptme_array.sv"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vector_sources = manifest.get("source_sha256") or {}
    if len(vector_sources) != 9:
        raise RuntimeError("Local5 ATLIF vector manifest must bind generator and eight payloads")
    for path_value, digest in vector_sources.items():
        _validate_path_sha(path_value, digest, "Local5 ATLIF vector source")


def validate_projection_provenance(report: dict[str, Any]) -> None:
    schema = report.get("schema")
    if schema not in {
        "h67_checkpoint_projection_rtl_exact_v2",
        "h67_checkpoint_projection_rtl_exact_v3",
        "h67_checkpoint_projection_rtl_exact_v4",
    }:
        raise RuntimeError("H67 projection provenance requires report schema v2/v3/v4")
    artifacts = report.get("source_artifacts") or {}
    for key in (
        "source_trace_manifest",
        "vector_manifest",
        "vector_generator",
        "vector_generator_tests",
        "runner",
        "testbench",
        "sva",
        "bind",
    ):
        if not isinstance(artifacts.get(key), dict):
            raise RuntimeError(f"H67 projection provenance missing {key}")
        _validate_binding(artifacts[key], key)
    rtl = artifacts.get("rtl") or []
    if len(rtl) != 7:
        raise RuntimeError("H67 projection provenance must bind seven RTL sources")
    for index, binding in enumerate(rtl):
        _validate_binding(binding, f"rtl[{index}]")

    if schema in {
        "h67_checkpoint_projection_rtl_exact_v3",
        "h67_checkpoint_projection_rtl_exact_v4",
    }:
        for key in (
            "tool_versions",
            "generator_unittest_log",
            "vector_generation_log",
        ):
            if not isinstance(artifacts.get(key), dict):
                raise RuntimeError(f"H67 projection v3 provenance missing {key}")
            _validate_binding(artifacts[key], key)
        simulation_logs = artifacts.get("simulation_logs") or []
        if len(simulation_logs) != 12:
            raise RuntimeError("H67 projection v3 must bind 12 dual-simulator logs")
        log_names = {str(row.get("name", "")) for row in simulation_logs}
        if log_names != EXPECTED_ALL12_NAMES:
            raise RuntimeError("H67 projection v3 simulation-log coverage is not exact all12")
        for row in simulation_logs:
            for simulator in ("icarus", "verilator"):
                _validate_binding(
                    row.get(simulator) or {},
                    f"{row.get('name')} {simulator} log",
                )
        if schema == "h67_checkpoint_projection_rtl_exact_v4":
            log_paths = {
                str((row.get(simulator) or {}).get("path", ""))
                for row in simulation_logs
                for simulator in ("icarus", "verilator")
            }
            if len(log_paths) != 24:
                raise RuntimeError("H67 projection v4 simulation logs must be 24 unique files")

    source_manifest = _validate_binding(
        artifacts["source_trace_manifest"], "source_trace_manifest"
    )
    vector_manifest_path = _validate_binding(
        artifacts["vector_manifest"], "vector_manifest"
    )
    if Path(str(report.get("source_manifest", ""))).resolve() != source_manifest.resolve():
        raise RuntimeError("H67 projection report/source manifest path mismatch")
    if report.get("source_manifest_sha256") != sha256_file(source_manifest):
        raise RuntimeError("H67 projection report/source manifest SHA mismatch")
    if Path(str(report.get("vector_manifest", ""))).resolve() != vector_manifest_path.resolve():
        raise RuntimeError("H67 projection report/vector manifest path mismatch")
    if report.get("vector_manifest_sha256") != sha256_file(vector_manifest_path):
        raise RuntimeError("H67 projection report/vector manifest SHA mismatch")

    vector_manifest = json.loads(vector_manifest_path.read_text(encoding="utf-8"))
    records = vector_manifest.get("records") or []
    names = [str(record.get("name", "")) for record in records]
    if len(names) != 12 or set(names) != EXPECTED_ALL12_NAMES:
        raise RuntimeError("H67 projection vector manifest is not exact all12")
    report_names = [str(record.get("name", "")) for record in report.get("records") or []]
    if len(report_names) != 12 or set(report_names) != EXPECTED_ALL12_NAMES:
        raise RuntimeError("H67 projection RTL report is not exact all12")

    payloads = report.get("vector_payloads") or []
    if len(payloads) != 12 or {str(row.get("name", "")) for row in payloads} != EXPECTED_ALL12_NAMES:
        raise RuntimeError("H67 projection payload provenance is not exact all12")
    payload_by_name = {str(row["name"]): row for row in payloads}
    for record in records:
        name = str(record["name"])
        payload = payload_by_name[name]
        vector_dir = Path(str(record["vector_dir"]))
        record_manifest = _validate_binding(
            payload.get("record_manifest") or {}, f"{name} record manifest"
        )
        if record_manifest.resolve() != (vector_dir / "manifest.json").resolve():
            raise RuntimeError(f"{name} record manifest path mismatch")
        expected_files = record.get("files") or {}
        file_bindings = payload.get("files") or []
        if {Path(str(row.get("path", ""))).name for row in file_bindings} != set(
            expected_files
        ):
            raise RuntimeError(f"{name} vector payload set mismatch")
        for binding in file_bindings:
            path = _validate_binding(binding, f"{name} vector payload")
            expected = expected_files[path.name]
            if (
                binding.get("sha256") != expected.get("sha256")
                or int(binding.get("bytes", -1)) != int(expected.get("bytes", -2))
            ):
                raise RuntimeError(f"{name} vector manifest payload mismatch: {path.name}")

    if schema == "h67_checkpoint_projection_rtl_exact_v4":
        source_paths = [
            Path(str(artifacts["runner"]["path"])),
            Path(str(artifacts["testbench"]["path"])),
            Path(str(artifacts["sva"]["path"])),
            Path(str(artifacts["bind"]["path"])),
            *(Path(str(binding["path"])) for binding in rtl),
        ]
        source_set_sha = _source_set_digest(source_paths)
        if report.get("source_set_sha256") != source_set_sha:
            raise RuntimeError("H67 projection v4 source-set SHA mismatch")
        receipts = report.get("run_receipts") or []
        if len(receipts) != 24:
            raise RuntimeError("H67 projection v4 must bind 24 run receipts")
        expected_receipts = {
            (name, simulator)
            for name in EXPECTED_ALL12_NAMES
            for simulator in ("icarus", "verilator")
        }
        actual_receipts = {
            (str(row.get("name", "")), str(row.get("simulator", "")))
            for row in receipts
        }
        if actual_receipts != expected_receipts:
            raise RuntimeError("H67 projection v4 receipt coverage is not all12 x dual simulator")
        record_by_name = {str(record["name"]): record for record in records}
        for receipt in receipts:
            name = str(receipt["name"])
            simulator = str(receipt["simulator"])
            expected_assertions = "enabled" if simulator == "verilator" else "none"
            if receipt.get("assertions") != expected_assertions:
                raise RuntimeError(f"{name} {simulator} assertion marker mismatch")
            record = record_by_name[name]
            expected_vector_sha = hashlib.sha256(
                json.dumps(
                    record["files"], sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if receipt.get("vector_aggregate_sha256") != expected_vector_sha:
                raise RuntimeError(f"{name} {simulator} vector aggregate SHA mismatch")
            if receipt.get("source_set_sha256") != source_set_sha:
                raise RuntimeError(f"{name} {simulator} source-set SHA mismatch")
            if receipt.get("vector_id") != Path(str(record["vector_dir"])).name:
                raise RuntimeError(f"{name} {simulator} vector ID mismatch")


def validate_motion_tesc_provenance(report: dict[str, Any]) -> None:
    if report.get("schema") != "motion_temporal_equivalence_v2":
        raise RuntimeError("Motion TESC provenance requires schema v2")
    provenance = report.get("provenance") or {}
    bindings = {}
    for key in (
        "profile",
        "config",
        "checkpoint",
        "analyzer",
        "validator",
        "watcher",
        "test_log",
    ):
        bindings[key] = _validate_binding(provenance.get(key) or {}, f"TESC {key}")
    tests = provenance.get("tests") or []
    if len(tests) != 3:
        raise RuntimeError("Motion TESC must bind three test sources")
    for index, binding in enumerate(tests):
        _validate_binding(binding, f"TESC test[{index}]")
    source = report.get("source") or {}
    if bindings["profile"].resolve() != Path(str(report.get("profile", ""))).resolve():
        raise RuntimeError("Motion TESC profile path mismatch")
    if bindings["profile"].resolve() != Path(str(source.get("profile", ""))).resolve():
        raise RuntimeError("Motion TESC source/profile path mismatch")
    for key in ("config", "checkpoint"):
        if bindings[key].resolve() != Path(str(source.get(f"{key}_path", ""))).resolve():
            raise RuntimeError(f"Motion TESC {key} path mismatch")
        if provenance[key].get("sha256") != source.get(f"{key}_sha256"):
            raise RuntimeError(f"Motion TESC {key} SHA mismatch")


def validate_motion_rqtb_provenance(report: dict[str, Any]) -> None:
    if report.get("schema") != "motion_reversible_quotient_bundle_v2":
        raise RuntimeError("Motion RQTB provenance requires schema v2")
    provenance = report.get("provenance") or {}
    bindings = {}
    for key in (
        "profile",
        "tesc_report",
        "config",
        "checkpoint",
        "model",
        "validator",
        "watcher",
        "test_log",
    ):
        bindings[key] = _validate_binding(provenance.get(key) or {}, f"RQTB {key}")
    tests = provenance.get("tests") or []
    if len(tests) != 3:
        raise RuntimeError("Motion RQTB must bind three test sources")
    for index, binding in enumerate(tests):
        _validate_binding(binding, f"RQTB test[{index}]")
    source = report.get("source") or {}
    if bindings["profile"].resolve() != Path(str(source.get("profile", ""))).resolve():
        raise RuntimeError("Motion RQTB source/profile path mismatch")
    for key in ("config", "checkpoint"):
        if bindings[key].resolve() != Path(str(source.get(f"{key}_path", ""))).resolve():
            raise RuntimeError(f"Motion RQTB {key} path mismatch")
        if provenance[key].get("sha256") != source.get(f"{key}_sha256"):
            raise RuntimeError(f"Motion RQTB {key} SHA mismatch")
    tesc = json.loads(bindings["tesc_report"].read_text(encoding="utf-8"))
    validate_motion_tesc_provenance(tesc)
    tesc_profile = (tesc.get("provenance") or {}).get("profile") or {}
    if tesc_profile.get("sha256") != provenance["profile"].get("sha256"):
        raise RuntimeError("Motion RQTB/TESC profile SHA mismatch")

def validate_local5_projection_provenance(scope: dict[str, Any]) -> None:
    if scope.get("schema") != "local5_checkpoint_bound_component_rtl_exact_v2":
        raise RuntimeError("Local5 component provenance requires scope schema v2")
    artifacts = scope.get("source_artifacts") or {}
    for key in (
        "projection_report",
        "projection_vector_manifest",
        "projection_source_manifest",
        "projection_source_trace_manifest",
        "projection_source_trace_payload",
        "score_report",
        "atlif_report",
        "runner",
        "projection_shell",
        "projection_generator",
        "projection_summarizer",
        "score_shell",
        "score_generator",
        "score_reporter",
        "provenance_validator",
    ):
        if not isinstance(artifacts.get(key), dict):
            raise RuntimeError(f"Local5 component provenance missing {key}")
        _validate_binding(artifacts[key], key)
    code_suffixes = {
        "runner": "scripts/run_local5_bb1e4_checkpoint_bound_rtl.py",
        "projection_shell": "sim_new_arch/run_local5_qgasr2c_fivebank_checks.sh",
        "projection_generator": "scripts/generate_local5_active_projection_postg0_vectors.py",
        "projection_summarizer": "scripts/summarize_local5_gasr2c_fivebank_rtl.py",
        "score_shell": "sim_local5/run_local5_checkpoint_score_trace_checks.sh",
        "score_generator": "scripts/generate_local5_checkpoint_score_vectors.py",
        "score_reporter": "scripts/report_local5_checkpoint_score_rtl.py",
        "provenance_validator": "scripts/evidence_provenance.py",
    }
    for key, suffix in code_suffixes.items():
        _require_suffix(Path(str(artifacts[key]["path"])), suffix, key)

    score_report = json.loads(
        Path(str(artifacts["score_report"]["path"])).read_text(encoding="utf-8")
    )
    atlif_report = json.loads(
        Path(str(artifacts["atlif_report"]["path"])).read_text(encoding="utf-8")
    )
    validate_local5_score_provenance(score_report)
    validate_local5_atlif_provenance(atlif_report)

    projection_sources = artifacts.get("projection_sources") or []
    if len(projection_sources) != 20:
        raise RuntimeError("Local5 projection provenance must bind 13 RTL/TB and 7 SVA sources")
    source_suffixes = set()
    for index, binding in enumerate(projection_sources):
        path = _validate_binding(binding, f"Local5 projection source[{index}]")
        matches = {
            suffix
            for suffix in LOCAL5_PROJECTION_SOURCE_SUFFIXES
            if path.as_posix().endswith("/" + suffix)
        }
        if len(matches) != 1:
            raise RuntimeError(f"unexpected Local5 projection source: {path}")
        source_suffixes.update(matches)
    if source_suffixes != LOCAL5_PROJECTION_SOURCE_SUFFIXES:
        raise RuntimeError("Local5 projection source set is incomplete or duplicated")

    vector_manifest_path = _validate_binding(
        artifacts["projection_vector_manifest"], "projection_vector_manifest"
    )
    vector_manifest = json.loads(vector_manifest_path.read_text(encoding="utf-8"))
    vector_dir = vector_manifest_path.parent
    payloads = artifacts.get("projection_payloads") or []
    expected_artifacts = vector_manifest.get("artifacts") or {}
    if len(payloads) != len(expected_artifacts):
        raise RuntimeError("Local5 projection payload count mismatch")
    payload_by_name = {Path(str(row.get("path", ""))).name: row for row in payloads}
    expected_names = {str(row.get("file", "")) for row in expected_artifacts.values()}
    if set(payload_by_name) != expected_names:
        raise RuntimeError("Local5 projection payload set mismatch")
    for artifact in expected_artifacts.values():
        name = str(artifact["file"])
        binding = payload_by_name[name]
        path = _validate_binding(binding, f"Local5 projection payload {name}")
        if path.resolve() != (vector_dir / name).resolve():
            raise RuntimeError(f"Local5 projection payload path mismatch: {name}")
        if binding.get("sha256") != artifact.get("sha256"):
            raise RuntimeError(f"Local5 projection payload manifest SHA mismatch: {name}")

    for manifest_key, artifact_key, sha_key in (
        ("source_manifest", "projection_source_trace_manifest", "source_manifest_sha256"),
        ("source_payload", "projection_source_trace_payload", "source_payload_sha256"),
    ):
        path = _validate_binding(artifacts[artifact_key], artifact_key)
        if Path(str(vector_manifest.get(manifest_key, ""))).resolve() != path.resolve():
            raise RuntimeError(f"Local5 {manifest_key} path mismatch")
        if vector_manifest.get(sha_key) != sha256_file(path):
            raise RuntimeError(f"Local5 {manifest_key} SHA mismatch")
