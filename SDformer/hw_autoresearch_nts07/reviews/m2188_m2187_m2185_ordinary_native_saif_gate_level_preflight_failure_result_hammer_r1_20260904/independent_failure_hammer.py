#!/usr/bin/python3.12
"""Read-only M2188 hammer for the sole failed M2187 quarantine."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
HERE = Path(__file__).resolve().parent
Q = HW / "results/m2187_m2185_m2018_ordinary_native_saif_gate_level_preflight_r1_20260904.failed.3245526.quarantine"
ATTEMPT = HW / "results/.m2187_m2185_ordinary_native_saif_gate_level_preflight_attempt_consumed"
CANONICAL = HW / "results/m2187_m2185_m2018_ordinary_native_saif_gate_level_preflight_r1_20260904"
LOCK = HW / "results/.m2187_m2185_ordinary_native_saif_gate_level_preflight_launch_lock"
M2186 = HW / "reviews/m2186_m2185_m2179_ordinary_native_saif_gate_level_preflight_source_hammer_r1_20260904"
PARSER = HW / "system_simulator/scripts/parse_m2176_m2018_ordinary_native_saif_reset_semantics_preflight.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "quarantine_manifest": "ae125f6532a79f65c8f3abb8ae93111e09e7554ca3c93a07e62a8b14cd52008e",
    "quarantine_outer": "5d3abf3c570045cde0f77241b6c275c2f1db4bbfe5c85e7786e312a036be8396",
    "attempt_manifest": "a5d57d94faa32f40f2638fabd14c1d6665fceed6ed937dab283fbcd8f4c8fa3c",
    "attempt_outer": "f8947170c969137c50cec194b6a4aa11d5aebf2008c3f32a58f6744741861b7c",
    "m2186_review": "599de567be014f3c7934b77b324f4c90ff07fb91c88e50b00f38608b96db16c2",
    "parser": "2dadf88ccfb4f4e43281203c67317b9f0bf91ed1fa3874eadb6015db9244438d",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "prehistory_saif": "96b488f38055b241f661888681794a397b79070d498a8a61d005f2bc4e615db3",
    "measurement_saif": "65725beefe9cfe26c9f0c243b2a0433c2910f70b50a1fcc24beab48648060dae",
    "runtime": "5a5ca05f69c97d01e72715289081c49983a6c4cd668f145a9a7a9a4569831e63",
}


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    need(isinstance(value, dict), "JSON object required")
    return value


def verify_seal(directory: Path) -> dict[str, object]:
    need(directory.is_dir() and not directory.is_symlink(), "sealed directory invalid")
    need(not any(path.is_symlink() for path in directory.rglob("*")), "sealed symlink")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    listed: set[str] = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe seal path")
        need((directory / rel).is_file() and sha(directory / rel) == digest,
             "member seal mismatch: " + rel.as_posix())
        listed.add(rel.as_posix())
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == listed, "non-exhaustive seal")
    return {"member_count": len(listed), "manifest_sha256": sha(manifest),
            "outer_sha256": sha(outer), "exhaustive": True}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "module load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile_saif(base, path: Path, role: str) -> dict[str, object]:
    text = base.read(path)
    root = base.parse_balanced_saif(text)
    duration = float(base.header_value(root, "DURATION"))
    timescales = [node for node in base.all_nodes(root) if base.head(node) == "TIMESCALE"]
    need(len(timescales) == 1, "TIMESCALE count")
    scale_atoms = [item for item in timescales[0][1:] if isinstance(item, str)]
    need(scale_atoms == ["1", "ns"], "unexpected TIMESCALE")
    instances = [node for node in base.all_nodes(root) if base.head(node) == "INSTANCE"]
    targets = [node for node in instances
               if base.atom_after_head(node) == base.TARGET_INSTANCE]
    need(len(targets) == 1, "target count")
    records, outside = base.collect_activity(root, targets[0])
    residual_histogram: dict[str, int] = {}
    sum_histogram: dict[str, int] = {}
    noninteger_activity_time_fields = 0
    tx_nonzero = 0
    tx_sum = 0.0
    toggled = 0
    named: dict[str, list[float]] = {}
    for record in records:
        name = base.activity_name(record)
        t0, t1, tx, tc = (base.numeric_field(record, field)
                          for field in ("T0", "T1", "TX", "TC"))
        need(min(t0, t1, tx, tc) >= 0.0, "negative activity")
        total = t0 + t1 + tx
        residual = round(duration - total, 9)
        residual_histogram[str(residual)] = residual_histogram.get(str(residual), 0) + 1
        sum_histogram[str(total)] = sum_histogram.get(str(total), 0) + 1
        noninteger_activity_time_fields += sum(not value.is_integer()
                                               for value in (t0, t1, tx))
        tx_nonzero += int(tx != 0.0)
        tx_sum += tx
        toggled += int(tc > 0.0)
        named.setdefault(name, []).append(tc)
    critical_tokens = base.CRITICAL if role == "measurement" else ("load_valid",)
    critical: dict[str, dict[str, float | int]] = {}
    for token in critical_tokens:
        values = [tc for name, counts in named.items()
                  if name == token or re.fullmatch(re.escape(token) + r"\\?\[[^]]+\]", name)
                  for tc in counts]
        critical[token] = {"record_count": len(values),
                           "nonzero_tc_records": sum(value > 0.0 for value in values),
                           "tc_sum": sum(values)}
    return {
        "file": path.name, "sha256": sha(path), "role": role,
        "timescale": "1 ns", "duration_raw": duration,
        "duration_floor": math.floor(duration),
        "duration_fraction": round(duration - math.floor(duration), 9),
        "instance_count": len(instances), "target_instance_count": len(targets),
        "record_count": len(records), "outside_target_record_count": len(outside),
        "residual_histogram": residual_histogram, "sum_histogram": sum_histogram,
        "noninteger_activity_time_fields": noninteger_activity_time_fields,
        "tx_nonzero_record_count": tx_nonzero, "tx_sum": tx_sum,
        "nonzero_toggle_record_count": toggled,
        "critical_nonzero_record_counts": critical,
    }


def main() -> int:
    quarantine_matches = sorted(HW.glob(
        "results/m2187_m2185_m2018_ordinary_native_saif_gate_level_preflight_r1_20260904.failed.*.quarantine"))
    need(quarantine_matches == [Q], "not exactly one frozen M2187 quarantine")
    need(not CANONICAL.exists() and not LOCK.exists(), "M2187 canonical/lock exists")
    quarantine_seal = verify_seal(Q)
    attempt_seal = verify_seal(ATTEMPT)
    source_review_seal = verify_seal(M2186)
    need(quarantine_seal["manifest_sha256"] == EXPECTED["quarantine_manifest"] and
         quarantine_seal["outer_sha256"] == EXPECTED["quarantine_outer"],
         "quarantine identity")
    need(attempt_seal["manifest_sha256"] == EXPECTED["attempt_manifest"] and
         attempt_seal["outer_sha256"] == EXPECTED["attempt_outer"], "attempt identity")
    need(sha(M2186 / "review.json") == EXPECTED["m2186_review"], "M2186 identity")
    need(sha(PARSER) == EXPECTED["parser"] and sha(DOC359) == EXPECTED["docs359"],
         "parser/docs identity")
    attempt = read_json(ATTEMPT / "attempt.json")
    need(attempt["status"] == "M2187_ATTEMPT_CONSUMED" and
         attempt["automatic_retry"] is False, "M2187 attempt status")
    counts = read_json(Q / "execution_counts.json")
    need(counts == {"license_queries": 1, "vcs_compiles": 1, "simv_runs": 1,
                    "raw_saif_files_written": 2, "diagnostic_saif_files_written": 1,
                    "admitted_measurement_saif_files": 0, "admitted_saif_files": 0,
                    "dc_runs": 0, "ptpx_runs": 0, "icc2_runs": 0, "gpu_runs": 0},
         "execution counts")
    commands = read_json(Q / "execution_commands.json")
    need("-debug_access+r" in commands["vcs_compile"] and
         commands["simv"].count("+M2160_AXIS_ORDINARY") == 1 and
         commands["simv"].count("+WORKLOAD_SLOT=42") == 1, "execution command surface")
    compile_log = (Q / "vcs_compile.log").read_text(errors="replace")
    runtime_log = (Q / "rtl_sim.log").read_text(errors="replace")
    need("All of 6 modules done" in compile_log and "CPU time:" in compile_log,
         "VCS compile completion")
    need("PASS_M2160_ORDINARY_SINGLE_AXIS_NATIVE_SAIF_PREFLIGHT" in runtime_log and
         "ledger_exact=1" in runtime_log and "arithmetic_scoreboard_exact=1" in runtime_log and
         "second_axis=0" in runtime_log, "runtime PASS fingerprint")
    need("Error-" not in compile_log and "Assertion failed" not in runtime_log and
         "assertion fail" not in runtime_log.casefold(), "compile/runtime hard failure")

    parser = load(PARSER, "m2176_parser_for_m2188")
    base = parser.BASE
    runtime = parser.parse_runtime(Q / "rtl_sim.log")
    need(runtime["completion_ledger"] == {"cycles": 20292, "rows": 149,
         "issues": 1278, "products": 29472, "commits": 24,
         "bundles": 1788, "reads": 14304}, "runtime ledger")
    prehistory = profile_saif(base, Q / "rtl_prehistory.saif", "diagnostic_prehistory")
    measurement = profile_saif(base, Q / "rtl_measurement.saif", "measurement")
    need(prehistory["sha256"] == EXPECTED["prehistory_saif"] and
         measurement["sha256"] == EXPECTED["measurement_saif"] and
         sha(Q / "rtl_sim.log") == EXPECTED["runtime"], "raw identity")
    need(prehistory["target_instance_count"] == 1 and
         prehistory["record_count"] == 93971 and
         prehistory["outside_target_record_count"] == 0 and
         prehistory["nonzero_toggle_record_count"] == 1285 and
         prehistory["critical_nonzero_record_counts"]["load_valid"]["nonzero_tc_records"] == 1,
         "prehistory hierarchy/activity")
    need(prehistory["duration_raw"] == 1167.01 and
         prehistory["duration_floor"] == 1167 and
         prehistory["duration_fraction"] == 0.01 and
         prehistory["noninteger_activity_time_fields"] == 0 and
         prehistory["residual_histogram"] == {"0.01": 93971} and
         prehistory["sum_histogram"] == {"1167.0": 93971},
         "uniform sub-tick quantization fingerprint")
    try:
        parser.parse_saif(Q / "rtl_prehistory.saif", role="diagnostic_prehistory")
    except parser.Failure as exc:
        need(str(exc) == "SAIF conservation failures: 93971", "unexpected diagnostic failure")
    else:
        raise RuntimeError("frozen parser unexpectedly accepts diagnostic SAIF")
    measurement_admissibility = parser.parse_saif(
        Q / "rtl_measurement.saif", role="measurement")
    need(measurement_admissibility["record_count"] == 93971 and
         measurement_admissibility["conservation_failures"] == 0 and
         measurement_admissibility["tx_nonzero_record_count"] == 0 and
         measurement["residual_histogram"] == {"0.0": 93971} and
         measurement["outside_target_record_count"] == 0 and
         all(item["nonzero_tc_records"] > 0
             for item in measurement["critical_nonzero_record_counts"].values()),
         "measurement SAIF admissibility")
    need(prehistory["tx_nonzero_record_count"] == 45 and prehistory["tx_sum"] == 45.0,
         "diagnostic unknown census")
    need((Q / "prehistory_saif_parse.log").read_text().strip() ==
         "M2176_PARSE_FAIL_CLOSED: SAIF conservation failures: 93971",
         "first failing gate fingerprint")

    result = {
        "schema": "m2188_m2187_m2185_ordinary_native_saif_gate_level_preflight_failure_mechanical_checks_r1_v1",
        "status": "PASS_M2188_FAILURE_ROOT_CAUSE_MECHANICAL_CHECKS",
        "seals": {"quarantine": quarantine_seal, "attempt": attempt_seal,
                  "m2186_source_review": source_review_seal},
        "census": {"m2187_attempt_consumed": True, "canonical_result_absent": True,
                   "launch_lock_absent": True, "failure_quarantine_count": 1},
        "execution_counts": counts,
        "vcs_and_runtime": {"compile_passed": True, "simulation_passed": True,
                            "runtime_parser_passed": True,
                            "completion_ledger": runtime["completion_ledger"],
                            "arithmetic_scoreboard_exact": True},
        "prehistory_saif": prehistory,
        "measurement_saif": measurement,
        "root_cause": {
            "classification": "DIAGNOSTIC_ONLY_SUB_TIMESCALE_QUANTIZATION_PARSER_FALSE_NEGATIVE",
            "activity_pollution_observed": False,
            "evidence": "At TIMESCALE 1 ns, all 93,971 diagnostic records contain integer T0/T1/TX fields summing to floor(1167.01)=1167, hence the same 0.01 residual; the measurement duration is integral and all 93,971 records conserve exactly.",
            "confidence": "high",
        },
        "claim_boundary": {"m2187_admitted": False, "m2187_paper_citable": False,
                           "m2187_retry": False, "measurement_raw_file_diagnostically_valid": True,
                           "power_or_energy": False, "system_speedup": False},
        "review_execution": {"vcs_runs": 0, "simv_runs": 0, "license_queries": 0,
                             "saif_files_written": 0, "dc_runs": 0, "ptpx_runs": 0,
                             "icc2_runs": 0, "gpu_runs": 0, "git_mutations": 0,
                             "m2187_retry": False, "docs359_modified": False},
    }
    out = HERE / "mechanical_checks.json"
    need(not out.exists(), "fresh mechanical output required")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
