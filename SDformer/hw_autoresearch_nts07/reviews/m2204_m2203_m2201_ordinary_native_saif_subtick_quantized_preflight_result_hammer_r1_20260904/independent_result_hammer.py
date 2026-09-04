#!/usr/bin/python3.12
"""Independent M2204 raw-result hammer; never invokes tools or modifies sources."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent / "mechanical_checks.json"
RESULT = HW / "results/m2203_m2201_m2018_ordinary_native_saif_subtick_quantized_preflight_r1_20260904"
ATTEMPT = HW / "results/.m2203_m2201_ordinary_native_saif_subtick_quantized_preflight_attempt_consumed"
M2202 = HW / "reviews/m2202_m2201_m2188_ordinary_native_saif_subtick_quantized_preflight_source_hammer_r1_20260904"
M2188 = HW / "reviews/m2188_m2187_m2185_ordinary_native_saif_gate_level_preflight_failure_result_hammer_r1_20260904"
M2187 = HW / "results/m2187_m2185_m2018_ordinary_native_saif_gate_level_preflight_r1_20260904.failed.3245526.quarantine"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
TARGET = "dut_ordinary"
CRITICAL = (
    "bridge_accept", "bridge_valid", "commit_accept", "commit_valid",
    "mem_req_accept", "mem_req_valid", "mem_rsp_accept", "mem_rsp_valid",
)
AUDIT_NAMES = CRITICAL + ("load_valid",)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exhaustive_seal(directory: Path) -> list[str]:
    assert directory.is_dir() and not directory.is_symlink()
    assert not any(path.is_symlink() for path in directory.rglob("*"))
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    listed: list[str] = []
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        assert sha(directory / name) == digest, name
        listed.append(name)
    actual = sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {manifest.name, outer.name}
    )
    assert sorted(listed) == actual
    digest, name = outer.read_text().split(maxsplit=1)
    assert name.strip().lstrip("*") == manifest.name and sha(manifest) == digest
    return actual


def file_seal(path: Path) -> dict[str, str]:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    digest, name = sidecar.read_text().split(maxsplit=1)
    assert name.strip().lstrip("*") == path.name and digest == sha(path)
    outer_digest, outer_name = outer.read_text().split(maxsplit=1)
    assert outer_name.strip().lstrip("*") == sidecar.name
    assert outer_digest == sha(sidecar)
    return {"sha256": digest, "sidecar_sha256": sha(sidecar),
            "outer_sha256": sha(outer)}


TOKEN = re.compile(r"/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|[()]|[^\s()]+", re.S)


def parse_saif_independent(path: Path) -> dict[str, object]:
    """Small independent streaming S-expression audit, unrelated to M2201 parser."""
    text = path.read_text(errors="strict")
    stack: list[dict[str, object]] = []
    duration = None
    timescale: tuple[float, str] | None = None
    instances = 0
    target_instances = 0
    records = 0
    outside = 0
    tx_nonzero = 0
    tx_sum = 0.0
    toggled = 0
    noninteger_fields = 0
    sums: list[float] = []
    critical = {name: 0 for name in AUDIT_NAMES}

    for match in TOKEN.finditer(text):
        token = match.group(0)
        if token.startswith("/*"):
            continue
        if token == "(":
            stack.append({"head": None, "atoms": [], "fields": {}})
            continue
        if token != ")":
            assert stack, token
            frame = stack[-1]
            if frame["head"] is None:
                frame["head"] = token
            else:
                frame["atoms"].append(token)  # type: ignore[union-attr]
            continue

        assert stack, "unbalanced close"
        frame = stack.pop()
        head = frame["head"]
        atoms = frame["atoms"]
        fields = frame["fields"]
        assert isinstance(atoms, list) and isinstance(fields, dict)
        if head in {"T0", "T1", "TX", "TC"}:
            assert len(atoms) == 1
            value = float(atoms[0])
            assert stack
            stack[-1]["fields"][head] = value  # type: ignore[index]
            continue
        if head == "DURATION":
            assert len(atoms) == 1
            duration = float(atoms[0])
        elif head == "TIMESCALE":
            assert len(atoms) == 2
            timescale = (float(atoms[0]), str(atoms[1]))
        elif head == "INSTANCE":
            instances += 1
            assert atoms
            if atoms[0] == TARGET:
                target_instances += 1

        if all(name in fields for name in ("T0", "T1", "TX", "TC")):
            assert isinstance(head, str)
            values = tuple(float(fields[name]) for name in ("T0", "T1", "TX", "TC"))
            assert all(math.isfinite(value) and value >= 0.0 for value in values)
            inside = any(
                ancestor["head"] == "INSTANCE"
                and isinstance(ancestor["atoms"], list)
                and ancestor["atoms"]
                and ancestor["atoms"][0] == TARGET
                for ancestor in stack
            )
            if inside:
                records += 1
                t0, t1, tx, tc = values
                sums.append(t0 + t1 + tx)
                tx_nonzero += int(tx != 0.0)
                tx_sum += tx
                toggled += int(tc > 0.0)
                noninteger_fields += sum(not value.is_integer() for value in values)
                for base in AUDIT_NAMES:
                    if (head == base or re.fullmatch(re.escape(base) + r"\\?\[[^]]+\]", head)) and tc > 0:
                        critical[base] += 1
            else:
                outside += 1

    assert not stack and duration is not None and timescale is not None
    scale, unit = timescale
    unit_ns = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0,
               "ps": 1e-3, "fs": 1e-6}[unit]
    return {
        "duration_raw": duration,
        "duration_ns": duration * scale * unit_ns,
        "instances": instances,
        "target_instances": target_instances,
        "records": records,
        "outside": outside,
        "tx_nonzero": tx_nonzero,
        "tx_sum": tx_sum,
        "toggled": toggled,
        "noninteger_fields": noninteger_fields,
        "min_sum": min(sums),
        "max_sum": max(sums),
        "critical_nonzero": critical,
    }


def main() -> int:
    result_members = exhaustive_seal(RESULT)
    attempt_members = exhaustive_seal(ATTEMPT)
    m2202_members = exhaustive_seal(M2202)
    m2188_members = exhaustive_seal(M2188)
    m2187_members = exhaustive_seal(M2187)
    assert len(result_members) == 16 and attempt_members == ["attempt.json"]
    assert sha(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

    raw = json.loads(RESULT.joinpath("result.json").read_text())
    assert raw["status"] == (
        "PASS_RAW_M2203_M2201_SUBTICK_NATIVE_SAIF_PREFLIGHT_PENDING_M2204_RESULT_HAMMER"
    )
    source_review = json.loads(M2202.joinpath("review.json").read_text())
    assert source_review["status"] == "PASS_M2202_M2201_SOURCE_HAMMER__M2203_ONE_SHOT_AUTHORIZED"
    assert raw["identity"]["m2202_review_sha256"] == sha(M2202 / "review.json")

    expected_counts = {
        "admitted_measurement_saif_files": 1, "admitted_saif_files": 1,
        "dc_runs": 0, "diagnostic_saif_files_written": 1, "gpu_runs": 0,
        "icc2_runs": 0, "license_queries": 1, "ptpx_runs": 0,
        "raw_saif_files_written": 2, "simv_runs": 1, "vcs_compiles": 1,
    }
    assert raw["execution_counts"] == expected_counts
    attempt = json.loads(ATTEMPT.joinpath("attempt.json").read_text())
    assert attempt["status"] == "M2203_ATTEMPT_CONSUMED"
    assert attempt["budget"] == expected_counts and attempt["automatic_retry"] is False
    commands = json.loads(RESULT.joinpath("execution_commands.json").read_text())
    assert set(commands) == {"license_preflight", "vcs_compile", "simv", "timing_surface"}
    assert commands["license_preflight"].count("lmstat") == 1
    assert commands["vcs_compile"].count("-full64") == 1
    assert commands["simv"][0] == "./simv"
    compile_log = RESULT.joinpath("vcs_compile.log").read_text(errors="replace")
    sim_log = RESULT.joinpath("rtl_sim.log").read_text(errors="replace")
    assert compile_log.count("Chronologic VCS (TM)") == 1
    assert compile_log.count("CPU time:") == 1
    assert not re.search(r"(?:Error-|Syntax error|Compiler directive error|UVM_FATAL)", compile_log)
    assert sim_log.count("PASS_M2160_ORDINARY_SINGLE_AXIS_NATIVE_SAIF_PREFLIGHT") == 1
    assert sim_log.count("M2160_RTL_SAIF_WINDOW_BEGIN") == 1
    assert sim_log.count("M2160_RTL_SAIF_WINDOW_END") == 1
    assert not re.search(r"(?:\$fatal|Assertion failed|Offending|UVM_FATAL)", sim_log)

    end = re.search(
        r"M2160_RTL_SAIF_WINDOW_END axis=ordinary_lru4 .*?"
        r"measurement_cycles=(\d+) rows=(\d+) issues=(\d+) products=(\d+) "
        r"commits=(\d+) bundles=(\d+) scalar_weight_reads=(\d+) duration_ns=([0-9.]+)",
        sim_log,
    )
    assert end
    ledger = tuple(map(int, end.groups()[:7]))
    assert ledger == (20292, 149, 1278, 29472, 24, 1788, 14304)
    assert float(end.group(8)) == 60876.0

    pre_path = RESULT / "rtl_prehistory.saif"
    meas_path = RESULT / "rtl_measurement.saif"
    pre_seal = file_seal(pre_path)
    meas_seal = file_seal(meas_path)
    pre = parse_saif_independent(pre_path)
    meas = parse_saif_independent(meas_path)

    assert pre["records"] == 93971 and pre["target_instances"] == 1 and pre["outside"] == 0
    assert pre["duration_ns"] == 1167.01
    assert pre["noninteger_fields"] == 0
    assert pre["min_sum"] == pre["max_sum"] == math.floor(float(pre["duration_raw"])) == 1167
    residual = float(pre["duration_raw"]) - float(pre["min_sum"])
    assert math.isclose(residual, 0.01, rel_tol=0.0, abs_tol=1e-9)
    assert 0.0 < residual < 1.0
    assert pre["tx_nonzero"] == 45 and pre["tx_sum"] == 45.0
    assert pre["critical_nonzero"]["load_valid"] > 0

    assert meas["records"] == 93971 and meas["target_instances"] == 1 and meas["outside"] == 0
    assert meas["duration_ns"] == 60876.0
    assert meas["min_sum"] == meas["max_sum"] == 60876.0
    assert meas["tx_nonzero"] == 0 and meas["tx_sum"] == 0.0
    assert all(meas["critical_nonzero"][name] > 0 for name in CRITICAL)
    assert meas["toggled"] == 76264

    # Result fields must reproduce independent raw parsing exactly.
    published_pre = raw["diagnostic_prehistory_saif"]
    published_meas = raw["measurement_saif"]
    assert published_pre["record_count"] == pre["records"]
    assert published_pre["outside_target_record_count"] == pre["outside"]
    assert published_pre["tx_nonzero_record_count"] == pre["tx_nonzero"]
    assert published_pre["tx_sum"] == pre["tx_sum"]
    assert published_pre["identity_seal"] == pre_seal
    assert published_meas["record_count"] == meas["records"]
    assert published_meas["outside_target_record_count"] == meas["outside"]
    assert published_meas["tx_nonzero_record_count"] == meas["tx_nonzero"]
    assert published_meas["nonzero_toggle_record_count"] == meas["toggled"]
    assert published_meas["critical_nonzero_record_counts"] == {
        name: meas["critical_nonzero"][name] for name in CRITICAL
    }
    assert published_meas["identity_seal"] == meas_seal

    # M2187 is a sealed failed regression only; neither raw file was reused.
    old_pre = M2187 / "rtl_prehistory.saif"
    old_meas = M2187 / "rtl_measurement.saif"
    assert sha(old_pre) != sha(pre_path) and sha(old_meas) != sha(meas_path)
    assert raw["identity"]["m2188_failure_review_sha256"] == sha(M2188 / "review.json")

    census = sorted(path.name for path in (HW / "results").glob("*m2203*")
                    if not path.name.startswith("."))
    hidden_census = sorted(path.name for path in (HW / "results").glob(".m2203*"))
    assert census == [RESULT.name]
    assert hidden_census == [ATTEMPT.name]
    assert not any((HW / "results").glob(".m2203_m2201_work*"))
    assert not any((HW / "results").glob(".m2203_m2201_stage*"))
    assert not any((HW / "results").glob(".m2203*m2201*lock*"))

    boundary = raw["claim_boundary"]
    assert boundary["vcs_native_rtl_saif_acquisition_preflight"] is True
    assert boundary["measurement_saif_candidate_only"] is True
    assert boundary["diagnostic_prehistory_never_annotated"] is True
    for key in ("dc_run", "ptpx_run", "icc2_run", "mapped_netlist_activity",
                "power_or_energy", "component_speedup_admitted", "system_speedup",
                "paper_citable"):
        assert boundary[key] is False, key

    result = {
        "schema": "m2204_m2203_m2201_native_saif_preflight_mechanical_checks_r1_v1",
        "status": "PASS_M2204_INDEPENDENT_RAW_AUDIT",
        "sealed_members": {"result": len(result_members), "attempt": len(attempt_members),
                           "m2202": len(m2202_members), "m2188": len(m2188_members),
                           "m2187_failed": len(m2187_members)},
        "execution": {"attempts": 1, "license_queries": 1, "vcs_compiles": 1,
                      "simv_runs": 1, "dc": 0, "ptpx": 0, "icc2": 0, "gpu": 0},
        "runtime_ledger": {"cycles": ledger[0], "rows": ledger[1], "issues": ledger[2],
                           "products": ledger[3], "commits": ledger[4],
                           "bundles": ledger[5], "reads": ledger[6]},
        "diagnostic": {**pre, "uniform_subtick_residual": residual,
                       "saif_sha256": pre_seal["sha256"]},
        "measurement": {**meas, "critical_classes_nonzero": 8,
                        "saif_sha256": meas_seal["sha256"]},
        "m2187_raw_reused": False,
        "m2203_visible_census": census,
        "m2203_hidden_census": hidden_census,
        "docs359_sha256": sha(DOCS359),
        "claim_boundary": "native RTL SAIF acquisition preflight only; no power, energy, PPA, speedup, or paper claim",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS_M2204_RAW_AUDIT records=93971/93971 measurement_tx=0 critical=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
