#!/usr/bin/python3.12
"""Independent no-EDA M2173 hammer for the committed M2172 source."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
PARSER = HW / "system_simulator/scripts/parse_m2172_m2018_ordinary_native_saif_balanced_scope_preflight.py"
RUNNER = HW / "dc_handoff/scripts/run_m2172_m2018_ordinary_native_saif_balanced_scope_preflight_one_shot.py"
TEST = HW / "tests/test_m2172_ordinary_native_saif_balanced_scope_preflight.py"
CONTRACT = HW / "contracts/m2172_m2018_ordinary_native_saif_balanced_scope_preflight_source_contract_r1_20260904.json"
AUTHOR = HW / "reviews/m2172_m2018_ordinary_native_saif_balanced_scope_preflight_source_author_receipt_r1_20260904"
TB = HW / "tb_m2018/tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.sv"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2160_m2018_ordinary_native_saif_report_reset_preflight_vcs.f"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2174_RESULT = HW / "results/m2174_m2172_m2018_ordinary_native_saif_balanced_scope_preflight_r1_20260904"
M2174_ATTEMPT = HW / "results/.m2174_m2172_ordinary_native_saif_balanced_scope_preflight_attempt_consumed"
M2174_LOCK = HW / "results/.m2174_m2172_ordinary_native_saif_balanced_scope_preflight_launch_lock"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load("m2173_reviewed_m2172_parser", PARSER)


def verify_dir_seal(path: Path) -> list[str]:
    assert path.is_dir() and not path.is_symlink()
    assert not any(node.is_symlink() for node in path.rglob("*"))
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    assert outer.read_text().split() == [sha(manifest), manifest.name]
    entries: list[str] = []
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        member = path / name
        assert member.is_file() and not member.is_symlink() and sha(member) == digest
        entries.append(name)
    actual = sorted(str(node.relative_to(path)) for node in path.rglob("*")
                    if node.is_file() and node.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    assert sorted(entries) == actual
    return entries


def seal_file(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{sha(path)}  {path.name}\n")
    Path(str(sidecar) + ".seal.sha256").write_text(
        f"{sha(sidecar)}  {sidecar.name}\n")


def rows(duration: float, count: int, *, tx_first: int = 0,
         mute_first_critical: bool = False) -> list[str]:
    names = list(M.CRITICAL)
    output: list[str] = []
    for index in range(count):
        name = names[index] if index < len(names) else f"filler_{index}"
        tx = tx_first if index == 0 else 0
        tc = 0 if mute_first_critical and index == 0 else 2
        output.append(
            f"({name} (T0 {duration - 1 - tx:.2f}) (T1 1) (TX {tx}) (TC {tc}) (IG 0))")
    return output


def saif(*, count: int, duration: float = 60876.0, target: str = "dut_ordinary",
         tx_first: int = 0, mute_first_critical: bool = False,
         outside_count: int = 0, empty_target: bool = False,
         duplicate_target: bool = False) -> str:
    inside = rows(duration, count, tx_first=tx_first,
                  mute_first_critical=mute_first_critical)
    outside = rows(duration, outside_count)
    body = [] if empty_target else ["(INSTANCE implementation", "(NET", *inside, ")", ")"]
    duplicate = ["(INSTANCE dut_ordinary)"] if duplicate_target else []
    return "\n".join([
        "/** M2173 legal synthetic SAIF */", "(SAIFILE", '(SAIFVERSION "2.0")',
        "(TIMESCALE 1 ns)", f"(DURATION {duration})", f"(INSTANCE {target}",
        *body, ")", *duplicate, "(NET", *outside, ")", ")", "",
    ])


def expect_saif_reject(path: Path, text: str, role: str = "measurement") -> None:
    path.write_text(text)
    seal_file(path)
    try:
        M.parse_saif(path, role=role)
    except M.Failure:
        return
    raise AssertionError("SAIF mutation unexpectedly admitted")


def run_static(command: list[str]) -> str:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPYCACHEPREFIX": "/tmp/m2173_pycache"}
    completed = subprocess.run(command, cwd=REPO, env=env, check=True,
                               capture_output=True, text=True, timeout=120)
    return completed.stdout


def main() -> int:
    contract = json.loads(CONTRACT.read_text())
    assert sha(RUNNER) == "828c743093afe0c1e506bd820d7cd2fcad0169ae0ea9e9ad8308ec1e3c9c27eb"
    assert sha(PARSER) == "42fd87d6991c46366e80db1d08c20ec5e0d463f3bca8c6050673093d04f3bfe2"
    assert sha(CONTRACT) == "ea1559ba381c58886175afb0031144b98d6edf3f45d1448f5f6ba613ab807738"
    assert sha(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    author_members = verify_dir_seal(AUTHOR)
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    assert sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name]
    assert outer.read_text().split() == [sha(sidecar), sidecar.name]
    assert contract["status"] == "SOURCE_ONLY__M2173_INDEPENDENT_REVIEW_REQUIRED__NO_EDA"
    assert contract["execution_authority"]["direct_execution_authorized_now"] is False
    assert all(not path.exists() for path in (M2174_RESULT, M2174_ATTEMPT, M2174_LOCK))

    author_tests = run_static([sys.executable, "-B", str(TEST)])
    assert "PASS_M2172_SOURCE_TESTS tests=42" in author_tests
    parser_static = json.loads(run_static([sys.executable, "-B", str(PARSER), "static"]))
    runner_static = json.loads(run_static([sys.executable, "-B", str(RUNNER), "--static"]))
    assert parser_static["status"] == "PASS_M2172_STATIC_PARSER"
    assert runner_static["status"] == "PASS_M2172_STATIC_RUNNER"
    topology = M.audit_single_axis_source(TB.read_text(), FILELIST.read_text())
    assert topology == contract["single_axis_topology"]

    failure_probes = [
        "Warning: reset ignored.", "Warning: reset rejected.",
        "Error: reset denied.", "Warning: reset unsupported.",
        "Warning: reset failed.", "Error: reset cannot complete.",
        "Warning: reset unable to complete.", "Error: reset remained uncleared.",
        "Warning: reset retained old counters.", "Error: reset remained active.",
        "Warning: reset not cleared.", "Error: reset not reset.",
        "Warning: clear failed.", "Error: clear request denied.",
    ]
    escaped = [line for line in failure_probes if not M.reset_failure_lines(line)]
    accepted_control = "Info: power reset request accepted and switching counters cleared."
    assert not M.reset_failure_lines(accepted_control)

    with tempfile.TemporaryDirectory(prefix="m2173_saif_") as raw:
        path = Path(raw) / "measurement.saif"
        path.write_text(saif(count=93971))
        seal_file(path)
        full = M.parse_saif(path, role="measurement")
        assert full["record_count"] == 93971
        assert full["outside_target_record_count"] == 0
        assert full["target_instance_count"] == 1
        assert full["tx_nonzero_record_count"] == 0
        assert full["conservation_failures"] == 0
        assert set(full["critical_nonzero_record_counts"]) == set(M.CRITICAL)

        M.EXPECTED["records"] = 32
        valid = saif(count=32)
        path.write_text(valid)
        seal_file(path)
        assert M.parse_saif(path, role="measurement")["record_count"] == 32
        expect_saif_reject(path, saif(count=32, tx_first=1))
        expect_saif_reject(path, saif(count=31))
        expect_saif_reject(path, saif(count=32, mute_first_critical=True))
        expect_saif_reject(path, saif(count=32, target="wrong_instance"))
        expect_saif_reject(path, saif(count=32, duplicate_target=True))
        expect_saif_reject(path, saif(count=0, empty_target=True, outside_count=32))
        expect_saif_reject(path, saif(count=32, outside_count=1))
        expect_saif_reject(path, valid[:-2])
        expect_saif_reject(path, valid + ")\n")
        path.write_text(valid.replace("(T0 60875.00)", "(T0 60874.00)", 1))
        seal_file(path)
        try:
            M.parse_saif(path, role="measurement")
        except M.Failure:
            pass
        else:
            raise AssertionError("conservation mutation admitted")
        path.write_text(valid)
        seal_file(path)
        Path(str(path) + ".sha256").write_text("0" * 64 + f"  {path.name}\n")
        try:
            M.parse_saif(path, role="measurement")
        except M.Failure:
            pass
        else:
            raise AssertionError("file-seal mutation admitted")

    output = {
        "status": "FAIL_M2173_MECHANICAL_CHECKS__RESET_SYNONYM_BYPASS",
        "author_receipt_members": len(author_members),
        "author_tests": 42,
        "parser_static": parser_static["status"],
        "runner_static": runner_static["status"],
        "single_axis_topology": topology,
        "full_saif_records": 93971,
        "full_saif_target_instances": 1,
        "full_saif_outside_records": 0,
        "saif_mutations_independently_rejected": 11,
        "reset_failure_probes": len(failure_probes),
        "reset_failure_probes_escaped": escaped,
        "reset_failure_probe_escape_count": len(escaped),
        "accepted_control_rejected": False,
        "m2174_artifacts": 0,
        "license_queries": 0,
        "vcs_compiles": 0,
        "simv_runs": 0,
        "eda_runs": 0,
        "gpu_runs": 0,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
