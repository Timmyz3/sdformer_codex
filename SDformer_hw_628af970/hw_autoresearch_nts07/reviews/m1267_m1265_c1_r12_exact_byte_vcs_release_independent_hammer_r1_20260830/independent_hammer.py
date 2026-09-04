#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1267 independent, source-only hammer for the M1265 exact-byte release.

This program never executes the release runner, VCS, simv, EDA, GPU, or remote
work.  It only audits immutable bytes and exercises pure checker predicates.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1265_m1258r12_m1162_c1_common_charge_protocol_exact_byte_r12.sh"
FILELIST = HW / "dc_handoff/filelists/date_m1265_m1258r12_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
CHECKER = HW / "verif_m1265_c1_r12_vcs_release/static_check_m1265_c1_r12_exact_byte_vcs_release_source.py"
TESTS = HW / "verif_m1265_c1_r12_vcs_release/test_m1265_c1_r12_exact_byte_vcs_release_source.py"
CONTRACT = HW / "contracts/m1265_c1_r12_exact_byte_vcs_release_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1265_c1_r12_exact_byte_vcs_launch_release_r1_20260830.json"
AUTHOR = HW / "reviews/m1265_c1_r12_exact_byte_vcs_release_source_author_r1_20260830"
REACH = HW / "reviews/m1265a_m1258_r12_exact_tb_reachability_read_only_audit_r1_20260830"
COMPAT = HW / "reviews/m1266_m1265_c1_r12_exact_byte_vcs_release_independent_hammer_r1_20260830"

EXPECTED = {
    RUNNER: "320e8f692557f8111c708f245987d2f831710204a23199030e7a90c3ba6bea28",
    FILELIST: "eb579191d78eee1870fb98866a3436db732db52fdb638e742151b0f10f849de0",
    CHECKER: "fec43010bbcb2f14515fef260fb26c95c8d135da69cda7a167c5140affd19752",
    TESTS: "ccbfd62c77ce2073afd6bd1ecfcda972612c03beacbb7af99cf4684bc8bde3e7",
    CONTRACT: "e9352a4f2c1cf90acf16dcde54427b37659c24ae99fdd6fa0c5b4a23c7cb40fa",
    RELEASE: "2ee20e2a773ab3c778fa09758f052b16538795903e4830122a0db8f2c6f0e022",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dir(root: Path) -> None:
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    assert root.is_dir() and not root.is_symlink()
    assert outer.read_text().split() == [sha(sums), "SHA256SUMS"]
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        assert name not in listed and not Path(name).is_absolute() and ".." not in Path(name).parts
        listed[name] = digest
    actual: set[str] = set()
    for base_text, dirs, files in os.walk(root, followlinks=False):
        base = Path(base_text)
        dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            path = base / name
            rel = path.relative_to(root).as_posix()
            if rel not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} and not path.is_symlink():
                actual.add(rel)
    assert actual == set(listed)
    for name, digest in listed.items():
        assert sha(root / name) == digest


def main() -> int:
    checks = 0
    for path, digest in EXPECTED.items():
        assert path.is_file() and not path.is_symlink() and sha(path) == digest
        checks += 1

    spec = importlib.util.spec_from_file_location("m1265_exact", CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # Canonical source gates and their declared unit tests must pass without
    # executing the release runner.
    source = subprocess.run([str(module.PYTHON), "-I", str(CHECKER)],
                            text=True, capture_output=True, check=False)
    assert source.returncode == 0 and "PASS_M1265_SOURCE_ONLY" in source.stdout
    checks += 1
    tests = subprocess.run([str(module.PYTHON), "-I", str(TESTS)],
                           text=True, capture_output=True, check=False,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert tests.returncode == 0 and "Ran 6 tests" in tests.stderr and "OK" in tests.stderr
    checks += 1

    # Every technical corpus byte is fail-closed.  The stronger mutations also
    # target the requested gate deletion, duplicate tool launch, old-TB seepage,
    # and claim-inflation attacks.
    for path in module.EXPECTED:
        text = path.read_text(errors="replace")
        assert not module.exact_byte_gate(path, text + "X")
        checks += 1
    runner = RUNNER.read_text()
    runner_attacks = [
        runner.replace('[[ ! -e "${ATTEMPT}"', '[[ -e "${ATTEMPT}"', 1),
        runner.replace('"${VCS_BIN}" -full64', '"${VCS_BIN}" -full64\n"${VCS_BIN}" -full64', 1),
        runner.replace('./simv -no_save', './simv -no_save\n./simv -no_save', 1),
        runner.replace('/usr/bin/timeout --signal=TERM --kill-after=30s', '/bin/true', 1),
        runner.replace('seal_dir "${WORK}" || true', 'true', 1),
        runner.replace('functional_vcs_verified=false', 'functional_vcs_verified=true', 1),
        runner.replace('automatic_retry=false', 'automatic_retry=true', 1),
    ]
    for mutant in runner_attacks:
        assert mutant != runner and not module.exact_byte_gate(RUNNER, mutant)
        checks += 1
    filelist = FILELIST.read_text()
    for mutant in (filelist + str(HW / "verif_m1232r11_c1_common_charge_protocol/old_tb.sv") + "\n",
                   filelist.replace("m1258r12", "m1232r11", 1),
                   filelist + "\n"):
        assert not module.exact_byte_gate(FILELIST, mutant)
        checks += 1

    # External pins reject every absent, malformed, uppercase, or wrong-length
    # value before attempt consumption.
    names = ("M1265_EXPECTED_RELEASE_SHA256",
             "M1265_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
             "M1265_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
             "M1265_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256")
    good = {name: "a" * 64 for name in names}
    assert module.env_gate(good)
    checks += 1
    for name in names:
        for replacement in (None, "b" * 63, "B" * 64, "0" * 65, "not-a-sha"):
            bad = dict(good)
            if replacement is None:
                bad.pop(name)
            else:
                bad[name] = replacement
            assert not module.env_gate(bad)
            checks += 1

    # Byte changes to either contract invalidate the checked sidecar digest.
    for path in (CONTRACT, RELEASE):
        canonical = sha(path)
        for mutant in (path.read_bytes() + b"\n", path.read_bytes().replace(b"false", b"true", 1)):
            assert hashlib.sha256(mutant).hexdigest() != canonical
            checks += 1

    verify_dir(AUTHOR)
    verify_dir(REACH)
    checks += 2
    reach = json.loads((REACH / "mechanical_checks.json").read_text())
    assert reach["status"] == "PASS_READ_ONLY_REACHABILITY_AUDIT_NO_RELEASE_AUTHORIZATION"
    assert reach["score"] == 100 and reach["severity_counts"] == {"P0": 0, "P1": 0, "P2": 0}
    assert reach["tb"]["sha256"] == module.EXPECTED[module.TB]
    assert reach["counts"]["initial_blocks"] == 1
    assert reach["counts"]["parent_force_statements"] == 0
    assert reach["counts"]["random_iterations"] == 24
    checks += 5

    contract = json.loads(CONTRACT.read_text())
    release = json.loads(RELEASE.read_text())
    author = json.loads((AUTHOR / "review.json").read_text())
    assert contract["identity"]["runner_sha256"] == sha(RUNNER)
    assert contract["identity"]["filelist_sha256"] == sha(FILELIST)
    assert contract["identity"]["release_checker_sha256"] == sha(CHECKER)
    assert contract["identity"]["release_tests_sha256"] == sha(TESTS)
    assert release["identity"]["runner_sha256"] == sha(RUNNER)
    assert release["identity"]["source_contract_sha256"] == sha(CONTRACT)
    assert author["bindings"]["runner_sha256"] == sha(RUNNER)
    assert author["bindings"]["filelist_sha256"] == sha(FILELIST)
    assert author["bindings"]["checker_sha256"] == sha(CHECKER)
    assert author["bindings"]["tests_sha256"] == sha(TESTS)
    assert author["bindings"]["source_contract_sha256"] == sha(CONTRACT)
    assert author["bindings"]["release_sha256"] == sha(RELEASE)
    checks += 12
    for data in (contract, release, author):
        for key in ("functional_vcs_verified", "timing_verified", "cycles_measured",
                    "speedup", "ppa", "power", "energy", "system_speedup", "paper_citable"):
            assert data["claim_boundary"][key] is False
            checks += 1

    # Exact source topology: a single compile, single simulation, two timeouts,
    # pre-work attempt consumption, fresh-namespace gates, and quarantine seal.
    assert runner.count('"${VCS_BIN}" -full64') == 1
    assert runner.count('./simv -no_save') == 1
    assert runner.count('/usr/bin/timeout --signal=TERM --kill-after=30s') == 2
    assert runner.count('/bin/mkdir -- "${ATTEMPT}"') == 1
    assert runner.count('/bin/mkdir -- "${WORK}"') == 1
    assert 'compgen -G "${HW_ROOT}/results/.m1265_m1258r12_m1162_c1_common_charge_protocol_vcs_r12_work.*"' in runner
    assert 'compgen -G "${RESULT}.failed_or_incomplete.*"' in runner
    assert 'seal_dir "${WORK}" || true' in runner
    assert 'mv -- "${WORK}" "${QUARANTINE}" || true' in runner
    attempt_i = runner.index('/bin/mkdir -- "${ATTEMPT}"')
    work_i = runner.index('/bin/mkdir -- "${WORK}"')
    compile_i = runner.index('"${VCS_BIN}" -full64')
    sim_i = runner.index('./simv -no_save')
    assert attempt_i < work_i < compile_i < sim_i
    checks += 10

    attempt = HW / "results/.m1265_m1258r12_m1162_c1_common_charge_protocol_vcs_r12_attempt_consumed"
    result = HW / "results/m1265_m1258r12_m1162_c1_common_charge_protocol_unit_delay_vcs_r12_20260830"
    works = list((HW / "results").glob(".m1265_m1258r12_m1162_c1_common_charge_protocol_vcs_r12_work.*"))
    failures = list((HW / "results").glob("m1265_m1258r12_m1162_c1_common_charge_protocol_unit_delay_vcs_r12_20260830.failed_or_incomplete.*"))
    assert not os.path.lexists(attempt) and not os.path.lexists(result) and not works and not failures
    checks += 1

    # The source predates an unrelated M1266 audit and therefore hard-codes an
    # M1266-compatible alias.  The true fresh author is M1267.  Compatibility is
    # safe only if both sealed directories bind the same alias identity.
    assert 'RELEASE_HAMMER="${HW_ROOT}/reviews/m1266_m1265_c1_r12_exact_byte_vcs_release_independent_hammer_r1_20260830"' in runner
    assert release["future_hammer"] == {
        "schema": "m1266_m1265_c1_r12_exact_byte_vcs_release_independent_hammer_r1_v1",
        "status": "PASS_M1266_AUTHORIZE_ONE_M1265_R12_UNIT_DELAY_VCS_LAUNCH",
        "minimum_score": 95,
        "maximum_p0": 0,
        "maximum_p1": 0,
        "maximum_p2": 0,
        "fresh_different_author": True,
    }
    checks += 2

    out = {
        "schema": "m1267_m1265_c1_r12_exact_byte_release_hammer_mechanical_r1_v1",
        "status": "PASS_SOURCE_ONLY_HAMMER__AUTHORIZE_EXACTLY_ONE_FUTURE_M1265_VCS",
        "checks_passed": checks,
        "severity_counts": {"P0": 0, "P1": 0, "P2": 0},
        "runner_sha256": sha(RUNNER),
        "filelist_sha256": sha(FILELIST),
        "checker_sha256": sha(CHECKER),
        "tests_sha256": sha(TESTS),
        "source_contract_sha256": sha(CONTRACT),
        "release_sha256": sha(RELEASE),
        "tb_sha256": sha(module.TB),
        "m1265a_reachability_status": reach["status"],
        "m1265a_reachability_tb_sha256": reach["tb"]["sha256"],
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "gpu_runs": 0,
        "remote_actions": 0,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
