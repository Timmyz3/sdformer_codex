#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author, local-only blind hammer for the M1497 C1 source.

This program never invokes VCS, simv, synthesis, STA, power, SSH, or GPU
tools.  Runtime-path attacks replace the tool call with deterministic local
mocks and publish only inside TemporaryDirectory instances.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "verif_m1497_c1_active_next_oracle_successor/check_m1497_source.py"
TESTS = HW / "verif_m1497_c1_active_next_oracle_successor/test_m1497_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1497_m1459_c1_active_next_oracle_clean_result_successor_one_shot.py"
CONTRACT = HW / "contracts/m1497_c1_active_next_oracle_clean_result_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1497_c1_active_next_oracle_clean_result_successor_source_author_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
UCLI = HW.parent / "ucli.key"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.argv)
    try:
        sys.argv = [str(path)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


C = load("m1498_bound_m1497_checker", CHECKER)
R = load("m1498_bound_m1497_runner", RUNNER)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


ATTACKS: list[dict[str, object]] = []


def attack(name: str, rejected: bool, detail: str) -> None:
    ATTACKS.append({"name": name, "rejected": bool(rejected), "detail": detail})


def expect_exception(name: str, function, detail: str) -> None:
    try:
        function()
    except BaseException as exc:
        attack(name, True, detail + "; rejected=" + type(exc).__name__)
    else:
        attack(name, False, detail + "; mutation was accepted")


def write_contract_fixture(root: Path, payload: dict) -> Path:
    path = root / CONTRACT.name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{sha(path)}  {path.name}\n")
    outer = Path(str(sidecar) + ".seal.sha256")
    outer.write_text(f"{sha(sidecar)}  {sidecar.name}\n")
    return path


def contract_mutation(name: str, mutator) -> None:
    with tempfile.TemporaryDirectory(prefix="m1498_contract_") as temp:
        payload = json.loads(CONTRACT.read_text())
        mutator(payload)
        candidate = write_contract_fixture(Path(temp), payload)
        with mock.patch.object(C, "CONTRACT", candidate):
            try:
                C.check_source(require_runtime_authority=False)
            except BaseException as exc:
                attack(name, True, "contract mutation rejected=" + type(exc).__name__)
            else:
                attack(name, False, "contract mutation accepted after valid sidecar reseal")


def oracle_attacks() -> None:
    base = dict(active=1, issue_valid=1, public_first=0, public_source=1,
                latched_first=0, latched_source=1, weight_accepted=0,
                psum_accepted=1, served_source=0)
    for field in tuple(base):
        row = dict(base); row[field] = None
        attack("oracle_unknown_" + field,
               not C.active_next_oracle(**row), "None models X/Z")
    wrong = (
        ("weight_accept_nonfirst", {"weight_accepted": 1}),
        ("psum_accept_nonfirst", {"psum_accepted": 0}),
        ("first_public_latched", {"public_first": 1}),
        ("source_public_latched", {"public_source": 2}),
        ("stale_source", {"served_source": 1}),
        ("missing_issue", {"issue_valid": 0}),
        ("active_bad", {"active": 2}),
        ("first_weight_accept", {"public_first": 1, "latched_first": 1,
                                 "psum_accepted": 0, "weight_accepted": 1}),
        ("first_wrong_psum", {"public_first": 1, "latched_first": 1,
                              "psum_accepted": 1}),
    )
    for name, changes in wrong:
        row = dict(base); row.update(changes)
        attack("oracle_" + name, not C.active_next_oracle(**row),
               "wrong active-next tuple")
    first = dict(base, public_first=1, latched_first=1, psum_accepted=0)
    attack("oracle_positive_nonfirst", C.active_next_oracle(**base),
           "exact weight=0, psum=!first non-first tuple")
    attack("oracle_positive_first", C.active_next_oracle(**first),
           "exact weight=0, psum=!first first tuple")
    idle = {key: None for key in base}; idle["active"] = 0
    attack("oracle_positive_idle", C.active_next_oracle(**idle),
           "inactive retired state ignores irrelevant X/Z")


def sole_delta_attacks() -> None:
    old = C.TB_R13.read_text()
    new = C.TB.read_text()
    attack("sole_delta_exact", old.count(C.OLD) == 1 and
           new == old.replace(C.OLD, C.NEW), "exact one-region replacement")
    with tempfile.TemporaryDirectory(prefix="m1498_tb_") as temp:
        temp = Path(temp)
        bad_tb = temp / "tb.sv"
        bad_tb.write_text(new + "\n// mutation\n")
        with mock.patch.object(C, "TB", bad_tb):
            expect_exception("tb_extra_delta", C.check_source,
                             "extra TB byte must fail")
        bad_r13 = temp / "r13.sv"
        bad_r13.write_text(old + "\n// mutation\n")
        with mock.patch.object(C, "TB_R13", bad_r13):
            expect_exception("r13_byte_drift", C.check_source,
                             "frozen R13 byte mutation must fail")
        bad_filelist = temp / "filelist.f"
        bad_filelist.write_text(C.FILELIST.read_text() + "/tmp/extra.sv\n")
        with mock.patch.object(C, "FILELIST", bad_filelist):
            expect_exception("filelist_extra_member", C.check_source,
                             "eighth filelist member must fail")


def contract_attacks() -> None:
    contract_mutation("contract_identity_runner_sha", lambda p:
                      p["identity"].__setitem__("runner_sha256", "0" * 64))
    contract_mutation("contract_identity_wrapper_omitted", lambda p:
                      p["identity"].__setitem__("frozen_r13_path", "wrong.sv"))
    contract_mutation("contract_oracle_weight_accept", lambda p:
                      p["oracle_contract"].__setitem__("active_next_weight_accepted", 1))
    contract_mutation("contract_result_hygiene_raw", lambda p:
                      p["result_hygiene"].__setitem__("raw_build_is_never_sealed_or_published", False))
    contract_mutation("contract_future_vcs_two", lambda p:
                      p["future_authority"].__setitem__("maximum_future_vcs_compiles", 2))
    contract_mutation("contract_future_retry", lambda p:
                      p["future_authority"].__setitem__("automatic_retry", True))
    contract_mutation("contract_authorize_simv", lambda p:
                      p["authorization"].__setitem__("simv_runs", 1))
    contract_mutation("contract_authorize_other_eda", lambda p:
                      p["authorization"].__setitem__("all_other_eda_runs", 1))
    contract_mutation("contract_authorize_retry", lambda p:
                      p["authorization"].__setitem__("automatic_retry", True))
    contract_mutation("contract_extra_top_level", lambda p:
                      p.__setitem__("unreviewed", True))


def runtime_identity_attack() -> None:
    called: set[Path] = set()
    real_exact = R.exact
    fake_release = {
        "status": R.RELEASE_STATUS,
        "authorization": copy.deepcopy(R.AUTHORIZATION),
        "claim_boundary": copy.deepcopy(R.CLAIMS),
    }
    author = {"status": R.AUTHOR_STATUS,
              "claim_boundary": copy.deepcopy(R.CLAIMS)}
    hammer = {"status": R.HAMMER_STATUS,
              "claim_boundary": copy.deepcopy(R.CLAIMS)}
    final = {"status": R.FINAL_STATUS,
             "claim_boundary": copy.deepcopy(R.CLAIMS)}

    def exact_spy(path: Path, digest: str) -> None:
        path = Path(path); called.add(path)
        if path == R.RELEASE:
            return
        real_exact(path, digest)

    def authority(root: Path, *_args):
        if root == R.AUTHOR:
            return author
        if root == R.HAMMER:
            return hammer
        if root == R.FINAL:
            return final
        raise RuntimeError("unexpected authority")

    def strict(path: Path):
        return json.loads(CONTRACT.read_text()) if path == R.CONTRACT else fake_release

    env = {name: "1" * 64 for name in R.ENV_PINS}
    env["M1497_EXPECTED_RUNNER_SHA256"] = sha(R.RUNNER)
    env["M1497_EXPECTED_CONTRACT_SHA256"] = sha(R.CONTRACT)
    with mock.patch.dict(os.environ, env, clear=False), \
            mock.patch.object(R, "exact", side_effect=exact_spy), \
            mock.patch.object(R.P, "verify_file_sidecar", return_value=None), \
            mock.patch.object(R.P, "verify_authority", side_effect=authority), \
            mock.patch.object(R, "strict_json", side_effect=strict):
        R.validate_authority()

    targets = {
        "parent_rtl": R.P.BASE.PARENT,
        "m935_rtl": R.P.BASE.M935,
        "wrapper_rtl": R.P.BASE.WRAPPER,
        "sva": R.P.BASE.SVA,
        "witness": R.P.BASE.WITNESS,
        "foundry_model": R.P.BASE.FOUNDRY,
        "source_checker": R.CHECKER,
        "source_tests": R.TESTS,
        "vcs_binary": R.P.BASE.VCS,
    }
    for label, path in targets.items():
        attack("runtime_unpinned_" + label, Path(path) in called,
               "mutation must force an exact read; exact_called=" +
               str(Path(path) in called))


def run_main_with_fake_log(log_text: str) -> tuple[int, dict]:
    with tempfile.TemporaryDirectory(prefix="m1498_main_") as temp_name:
        temp = Path(temp_name)
        paths = {
            "ATTEMPT": temp / "attempt",
            "RESULT": temp / "result",
            "QUARANTINE": temp / "quarantine",
            "RAW_BUILD": temp / "raw",
            "CLEAN_RESULT_STAGE": temp / "clean",
            "ATTEMPT_STAGE": temp / "attempt_stage",
            "FAILURE_STAGE": temp / "failure_stage",
        }

        def fake_run_tool(_command, log: Path, _timeout, _environment):
            if log.name == "compile.log":
                log.write_text("compile mocked; no EDA invoked\n")
                simv = paths["RAW_BUILD"] / "simv"
                simv.write_text("mock executable; never run\n")
                simv.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            else:
                log.write_text(log_text)
            return 0

        completed = SimpleNamespace(returncode=0, stderr="", stdout="source gate mocked")
        with ExitStack() as stack:
            for name, value in paths.items():
                stack.enter_context(mock.patch.object(R, name, value))
            stack.enter_context(mock.patch.object(R, "validate_authority", return_value=None))
            stack.enter_context(mock.patch.object(R.subprocess, "run", return_value=completed))
            stack.enter_context(mock.patch.object(R, "namespace_gate", return_value=None))
            stack.enter_context(mock.patch.object(R.P.BASE, "collision_gate", return_value=None))
            stack.enter_context(mock.patch.object(R.P.BASE, "resource_gate", return_value=None))
            stack.enter_context(mock.patch.object(R, "run_tool", side_effect=fake_run_tool))
            stack.enter_context(mock.patch.object(
                R, "publish_no_replace", side_effect=lambda source, destination:
                os.rename(source, destination)))
            rc = R.main()
        receipt = json.loads((paths["RESULT"] /
            "m1497_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json").read_text())
        return rc, receipt


def log_acceptance_attacks() -> None:
    tokens = R.P.BASE.R13_PASS + "\n" + R.P.BASE.R15_PASS + "\n"
    rc, receipt = run_main_with_fake_log(tokens)
    accepted = rc == 0 and receipt["status"].startswith("PASS_")
    attack("runtime_missing_phase_coverage_accepted", not accepted,
           "only two PASS tokens supplied; accepted=" + str(accepted))
    rc, receipt = run_main_with_fake_log(
        tokens + "Error: assertion failure deliberately injected\n")
    accepted = rc == 0 and receipt["status"].startswith("PASS_")
    attack("runtime_assertion_error_line_accepted", not accepted,
           "PASS tokens plus Error/assertion supplied; accepted=" + str(accepted))


def pretry_failure_attack() -> None:
    with tempfile.TemporaryDirectory(prefix="m1498_pretry_") as temp_name:
        temp = Path(temp_name)
        raw = temp / "raw"
        raw.mkdir()
        paths = {
            "ATTEMPT": temp / "attempt",
            "RESULT": temp / "result",
            "QUARANTINE": temp / "quarantine",
            "RAW_BUILD": raw,
            "CLEAN_RESULT_STAGE": temp / "clean",
            "ATTEMPT_STAGE": temp / "attempt_stage",
            "FAILURE_STAGE": temp / "failure_stage",
        }
        completed = SimpleNamespace(returncode=0, stderr="", stdout="")
        caught = False
        with ExitStack() as stack:
            for name, value in paths.items():
                stack.enter_context(mock.patch.object(R, name, value))
            stack.enter_context(mock.patch.object(R, "validate_authority", return_value=None))
            stack.enter_context(mock.patch.object(R.subprocess, "run", return_value=completed))
            stack.enter_context(mock.patch.object(R, "namespace_gate", return_value=None))
            stack.enter_context(mock.patch.object(R.P.BASE, "collision_gate", return_value=None))
            stack.enter_context(mock.patch.object(R.P.BASE, "resource_gate", return_value=None))
            stack.enter_context(mock.patch.object(
                R, "publish_no_replace", side_effect=lambda source, destination:
                os.rename(source, destination)))
            try:
                R.main()
            except FileExistsError:
                caught = True
        attempt_consumed = paths["ATTEMPT"].is_dir()
        quarantine = paths["QUARANTINE"].exists()
        rejected = not (caught and attempt_consumed and not quarantine)
        attack("attempt_consumed_pretry_failure_unsealed", rejected,
               f"caught={caught}, attempt={attempt_consumed}, quarantine={quarantine}")


def result_hygiene_attacks() -> None:
    with tempfile.TemporaryDirectory(prefix="m1498_clean_") as temp_name:
        root = Path(temp_name) / "clean"
        root.mkdir()
        for name in R.CLEAN_PAYLOAD:
            (root / name).write_text("regular\n")
        target = root / "compile.log"
        target.unlink()
        target.symlink_to(root / "sim.log")
        expect_exception("clean_result_symlink", lambda: R.seal_clean_result(root),
                         "clean stage must reject a symlink")


def seal_authority_baseline() -> dict[str, object]:
    review_sha = sha(AUTHOR / "review.json")
    manifest_sha = sha(AUTHOR / "SHA256SUMS")
    outer_sha = sha(AUTHOR / "SHA256SUMS.seal.sha256")
    review = R.P.verify_authority(AUTHOR, review_sha, manifest_sha, outer_sha)
    return {"status": review["status"], "review_sha256": review_sha,
            "manifest_sha256": manifest_sha, "outer_file_sha256": outer_sha}


def main() -> int:
    docs_before = sha(DOCS359)
    ucli_before = sha(UCLI)
    author_run = subprocess.run([str(PYTHON), "-I", str(TESTS)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=120, check=False)
    checker_run = subprocess.run([str(PYTHON), "-I", str(CHECKER),
                                  "--mode", "source_only"],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=120, check=False)
    baseline = {
        "author_tests_returncode": author_run.returncode,
        "author_tests_output": author_run.stdout,
        "source_checker_returncode": checker_run.returncode,
        "source_checker_output": checker_run.stdout,
        "author_seal": seal_authority_baseline(),
    }
    oracle_attacks()
    sole_delta_attacks()
    contract_attacks()
    runtime_identity_attack()
    log_acceptance_attacks()
    pretry_failure_attack()
    result_hygiene_attacks()
    false_negatives = [row for row in ATTACKS if not row["rejected"]]
    output = {
        "schema": "m1498_m1497_c1_active_next_oracle_source_blind_hammer_output_r1_v1",
        "status": ("PASS_ZERO_FALSE_NEGATIVES" if not false_negatives
                   else "FAIL_FALSE_NEGATIVES__DO_NOT_AUTHOR_RELEASE"),
        "commit_under_review": "6af1e2d1",
        "baseline": baseline,
        "attacks": ATTACKS,
        "attack_count": len(ATTACKS),
        "rejected_count": len(ATTACKS) - len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_negative_names": [row["name"] for row in false_negatives],
        "eda_invocations": 0,
        "ssh_invocations": 0,
        "gpu_invocations": 0,
        "docs359_sha256_before": docs_before,
        "docs359_sha256_after": sha(DOCS359),
        "ucli_key_sha256_before": ucli_before,
        "ucli_key_sha256_after": sha(UCLI),
    }
    encoded = json.dumps(output, indent=2, sort_keys=True) + "\n"
    (HERE / "hammer_output.json").write_text(encoded)
    print(encoded, end="")
    return 0 if not false_negatives else 2


if __name__ == "__main__":
    raise SystemExit(main())
