#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh source-only hammer for M1354 C1/R16 one-shot VCS release source."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CHECKER = HW / "verif_m1354_c1_r16_vcs_release/check_m1354_c1_r16_vcs_release_source.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("m1355_bound_m1354_checker", CHECKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("checker import spec failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_checker()
RUNNER_TEXT = M.RUNNER.read_text()
CONTRACT = json.loads(M.CONTRACT.read_text())


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) < 1:
        raise AssertionError(f"mutation anchor absent old={old!r}")
    return text.replace(old, new, 1)


def main() -> int:
    outcomes: list[dict[str, object]] = []

    def record(name: str, category: str, thunk) -> None:
        rejected = False
        message = ""
        try:
            accepted = bool(thunk())
            rejected = not accepted
            if accepted:
                message = "mutation accepted"
        except Exception as exc:
            rejected = True
            message = f"{type(exc).__name__}: {exc}"
        outcomes.append({"attack": name, "category": category,
                         "rejected": rejected, "message": message})

    def exact_runner(name: str, old: str, new: str) -> None:
        mutant = replace_once(RUNNER_TEXT, old, new).encode()
        record(name, "runner_exact_byte_and_protocol", lambda: M.exact_byte_gate(M.RUNNER, mutant))

    # Runner semantics: all must be rejected by the frozen exact-byte authority.
    for args in (
        ("runner_drop_pipefail", "set -euo pipefail", "set -eu"),
        ("runner_relax_umask", "umask 077", "umask 022"),
        ("runner_accept_arguments", "[[ $# -eq 0 ]]", "[[ $# -le 1 ]]"),
        ("runner_release_env_rename", "M1354_EXPECTED_RELEASE_SHA256", "M1354_RELEASE_SHA256"),
        ("runner_hammer_review_env_rename", "M1354_EXPECTED_HAMMER_REVIEW_SHA256", "M1354_HAMMER_REVIEW_SHA256"),
        ("runner_hammer_manifest_env_rename", "M1354_EXPECTED_HAMMER_MANIFEST_SHA256", "M1354_HAMMER_MANIFEST_SHA256"),
        ("runner_hammer_outer_env_rename", "M1354_EXPECTED_HAMMER_OUTER_SEAL_FILE_SHA256", "M1354_HAMMER_OUTER_SHA256"),
        ("runner_env_uppercase_digest", "^[0-9a-f]{64}$", "^[0-9A-Fa-f]{64}$"),
        ("runner_compile_timeout_unbounded", "COMPILE_TIMEOUT_SECONDS=1200", "COMPILE_TIMEOUT_SECONDS=0"),
        ("runner_sim_timeout_unbounded", "SIM_TIMEOUT_SECONDS=1800", "SIM_TIMEOUT_SECONDS=0"),
        ("runner_compile_timeout_call_removed", "/usr/bin/timeout --signal=TERM --kill-after=30s \"${COMPILE_TIMEOUT_SECONDS}s\"", ""),
        ("runner_sim_timeout_call_removed", "/usr/bin/timeout --signal=TERM --kill-after=30s \"${SIM_TIMEOUT_SECONDS}s\"", ""),
        ("runner_compile_call_duplicated", '"${VCS_BIN}" -full64', '"${VCS_BIN}" -full64\n"${VCS_BIN}" -full64'),
        ("runner_sim_call_duplicated", "./simv -no_save", "./simv -no_save\n./simv -no_save"),
        ("runner_attempt_marker_removed", '/bin/mkdir -- "${ATTEMPT}"', ':'),
        ("runner_attempt_after_license", 'export VCS_HOME="/opt/synopsys/vcs/V-2023.12-SP1"', 'export VCS_HOME="/opt/synopsys/vcs/V-2023.12-SP1"\n/bin/mkdir -- "${ATTEMPT}"'),
        ("runner_eda_collision_drop_vcs", "blocked={'vcs','vcs1','simv'", "blocked={'vcs1','simv'"),
        ("runner_memory_gate_lowered", '"${mem_kib}" -ge 67108864', '"${mem_kib}" -ge 1'),
        ("runner_quarantine_disabled", "trap on_exit EXIT", "# trap disabled"),
        ("runner_failure_seal_disabled", 'seal_dir "${WORK}" || true', ':'),
        ("runner_r13_pass_token_weakened", 'R13_PASS="PASS_M1270R13_REAL_M935_INTEGRATED_PROTOCOL_SOURCE_CANDIDATE', 'R13_PASS="PASS_M1270R13'),
        ("runner_r15_pass_token_weakened", 'R15_PASS="PASS_M1337R15_REAL_M935_RUNTIME_WITNESS', 'R15_PASS="PASS_M1337R15'),
        ("runner_accept_zero_r13_pass", 'rg -Fxc "${R13_PASS}"', 'rg -Fxc "DOES_NOT_EXIST"'),
        ("runner_accept_zero_r15_pass", 'rg -Fxc "${R15_PASS}"', 'rg -Fxc "DOES_NOT_EXIST"'),
        ("runner_retry_claim_true", "automatic_retry=false", "automatic_retry=true"),
        ("runner_headline_claim_true", "headline=false", "headline=true"),
    ):
        exact_runner(*args)

    # Filelist and exact seven technical members.
    filelist = M.FILELIST.read_bytes()
    record("filelist_append_member", "filelist_and_seven_members",
           lambda: M.exact_byte_gate(M.FILELIST, filelist + b"/tmp/injected.sv\n"))
    lines = filelist.splitlines(keepends=True)
    record("filelist_reorder_members", "filelist_and_seven_members",
           lambda: M.exact_byte_gate(M.FILELIST, b"".join(lines[1:2] + lines[0:1] + lines[2:])))
    record("filelist_drop_member", "filelist_and_seven_members",
           lambda: M.exact_byte_gate(M.FILELIST, b"".join(lines[:-1])))
    for path in (M.FOUNDRY, M.PARENT, M.M935, M.WRAPPER, M.SVA, M.TB, M.WITNESS):
        record("exact_member_mutate_" + path.name, "filelist_and_seven_members",
               lambda path=path: M.exact_byte_gate(path, path.read_bytes() + b"\nM1355_MUTATION"))

    # Four external digest pins: missing, malformed and uppercase variants.
    env_names = (
        "M1354_EXPECTED_RELEASE_SHA256",
        "M1354_EXPECTED_HAMMER_REVIEW_SHA256",
        "M1354_EXPECTED_HAMMER_MANIFEST_SHA256",
        "M1354_EXPECTED_HAMMER_OUTER_SEAL_FILE_SHA256",
    )
    good_env = {name: "a" * 64 for name in env_names}
    for name in env_names:
        for suffix, value in (("missing", None), ("short", "a" * 63),
                              ("uppercase", "A" * 64)):
            mutant = dict(good_env)
            if value is None:
                mutant.pop(name)
            else:
                mutant[name] = value
            record(f"env_{name}_{suffix}", "four_env_pins",
                   lambda mutant=mutant: M.env_gate(mutant))

    # Seal attacks against copies only; never touch the authoritative trees.
    with tempfile.TemporaryDirectory(prefix="m1355_seals_") as temporary:
        temp = Path(temporary)
        seal_cases = []
        for index, (root, pins) in enumerate(M.R16_SEALS.items()):
            base = temp / f"base_{index}"
            shutil.copytree(root, base)
            seal_cases.append((root.name + "_review_tamper", base, pins, "review"))
            extra = temp / f"extra_{index}"; shutil.copytree(root, extra)
            (extra / "UNLISTED.txt").write_text("x")
            seal_cases.append((root.name + "_extra_member", extra, pins, "extra"))
            outer = temp / f"outer_{index}"; shutil.copytree(root, outer)
            (outer / "SHA256SUMS.seal.sha256").write_text("0" * 64 + "  SHA256SUMS\n")
            seal_cases.append((root.name + "_outer_semantics", outer, pins, "outer"))
        for name, copied, pins, kind in seal_cases:
            if kind == "review":
                review = copied / "review.json"; review.write_bytes(review.read_bytes() + b"\n")
            record(name, "r16_author_and_blind_seals",
                   lambda copied=copied, pins=pins: (M.verify_dir(copied, pins), True)[1])

    # Contract mutations exercise exact set/value semantics directly.
    def contract_attack(name: str, mutate) -> None:
        candidate = copy.deepcopy(CONTRACT)
        mutate(candidate)
        record(name, "contract_exact_set_value", lambda: (M.check_contract_dict(candidate), True)[1])

    contract_attack("contract_extra_top_level", lambda c: c.__setitem__("unexpected", True))
    contract_attack("contract_date_changed", lambda c: c.__setitem__("date", "2099-01-01"))
    contract_attack("contract_future_execution_removed", lambda c: c.pop("future_execution"))
    contract_attack("contract_future_execution_extra", lambda c: c["future_execution"].__setitem__("retry_alias", True))
    for key, value in (
        ("maximum_vcs_compiles", 2), ("maximum_simv_runs", 2),
        ("all_other_eda_runs", 1), ("compile_timeout_seconds", 1),
        ("simulation_timeout_seconds", 1), ("attempt_consumed_before_tool", False),
        ("fresh_attempt_namespace", "results/fake_attempt"),
        ("fresh_result_namespace", "results/fake_result"),
        ("failure_quarantine_recursive_seal", False), ("automatic_retry", True),
    ):
        contract_attack("future_execution_" + key,
                        lambda c, key=key, value=value: c["future_execution"].__setitem__(key, value))
    contract_attack("author_execution_extra", lambda c: c["author_execution"].__setitem__("retry", True))
    contract_attack("claim_boundary_extra", lambda c: c["claim_boundary"].__setitem__("paper_claim", True))
    contract_attack("identity_extra", lambda c: c["identity"].__setitem__("extra_sha256", "0" * 64))
    contract_attack("r16_authority_extra", lambda c: c["r16_authority"].__setitem__("extra", True))
    contract_attack("future_release_extra", lambda c: c["future_release"].__setitem__("extra", True))
    for key in ("release", "vcs", "simv", "dc", "pt", "ptpx", "eda", "gpu", "remote"):
        contract_attack("author_execution_true_" + key,
                        lambda c, key=key: c["author_execution"].__setitem__(key, True))
    for key in ("functional_vcs", "timing_verified", "cycles_measured", "speedup",
                "ppa", "power", "energy", "system_speedup", "headline"):
        contract_attack("claim_true_" + key,
                        lambda c, key=key: c["claim_boundary"].__setitem__(key, True))
    for key, value in (("launch_authorized", True), ("vcs_compiles_now", 1),
                       ("simv_runs_now", 1), ("automatic_retry", True)):
        contract_attack("future_release_" + key,
                        lambda c, key=key, value=value: c["future_release"].__setitem__(key, value))

    false_negatives = [str(row["attack"]) for row in outcomes if not row["rejected"]]
    categories: dict[str, dict[str, int]] = {}
    for row in outcomes:
        item = categories.setdefault(str(row["category"]), {"attacks": 0, "rejected": 0, "false_negatives": 0})
        item["attacks"] += 1
        item["rejected" if row["rejected"] else "false_negatives"] += 1
    output = {
        "schema": "m1355_m1354_c1_r16_vcs_release_blind_hammer_r1_v1",
        "status": "PASS" if not false_negatives else "FAIL_DO_NOT_CITE",
        "attack_count": len(outcomes),
        "rejected_count": len(outcomes) - len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_negatives": false_negatives,
        "categories": categories,
        "vcs_runs": 0,
        "simv_runs": 0,
        "eda_runs": 0,
        "release_created": False,
        "outcomes": outcomes,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not false_negatives else 1


if __name__ == "__main__":
    raise SystemExit(main())
