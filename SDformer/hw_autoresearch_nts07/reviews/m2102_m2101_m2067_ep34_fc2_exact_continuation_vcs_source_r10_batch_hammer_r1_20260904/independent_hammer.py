#!/opt/anaconda3/bin/python3.12
"""Read-only M2102 hammer for the M2101/R10 batch-VCS source.

This checker deliberately does not invoke lmstat, VCS, or any EDA tool.  A
source review is not an execution release.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m2067_ep34_fc2_exact_continuation_vcs_one_shot_codex_batch_r10_20260904.py"
PARSER = HW / "system_simulator/scripts/parse_m2067_ep34_fc2_exact_continuation_vcs_codex_batch_r10_20260904.py"
CONTRACT = HW / "contracts/m2101_m2067_ep34_fc2_exact_continuation_vcs_source_contract_r10_codex_batch_20260904.json"
FILELIST = HW / "dc_handoff/filelists/iscas_m2067_ep34_fc2_exact_continuation_vcs_codex_batch_r10_20260904.f"
TB = HW / "tb_m2018/tb_m2067_ep34_fc2_exact_continuation_s960_codex_batch_r10_20260904.sv"
R9_ATTEMPT = HW / "results/.m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_attempt_consumed"
R9_FAILURE = HW / "results/m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_20260904.failed_or_incomplete.quarantine"
R9_RESULT = HW / "results/m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_20260904"
R9_RUNNER = HW / "dc_handoff/scripts/run_m2067_ep34_fc2_exact_continuation_vcs_one_shot_codex_r9_ownerfix_20260904.py"
M2103_RELEASE = HW / "contracts/m2103_m2102_m2101_m2067_ep34_fc2_exact_continuation_vcs_r10_batch_launch_release_r1_20260904.json"


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def check_seal(root: Path) -> tuple[str, str]:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha256(manifest), "SHA256SUMS"],
         "outer seal")
    listed = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        relative = Path(fields[1].lstrip("*"))
        need(not relative.is_absolute() and ".." not in relative.parts,
             "unsafe manifest member")
        need(relative.as_posix() not in listed, "duplicate manifest member")
        need(sha256(root / relative) == fields[0], "member identity")
        listed[relative.as_posix()] = fields[0]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {
                  "SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(not any(p.is_symlink() for p in root.rglob("*")), "linked seal")
    need(actual == set(listed), "non-exhaustive seal")
    return sha256(manifest), sha256(outer)


def main() -> int:
    contract = strict_json(CONTRACT)
    contract_sha = sha256(CONTRACT)
    contract_line = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
    contract_outer = Path(str(contract_line) + ".seal.sha256")
    need(contract_line.read_text().split() == [contract_sha, CONTRACT.name],
         "contract seal")
    need(contract_outer.read_text().split() ==
         [sha256(contract_line), contract_line.name], "contract outer seal")
    inventory = {row["path"]: row["sha256"]
                 for row in contract["frozen_source_inventory"]}
    need(len(inventory) == len(contract["frozen_source_inventory"]),
         "duplicate inventory")
    for relative, digest in inventory.items():
        need(sha256(HW / relative) == digest, "inventory drift " + relative)

    spec = importlib.util.spec_from_file_location("m2101_r10_parser", PARSER)
    need(spec is not None and spec.loader is not None, "parser spec")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    static = parser.validate_source()
    need(static["workloads"] == 960, "workload count")
    metadata = strict_json(
        HW / "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.json")
    need(len(metadata["rows"]) == 960, "metadata rows")
    need(sum(row["integer_checks"] for row in metadata["rows"]) == 1843200,
         "integer checks")
    need(sum(row["expected_commits"] for row in metadata["rows"]) == 115200,
         "commits")
    need(sum(row["output_tiles"] * row["chunks"]
             for row in metadata["rows"]) == 13440, "row/chunk records")

    mutations = {}
    for label, pass_count, reset_count in (
            ("partial_pass", 959, 0), ("duplicate_pass", 961, 0),
            ("partial_reset", 960, 959)):
        with tempfile.NamedTemporaryFile("w", delete=False) as stream:
            path = Path(stream.name)
            for _ in range(pass_count):
                stream.write(parser.PASS_PREFIX + "workload_slot=0\n")
            for _ in range(reset_count):
                stream.write(parser.RESET_PREFIX + "workload_slot=0\n")
        try:
            parser.parse_log(path)
        except Exception as exc:  # expected fail-closed outcome
            mutations[label] = type(exc).__name__ + ": " + str(exc)
        else:
            raise RuntimeError("mutation accepted " + label)
        finally:
            path.unlink()

    failure_manifest, failure_outer = check_seal(R9_FAILURE)
    failure = strict_json(R9_FAILURE / "failure.json")
    owner = strict_json(R9_ATTEMPT / "owner.json")
    need(not R9_RESULT.exists(), "R9 success exists")
    need(failure["status"] == "FAILED_DO_NOT_CITE_NO_RETRY",
         "R9 failure status")
    need(failure["automatic_retry"] is False, "R9 retry")
    need(failure["owner_nonce"] == owner["nonce"], "R9 nonce")
    need(failure["runner_sha256"] == owner["runner_sha256"], "R9 runner")
    need(sha256(R9_RUNNER) ==
         "3423f358fd1b91f92058c1ab5aac2f15add72787650962af94f28e4841e9d2c4",
         "R9 source drift")
    need(not (Path("/proc") / str(owner["pid"])).exists(), "R9 owner live")

    runner_text = RUNNER.read_text()
    contract_gate = contract["predecessor_and_release_gate"]
    # M2102 is source review only.  A future, separately double-sealed M2103
    # must bind this exact review and own the one-shot production budget.
    review_only_authority = all(token in runner_text for token in (
        '"m2103_release_authoring": 1', '"vcs_execution": 0',
        '"license_queries": 0', '"automatic_retry": False'))
    release_gate = all(token in runner_text for token in (
        "M2103_RELEASE", "verify_double_sealed_file(",
        'authority("M2067_R10_EXPECTED_M2103_RELEASE_SHA256")',
        '"vcs_compiles": 1', '"simv_runs": 1',
        '"all_other_eda_runs": 0'))
    need(review_only_authority, "M2102 direct-execution authority")
    need(release_gate, "M2103 independent release gate")
    need(contract_gate.get(
        "m2102_must_authorize_only_m2103_release_authoring") is True,
         "contract source-review boundary")
    need(contract_gate.get("m2102_direct_vcs_execution_must_be_zero") is True,
         "contract direct VCS boundary")
    need(contract_gate.get("m2103_double_sealed_release_required") is True,
         "contract release seal boundary")
    need(not M2103_RELEASE.exists(), "M2103 prematurely authored")

    need(runner_text.count("run_checked(compile_command") == 1,
         "compile-call cardinality")
    need(runner_text.count('run_checked(RUN_STATE["command"], WORK, batch_log')
         == 1, "simv-call cardinality")
    need('"vcs_compiles_budget": 1, "simv_runs_budget": 1' in runner_text,
         "attempt budget")
    need('"release_sha256": release_sha' in runner_text,
         "attempt release binding")
    need('"m2103_release_sha256": release_sha' in runner_text,
         "result release binding")
    need('publish_no_replace(STAGE, RESULT)' in runner_text
         and 'publish_no_replace(FAIL_STAGE, FAILURE)' in runner_text,
         "no-replace publication")
    need('or not attempt_owned_by_this_process()' in runner_text,
         "failure owner boundary")
    need('"automatic_retry": False' in runner_text,
         "no-retry boundary")
    need("def verify_double_sealed_file(" in runner_text
         and "not sidecar.is_file() or sidecar.is_symlink()" in runner_text
         and "not outer.is_file() or outer.is_symlink()" in runner_text,
         "future release regular-file double seal")
    need('release.get("authorization") != {' in runner_text
         and 'release.get("claim_boundary") != {' in runner_text,
         "release semantic authorization/claim checks")
    need('"m2102_review_sha256": sha256(M2102 / "review.json")' in runner_text
         and '"m2102_manifest_sha256": sha256(M2102 / "SHA256SUMS")'
         in runner_text
         and '"r9_failure_json_sha256": sha256(R9_FAILURE / "failure.json")'
         in runner_text,
         "release identity source-review/predecessor binding")
    need(runner_text.count("verify_authority()") >= 6
         and "def run_checked(" in runner_text
         and "    verify_authority()\n    collision_gate()" in runner_text,
         "repeated authority verification")

    tb_text = TB.read_text()
    start = tb_text.index("task automatic prepare_workload_boundary;")
    end = tb_text.index("endtask", start)
    boundary = tb_text[start:end]
    for token in (
            "rst_core=1", "initialize_drives();", "alias_attacks=0",
            "row_chunk_records=0", "address_checks_base=0",
            "address_checks_tsbg=0", "workload_commits_base=0",
            "workload_commits_tsbg=0", "workload_checks_base=0",
            "workload_checks_tsbg=0", "total_base_cycles=0",
            "total_tsbg_cycles=0", "observed_base[context][slice]=0",
            "observed_tsbg[context][slice]=0",
            "expected[context][slice][lane]=0"):
        need(token in boundary, "workload reset state " + token)
    need("for (workload_slot=0; workload_slot<WORKLOADS;" in tb_text,
         "exact-order batch loop")
    need(tb_text.count("prepare_workload_boundary();") == 1,
         "boundary call source cardinality")

    result = {
        "status": "PASS_M2102_M2101_R10_BATCH_SOURCE_HAMMER__AUTHORIZE_M2103_RELEASE_ONLY",
        "static_parser_status": static["status"],
        "source_identity": {
            "runner_sha256": sha256(RUNNER),
            "parser_sha256": sha256(PARSER),
            "contract_sha256": contract_sha,
            "filelist_sha256": sha256(FILELIST),
            "tb_sha256": sha256(TB),
        },
        "cardinality": {
            "workloads": 960, "row_chunk_records": 13440,
            "integer_checks_per_axis": 1843200,
            "commits_per_axis": 115200,
        },
        "r9_failure": {
            "failure_json_sha256": sha256(R9_FAILURE / "failure.json"),
            "manifest_sha256": failure_manifest,
            "outer_file_sha256": failure_outer,
            "completed_slots": failure["completed_slots"],
            "simv_runs": failure["simv_runs"],
            "owner_pid_dead": True,
            "preserved_work_and_stage_tolerated": True,
        },
        "mutations": mutations,
        "authority_layering": {
            "m2102_direct_vcs_execution": 0,
            "m2102_license_queries": 0,
            "m2103_release_authoring": 1,
            "future_m2103_double_sealed_release_required": True,
            "pass": True,
        },
        "tools_launched": {"vcs": 0, "lmstat": 0, "eda": 0, "gpu": 0},
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
