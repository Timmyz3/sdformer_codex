#!/usr/bin/python3.12
"""Read-only M2200 failure hammer; runs no VCS, license query, or EDA."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
HERE = Path(__file__).resolve().parent
Q = HW / "results/m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904.failed_or_incomplete.3417494.quarantine"
ATTEMPT = HW / "results/.m2199_m2197_selective_bank_fill_vcs_attempt_consumed"
CANONICAL = HW / "results/m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904"
LOCK = HW / "results/.m2199_m2197_selective_bank_fill_vcs_launch_lock"
RUNNER = HW / "dc_handoff/scripts/run_m2199_m2198_m2197_selective_bank_fill_directed_vcs_one_shot.sh"
PARSER = HW / "system_simulator/scripts/parse_m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs.py"
RTL = HW / "rtl_m2193/m2193_c2_tsbg_b4_selective_bank_fill_frontend.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
SVA = HW / "verif_m2197/m2197_c2_tsbg_selective_bank_fill_assertions.sv"
TB = HW / "tb_m2197/tb_m2197_c2_tsbg_selective_bank_fill_directed.sv"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2197_c2_tsbg_selective_bank_fill_directed_vcs.f"
CONTRACT = HW / "contracts/m2197_c2_tsbg_selective_bank_fill_source_contract_r1_20260904.json"
M2198 = HW / "reviews/m2198_m2197_c2_tsbg_selective_bank_fill_source_hammer_r1_20260904"
PYTHON = Path("/opt/anaconda3/bin/python3.12")
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    RUNNER: "745da777421e5601776f1caf158f4905fdbe8c82f6c0095c118d7b2d98ceb3fb",
    PARSER: "fde65c8372c9eab82ae49caea03137cdd93d0bd996fe65e9549220869a743571",
    RTL: "f651ea3a3b4dfab04d021a1e44797e7ab72c244cb7edf7496e18ac1ac033339e",
    M803: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    SVA: "8003115edb919e9c5c6c9c36ce4ba75dfb37d9ec9f23e7c4cf59e2aed3b461b4",
    TB: "a8a954826324aa20443e7b2acbbc6a0b1b2a92f83ebdd84bfdbb0879920526e3",
    FILELIST: "5beddf477b6938b599cfab962eba60f6d79dceeb825380f2e5cdc6f22b49dc13",
    CONTRACT: "01aa9873330dddbc837929032bee18b89320a601a0ac491680d64339454577ed",
    PYTHON: "873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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
    need(not any(path.is_symlink() for path in directory.rglob("*")), "symlink in seal")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    listed: set[str] = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe seal path")
        need((directory / rel).is_file() and sha(directory / rel) == digest,
             "member mismatch: " + rel.as_posix())
        listed.add(rel.as_posix())
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(listed == actual, "non-exhaustive seal")
    return {"member_count": len(listed), "manifest_sha256": sha(manifest),
            "outer_sha256": sha(outer), "exhaustive": True}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "module load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_unsealed_quarantine() -> dict[str, object]:
    files = []
    links = []
    for path in sorted(Q.rglob("*")):
        rel = path.relative_to(Q).as_posix()
        if path.is_symlink():
            links.append({"path": rel, "target": os.readlink(path),
                          "mode": oct(stat.S_IMODE(path.lstat().st_mode))})
        elif path.is_file():
            files.append({"path": rel, "sha256": sha(path),
                          "size_bytes": path.stat().st_size,
                          "mode": oct(stat.S_IMODE(path.stat().st_mode))})
    snapshot = {
        "schema": "m2200_m2199_unsealed_quarantine_read_only_snapshot_r1_v1",
        "root": Q.name, "regular_file_count": len(files),
        "symlink_count": len(links), "files": files, "symlinks": links,
        "original_manifest_present": (Q / "SHA256SUMS").exists(),
        "original_outer_seal_present": (Q / "SHA256SUMS.seal.sha256").exists(),
        "quarantine_modified": False,
    }
    (HERE / "quarantine_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    return snapshot


def main() -> int:
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "identity drift: " + str(path))
    need(stat.S_IMODE(PARSER.stat().st_mode) == 0o664 and not os.access(PARSER, os.X_OK),
         "parser mode is not exact non-executable 0644/0664 class")
    need(stat.S_IMODE(PYTHON.stat().st_mode) == 0o755 and os.access(PYTHON, os.X_OK),
         "fixed Python mode")
    source_review_seal = verify_seal(M2198)
    attempt_seal = verify_seal(ATTEMPT)
    review = read_json(M2198 / "review.json")
    need(review["status"] ==
         "PASS_M2198_M2197_SOURCE_HAMMER__M2199_ONE_SHOT_VCS_AUTHORIZED" and
         review["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0},
         "M2198 source authority")
    for key, path in (("runner_sha256", RUNNER), ("filelist_sha256", FILELIST),
                      ("rtl_sha256", RTL), ("m803_sha256", M803),
                      ("sva_sha256", SVA), ("tb_sha256", TB),
                      ("parser_sha256", PARSER), ("contract_sha256", CONTRACT)):
        need(review["identity"][key] == sha(path), "M2198 identity: " + key)
    need(ATTEMPT.is_dir() and
         (ATTEMPT / "ATTEMPT_CONSUMED.txt").read_text().splitlines() == [
             "status=M2199_ATTEMPT_CONSUMED", "license_queries=1",
             "vcs_compiles=1", "simv_runs=1", "retry=false"],
         "M2199 consumed budget")
    matches = sorted(HW.glob(
        "results/m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904.failed_or_incomplete.*.quarantine"))
    need(matches == [Q] and not CANONICAL.exists() and not LOCK.exists(), "M2199 census")
    need(not list((HW / "results").glob(".m2199_m2197_selective_bank_fill_vcs_work.*")),
         "M2199 work census")

    snapshot = snapshot_unsealed_quarantine()
    need(snapshot["regular_file_count"] == 92 and snapshot["symlink_count"] == 2 and
         snapshot["original_manifest_present"] is False and
         snapshot["original_outer_seal_present"] is False,
         "unsealed quarantine fingerprint")
    need((Q / "RUN_FAILED_OR_INCOMPLETE.txt").read_text().splitlines() == [
        "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE", "exit_code=126", "retry=false"],
         "failure status")
    need((Q / "simv.rc").read_text().strip() == "0" and
         (Q / "parser.log").stat().st_size == 0 and not (Q / "receipt.json").exists(),
         "parser launch boundary")
    runner_text = RUNNER.read_text()
    direct = '"${PARSER}" --sim-log "${WORK}/simv.log" --compile-log "${WORK}/vcs_compile.log"'
    need(runner_text.count(direct) == 1 and
         "/opt/anaconda3/bin/python3.12" not in runner_text and
         "chmod" not in runner_text, "direct parser invocation fingerprint")
    need("seal_dir \"${WORK}\" || true" in runner_text,
         "failure-seal best-effort fingerprint")
    compile_log = (Q / "vcs_compile.log").read_text(errors="replace")
    sim_log = (Q / "simv.log").read_text(errors="replace")
    need("All of 7 modules done" in compile_log and "CPU time:" in compile_log and
         "Error-" not in compile_log, "VCS compile diagnostic")
    need("PASS_M2197_C2_TSBG_SELECTIVE_BANK_FILL_DIRECTED" in sim_log and
         "$finish at simulation time" in sim_log and "Assertion failed" not in sim_log and
         "$fatal" not in sim_log, "simulation diagnostic")
    parser = load(PARSER, "m2199_parser_read_only_for_m2200")
    parsed = parser.parse(Q / "simv.log", Q / "vcs_compile.log", Q / "simv.rc")
    need(parsed["status"] ==
         "RAW_PASS_M2199_M2197_DIRECTED_VCS_PENDING_M2200_RESULT_HAMMER",
         "fixed-interpreter diagnostic parse")
    need(parsed["ledger"] == {"bundles": 3, "commits_ordinary": 72,
         "commits_tsbg": 72, "identity_checks_ordinary": 72,
         "identity_checks_tsbg": 72, "refill_banks_ordinary": 588,
         "refill_banks_tsbg": 156, "scalar_requests_ordinary": 588,
         "scalar_requests_tsbg": 156, "products_each": 3264}, "ledger")
    need(all((not isinstance(value, int)) or value > 0
             for key, value in parsed["coverage"].items()
             if key not in {"golden_commit_tag_context_slice_terminal_checks_per_mode",
                            "exact_acc24_commits_per_mode"}), "coverage nonzero")

    result = {
        "schema": "m2200_m2199_m2197_selective_bank_fill_failure_mechanical_checks_r1_v1",
        "status": "PASS_M2200_FAILURE_ROOT_CAUSE_MECHANICAL_CHECKS",
        "source_review_seal": source_review_seal,
        "attempt_seal": attempt_seal,
        "quarantine": {"exhaustively_double_sealed": False,
                       "manifest_present": False, "outer_seal_present": False,
                       "regular_file_count": snapshot["regular_file_count"],
                       "symlink_count": snapshot["symlink_count"],
                       "independent_snapshot_sha256": sha(HERE / "quarantine_snapshot.json")},
        "root_cause": {"exit_code": 126, "parser_mode_octal": "0664",
                       "parser_executable": False, "runner_direct_exec": True,
                       "parser_started": False, "parser_log_bytes": 0,
                       "classification": "DIRECT_EXEC_OF_NON_EXECUTABLE_PARSER"},
        "diagnostic_execution": {"license_query_completed": True,
                                 "vcs_compile_completed": True,
                                 "simv_completed": True, "simv_rc": 0,
                                 "parser_read_only_fixed_python_parse": "PASS",
                                 "ledger": parsed["ledger"],
                                 "coverage": parsed["coverage"]},
        "fixed_python": {"path": str(PYTHON), "sha256": sha(PYTHON),
                         "mode_octal": "0755", "regular_executable": True},
        "census": {"attempt_count": 1, "quarantine_count": 1,
                   "canonical_result_absent": True, "lock_absent": True,
                   "work_absent": True},
        "claim_boundary": {"m2199_admitted": False, "m2199_retry": False,
                           "m2199_paper_citable": False,
                           "diagnostic_functional_pass_only": True,
                           "rtl_verification_claim": False,
                           "component_speedup": False, "system_speedup": False,
                           "area": False, "timing": False, "energy": False},
        "review_execution": {"vcs_runs": 0, "simv_runs": 0, "license_queries": 0,
                             "eda_runs": 0, "gpu_runs": 0, "git_mutations": 0,
                             "m2199_retry": False, "quarantine_modified": False,
                             "docs359_modified": False},
    }
    out = HERE / "mechanical_checks.json"
    need(not out.exists(), "fresh mechanical output required")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
