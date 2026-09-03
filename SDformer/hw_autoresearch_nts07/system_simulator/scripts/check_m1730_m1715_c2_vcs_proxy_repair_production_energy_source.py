#!/usr/bin/env python3
"""No-EDA checker plus unchanged M1715 runtime parsers for M1730."""
from __future__ import print_function

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
OLD_CHECKER = HW / "system_simulator/scripts/check_m1715_m1710_m1684_c2_queue_order_repair_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1715_checker_for_m1730", str(OLD_CHECKER))
OLD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(OLD)

RUNNER = HW / "dc_handoff/scripts/run_m1730_m1715_c2_vcs_proxy_repair_production_energy_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1730_m1715_c2_vcs_proxy_repair_production_energy_source.py"
CONTRACT = HW / "contracts/m1730_m1715_c2_vcs_proxy_repair_production_energy_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1730_m1715_c2_vcs_proxy_repair_production_energy_source_author_receipt_r1_20260901"
M1731 = HW / "reviews/m1731_m1730_m1715_c2_vcs_proxy_repair_production_energy_source_hammer_r1_20260901"
M1732 = HW / "contracts/m1732_m1731_m1730_m1715_c2_vcs_proxy_repair_production_energy_launch_release_r1_20260901.json"

M1715_RUNNER = HW / "dc_handoff/scripts/run_m1715_m1710_m1684_m1661_c2_queue_order_repair_production_energy_one_shot.py"
M1715_CHECKER = OLD.CHECKER
M1715_TEST = OLD.TEST
M1715_CONTRACT = OLD.CONTRACT
M1715_AUTHOR = OLD.AUTHOR
M1716 = OLD.M1716
M1717 = OLD.M1717
M1715_ATTEMPT = HW / "results/.m1715_c2_queue_order_repair_production_energy_attempt_consumed"
M1715_FAILURE = HW / "results/m1715_c2_queue_order_repair_production_energy_r1_20260901.failed_or_incomplete.quarantine"
M1715_RESULT = HW / "results/m1715_c2_queue_order_repair_production_energy_r1_20260901"
M1715_PRIVATE = HW / "results/m1715_c2_queue_order_repair_production_energy_r1_20260901.private_build.unsealed_do_not_cite"
DOCS359 = OLD.OLD.OLD.DOCS359

CLAIMS = dict((key, False) for key in (
    "vcs", "mapped_functionality", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready", "headline"))
COUNTS = {"vcs_compiles": 2, "simv_runs": 10,
          "saif_files": 10, "ptpx_runs": 10}
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
              "http_proxy", "https_proxy")
EXPECTED_PROXY = {
    "HTTP_PROXY": "http://127.0.0.1:7897",
    "HTTPS_PROXY": "http://127.0.0.1:7897",
    "ALL_PROXY": "http://127.0.0.1:7897",
    "NO_PROXY": "localhost,127.0.0.1,::1",
    "http_proxy": "http://127.0.0.1:7897",
    "https_proxy": "http://127.0.0.1:7897",
}
FIXED = {
    M1715_RUNNER: "b5010f8d3d70ea2e029a3636b7b338d61b3b6f02ae8fa130dfbe279e736a2225",
    M1715_CHECKER: "1c2e917335be2e018c8df526e38265b94a7737241ac5ed1af3fbb02082538988",
    M1715_TEST: "bf76d2242fca01cee07a12f243ed53c3a755e2c15e79274e92f0a4de5cede11c",
    M1715_CONTRACT: "d775ca27bc3fa017751582aba34db8f3ebceac3aa9ecd8a550f413731ab68fc7",
    M1715_AUTHOR / "author_receipt.json": "a584b48d3507320180fcfb3a5b66d5fe1ba426ae2434bf41f132c569153d6c2d",
    M1715_AUTHOR / "SHA256SUMS": "d7cc3a26ceb1418783270179d802f93182c6cc098f5580e9a2f9aef863261e2a",
    M1715_AUTHOR / "SHA256SUMS.seal.sha256": "1869f59abc911c43f422b6b6b052473c2c679fbba7846afcb7bb11d087106379",
    M1716 / "review.json": "68c132bff9e52b2849c15e240e1cbc860706bdb24f4b3a1b42ddeb54423aeff2",
    M1716 / "SHA256SUMS": "72f7cfd0ea0569cb3fadf8009e47aad8420cd90936fcf51eba4359914b3f86d8",
    M1716 / "SHA256SUMS.seal.sha256": "67add3d29c18ed7e1f960078a2b358fd2f51ac09ca4d21d3e3bc19c577e686af",
    M1717: "32ae066e43a0889852e6eed7aa7b3a7cedf1a594a3e60116a629912ab9c2b6c8",
    Path(str(M1717) + ".sha256"): "8eaecbb4b87332e4afbb99819459127d06fce1442f611e7588324736817f8905",
    Path(str(M1717) + ".sha256.seal.sha256"): "501012d3f2d1ae666673e4591a1c92dcd398084d17dff3b816fc01d3163a86f5",
    M1715_ATTEMPT / "attempt.json": "9cdfebe0acd0bd81d1ff92f7423020b336cbe4c7ebc0dbcb0118d7ceab068c53",
    M1715_ATTEMPT / "SHA256SUMS": "18657601c99e8eb4141e3b175976a84bb491e04d7d1e85c46e82d6b463f71cf3",
    M1715_ATTEMPT / "SHA256SUMS.seal.sha256": "ab8f355009cb82e9c1211ff39a92295fcbe776ac26c856a804c2e52b903b7835",
    M1715_FAILURE / "failure.json": "e9e09df40e1e1b6e02064150b7d8752d2303bb1edabddc3f38085021cc6f3c02",
    M1715_FAILURE / "SHA256SUMS": "2d2a24c7499c31a2667cc097037ffa3fa9d270606f104b3d85aac145712d49fe",
    M1715_FAILURE / "SHA256SUMS.seal.sha256": "d09f54d3acc5f1dba9f0cc0387b9b003bf3a9ff136ff804f2ae198055cd8f971",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON: " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_sealed_directory(root, manifest_sha, outer_sha):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(root.is_dir() and not root.is_symlink(), "sealed root invalid")
    need(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
         "seal hash drift")
    need(outer.read_text() == manifest_sha + "  SHA256SUMS\n",
         "outer seal content drift")
    listed = set()
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        digest, name = fields[0], fields[1].lstrip("*")
        rel = Path(name)
        need(name not in listed and not rel.is_absolute() and ".." not in rel.parts,
             "manifest member unsafe/duplicate")
        path = root / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "manifest member drift: " + name)
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    need(actual == listed, "sealed population drift")


def verify_m1715_failure():
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift: " + str(path))
    verify_sealed_directory(M1715_AUTHOR, FIXED[M1715_AUTHOR / "SHA256SUMS"],
                            FIXED[M1715_AUTHOR / "SHA256SUMS.seal.sha256"])
    verify_sealed_directory(M1716, FIXED[M1716 / "SHA256SUMS"],
                            FIXED[M1716 / "SHA256SUMS.seal.sha256"])
    verify_sealed_directory(M1715_ATTEMPT, FIXED[M1715_ATTEMPT / "SHA256SUMS"],
                            FIXED[M1715_ATTEMPT / "SHA256SUMS.seal.sha256"])
    verify_sealed_directory(M1715_FAILURE, FIXED[M1715_FAILURE / "SHA256SUMS"],
                            FIXED[M1715_FAILURE / "SHA256SUMS.seal.sha256"])
    attempted = strict_json(M1715_ATTEMPT / "attempt.json")
    failed = strict_json(M1715_FAILURE / "failure.json")
    need(attempted.get("status") ==
         "M1715_C2_QUEUE_ORDER_REPAIR_PRODUCTION_ENERGY_ATTEMPT_CONSUMED",
         "M1715 attempt status")
    need(attempted.get("budget") == COUNTS and
         attempted.get("automatic_retry") is False, "M1715 attempt budget")
    need(failed == {
        "attempt_consumed": True, "automatic_retry": False,
        "canonical_result": False,
        "counts": {"ptpx_runs": 0, "saif_files": 0,
                   "simv_runs": 0, "vcs_compiles": 1},
        "error": "KeyboardInterrupt", "partial_axis_citable": False,
        "phase": "COMPILE_k8", "status": "FAILED_OR_INCOMPLETE"},
        "M1715 failure semantics")
    need(not os.path.lexists(M1715_RESULT), "M1715 canonical result exists")
    return failed


def _function_body(text, name, next_name):
    start = text.index("def " + name + "(")
    end = text.index("def " + next_name + "(", start)
    return text[start:end]


def validate_proxy_source(runner_text=None):
    text = RUNNER.read_text() if runner_text is None else runner_text
    start = text.index("EXPECTED_PROXY = {") + len("EXPECTED_PROXY = ")
    end = text.index("\n}\nPROXY_HOST", start) + 2
    parsed = ast.literal_eval(text[start:end])
    need(parsed == EXPECTED_PROXY, "exact proxy tuple drift")
    need('PROXY_HOST = "127.0.0.1"' in text and
         "PROXY_PORT = 7897" in text and
         "PROXY_CONNECT_TIMEOUT_S = 2.0" in text,
         "proxy endpoint/timeout constant drift")
    need('PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",\n'
         '              "http_proxy", "https_proxy")' in text,
         "proxy allowlist drift")
    capture = _function_body(text, "capture_exact_compile_proxy_from_launch",
                             "preflight_compile_proxy")
    need(capture.count("os.environ.get(name)") == 1,
         "launch proxy copy primitive drift")
    need("if value != EXPECTED_PROXY[name]:" in capture and
         "captured[name] = value" in capture,
         "per-key exact launch value gate drift")
    for forbidden in ("os.environ.copy", "dict(os.environ)", "os.getenv"):
        need(forbidden not in capture, "broad/alternate environment copy")
    preflight = _function_body(text, "preflight_compile_proxy", "clean_env")
    need("proxy != EXPECTED_PROXY" in preflight and
         "socket.create_connection(" in preflight and
         "(PROXY_HOST, PROXY_PORT), timeout=PROXY_CONNECT_TIMEOUT_S" in preflight,
         "proxy TCP preflight drift")
    clean = _function_body(text, "clean_env", "run")
    need("vcs_compile_proxy: dict[str, str] | None = None" in clean,
         "proxy injection gate absent")
    need("if vcs_compile_proxy is not None:" in clean and
         "if vcs_compile_proxy != EXPECTED_PROXY:" in clean and
         "value.update(vcs_compile_proxy)" in clean,
         "exact proxy injection gate drift")
    need("os.environ" not in clean, "clean_env copied ambient environment")
    main = text[text.index("def main("):]
    capture_pos = main.index("compile_proxy = capture_exact_compile_proxy_from_launch()")
    preflight_pos = main.index("preflight_compile_proxy(compile_proxy)")
    attempt_pos = main.index("ATTEMPT.mkdir()")
    need(capture_pos < preflight_pos < attempt_pos,
         "proxy capture/TCP preflight must precede attempt")
    need(text.count("vcs_compile_proxy=compile_proxy") == 1,
         "proxy must be injected into exactly one compile call site")
    need('state["phase"] = "PROXY_PREFLIGHT"\n'
         '        compile_proxy = capture_exact_compile_proxy_from_launch()\n'
         '        preflight_compile_proxy(compile_proxy)\n'
         '        state["phase"] = "PRE_ATTEMPT_RUNTIME_REBIND"' in main,
         "unconditional proxy preflight adjacency drift")
    compile_start = main.index('state["phase"] = "COMPILE_" + axis')
    sim_start = main.index('state["phase"] = "SIM_" + axis', compile_start)
    pt_start = main.index('state["phase"] = "PTPX_" + axis', sim_start)
    need("vcs_compile_proxy=compile_proxy" in main[compile_start:sim_start],
         "compile proxy injection absent")
    need("vcs_compile_proxy" not in main[sim_start:pt_start],
         "proxy leaked into sim/checker")
    need("vcs_compile_proxy" not in main[pt_start:],
         "proxy leaked into PTPX/result")
    need("M1715_PRIVATE is deliberately neither sealed nor inspected for claims" in text,
         "private forensic non-citation boundary absent")
    predecessor = _function_body(text, "verify_predecessors_and_inputs",
                                 "namespaces_fresh")
    need("verify_m1715_consumed_failure()" in predecessor,
         "M1715 exhausted failure binder absent")


def validate_sources():
    verify_m1715_failure()
    validate_proxy_source()
    runner = RUNNER.read_text()
    need(runner.count('for axis in ("k8", "k1x8"):') >= 3,
         "axis geometry drift")
    need(runner.count("for case_id in range(5):") >= 2,
         "case geometry drift")
    for token in ('"vcs_compiles": 2', '"simv_runs": 10',
                  '"saif_files": 10', '"ptpx_runs": 10'):
        need(token in runner, "execution budget drift")
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m1730_m1715_c2_vcs_proxy_repair_production_energy_source_contract_r1_v1",
         "contract schema drift")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1731_REVIEW_AND_M1732_RELEASE_REQUIRED__NO_EDA",
         "contract status drift")
    need(contract.get("claim_boundary") == CLAIMS, "claim promotion")
    rows = contract.get("source_files", [])
    mapping = dict((row.get("path"), row.get("sha256")) for row in rows)
    expected = (RUNNER, CHECKER, TEST)
    need(len(mapping) == len(rows) == len(expected), "source inventory cardinality")
    for path in expected:
        rel = path.relative_to(HW).as_posix()
        need(mapping.get(rel) == sha(path), "source SHA drift: " + rel)
    future = (M1731, M1732, Path(str(M1732) + ".sha256"),
              Path(str(M1732) + ".sha256.seal.sha256"),
              HW / "results/.m1730_c2_vcs_proxy_repair_production_energy_attempt_consumed",
              HW / "results/m1730_c2_vcs_proxy_repair_production_energy_r1_20260901",
              HW / "results/m1730_c2_vcs_proxy_repair_production_energy_r1_20260901.failed_or_incomplete.quarantine",
              HW / "results/m1730_c2_vcs_proxy_repair_production_energy_r1_20260901.private_build.unsealed_do_not_cite")
    for path in future:
        need(not os.path.lexists(path), "future/result namespace exists: " + str(path))
    return {
        "schema": "m1730_m1715_c2_vcs_proxy_repair_source_check_r1_v1",
        "status": "PASS_M1730_SOURCE_ONLY_NO_EDA",
        "m1715_attempt_consumed": True,
        "m1715_failure_phase": "COMPILE_k8",
        "m1715_counts": {"vcs_compiles": 1, "simv_runs": 0,
                         "saif_files": 0, "ptpx_runs": 0},
        "m1715_retry_forbidden": True,
        "proxy_keys": list(PROXY_KEYS),
        "proxy_exact_value_pin": True,
        "proxy_tcp_preflight_before_attempt": True,
        "proxy_scope": "VCS_COMPILE_ONLY",
        "private_forensic_tree_citable": False,
        "future_budget": COUNTS,
        "claim_boundary": CLAIMS,
    }


active_force_present = OLD.active_force_present
validate_saif = OLD.validate_saif
validate_runtime_log = OLD.validate_runtime_log
parse_power_report = OLD.parse_power_report
aggregate_metrics = OLD.aggregate_metrics
AXES = OLD.AXES
EVENTS = OLD.EVENTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source", "saif", "power"), required=True)
    parser.add_argument("--axis", choices=sorted(AXES))
    parser.add_argument("--case", dest="case_id", type=int)
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--saif", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--power-report", type=Path)
    args = parser.parse_args()
    if args.mode == "source":
        output = validate_sources()
    elif args.mode == "saif":
        need(args.axis is not None and args.case_id is not None and
             args.cycles is not None and args.saif and args.log, "saif arguments")
        output = validate_saif(args.saif, args.axis, args.case_id, args.cycles)
        output["runtime"] = validate_runtime_log(args.log, args.axis, args.case_id)
    else:
        need(args.power_report is not None, "power report argument")
        output = parse_power_report(args.power_report)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
