#!/opt/anaconda3/bin/python3.12
"""Independent M2230 CPU-only source review; no receipt or EDA is produced."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import re

HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "system_simulator/scripts/run_m2231_m2229_causal_parse_only_successor.py"
spec = importlib.util.spec_from_file_location("m2230_reviewed_source", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def main():
    contract = json.loads(m.CONTRACT.read_text())
    m.validate_inputs(contract)
    identities_before = {name: m.sha(m.REPO / name) for name in contract["pinned_files"]}
    comp, sim, rc = [(m.Q / name).read_text() for name in ("vcs_compile.log", "simv.log", "simv.rc")]
    tb = (m.REPO / m.TB_REL).read_text()
    actual = m.parse_raw(comp, sim, rc, tb)
    rejected = []
    def reject(label, comp_arg=comp, sim_arg=sim, rc_arg=rc, tb_arg=tb):
        try:
            m.parse_raw(comp_arg, sim_arg, rc_arg, tb_arg)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError("mutation accepted: " + label)
    for key, value in m.EXPECTED.items():
        reject("ledger_" + key, sim_arg=re.sub(r"\b" + key + r"=\d+", key + "=" + str(value + 1), sim))
    for key in actual["sva_matches"]:
        reject("missing_" + key, sim_arg="\n".join(line for line in sim.splitlines() if key not in line))
    for label, changed in (
        ("duplicate_cover", sim + "\n" + next(line for line in sim.splitlines() if line.startswith("M2213_COVER"))),
        ("duplicate_sva", sim + "\n" + next(line for line in sim.splitlines() if "cp_real_postread_request" in line)),
        ("unknown_ledger_field", sim.replace("golden_mismatches=0", "golden_mismatches=0 injected=0")),
        ("malformed_ledger_field", sim.replace("golden_mismatches=0", "golden_mismatches=NaN")),
        ("missing_finish", sim.replace("$finish called from file", "removed")),
        ("assertion_error", sim + "\nError: assertion ap_req_accept_definition failed\n"),
    ):
        reject(label, sim_arg=changed)
    reject("compiler_banner_in_wrong_file", comp_arg=comp.replace("Chronologic VCS (TM)", "removed"),
           sim_arg=sim + "\nChronologic VCS (TM)\n")
    reject("runtime_banner_in_wrong_file", comp_arg=comp + "\nChronologic VCS simulator copyright\n",
           sim_arg=sim.replace("Chronologic VCS simulator copyright", "removed"))
    reject("compile_error", comp_arg=comp + "\nError-[TEST] failed\n")
    reject("unreviewed_warning", comp_arg=comp + "\nWarning: injected\n")
    reject("source_warning_mismatch", tb_arg=tb.replace("context", "ctx"))
    review = {
        "status": "PASS_M2230_M2229_PARSE_ONLY_SOURCE__M2231_CPU_PARSE_AUTHORIZED",
        "score_over_100": 97,
        "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
        "identity": {"source_contract_sha256": m.sha(m.CONTRACT), "parser_runner_sha256": m.sha(SOURCE)},
        "authorization": {"cpu_parse_runs": 1, "license_queries": 0, "eda_runs": 0, "gpu_runs": 0, "automatic_retry": False}}
    m.authorize(review, m.sha(m.CONTRACT))
    review_rejected = []
    for label, path, value in (
        ("status", ("status",), "FAIL"),
        ("score", ("score_over_100",), 94),
        ("p0", ("severity_counts", "p0"), 1),
        ("p1", ("severity_counts", "p1"), 1),
        ("contract", ("identity", "source_contract_sha256"), "0" * 64),
        ("runner", ("identity", "parser_runner_sha256"), "0" * 64),
        ("retry", ("authorization", "automatic_retry"), True),
        ("eda", ("authorization", "eda_runs"), 1),
    ):
        changed = copy.deepcopy(review)
        node = changed
        for key in path[:-1]: node = node[key]
        node[path[-1]] = value
        try:
            m.authorize(changed, m.sha(m.CONTRACT))
        except ValueError:
            review_rejected.append(label)
        else:
            raise AssertionError("authority mutation accepted: " + label)
    tree = ast.parse(SOURCE.read_text())
    imports = sorted({alias.name.split(".")[0] for node in ast.walk(tree)
                      if isinstance(node, ast.Import) for alias in node.names}
                     | {node.module.split(".")[0] for node in ast.walk(tree)
                        if isinstance(node, ast.ImportFrom) and node.module != "__future__"})
    assert set(imports) <= {"argparse", "contextlib", "hashlib", "json", "os", "re", "sys", "traceback", "pathlib"}
    forbidden = {"system", "popen", "spawn", "fork", "execv", "execve", "Popen", "subprocess", "eval", "exec"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
            assert name not in forbidden, name
    m.validate_inputs(contract)
    assert identities_before == {name: m.sha(m.REPO / name) for name in contract["pinned_files"]}
    assert not any(path.exists() or path.is_symlink() for path in (m.RESULT, m.ATTEMPT, m.LOCK))
    print(json.dumps({
        "status": "PASS_M2230_INDEPENDENT_CPU_SOURCE_CHECKS",
        "actual_frozen_raw_parse": actual,
        "rejected_raw_mutations": rejected,
        "raw_mutation_count": len(rejected),
        "rejected_authority_mutations": review_rejected,
        "authority_mutation_count": len(review_rejected),
        "only_standard_library_imports": imports,
        "process_launch_calls": 0,
        "pinned_input_count": len(identities_before),
        "all_pinned_inputs_unchanged": True,
        "m2231_namespace_unconsumed": True,
        "source_or_raw_edits": False,
        "production_receipt_created": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
