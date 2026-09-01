#!/usr/bin/env python3
"""Different-author, source-only hammer for M1698.  Never imports the runner."""
from __future__ import print_function

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RUNNER = HW / "dc_handoff/scripts/run_m1698_m1684_m1661_c2_shared_eda_queue_production_energy_one_shot.py"
CHECKER = HW / "system_simulator/scripts/check_m1698_m1684_c2_shared_eda_queue_production_energy_source.py"
TEST = HW / "system_simulator/tests/test_m1698_m1684_c2_shared_eda_queue_production_energy_source.py"
CONTRACT = HW / "contracts/m1698_m1684_c2_shared_eda_queue_production_energy_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1698_m1684_c2_shared_eda_queue_production_energy_source_author_receipt_r1_20260901"
OLD_CONTRACT = HW / "contracts/m1684_m1661_c2_m1609_fresh_mapped_production_energy_source_contract_r1_20260901.json"
M1686 = HW / "contracts/m1686_m1685_m1684_m1661_c2_m1609_fresh_mapped_production_energy_launch_release_r1_20260901.json"

IDENTITY = {
    "runner_sha256": "60bf8751d6d8eecf3aa469b1bda113b69172f05295bdd712413c9b8c543049be",
    "checker_sha256": "e992cf05332ac5777541888a2c8b39b43fc2f892b99763a383378af79b19eb2b",
    "test_sha256": "102b6d8244221d52fb8541fa63f3c40f6b29bfa6dd34fda5e7e71ac5cd6efa5e",
    "contract_sha256": "2b1ec47117ac3bdc71be8bd6eecff454f84a5ee0c4ee16d6095440a00a81e9de",
    "author_receipt_sha256": "61420ee154335682885e574d3e72a0eb56c932183386cc3a9e5642398a1438f4",
    "author_manifest_sha256": "97091f7e9b6d548412b93676a15dd369e1f5169041aa5d0aaca72ef82c48cf54",
    "author_outer_sha256": "cc675964c73b6e33e1470143f77b6ce1c95a26abf589ee8d5a1931c4c4bdb76c",
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


def verify_seal(root, manifest_sha, outer_sha):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(root.is_dir() and not root.is_symlink(), "author root")
    need(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
         "author seal identity")
    need(outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
         "author outer content")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        digest, name = fields[0], fields[1].lstrip("*")
        rel = Path(name)
        need(name not in listed and not rel.is_absolute() and ".." not in rel.parts,
             "manifest member")
        path = root / rel
        need(path.is_file() and not path.is_symlink()
             and stat.S_ISREG(path.lstat().st_mode) and sha(path) == digest,
             "manifest identity")
        listed.add(name)
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name not in
                 {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(actual == listed, "author population")


def load_checker():
    spec = importlib.util.spec_from_file_location("m1698_checked_only", str(CHECKER))
    module = importlib.util.module_from_spec(spec)
    need(spec is not None and spec.loader is not None, "checker spec")
    spec.loader.exec_module(module)
    return module


def selected_runner_functions(text):
    tree = ast.parse(text)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"_parent_pid", "_owned_or_ancestor"}]
    need(len(selected) == 2, "ancestry helpers")
    for node in selected:
        node.returns = None
        for argument in list(node.args.args) + list(node.args.kwonlyargs):
            argument.annotation = None
    try:
        module = ast.Module(body=selected, type_ignores=[])
    except TypeError:
        module = ast.Module(body=selected)
    ast.fix_missing_locations(module)
    namespace = {"Path": Path, "os": os}
    exec(compile(module, str(RUNNER), "exec"), namespace)
    return namespace


def main():
    for path, key in ((RUNNER, "runner_sha256"), (CHECKER, "checker_sha256"),
                      (TEST, "test_sha256"), (CONTRACT, "contract_sha256"),
                      (AUTHOR / "author_receipt.json", "author_receipt_sha256")):
        need(path.is_file() and not path.is_symlink() and sha(path) == IDENTITY[key],
             "identity: " + str(path))
    verify_seal(AUTHOR, IDENTITY["author_manifest_sha256"],
                IDENTITY["author_outer_sha256"])

    checker = load_checker()
    text = RUNNER.read_text()
    contract = strict_json(CONTRACT)
    old_contract = strict_json(OLD_CONTRACT)
    checker.validate_queue_source(text)

    # Shared queue and launch adjacency are structurally present.
    need('LOCK = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")' in text,
         "shared lock")
    main_text = text[text.index("def main("):]
    lock_index = main_text.index("fcntl.flock(")
    post_lock_index = main_text.index("collision_gate()", lock_index)
    attempt_index = main_text.index("ATTEMPT.mkdir()")
    first_compile_index = main_text.index('for axis in ("k8", "k1x8"):')
    need(lock_index < post_lock_index < attempt_index < first_compile_index,
         "lock/order")
    run_text = text[text.index("def run("):text.index("def result_identity")]
    need('Path(command[0]).name in {"vcs", "pt_shell"}' in run_text,
         "tool selector")
    need(run_text.index("collision_gate()") < run_text.index("subprocess.run("),
         "prelaunch rescan")
    ancestry = selected_runner_functions(text)
    need(ancestry["_owned_or_ancestor"](os.getpid(), os.getpid()), "own pid")
    need(not ancestry["_owned_or_ancestor"](1, os.getpid()), "external pid")

    # Exact workload and accounting geometry remain intact.
    need(contract["fair_campaign"]["clock_period_ns"] == 3.0, "period")
    need(contract["fair_campaign"]["accepted_sources_per_axis"] == 261,
         "sources")
    need(contract["fair_campaign"]["cases"] == [0, 1, 2, 3, 4], "cases")
    need(contract["future_execution"] == {
        "authorized_now": False, "attempts": 1, "vcs_compiles": 2,
        "simv_runs": 10, "production_saif_files": 10, "ptpx_runs": 10,
        "automatic_retry": False,
        "fresh_result_namespace": "results/m1698_c2_shared_eda_queue_production_energy_r1_20260901",
        "fresh_attempt_namespace": "results/.m1698_c2_shared_eda_queue_production_energy_attempt_consumed"},
        "execution budget")
    need(text.index("ATTEMPT.mkdir()") < text.index("command = [str(VCS)"),
         "attempt before VCS")

    # Current sources are clean, but the executable runner does not re-bind them.
    for path in checker.EXECUTION_SOURCES:
        need(not checker.active_force_present(path), "current active force")
        need("initreg" not in path.read_text().lower(), "current initreg")
    old_rows = old_contract["source_files"]
    direct_runtime = [row["path"] for row in old_rows[3:]]
    predecessor_body = text[text.index("def verify_predecessors_and_inputs"):
                            text.index("def namespaces_fresh")]
    need("for row in" not in predecessor_body and "source_files" not in predecessor_body,
         "unexpected old source runtime traversal")
    missing_runtime_binding = []
    for rel in direct_runtime:
        name = Path(rel).name
        if name not in predecessor_body:
            missing_runtime_binding.append(rel)
    need(len(missing_runtime_binding) == 6, "direct source omission count")

    # The checker also misses a live inline Tcl force command.
    inline_tcl = "if {1} { force dut/q 0 }\n"
    tcl_force_bypass = not checker.active_force_present(Path("probe.tcl"), inline_tcl)
    need(tcl_force_bypass, "expected Tcl bypass disappeared")

    # M1686 is checked only now, with exists(); the production runner has no gate.
    need(not os.path.lexists(M1686), "M1686 currently exists")
    need("M1686 =" not in text and "os.path.lexists(M1686)" not in text,
         "unexpected M1686 runtime gate")
    checker_text = CHECKER.read_text()
    need("need(not M1686.exists()" in checker_text,
         "expected author-only exists gate absent")

    # Independent source mutations: good queue defenses reject; omissions remain real.
    queue_mutations = {
        "private_lock": text.replace("/tmp/date_dual_synopsys_same_uid_eda_queue.lock",
                                     "/tmp/m1698_private.lock", 1),
        "remove_flock": text.replace(
            "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)",
            "pass", 1),
        "vcs_only_selector": text.replace(
            'Path(command[0]).name in {"vcs", "pt_shell"}',
            'Path(command[0]).name == "vcs"', 1),
        "remove_prelaunch_scan": text.replace(
            "        collision_gate()\n    with output.open",
            "        pass\n    with output.open", 1),
    }
    rejected = []
    for name, mutation in sorted(queue_mutations.items()):
        try:
            checker.validate_queue_source(mutation)
        except RuntimeError:
            rejected.append(name)
    need(len(rejected) == len(queue_mutations), "queue mutation escape")

    output = {
        "schema": "m1699_m1698_c2_shared_queue_source_independent_hammer_r1_v1",
        "status": "FAIL_M1699_M1698_C2_SHARED_EDA_QUEUE_PRODUCTION_ENERGY_SOURCE_HAMMER__NO_M1700_RELEASE__RUNTIME_SOURCE_AND_M1686_GATES_REQUIRED",
        "score_over_100": 88,
        "p0_count": 0,
        "p1_count": 2,
        "p2_count": 1,
        "verified": {
            "shared_lock_whole_campaign_structure": True,
            "post_lock_collision_scan": True,
            "pre_each_vcs_ptpx_collision_scan": True,
            "ancestry_helpers_executed": True,
            "current_execution_sources_force_clean": True,
            "five_cases_3ns_261_sources_per_axis": True,
            "execution_budget_2_10_10_10": True,
            "attempt_before_first_vcs": True,
            "automatic_retry": False,
            "queue_mutations_rejected": rejected,
            "fresh_m1661_m1677_contract_chain": True,
            "missing_runtime_bound_direct_sources": missing_runtime_binding,
            "inline_tcl_active_force_bypass": tcl_force_bypass,
            "m1686_currently_absent": True,
            "m1686_runtime_lexists_gate": False,
            "eda_executed": False,
            "attempt_created": False,
            "release_created": False,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
