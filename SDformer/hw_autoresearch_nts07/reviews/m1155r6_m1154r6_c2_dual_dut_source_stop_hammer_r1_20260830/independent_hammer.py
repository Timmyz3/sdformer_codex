#!/usr/bin/env python3
"""Independent fail-closed hammer for M1154R6 source/author receipt.

Read-only: no VCS/DC, no namespace consumption, and no old attempt retry.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import sys


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "dc_handoff/scripts/run_m1154r6_c2_dual_dut_vcs_root_diagnostic_source_r1.py"
SOURCE_SHA = "5e39999037463a5b190f61c66c9d895f7ad1af93bb7d9d7503d737b8133350e4"
CONTRACT = HW / "contracts/m1154r6_c2_dual_dut_vcs_root_diagnostic_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "52a27a267f2064efc3db8cffe3c705775eb97887b639114ba6e83cab3537df6e",
    "f1e3173dea034247627c31fe444b43f7bc0094b1a162c7e9cf4f450160cb107c",
    "75121250728947363d63ca9c8ed562961eb3dbb1fd9c3207806570304d8b2aa0",
)
AUTHOR = HW / "reviews/m1154r6_c2_dual_dut_vcs_root_diagnostic_author_receipt_r1_20260830"
AUTHOR_OUTER_FILE_SHA = "cfb73bb51d4681e131645b81f9b4f93a3c70f88732fbcc828a3efc4621ef5197"
NETLIST = HW / ("results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830."
                "failed_or_incomplete.1172090.quarantine/dc/netlist/"
                "m1129r5_c2_k1_async_observation_shadow_wrapper_mapped.v")
NETLIST_SHA = "362e855cd3b4391d31dc7a08e5388d9545f289c81d291c512d25294a8539cbc4"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
RESULTS = HW / "results"

RETAINED = (
    "implementation_core_frontend_compactor_fault_q",
    "implementation_core_frontend_paired_sink_fault_q",
    "implementation_core_adapter_fault_q",
    "implementation_core_g_k1_service_fault_q",
    "implementation_memory_adapter_fault_q",
)
PAIRED = (
    "implementation_core_mem_req_accept",
    "implementation_adapter_core_mem_req_accept",
    "implementation_core_mem_rsp_accept",
    "implementation_adapter_core_mem_rsp_accept",
)
CONSISTENCY = (
    "implementation_consistency_fault_now",
    "implementation_consistency_fault_q",
)
PROTOCOL = (
    "implementation_core_protocol_error",
    "implementation_adapter_protocol_error",
)
REQUIRED = RETAINED + PAIRED + CONSISTENCY + PROTOCOL


class Reject(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Reject(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(Reject("nonfinite JSON")))


def verify_double(path: Path, identities: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require((sha256(path), sha256(side), sha256(outer)) == identities, "contract triple")
    require(side.read_text(encoding="utf-8").split() == [identities[0], path.name], "sidecar content")
    require(outer.read_text(encoding="utf-8").split() == [identities[1], side.name], "outer content")


def verify_flat_tree(directory: Path) -> dict[str, str]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(sha256(outer) == AUTHOR_OUTER_FILE_SHA, "author outer file identity")
    require(outer.read_text(encoding="utf-8") == f"{sha256(manifest)}  SHA256SUMS\n", "author outer")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        member = directory / name
        require(name not in listed and member.is_file() and not member.is_symlink() and sha256(member) == digest,
                "author member")
        listed[name] = digest
    actual = {p.name for p in directory.iterdir() if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(listed), "author coverage")
    return {"manifest_sha256": sha256(manifest), "outer_file_sha256": sha256(outer)}


def declaration_prefix(path: Path) -> str:
    rows = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for row in stream:
            if re.match(r"^\s*[A-Z][A-Z0-9]*BWP35P140\s+", row):
                break
            rows.append(row)
    value = "".join(rows)
    require("module m1129r5_c2_k1_async_observation_shadow_wrapper" in value, "module declaration")
    return value


def census(declarations: str) -> dict:
    found = []
    missing = []
    for name in REQUIRED:
        if re.search(r"(?<![A-Za-z0-9_$])" + re.escape(name) + r"(?![A-Za-z0-9_$])", declarations):
            found.append(name)
        else:
            missing.append(name)
    anonymous = set(re.findall(r"\bn[0-9]+\b", declarations))
    return {"present": found, "missing": missing, "anonymous_n_declarations": len(anonymous)}


def namespace_fresh() -> bool:
    fixed = (
        RESULTS / "m1154r6_c2_dual_dut_vcs_root_diagnostic_r1_20260830",
        RESULTS / ".m1154r6_c2_dual_dut_vcs_root_diagnostic_attempt_consumed",
        Path("/tmp/m1154r6_c2_dual_dut_vcs_root_diagnostic.lock"),
    )
    variable = tuple(RESULTS.glob(".m1154r6_c2_dual_dut_vcs_root_diagnostic_work.*"))
    variable += tuple(RESULTS.glob("m1154r6_c2_dual_dut_vcs_root_diagnostic_r1_20260830.failed_or_incomplete.*"))
    return not any(path.exists() or path.is_symlink() for path in fixed + variable)


def admission_gate(*, present: int, anonymous_binding: bool, wildcard_binding: bool,
                   force_x: bool, observe_five_only: bool, retry_m1146: bool) -> None:
    require(present == 13, "not all semantic taps retained")
    require(not anonymous_binding, "anonymous n* binding")
    require(not wildcard_binding, "wildcard hierarchy")
    require(not force_x, "force/X masking")
    require(not observe_five_only, "five-tap observation is incomplete")
    require(not retry_m1146, "M1146 retry forbidden")


def main() -> None:
    checks = 0
    require(sha256(SOURCE) == SOURCE_SHA, "source identity")
    verify_double(CONTRACT, CONTRACT_ID)
    author_seal = verify_flat_tree(AUTHOR)
    require(sha256(NETLIST) == NETLIST_SHA, "netlist identity")
    require(sha256(DOCS359) == DOCS359_SHA, "docs359 identity")
    checks += 14

    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "review.json")
    require(contract["authorization"]["root_vcs_execution"] is False, "VCS authorization")
    require(contract["authorization"]["dc"] is False, "DC authorization")
    require(contract["authorization"]["retry_old_namespace"] is False, "old retry authorization")
    require(author["authorization"]["root_vcs_execution"] is False and author["authorization"]["dc"] is False and author["authorization"]["retry"] is False, "author authorization")
    require(namespace_fresh(), "M1154 namespace not fresh")
    checks += 7

    source_text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    require("subprocess" not in imported, "subprocess imported")
    forbidden_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"system", "popen", "spawn", "execv", "execve"}:
                forbidden_calls.append(node.func.attr)
    require(not forbidden_calls, "tool-spawn call present")
    require("raise Failure(preflight[\"status\"]" in source_text, "zero-argument stop path")
    checks += 3

    declarations = declaration_prefix(NETLIST)
    taps = census(declarations)
    require(taps["present"] == list(RETAINED), "retained tap set")
    require(taps["missing"] == list(PAIRED + CONSISTENCY + PROTOCOL), "missing tap set")
    require(taps["anonymous_n_declarations"] > 100000, "anonymous net population")
    checks += 16

    # The source's bounded self-test is safe after the independent AST gate:
    # it has no process-spawn primitive and the real namespace is required fresh.
    spec = importlib.util.spec_from_file_location("m1154r6_independent_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    bounded = module.source_bounded_mock_self_test()
    require(bounded["status"] == "PASS_SOURCE_AND_BOUNDED_MOCK__REAL_STABLE_TAP_GATE_STOP", "bounded status")
    require(bounded["real_preflight"]["stable_tap_census"]["present_count"] == 5, "bounded real present")
    require(bounded["real_preflight"]["stable_tap_census"]["missing_count"] == 8, "bounded real missing")
    require(bounded["bounded_mock_required_taps"] == 13 and bounded["bounded_mock_missing_taps"] == 0, "bounded synthetic taps")
    require(bounded["attempt_created"] is False and bounded["vcs_calls"] == 0 and bounded["dc_calls"] == 0, "bounded side effects")
    require(namespace_fresh(), "namespace mutated by bounded mock")
    checks += 7

    endpoint = module.valid_qualified_endpoint_template()
    probe = module.dual_dut_probe_template()
    require("request_payload_known=!$isunknown" in endpoint, "known-payload predicate")
    require("mem_req_ready=(mem_req_valid&&request_payload_known)?inner_req_ready:1'b0" in endpoint, "valid-qualified ready")
    require("mem_req_accept&&mem_req_valid&&request_payload_known" in endpoint, "valid-qualified accept")
    require("endpoint_protocol_fault_now=mem_req_valid&&!request_payload_known" in endpoint, "invalid-payload diagnostic")
    require(all(probe.count("orig.dut_orig." + tap) == 1 and probe.count("qualified.dut_qualified." + tap) == 1 for tap in REQUIRED), "dual-DUT tap coverage")
    # Important boundary: the generated probe is a specification module with comments,
    # not executable bitmap capture logic or DUT instances.
    probe_without_comments = "\n".join(row.split("//", 1)[0] for row in probe.splitlines())
    require("dut_orig" not in probe_without_comments and "dut_qualified" not in probe_without_comments, "unexpected executable DUT")
    require("bitmap" not in probe_without_comments and "always" not in probe_without_comments, "unexpected executable first-X implementation")
    checks += 7

    rejected = []
    attacks = {
        "anonymous_n_guess": dict(present=13, anonymous_binding=True, wildcard_binding=False, force_x=False, observe_five_only=False, retry_m1146=False),
        "wildcard_hierarchy": dict(present=13, anonymous_binding=False, wildcard_binding=True, force_x=False, observe_five_only=False, retry_m1146=False),
        "authorize_with_only_five_taps": dict(present=5, anonymous_binding=False, wildcard_binding=False, force_x=False, observe_five_only=True, retry_m1146=False),
        "force_x_masking": dict(present=13, anonymous_binding=False, wildcard_binding=False, force_x=True, observe_five_only=False, retry_m1146=False),
        "rerun_m1146": dict(present=13, anonymous_binding=False, wildcard_binding=False, force_x=False, observe_five_only=False, retry_m1146=True),
    }
    for name, values in attacks.items():
        try:
            admission_gate(**values)
        except Reject:
            rejected.append(name)
        else:
            raise Reject("attack survived: " + name)
    require(len(rejected) == 5, "attack count")
    checks += 6

    print(json.dumps({
        "status": "PASS_M1155R6_INDEPENDENT_FAIL_CLOSED_HAMMER__STOP_FROZEN_NETLIST_OBSERVATION_EXPANSION",
        "checks": checks,
        "source_sha256": SOURCE_SHA,
        "contract_identity": list(CONTRACT_ID),
        "author_seal": author_seal,
        "netlist_sha256": NETLIST_SHA,
        "stable_taps": {"required": 13, "present": taps["present"], "missing": taps["missing"], "anonymous_n_declarations": taps["anonymous_n_declarations"]},
        "bounded_mock": {
            "synthetic_census_pass": True,
            "valid_qualified_endpoint_contract_pass": True,
            "dual_dut_tap_name_spec_pass": True,
            "first_x_atomic_bitmap": "SPECIFICATION_ONLY_NOT_EXECUTABLE_OR_SIMULATED",
        },
        "real_execution": {"attempts": 0, "vcs_calls": 0, "dc_calls": 0, "namespace_fresh": True},
        "controlled_attacks_rejected": rejected,
        "verdict": "STOP_OLD_FROZEN_NETLIST_OBSERVATION_EXPANSION__RETAIN_M903_LOGIC_ONLY_UNLESS_NEW_SYNTHESIS_EXPLICITLY_PRESERVES_ALL_TAPS",
        "docs359_sha256": DOCS359_SHA,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
