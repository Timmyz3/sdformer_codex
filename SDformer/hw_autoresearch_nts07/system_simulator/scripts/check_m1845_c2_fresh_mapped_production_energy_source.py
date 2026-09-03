#!/usr/bin/env python3
"""CPU-only checker for the additive M1845 fresh-mapped energy source."""
from __future__ import print_function

import argparse
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tokenize


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RUNNER = HW / "dc_handoff/scripts/run_m1845_c2_fresh_mapped_production_energy_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1845_c2_fresh_mapped_production_energy_source.py"
CONTRACT = HW / "contracts/m1845_m1833_m1831_m1830_m1811_c2_fresh_mapped_production_energy_source_contract_r1_20260902.json"
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
CORE = HW / "dc_handoff/tb/tb_m1831_c2_fresh_mapped_gate_case_core.sv"
FAULT = HW / "dc_handoff/tb/m1831_c2_registered_public_fault_production_assertions.sv"
TOP_TB = HW / "dc_handoff/tb/tb_m1831_c2_fresh_mapped_production_energy.sv"
MEM = HW / "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv"
PROD_ASSERT = HW / "dc_handoff/tb/m1334_c2_production_activity_assertions.sv"
M979 = HW / "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv"
UCLI = HW / "dc_handoff/scripts/m1831_c2_fresh_mapped_production_energy.ucli.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1831_c2_fresh_mapped_production_energy_tt0p9v25c.tcl"
FILELISTS = {
    "k8": HW / "dc_handoff/filelists/date_m1831_c2_k8_fresh_mapped_production_energy.f",
    "k1x8": HW / "dc_handoff/filelists/date_m1831_c2_k1x8_fresh_mapped_production_energy.f",
}
M1811 = HW / "dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902"
M1830 = HW / "reviews/m1830_m1811_c2_registered_fault_matched_two_axis_dc_result_hammer_r1_20260902"
M1831_FAILED_CONTRACT = HW / "contracts/m1831_m1830_m1811_c2_fresh_mapped_production_energy_source_contract_r1_20260902.json"
M1833_FAILED_REVIEW = HW / "reviews/m1833_m1831_c2_fresh_mapped_production_energy_source_hammer_r1_20260902"
DESIGN_BASE = "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24"
MAPPED = dict((axis, {
    "netlist": M1811 / axis / "netlist" / (DESIGN_BASE + "_mapped.v"),
    "sdc": M1811 / axis / "netlist" / (DESIGN_BASE + "_mapped.sdc")})
    for axis in ("k8", "k1x8"))
CELL_V = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
TT_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140tt0p9v25c.db")
TOP = "tb_m1831_c2_fresh_mapped_production_energy"
SAIF_SCOPE = TOP + ".core.dut.implementation"
EVENTS = [20, 41, 90, 110, 0]
PACKETS = [1, 2, 4, 8, 1]
AXES = {
    "k8": {"define": "M1831_AXIS_K8",
           "derived_top": "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE0",
           "cycles": [51, 131, 486, 1231, 14],
           "netlist_sha256": "63605469818c36574ce9719130877610e79cf0c3b7317c0e69848539afa6b792",
           "sdc_sha256": "af2fbde96a5046053aed137facc4fd2741b3f517eb678710c81eef9f7ed49018"},
    "k1x8": {"define": "M1831_AXIS_K1X8",
             "derived_top": "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE1",
             "cycles": [53, 133, 499, 1246, 14],
             "netlist_sha256": "8698d227f3408b6e40c03bfe9282de458b0ba5cba4e22ec5f0c9bfd4ff16fc1b",
             "sdc_sha256": "1631f7d0cc3d0257439dea5f9ed2a2fc004556dc0f8f5657152a7d3f5f3e6c0a"},
}
CLAIMS = dict((key, False) for key in (
    "source_complete", "source_reviewed", "release", "mapped_vcs",
    "production_saif", "ptpx", "power", "energy", "same_resource_result",
    "paper_admitted", "component_speedup", "system_speedup", "headline"))
CLAIMS["source_complete"] = True

CANONICAL = {
    "m1811_canonical_directory":
        "dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902",
    "m1811_receipt_sha256":
        "3bec6bb629d81a756b5eb9bb4570b04fc1de17a21c0a6143bca6b6c886945e6d",
    "m1811_manifest_sha256":
        "695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066",
    "m1811_outer_file_sha256":
        "04aa6bea4a06a8be3c441ddb984c68a046810a137fd2eca096adf513af0d324b",
    "m1830_review_directory":
        "reviews/m1830_m1811_c2_registered_fault_matched_two_axis_dc_result_hammer_r1_20260902",
    "m1830_review_sha256":
        "79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b",
    "m1830_manifest_sha256":
        "d0ef8172f33378e9b025aab18043da19335fd9f00d1cd8d240bfb620997c0d06",
    "m1830_outer_file_sha256":
        "0b9dc1915096db8df6702e3ab5027d267fb99a3178bc2288a8b5625e611e343d",
    "m1831_failed_contract_path":
        "contracts/m1831_m1830_m1811_c2_fresh_mapped_production_energy_source_contract_r1_20260902.json",
    "m1831_failed_contract_sha256":
        "79052bcd5da95aa0ed441f5efbd403d83605b7ae6dd14f1a4c9a3cbaf746fdc3",
    "m1831_failed_contract_sidecar_sha256":
        "875631f366ce10b6602f7fec1319c921628312a882bb0ca186164bc4b7333d13",
    "m1831_failed_contract_outer_file_sha256":
        "61bce3bc6d2dd5155890d00461aafc76b5c649a54806c50f049bcf695a9b4f26",
    "m1833_failed_review_directory":
        "reviews/m1833_m1831_c2_fresh_mapped_production_energy_source_hammer_r1_20260902",
    "m1833_failed_review_sha256":
        "877254bf4de7b25255226e85c535637d2711413a82fc0feb20f2692aac074918",
    "m1833_failed_manifest_sha256":
        "dc48c4afb7943d7237ceef5cea6aaae3b73620634acd8ed0df98dfd5adcc5884",
    "m1833_failed_outer_file_sha256":
        "24e66afa58e38cf6680244f970826f2a295005a0da4b6f16f1d71b078bdb1b08",
    "k8_mapped_netlist_sha256": AXES["k8"]["netlist_sha256"],
    "k8_mapped_sdc_sha256": AXES["k8"]["sdc_sha256"],
    "k1x8_mapped_netlist_sha256": AXES["k1x8"]["netlist_sha256"],
    "k1x8_mapped_sdc_sha256": AXES["k1x8"]["sdc_sha256"],
}
SOURCE_PATHS = (CORE, FAULT, TOP_TB, FILELISTS["k8"], FILELISTS["k1x8"],
                UCLI, PT_TCL, RUNNER, CHECKER, TEST)
REUSED_SOURCE_DIGESTS = {
    MEM: "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
    PROD_ASSERT: "86be3fa541bf65afa6ada99aa3e2bd494ed689594fece18cfea135b91420c32a",
    M979: "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c",
}
TECH_DIGESTS = {
    CELL_V: "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    TT_DB: "d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070",
}

# Populated from syntax-token normalization, not from raw source bytes.  The
# independent test synchronizes source_inventory after every mutation; these
# guards therefore exercise effective control flow instead of identity drift.
RUNNER_SEMANTIC_TOKEN_SHA256 = {
    "exact": "8940e4e3fd482f0598d9bfc999e9b4fa527dcf9af1b86ec96acf17ca5f0b0941",
    "verify_directory_seal": "7b79a880d166f3802fc96141f1b1ab98e4925ba1605700cba27a690afa36abe1",
    "verify_file_double_seal": "2b3939aebf23e86ecaf15cb493229ddc0424676f4242f5793cd53094721f3624",
    "verify_authority_and_canonical": "609fe2b334d33a13443ce127f9b6f16b37df90065fe1c6cf266b67286254e55c",
    "collision_gate": "22a176fb78c983345584c050f6f572fbecdf503da6db92e2c56ae6456c4c79a0",
    "resource_gate": "e2e656ae61d1f2c9b41a7536f97b97d48e9199a23e764bef898e2ed4feba903a",
    "run": "1d2e804563d5dc0d422ed7f665e65233cedf27addb0d0a34e4df89ff7fca8903",
    "seal_dir": "965b8300c17441acc0e83d7afe5256f2513167825124365c722107afd1260acc",
    "publish_no_replace": "685726b550ab00ccb0e86dcd8e72faa0062d2eb79de1ca09e42f544c199ba7e2",
    "main": "d0de30e76aaf38ebe9bb9b0dca95da4eef0ccfb2657c2ca239f42811e51ca016",
}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    import hashlib
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
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_directory_seal(root, manifest_sha, outer_sha):
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), "sealed directory")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and not manifest.is_symlink()
         and sha(manifest) == manifest_sha, "manifest identity")
    need(outer.is_file() and not outer.is_symlink() and sha(outer) == outer_sha,
         "outer identity")
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer content")
    mapping = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts
             and name not in mapping, "manifest path")
        path = root / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == fields[0],
             "manifest drift " + name)
        mapping[name] = fields[0]
    actual = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "symlink in sealed directory")
        if (path.is_file() and path.name not in
                ("SHA256SUMS", "SHA256SUMS.seal.sha256")):
            actual.add(path.relative_to(root).as_posix())
    need(actual == set(mapping), "manifest not exhaustive")
    return mapping


def verify_contract_seal():
    need(CONTRACT.is_file() and not CONTRACT.is_symlink()
         and CONTRACT_SIDECAR.is_file() and not CONTRACT_SIDECAR.is_symlink()
         and CONTRACT_OUTER.is_file() and not CONTRACT_OUTER.is_symlink(),
         "contract seal inputs")
    need(CONTRACT_SIDECAR.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(CONTRACT_OUTER.read_text().split() ==
         [sha(CONTRACT_SIDECAR), CONTRACT_SIDECAR.name], "contract outer")


def verify_file_double_seal(path, file_sha, sidecar_sha, outer_sha):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    need(path.is_file() and not path.is_symlink() and sha(path) == file_sha,
         "sealed file identity")
    need(sidecar.is_file() and not sidecar.is_symlink()
         and sha(sidecar) == sidecar_sha, "sealed sidecar identity")
    need(outer.is_file() and not outer.is_symlink() and sha(outer) == outer_sha,
         "sealed outer identity")
    need(sidecar.read_text().split() == [sha(path), path.name],
         "sealed sidecar content")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name],
         "sealed outer content")


def active_lines(path):
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def function_source(text, name):
    match = re.search(r"(?ms)^def " + re.escape(name)
                      + r"\([^\n]*\):\n.*?(?=^(?:def |class |if __name__))", text)
    need(match is not None, "runner function absent " + name)
    return match.group(0).rstrip() + "\n"


def semantic_token_sha(text, name):
    source = function_source(text, name)
    ignored = set((tokenize.COMMENT, tokenize.NL, tokenize.ENCODING,
                   tokenize.ENDMARKER))
    rows = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in ignored:
            rows.append(tokenize.tok_name[token.type] + ":" + token.string)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return (base + "." if base else "") + node.attr
    return ""


def validate_runner_semantics(runner):
    tree = ast.parse(runner)
    functions = dict((node.name, node) for node in tree.body
                     if isinstance(node, ast.FunctionDef))
    for name, expected in RUNNER_SEMANTIC_TOKEN_SHA256.items():
        need(name in functions, "semantic function absent " + name)
        need(expected != "PENDING" and semantic_token_sha(runner, name) == expected,
             "semantic function drift " + name)
    false_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.NameConstant) and node.value is False:
            false_nodes.append(node)
        elif hasattr(ast, "Constant") and isinstance(node, ast.Constant) \
                and node.value is False:
            false_nodes.append(node)
    # Literal False is legitimate only in data dictionaries and check=False;
    # never as the left operand of a guard-neutralizing BoolOp.
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp):
            need(not any(item in false_nodes for item in node.values),
                 "constant-false control-flow bypass")
    exact_returns = [node for node in ast.walk(functions["exact"])
                     if isinstance(node, ast.Return)]
    resource_returns = [node for node in ast.walk(functions["resource_gate"])
                        if isinstance(node, ast.Return)]
    file_seal_returns = [node for node in ast.walk(
        functions["verify_file_double_seal"]) if isinstance(node, ast.Return)]
    publish_returns = [node for node in ast.walk(functions["publish_no_replace"])
                       if isinstance(node, ast.Return)]
    need(not exact_returns and not resource_returns and not file_seal_returns
         and not publish_returns, "early-return semantic bypass")
    calls = [call_name(node.func) for node in ast.walk(tree)
             if isinstance(node, ast.Call)]
    for required in ("CHECK.validate_sources", "CHECK.validate_runtime_log",
                     "CHECK.validate_compile_log",
                     "CHECK.validate_sealed_result_stage", "hits.append",
                     "shutil.copy2", "publish_no_replace"):
        need(required in calls, "required effective call absent " + required)
    need("os.replace" not in calls, "replace publication bypass")
    for node in ast.walk(functions["main"]):
        if isinstance(node, ast.Call) and call_name(node.func) == "ATTEMPT.mkdir":
            need(not node.keywords, "attempt latch reusable")


def validate_source_texts(texts):
    runner = texts[RUNNER]
    core = texts[CORE]
    fault = texts[FAULT]
    ucli = texts[UCLI]
    pt = texts[PT_TCL]
    contract = json.loads(texts[CONTRACT])
    validate_runner_semantics(runner)
    for forbidden in (".m1811_", "work.894487", "m1661_m1652",
                      "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v",
                      "reviews/m1847_m1845_c2_fresh_mapped_production_energy_source_hammer",
                      "contracts/m1849_m1847_m1845_c2_fresh_mapped_production_energy_launch_release",
                      "M1845_EXPECTED_M1847_",
                      "M1845_EXPECTED_M1811_", "M1845_EXPECTED_M1830_",
                      "M1845_EXPECTED_K8_NETLIST_SHA256",
                      "M1845_EXPECTED_K1X8_NETLIST_SHA256"):
        need(forbidden not in runner, "runner binds forbidden partial/old netlist " + forbidden)
        need(all(forbidden not in texts[path] for path in FILELISTS.values()),
             "filelist binds forbidden partial/old netlist " + forbidden)
    need(all("mapped.v" not in texts[path] for path in FILELISTS.values()),
         "filelist must receive the canonical netlist from the runner")
    for axis in AXES:
        need(AXES[axis]["derived_top"] in core, "core derived top " + axis)
        need(AXES[axis]["derived_top"] in runner, "runner derived top " + axis)
        need(str(AXES[axis]["cycles"]) in runner, "runner cycle anchor " + axis)
    for token in ("implementation (", "`include \"tb_m979_c2_three_axis_mapped_gate_case_saif.sv\"",
                  "M1831_AXIS_K8", "M1831_AXIS_K1X8"):
        need(token in core, "core omits " + token)
    for token in ("ap_public_fault_binary", "ap_registered_public_fault_zero",
                  "@(negedge clk_core)", "$isunknown({protocol_error",
                  "accepted_sources != expected_sources(case_id)",
                  "registered_fault_public_zero=1"):
        need(token in fault, "fault assertion omits " + token)
    need(fault.count("$isunknown({protocol_error, numeric_overflow,") == 2,
         "fault X/Z checks must remain procedural plus SVA")
    expected_ucli = [
        "power -gate_level all mda sv", "power " + SAIF_SCOPE, "run",
        "power -enable", "run", "power -disable",
        "power -report $::env(M1831_SAIF_FILE) 1e-9 " + SAIF_SCOPE,
        "quit"]
    need([line for line in ucli.splitlines()
          if line.strip() and not line.lstrip().startswith("#")] == expected_ucli,
         "UCLI exact scope/order")
    for token in ("M1831_DESIGN_NAME", "M1831_SAIF_INSTANCE",
                  "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE0",
                  "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE1",
                  "read_saif -strip_path", "annotated_nets != $total_nets",
                  "annotated_leaf_cells != $total_leaf_cells",
                  'puts $scope_fp "macro_count=0"',
                  'puts $marker "macro_count=0"',
                  "external_weight_sram_excluded=true"):
        need(token in pt, "PTPX omits " + token)
    normalized_pt = re.sub(r"\s+", " ", pt)
    need("if {$total_nets <= 0 || $annotated_nets != $total_nets || $annotated_percent != 100.0 || $total_leaf_cells <= 0 || $annotated_leaf_cells != $total_leaf_cells || $annotated_leaf_percent != 100.0} { error \"M1831_FAIL_EXACT_ANNOTATION_GATE\" }" in normalized_pt,
         "PTPX exact annotation semantic gate")
    for token in ('"vcs_compiles": 2', '"simv_runs": 10',
                  '"saif_files": 10', '"ptpx_runs": 10',
                  'for axis in AXIS_ORDER:', 'for case_id in CASES:',
                  'all ten mapped SAIF coordinates required before PTPX',
                  CANONICAL["m1811_manifest_sha256"],
                  CANONICAL["m1811_outer_file_sha256"],
                  CANONICAL["m1811_receipt_sha256"],
                  CANONICAL["m1830_review_sha256"],
                  CANONICAL["m1830_manifest_sha256"],
                  CANONICAL["m1830_outer_file_sha256"],
                  CANONICAL["m1831_failed_contract_sha256"],
                  CANONICAL["m1831_failed_contract_sidecar_sha256"],
                  CANONICAL["m1831_failed_contract_outer_file_sha256"],
                  CANONICAL["m1833_failed_review_sha256"],
                  CANONICAL["m1833_failed_manifest_sha256"],
                  CANONICAL["m1833_failed_outer_file_sha256"],
                  AXES["k8"]["netlist_sha256"],
                  AXES["k8"]["sdc_sha256"],
                  AXES["k1x8"]["netlist_sha256"],
                  AXES["k1x8"]["sdc_sha256"],
                  'reviews/m1848_m1845_c2_fresh_mapped_production_energy_source_hammer_r1_20260902',
                  'M1845_EXPECTED_M1848_MANIFEST_SHA256',
                  'M1845_EXPECTED_M1848_REVIEW_SHA256',
                  'm1848_m1845_c2_fresh_mapped_production_energy_source_hammer_review_r1_v1',
                  'PASS_M1848_M1845_C2_FRESH_MAPPED_PRODUCTION_ENERGY_SOURCE_HAMMER__P0_0_P1_0_P2_0__AUTHORIZED_FOR_M1849_RELEASE',
                  'M1848 source review schema/status/severity/authorization',
                  'contracts/m1849_m1848_m1845_c2_fresh_mapped_production_energy_launch_release_r1_20260902.json',
                  'M1845_EXPECTED_M1849_RELEASE_SHA256',
                  'M1845_EXPECTED_M1849_SIDECAR_SHA256',
                  'M1845_EXPECTED_M1849_OUTER_FILE_SHA256',
                  'm1849_m1848_m1845_c2_fresh_mapped_production_energy_launch_release_r1_v1',
                  'release.get("identity") != expected_release_identity',
                  'release.get("prelaunch_claim_boundary") != CHECK.CLAIMS',
                  'compile_evidence = STAGE / "compile_logs"',
                  'CHECK.validate_compile_log(',
                  'CHECK.validate_sealed_result_stage(STAGE)'):
        need(token in runner, "runner omits " + token)
    need(runner.index("all ten mapped SAIF coordinates required before PTPX")
         < runner.index('state["phase"] = "PTPX_"'),
         "PTPX precedes complete SAIF gate")
    need(contract.get("schema") ==
         "m1845_m1833_m1831_m1830_m1811_c2_fresh_mapped_production_energy_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_COMPLETE__M1811_M1830_CANONICAL_AND_M1831_M1833_FAILURE_BOUND__M1848_REVIEW_AND_M1849_RELEASE_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("canonical_identity") == CANONICAL,
         "canonical identity")
    future = contract.get("future_authority")
    need(type(future) is dict, "future authority")
    need(future.get("m1848_source_review_directory") ==
         "reviews/m1848_m1845_c2_fresh_mapped_production_energy_source_hammer_r1_20260902",
         "M1848 source-review identity")
    need(future.get("m1849_launch_release_contract") ==
         "contracts/m1849_m1848_m1845_c2_fresh_mapped_production_energy_launch_release_r1_20260902.json",
         "M1849 launch-release identity")
    for field in ("m1848_source_review_sha256",
                  "m1848_source_review_manifest_sha256",
                  "m1848_source_review_outer_file_sha256",
                  "m1849_launch_release_sha256",
                  "m1849_launch_release_sidecar_sha256",
                  "m1849_launch_release_outer_file_sha256"):
        need(future.get(field) == "PENDING_EXTERNAL_PIN",
             "future authority field " + field)
    need(contract.get("claim_boundary") == CLAIMS, "source claim boundary")
    need(contract.get("future_execution_budget") == {
        "license_queries": 1, "vcs_compiles": 2, "simv_runs": 10,
        "saif_files": 10, "ptpx_runs": 10, "automatic_retry": False},
        "future budget")
    inventory = contract.get("source_inventory")
    need(type(inventory) is list and len(inventory) == len(SOURCE_PATHS),
         "source inventory cardinality")
    mapping = dict((row.get("path"), row.get("sha256")) for row in inventory)
    need(len(mapping) == len(SOURCE_PATHS), "source inventory uniqueness")
    for path in SOURCE_PATHS:
        need(mapping.get(str(path.relative_to(HW))) == text_sha(texts[path]),
             "source inventory drift " + str(path))
    reused = contract.get("reused_source_identity")
    need(reused == dict((str(path.relative_to(HW)), digest)
                        for path, digest in REUSED_SOURCE_DIGESTS.items()),
         "reused source identity")
    need(contract.get("technology_identity") == {
        "cell_verilog_path": str(CELL_V),
        "cell_verilog_sha256": TECH_DIGESTS[CELL_V],
        "tt_db_path": str(TT_DB),
        "tt_db_sha256": TECH_DIGESTS[TT_DB]}, "technology identity")


def validate_sources():
    paths = (RUNNER, CHECKER, TEST, CONTRACT, CORE, FAULT, TOP_TB, MEM,
             PROD_ASSERT, M979, UCLI, PT_TCL, FILELISTS["k8"],
             FILELISTS["k1x8"])
    for path in paths:
        need(path.is_file() and not path.is_symlink(), "source absent " + str(path))
    texts = dict((path, path.read_text()) for path in paths)
    validate_source_texts(texts)
    verify_contract_seal()
    m1811_map = verify_directory_seal(
        M1811, CANONICAL["m1811_manifest_sha256"],
        CANONICAL["m1811_outer_file_sha256"])
    m1830_map = verify_directory_seal(
        M1830, CANONICAL["m1830_manifest_sha256"],
        CANONICAL["m1830_outer_file_sha256"])
    need(sha(M1811 / "receipt.json") == CANONICAL["m1811_receipt_sha256"]
         and m1811_map.get("receipt.json") == CANONICAL["m1811_receipt_sha256"],
         "M1811 receipt identity")
    need(sha(M1830 / "review.json") == CANONICAL["m1830_review_sha256"]
         and m1830_map.get("review.json") == CANONICAL["m1830_review_sha256"],
         "M1830 review identity")
    review = strict_json(M1830 / "review.json")
    need(review.get("status") ==
         "PASS_M1830_M1811_C2_REGISTERED_FAULT_MATCHED_TWO_AXIS_DC_RESULT_HAMMER__P0_0_P1_0_P2_0__SETUP_AREA_ADMITTED"
         and review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
         "M1830 admission")
    verify_file_double_seal(
        M1831_FAILED_CONTRACT,
        CANONICAL["m1831_failed_contract_sha256"],
        CANONICAL["m1831_failed_contract_sidecar_sha256"],
        CANONICAL["m1831_failed_contract_outer_file_sha256"])
    failed_map = verify_directory_seal(
        M1833_FAILED_REVIEW, CANONICAL["m1833_failed_manifest_sha256"],
        CANONICAL["m1833_failed_outer_file_sha256"])
    need(failed_map.get("review.json") ==
         CANONICAL["m1833_failed_review_sha256"],
         "M1833 failed review sealed member")
    failed_review = strict_json(M1833_FAILED_REVIEW / "review.json")
    need(failed_review.get("schema") ==
         "m1833_m1831_c2_fresh_mapped_production_energy_source_hammer_review_r1_v1"
         and failed_review.get("status") ==
         "FAIL_CLOSED_M1833_M1831_C2_FRESH_MAPPED_PRODUCTION_ENERGY_SOURCE_HAMMER__P0_0_P1_3_P2_0__NO_EDA_NO_LICENSE_NO_RELEASE"
         and failed_review.get("severity_counts") ==
         {"p0": 0, "p1": 3, "p2": 0}, "M1833 failure chain")
    for axis in AXES:
        for kind in ("netlist", "sdc"):
            need(MAPPED[axis][kind].is_file()
                 and not MAPPED[axis][kind].is_symlink()
                 and sha(MAPPED[axis][kind]) == AXES[axis][kind + "_sha256"],
                 "mapped identity " + axis + " " + kind)
        modules = re.findall(r"(?m)^\s*module\s+([^\s(]+)",
                             MAPPED[axis]["netlist"].read_text(errors="strict"))
        need(modules.count(AXES[axis]["derived_top"]) == 1,
             "mapped derived top " + axis)
    for path, digest in REUSED_SOURCE_DIGESTS.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "reused source identity " + str(path))
    for path, digest in TECH_DIGESTS.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "technology identity " + str(path))
    for axis, path in FILELISTS.items():
        lines = active_lines(path)
        need(lines[0] == "+define+" + AXES[axis]["define"],
             "filelist axis define")
        need(not any("mapped.v" in line for line in lines),
             "filelist must not bind mapped netlist")
    return {"schema": "m1845_c2_fresh_mapped_energy_source_check_r1_v1",
            "status": "PASS_M1845_SOURCE_COMPLETE_PENDING_M1848_REVIEW_M1849_RELEASE",
            "axes": ["k8", "k1x8"], "cases_per_axis": 5,
            "accepted_sources_per_axis": sum(EVENTS),
            "canonical_identity": CANONICAL,
            "source_files": len(SOURCE_PATHS),
            "future_budget": {"vcs_compiles": 2, "simv_runs": 10,
                              "saif_files": 10, "ptpx_runs": 10},
            "claim_boundary": CLAIMS}


def sexpr_tokens(text):
    return re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+', text)


def parse_saif(text):
    tokens = sexpr_tokens(text); pos = [0]
    def parse_one():
        need(pos[0] < len(tokens) and tokens[pos[0]] == "(", "malformed SAIF")
        pos[0] += 1; node = []
        while pos[0] < len(tokens) and tokens[pos[0]] != ")":
            if tokens[pos[0]] == "(": node.append(parse_one())
            else: node.append(tokens[pos[0]]); pos[0] += 1
        need(pos[0] < len(tokens), "unterminated SAIF"); pos[0] += 1
        return node
    root = parse_one()
    need(pos[0] == len(tokens) and root and root[0] == "SAIFILE", "SAIF root")
    return root


def forms(node, tag):
    return [item for item in node[1:]
            if isinstance(item, list) and item and item[0] == tag]


def all_forms(node, tag):
    found = []
    if isinstance(node, list):
        if node and node[0] == tag: found.append(node)
        for item in node:
            if isinstance(item, list): found.extend(all_forms(item, tag))
    return found


def direct_instance(node, name):
    hits = [item for item in forms(node, "INSTANCE")
            if len(item) >= 2 and item[1].lstrip("\\") == name]
    need(len(hits) == 1, "SAIF instance " + name)
    return hits[0]


def activity_under(node):
    activity = {}
    def walk(value):
        if not isinstance(value, list): return
        tc = forms(value, "TC")
        if value and isinstance(value[0], str) and len(tc) == 1:
            need(len(tc[0]) == 2, "malformed TC")
            name = value[0].lstrip("\\")
            activity[name] = activity.get(name, 0.0) + float(tc[0][1])
        for child in value[1:]: walk(child)
    walk(node); return activity


def cone(activity, prefixes):
    return sum(value for name, value in activity.items()
               if any(name == prefix or name.startswith(prefix + "[")
                      for prefix in prefixes))


def expected_compile_command(axis):
    need(axis in AXES, "compile axis")
    return ["/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs", "-full64",
            "-sverilog", "+v2k", "-timescale=1ns/1ps", "-assert", "svaext",
            "-debug_access+r", "-lca", "+vcs+lic+wait", "-Mdir=csrc",
            str(MAPPED[axis]["netlist"]), "-f", str(FILELISTS[axis]),
            "-top", TOP, "-o", "simv"]


def validate_compile_log(path, axis, expected_command=None):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "compile log regular")
    text = path.read_text(errors="strict")
    lines = text.splitlines()
    need(lines and lines[0].startswith("M1845_COMMAND_JSON="),
         "compile command record absent")
    command = json.loads(lines[0].split("=", 1)[1])
    expected = expected_compile_command(axis)
    need(type(command) is list and command == expected,
         "compile command identity")
    if expected_command is not None:
        need(list(expected_command) == expected, "caller compile command identity")
    lowered = "\n".join(lines[1:]).lower()
    forbidden = ("error-", "fatal", "unresolved", "undefined module",
                 "black box", "black-box", "compile error")
    need(not any(token in lowered for token in forbidden),
         "compile fatal/unresolved/black-box diagnostic")
    return {"axis": axis, "compile_log_sha256": sha(path),
            "derived_top": AXES[axis]["derived_top"],
            "mapped_netlist_sha256": AXES[axis]["netlist_sha256"],
            "mapped_sdc_sha256": AXES[axis]["sdc_sha256"],
            "command_sha256": text_sha(json.dumps(
                expected, separators=(",", ":"), ensure_ascii=True))}


def validate_sealed_result_stage(root):
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), "result stage directory")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and not manifest.is_symlink(), "result manifest")
    need(outer.is_file() and not outer.is_symlink(), "result outer seal")
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "result outer content")
    mapping = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "result manifest syntax")
        rel = Path(fields[1].lstrip("*")); name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts
             and name not in mapping, "result manifest path")
        member = root / rel
        need(member.is_file() and not member.is_symlink()
             and sha(member) == fields[0], "result member drift " + name)
        mapping[name] = fields[0]
    actual = set()
    for member in root.rglob("*"):
        need(not member.is_symlink(), "result symlink")
        if (member.is_file() and member.name not in
                ("SHA256SUMS", "SHA256SUMS.seal.sha256")):
            actual.add(member.relative_to(root).as_posix())
    need(actual == set(mapping), "result manifest not exhaustive")
    required = {"compile_logs/k8.compile.log",
                "compile_logs/k1x8.compile.log", "compile_log_rows.json"}
    need(required.issubset(set(mapping)), "sealed compile evidence absent")
    rows = json.loads((root / "compile_log_rows.json").read_text())
    need(type(rows) is list and len(rows) == 2, "compile evidence rows")
    checked = [validate_compile_log(root / "compile_logs" /
               (axis + ".compile.log"), axis) for axis in ("k8", "k1x8")]
    need(rows == checked, "compile evidence row drift")
    return {"members": len(mapping), "compile_logs": checked,
            "manifest_sha256": sha(manifest), "outer_file_sha256": sha(outer)}


def validate_runtime_log(path, axis, case_id):
    text = Path(path).read_text(errors="strict")
    need(not any(token in text for token in
                 ("Assertion failed", "Fatal:", "$fatal", "Error-[",
                  "contains X/Z", "raised fault", "coverage incomplete")),
         "runtime fatal/assertion")
    pattern = (r"PASS M1831 registered-fault production case=" + str(case_id)
               + r" accepted_sources=([0-9]+) source_packets=([0-9]+)"
               + r" endpoint_accepts=([0-9]+) result_accepts=([1-9][0-9]*)"
               + r" done_accepts=1 fault_binary_clean=1 registered_fault_public_zero=1")
    hits = re.findall(pattern, text)
    need(len(hits) == 1, "M1831 runtime PASS")
    need(int(hits[0][0]) == EVENTS[case_id]
         and int(hits[0][1]) == PACKETS[case_id], "source denominator")
    display = "K8" if axis == "k8" else "K1x8"
    old_pass = ("PASS M979 mapped replay axis=" + display
                + " case=" + str(case_id) + " events=" + str(EVENTS[case_id])
                + " cycles=" + str(AXES[axis]["cycles"][case_id]))
    need(text.count(old_pass) == 1, "M979 mapped cycle PASS")
    need(text.count("PASS M1334 coverage case=" + str(case_id)) == 1,
         "M1334 coverage PASS")
    return {"log_sha256": sha(path), "accepted_sources": EVENTS[case_id],
            "endpoint_accepts": int(hits[0][2])}


def validate_saif(path, axis, case_id, cycles):
    need(axis in AXES and case_id in range(5), "axis/case")
    need(cycles == AXES[axis]["cycles"][case_id], "cycle anchor")
    path = Path(path); need(path.is_file() and not path.is_symlink(), "SAIF regular")
    root = parse_saif(path.read_text(errors="strict"))
    duration = forms(root, "DURATION")
    need(len(duration) == 1 and abs(float(duration[0][1]) - cycles * 3.0) <= 1e-6,
         "SAIF duration")
    tx = all_forms(root, "TX")
    need(tx and all(len(item) == 2 and float(item[1]) == 0.0 for item in tx),
         "SAIF TX")
    top = direct_instance(root, TOP)
    implementation = direct_instance(direct_instance(
        direct_instance(top, "core"), "dut"), "implementation")
    activity = activity_under(implementation); need(activity, "empty DUT SAIF")
    for name, prefixes in {"clock": ("clk_core",),
            "source": ("raw_valid", "raw_accept", "raw_bitmap"),
            "commit": ("result_valid", "result_accept", "result_accumulator"),
            "done": ("token_done_valid", "token_done_accept")}.items():
        need(cone(activity, prefixes) > 0.0, "zero cone " + name)
    for fault in ("protocol_error", "numeric_overflow", "stale_response_seen"):
        hits = [name for name in activity if name == fault or name.startswith(fault + "[")]
        need(hits and sum(activity[name] for name in hits) == 0.0,
             "fault toggled " + fault)
    need(cone(activity, ("rst_core",)) == 0.0, "reset toggled")
    return {"status": "PASS_M1831_DUT_ONLY_MAPPED_SAIF",
            "axis": axis, "case": case_id, "cycles": cycles,
            "accepted_sources": EVENTS[case_id], "saif_sha256": sha(path)}


POWER_FIELDS = ("Net Switching Power", "Cell Internal Power",
                "Cell Leakage Power", "Total Power")


def parse_power_report(path):
    text = Path(path).read_text(errors="strict"); values = {}
    need("Report : Averaged Power" in text and "-unit mW" in text,
         "power mode/unit")
    for field in POWER_FIELDS:
        hits = re.findall(re.escape(field) + r"\s*=\s*([0-9.eE+-]+)", text)
        need(len(hits) == 1, "power field " + field)
        value = float(hits[0]); need(math.isfinite(value) and value >= 0.0,
                                     "power value " + field)
        values[field] = value
    need(values["Total Power"] > 0.0, "total power")
    need(abs(sum(values[field] for field in POWER_FIELDS[:3]) - values["Total Power"])
         <= max(1e-6, values["Total Power"] * 1e-4), "power subtotal")
    return {"net_switching_mw": values[POWER_FIELDS[0]],
            "cell_internal_mw": values[POWER_FIELDS[1]],
            "cell_leakage_mw": values[POWER_FIELDS[2]],
            "total_mw": values[POWER_FIELDS[3]]}


def aggregate_metrics(entries):
    need(len(entries) == 10, "ten coordinates")
    need(set((row["axis"], row["case"]) for row in entries) ==
         set((axis, case) for axis in AXES for case in range(5)),
         "Cartesian product")
    axes = {}
    for axis in AXES:
        rows = sorted((row for row in entries if row["axis"] == axis),
                      key=lambda row: row["case"])
        need([row["cycles"] for row in rows] == AXES[axis]["cycles"],
             "cycle rows")
        total_cycles = sum(row["cycles"] for row in rows)
        total_energy = sum(row["total_mw"] * row["cycles"] * 3.0 for row in rows)
        axes[axis] = {"cycles": total_cycles, "total_energy_pj": total_energy,
                      "average_power_mw": total_energy / (total_cycles * 3.0)}
    return {"axes": axes,
            "equal_bandwidth_cycle_speedup_k8_vs_k1x8":
                axes["k1x8"]["cycles"] / axes["k8"]["cycles"],
            "equal_bandwidth_energy_ratio_k1x8_over_k8":
                axes["k1x8"]["total_energy_pj"] / axes["k8"]["total_energy_pj"],
            "logic_only_premacro": True, "external_weight_sram_excluded": True}


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
    if args.mode == "source": output = validate_sources()
    elif args.mode == "saif":
        need(args.axis is not None and args.case_id is not None
             and args.cycles is not None and args.saif and args.log, "SAIF args")
        output = validate_saif(args.saif, args.axis, args.case_id, args.cycles)
        output["runtime"] = validate_runtime_log(args.log, args.axis, args.case_id)
    else:
        need(args.power_report is not None, "power args")
        output = parse_power_report(args.power_report)
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
