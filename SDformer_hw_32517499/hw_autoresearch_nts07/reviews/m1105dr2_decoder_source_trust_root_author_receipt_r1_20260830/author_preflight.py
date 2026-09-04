#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1105Dr2 author preflight; source-only, no production/EDA/RTL."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "system_simulator/scripts/build_m1105dr2_decoder_only_address_timed_source.py"
SOURCE_SHA = "b2d8ef4139283de06b7e332429bdf752ad16122ffbeda0ff7d75bce6d816a5c4"
CONTRACT = HW / "contracts/m1105dr2_decoder_only_address_timed_source_contract_r2_20260830.json"
CONTRACT_SHA = "cdbae0362d3ea093dbcb318aa2efad04e70677f8d984a9908cda44b0de3b80a4"
CONTRACT_SIDE_SHA = "37cdc8aa6b0c31103affa46f1aea80f073689540b16a40ea0eec68904a0fb4fe"
CONTRACT_OUTER_SHA = "4f95a616e16530bc30f94b68235247f7c7abe1b32956fc981412b3b1576193d3"
M1106D_OUTER = "eb5fc732c83c533f4637f87e0727dfaa57019014f14cb43423f26fc736ff1132"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


checks: list[str] = []


def require(value: bool, label: str) -> None:
    if not value:
        raise RuntimeError(label)
    checks.append(label)


def reject(function, label: str) -> None:
    try:
        function()
    except (RuntimeError, TypeError, ValueError, SystemExit):
        checks.append(label)
    else:
        raise RuntimeError(label + " accepted")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " regular")
    require(sha(path) == expected, label + " sha")


regular(SOURCE, SOURCE_SHA, "source")
regular(CONTRACT, CONTRACT_SHA, "contract")
regular(Path(str(CONTRACT) + ".sha256"), CONTRACT_SIDE_SHA, "contract side")
regular(Path(str(CONTRACT) + ".sha256.seal.sha256"), CONTRACT_OUTER_SHA,
        "contract outer")
regular(DOCS359, DOCS359_SHA, "docs359")
require(Path(str(CONTRACT) + ".sha256").read_text().split() ==
        [CONTRACT_SHA, CONTRACT.name], "contract side content")
require(Path(str(CONTRACT) + ".sha256.seal.sha256").read_text().split() ==
        [CONTRACT_SIDE_SHA, CONTRACT.name + ".sha256"], "contract outer content")

source_text = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(source_text)
require("argparse" not in source_text and "--repo-root" not in source_text and
        "--contract" not in source_text and "--output" not in source_text,
        "no caller path CLI")
require("os.environ" not in source_text and "getenv" not in source_text,
        "no caller environment read")
require('CONTRACT_SHA256 = "' + CONTRACT_SHA + '"' in source_text and
        'CONTRACT_SIDECAR_SHA256 = "' + CONTRACT_SIDE_SHA + '"' in source_text and
        'CONTRACT_OUTER_SHA256 = "' + CONTRACT_OUTER_SHA + '"' in source_text,
        "contract triple literal pin")
require("Path(__file__).absolute()" in source_text and
        "PAYLOAD_ROOT = HW /" in source_text, "source-derived roots")
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
require(len(functions["build_canonical"].args.args) == 0,
        "build canonical zero arguments")

spec = importlib.util.spec_from_file_location("m1105dr2_author_preflight_source", SOURCE)
require(spec is not None and spec.loader is not None, "source import spec")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

reject(lambda: module.build_canonical(CONTRACT), "build caller contract rejected")
reject(lambda: module.build_canonical(HW.parent, CONTRACT), "build caller repo rejected")
for argv in (["--contract", "/tmp/forged"], ["--repo-root", "/tmp"],
             ["--output", "/tmp/out"], ["unexpected"]):
    reject(lambda argv=argv: module.main(argv), "main caller argv rejected " + argv[0])

contract = module.strict_json(CONTRACT)
mutations = [
    ("lanes", lambda value: value["common_resource_schedule_schema"].__setitem__("lanes", 95)),
    ("acc24", lambda value: value["common_resource_schedule_schema"].__setitem__("accumulator_bits", 23)),
    ("clock", lambda value: value["common_resource_schedule_schema"].__setitem__("clock_ns", 4.0)),
    ("bandwidth", lambda value: value["common_resource_schedule_schema"].__setitem__("external_bytes_per_cycle", 191)),
    ("sram", lambda value: value["common_resource_schedule_schema"].__setitem__("onchip_sram_bytes_macro_rounded", 262144)),
    ("address", lambda value: value["common_resource_schedule_schema"]["address_regions"].__setitem__("psum", value["common_resource_schedule_schema"]["address_regions"]["input_descriptor"])),
    ("dependency", lambda value: value["transaction_event_schema"].__setitem__("required_dependency_fields", [])),
    ("timestamp", lambda value: value["transaction_event_schema"].__setitem__("time_policy", "caller timestamps")),
    ("theta", lambda value: value["d1_numeric_contract"].__setitem__("theta_word_uint32", 1065353216)),
    ("fold", lambda value: value["d1_numeric_contract"].__setitem__("weight_folding_allowed", True)),
    ("coerce", lambda value: value["d1_numeric_contract"].__setitem__("coercion_to_binary_one_allowed", True)),
    ("calls", lambda value: value["population"].__setitem__("expected_calls", 119)),
    ("checkpoint", lambda value: value["population"].__setitem__("checkpoint_sha256", "0" * 64)),
    ("rebind", lambda value: value["population"].__setitem__("final_checkpoint_rebind_required_if_changed", False)),
    ("caller_contract", lambda value: value["trust_root"].__setitem__("caller_contract_path_allowed", True)),
    ("production", lambda value: value["release"].__setitem__("production_run_allowed", True)),
    ("speedup", lambda value: value["claim_boundary"].__setitem__("speedup", True)),
    ("external_result", lambda value: value.__setitem__("m700_speedup", 3.088)),
]
for label, mutation in mutations:
    changed = copy.deepcopy(contract)
    mutation(changed)
    reject(lambda changed=changed: module.validate_contract(changed),
           "contract mutation rejected " + label)

with tempfile.TemporaryDirectory(prefix="m1105dr2_contract_seal_mutation.") as raw:
    changed = Path(raw) / CONTRACT.name
    changed.write_bytes(CONTRACT.read_bytes() + b"\n")
    require(sha(changed) != CONTRACT_SHA, "stale contract seal rejected")

old_environment = dict(os.environ)
try:
    os.environ["M1105D_REPO_ROOT"] = "/tmp/forged-repo"
    os.environ["M1105D_CONTRACT"] = "/tmp/forged-contract"
    os.environ["M1105D_OUTPUT"] = "/tmp/forged-output"
    os.environ["M1105D_EXPECTED_SHA256"] = "0" * 64
    result = module.build_canonical()
finally:
    os.environ.clear()
    os.environ.update(old_environment)

require(result["status"] ==
        "PASS_M1105DR2_FIXED_TRUST_SOURCE_PREFLIGHT__PRODUCTION_NOT_RELEASED",
        "canonical preflight status")
require(result["population"] == {"sequences": 3, "samples": 30, "calls": 120,
        "packed_bytes": 261090000, "global_ordinals_contiguous": True,
        "per_sample_module_order": ["D0", "D1", "D2", "D3"]},
        "canonical population")
require(result["d1_exact_scaled_binary_miter"]["calls_checked"] == 30 and
        result["d1_exact_scaled_binary_miter"]["mismatches"] == 0 and
        result["d1_exact_scaled_binary_miter"]["theta_word"] == 1065353139 and
        result["d1_exact_scaled_binary_miter"]["folded_weights"] is False and
        result["d1_exact_scaled_binary_miter"]["coerced_to_one"] is False,
        "canonical D1 miter")
require(result["common_resource_schedule_schema"] == module.EXPECTED_RESOURCE and
        result["transaction_event_schema"] == module.EXPECTED_TRANSACTION_SCHEMA,
        "canonical resource/transaction schema")
require(result["external_baseline_rejection"] == {
        "m700_admitted": False, "ours_cycles_from_external_artifact": False},
        "M700 rejection")
require(result["input_identity"]["final_checkpoint_rebind_required_if_changed"] is True and
        result["release"]["production_run_allowed"] is False and
        all(result["claim_boundary"][key] is False for key in
            ("production_transactions", "cycles", "traffic", "speedup",
             "system_speedup", "ours_performance", "rtl", "eda", "energy", "ppa")),
        "rebind/release/claim boundary")
call_digest = hashlib.sha256(json.dumps(result["calls"], sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
d1_digest = hashlib.sha256(json.dumps(
    result["d1_exact_scaled_binary_miter"]["records"], sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
require(call_digest == "c7bc8d82468d1ba604ae676e568117451de7ddf47b13a7dee91a78766a9f1552",
        "120-call digest")
require(d1_digest == "416151dc4e70b9c0eb7c02065165ba390ccf8b6600bf97554a8ef97cad266d7d",
        "D1 record digest")
require(sha(DOCS359) == DOCS359_SHA, "docs359 unchanged")

runner_candidates = list(HW.glob("**/*m1105dr2*runner*"))
attempt_candidates = list((HW / "results").glob("*m1105dr2*attempt*"))
require(not runner_candidates, "no r2 runner")
require(not attempt_candidates, "no r2 attempt")

print(json.dumps({
    "schema": "m1105dr2_decoder_source_trust_root_author_preflight_v1",
    "status": "PASS_M1105DR2_FIXED_TRUST_AUTHOR_PREFLIGHT__INDEPENDENT_HAMMER_REQUIRED",
    "checks_passed": len(checks),
    "identity": {"source_sha256": SOURCE_SHA, "contract_sha256": CONTRACT_SHA,
        "contract_sidecar_sha256": CONTRACT_SIDE_SHA,
        "contract_outer_seal_file_sha256": CONTRACT_OUTER_SHA,
        "m1106d_stop_outer_seal_file_sha256": M1106D_OUTER,
        "docs359_sha256": DOCS359_SHA},
    "trust": {"contract_leaf_count": 136,
        "contract_leaf_digest_sha256":
            "a4551d23ed3298206e4f1e1c2a36a943f1fbf66f46e1b0b49f6610dc5160a9de",
        "caller_path_arguments_rejected": 6,
        "caller_contract_mutations_rejected": len(mutations),
        "caller_environment_authority": False,
        "source_contract_hash_cycle": False,
        "source_and_contract_bound_here": True},
    "preflight": {"calls": 120, "d1_calls": 30, "d1_mismatches": 0,
        "call_digest_sha256": call_digest, "d1_records_digest_sha256": d1_digest,
        "m700_admitted": False, "final_checkpoint_rebind_required": True,
        "production_run_allowed": False},
    "execution": {"runner_created": False, "attempt_created": False,
        "production_transactions_enumerated": False, "eda_rtl_commands": 0},
}, indent=2, sort_keys=True))
