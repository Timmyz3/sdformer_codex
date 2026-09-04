#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1110D independent hammer of M1105Dr2; source-only, no runner/production/EDA/RTL."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
import tempfile
from typing import Any, Callable


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1105dr2_decoder_only_address_timed_source.py"
CONTRACT = HW / "contracts/m1105dr2_decoder_only_address_timed_source_contract_r2_20260830.json"
AUTHOR = HW / "reviews/m1105dr2_decoder_source_trust_root_author_receipt_r1_20260830"
OLD_STOP = HW / "reviews/m1106d_m1105d_decoder_source_contract_independent_hammer_r1_20260830"
PAYLOAD = HW / "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = HERE / "mechanical_checks.json"

EXPECTED = {
    "source": "b2d8ef4139283de06b7e332429bdf752ad16122ffbeda0ff7d75bce6d816a5c4",
    "contract": "cdbae0362d3ea093dbcb318aa2efad04e70677f8d984a9908cda44b0de3b80a4",
    "contract_side": "37cdc8aa6b0c31103affa46f1aea80f073689540b16a40ea0eec68904a0fb4fe",
    "contract_outer": "4f95a616e16530bc30f94b68235247f7c7abe1b32956fc981412b3b1576193d3",
    "author_review": "16a628bb69d12b41a421d16dc1af5a9da0ae7593cfeeb9105a71ebc57bd9f952",
    "author_manifest": "e05ddc0c29c371e6a9b719a9e167b59ac2cecc33a51aac2959d0bd4b2a558cd8",
    "author_outer": "d16257e342be49f6e895bd1ca4b4c764eb6200da47bd27aa221abba7e6f6af25",
    "old_stop_outer": "eb5fc732c83c533f4637f87e0727dfaa57019014f14cb43423f26fc736ff1132",
    "manifest": "e2d7c92a038c213b590603ff534a33f3579bf1224cc3f56c11629e1d4c813dc0",
    "payload_outer": "eaf975a9a1a4829b2c0a2251e7ef297abd53b83b30e23630e5ce51db5c5de18c",
    "checkpoint": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "contract_leaf_digest": "a4551d23ed3298206e4f1e1c2a36a943f1fbf66f46e1b0b49f6610dc5160a9de",
    "call_digest": "c7bc8d82468d1ba604ae676e568117451de7ddf47b13a7dee91a78766a9f1552",
    "d1_digest": "416151dc4e70b9c0eb7c02065165ba390ccf8b6600bf97554a8ef97cad266d7d",
}
SEQUENCES = ["interlaken_01_a", "thun_01_b", "zurich_city_12_a"]
MODULES = {
    0: ("sttmultires_unet.decoders.0.deconv.0", (10, 1, 1536, 15, 20),
        "EXACT_BINARY_BITPACK"),
    1: ("sttmultires_unet.decoders.1.deconv.0", (10, 1, 770, 30, 40),
        "EXACT_SCALED_BINARY_BITPACK"),
    2: ("sttmultires_unet.decoders.2.deconv.0", (10, 1, 386, 60, 80),
        "EXACT_BINARY_BITPACK"),
    3: ("sttmultires_unet.decoders.3.deconv.0", (10, 1, 194, 120, 160),
        "EXACT_BINARY_BITPACK"),
}
THETA = 1065353139
THETA_HEX = "b3ff7f3f"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " is not canonical regular file")
    require(sha(path) == expected, label + " SHA drift")


def strict_json(path: Path) -> Any:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("non-finite JSON " + token)))


def safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    require(not member.is_absolute() and ".." not in member.parts and
            member.as_posix() == name, "unsafe sealed member")
    return member


def verify_sealed(directory: Path, outer_sha: str,
                  selected: dict[str, str] | None = None) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(), "sealed directory symlink/absent")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(outer, outer_sha, "sealed outer")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64 and fields[1] not in expected,
                "bad/duplicate sealed member")
        member = directory.joinpath(*safe_member(fields[1]).parts)
        regular(member, fields[0], "sealed member " + fields[1])
        expected[fields[1]] = fields[0]
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(expected), "sealed directory coverage drift")
    require(sha(manifest) == outer.read_text(encoding="utf-8").split()[0] and
            outer.read_text(encoding="utf-8").split()[1] == "SHA256SUMS",
            "sealed directory outer content drift")
    if selected:
        for name, digest in selected.items():
            require(expected.get(name) == digest, "selected sealed member drift " + name)
    return {"members": len(expected), "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def leaves(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    def walk(item: Any, path: tuple[str, ...]) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                walk(item[key], path + (key,))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, path + (str(index),))
        else:
            result.append({"path": "/".join(path), "type": type(item).__name__,
                           "value": item})
    walk(value, ())
    return result


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_source():
    regular(SOURCE, EXPECTED["source"], "source")
    spec = importlib.util.spec_from_file_location("m1110d_frozen_m1105dr2", SOURCE)
    require(spec is not None and spec.loader is not None, "source import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def independently_validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    projection = leaves(contract)
    require(len(projection) == 136 and json_digest(projection) == EXPECTED["contract_leaf_digest"],
            "136-leaf projection drift")
    resource = contract["common_resource_schedule_schema"]
    require(resource["lanes"] == 96 and resource["accumulator_bits"] == 24 and
            resource["clock_ns"] == 3.0 and resource["external_bytes_per_cycle"] == 192 and
            resource["onchip_sram_bytes_macro_rounded"] == 245760 and
            sum(resource["partitions"].values()) == 245760 and
            resource["ports"]["weight"] == {"banks": 8, "mode": "1R1W", "row_bytes": 16,
                "read_latency_cycles": 4, "initiation_interval": 1, "outstanding_per_bank": 8} and
            resource["ports"]["psum"] == {"banks": 6, "mode": "1RW", "row_bytes": 48,
                "read_latency_cycles": 2, "write_latency_cycles": 1,
                "initiation_interval": 1, "outstanding_per_bank": 8},
            "resource contract drift")
    require(len(set(resource["address_regions"].values())) == 5 and
            all(text.startswith("0x" + str(index) + "_") for index, text in enumerate(
                resource["address_regions"].values(), start=1)), "address contract drift")
    event = contract["transaction_event_schema"]
    require(event["required_dependency_fields"] == ["dependency_tokens", "produces_token"] and
            event["required_time_fields"] == ["earliest_issue_cycle", "dependency_ready_cycle",
                "issue_cycle", "return_cycle", "commit_cycle", "stall_class"] and
            "remain absent" in event["time_policy"], "dependency/time contract drift")
    d1 = contract["d1_numeric_contract"]
    require(d1["theta_word_uint32"] == THETA and d1["theta_ieee754_le_hex"] == THETA_HEX and
            d1["weight_folding_allowed"] is False and
            d1["coercion_to_binary_one_allowed"] is False, "D1 contract drift")
    population = contract["population"]
    require(population["checkpoint"] == "H67_ep35" and
            population["checkpoint_sha256"] == EXPECTED["checkpoint"] and
            population["expected_calls"] == 120 and population["expected_packed_bytes"] == 261090000 and
            population["final_checkpoint_rebind_required_if_changed"] is True and
            set(population["final_checkpoint_rebind_scope"]) == {"payload_activity", "D1_theta_identity",
                "weight_identity", "numeric_miters", "transaction_population", "cycles", "traffic",
                "energy", "system_table"}, "checkpoint/rebind contract drift")
    require("m700" not in json.dumps(contract, sort_keys=True).lower() and
            contract["release"]["production_run_allowed"] is False and
            contract["release"]["production_replay_authorized"] is False and
            all(contract["claim_boundary"][key] is False for key in ("external_opportunity_result_admitted",
                "production_transactions", "cycles", "traffic", "speedup", "system_speedup",
                "ours_performance", "rtl", "eda", "energy", "ppa")), "M700/release/claim drift")
    return {"leaf_count": len(projection), "leaf_digest_sha256": json_digest(projection)}


def reconstruct_d1(path: Path, elements: int) -> str:
    digest = hashlib.sha256()
    seen = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(1 << 18)
            if not block:
                break
            take = min(elements - seen, len(block) * 8)
            words = bytearray(take * 4)
            for index in range(take):
                bit = (block[index >> 3] >> (index & 7)) & 1
                struct.pack_into("<I", words, index * 4, THETA if bit else 0)
            digest.update(words)
            seen += take
    require(seen == elements and path.stat().st_size == (elements + 7) // 8,
            "independent D1 packed population drift")
    return digest.hexdigest()


def validate_population(result: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    require(result["population"] == {"sequences": 3, "samples": 30, "calls": 120,
        "packed_bytes": 261090000, "global_ordinals_contiguous": True,
        "per_sample_module_order": ["D0", "D1", "D2", "D3"]}, "result population drift")
    rows = manifest["records"]
    calls = result["calls"]
    require(len(rows) == len(calls) == 120, "120-call population drift")
    packed_bytes = 0
    d1_records: list[dict[str, Any]] = []
    for ordinal, (row, call) in enumerate(zip(rows, calls)):
        module = ordinal % 4
        sample = ordinal // 4
        name, shape, route = MODULES[module]
        require(row["global_call_index"] == call["global_ordinal"] == ordinal and
                row["global_sample_id"] == call["global_sample_id"] == sample and
                row["module_index"] == call["module_ordinal"] == module and
                row["sequence"] == call["sequence"] == SEQUENCES[sample // 10] and
                row["sequence_sample_id"] == call["sequence_sample_id"] == sample % 10 and
                row["name"] == call["module"] == name and tuple(row["input_shape"]) == shape and
                row["route"] == call["route"] == route, "call order/identity drift")
        stats = row["statistics"]["scaled_binary_audit"] if module == 1 else row["statistics"]
        member = safe_member(row["relative_path"])
        path = PAYLOAD.joinpath(*member.parts)
        regular(path, stats["packed_sha256"], "payload " + str(ordinal))
        require(path.stat().st_size == stats["packed_bytes"] and
                call["payload_sha256"] == stats["packed_sha256"] and
                call["payload_relative_path"] == row["relative_path"], "payload projection drift")
        packed_bytes += path.stat().st_size
        addresses = call["address_regions"]
        stride = 1 << 32
        expected_addresses = {"input_descriptor_base": (1 << 60) + ordinal * stride,
            "weight_base": (2 << 60) + module * stride,
            "psum_base": (3 << 60) + ordinal * stride,
            "output_commit_base": (4 << 60) + ordinal * stride,
            "control_descriptor_base": (5 << 60) + ordinal * stride,
            "per_call_region_bytes": stride}
        require(addresses == expected_addresses and len({value >> 60 for key, value in addresses.items()
            if key.endswith("_base")}) == 5, "per-call address overlap/drift")
        if module == 1:
            rebuilt = reconstruct_d1(path, stats["elements"])
            require(rebuilt == row["raw_fp32_content_sha256"], "independent D1 miter mismatch")
            d1_records.append({"global_call_index": ordinal,
                "packed_sha256": stats["packed_sha256"],
                "reconstructed_raw_fp32_sha256": rebuilt,
                "expected_raw_fp32_sha256": row["raw_fp32_content_sha256"], "mismatch": False})
    require(packed_bytes == 261090000 and len(d1_records) == 30 and
            json_digest(calls) == EXPECTED["call_digest"] and
            json_digest(d1_records) == EXPECTED["d1_digest"], "population/digest drift")
    return {"calls": len(calls), "packed_bytes": packed_bytes, "d1_calls": len(d1_records),
            "d1_mismatches": 0, "call_digest_sha256": json_digest(calls),
            "d1_records_digest_sha256": json_digest(d1_records)}


def reject(function: Callable[[], Any]) -> bool:
    try:
        function()
    except (Exception, SystemExit):
        return True
    return False


def contract_mutations(module, contract: dict[str, Any]) -> dict[str, bool]:
    tests: dict[str, bool] = {}
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("leaf_delete", lambda x: x["claim_boundary"].pop("ppa")),
        ("leaf_add", lambda x: x.__setitem__("forged", 1)),
        ("lanes", lambda x: x["common_resource_schedule_schema"].__setitem__("lanes", 95)),
        ("accumulator", lambda x: x["common_resource_schedule_schema"].__setitem__("accumulator_bits", 23)),
        ("clock", lambda x: x["common_resource_schedule_schema"].__setitem__("clock_ns", 4.0)),
        ("bandwidth", lambda x: x["common_resource_schedule_schema"].__setitem__("external_bytes_per_cycle", 191)),
        ("sram", lambda x: x["common_resource_schedule_schema"].__setitem__("onchip_sram_bytes_macro_rounded", 262144)),
        ("port", lambda x: x["common_resource_schedule_schema"]["ports"]["psum"].__setitem__("mode", "1R1W")),
        ("address_overlap", lambda x: x["common_resource_schedule_schema"]["address_regions"].__setitem__(
            "psum", x["common_resource_schedule_schema"]["address_regions"]["input_descriptor"])),
        ("dependency", lambda x: x["transaction_event_schema"].__setitem__("required_dependency_fields", [])),
        ("time", lambda x: x["transaction_event_schema"].__setitem__("time_policy", "caller timestamps")),
        ("theta_word", lambda x: x["d1_numeric_contract"].__setitem__("theta_word_uint32", 1065353216)),
        ("theta_endian", lambda x: x["d1_numeric_contract"].__setitem__("theta_ieee754_le_hex", "0000803f")),
        ("fold", lambda x: x["d1_numeric_contract"].__setitem__("weight_folding_allowed", True)),
        ("coerce", lambda x: x["d1_numeric_contract"].__setitem__("coercion_to_binary_one_allowed", True)),
        ("checkpoint", lambda x: x["population"].__setitem__("checkpoint_sha256", "0" * 64)),
        ("rebind", lambda x: x["population"].__setitem__("final_checkpoint_rebind_required_if_changed", False)),
        ("rebind_scope", lambda x: x["population"].__setitem__("final_checkpoint_rebind_scope", ["cycles"])),
        ("caller_contract", lambda x: x["trust_root"].__setitem__("caller_contract_path_allowed", True)),
        ("production", lambda x: x["release"].__setitem__("production_run_allowed", True)),
        ("m700", lambda x: x.__setitem__("m700_speedup", 3.088)),
        ("speedup", lambda x: x["claim_boundary"].__setitem__("speedup", True)),
    ]
    for name, mutation in mutations:
        changed = copy.deepcopy(contract)
        mutation(changed)
        tests[name] = reject(lambda changed=changed: module.validate_contract(changed))
    require(all(tests.values()), "contract mutation escaped")
    return tests


def isolated_bytes_and_symlink_attacks(module) -> dict[str, bool]:
    tests: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="m1110d_bytes.") as raw:
        root = Path(raw)
        source_bad = root / "source.py"
        source_bad.write_bytes(SOURCE.read_bytes() + b"\n# mutation\n")
        tests["source_bytes"] = sha(source_bad) != EXPECTED["source"]
        contract_bad = root / CONTRACT.name
        contract_bad.write_bytes(CONTRACT.read_bytes() + b"\n")
        tests["contract_bytes"] = sha(contract_bad) != EXPECTED["contract"]
        source_link = root / "source_link.py"
        source_link.symlink_to(SOURCE)
        tests["source_symlink"] = reject(lambda: module.verify_regular(source_link, EXPECTED["source"]))
        tests["unsafe_parent_member"] = reject(lambda: module.safe_member("../escape"))
        tests["unsafe_absolute_member"] = reject(lambda: module.safe_member("/absolute"))

        copied = root / "author_copy"
        copied.mkdir()
        manifest = AUTHOR / "SHA256SUMS"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            _, name = line.split("  ", 1)
            (copied / name).write_bytes((AUTHOR / name).read_bytes())
        (copied / "SHA256SUMS").write_bytes(manifest.read_bytes())
        (copied / "SHA256SUMS.seal.sha256").write_bytes(
            (AUTHOR / "SHA256SUMS.seal.sha256").read_bytes())
        (copied / "review.json").write_bytes((copied / "review.json").read_bytes() + b"\n")
        tests["author_receipt_bytes"] = reject(lambda: verify_sealed(copied, EXPECTED["author_outer"]))

        member_root = root / "member_symlink"
        member_root.mkdir()
        target = root / "target"
        target.write_text("canonical\n", encoding="utf-8")
        (member_root / "member").symlink_to(target)
        member_hash = sha(target)
        (member_root / "SHA256SUMS").write_text(member_hash + "  member\n", encoding="utf-8")
        manifest_hash = sha(member_root / "SHA256SUMS")
        (member_root / "SHA256SUMS.seal.sha256").write_text(
            manifest_hash + "  SHA256SUMS\n", encoding="utf-8")
        tests["sealed_member_symlink"] = reject(lambda: verify_sealed(
            member_root, sha(member_root / "SHA256SUMS.seal.sha256")))
        directory_link = root / "directory_link"
        directory_link.symlink_to(AUTHOR, target_is_directory=True)
        tests["sealed_directory_symlink"] = reject(lambda: verify_sealed(
            directory_link, EXPECTED["author_outer"]))
    require(all(tests.values()), "byte/symlink mutation escaped")
    return tests


def main() -> None:
    regular(CONTRACT, EXPECTED["contract"], "contract")
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    regular(side, EXPECTED["contract_side"], "contract side")
    regular(outer, EXPECTED["contract_outer"], "contract outer")
    require(side.read_text(encoding="utf-8").split() == [EXPECTED["contract"], CONTRACT.name] and
            outer.read_text(encoding="utf-8").split() == [EXPECTED["contract_side"], side.name],
            "contract triple content drift")
    author_identity = verify_sealed(AUTHOR, EXPECTED["author_outer"],
        {"review.json": EXPECTED["author_review"]})
    require(author_identity["manifest_sha256"] == EXPECTED["author_manifest"],
            "author receipt manifest drift")
    author_review = strict_json(AUTHOR / "review.json")
    require(author_review["identity"]["source_sha256"] == EXPECTED["source"] and
            author_review["identity"]["contract_sha256"] == EXPECTED["contract"] and
            author_review["identity"]["contract_sidecar_sha256"] == EXPECTED["contract_side"] and
            author_review["identity"]["contract_outer_seal_file_sha256"] == EXPECTED["contract_outer"] and
            author_review["identity"]["m1106d_stop_outer_seal_file_sha256"] == EXPECTED["old_stop_outer"],
            "author source/contract/STOP binding drift")
    old_stop_identity = verify_sealed(OLD_STOP, EXPECTED["old_stop_outer"])
    require(strict_json(OLD_STOP / "review.json")["status"] ==
            "STOP_M1106D_CALLER_CONTRACT_FORGERY__NO_PRODUCTION_RUNNER",
            "M1106D STOP drift")
    regular(DOCS359, EXPECTED["docs359"], "docs359")
    payload_identity = verify_sealed(PAYLOAD, EXPECTED["payload_outer"],
        {"manifest.json": EXPECTED["manifest"]})

    module = load_source()
    source_text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require(len(functions["build_canonical"].args.args) == 0 and
            "argparse" not in source_text and "os.environ" not in source_text and
            "getenv" not in source_text and "subprocess" not in source_text and
            "os.system" not in source_text and "Popen" not in source_text,
            "caller path/env or execution primitive present")
    caller_argument_rejections = {
        token: reject(lambda token=token: module.main([token, "/tmp/forged"]))
        for token in ("--repo-root", "--contract", "--output", "--runner", "--production")
    }
    require(all(caller_argument_rejections.values()) and
            reject(lambda: module.build_canonical(CONTRACT)) and
            reject(lambda: module.build_canonical(HW.parent, CONTRACT)),
            "caller authority accepted")

    contract = strict_json(CONTRACT)
    contract_projection = independently_validate_contract(contract)
    mutation_results = contract_mutations(module, contract)
    byte_symlink_results = isolated_bytes_and_symlink_attacks(module)

    old_environment = dict(os.environ)
    try:
        os.environ.update({"M1105D_REPO_ROOT": "/tmp/forged-repo",
            "M1105D_CONTRACT": "/tmp/forged-contract", "M1105D_OUTPUT": "/tmp/forged-output",
            "M1105DR2_EXPECTED_SHA256": "0" * 64, "M700_SPEEDUP": "99.0"})
        result = module.build_canonical()
    finally:
        os.environ.clear()
        os.environ.update(old_environment)
    require(result["status"] ==
            "PASS_M1105DR2_FIXED_TRUST_SOURCE_PREFLIGHT__PRODUCTION_NOT_RELEASED",
            "canonical source preflight status drift")
    manifest = strict_json(PAYLOAD / "manifest.json")
    population = validate_population(result, manifest)
    require(result["common_resource_schedule_schema"] == contract["common_resource_schedule_schema"] and
            result["transaction_event_schema"] == contract["transaction_event_schema"],
            "result resource/event schema drift")
    require(result["d1_exact_scaled_binary_miter"]["calls_checked"] == 30 and
            result["d1_exact_scaled_binary_miter"]["mismatches"] == 0 and
            result["d1_exact_scaled_binary_miter"]["theta_word"] == THETA and
            result["d1_exact_scaled_binary_miter"]["folded_weights"] is False and
            result["d1_exact_scaled_binary_miter"]["coerced_to_one"] is False,
            "source D1 result drift")
    require(result["external_baseline_rejection"] == {"m700_admitted": False,
            "ours_cycles_from_external_artifact": False} and
            result["input_identity"]["checkpoint_sha256"] == EXPECTED["checkpoint"] and
            result["input_identity"]["final_checkpoint_rebind_required_if_changed"] is True and
            result["release"]["production_run_allowed"] is False and
            result["release"]["production_cycles"] is None and result["release"]["speedup"] is None and
            all(result["claim_boundary"][key] is False for key in ("production_transactions", "cycles",
                "traffic", "speedup", "system_speedup", "ours_performance", "rtl", "eda", "energy", "ppa")),
            "checkpoint/M700/release boundary drift")
    require(reject(lambda: module.d1_scaled_binary_raw_sha(
        PAYLOAD / manifest["records"][1]["relative_path"], 1, 1065353216)),
        "wrong D1 theta accepted")
    require(sha(DOCS359) == EXPECTED["docs359"], "docs359 changed during hammer")

    output = {
        "schema": "m1110d_m1105dr2_decoder_source_contract_receipt_independent_hammer_checks_v1",
        "status": "PASS_M1110D_M1105DR2_FIXED_TRUST_SOURCE_HAMMER__RUNNER_AUTHORING_ONLY",
        "score": 100,
        "identity": {"source_sha256": sha(SOURCE), "contract_sha256": sha(CONTRACT),
            "contract_sidecar_sha256": sha(side), "contract_outer_seal_file_sha256": sha(outer),
            "author_receipt_review_sha256": sha(AUTHOR / "review.json"),
            "author_receipt_manifest_sha256": sha(AUTHOR / "SHA256SUMS"),
            "author_receipt_outer_seal_file_sha256": sha(AUTHOR / "SHA256SUMS.seal.sha256"),
            "m1106d_stop_outer_seal_file_sha256": sha(OLD_STOP / "SHA256SUMS.seal.sha256"),
            "docs359_sha256": sha(DOCS359)},
        "contract_projection": contract_projection,
        "canonical_population": population,
        "sealed_inputs": {"author_receipt": author_identity, "m1106d_stop": old_stop_identity,
            "m699_payload": payload_identity},
        "caller_authority": {"path_arguments_rejected": caller_argument_rejections,
            "build_positional_arguments_rejected": 2, "environment_ignored": True,
            "canonical_result_under_forged_environment": True},
        "mutation_rejections": {"contract": mutation_results,
            "bytes_and_symlinks": byte_symlink_results,
            "contract_mutations_rejected": sum(mutation_results.values()),
            "byte_symlink_mutations_rejected": sum(byte_symlink_results.values())},
        "numeric_resource_policy": {"resource_exact": True, "address_disjoint": True,
            "dependency_time_schema_exact": True, "d1_theta_exact": True,
            "checkpoint_rebind_required": True, "m700_admitted": False},
        "scope": {"production_run": False, "production_transactions": False,
            "runner_created": False, "eda": False, "rtl": False, "gpu_remote": False,
            "canonical_source_contract_receipt_modified": False, "docs359_modified": False},
        "authorization": {"different_author_runner_authoring": True,
            "production_run_now": False, "launch_now": False,
            "different_author_runner_and_launch_hammer_required": True},
        "claim_boundary": {"identity_source_only": True, "paper_citable_performance": False,
            "cycles": False, "traffic": False, "speedup": False, "system_speedup": False,
            "ours_performance": False, "rtl": False, "eda": False, "energy": False, "ppa": False},
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(output["status"])


if __name__ == "__main__":
    main()
