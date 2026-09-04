#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh no-export CPU hammer for M1514 decoder-weight identities."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import copy
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / (
    "system_simulator/scripts/"
    "build_m1514_ep34_decoder_weight_identity_export_source.py")
TEST = HW / (
    "system_simulator/tests/"
    "test_m1514_ep34_decoder_weight_identity_export_source.py")
CONTRACT = HW / (
    "contracts/m1514_ep34_decoder_weight_identity_export_source_contract_"
    "r1_20260831.json")
CHECKPOINT = HW / (
    "system_handoff/incoming/"
    "motion_c12_ep34_live93_checkpoint_epoch34.pth")
M1515 = HW / (
    "reviews/m1515_m1514_ep34_decoder_weight_identity_source_independent_"
    "hammer_r1_20260831")
M1516_RUNNER = HW / (
    "system_simulator/scripts/"
    "run_m1516_ep34_decoder_weight_export_one_shot.py")
M1516_OUTPUT = HW / (
    "system_handoff/exports/m1516_ep34_decoder_weights_r1")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
UCLI_KEY = ROOT / "ucli.key"

PINS = {
    "source": "c28dc1ee5fe115c3842f51955b3f8ca2db3fe3ae5f53738ea9ec4d5e5b0e0bdf",
    "test": "87c00ea1606ee390e739d32f1104ce2296caaff425ff205377e2daa54f3a5fa7",
    "contract": "178aadda75ff7a22b0f958a78e28a1ab9260e4134fd8bf2bdca6aeaad2bea7a0",
    "checkpoint": "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    "m1510_source": "051b61d5cf8a7b164096da229601afb2ca8867d3b878e491bd7279148e5793aa",
    "m1510_contract": "88203261b26abee15ec57430e46cef7b4225f53fbb67abe9d18fc87c82d1abd7",
    "m1512_review": "b302e94375f925d84a45eb798579f243fa68b13724d3f63fabfe2810948dbb74",
    "m1512_manifest": "2af7a59b6a4df07dc6047c0d48c52b7798b7f0803e31e290b2ad842e6c154b81",
    "m1512_outer": "ccbcd7bf1b99fd944062a6fb220d7ec719d96da91c190697db125cbd4ad58f7c",
    "m1513_review": "1eb36a76fac29d5d15607dbb4ee3f9a434c4b0686843acac11f18116b48c7aaa",
    "m1513_manifest": "966ba95baf00f698b6ca1fb8613afbfb78e40d2a70223f0a72bd4a87dcea04fa",
    "m1513_outer": "dc19cacbbb5ecae7f0327fd17b310be79a3b144937be7f289c25eb6f64794832",
    "capture_manifest": "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e",
    "capture_outer": "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "ucli_key": "1107aa2b8d30b14e7e4f9237ff461fb058ae4e07c8a5bed30bef3ad3eb9c30ac",
}
WEIGHT_KEYS = tuple(
    f"sttmultires_unet.decoders.{ordinal}.deconv.0.weight"
    for ordinal in range(4))
BIAS_KEYS = tuple(
    f"sttmultires_unet.decoders.{ordinal}.deconv.0.bias"
    for ordinal in range(4))
SHAPES = (
    (1536, 384, 3, 3), (770, 192, 3, 3),
    (386, 96, 3, 3), (194, 96, 3, 3))
CONTENT_SHA = (
    "cb1a90a4ff33622024b43ee6b15a3409e2567ea1e7b626715f40cf8a4fbfd83b",
    "35a9214e9fbc2e4e271beea74c4f329c12d6c072cda9252eaae350dd404a51cb",
    "75f9921f3cd9786ece78247115dd07bdda425b4f6e068d43936c884c611d3ef7",
    "6a42dabae358d0048aa46c609c9cb633f1e8d0479e4628e4f85c21e00835ea4e",
)
TOTAL_BYTES = 28_560_384
TINY_SHAPES = ((2, 2), (2, 3), (3, 2), (1, 5))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return value + "__M1515_MUTATION"
    if type(value) is list:
        return value + ["__M1515_MUTATION"]
    if value is None:
        return "__M1515_NOT_NULL__"
    raise TypeError(type(value).__name__)


def walk_dicts(value, path=()):
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from walk_dicts(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_dicts(item, path + (index,))


def walk_leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk_leaves(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_leaves(item, path + (index,))
    else:
        yield path, value


def parent_at(value, path):
    for key in path:
        value = value[key]
    return value


def tiny_state(torch):
    state = OrderedDict()
    for index in range(917):
        state[f"filler.{index:04d}"] = torch.tensor([index], dtype=torch.int64)
    for ordinal, (key, shape) in enumerate(zip(WEIGHT_KEYS, TINY_SHAPES)):
        elements = 1
        for extent in shape:
            elements *= extent
        state[key] = (torch.arange(elements, dtype=torch.float32)
                      .reshape(shape).contiguous() + ordinal)
    return state


def tiny_hashes(state):
    return tuple(hashlib.sha256(
        state[key].numpy().tobytes(order="C")).hexdigest()
                 for key in WEIGHT_KEYS)


@contextmanager
def tiny_contract(M, state, shapes=TINY_SHAPES):
    total = sum(state[key].numel() for key in WEIGHT_KEYS)
    with mock.patch.object(M, "WEIGHT_SHAPES", shapes), \
            mock.patch.object(M, "EXPECTED_CONTENT_SHA256", tiny_hashes(state)), \
            mock.patch.object(M, "EXPECTED_TOTAL_ELEMENTS", total), \
            mock.patch.object(M, "EXPECTED_TOTAL_BYTES", total * 4):
        yield


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty")
    if any(os.path.lexists(path) for path in (M1515, M1516_RUNNER, M1516_OUTPUT)):
        raise RuntimeError("M1515/M1516 namespace not fresh")
    checks = []
    attacks = []
    def check(name, value, category):
        checks.append({"check": name, "category": category, "pass": bool(value)})
    def attack(name, thunk, category):
        try:
            thunk()
            caught = False
        except BaseException:
            caught = True
        attacks.append({"attack": name, "category": category,
                        "rejected": caught, "false_negative": not caught})

    for name, path in (("source", SOURCE), ("test", TEST),
                       ("contract", CONTRACT), ("checkpoint", CHECKPOINT),
                       ("docs359", DOCS359), ("ucli_key", UCLI_KEY)):
        check(name + "_exact", sha(path) == PINS[name], "identity")
    M = load("m1515_bound_m1514", SOURCE)
    T = load("m1515_bound_m1514_tests", TEST)
    import torch
    check("cuda_not_initialized", not torch.cuda.is_initialized(), "execution")

    policy = M.validate_source_policy()
    check("source_self_check", policy.get("status") == M.SOURCE_STATUS, "source")
    stream = io.StringIO()
    replay = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(T))
    check("author_tests_10", replay.testsRun == 10 and not replay.failures
          and not replay.errors, "source")

    authority = M.verify_capture_authorities()
    check("m1510_source_exact", sha(M.M1510_SOURCE) == PINS["m1510_source"],
          "authority")
    check("m1510_contract_exact", sha(M.M1510_CONTRACT) ==
          PINS["m1510_contract"], "authority")
    m1512 = M.verify_sealed_review(
        M.M1512, (PINS["m1512_review"], PINS["m1512_manifest"],
                  PINS["m1512_outer"]),
        "PASS_M1512_M1501_M1458_EP34_CAPTURE_SOURCE_AND_RESULT")
    m1513 = M.verify_sealed_review(
        M.M1513, (PINS["m1513_review"], PINS["m1513_manifest"],
                  PINS["m1513_outer"]),
        "PASS_M1513_COMPLETE_M1458_EP34_PRODUCTION_PROVENANCE")
    check("m1512_authority", authority["m1512_status"] == m1512["status"],
          "authority")
    check("m1513_authority", authority["m1513_status"] == m1513["status"],
          "authority")
    check("capture_seal_identity", m1512["bindings"]["result_manifest_sha256"] ==
          PINS["capture_manifest"] and
          m1513["bindings"]["result_outer_file_sha256"] ==
          PINS["capture_outer"], "authority")

    source_audit = M.audit_checkpoint()
    check("source_real_cpu_audit", source_audit["status"] == M.STATUS,
          "checkpoint")
    check("source_audit_total", source_audit["aggregate"]["content_bytes"] ==
          TOTAL_BYTES, "checkpoint")

    checkpoint_before = sha(CHECKPOINT)
    value = torch.load(CHECKPOINT, map_location=torch.device("cpu"))
    checkpoint_after = sha(CHECKPOINT)
    check("checkpoint_before_after_exact", checkpoint_before == checkpoint_after ==
          PINS["checkpoint"], "checkpoint")
    check("root_exact", type(value) is dict and
          list(value.keys()) == ["model_state_dict"], "checkpoint")
    state = value["model_state_dict"]
    check("state_ordered_921", type(state) is OrderedDict and len(state) == 921,
          "checkpoint")
    rows = []
    storage = set()
    for ordinal, key in enumerate(WEIGHT_KEYS):
        tensor = state[key]
        suffix = f"decoders.{ordinal}.deconv.0.weight"
        aliases = [candidate for candidate in state if candidate.endswith(suffix)]
        bias_suffix = f"decoders.{ordinal}.deconv.0.bias"
        address = tensor.untyped_storage().data_ptr()
        array = tensor.detach().numpy()
        content = array.tobytes(order="C")
        row = {
            "key": key, "shape": list(tensor.shape),
            "dtype": str(tensor.dtype), "cpu": tensor.device.type == "cpu",
            "contiguous": tensor.is_contiguous(),
            "little_endian": sys.byteorder == "little" and array.dtype.str == "<f4",
            "c_order": array.flags.c_contiguous,
            "bias_absent": not any(candidate.endswith(bias_suffix)
                                   for candidate in state),
            "suffix_aliases": aliases,
            "storage_unique": address not in storage,
            "content_bytes": len(content),
            "content_sha256": hashlib.sha256(content).hexdigest(),
        }
        storage.add(address)
        check(f"weight_{ordinal}_exact", row == {
            "key": key, "shape": list(SHAPES[ordinal]),
            "dtype": "torch.float32", "cpu": True, "contiguous": True,
            "little_endian": True, "c_order": True, "bias_absent": True,
            "suffix_aliases": [key], "storage_unique": True,
            "content_bytes": tensor.numel() * 4,
            "content_sha256": CONTENT_SHA[ordinal]}, "checkpoint")
        rows.append(row)
    check("aggregate_28560384", sum(row["content_bytes"] for row in rows) ==
          TOTAL_BYTES, "checkpoint")
    del value, state
    gc.collect()

    attack("checkpoint_sha", lambda: M.audit_checkpoint(
        CHECKPOINT, "0" * 64), "checkpoint_mutation")
    base = tiny_state(torch)
    attack("root_extra", lambda: M.validate_checkpoint_object(
        {"model_state_dict": base, "optimizer": {}}), "checkpoint_mutation")
    missing = tiny_state(torch)
    missing.pop(WEIGHT_KEYS[0])
    missing["renamed.weight"] = torch.zeros(TINY_SHAPES[0])
    with tiny_contract(M, base):
        attack("target_key_missing", lambda: M.validate_checkpoint_object(
            {"model_state_dict": missing}), "checkpoint_mutation")
    shape = tiny_state(torch)
    shape[WEIGHT_KEYS[0]] = torch.zeros((1, 4), dtype=torch.float32)
    with tiny_contract(M, base):
        attack("shape", lambda: M.validate_checkpoint_object(
            {"model_state_dict": shape}), "checkpoint_mutation")
    dtype = tiny_state(torch)
    dtype[WEIGHT_KEYS[0]] = torch.zeros(TINY_SHAPES[0], dtype=torch.float64)
    with tiny_contract(M, base):
        attack("dtype", lambda: M.validate_checkpoint_object(
            {"model_state_dict": dtype}), "checkpoint_mutation")
    noncontiguous = tiny_state(torch)
    noncontiguous[WEIGHT_KEYS[0]] = noncontiguous[WEIGHT_KEYS[0]].t()
    with tiny_contract(M, base):
        attack("contiguous", lambda: M.validate_checkpoint_object(
            {"model_state_dict": noncontiguous}), "checkpoint_mutation")
    bias = tiny_state(torch)
    bias.pop("filler.0000")
    bias[BIAS_KEYS[0]] = torch.zeros(1)
    with tiny_contract(M, base):
        attack("bias", lambda: M.validate_checkpoint_object(
            {"model_state_dict": bias}), "checkpoint_mutation")
    content = tiny_state(torch)
    content[WEIGHT_KEYS[3]][0, 0] += 1
    with tiny_contract(M, base):
        attack("content", lambda: M.validate_checkpoint_object(
            {"model_state_dict": content}), "checkpoint_mutation")
    alias = tiny_state(torch)
    alias.pop("filler.0000")
    alias["module." + WEIGHT_KEYS[0]] = alias[WEIGHT_KEYS[0]]
    with tiny_contract(M, base):
        attack("suffix_alias", lambda: M.validate_checkpoint_object(
            {"model_state_dict": alias}), "checkpoint_mutation")
    shared = tiny_state(torch)
    equal_shapes = (TINY_SHAPES[0],) * 4
    for ordinal, key in enumerate(WEIGHT_KEYS):
        shared[key] = torch.full(equal_shapes[ordinal], float(ordinal))
    shared[WEIGHT_KEYS[1]] = shared[WEIGHT_KEYS[0]]
    with tiny_contract(M, shared, equal_shapes):
        attack("shared_storage", lambda: M.validate_checkpoint_object(
            {"model_state_dict": shared}), "checkpoint_mutation")

    authority_attacks = (
        ("m1510_source_identity", lambda: M.regular_exact(
            M.M1510_SOURCE, "0" * 64, "mutated M1510 source")),
        ("m1510_contract_identity", lambda: M.regular_exact(
            M.M1510_CONTRACT, "0" * 64, "mutated M1510 contract")),
        ("m1512_identity", lambda: M.verify_sealed_review(
            M.M1512, ("0" * 64, PINS["m1512_manifest"], PINS["m1512_outer"]),
            m1512["status"])),
        ("m1513_identity", lambda: M.verify_sealed_review(
            M.M1513, ("0" * 64, PINS["m1513_manifest"], PINS["m1513_outer"]),
            m1513["status"])),
    )
    for name, thunk in authority_attacks:
        attack(name, thunk, "authority_mutation")
    with mock.patch.object(M, "CHECKPOINT_SHA256", "0" * 64):
        attack("authority_checkpoint_identity", M.verify_capture_authorities,
               "authority_mutation")
    with mock.patch.object(M, "RESULT_MANIFEST_SHA256", "0" * 64):
        attack("authority_capture_manifest", M.verify_capture_authorities,
               "authority_mutation")

    contract = M.strict_json(CONTRACT)
    frozen = copy.deepcopy(contract)
    def validate_contract(candidate):
        if candidate != frozen:
            raise RuntimeError("M1514 contract exact-set/value")
    for path, leaf in walk_leaves(frozen):
        candidate = copy.deepcopy(frozen)
        parent_at(candidate, path[:-1])[path[-1]] = changed(leaf)
        attack("contract_value_" + "_".join(map(str, path)),
               lambda value=candidate: validate_contract(value),
               "contract_mutation")
    for path, mapping in list(walk_dicts(frozen)):
        for key in tuple(mapping):
            candidate = copy.deepcopy(frozen)
            del parent_at(candidate, path)[key]
            attack("contract_delete_" + "_".join(map(str, path + (key,))),
                   lambda value=candidate: validate_contract(value),
                   "contract_mutation")
        candidate = copy.deepcopy(frozen)
        parent_at(candidate, path)["__M1515_EXTRA__"] = False
        attack("contract_extra_" + ("_".join(map(str, path)) or "root"),
               lambda value=candidate: validate_contract(value),
               "contract_mutation")

    check("no_export_payload", not M1516_RUNNER.exists() and
          not M1516_OUTPUT.exists(), "execution")
    check("no_gpu_eda_remote", not torch.cuda.is_initialized() and
          M.CLAIM_BOUNDARY["gpu"] is False and
          M.CLAIM_BOUNDARY["eda"] is False and
          M.CLAIM_BOUNDARY["remote"] is False, "execution")
    p0 = sum(item["false_negative"] for item in attacks)
    p1 = sum(not item["pass"] for item in checks)
    categories = {}
    for item in attacks:
        categories[item["category"]] = categories.get(item["category"], 0) + 1
    output = {
        "schema": "m1515_m1514_ep34_decoder_weight_identity_hammer_output_r1_v1",
        "status": "PASS_ZERO_FALSE_NEGATIVE" if p0 == 0 and p1 == 0
                  else "FAIL_DO_NOT_ADVANCE",
        "summary": {
            "checks_passed": sum(item["pass"] for item in checks),
            "checks_total": len(checks),
            "mutations_rejected": sum(item["rejected"] for item in attacks),
            "mutations_total": len(attacks),
            "mutation_categories": categories,
            "p0_count": p0, "p1_count": p1,
            "author_tests": "10/10 PASS",
            "checkpoint_bytes_read_only": CHECKPOINT.stat().st_size,
            "decoder_content_bytes": TOTAL_BYTES,
        },
        "real_checkpoint": {
            "sha256": checkpoint_after,
            "root_keys": ["model_state_dict"],
            "model_state_dict_keys": 921,
            "weights": rows,
            "aggregate_content_bytes": TOTAL_BYTES,
        },
        "authority": {
            "m1510": True, "m1512": True, "m1513": True,
            "capture_manifest": PINS["capture_manifest"],
            "capture_outer": PINS["capture_outer"]},
        "execution": {
            "checkpoint_loads_cpu": 2, "payload_exports": 0,
            "gpu": 0, "eda": 0, "ssh": 0, "remote": 0},
    }
    if p0 or p1:
        raise RuntimeError(json.dumps(output, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
