#!/usr/bin/env python3
"""M38-r5 recursive type-strict fail-closed reference."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
R4_ANALYZER = (
    HW_ROOT / "system_simulator/scripts/"
    "analyze_m38_rst_math_protocol_reachable_r4.py"
)
R4_CONTRACT = HW_ROOT / "contracts/m38_rst_math_input_contract_r4_20260822.json"
DEFAULT_CONTRACT = HW_ROOT / "contracts/m38_rst_math_input_contract_r5_20260822.json"
R4_ANALYZER_SHA256 = (
    "169e5dc3085cdcb6a87d945b53f8bb9f5420242e81534ecbe6fd4ac98ceabf21"
)
R4_CONTRACT_SHA256 = (
    "d32c6437fc2a70001da1ebeb8f3d52f0acba2f07a81398368abaf852d3f3590c"
)
EXPECTED_CLAIM = (
    "M38-r5 executable Python3.6 type-strict fail-closed reference for 768 "
    "scalar q8-by-ternary pairs, constructive coverage of every integer rank "
    "sum from -384 through 384, exact Q24 saturation/threshold semantics, "
    "strict canonical CRC32C fragment loading with type/drain failure reset "
    "and fragment-zero recovery, exact typed offer validation with state-atomic "
    "rejection, a complete finite reachable-state safety graph, directed drain "
    "liveness, recursive type-strict semantic-contract and review-admission "
    "binding, duplicate-key-safe JSON loading, and conditional kernel "
    "scheduling only. Boolean and integer JSON values are never interchangeable. "
    "Recursive M31-r4/M37-r8 identity is admitted only through both hash-bound "
    "independent VCS-only review artifacts; both review payloads are compared "
    "type-strictly with their canonical evidence and the M31 admission is "
    "rebuilt in full. Those artifacts admit no DC/STA/Formality/PPA/system "
    "claims. Integrated RTL, VCS of integrated RTL, DC/STA/Formality, PPA, "
    "power, energy, memory, trained coverage, Local/Motion system cycles, "
    "speedup, and headline claims remain unadmitted."
)
EXPECTED_SUPERSEDES = {
    "contract": [
        "hw_autoresearch_nts07/contracts/m38_rst_math_input_contract_r4_20260822.json",
        R4_CONTRACT_SHA256],
    "analyzer": [
        "hw_autoresearch_nts07/system_simulator/scripts/analyze_m38_rst_math_protocol_reachable_r4.py",
        R4_ANALYZER_SHA256],
    "regression": [
        "hw_autoresearch_nts07/system_simulator/tests/test_m38_rst_math_protocol_reachable_r4.py",
        "b19774fd6dde6bed0f42d7c74d11619c1ff45077af1f86629739ab7a2017f2d7"],
    "result": [
        "hw_autoresearch_nts07/results/m38_rst_math_protocol_reachable_r4_20260822/m38_rst_math_protocol_reachable_state.json",
        "b2b79a148f738fedb9c67529a991b1549e29c25a3c6dd2b8300bc14aa9673075"],
    "specification": [
        "hw_autoresearch_nts07/rtl_m38/M38_RST_FAIL_CLOSED_REFERENCE_R4.md",
        "ed06405a63b772411b972ffa5dc235c1cdec1b0b0d03a88c764d3ffbdf35c080"],
    "independent_nogo_review": [
        "hw_autoresearch_nts07/results/m38_rst_math_protocol_reachable_r4_20260822/m38_r4_independent_hammer_nogo_review.json",
        "d45406ce03b486d98a33e0a8fdf486dc3c1e1bde662392d38aec82865857f14a"],
    "prior_r3_independent_nogo_review": [
        "hw_autoresearch_nts07/results/m38_rst_math_protocol_reachable_r3_20260822/m38_r3_independent_hammer_nogo_review.json",
        "d93335610d5d01d02a33507014188e9348f8a33e3c16035c7b51c640747ff9d6"],
    "state": "NO_GO_TYPE_STRICT_REVIEW_SUPERSEDED_DO_NOT_CITE",
    "reasons": [
        "r4 used Python container equality for exact JSON objects, so four boolean/integer type-confusion forgeries still returned PASS"],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if sha256(R4_ANALYZER) != R4_ANALYZER_SHA256:
    raise RuntimeError("M38-r4 base analyzer identity drift")
R4_SPEC = importlib.util.spec_from_file_location("m38_r4_frozen_base", str(R4_ANALYZER))
R4 = importlib.util.module_from_spec(R4_SPEC)
R4_SPEC.loader.exec_module(R4)
BASE = R4.BASE


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_keys(payload, expected, label):
    require(isinstance(payload, dict) and set(payload) == set(expected),
            "{} population drift".format(label))


def reject_nonstandard_constant(raw):
    raise ValueError("non-standard JSON numeric constant: {}".format(raw))


def read_json_no_duplicates(path):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_nonstandard_constant)


def type_strict_mismatch(actual, expected, path="$"):
    if type(actual) is not type(expected):
        return "{} type {} != {}".format(
            path, type(actual).__name__, type(expected).__name__)
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            return "{} key population differs".format(path)
        for key in sorted(actual):
            mismatch = type_strict_mismatch(
                actual[key], expected[key], "{}.{}".format(path, key))
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return "{} list length differs".format(path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            mismatch = type_strict_mismatch(
                left, right, "{}[{}]".format(path, index))
            if mismatch is not None:
                return mismatch
        return None
    if actual != expected:
        return "{} value differs".format(path)
    return None


def require_type_strict_equal(actual, expected, label):
    mismatch = type_strict_mismatch(actual, expected)
    require(mismatch is None, "{} type-strict drift: {}".format(label, mismatch))


def resolve(raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_python_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "validator import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_r4_contract():
    require(sha256(R4_CONTRACT) == R4_CONTRACT_SHA256,
            "M38-r4 base contract identity drift")
    return read_json_no_duplicates(R4_CONTRACT)


def load_contract(path=DEFAULT_CONTRACT):
    path = Path(path)
    contract = read_json_no_duplicates(path)
    exact_keys(contract, BASE.CONTRACT_TOP_KEYS, "M38-r5 contract")
    require(contract["schema"] == "m38_rst_math_input_contract_v5",
            "M38-r5 contract schema drift")
    require(contract["identity"] ==
            "M31_r4_M37_r8_M38_R5_type_strict_fail_closed_reference_only",
            "M38-r5 contract identity drift")
    require(contract["claim_boundary"] == EXPECTED_CLAIM,
            "M38-r5 claim boundary drift")
    require_type_strict_equal(contract["supersedes"], EXPECTED_SUPERSEDES,
                              "M38-r5 supersession")
    for name in (
            "contract", "analyzer", "regression", "result", "specification",
            "independent_nogo_review", "prior_r3_independent_nogo_review"):
        pair = contract["supersedes"][name]
        target = resolve(pair[0])
        require(target.is_file() and sha256(target) == pair[1],
                "M38-r4 superseded {} identity drift".format(name))
    require((resolve(contract["supersedes"]["independent_nogo_review"][0])
             .stat().st_mode & 0o777) == 0o444,
            "M38-r4 NO-GO review mode drift")

    reference = frozen_r4_contract()
    require_type_strict_equal(contract["inputs"], reference["inputs"],
                              "M38-r5 frozen input identities")
    exact_keys(contract["inputs"], BASE.INPUT_KEYS, "M38-r5 inputs")
    payloads, hashes = {}, {}
    for name, item in sorted(contract["inputs"].items()):
        exact_keys(item, {"path", "sha256"}, "M38-r5 input {}".format(name))
        source = resolve(item["path"])
        require(source.is_file() and sha256(source) == item["sha256"],
                "M38-r5 input identity drift: {}".format(name))
        hashes[name] = item["sha256"]
        if source.suffix == ".json":
            payloads[name] = read_json_no_duplicates(source)
        else:
            payloads[name] = source.read_text(encoding="utf-8")
    return contract, payloads, hashes


def validate_frozen_contract(contract, payloads):
    reference = frozen_r4_contract()
    for section in (
            "frozen_architecture", "canonical_configuration_frame",
            "offer_schemas", "reachable_state_model", "theory_rules"):
        require_type_strict_equal(
            contract[section], reference[section],
            "M38-r5 exact {}".format(section))
    # Retain every r4 executable/subset assertion after the strict recursive
    # equality gate succeeds.
    R4.validate_frozen_contract(contract, payloads)


def validate_review_admissions(contract, payloads, hashes):
    reference = frozen_r4_contract()
    canonical_payloads = {}
    current_payloads = {}
    for name, spec in sorted(contract["independent_review_admissions"].items()):
        require(spec["state"] == "BOUND_PASS", "M38-r5 review is not bound")
        canonical_spec = reference["independent_review_admissions"][name]
        canonical_payloads[name] = read_json_no_duplicates(
            resolve(canonical_spec["path"]))
        current_payloads[name] = read_json_no_duplicates(resolve(spec["path"]))
        require(sha256(resolve(spec["path"])) == spec["sha256"],
                "M38-r5 review identity drift")
        require_type_strict_equal(
            current_payloads[name], canonical_payloads[name],
            "M38-r5 {} review payload".format(name))

    audits, all_bound = R4.validate_review_admissions(contract, payloads, hashes)
    m31_spec = contract["independent_review_admissions"]["m31_r4"]
    validator_path = resolve(m31_spec["validator_path"])
    require(sha256(validator_path) == m31_spec["validator_sha256"],
            "M31-r4 validator identity drift")
    validator = load_python_module(
        validator_path, "m38_r5_m31_type_strict_admission_validator")
    rebuilt = validator.build(
        resolve(contract["inputs"]["m31_vcs_receipt"]["path"]))
    require_type_strict_equal(
        current_payloads["m31_r4"], rebuilt,
        "M31-r4 independent admission full rebuild")
    audits["m31_r4"]["full_payload_type_strict_rebuild_match"] = True
    audits["m37_r8"]["full_payload_type_strict_canonical_match"] = True
    return audits, all_bound


# The frozen r4 build resolves these names through its r3 base module.
BASE.load_contract = load_contract
BASE.validate_frozen_contract = validate_frozen_contract
BASE.validate_review_admissions = validate_review_admissions


def build(contract_path=DEFAULT_CONTRACT):
    result = R4.build(contract_path)
    result["schema"] = "m38_rst_math_protocol_reachable_state_audit_v5"
    result["status"] = (
        "PASS_M38_R5_TYPE_STRICT_MATH_PROTOCOL_COMPLETE_REACHABLE_STATE_ONLY")
    result["identity"]["contract"] = (
        "hw_autoresearch_nts07/contracts/{}".format(Path(contract_path).name))
    result["identity"]["contract_sha256"] = sha256(contract_path)
    result["identity"]["analyzer"] = (
        "hw_autoresearch_nts07/system_simulator/scripts/"
        "analyze_m38_rst_math_protocol_reachable_r5.py")
    result["identity"]["analyzer_sha256"] = sha256(Path(__file__).resolve())
    result["identity"]["frozen_r4_analyzer"] = [
        "hw_autoresearch_nts07/system_simulator/scripts/"
        "analyze_m38_rst_math_protocol_reachable_r4.py",
        R4_ANALYZER_SHA256]
    result["admission"].update({
        "recursive_type_strict_semantic_binding_admitted": True,
        "boolean_integer_interchange_rejected": True,
        "nonstandard_json_numeric_constants_rejected": True,
        "both_review_payloads_type_strict_canonical_match": True,
        "m31_independent_admission_type_strict_rebuild_match": True,
    })
    result["claim_boundary"] = EXPECTED_CLAIM
    return result


def write_output(path, payload):
    R4.write_output(path, payload)


GOLDEN_CONFIG = R4.GOLDEN_CONFIG
StrictFragmentLoader = R4.StrictFragmentLoader
pack_configuration_frame = R4.pack_configuration_frame
make_fragments = R4.make_fragments
decode_configuration_frame = R4.decode_configuration_frame
crc32c = R4.crc32c
validate_fragment = R4.validate_fragment
generation_is_newer = R4.generation_is_newer
IntegratedCycleModel = R4.IntegratedCycleModel
ternary_product = R4.ternary_product
constructive_rank3_decomposition = R4.constructive_rank3_decomposition
saturate_q24 = R4.saturate_q24


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.contract)
    write_output(args.output, result)
    print(json.dumps({
        "status": result["status"], "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
