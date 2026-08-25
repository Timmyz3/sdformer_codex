#!/usr/bin/env python3
"""M38-r4 fail-closed repair over the frozen r3 executable reference."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
R3_ANALYZER = (
    HW_ROOT / "system_simulator/scripts/"
    "analyze_m38_rst_math_protocol_reachable_r3.py"
)
R3_CONTRACT = HW_ROOT / "contracts/m38_rst_math_input_contract_r3_20260822.json"
DEFAULT_CONTRACT = HW_ROOT / "contracts/m38_rst_math_input_contract_r4_20260822.json"
R3_ANALYZER_SHA256 = (
    "1efaaad25e6dabfbe76870bad95fde371470c598f13438de018087c7a4b050c6"
)
R3_CONTRACT_SHA256 = (
    "96198cd2f40be1edcd750d1d8f7b35ca03a24e4cbc348c47b24ade596750315a"
)
EXPECTED_CLAIM = (
    "M38-r4 executable Python3.6 fail-closed reference for 768 scalar "
    "q8-by-ternary pairs, constructive coverage of every integer rank sum "
    "from -384 through 384, exact Q24 saturation/threshold semantics, strict "
    "canonical CRC32C fragment loading with type/drain failure reset and "
    "fragment-zero recovery, exact typed offer validation with state-atomic "
    "rejection, a complete finite reachable-state safety graph, directed "
    "drain liveness, full exact semantic-contract binding, duplicate-key-safe "
    "JSON loading, and conditional kernel scheduling only. Recursive M31-r4/"
    "M37-r8 identity is admitted only through both hash-bound independent "
    "VCS-only review artifacts; the M31 admission is rebuilt and compared in "
    "full. Those artifacts admit no DC/STA/Formality/PPA/system claims. "
    "Integrated RTL, VCS of integrated RTL, DC/STA/Formality, PPA, power, "
    "energy, memory, trained coverage, Local/Motion system cycles, speedup, "
    "and headline claims remain unadmitted."
)
EXPECTED_SUPERSEDES = {
    "contract": [
        "hw_autoresearch_nts07/contracts/m38_rst_math_input_contract_r3_20260822.json",
        R3_CONTRACT_SHA256],
    "analyzer": [
        "hw_autoresearch_nts07/system_simulator/scripts/analyze_m38_rst_math_protocol_reachable_r3.py",
        R3_ANALYZER_SHA256],
    "regression": [
        "hw_autoresearch_nts07/system_simulator/tests/test_m38_rst_math_protocol_reachable_r3.py",
        "7b2bc0f52462e95727ca536b587b894ad8fd656710c7bc3a936a1042d081624e"],
    "result": [
        "hw_autoresearch_nts07/results/m38_rst_math_protocol_reachable_r3_20260822/m38_rst_math_protocol_reachable_state.json",
        "c4158ef218c06263bb1976bb8c2a89dfd39d6c4b963fae5c1c062f91b807a2dc"],
    "specification": [
        "hw_autoresearch_nts07/rtl_m38/M38_RST_MATH_PROTOCOL_REACHABLE_R3.md",
        "a04296b57ed258a54a2cca07e411ba3df94046bd0135e7549b1687adae547cfa"],
    "independent_nogo_review": [
        "hw_autoresearch_nts07/results/m38_rst_math_protocol_reachable_r3_20260822/m38_r3_independent_hammer_nogo_review.json",
        "d93335610d5d01d02a33507014188e9348f8a33e3c16035c7b51c640747ff9d6"],
    "state": "NO_GO_FAIL_CLOSED_REVIEW_SUPERSEDED_DO_NOT_CITE",
    "reasons": [
        "r3 accepted fourteen forged canonical-frame offer-schema and reachable-state semantic fields because validation checked only a subset",
        "r3 allowed a non-boolean datapath_drained failure mid-frame without clearing loader shadow state or requiring fragment-zero recovery",
        "r3 only partially validated the M31 independent admission and accepted five forged unvalidated fields",
        "r3 JSON loading did not reject duplicate keys before a later legal duplicate"],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if sha256(R3_ANALYZER) != R3_ANALYZER_SHA256:
    raise RuntimeError("M38-r3 base analyzer identity drift")
BASE_SPEC = importlib.util.spec_from_file_location("m38_r3_frozen_base", str(R3_ANALYZER))
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

ORIGINAL_VALIDATE_FROZEN = BASE.validate_frozen_contract
ORIGINAL_VALIDATE_REVIEWS = BASE.validate_review_admissions
ORIGINAL_BUILD_PROTOCOL = BASE.build_protocol_audit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_keys(payload, expected, label):
    require(isinstance(payload, dict) and set(payload) == set(expected),
            "{} population drift".format(label))


def read_json_no_duplicates(path):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook)


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


def frozen_r3_contract():
    require(sha256(R3_CONTRACT) == R3_CONTRACT_SHA256,
            "M38-r3 base contract identity drift")
    return read_json_no_duplicates(R3_CONTRACT)


def load_contract(path=DEFAULT_CONTRACT):
    path = Path(path)
    contract = read_json_no_duplicates(path)
    exact_keys(contract, BASE.CONTRACT_TOP_KEYS, "M38-r4 contract")
    require(contract["schema"] == "m38_rst_math_input_contract_v4",
            "M38-r4 contract schema drift")
    require(contract["identity"] ==
            "M31_r4_M37_r8_M38_R4_fail_closed_math_strict_protocol_and_reachable_state_reference_only",
            "M38-r4 contract identity drift")
    require(contract["claim_boundary"] == EXPECTED_CLAIM,
            "M38-r4 claim boundary drift")
    require(contract["supersedes"] == EXPECTED_SUPERSEDES,
            "M38-r4 supersession drift")
    for name in ("contract", "analyzer", "regression", "result", "specification",
                 "independent_nogo_review"):
        pair = contract["supersedes"][name]
        target = resolve(pair[0])
        require(target.is_file() and sha256(target) == pair[1],
                "M38-r3 superseded {} identity drift".format(name))

    reference = frozen_r3_contract()
    require(contract["inputs"] == reference["inputs"],
            "M38-r4 frozen input identity drift")
    exact_keys(contract["inputs"], BASE.INPUT_KEYS, "M38-r4 inputs")
    payloads, hashes = {}, {}
    for name, item in sorted(contract["inputs"].items()):
        exact_keys(item, {"path", "sha256"}, "M38-r4 input {}".format(name))
        source = resolve(item["path"])
        require(source.is_file() and sha256(source) == item["sha256"],
                "M38-r4 input identity drift: {}".format(name))
        hashes[name] = item["sha256"]
        if source.suffix == ".json":
            payloads[name] = read_json_no_duplicates(source)
        else:
            payloads[name] = source.read_text(encoding="utf-8")
    return contract, payloads, hashes


def validate_frozen_contract(contract, payloads):
    reference = frozen_r3_contract()
    for section in (
            "frozen_architecture", "canonical_configuration_frame",
            "offer_schemas", "reachable_state_model", "theory_rules"):
        exact_keys(contract[section], set(reference[section]),
                   "M38-r4 exact {}".format(section))
        require(contract[section] == reference[section],
                "M38-r4 exact {} value drift".format(section))
    ORIGINAL_VALIDATE_FROZEN(contract, payloads)


def validate_review_admissions(contract, payloads, hashes):
    for name, spec in sorted(contract["independent_review_admissions"].items()):
        require(spec["state"] == "BOUND_PASS", "M38-r4 review is not bound")
        artifact = resolve(spec["path"])
        payload = read_json_no_duplicates(artifact)
        require(sha256(artifact) == spec["sha256"],
                "M38-r4 review admission identity drift")
        require(payload.get("schema") == spec["expected_schema"]
                and payload.get("status") == spec["expected_status"],
                "M38-r4 review schema/status drift")
    audits, all_bound = ORIGINAL_VALIDATE_REVIEWS(contract, payloads, hashes)

    m31_spec = contract["independent_review_admissions"]["m31_r4"]
    m31_payload = read_json_no_duplicates(resolve(m31_spec["path"]))
    exact_keys(m31_payload, {
        "schema", "status", "identity", "manifest_audit", "log_audit",
        "observed", "r3_regression", "source_audit",
        "current_formality_filter_audit", "admission", "claim_boundary"},
        "M31-r4 independent admission")
    validator_path = resolve(m31_spec["validator_path"])
    require(sha256(validator_path) == m31_spec["validator_sha256"],
            "M31-r4 validator identity drift")
    validator = load_python_module(validator_path,
                                   "m38_r4_m31_full_admission_validator")
    rebuilt = validator.build(resolve(contract["inputs"]["m31_vcs_receipt"]["path"]))
    require(rebuilt == m31_payload,
            "M31-r4 independent admission full rebuild drift")
    audits["m31_r4"]["full_payload_exact_rebuild_match"] = True
    audits["m31_r4"]["top_key_population_exact"] = True
    audits["m37_r8"]["duplicate_key_safe_preload"] = True
    return audits, all_bound


class StrictFragmentLoader(BASE.StrictFragmentLoader):
    def accept(self, fragment, datapath_drained=False):
        try:
            index = BASE.validate_fragment(fragment)
            require(isinstance(datapath_drained, bool),
                    "datapath_drained type violation")
        except ValueError as error:
            self._fail(str(error))
        if self.failed:
            if index != 0:
                self._fail("failed load restart requires fragment zero")
            self.failed = False
        if index == 0 and self.next_index != 0:
            self._fail("duplicate or premature nonzero fragment zero")
        if index != self.next_index:
            self._fail("configuration fragment order or duplicate violation")
        self.shadow.extend(fragment["data_u64"].to_bytes(8, byteorder="little"))
        self.next_index += 1
        if index != 9:
            return False
        frame = bytes(self.shadow[:78])
        try:
            candidate = BASE.decode_configuration_frame(frame)
            active_generation = (None if self.active_config is None
                                 else self.active_config["generation_u16"])
            require(BASE.generation_is_newer(
                candidate["generation_u16"], active_generation),
                "configuration generation is stale or ambiguous")
            require(datapath_drained,
                    "configuration activation requires drained datapath")
        except ValueError as error:
            self._fail(str(error))
        self.active_config = candidate
        self.next_index = 0
        self.shadow = bytearray()
        return True


def build_protocol_audit(contract):
    audit = ORIGINAL_BUILD_PROTOCOL(contract)
    frame = BASE.pack_configuration_frame(BASE.GOLDEN_CONFIG)
    fragments = BASE.make_fragments(frame)
    loader = StrictFragmentLoader()
    loader.accept(fragments[0], datapath_drained=True)
    try:
        loader.accept(fragments[1], datapath_drained=1)
    except ValueError:
        pass
    else:
        raise ValueError("mid-frame non-boolean datapath_drained was accepted")
    require(loader.failed is True and loader.next_index == 0
            and bytes(loader.shadow) == b"" and loader.active_config is None,
            "mid-frame type failure did not reset complete shadow state")
    try:
        loader.accept(fragments[1], datapath_drained=True)
    except ValueError:
        pass
    else:
        raise ValueError("mid-frame type failure did not require fragment zero")
    require(loader.failed is True and loader.next_index == 0
            and bytes(loader.shadow) == b"",
            "failed continuation changed reset shadow state")
    for fragment in fragments:
        activated = loader.accept(fragment, datapath_drained=True)
    require(activated and loader.active_config == BASE.GOLDEN_CONFIG,
            "fragment-zero recovery after type failure failed")
    audit["negative_cases_rejected"].append(
        "datapath_drained_type_midframe_reset")
    audit["midframe_type_failure_state"] = {
        "failed": True, "next_index": 0, "shadow_bytes": 0,
        "continuation_fragment1_rejected": True,
        "fragment0_full_recovery": True}
    return audit


# The frozen r3 build resolves these names in its own module globals. Replace
# only the reviewed gates and loader; arithmetic/cycle/BFS logic remains the
# exact SHA-bound r3 implementation.
BASE.load_contract = load_contract
BASE.validate_frozen_contract = validate_frozen_contract
BASE.validate_review_admissions = validate_review_admissions
BASE.StrictFragmentLoader = StrictFragmentLoader
BASE.build_protocol_audit = build_protocol_audit


def build(contract_path=DEFAULT_CONTRACT):
    result = BASE.build(contract_path)
    result["schema"] = "m38_rst_math_protocol_reachable_state_audit_v4"
    result["status"] = (
        "PASS_M38_R4_FAIL_CLOSED_MATH_PROTOCOL_COMPLETE_REACHABLE_STATE_ONLY")
    result["identity"]["contract"] = (
        "hw_autoresearch_nts07/contracts/{}".format(Path(contract_path).name))
    result["identity"]["contract_sha256"] = sha256(contract_path)
    result["identity"]["analyzer"] = (
        "hw_autoresearch_nts07/system_simulator/scripts/"
        "analyze_m38_rst_math_protocol_reachable_r4.py")
    result["identity"]["analyzer_sha256"] = sha256(Path(__file__).resolve())
    result["identity"]["frozen_r3_analyzer"] = [
        "hw_autoresearch_nts07/system_simulator/scripts/"
        "analyze_m38_rst_math_protocol_reachable_r3.py",
        R3_ANALYZER_SHA256]
    result["admission"].update({
        "full_exact_semantic_contract_binding_admitted": True,
        "duplicate_json_key_rejection_admitted": True,
        "midframe_type_failure_complete_reset_admitted": True,
        "m31_independent_admission_full_rebuild_match": True,
    })
    result["claim_boundary"] = EXPECTED_CLAIM
    return result


def write_output(path, payload):
    BASE.write_output(path, payload)


# Public aliases retained for the r4 regression and downstream users.
GOLDEN_CONFIG = BASE.GOLDEN_CONFIG
pack_configuration_frame = BASE.pack_configuration_frame
make_fragments = BASE.make_fragments
decode_configuration_frame = BASE.decode_configuration_frame
crc32c = BASE.crc32c
validate_fragment = BASE.validate_fragment
generation_is_newer = BASE.generation_is_newer
IntegratedCycleModel = BASE.IntegratedCycleModel
ternary_product = BASE.ternary_product
constructive_rank3_decomposition = BASE.constructive_rank3_decomposition
saturate_q24 = BASE.saturate_q24


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
