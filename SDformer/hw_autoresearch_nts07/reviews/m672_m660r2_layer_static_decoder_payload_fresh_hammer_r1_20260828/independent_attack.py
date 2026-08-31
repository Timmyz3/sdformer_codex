#!/usr/bin/env python3
"""Fresh CPU/static attacks for the frozen M660-r2 author candidate.

This script must not execute a model forward, touch CUDA, consume the one-shot,
or mutate any frozen author artifact.  It independently reconstructs the real
checkpoint topology on CPU and demonstrates the reviewed-preflight identity
binding gap with a private temporary copy.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

import torch


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
REQUEST_DIR = HW / "reviews/m670_m660r2_layer_static_decoder_payload_fresh_hammer_r1_REQUEST_20260828"
REQUEST_PATH = REQUEST_DIR / "request.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def verify_double_seal(directory):
    directory = Path(directory)
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(),
            "unsafe seal directory")
    require(manifest.is_file() and not manifest.is_symlink() and
            outer.is_file() and not outer.is_symlink(), "missing seal")
    outer_fields = outer.read_text(encoding="utf-8").strip().split()
    require(outer_fields == [sha256(manifest), "SHA256SUMS"],
            "outer seal mismatch")
    sealed = set()
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = raw.split(None, 1)
        name = name.strip()
        require(name not in sealed and not Path(name).is_absolute() and
                ".." not in Path(name).parts, "unsafe seal member")
        member = directory / name
        require(member.is_file() and not member.is_symlink() and
                sha256(member) == expected, "seal member mismatch: " + name)
        sealed.add(name)
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.relative_to(directory).as_posix()
        not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    require(actual == sealed, "sealed population mismatch")
    try:
        display_directory = str(directory.relative_to(ROOT))
    except ValueError:
        display_directory = "PRIVATE_TEMPORARY_COPY"
    return {
        "directory": display_directory,
        "members": len(sealed),
        "manifest_sha256": sha256(manifest),
        "outer_seal_file_sha256": sha256(outer),
    }


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reseal(directory):
    directory = Path(directory)
    members = sorted(
        path.relative_to(directory)
        for path in directory.rglob("*")
        if path.is_file() and path.name not in
        {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    )
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(
        "{}  {}\n".format(sha256(directory / member), member.as_posix())
        for member in members), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")


def runner_preflight_semantics(receipt, expected_contract):
    """Literal semantic predicates from runner lines 197--208."""
    return bool(
        receipt.get("status") ==
        "PASS_CPU_EXACT_LOAD_REAL_WRAPPER_AND_ATLIF_LEAF" and
        receipt["contract"]["sha256"] == expected_contract and
        receipt["d1_threshold_identity"]["parameter_name"] ==
        "sttmultires_unet.decoders.1.sn.spiking_neuron.thresh" and
        not receipt["checkpoint_load_audit"].get("missing_count") and
        not receipt["checkpoint_load_audit"].get("unexpected_count")
    )


def main():
    request = strict_json(REQUEST_PATH)
    require(request["status"] == "REQUEST_ONLY__NO_EXECUTION_AUTHORIZATION",
            "request boundary drift")

    target_hashes = {}
    for name, entry in request["target"].items():
        path = ROOT / entry["path"]
        observed = sha256(path)
        require(path.is_file() and not path.is_symlink() and
                observed == entry["sha256"], "target drift: " + name)
        target_hashes[name] = observed

    nested_seals = []
    nested_seals.append(verify_double_seal(REQUEST_DIR))
    nested_seals.append(verify_double_seal(
        HW / "reviews/m669_m660r2_layer_static_decoder_payload_author_handoff_r1_20260828"))
    nested_seals.append(verify_double_seal(
        HW / "results/m660r2_h67_ep35_cpu_exact_load_preflight_r1_20260828"))
    nested_seals.append(verify_double_seal(
        HW / "reviews/m666_m660_layer_static_decoder_payload_fresh_hammer_r1_20260828"))

    producer_path = ROOT / request["target"]["producer"]["path"]
    runner_path = ROOT / request["target"]["runner"]["path"]
    contract_path = ROOT / request["target"]["contract"]["path"]
    producer = load_module("m672_frozen_m660r2", producer_path)
    contract = producer.strict_json(contract_path)
    contract_inputs = producer.verify_contract_inputs(contract, producer_path)
    predecessor = producer.verify_predecessor_evidence(contract)

    m511_path = ROOT / contract["inputs"]["m511_producer"]["path"]
    m511_contract_path = ROOT / contract["inputs"]["m511_contract"]["path"]
    config_path = ROOT / contract["inputs"]["config"]["path"]
    checkpoint_path = ROOT / contract["inputs"]["checkpoint"]["path"]
    m511 = producer.load_frozen_m511(m511_path)
    m511_contract = producer.strict_json(m511_contract_path)
    frozen_m511_inputs = m511.verify_inputs(
        m511_contract, producer.M511_PRODUCER_SHA256)

    determinism = producer.configure_deterministic_execution()
    producer.require_deterministic_execution(determinism)
    config, _ = m511.profile.load_config(config_path)
    model = m511.profile.build_model(config, checkpoint_path,
                                     torch.device("cpu"))
    load_audit = m511.profile.validate_h9_load_audit(model, config)
    require(load_audit is not None and
            int(load_audit.get("missing_count", 0)) == 0 and
            int(load_audit.get("unexpected_count", 0)) == 0,
            "fresh CPU exact-load failed")
    convtranspose_names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.ConvTranspose2d)
    ]
    expected_names = [row["name"] for row in m511_contract["modules"]]
    require(convtranspose_names == expected_names,
            "fresh ConvTranspose topology mismatch")
    d1_expected = m511_contract["modules"][1]
    theta_snapshot, threshold_identity = producer.decoder_threshold_identity(
        model, d1_expected)
    named = dict(model.named_modules())
    live_theta = named[
        "sttmultires_unet.decoders.1.sn.spiking_neuron"].thresh
    require(theta_snapshot.data_ptr() != live_theta.data_ptr(),
            "theta snapshot aliases live storage")
    require(torch.equal(theta_snapshot, live_theta.detach()),
            "theta clone content mismatch")

    preflight_dir = ROOT / contract["cpu_exact_load_preflight"][
        "canonical_directory"]
    preflight_receipt = strict_json(preflight_dir / "preflight.json")
    require(preflight_receipt["checkpoint_load_audit"] == load_audit,
            "frozen preflight load audit differs from fresh reconstruction")
    require(preflight_receipt["convtranspose_names"] == convtranspose_names and
            preflight_receipt["d1_threshold_identity"] == threshold_identity,
            "frozen preflight topology differs from fresh reconstruction")
    require(preflight_receipt["device"] == "cpu" and
            preflight_receipt["forward_executed"] is False,
            "preflight claim boundary drift")
    topology = {
        "device": "cpu", "forward_executed": False,
        "missing_count": int(load_audit["missing_count"]),
        "unexpected_count": int(load_audit["unexpected_count"]),
        "convtranspose_names": convtranspose_names,
        "wrapper_class": threshold_identity["wrapper_class"],
        "leaf_class": threshold_identity["leaf_class"],
        "parameter_name": threshold_identity["parameter_name"],
        "theta_ieee754_le_hex": threshold_identity["ieee754_le_hex"],
        "theta_ieee754_uint32": threshold_identity["ieee754_uint32"],
        "clone_storage_distinct": True,
        "receipt_exact_match": True,
    }
    del model, live_theta, theta_snapshot
    gc.collect()

    # Independent raw-FP32 attacks, including a chunk-boundary mismatch.
    exact = torch.tensor([0.0, -0.0, 1.0, float("inf")], dtype=torch.float32)
    exact_result = producer.compare_tensors_streaming(exact, exact.clone(), 2)
    require(exact_result["bit_exact"] and exact_result["hashes_equal"] and
            exact_result["max_ulp_error"] == 0, "exact miter rejected")
    signed_left = torch.tensor([0.0, 1.0], dtype=torch.float32)
    signed_right = signed_left.clone()
    signed_right.view(torch.int32)[0] = -2147483648
    signed_result = producer.compare_tensors_streaming(
        signed_left, signed_right, 1)
    require(not signed_result["bit_exact"] and
            signed_result["signed_zero_bit_mismatch_count"] == 1 and
            signed_result["max_ulp_error"] == 1 and
            not signed_result["hashes_equal"], "signed-zero attack escaped")
    ulp_left = torch.ones(5, dtype=torch.float32)
    ulp_right = ulp_left.clone()
    ulp_right.view(torch.int32)[2] += 1
    ulp_result = producer.compare_tensors_streaming(ulp_left, ulp_right, 2)
    require(not ulp_result["bit_exact"] and
            ulp_result["bit_exact_mismatch_count"] == 1 and
            ulp_result["max_ulp_error"] == 1 and
            not ulp_result["hashes_equal"], "chunk-boundary ULP escaped")
    nan_bits = torch.tensor([0x7FC00001, 0x7FC00002], dtype=torch.int32)
    nan_left = nan_bits.view(torch.float32)
    nan_right = nan_left.clone()
    nan_right.view(torch.int32)[1] += 1
    nan_result = producer.compare_tensors_streaming(nan_left, nan_right, 1)
    require(not nan_result["bit_exact"] and
            nan_result["bit_exact_mismatch_count"] == 1 and
            nan_result["max_ulp_error"] == 1, "NaN payload attack escaped")

    good_miter = exact_result
    rows = [{"folded_weight_miter": dict(good_miter)} for _ in range(10)]
    require(producer.folded_miter_admitted(rows, True),
            "all-ten exact gate rejected")
    gate_attacks = {}
    for field, value in (
            ("bit_exact", False),
            ("bit_exact_mismatch_count", 1),
            ("signed_zero_bit_mismatch_count", 1),
            ("max_ulp_error", 1),
            ("hashes_equal", False),
            ("folded_reference_output_sha256", "0" * 64)):
        attacked = [{"folded_weight_miter": dict(good_miter)}
                    for _ in range(10)]
        attacked[9]["folded_weight_miter"][field] = value
        escaped = producer.folded_miter_admitted(attacked, True)
        require(not escaped, "all-ten conjunct escaped: " + field)
        gate_attacks[field] = "REJECTED"
    require(not producer.folded_miter_admitted(rows, False),
            "theta S10 false escaped deployment gate")

    # Exercise the scrub population independently at a post-finalization-like
    # state.  The temporary tree is private and no author artifact is touched.
    with tempfile.TemporaryDirectory(prefix="m672_scrub_") as tmp:
        staging = Path(tmp) / "staging"
        (staging / "d1_candidate").mkdir(parents=True)
        (staging / "weights").mkdir()
        (staging / "calls").mkdir()
        for path in (
                staging / "d1_candidate/mask.bitpack",
                staging / "weights/d1.weight.folded_theta.f32le",
                staging / "weights/d1.original_weight_output_scale.sidecar.json",
                staging / "calls/s09_d1.activation.theta.le.bitpack",
                staging / "weights/SHA256SUMS",
                staging / "weights/SHA256SUMS.seal.sha256",
                staging / "manifest.json",
                staging / "SHA256SUMS",
                staging / "SHA256SUMS.seal.sha256",
                staging / "RUN_COMPLETE.txt"):
            path.write_bytes(b"candidate")
        removed = producer.scrub_d1_candidates(staging)
        require(len(removed) == 10 and not any(
            path.is_file() for path in staging.rglob("*")),
            "post-finalization scrub incomplete")
    scrub = {"removed_count": len(removed), "candidate_survivors": 0}

    # Prove the runner accepts a semantically valid but independently resealed
    # preflight with identities different from the two SHA roots frozen by the
    # review request.  This mirrors its literal lines 186--208 without changing
    # the canonical preflight directory or consuming the one-shot.
    with tempfile.TemporaryDirectory(prefix="m672_preflight_reseal_") as tmp:
        substituted = Path(tmp) / "preflight"
        shutil.copytree(preflight_dir, substituted)
        substituted_receipt_path = substituted / "preflight.json"
        substituted_receipt = strict_json(substituted_receipt_path)
        substituted_receipt["adversarial_reseal_marker"] = (
            "different identity accepted by current runner predicates")
        substituted_receipt_path.write_text(
            json.dumps(substituted_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        reseal(substituted)
        substituted_seal = verify_double_seal(substituted)
        require(runner_preflight_semantics(
            substituted_receipt, sha256(contract_path)),
            "substituted receipt did not pass literal runner predicates")
        frozen_receipt_sha = request["target"]["cpu_preflight_receipt"]["sha256"]
        frozen_outer_sha = request["target"][
            "cpu_preflight_outer_seal_file"]["sha256"]
        require(sha256(substituted_receipt_path) != frozen_receipt_sha and
                substituted_seal["outer_seal_file_sha256"] != frozen_outer_sha,
                "substitution did not change frozen identities")
        substitution = {
            "runner_semantics_accept": True,
            "receipt_sha_differs_from_reviewed": True,
            "outer_seal_file_sha_differs_from_reviewed": True,
            "canonical_artifact_mutated": False,
            "attempt_consumed": False,
        }

    runner_source = runner_path.read_text(encoding="utf-8")
    require("M660R2_EXPECTED_RUNNER_SHA256" in runner_source and
            "M660R2_EXPECTED_CONTRACT_SHA256" in runner_source,
            "runner external roots drift")
    require("M660R2_EXPECTED_PREFLIGHT_RECEIPT_SHA256" not in runner_source and
            "M660R2_EXPECTED_PREFLIGHT_OUTER_SEAL_SHA256" not in runner_source,
            "expected P1 gap unexpectedly repaired")

    attempt = HW / "results/.m660r2_h67_ep35_layer_static_decoder_payload_r1_attempt_consumed"
    canonical = HW / "system_handoff/outgoing/m660r2_h67_ep35_layer_static_decoder_payload_s10_r1_20260828"
    docs359 = HW / "docs/359_DATE终局冻结_20260813.md"
    require(not os.path.lexists(str(attempt)) and
            not os.path.lexists(str(canonical)),
            "one-shot/canonical path is not absent")
    require(sha256(docs359) ==
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            "docs359 drift")

    result = {
        "schema": "m672_m660r2_fresh_independent_attack_result_v1",
        "status": "PASS_ATTACKS_WITH_P1_PREFLIGHT_REVIEW_IDENTITY_BINDING_GAP",
        "execution": {
            "gpu": False, "model_forward": False,
            "one_shot_consumed": False, "rtl": False, "eda": False,
        },
        "target_hashes": target_hashes,
        "contract_inputs_rehashed": len(contract_inputs),
        "predecessor_evidence_roots_reconstructed": len(predecessor),
        "m511_inputs_rehashed": len(frozen_m511_inputs),
        "nested_seals": nested_seals,
        "fresh_cpu_exact_load": topology,
        "deterministic_execution": determinism,
        "raw_fp32_attacks": {
            "exact": exact_result,
            "signed_zero": signed_result,
            "adjacent_ulp_chunk_boundary": ulp_result,
            "nan_payload": nan_result,
        },
        "all_ten_conjunctive_gate_attacks": gate_attacks,
        "scrub_attack": scrub,
        "p1_preflight_review_identity_substitution": substitution,
        "boundary": {
            "m658_p2": "PENDING_POST_RESULT_INDEPENDENT_HAMMER",
            "canonical_output_absent": True,
            "attempt_absent": True,
            "docs359_sha256": sha256(docs359),
            "m665_author_tests": "SEPARATE_PYTEST_39_OF_39_COMBINED_PASS",
        },
    }
    output = REVIEW / "independent_attack_result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "fresh_cpu_exact_load": topology,
        "p1": substitution,
        "result": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
