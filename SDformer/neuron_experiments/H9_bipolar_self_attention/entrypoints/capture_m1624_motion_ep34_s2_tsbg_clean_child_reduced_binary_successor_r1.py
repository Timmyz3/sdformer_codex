#!/usr/bin/env python3
"""M1624 clean-child successor for one future ep34 reduced-binary capture.

The parent accepts no capture inputs.  After a separately sealed review and
release exist, it starts exactly one fixed interpreter/source child.  The
child independently revalidates every fixed disk identity, the ep34 live93
manifest/checkpoint/config/cohort, free space and fresh namespaces, consumes
the attempt atomically, and only then constructs the fixed M1558 producer.

No Python permit, registry, provider, free-space value, provenance value or
callable crosses the process boundary.  M1558's in-process permit is used only
as a local implementation detail after the one-shot child is isolated.  This
source authoring stage is inert: no review/release exists and no GPU, payload,
capture, AEE, DSE, RTL or EDA operation is authorized.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
TEST = HW / "tests/test_m1624_motion_ep34_s2_tsbg_clean_child_source.py"
SOURCE_CONTRACT = HW / (
    "contracts/m1624_motion_ep34_s2_tsbg_clean_child_reduced_binary_"
    "source_contract_r1_20260901.json")
M1434_SOURCE = SOURCE.with_name(
    "capture_m1434_motion_ep34_live93_runtime_successor_r1.py")
M1558_SOURCE = SOURCE.with_name(
    "capture_m1558_motion_ep34_s2_tsbg_reduced_binary_source_r1.py")
PROFILE_SOURCE = SOURCE.with_name("profile_nts11_hardware_p0.py")
M1458_ROOT = HW / (
    "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_"
    "r1_20260831")
M1458_MANIFEST = M1458_ROOT / "manifest.json"
M1458_SUMS = M1458_ROOT / "SHA256SUMS"
M1458_OUTER = M1458_ROOT / "SHA256SUMS.seal.sha256"
SAMPLE_ORDER = HW / (
    "system_handoff/m1544_ep34_sparse_capture_handoff_r1_20260831/"
    "sample_order.json")
M1512 = HW / (
    "reviews/m1512_m1501_m1458_ep34_capture_source_result_independent_"
    "hammer_r1_20260831")
M1598 = HW / (
    "reviews/m1598_m1582_m1574_tsbg_capture_permit_independent_rehammer_"
    "r1_20260901")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

FUTURE_REVIEW = HW / (
    "reviews/m1625_m1624_motion_ep34_s2_tsbg_clean_child_source_hammer_"
    "r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1626_m1625_m1624_motion_ep34_s2_tsbg_clean_child_capture_"
    "release_r1_20260901.json")
CHILD_PYTHON = Path("/opt/conda/envs/sdformerflow/bin/python3.10")
REMOTE_ROOT = Path("/root/private_data/work/sdformer_codex/SDformer")
CHECKPOINT = REMOTE_ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "dsec_c12_alpha0125_ep29_resume5_20260830/checkpoint_epoch34.pth")
CONFIG = REMOTE_ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_c12_alpha0125_ep29_resume5_20260830.yml")

RESULT = HW / (
    "results/m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901")
ATTEMPT = HW / (
    "results/.m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901.attempt_consumed")
WORK = HW / (
    "results/.m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901.work")
FAILURE = HW / (
    "results/m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_"
    "r1_20260901.failed_no_retry")

SOURCE_SCHEMA = (
    "m1624_motion_ep34_s2_tsbg_clean_child_reduced_binary_source_r1_v1")
SOURCE_STATUS = (
    "SOURCE_ONLY__CLEAN_FIXED_CHILD__DIFFERENT_AUTHOR_REVIEW_REQUIRED__"
    "NO_CAPTURE")
REVIEW_STATUS = (
    "PASS_M1625_M1624_CLEAN_CHILD_SOURCE__AUTHORIZE_RELEASE_AUTHORING__"
    "NO_CAPTURE")
RELEASE_STATUS = (
    "AUTHORIZE_ONE_M1624_EP34_S2_TSBG_REDUCED_BINARY_CLEAN_CHILD_CAPTURE")
ATTEMPT_TOKEN = (
    "M1624_ATTEMPT_CONSUMED__ONE_CLEAN_CHILD__AUTOMATIC_RETRY_FALSE\n")
PASS_TOKEN = (
    "PASS_M1624_EP34_S2_TSBG_REDUCED_BINARY_CLEAN_CHILD_CAPTURE__"
    "FRESH_RESULT_HAMMER_REQUIRED")

M1434_SHA256 = "b28c8507f077b754048fc54afd9fe04900dac854b273df2ba1981fa5f892b6ed"
M1558_SHA256 = "e6686564064ae3acda2bfcfc8c2d75061eb9cb591bc739d090bc03911469b089"
PROFILE_SHA256 = "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684"
M1458_MANIFEST_SHA256 = "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d"
M1458_SUMS_SHA256 = "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e"
M1458_OUTER_SHA256 = "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed"
SAMPLE_ORDER_SHA256 = "d4f1f6e140b531b972d53b48aa64e5f0aa5497b79d460616a0b3f89139a4f773"
M1512_REVIEW_SHA256 = "b302e94375f925d84a45eb798579f243fa68b13724d3f63fabfe2810948dbb74"
M1512_SUMS_SHA256 = "2af7a59b6a4df07dc6047c0d48c52b7798b7f0803e31e290b2ad842e6c154b81"
M1512_OUTER_SHA256 = "ccbcd7bf1b99fd944062a6fb220d7ec719d96da91c190697db125cbd4ad58f7c"
M1598_REVIEW_SHA256 = "e887266475d28f7c2cfba3f69cbbbd103eed9db08905eebe042528f2baea1065"
M1598_SUMS_SHA256 = "2dc5ed7e2f2fbc26b7177b889bae0fafd1e3c2dd8a51da4896507aa7d812781d"
M1598_OUTER_SHA256 = "068cc1a6dcf50c827d9ea883a349a96f0d8114034736eda82fff03ec4f10dd05"
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
CHECKPOINT_BYTES = 225504447
CONFIG_SHA256 = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


class M1624Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1624Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1624Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1624Error("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def verify_tree_seal(root, review_sha, sums_sha, outer_sha):
    root = Path(root)
    review = root / "review.json"
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular_exact(review, review_sha, root.name + " review")
    regular_exact(sums, sums_sha, root.name + " manifest")
    regular_exact(outer, outer_sha, root.name + " outer seal")
    require(outer.read_text(encoding="ascii") == sums_sha + "  SHA256SUMS\n",
            root.name + " outer content mismatch")
    sealed_review = False
    for row in sums.read_text(encoding="utf-8").splitlines():
        fields = row.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                root.name + " malformed manifest")
        relative = Path(fields[1])
        require(not relative.is_absolute() and ".." not in relative.parts,
                root.name + " unsafe manifest member")
        member = root / relative
        regular_exact(member, fields[0], root.name + " member")
        if relative.as_posix() == "review.json":
            sealed_review = fields[0] == review_sha
    require(sealed_review, root.name + " review is not sealed")
    return strict_json(review)


def verify_file_seal(path):
    path = Path(path)
    sums = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(path.is_file() and not path.is_symlink(), "release absent/nonregular")
    require(sums.is_file() and not sums.is_symlink(), "release manifest absent")
    require(outer.is_file() and not outer.is_symlink(), "release outer absent")
    require(sums.read_text(encoding="ascii") ==
            sha256(path) + "  " + path.name + "\n", "release manifest mismatch")
    require(outer.read_text(encoding="ascii") ==
            sha256(sums) + "  " + sums.name + "\n", "release outer mismatch")


def validate_source_contract():
    value = strict_json(SOURCE_CONTRACT)
    require(value.get("schema") == SOURCE_SCHEMA and
            value.get("status") == SOURCE_STATUS,
            "M1624 source contract identity mismatch")
    require(value.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)},
        "M1624 source contract source mismatch")
    require(value.get("test") == {
        "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
        "M1624 source contract test mismatch")
    require(value.get("authorization", {}).get("different_author_review") is True and
            value.get("authorization", {}).get("capture") is False and
            value.get("authorization", {}).get("gpu") is False,
            "M1624 authoring cannot authorize capture")
    return value


def verify_fixed_metadata(expect_future_absent):
    for path, digest, label in (
        (M1434_SOURCE, M1434_SHA256, "M1434 source"),
        (M1558_SOURCE, M1558_SHA256, "M1558 source"),
        (PROFILE_SOURCE, PROFILE_SHA256, "fixed profile source"),
        (M1458_MANIFEST, M1458_MANIFEST_SHA256, "M1458 manifest"),
        (M1458_SUMS, M1458_SUMS_SHA256, "M1458 inner manifest"),
        (M1458_OUTER, M1458_OUTER_SHA256, "M1458 outer seal"),
        (SAMPLE_ORDER, SAMPLE_ORDER_SHA256, "M1458 sample order"),
        (DOCS359, DOCS359_SHA256, "protected docs359"),
    ):
        regular_exact(path, digest, label)
    require(M1458_OUTER.read_text(encoding="ascii") ==
            M1458_SUMS_SHA256 + "  SHA256SUMS\n", "M1458 outer content drift")
    manifest_row = M1458_MANIFEST_SHA256 + "  manifest.json"
    require(manifest_row in M1458_SUMS.read_text(encoding="utf-8").splitlines(),
            "M1458 manifest not bound by inner manifest")

    result_review = verify_tree_seal(
        M1512, M1512_REVIEW_SHA256, M1512_SUMS_SHA256, M1512_OUTER_SHA256)
    require(result_review.get("status") ==
            "PASS_M1512_M1501_M1458_EP34_CAPTURE_SOURCE_AND_RESULT" and
            result_review.get("p0_count") == 0 and
            result_review.get("p1_count") == 0,
            "M1458 result hammer drift")
    no_go = verify_tree_seal(
        M1598, M1598_REVIEW_SHA256, M1598_SUMS_SHA256, M1598_OUTER_SHA256)
    require(no_go.get("status") ==
            "NO_GO_M1598_M1582_CLOSURE_REGISTRY_NOT_A_PYTHON_SECURITY_BOUNDARY__"
            "SUCCESSOR_FIX_ONLY__NO_CAPTURE" and
            no_go.get("authorization", {}).get(
                "minimal_source_only_boundary_successor") is True and
            no_go.get("authorization", {}).get("capture") is False,
            "M1598 NO-GO boundary drift")

    capture = strict_json(M1458_MANIFEST)
    selected = capture.get("identity", {}).get("selection", {}).get("selected", {})
    require(capture.get("schema") ==
            "m1434_motion_ep34_live93_unified_hardware_capture_r1_v1" and
            capture.get("status") ==
            "CAPTURE_COMPLETE__FRESH_M1434_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM" and
            selected.get("candidate_id") == "resume_ep34" and
            selected.get("epoch") == 34 and
            selected.get("checkpoint", {}).get("absolute_path") == str(CHECKPOINT) and
            selected.get("checkpoint", {}).get("sha256") == CHECKPOINT_SHA256 and
            selected.get("checkpoint", {}).get("size_bytes") == CHECKPOINT_BYTES and
            selected.get("configuration", {}).get("absolute_path") == str(CONFIG) and
            selected.get("configuration", {}).get("sha256") == CONFIG_SHA256 and
            capture.get("cohort", {}).get("population") == 40 and
            capture.get("ordered_population", {}).get("records") == 9880,
            "final ep34 live93 capture identity drift")
    audit = capture.get("identity", {}).get("checkpoint_load_audit", {})
    require(audit.get("missing_count") == 0 and
            audit.get("unexpected_count") == 0 and
            audit.get("overlay_missing_count") == 0 and
            audit.get("overlay_unexpected_count") == 0,
            "M1458 checkpoint load audit drift")
    samples = strict_json(SAMPLE_ORDER)
    require(samples.get("identity", {}).get("checkpoint_sha256") ==
            CHECKPOINT_SHA256 and len(samples.get("samples", [])) == 40 and
            [row.get("global_sample_id") for row in samples["samples"]] ==
            list(range(40)), "M1458 sample-order identity drift")
    validate_source_contract()
    if expect_future_absent:
        require(not os.path.lexists(str(FUTURE_REVIEW)) and
                not os.path.lexists(str(FUTURE_RELEASE)) and
                not os.path.lexists(str(Path(str(FUTURE_RELEASE) + ".sha256"))) and
                not os.path.lexists(str(Path(str(FUTURE_RELEASE) +
                                             ".sha256.seal.sha256"))),
                "future M1625/M1626 authority must be absent at authoring")
    return {"checkpoint_sha256": CHECKPOINT_SHA256, "samples": 40,
            "m1558_sha256": M1558_SHA256,
            "m1598_review_sha256": M1598_REVIEW_SHA256}


def validate_future_authorities():
    require(FUTURE_REVIEW.is_dir() and not FUTURE_REVIEW.is_symlink(),
            "fresh M1625 review is absent")
    review_manifest = FUTURE_REVIEW / "SHA256SUMS"
    review_outer = FUTURE_REVIEW / "SHA256SUMS.seal.sha256"
    require(review_manifest.is_file() and review_outer.is_file(),
            "M1625 review is not sealed")
    review = verify_tree_seal(
        FUTURE_REVIEW, sha256(FUTURE_REVIEW / "review.json"),
        sha256(review_manifest), sha256(review_outer))
    expected_identity = {
        "source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "m1434_source_sha256": M1434_SHA256,
        "m1558_source_sha256": M1558_SHA256,
        "m1458_manifest_sha256": M1458_MANIFEST_SHA256,
        "m1512_review_sha256": M1512_REVIEW_SHA256,
        "m1598_review_sha256": M1598_REVIEW_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "docs359_sha256": DOCS359_SHA256,
    }
    require(review.get("status") == REVIEW_STATUS and
            review.get("score", 0) >= 95 and
            review.get("p0_count") == 0 and review.get("p1_count") == 0 and
            review.get("identity") == expected_identity and
            review.get("authorization") == {
                "release_authoring": True, "capture": False,
                "gpu": False, "automatic_retry": False},
            "M1625 different-author review mismatch")

    verify_file_seal(FUTURE_RELEASE)
    release = strict_json(FUTURE_RELEASE)
    interpreter = release.get("child_interpreter", {})
    require(interpreter.get("path") == str(CHILD_PYTHON) and
            len(interpreter.get("sha256", "")) == 64,
            "fixed child interpreter binding mismatch")
    regular_exact(CHILD_PYTHON, interpreter["sha256"],
                  "fixed child interpreter")
    expected_release_identity = dict(expected_identity,
        review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        review_manifest_sha256=sha256(review_manifest),
        review_outer_file_sha256=sha256(review_outer))
    require(release.get("schema") ==
            "m1626_m1625_m1624_clean_child_capture_release_r1_v1" and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == expected_release_identity and
            release.get("authorization") == {
                "parent_calls": 1, "clean_child_processes": 1,
                "gpu_runs": 1, "production_captures": 1,
                "automatic_retry": False, "all_other_runs": 0} and
            release.get("namespaces") == {
                "result": str(RESULT.relative_to(ROOT)),
                "attempt": str(ATTEMPT.relative_to(ROOT)),
                "work": str(WORK.relative_to(ROOT)),
                "failure": str(FAILURE.relative_to(ROOT))} and
            release.get("claim_boundary") == {
                "tsbg_dse": False, "aee": False, "rtl": False,
                "eda": False, "performance": False,
                "paper_result": False},
            "M1626 release mismatch")
    return release


def require_fresh_namespaces():
    paths = (RESULT, ATTEMPT, WORK, FAILURE)
    require(len(set(paths)) == 4 and all("m1624_" in path.name for path in paths),
            "M1624 namespace identity drift")
    require(all(not os.path.lexists(str(path)) for path in paths),
            "M1624 result/attempt/work/failure namespace is not fresh")


def consume_attempt(release):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(ATTEMPT), flags, 0o400)
    try:
        value = (ATTEMPT_TOKEN + "release_sha256=" +
                 sha256(FUTURE_RELEASE) + "\nsource_sha256=" +
                 sha256(SOURCE) + "\n")
        os.write(descriptor, value.encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(release.get("authorization", {}).get("automatic_retry") is False,
            "attempt consumed under retryable release")


def load_m1434():
    regular_exact(M1434_SOURCE, M1434_SHA256, "M1434 source before import")
    spec = importlib.util.spec_from_file_location(
        "m1624_fixed_m1434", str(M1434_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot load fixed M1434")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    regular_exact(M1434_SOURCE, M1434_SHA256, "M1434 source after import")
    return module


def load_m1558():
    regular_exact(M1558_SOURCE, M1558_SHA256, "M1558 source before import")
    spec = importlib.util.spec_from_file_location(
        "m1624_fixed_m1558", str(M1558_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot load fixed M1558")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    regular_exact(M1558_SOURCE, M1558_SHA256, "M1558 source after import")
    return module


def seal_result(root):
    members = sorted(path.relative_to(root) for path in root.rglob("*")
                     if path.is_file() and path.name not in
                     ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    sums = root / "SHA256SUMS"
    sums.write_text("".join("{}  {}\n".format(sha256(root / member),
                                              member.as_posix())
                            for member in members), encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(sums)), encoding="ascii")


def write_child_receipt(root, release, load_audit, validation):
    receipt = {
        "schema": "m1624_ep34_s2_tsbg_clean_child_capture_receipt_r1_v1",
        "status": "PAYLOAD_COMPLETE__FRESH_DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED",
        "identity": {
            "source_sha256": sha256(SOURCE),
            "source_contract_sha256": sha256(SOURCE_CONTRACT),
            "release_sha256": sha256(FUTURE_RELEASE),
            "m1558_source_sha256": M1558_SHA256,
            "m1458_manifest_sha256": M1458_MANIFEST_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": CONFIG_SHA256,
        },
        "checkpoint_load": {
            "missing_count": int(load_audit.get("missing_count", -1)),
            "unexpected_count": int(load_audit.get("unexpected_count", -1)),
            "overlay_missing_count": int(load_audit.get("overlay_missing_count", -1)),
            "overlay_unexpected_count": int(
                load_audit.get("overlay_unexpected_count", -1)),
        },
        "population": {
            "samples": 40, "frames": int(validation["frames"]),
            "fc_tokens": int(validation["fc_tokens"]),
            "patch_histogram_rows": int(validation["patch_histogram_rows"]),
        },
        "execution": {
            "clean_child_processes": 1, "automatic_retry": False,
            "provider_crossed_parent_boundary": False,
            "permit_crossed_parent_boundary": False,
            "free_space_crossed_parent_boundary": False,
            "provenance_crossed_parent_boundary": False,
            "callable_crossed_parent_boundary": False,
        },
        "claim_boundary": {
            "capture_payload_only": True, "fresh_result_hammer_required": True,
            "hardware_quantization_authority": False,
            "model_bit_exact": False, "tsbg_dse": False, "aee": False,
            "cycles": False, "traffic": False, "energy": False,
            "speedup": False, "rtl": False, "eda": False,
            "paper_result": False,
        },
    }
    (root / "m1624_clean_child_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    seal_result(root)
    return receipt


def close_failed_producer(producer):
    if producer is None:
        return
    while producer.handles:
        producer.handles.pop().remove()
    for writer in (producer.binary, producer.patch):
        try:
            writer.close()
        except Exception:
            pass


def fixed_clean_child():
    verify_fixed_metadata(expect_future_absent=False)
    release = validate_future_authorities()
    require_fresh_namespaces()
    require(Path(sys.executable).resolve() == CHILD_PYTHON.resolve(),
            "child did not run under fixed interpreter")

    m1434 = load_m1434()
    m1558 = load_m1558()
    m1434.verify_predecessors()
    runtime, binding = m1434.build_runtime()
    require(binding.get("identity", {}).get("checkpoint_sha256") ==
            CHECKPOINT_SHA256 and
            binding.get("identity", {}).get("config_sha256") == CONFIG_SHA256 and
            Path(binding.get("checkpoint_path")) == CHECKPOINT and
            Path(binding.get("config_path")) == CONFIG,
            "fixed ep34 runtime binding drift")
    regular_exact(CHECKPOINT, CHECKPOINT_SHA256, "ep34 checkpoint")
    require(CHECKPOINT.stat().st_size == CHECKPOINT_BYTES,
            "ep34 checkpoint size drift")
    regular_exact(CONFIG, CONFIG_SHA256, "ep34 config")
    samples = m1434.R1.validate_cohort(runtime["cohort"]["samples"])
    sample_order = m1558.M1552.verify_bindings()
    require([m1558.M1552.project_m1434_sample(row) for row in samples] ==
            sample_order["samples"], "runtime/sample-order projection drift")
    specs = m1558.frozen_layer_specs()
    estimate = m1558.estimate_from_specs(specs, 40)
    available = shutil.disk_usage(str(RESULT.parent)).free
    require(available - int(estimate["result_upper_bytes"]) >
            m1558.MIN_FREE_AFTER_BYTES,
            "clean child would not leave strictly more than 16 GiB free")

    substrate = m1434.R1.load_substrate()
    producer = None
    published = False
    try:
        with substrate.exclusive_gpu_lease(m1434.R1.CANONICAL_LEASE):
            # The one shot is burned before checkpoint/model load or producer
            # construction.  Any later failure is terminal and is never retried.
            consume_attempt(release)
            profile = substrate.load_source(
                "m1624_fixed_profile", PROFILE_SOURCE, PROFILE_SHA256)
            torch = profile.torch
            import numpy as np
            config, device = profile.load_config(CONFIG)
            require(str(device).startswith("cuda") and torch.cuda.is_available(),
                    "M1624 production child requires CUDA")
            model = profile.build_model(config, CHECKPOINT, device)
            load_audit = profile.validate_h9_load_audit(model, config)
            require(load_audit is not None and
                    int(load_audit.get("missing_count", -1)) == 0 and
                    int(load_audit.get("unexpected_count", -1)) == 0 and
                    int(load_audit.get("overlay_missing_count", -1)) == 0 and
                    int(load_audit.get("overlay_unexpected_count", -1)) == 0,
                    "ep34 checkpoint load is not exact")
            require(profile.h9_module_counts(model) == {
                "ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
                "ep34 model topology drift")
            bn_policy = config.get("test", {}).get("bn_policy", "running")
            require(bn_policy == "no_running", "ep34 BN policy drift")
            profile.configure_batch_norm_evaluation(model, bn_policy)

            # The permit is born, consumed by the fixed producer and discarded
            # entirely inside this child.  It is never an interprocess authority.
            permit = m1558.issue_preload_permit(WORK)
            producer = m1558.ReducedBinaryProducer(
                model, m1558.TorchBinaryAdapter(torch), WORK, specs,
                sample_order, permit, production_inventory=True)
            with torch.no_grad():
                for row in samples:
                    profile.functional.reset_net(model)
                    producer.begin_sample(row)
                    array = np.load(row["resolved_path"], allow_pickle=False)
                    require(array.shape == (10, 480, 640) and
                            array.dtype == np.float32,
                            "raw input tensor identity drift")
                    chunk = torch.from_numpy(array.copy()).unsqueeze(0)
                    label = torch.zeros((1, 2, 480, 640), dtype=torch.float32)
                    mask = torch.ones((1, 480, 640), dtype=torch.float32)
                    x, _, _ = profile.preprocess_chunk(
                        config, chunk, label, mask, None, device)
                    model(x)
                    torch.cuda.synchronize(device)
                    producer.end_sample()
            output = producer.finalize_source_result()
            require(Path(output) == WORK, "M1558 returned noncanonical work root")
            validation = m1558.validate_binary_result(WORK, specs, sample_order)
            write_child_receipt(WORK, release, load_audit, validation)
            validation = m1558.validate_binary_result(WORK, specs, sample_order)
            require(validation.get("status") ==
                    "PASS_M1558_INCREMENTAL_BINARY_VALIDATION",
                    "M1558 final validation failed")
            require(not os.path.lexists(str(RESULT)),
                    "canonical result appeared before publication")
            WORK.rename(RESULT)
            require(RESULT.is_dir() and not WORK.exists(),
                    "atomic result publication failed")
            published = True
    except BaseException:
        close_failed_producer(producer)
        if WORK.is_dir() and not os.path.lexists(str(FAILURE)):
            WORK.rename(FAILURE)
        raise
    finally:
        if not published:
            close_failed_producer(producer)
    print(PASS_TOKEN + " " + str(RESULT), flush=True)
    return 0


def launch_parent():
    verify_fixed_metadata(expect_future_absent=False)
    validate_future_authorities()
    require_fresh_namespaces()
    # No inherited module-search override or caller capability crosses this edge.
    child_environment = {
        "PATH": "/opt/conda/envs/sdformerflow/bin:/usr/bin:/bin",
        "HOME": "/root",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    command = [str(CHILD_PYTHON), "-I", str(SOURCE), "--fixed-clean-child"]
    completed = subprocess.run(command, cwd=str(ROOT), env=child_environment,
                               stdin=subprocess.DEVNULL, check=False)
    require(completed.returncode == 0, "fixed clean child failed; no retry")
    return 0


def source_self_check():
    value = verify_fixed_metadata(expect_future_absent=True)
    require_fresh_namespaces()
    value.update({
        "status": "PASS_M1624_SOURCE_SELF_CHECK__NO_PAYLOAD_NO_GPU_NO_CAPTURE",
        "future_review_present": False, "future_release_present": False,
        "child_processes": 0, "gpu": False, "capture": False,
        "aee": False, "tsbg_dse": False, "rtl": False, "eda": False,
    })
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--source-self-check", action="store_true")
    modes.add_argument("--launch-parent", action="store_true")
    modes.add_argument("--fixed-clean-child", action="store_true",
                       help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.source_self_check:
        print(json.dumps(source_self_check(), indent=2, sort_keys=True,
                         allow_nan=False))
        return 0
    if args.launch_parent:
        return launch_parent()
    return fixed_clean_child()


if __name__ == "__main__":
    raise SystemExit(main())
