#!/usr/bin/env python3
"""Read-only M1180 source hammer. Never calls capture, remote, GPU, or EDA."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / ("neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                 "capture_m1180_motion_checkpoint_parametric_unified_hardware_r2.py")
CONTRACT = HW / "contracts/m1180_motion_checkpoint_parametric_unified_capture_source_contract_r2_20260830.json"
TEST = HW / "tests/test_m1180_motion_checkpoint_parametric_unified_capture_r2_source.py"
AUTHOR = HW / "reviews/m1180_motion_checkpoint_parametric_unified_capture_r2_author_r1_20260830"
M1178 = HW / "reviews/m1178_m1177_motion_checkpoint_parametric_unified_capture_r2_source_hammer_r1_20260830"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def rejected(fn, label: str) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError("mutation accepted: " + label)


def main() -> int:
    checks: list[str] = []
    need(sha(SOURCE) == "f88426c789c99a0d56c34ffaa742b052c73fcbad600c4ecd5797a62e2cf26479", "source SHA")
    need(sha(CONTRACT) == "bcc91d46cf02b3b3d1011287fb7c4d287431db08dba71eef22d5037b06c1d8df", "contract SHA")
    need(sha(TEST) == "6cb33ac3abcbc8678d8a3038afb87a895d7cc65cdd3f5d2fe4307b19f96ad57d", "test SHA")
    checks.append("exact source/contract/test SHA")

    spec = importlib.util.spec_from_file_location("m1181_hammered_m1180", SOURCE)
    need(spec is not None and spec.loader is not None, "source import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    policy = module.strict_json(CONTRACT)
    technical = module.load_technical_policy(policy)

    module.canonical_verify_double_seal(
        AUTHOR,
        "1363a7256655b8b64874099b6de7d4ac87a93ffe5712afa5fdfcb94371393547",
        "d7bc3196af16c8f97fbc07bd11ac477f8b942b222042b372e459843a6cfe7e36")
    receipt = module.strict_json(AUTHOR / "author_receipt.json")
    need(receipt["artifacts"]["source"]["sha256"] == sha(SOURCE), "author source binding")
    need(receipt["artifacts"]["contract"]["sha256"] == sha(CONTRACT), "author contract binding")
    need(receipt["artifacts"]["test"]["sha256"] == sha(TEST), "author test binding")
    checks.append("author recursive seal and exact artifact bindings")

    module.canonical_verify_double_seal(
        M1178,
        "eb55c07ce22b3512081bafa4bb6d7cc04614e22a721e0fcb871975837fc6ee12",
        "100e5bd184c860474e7708bfc95bc7eaf035a994359ea344a78b81d41c2ac677")
    old_review = module.strict_json(M1178 / "review.json")
    need(old_review["verdict"] == "FAIL_CLOSED" and
         old_review["blocking_finding"]["id"] == "B1_M1177_NAMESPACE_COLLISION", "M1178 blocker")
    pinned = {
        "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1177_motion_checkpoint_parametric_unified_hardware_r2.py": "b2b578f7f38643c7e05bd3413101bd5a6eb4f5224e0c0468ff088997f7602184",
        "hw_autoresearch_nts07/contracts/m1177_motion_checkpoint_parametric_unified_capture_source_contract_r2_20260830.json": "5e15abe12a2640df3f8ec22c2f9e64c2fc9fb32e7a9cd78a4b0a93c2077b83d1",
        "hw_autoresearch_nts07/tests/test_m1177_motion_checkpoint_parametric_unified_capture_r2_source.py": "b2587c7818699f285cc709b1c7a8e5d7517e8ee4f7fb0ee486e8f658f06cb9fb",
        "hw_autoresearch_nts07/system_handoff/scripts/run_m1177_motion_ep29_e1e8_closure_source.py": "2f15c406ac8238f1389ead96b044848e0debc1529a13326a488299c42067a19d",
        "hw_autoresearch_nts07/contracts/m1177_motion_ep29_e1e8_source_contract_r1_20260830.json": "d27fb4eebb60f4d828775838539f10080990f8a5262a44134a60c9a55337dfb7",
        "hw_autoresearch_nts07/reviews/m1177_motion_ep29_e1e8_source_author_r1_20260830/author_receipt.json": "671d42b3a99318ed839751f29d70e42306436183bf7d2e2a4e029351cafbb6a7",
        "hw_autoresearch_nts07/reviews/m1179_m1177_motion_ep29_e1e8_source_hammer_r1_20260830/review.json": "8f679f1416e45204c027484bd68376726fdea3b021d5479f5236a76cd185996d",
    }
    for name, expected in pinned.items():
        need(sha(ROOT / name) == expected, "immutable M1177 artifact " + name)
    checks.append("both pre-existing M1177 packages unchanged; sole collision removed by M1180")

    run = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "unittest",
         "hw_autoresearch_nts07.tests.test_m1180_motion_checkpoint_parametric_unified_capture_r2_source", "-v"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (OUT / "controlled_test_log.txt").write_text(run.stdout, encoding="utf-8")
    need(run.returncode == 0 and "Ran 12 tests" in run.stdout and "OK" in run.stdout, "controlled tests")
    checks.append("controlled tests 12/12")

    selected = module.validate_m1175()
    need(selected["selection"]["epoch"] == 29, "M1175 selection")
    checks.append("M1175 exact semantic identity pin")

    # Prove that a future release consumes the actual recursively sealed hammer,
    # then reject semantic and identity mutations of that object.
    with tempfile.TemporaryDirectory() as name:
        temp_root = Path(name)
        temp_hw = temp_root / "hw_autoresearch_nts07"
        hammer = temp_hw / "reviews/m1181_m1180_motion_checkpoint_parametric_unified_capture_r2_source_hammer_fake"
        hammer.mkdir(parents=True)
        review = {
            "schema": module.HAMMER_SCHEMA, "status": "PASS",
            "source_sha256": sha(SOURCE), "contract_sha256": sha(CONTRACT),
            "test_sha256": sha(TEST), "authorization": {"production_release": True},
        }
        (hammer / "review.json").write_text(json.dumps(review) + "\n", encoding="utf-8")
        module.canonical_write_double_seal(hammer)
        launch = {"inputs": {"m1180_source_hammer": {
            "path": str(hammer.relative_to(temp_root)),
            "manifest_sha256": sha(hammer / "SHA256SUMS"),
            "outer_file_sha256": sha(hammer / "SHA256SUMS.seal.sha256"),
            "review_sha256": sha(hammer / "review.json")}}}
        with mock.patch.object(module, "ROOT", temp_root), mock.patch.object(module, "HW", temp_hw):
            need(module.validate_m1181_hammer(launch, policy)["status"] == "PASS", "future hammer consume")
        for field, value in (("status", "FAIL"), ("source_sha256", "0" * 64),
                             ("contract_sha256", "0" * 64), ("test_sha256", "0" * 64)):
            bad = copy.deepcopy(review); bad[field] = value
            target = temp_hw / ("reviews/bad_" + field)
            target.mkdir()
            (target / "review.json").write_text(json.dumps(bad) + "\n", encoding="utf-8")
            module.canonical_write_double_seal(target)
            bad_launch = copy.deepcopy(launch)
            entry = bad_launch["inputs"]["m1180_source_hammer"]
            entry.update({"path": str(target.relative_to(temp_root)),
                          "manifest_sha256": sha(target / "SHA256SUMS"),
                          "outer_file_sha256": sha(target / "SHA256SUMS.seal.sha256"),
                          "review_sha256": sha(target / "review.json")})
            with mock.patch.object(module, "ROOT", temp_root), mock.patch.object(module, "HW", temp_hw):
                rejected(lambda b=bad_launch: module.validate_m1181_hammer(b, policy), "hammer " + field)
    checks.append("future actual hammer consume plus status/source/contract/test rejection")

    launch = {"cohort": {"samples": copy.deepcopy(technical["frozen_samples"])}}
    rows = module.validate_fixed_samples(launch, technical)
    need(len(rows) == 40 and len({row["path"] for row in rows}) == 40 and
         len({row["sha256"] for row in rows}) == 40, "forty identities")
    for mutation in ("duplicate", "order", "cohort", "path", "bytes", "sha", "key"):
        bad = copy.deepcopy(launch)
        samples = bad["cohort"]["samples"]
        if mutation == "duplicate": samples[1] = copy.deepcopy(samples[0]); samples[1]["global_sample_id"] = 1
        elif mutation == "order": samples[10], samples[11] = samples[11], samples[10]
        elif mutation == "cohort": samples[10]["cohort"] = "c1"
        elif mutation == "path": samples[0]["path"] = samples[1]["path"]
        elif mutation == "bytes": samples[0]["bytes"] += 1
        elif mutation == "sha": samples[0]["sha256"] = "0" * 64
        else: samples[0]["sample_key"] = "wrong.npy"
        rejected(lambda b=bad: module.validate_fixed_samples(b, technical), "cohort " + mutation)
    checks.append("exact ordered unique 40 path/size/SHA cohort and mutations")

    inventory = module.frozen_inventory(technical)
    need({key: len(value) for key, value in inventory.items()} == {
        "c1_conv3x3": 4, "decoder_convtranspose": 4, "fc1": 12, "fc2": 12,
        "qkv": 24, "patch_embed": 8, "batch_norm": 78, "attention": 12}, "inventory")
    need(len(module.ATTENTION_ALIASES) == 12, "attention aliases")
    checks.append("4 C1 + 4 decoder + 105/12/12/8/78/24/12x40 inventory gates inherited")

    with tempfile.TemporaryDirectory() as name:
        payload = Path(name) / "attention.npz"
        np.savez(payload, q_bits_packed=np.array([1], dtype=np.uint8),
                 k_bits_packed=np.array([1], dtype=np.uint8), gate_q17=np.array([1], dtype=np.int32))
        base = {"windows_captured": 1, "q_active_bits": 1, "k_active_bits": 1,
                "gate_nonzero": 1, "file": str(payload), "sha256": sha(payload)}
        writer = object.__new__(module.StrictAttentionWriter)
        writer.records = [{**base, "sample_id": sample, "name": alias}
                          for sample in range(40) for alias in module.ATTENTION_ALIASES]
        writer._assert_complete()
        bad = copy.deepcopy(writer.records); bad.pop()
        writer.records = bad
        rejected(writer._assert_complete, "attention 479")
        writer.records = [{**base, "sample_id": sample, "name": alias}
                          for sample in range(40) for alias in module.ATTENTION_ALIASES]
        writer.records[0]["gate_nonzero"] = -1
        rejected(writer._assert_complete, "attention partial")
    checks.append("attention exact 480 Cartesian Q/K/gate payload gate and mutations")

    with tempfile.TemporaryDirectory() as name:
        root = Path(name); (root / "deep").mkdir(); item = root / "deep/x"; item.write_bytes(b"x")
        module.canonical_write_double_seal(root)
        need(set(module.canonical_verify_double_seal(root)) == {"deep/x"}, "recursive seal")
        item.write_bytes(b"tamper")
        rejected(lambda: module.canonical_verify_double_seal(root), "seal tamper")
    checks.append("recursive nested double seal plus tamper rejection")

    need(module.CANONICAL_LEASE == HW / "results/gpu_profile_lease.lock", "canonical lease")
    need(module.CANONICAL_RESULT.name.startswith("m1180_") and
         module.CANONICAL_ATTEMPT.name.startswith(".m1180_") and
         module.CANONICAL_LOG.name.startswith(".m1180_"), "M1180 namespaces")
    need(module.PASS_TOKEN.startswith("PASS_M1180_") and module.ATTEMPT_TOKEN.startswith("M1180_"), "M1180 tokens")
    need(sha(HW / "docs/359_DATE终局冻结_20260813.md") == policy["docs359_sha256"], "docs359")
    checks.append("canonical lease, namespace closure, docs359 invariant")

    output = {"status": "PASS", "checks": checks, "checks_passed": len(checks),
              "remote": False, "gpu": False, "eda": False, "production": False}
    (OUT / "hammer_output.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
