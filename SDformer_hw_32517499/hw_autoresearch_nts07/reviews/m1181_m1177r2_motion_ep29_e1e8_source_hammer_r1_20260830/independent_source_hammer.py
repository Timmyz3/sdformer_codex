#!/usr/bin/env python3
"""Fresh different-author fail-closed source hammer for M1177r2.

This hammer is intentionally source-only.  It does not contact a remote host,
open the selected checkpoint, run valid825/range capture, use a GPU, invoke EDA,
or authorize production.  It verifies immutable checked-in authorities, runs
the 17 controlled author tests, and independently attacks the B1--B8 gates.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
from typing import Any, Callable

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_handoff/scripts/run_m1177r2_motion_ep29_e1e8_closure_source.py"
CONTRACT = HW / "contracts/m1177r2_motion_ep29_e1e8_source_contract_r1_20260830.json"
TESTS = HW / "tests/test_run_m1177r2_motion_ep29_e1e8_closure_source.py"
COHORT = HW / "contracts/m1177r2_motion_ep29_e8_canonical_40_source_manifest_r1_20260830.json"
M1175 = HW / "reviews/m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_20260830/review.json"
PROFILE = ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py"
EVALUATOR = ROOT / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "source": "b1fae4dd647ef159d4297fdc413f2415a5ffb8347234635f375ff6a7152916b3",
    "contract": "25a833c7de5e537d41988dd7b613f52e7b67b908655264ea546185a5b450292b",
    "tests": "6eca12c2f34acf004d7aefbf9dc78e4696777da48555703cc3d5b3813581c650",
    "cohort": "56bc2e9b032a895c9700d5a6e83cc85c9f32e3f1505848264ad9ee5f38c000db",
    "m1175": "8b83690b8b1130d2335bb118d35645ae4d172740966ab69c6fcea9bc8b5d307b",
    "profile": "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684",
    "evaluator": "ba40b42c7395fd703c59a183a19b6a4fd38fa08ed75201008f03fd71b82aaef1",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            RuntimeError("non-finite JSON: " + token)))
    need(isinstance(result, dict), "JSON root is not object")
    return result


def expect_reject(label: str, callback: Callable[[], Any], attacks: list[str]) -> None:
    try:
        callback()
    except Exception:
        attacks.append(label)
        return
    raise RuntimeError("mutation was accepted: " + label)


class ATLIFTernaryPSN(torch.nn.Module):
    def forward(self, value: Any) -> Any:
        return value


class FixtureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 3, bias=False)
        self.bn = torch.nn.BatchNorm2d(3)
        self.atlif = ATLIFTernaryPSN()
        self.deconv = torch.nn.ConvTranspose2d(3, 2, 3)
        self.fc = torch.nn.Linear(2, 2)


def seal_directory(directory: Path, review: dict[str, Any]) -> dict[str, str]:
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()
    review_path = directory / "review.json"
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    manifest = directory / "SHA256SUMS"
    manifest.write_text(sha(review_path) + "  review.json\n", encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(sha(manifest) + "  SHA256SUMS\n", encoding="utf-8")
    return {"path": str(review_path.relative_to(ROOT)),
            "review_sha256": sha(review_path),
            "manifest_sha256": sha(manifest), "outer_sha256": sha(outer)}


class Fixture:
    """Controlled bytes only; no selected checkpoint or production input."""
    def __init__(self, m: Any, mode: str = "e1") -> None:
        self.m = m
        self.tmp = tempfile.TemporaryDirectory(prefix=".m1181_fixture.", dir=HW / "results")
        self.root = Path(self.tmp.name)
        self.saved = {name: getattr(m, name) for name in (
            "DOCS359", "DOCS359_SHA256", "M1175_REVIEW", "M1175_REVIEW_SHA256",
            "PROFILE", "PROFILE_SHA256", "EVALUATOR", "EVALUATOR_SHA256",
            "EXPECTED_SOURCE_HAMMER_REVIEW", "EXPECTED_CHECKPOINT",
            "EXPECTED_CONFIG_SHA256")}
        self.checkpoint = self.root / "synthetic_fixture.bin"
        self.checkpoint.write_bytes(b"synthetic-not-a-checkpoint")
        self.config = self.root / "config.yml"
        self.config.write_text("bsa_attention:\n  enabled: true\n", encoding="utf-8")
        self.docs = self.root / "docs359"
        self.docs.write_text("synthetic protected bytes\n", encoding="utf-8")
        self.profile = self.root / "profile.py"
        self.profile.write_text("# synthetic pinned profile\n", encoding="utf-8")
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text("# synthetic pinned evaluator\n", encoding="utf-8")
        m.DOCS359, m.DOCS359_SHA256 = self.docs, sha(self.docs)
        m.PROFILE, m.PROFILE_SHA256 = self.profile, sha(self.profile)
        m.EVALUATOR, m.EVALUATOR_SHA256 = self.evaluator, sha(self.evaluator)
        m.EXPECTED_CHECKPOINT = {"epoch": 29, "sha256": sha(self.checkpoint),
                                 "size_bytes": self.checkpoint.stat().st_size,
                                 "mtime_ns": self.checkpoint.stat().st_mtime_ns}
        m.EXPECTED_CONFIG_SHA256 = sha(self.config)
        self.m1175 = self.root / "m1175.json"
        self.write_m1175()
        m.M1175_REVIEW, m.M1175_REVIEW_SHA256 = self.m1175, sha(self.m1175)
        self.hammer_dir = self.root / "hammer"
        self.hammer_dir.mkdir()
        self.hammer_review = self.hammer_dir / "review.json"
        m.EXPECTED_SOURCE_HAMMER_REVIEW = self.hammer_review
        self.hammer = self.write_hammer()
        self.launch = self.root / "launch.json"
        self.contract = self.make_contract(mode)

    def write_m1175(self, *, status: str = "PASS", e0: str = "ADMITTED",
                    load: bool = True) -> None:
        payload = {
            "schema": "m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_v1",
            "status": status,
            "verified": {"samples825_and_zero_load_audits": load,
                         "module_counts_atlif105_attention12": True,
                         "semantic_mutations_fail_closed": True},
            "selection": {"epoch": 29,
                          "checkpoint_sha256": self.m.EXPECTED_CHECKPOINT["sha256"],
                          "checkpoint_size_bytes": self.m.EXPECTED_CHECKPOINT["size_bytes"],
                          "checkpoint_mtime_ns": self.m.EXPECTED_CHECKPOINT["mtime_ns"],
                          "configuration_sha256": self.m.EXPECTED_CONFIG_SHA256,
                          "samples": 825, "AEE": "1.209876834190253",
                          "AAE": "5.406798340046045",
                          "AAE_Benchmark": "5.148612399245754"},
            "authorization_after_hammer": {
                "E0_final_checkpoint_and_deployment_identity": e0},
        }
        self.m1175.write_text(json.dumps(payload), encoding="utf-8")

    def write_hammer(self, *, status: str =
                     "PASS_SOURCE_HAMMER__RELEASE_AUTHORING_ALLOWED",
                     verified: bool = True, artifact_source: str | None = None) -> dict[str, str]:
        review = {
            "schema": "m1181_m1177r2_motion_ep29_e1e8_source_hammer_review_r1_v1",
            "status": status, "production_authorized": False,
            "artifacts": {
                "source": {"path": str(SOURCE.relative_to(ROOT)),
                           "sha256": artifact_source or sha(SOURCE)},
                "contract": {"path": str(CONTRACT.relative_to(ROOT)), "sha256": sha(CONTRACT)},
                "tests": {"path": str(TESTS.relative_to(ROOT)), "sha256": sha(TESTS)}},
            "verified": {"B" + str(index): verified for index in range(1, 9)},
        }
        return seal_directory(self.hammer_dir, review)

    def make_contract(self, mode: str) -> dict[str, Any]:
        common = {
            "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha(SOURCE)},
            "selection": {"epoch": 29,
                          "checkpoint_sha256": self.m.EXPECTED_CHECKPOINT["sha256"],
                          "checkpoint_size_bytes": self.m.EXPECTED_CHECKPOINT["size_bytes"],
                          "checkpoint_mtime_ns": self.m.EXPECTED_CHECKPOINT["mtime_ns"],
                          "config_sha256": self.m.EXPECTED_CONFIG_SHA256,
                          "standard_valid825": self.m.EXPECTED_STANDARD},
            "checkpoint_path": str(self.checkpoint), "config_path": str(self.config),
            "m1175_result_hammer": {"path": str(self.m1175.relative_to(ROOT)),
                                    "sha256": sha(self.m1175)},
            "m1177r2_source_hammer": self.hammer,
        }
        result: dict[str, Any] = {
            "schema": "m1177r2_motion_ep29_e1e8_launch_v2",
            "status": "HAMMERED_R2_SOURCE__M1175_BOUND__EXACTLY_ONE_MODE_AUTHORIZED",
            "mode": mode, "contract_path": str(self.launch.relative_to(ROOT)),
            "common": common, "output": {"path": "hw_autoresearch_nts07/results/synthetic"},
            "one_shot": {"attempt_marker": "hw_autoresearch_nts07/results/.synthetic"},
            "gpu_ownership": {"lease_path": str(self.m.LEASE.relative_to(ROOT))},
        }
        if mode == "e1":
            result["e1"] = {"fixed_modes": ["dyadic", "hardware_order"],
                            "standard_valid825": self.m.EXPECTED_STANDARD,
                            "evaluator": {"path": str(self.evaluator.relative_to(ROOT)),
                                          "sha256": sha(self.evaluator)}}
        else:
            result["e8"] = {"canonical_cohort_manifest": {
                "path": str(self.m.EXPECTED_COHORT.relative_to(ROOT)),
                "size_bytes": self.m.EXPECTED_COHORT_SIZE,
                "sha256": self.m.EXPECTED_COHORT_SHA256,
                "inner_sha256": self.m.EXPECTED_COHORT_INNER_SHA256,
                "outer_sha256": self.m.EXPECTED_COHORT_OUTER_SHA256},
                "profile": {"path": str(self.profile.relative_to(ROOT)),
                            "sha256": sha(self.profile)}, "expected_dynamic_samples": 40}
        self.launch.write_text(json.dumps(result), encoding="utf-8")
        return result

    def refresh_m1175_binding(self) -> None:
        self.m.M1175_REVIEW_SHA256 = sha(self.m1175)
        self.contract["common"]["m1175_result_hammer"] = {
            "path": str(self.m1175.relative_to(ROOT)), "sha256": sha(self.m1175)}

    def close(self) -> None:
        for key, value in self.saved.items():
            setattr(self.m, key, value)
        self.tmp.cleanup()


def main() -> int:
    artifacts = {"source": SOURCE, "contract": CONTRACT, "tests": TESTS,
                 "cohort": COHORT, "m1175": M1175, "profile": PROFILE,
                 "evaluator": EVALUATOR, "docs359": DOCS359}
    for key, path in artifacts.items():
        need(path.is_file() and not path.is_symlink(), key + " not regular")
        need(sha(path) == EXPECTED[key], key + " SHA drift")

    source_contract = strict_json(CONTRACT)
    need(source_contract.get("schema") == "m1177r2_motion_ep29_e1e8_source_contract_r1_v1",
         "source contract schema drift")
    need(source_contract.get("status") ==
         "R2_SOURCE_ONLY__M1179_FAILURE_CLOSED__FRESH_M1181_DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_PRODUCTION",
         "source contract status drift")
    need(source_contract["source"]["path"] == str(SOURCE.relative_to(ROOT)) and
         source_contract["source"]["sha256"] == EXPECTED["source"],
         "source contract source binding drift")
    need(source_contract["tests"] == {"path": str(TESTS.relative_to(ROOT)),
                                      "sha256": EXPECTED["tests"]},
         "source contract test binding drift")
    need(source_contract["future_hammer_contract"] == {
        "canonical_review_path": str((HERE / "review.json").relative_to(ROOT)),
        "required_schema": "m1181_m1177r2_motion_ep29_e1e8_source_hammer_review_r1_v1",
        "required_status": "PASS_SOURCE_HAMMER__RELEASE_AUTHORING_ALLOWED",
        "production_authorized_by_hammer": False,
        "required_exact_artifacts": ["source", "contract", "tests"],
        "required_verified_keys": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"],
        "fresh_release_contract_after_hammer": True}, "future hammer contract drift")

    py_compile.compile(str(SOURCE), doraise=True)
    py_compile.compile(str(TESTS), doraise=True)
    tests_run = subprocess.run(
        [sys.executable, "-m", "unittest", "-v",
         "hw_autoresearch_nts07.tests.test_run_m1177r2_motion_ep29_e1e8_closure_source"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (HERE / "author_tests.log").write_text(tests_run.stdout, encoding="utf-8")
    need(tests_run.returncode == 0 and "Ran 17 tests" in tests_run.stdout and
         tests_run.stdout.rstrip().endswith("OK"), "17 author tests did not pass exactly")

    m = load("m1181_m1177r2_under_hammer", SOURCE)
    need(m.M1175_REVIEW.resolve() == M1175.resolve() and
         m.M1175_REVIEW_SHA256 == EXPECTED["m1175"], "B1 compiled authority not exact")
    need(m.PROFILE.resolve() == PROFILE.resolve() and m.PROFILE_SHA256 == EXPECTED["profile"] and
         m.EVALUATOR.resolve() == EVALUATOR.resolve() and
         m.EVALUATOR_SHA256 == EXPECTED["evaluator"], "B2 compiled authority not exact")
    need(m.LEASE == HW / "results/gpu_profile_lease.lock", "B3 compiled lease not canonical")
    need(m.EXPECTED_COHORT.resolve() == COHORT.resolve() and
         m.EXPECTED_COHORT_SHA256 == EXPECTED["cohort"], "B5 cohort binding drift")

    actual_m1175 = m.validate_m1175()
    need(actual_m1175["authorization_after_hammer"][
        "E0_final_checkpoint_and_deployment_identity"] == "ADMITTED", "B1 E0 missing")
    cohort_rows = m.load_canonical_cohort(verify_source_bytes=False)
    need(len(cohort_rows) == 40 and
         [row["global_sample_id"] for row in cohort_rows] == list(range(40)) and
         len({(row["path"], row["sha256"]) for row in cohort_rows}) == 40,
         "B5 exact cohort population/order/identity drift")

    attacks: list[str] = []

    # B1: exact authority binding plus semantic status/E0/load admission.
    f = Fixture(m)
    try:
        bad = deepcopy(f.contract)
        bad["common"]["m1175_result_hammer"]["path"] = str(f.config.relative_to(ROOT))
        expect_reject("B1 arbitrary M1175 path", lambda: m.validate_launch(bad, f.launch), attacks)
    finally:
        f.close()
    for label, kwargs in (("B1 bad status", {"status": "NOT_PASS"}),
                          ("B1 E0 absent", {"e0": "NOT_ADMITTED"}),
                          ("B1 zero-load absent", {"load": False})):
        f = Fixture(m)
        try:
            f.write_m1175(**kwargs)
            f.refresh_m1175_binding()
            expect_reject(label, lambda f=f: m.validate_launch(f.contract, f.launch), attacks)
        finally:
            f.close()
    f = Fixture(m)
    try:
        payload = json.loads(f.m1175.read_text(encoding="utf-8"))
        payload["selection"]["epoch"] = 24
        f.m1175.write_text(json.dumps(payload), encoding="utf-8")
        f.refresh_m1175_binding()
        expect_reject("B1 selected epoch drift",
                      lambda: m.validate_launch(f.contract, f.launch), attacks)
    finally:
        f.close()

    # B2/B3/B4: executable dependency bytes, canonical lease, exact mode schemas.
    f = Fixture(m)
    try:
        f.profile.write_text("# mutated synthetic profile\n", encoding="utf-8")
        expect_reject("B2 profile byte mutation", lambda: m.validate_launch(f.contract, f.launch), attacks)
    finally:
        f.close()
    f = Fixture(m)
    try:
        f.evaluator.write_text("# mutated synthetic evaluator\n", encoding="utf-8")
        expect_reject("B2 evaluator byte mutation", lambda: m.validate_launch(f.contract, f.launch), attacks)
    finally:
        f.close()
    f = Fixture(m)
    try:
        f.contract["gpu_ownership"]["lease_path"] = "hw_autoresearch_nts07/results/bypass.lock"
        expect_reject("B3 alternate lease", lambda: m.validate_launch(f.contract, f.launch), attacks)
    finally:
        f.close()
    for label, mutation in (
        ("B4 top extra", lambda c: c.update({"extra": True})),
        ("B4 common extra", lambda c: c["common"].update({"cohort": []})),
        ("B4 E1 extra", lambda c: c["e1"].update({"epsilon": 0})),
        ("B4 mode mix", lambda c: c.update({"e8": {}})),
        ("B4 E1 fixed-mode drift", lambda c: c["e1"].update({"fixed_modes": ["dyadic"]})),
    ):
        f = Fixture(m)
        try:
            mutation(f.contract)
            expect_reject(label, lambda f=f: m.validate_launch(f.contract, f.launch), attacks)
        finally:
            f.close()

    # B5: even a self-consistent reseal cannot replace the compiled canonical cohort.
    original = {key: getattr(m, key) for key in (
        "EXPECTED_COHORT", "EXPECTED_COHORT_SHA256", "EXPECTED_COHORT_SIZE",
        "EXPECTED_COHORT_INNER_SHA256", "EXPECTED_COHORT_OUTER_SHA256")}
    with tempfile.TemporaryDirectory(prefix=".m1181_cohort.", dir=HW / "results") as temporary:
        bad = deepcopy(strict_json(COHORT))
        bad["rows"][1] = deepcopy(bad["rows"][0])
        path = Path(temporary) / "cohort.json"
        path.write_text(json.dumps(bad, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        inner = path.with_name(path.name + ".sha256")
        inner.write_text(sha(path) + "  " + path.name + "\n", encoding="utf-8")
        outer = path.with_name(path.name + ".sha256.seal.sha256")
        outer.write_text(sha(inner) + "  " + inner.name + "\n", encoding="utf-8")
        m.EXPECTED_COHORT, m.EXPECTED_COHORT_SHA256 = path, sha(path)
        m.EXPECTED_COHORT_SIZE = path.stat().st_size
        m.EXPECTED_COHORT_INNER_SHA256, m.EXPECTED_COHORT_OUTER_SHA256 = sha(inner), sha(outer)
        expect_reject("B5 duplicate row after reseal",
                      lambda: m.load_canonical_cohort(verify_source_bytes=False), attacks)
    for key, value in original.items():
        setattr(m, key, value)

    # B6: exact name/type/shape static census and exactly-once dynamic coverage.
    model = FixtureModel()
    census = m.build_model_census(model)
    need(census["counts"] == {"dynamic": 5, "weights": 3, "batch_norm": 1},
         "B6 fixture census population drift")
    with tempfile.TemporaryDirectory(prefix=".m1181_static.", dir=HW / "results") as temporary:
        missing = deepcopy(census["weights"][:-1])
        expect_reject("B6 omitted static layer",
                      lambda: m.export_static(torch, model, Path(temporary), missing), attacks)
    capture = m.RangeCapture(torch)
    capture.attach(model, census["dynamic"])
    capture.begin({"global_sample_id": 0})
    expect_reject("B6 omitted dynamic layers",
                  lambda: capture.end({row["name"] for row in census["dynamic"]}), attacks)
    capture.close()

    # B7: four tensors, finite channels and positive epsilon.
    nan_bn = FixtureModel()
    nan_bn.bn.running_var[0] = float("nan")
    nan_census = m.build_model_census(nan_bn)
    with tempfile.TemporaryDirectory(prefix=".m1181_bn.", dir=HW / "results") as temporary:
        expect_reject("B7 nonfinite running_var",
                      lambda: m.export_bn(nan_bn, Path(temporary), nan_census["batch_norm"]), attacks)
    missing_bn = torch.nn.Sequential(torch.nn.BatchNorm2d(3, track_running_stats=False))
    expect_reject("B7 missing running tensors", lambda: m.build_model_census(missing_bn), attacks)
    bad_eps = FixtureModel()
    bad_eps.bn.eps = 0.0
    bad_eps_census = m.build_model_census(bad_eps)
    with tempfile.TemporaryDirectory(prefix=".m1181_eps.", dir=HW / "results") as temporary:
        expect_reject("B7 nonpositive epsilon",
                      lambda: m.export_bn(bad_eps, Path(temporary), bad_eps_census["batch_norm"]), attacks)

    # B8: canonical path, PASS semantics, exact artifacts/B1-B8 and outer seal.
    for label, parameters in (
        ("B8 non-PASS status", {"status": "NOT_PASS"}),
        ("B8 false verified bit", {"verified": False}),
        ("B8 source artifact drift", {"artifact_source": "0" * 64}),
    ):
        f = Fixture(m)
        try:
            f.contract["common"]["m1177r2_source_hammer"] = f.write_hammer(**parameters)
            expect_reject(label, lambda f=f: m.validate_launch(f.contract, f.launch), attacks)
        finally:
            f.close()
    f = Fixture(m)
    try:
        outer = f.hammer_dir / "SHA256SUMS.seal.sha256"
        outer.write_text("0" * 64 + "  SHA256SUMS\n", encoding="utf-8")
        f.contract["common"]["m1177r2_source_hammer"]["outer_sha256"] = sha(outer)
        expect_reject("B8 outer seal mutation", lambda: m.validate_launch(f.contract, f.launch), attacks)
    finally:
        f.close()
    f = Fixture(m)
    try:
        (f.hammer_dir / "UNSEALED").write_text("extra\n", encoding="utf-8")
        expect_reject("B8 unsealed extra member", lambda: m.validate_launch(f.contract, f.launch), attacks)
    finally:
        f.close()
    f = Fixture(m)
    try:
        f.contract["common"]["m1177r2_source_hammer"]["path"] = str(
            f.config.relative_to(ROOT))
        expect_reject("B8 noncanonical review path",
                      lambda: m.validate_launch(f.contract, f.launch), attacks)
    finally:
        f.close()

    need(len(attacks) == 25, "independent mutation population drift: " + str(len(attacks)))
    need(sha(DOCS359) == EXPECTED["docs359"], "docs359 changed during hammer")
    output = {
        "schema": "m1181_m1177r2_motion_ep29_e1e8_source_hammer_output_r1_v1",
        "status": "PASS_17_AUTHOR_TESTS_AND_25_INDEPENDENT_MUTATIONS",
        "author_tests": 17, "author_test_failures": 0,
        "independent_mutations": len(attacks), "accepted_mutations": 0,
        "rejected_attacks": attacks,
        "artifacts": {key: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
                      for key, path in artifacts.items()},
        "execution_boundary": {"remote": False, "gpu": False,
                               "selected_checkpoint_opened": False,
                               "valid825": False, "range": False,
                               "eda": False, "production": False},
    }
    (HERE / "hammer_output.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
    print("PASS M1181 author_tests=17 mutations=25", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
