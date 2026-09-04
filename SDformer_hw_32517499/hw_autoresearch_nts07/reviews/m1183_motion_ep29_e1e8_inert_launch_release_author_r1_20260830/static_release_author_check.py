#!/usr/bin/env python3
"""Static-only author check for two inert M1177r2 launch releases."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_handoff/scripts/run_m1177r2_motion_ep29_e1e8_closure_source.py"
BASE_CONTRACT = HW / "contracts/m1177r2_motion_ep29_e1e8_source_contract_r1_20260830.json"
E8 = HW / "contracts/m1183_motion_ep29_e8_inert_launch_release_r1_20260830.json"
E1 = HW / "contracts/m1183_motion_ep29_e1_inert_launch_release_r1_20260830.json"
M1175_DIR = HW / "reviews/m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_20260830"
M1181_DIR = HW / "reviews/m1181_m1177r2_motion_ep29_e1e8_source_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "b1fae4dd647ef159d4297fdc413f2415a5ffb8347234635f375ff6a7152916b3",
    BASE_CONTRACT: "25a833c7de5e537d41988dd7b613f52e7b67b908655264ea546185a5b450292b",
    E8: "3bc14a2e45837be5e1c5f4c2f0042634b8428f6beaa2152a1f818e0531aa43f5",
    E1: "78995d3a37c9d02527052112129ea11d3f84ce885e9e241e759673fd3c2934c2",
    M1175_DIR / "review.json": "8b83690b8b1130d2335bb118d35645ae4d172740966ab69c6fcea9bc8b5d307b",
    M1175_DIR / "SHA256SUMS": "2a4481491d3d12bcba17263260a87e6511e523b4b410e18f3c7fecada07ab247",
    M1175_DIR / "SHA256SUMS.seal.sha256": "17a306168fc3c39b86e869f2213a6592c677203bed52913bf4e6fff29390199e",
    M1181_DIR / "review.json": "2b8166cca43fff4d6c153c824734d36f39a452ba53aaff77be08ca4ed610db95",
    M1181_DIR / "SHA256SUMS": "915cc243ed23496704cbce00b6f933e2fda7c0062a24e800430a1a210e3ce98f",
    M1181_DIR / "SHA256SUMS.seal.sha256": "8c376768a4a3942f9c20d509d511691f5b947f128a3dbd1608f0a60ff4f194e1",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
REMOTE_REPO = "/root/private_data/work/sdformer_codex/SDformer"
REMOTE_INTERPRETER = "/opt/conda/envs/sdformerflow/bin/python"
CHECKPOINT = REMOTE_REPO + "/neuron_experiments/H9_bipolar_self_attention/results/date_two_contribution_full30_20260826/c12_binary_motion_ttx/checkpoint_epoch29.pth"
CONFIG = REMOTE_REPO + "/neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def strict(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            need(key not in result, "duplicate key: " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(isinstance(value, dict), "non-object JSON")
    return value


def check_sidecars(path: Path) -> None:
    inner = path.with_name(path.name + ".sha256")
    outer = path.with_name(path.name + ".sha256.seal.sha256")
    need(inner.read_text(encoding="utf-8").split() == [sha(path), path.name],
         "inner sidecar drift: " + path.name)
    need(outer.read_text(encoding="utf-8").split() == [sha(inner), inner.name],
         "outer sidecar drift: " + path.name)


def check_launch(path: Path, mode: str) -> dict[str, Any]:
    value = strict(path)
    need(set(value) == {"schema", "status", "mode", "contract_path", "common",
                        "output", "one_shot", "gpu_ownership", mode},
         mode + " top exact keys")
    need(value["schema"] == "m1177r2_motion_ep29_e1e8_launch_v2" and
         value["status"] == "HAMMERED_R2_SOURCE__M1175_BOUND__EXACTLY_ONE_MODE_AUTHORIZED" and
         value["mode"] == mode and value["contract_path"] == str(path.relative_to(ROOT)),
         mode + " schema/status/path")
    common = value["common"]
    need(set(common) == {"source", "selection", "checkpoint_path", "config_path",
                         "m1175_result_hammer", "m1177r2_source_hammer"},
         mode + " common exact keys")
    need(common["source"] == {"path": str(SOURCE.relative_to(ROOT)),
                              "sha256": EXPECTED[SOURCE]}, mode + " source binding")
    need(common["checkpoint_path"] == CHECKPOINT and common["config_path"] == CONFIG,
         mode + " remote selected paths")
    selection = common["selection"]
    need(selection == {"epoch": 29,
                       "checkpoint_sha256": "2144dfd628cd928bfb768b92d4fa097b720db112c32d930b9f3cd85c6217286a",
                       "checkpoint_size_bytes": 225504447,
                       "checkpoint_mtime_ns": 1788057827000000000,
                       "config_sha256": "c7b5b994cb9f9a43478f3cb7c09e52a7aecf529fcd6a590f982a291e9eeed955",
                       "standard_valid825": {"samples": 825,
                                             "AEE": 1.209876834190253,
                                             "AAE": 5.406798340046045,
                                             "AAE_Benchmark": 5.148612399245754}},
         mode + " ep29 identity")
    need(common["m1175_result_hammer"] == {
        "path": str((M1175_DIR / "review.json").relative_to(ROOT)),
        "sha256": EXPECTED[M1175_DIR / "review.json"]}, mode + " M1175 binding")
    need(common["m1177r2_source_hammer"] == {
        "path": str((M1181_DIR / "review.json").relative_to(ROOT)),
        "review_sha256": EXPECTED[M1181_DIR / "review.json"],
        "manifest_sha256": EXPECTED[M1181_DIR / "SHA256SUMS"],
        "outer_sha256": EXPECTED[M1181_DIR / "SHA256SUMS.seal.sha256"]},
         mode + " M1181 binding")
    need(value["gpu_ownership"] == {
        "lease_path": "hw_autoresearch_nts07/results/gpu_profile_lease.lock"},
         mode + " canonical lease")
    need(set(value["output"]) == {"path"} and
         set(value["one_shot"]) == {"attempt_marker"}, mode + " one-shot exact keys")
    check_sidecars(path)
    return value


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    rows = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    need(len(rows) == 1, "function population drift: " + name)
    return rows[0]


def main() -> int:
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "authority drift: " + str(path))
    m1175 = strict(M1175_DIR / "review.json")
    m1181 = strict(M1181_DIR / "review.json")
    need(m1175["schema"] == "m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_v1" and
         m1175["status"] == "PASS" and
         m1175["authorization_after_hammer"]["E0_final_checkpoint_and_deployment_identity"] == "ADMITTED",
         "M1175 semantic authority drift")
    need(m1181["schema"] == "m1181_m1177r2_motion_ep29_e1e8_source_hammer_review_r1_v1" and
         m1181["status"] == "PASS_SOURCE_HAMMER__RELEASE_AUTHORING_ALLOWED" and
         m1181["production_authorized"] is False and
         all(m1181["verified"]["B" + str(index)] is True for index in range(1, 9)),
         "M1181 semantic authority drift")
    e8 = check_launch(E8, "e8")
    e1 = check_launch(E1, "e1")
    need(e8["output"]["path"] != e1["output"]["path"] and
         e8["one_shot"]["attempt_marker"] != e1["one_shot"]["attempt_marker"],
         "E8/E1 namespace collision")
    need(e8["e8"]["expected_dynamic_samples"] == 40 and
         set(e8["e8"]) == {"canonical_cohort_manifest", "profile", "expected_dynamic_samples"},
         "E8 exact single cohort contract")
    need(e1["e1"]["fixed_modes"] == ["dyadic", "hardware_order"] and
         set(e1["e1"]) == {"fixed_modes", "standard_valid825", "evaluator"},
         "E1 exact fixed modes")
    source_text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    e8_fn = function(tree, "run_e8")
    e1_fn = function(tree, "run_e1")
    build_calls = [node for node in ast.walk(e8_fn) if isinstance(node, ast.Call) and
                   isinstance(node.func, ast.Attribute) and node.func.attr == "build_model"]
    cohort_calls = [node for node in ast.walk(e8_fn) if isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Name) and node.func.id == "load_canonical_cohort"]
    need(len(build_calls) == 1 and len(cohort_calls) == 1,
         "E8 is not one-load/one-canonical-cohort source")
    need('for mode in ("dyadic", "hardware_order")' in source_text and
         'parameter_search": False' in source_text and
         'validation_selection": False' in source_text,
         "E1 fixed candidate/no-search source drift")
    need("fcntl.LOCK_EX | fcntl.LOCK_NB" in source_text and
         source_text.count("running_legacy_watchers()") >= 2 and
         "os.O_EXCL" in source_text and "ATTEMPT_CONSUMED__NO_RETRY" in source_text,
         "GPU lease/watcher/one-shot fail-closed source drift")
    need(sha(DOCS359) == EXPECTED[DOCS359], "docs359 drift")
    print(json.dumps({"status": "PASS_STATIC_INERT_RELEASE_AUTHOR_CHECK",
                      "E8_release_sha256": sha(E8), "E1_release_sha256": sha(E1),
                      "remote_repo": REMOTE_REPO, "remote_interpreter": REMOTE_INTERPRETER,
                      "E8_one_model_load": True, "E8_exact_cohort_samples": 40,
                      "E1_fixed_modes": ["dyadic", "hardware_order"],
                      "release_executed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
