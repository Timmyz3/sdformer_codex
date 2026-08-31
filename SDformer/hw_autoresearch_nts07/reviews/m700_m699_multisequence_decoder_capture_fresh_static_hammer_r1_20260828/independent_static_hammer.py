#!/usr/bin/env python3
"""Fresh, GPU-free static hammer for the M699 multi-sequence capture.

This audit deliberately does not execute the H67 model, CUDA, or any EDA
tool.  It independently checks every frozen byte root and exercises only
small CPU fixtures against the representation helpers.
"""

from __future__ import print_function

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile

import numpy as np
import torch


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
AUTHOR = HW / "reviews/m699_multisequence_decoder_capture_author_handoff_r1_20260828"
PRODUCER = ROOT / ("neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                   "capture_m699_h67_ep35_multisequence_decoder_payload.py")
RUNNER = HW / ("system_handoff/scripts/"
               "run_m699_h67_ep35_multisequence_decoder_payload_one_shot.sh")
CONTRACT = HW / ("contracts/"
                 "m699_h67_ep35_multisequence_decoder_payload_contract_r1_20260828.json")
TESTS = HW / "system_simulator/tests/test_m699_multisequence_decoder_capture_author.py"
OUTPUT = HW / ("system_handoff/outgoing/"
               "m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828")
ATTEMPT = HW / "results/.m699_h67_ep35_multisequence_decoder_payload_r1_attempt_consumed"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "producer": "fdd88b0285c329ea13466093479b2dc52e9242a7312d4fdce14903cdef1a1769",
    "runner": "9c0e8052577fce7e306ee41bae1a9c27434d0779511a1e6f910bfa5bdf75b958",
    "contract": "43d3b024c1a78d8bc2422af3846c9a376a67bedbecb2ff7396a17bc51ec68fc7",
    "tests": "c58b8591ede93f7b7a153bfa4524af25da9359f1b0cacd2d72f3b69776e73fe7",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "m511": "e16a454d532acd15d96527cfddf43ebf9f95338a34ce9aeedbb10032cb26230a",
    "m686": "1bcff2257e95983ddc77485a41cc4727e082c9297e7312ad534abbb28cf2c630",
}
SEQUENCES = ("interlaken_01_a", "thun_01_b", "zurich_city_12_a")
EXPECTED_INDICES = {
    "interlaken_01_a": [0, 12, 24, 36, 48, 59, 71, 83, 95, 107],
    "thun_01_b": [0, 8, 16, 25, 33, 41, 49, 58, 66, 74],
    "zurich_city_12_a": [0, 8, 16, 25, 33, 41, 49, 58, 66, 74],
}


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
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def tree_files(directory):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "unsafe tree root: " + str(directory))
    result = set()
    for base, dirnames, filenames in os.walk(str(directory), followlinks=False):
        base_path = Path(base)
        for name in dirnames:
            mode = os.lstat(str(base_path / name)).st_mode
            require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),
                    "unsafe directory member")
        for name in filenames:
            path = base_path / name
            mode = os.lstat(str(path)).st_mode
            require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode),
                    "unsafe file member")
            result.add(path.relative_to(directory).as_posix())
    return result


def safe_member(name):
    member = PurePosixPath(name)
    require(not member.is_absolute() and member.parts and
            ".." not in member.parts and member.parts[0] not in ("", "."),
            "unsafe sealed member")
    return member


def verify_double_seal(directory):
    directory = Path(directory)
    files = tree_files(directory)
    seal = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    fields = outer.read_text(encoding="utf-8").strip().split()
    require(fields == [sha256(seal), "SHA256SUMS"], "outer seal mismatch")
    sealed = set()
    for raw in seal.read_text(encoding="utf-8").splitlines():
        fields = raw.split(None, 1)
        require(len(fields) == 2, "malformed seal line")
        expected, name = fields[0], safe_member(fields[1].strip()).as_posix()
        require(name not in sealed and sha256(directory / name) == expected,
                "member seal mismatch: " + name)
        sealed.add(name)
    require(sealed == files - {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
            "sealed population mismatch")
    return {"members": len(sealed), "manifest_sha256": sha256(seal),
            "outer_seal_file_sha256": sha256(outer)}


def repo_path(relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "unsafe repo path")
    target = ROOT / relative
    cursor = Path(target.anchor)
    for part in target.parts[1:]:
        cursor /= part
        require(os.path.lexists(str(cursor)) and not cursor.is_symlink(),
                "missing/symlink path component: " + str(cursor))
    return target.resolve(strict=True)


def verify_claim_boundary(contract):
    boundary = contract["claim_boundary"]
    require(boundary["payload"] is True and boundary["density"] is True and
            boundary["same_checkpoint_multisequence"] is True,
            "capture claim population drift")
    for key in ("accuracy", "cycles", "speedup", "system_speedup", "rtl",
                "vcs", "eda", "dc", "formality", "ptpx", "energy",
                "ppa", "date_headline"):
        require(boundary.get(key) is False, "claim upgrade: " + key)


def verify_sources(contract, check_bytes=True):
    rows = contract["selected_sources"]
    require(len(rows) == 30 and
            [row["global_sample_id"] for row in rows] == list(range(30)),
            "global source lattice drift")
    total = 0
    for sequence_position, sequence in enumerate(SEQUENCES):
        cohort = rows[10 * sequence_position:10 * (sequence_position + 1)]
        require([row["sequence"] for row in cohort] == [sequence] * 10 and
                [row["sequence_sample_id"] for row in cohort] == list(range(10)),
                "sequence/sample substitution: " + sequence)
        directory = repo_path(contract["source_root"] + "/" + sequence)
        population = sorted(directory.glob("*.npy"))
        require(len(population) == int(cohort[0]["source_population"]) and
                not any(path.is_symlink() for path in population),
                "source population drift: " + sequence)
        indices = [round(index * (len(population) - 1) / 9)
                   for index in range(10)]
        require(indices == EXPECTED_INDICES[sequence] and
                indices == [int(row["source_index"]) for row in cohort],
                "selection/index substitution: " + sequence)
        for row, index in zip(cohort, indices):
            path = repo_path(row["path"])
            require(path == population[index].resolve() and
                    path.stat().st_size == int(row["bytes"]),
                    "selected path/size substitution: " + row["path"])
            if check_bytes:
                require(sha256(path) == row["sha256"],
                        "selected content substitution: " + row["path"])
                tensor = np.load(str(path), mmap_mode="r", allow_pickle=False)
                require(tensor.shape == (10, 480, 640) and
                        tensor.dtype == np.dtype("float32"),
                        "selected tensor identity drift")
            total += path.stat().st_size
    return {"files": 30, "bytes": total}


def verify_core(contract):
    expected_names = {"launcher", "runner", "m511_producer", "m686_helper",
                      "m511_contract", "config", "checkpoint", "docs359"}
    require(set(contract["inputs"]) == expected_names,
            "core input population drift")
    for name, entry in contract["inputs"].items():
        path = repo_path(entry["path"])
        require(path.is_file() and path.stat().st_size == int(entry["bytes"]) and
                sha256(path) == entry["sha256"],
                "core input identity drift: " + name)
    require(contract["inputs"]["launcher"]["sha256"] == EXPECTED["producer"] and
            contract["inputs"]["runner"]["sha256"] == EXPECTED["runner"] and
            contract["inputs"]["m511_producer"]["sha256"] == EXPECTED["m511"] and
            contract["inputs"]["m686_helper"]["sha256"] == EXPECTED["m686"] and
            contract["inputs"]["docs359"]["sha256"] == EXPECTED["docs359"],
            "critical dependency reverse binding drift")


def require_fragments(text, fragments, label):
    for fragment in fragments:
        require(fragment in text, label + " missing fragment: " + fragment)


def verify_static_semantics(producer_text, runner_text):
    ast.parse(producer_text)
    require_fragments(producer_text, [
        'missing_count", 0)) == 0', 'unexpected_count", 0)) == 0',
        'current["order"] == index', 'current["order"] == 4',
        'len(records) == 120', 'register_forward_hook',
        'raw["nonbinary_finite_count"] == 0',
        'if scaled["theta_gate_pass"]:',
        'route = "COMMON_FP32_HASH_ONLY_FALLBACK"',
        '"thresholded": False, "rounded": False', '"coerced": False',
        'm686.require_deterministic_execution(determinism)',
        'm686.require_deterministic_execution(m686.observe_execution_controls())',
        'torch.cuda.synchronize(device)', 'verify_core_inputs(contract, launcher)',
        'sources_final = verify_selected_sources(contract)',
        'os.replace(staging, output)', 'verify_double_seal(output)',
    ], "producer semantics")
    require_fragments(runner_text, [
        "set -euo pipefail", '[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" ]]',
        'M699_EXPECTED_RUNNER_SHA256', 'M699_EXPECTED_CONTRACT_SHA256',
        'M699_EXPECTED_REVIEW_SHA256', 'M699_EXPECTED_REVIEW_OUTER_SEAL_SHA256',
        'GO_M699_GPU_ONE_SHOT__CAPTURE_ONLY__P0_0_P1_0',
        'mkdir "${m699_attempt}"', '/usr/bin/env -i',
        'm699_started=1', 'FAIL_CLOSED_NO_CANONICAL_RESULT',
        'mv -- "${m699_output}" "${m699_quarantine}"',
        'sha256sum -c "${m699_attempt}/identity.sha256"',
        'm699_success=1', 'trap - EXIT',
    ], "runner semantics")
    require(runner_text.index('mkdir "${m699_attempt}"') <
            runner_text.index('"${m699_python}" "${m699_producer}"'),
            "attempt is not consumed before producer")
    require(runner_text.count('(cd "${m699_review}" && sha256sum -c SHA256SUMS') == 2,
            "review is not fully rehashed twice")


def load_helper(path, expected_sha):
    require(sha256(path) == expected_sha, "helper changed before CPU fixture")
    spec = importlib.util.spec_from_file_location("m700_m686_fixture", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(sha256(path) == expected_sha, "helper changed during CPU fixture")
    return module


def representation_attacks(contract):
    helper_path = repo_path(contract["inputs"]["m686_helper"]["path"])
    helper = load_helper(helper_path, EXPECTED["m686"])
    with tempfile.TemporaryDirectory(prefix="m700_representation_attack_") as tmp:
        tmp = Path(tmp)
        theta = torch.tensor(0.9999954104423523, dtype=torch.float32)
        scaled = torch.stack([torch.tensor(0.0), theta] * 8)
        raw = helper.summarize_d1_fallback(scaled, 8)
        require(raw["nonbinary_finite_count"] == 8,
                "scaled values were incorrectly admitted as exact binary")
        audit = helper.stream_theta_binary_candidate(
            scaled, theta, 8, tmp / "scaled.bitpack")
        require(audit["theta_gate_pass"] is True and
                audit["other_finite_count"] == 0 and
                audit["thresholded"] is False and audit["rounded"] is False,
                "exact scaled-binary route drift")
        poison = scaled.clone()
        poison[3] = torch.nextafter(theta, torch.tensor(0.0))
        rejected = helper.stream_theta_binary_candidate(
            poison, theta, 8, tmp / "poison.bitpack")
        require(rejected["theta_gate_pass"] is False and
                rejected["other_finite_count"] == 1 and
                not (tmp / "poison.bitpack").exists(),
                "near-theta value was rounded/coerced into scaled binary")
    return "BLOCKED_EXACT_VALUE_GATES__NO_THRESHOLD_ROUND_OR_COERCE"


def attack_suite(contract, producer_text, runner_text):
    attacks = {}
    mutant = copy.deepcopy(contract)
    mutant["selected_sources"][0]["sha256"] = "0" * 64
    try:
        verify_sources(mutant, check_bytes=True)
    except RuntimeError:
        attacks["input_content_replacement"] = "BLOCKED_BY_EXACT_BYTES_SHA"
    else:
        raise RuntimeError("input content attack survived")

    mutant = copy.deepcopy(contract)
    mutant["selected_sources"][0], mutant["selected_sources"][10] = (
        mutant["selected_sources"][10], mutant["selected_sources"][0])
    try:
        verify_sources(mutant, check_bytes=False)
    except RuntimeError:
        attacks["sequence_sample_substitution"] = "BLOCKED_BY_ORDERED_3X10_LATTICE"
    else:
        raise RuntimeError("sequence/sample attack survived")

    mutant_text = producer_text.replace('current["order"] == 4',
                                         'current["order"] >= 0', 1)
    try:
        verify_static_semantics(mutant_text, runner_text)
    except RuntimeError:
        attacks["missing_hook_acceptance"] = "BLOCKED_BY_PER_SAMPLE_4_AND_FINAL_120_GATES"
    else:
        raise RuntimeError("missing-hook attack survived")

    attacks["scaled_as_binary_or_near_theta_coercion"] = representation_attacks(contract)

    mutant = copy.deepcopy(contract)
    mutant["claim_boundary"]["speedup"] = True
    try:
        verify_claim_boundary(mutant)
    except RuntimeError:
        attacks["claim_upgrade"] = "BLOCKED_CAPTURE_DENSITY_ONLY"
    else:
        raise RuntimeError("claim upgrade survived")

    fake_review = {
        "status": "GO_M699_GPU_ONE_SHOT__CAPTURE_ONLY__P0_0_P1_0",
        "severity": {"p0": 0, "p1": 0, "p2": 0},
        "execution_authorized": True,
        "reviewed_inputs": {"runner_sha256": "0" * 64,
                            "contract_sha256": EXPECTED["contract"]},
        "claim_boundary": {"cycles": False, "speedup": False,
                           "system_speedup": False},
    }
    accepted = (fake_review["status"] ==
                "GO_M699_GPU_ONE_SHOT__CAPTURE_ONLY__P0_0_P1_0" and
                fake_review["severity"]["p0"] == 0 and
                fake_review["severity"]["p1"] == 0 and
                fake_review["execution_authorized"] and
                fake_review["reviewed_inputs"]["runner_sha256"] ==
                EXPECTED["runner"] and
                fake_review["reviewed_inputs"]["contract_sha256"] ==
                EXPECTED["contract"])
    require(not accepted, "old-review forgery survived reviewed-input binding")
    attacks["old_review_forgery"] = "BLOCKED_BY_EXACT_REVIEW_ROOT_AND_INPUT_REVERSE_BINDING"

    mutant_runner = runner_text.replace(
        'echo "FAIL_CLOSED_NO_CANONICAL_RESULT rc=${rc}"',
        'echo "FAIL_OPEN rc=${rc}"', 1)
    try:
        verify_static_semantics(producer_text, mutant_runner)
    except RuntimeError:
        attacks["failure_receipt_removal"] = "BLOCKED_BY_STATIC_FAIL_CLOSED_SENTINEL_GATE"
    else:
        raise RuntimeError("failure-receipt attack survived")
    return attacks


def main():
    require(sha256(PRODUCER) == EXPECTED["producer"] and
            sha256(RUNNER) == EXPECTED["runner"] and
            sha256(CONTRACT) == EXPECTED["contract"] and
            sha256(TESTS) == EXPECTED["tests"] and
            sha256(DOCS359) == EXPECTED["docs359"],
            "reviewed top-level identity drift")
    author_seal = verify_double_seal(AUTHOR)
    author = strict_json(AUTHOR / "author_handoff.json")
    require(author["status"] ==
            "STATIC_AUTHOR_HANDOFF_COMPLETE__FRESH_HAMMER_REQUIRED_BEFORE_GPU" and
            author["severity"]["p0"] == 0 and author["severity"]["p1"] == 0,
            "author handoff status drift")
    contract = strict_json(CONTRACT)
    require(contract["schema"] ==
            "m699_h67_ep35_multisequence_decoder_payload_contract_v1" and
            contract["status"] ==
            "STATIC_AUTHOR_HANDOFF__FRESH_HAMMER_REQUIRED_BEFORE_GPU",
            "contract schema/status drift")
    verify_claim_boundary(contract)
    verify_core(contract)
    source_inventory = verify_sources(contract)
    producer_text = PRODUCER.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    verify_static_semantics(producer_text, runner_text)
    subprocess.run(["bash", "-n", str(RUNNER)], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(not os.path.lexists(str(OUTPUT)) and
            not os.path.lexists(str(ATTEMPT)),
            "one-shot output/attempt already consumed")
    runtime = contract["runtime"]
    require(runtime["deterministic_execution"] == {
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cublas_workspace_config": ":4096:8",
    }, "native deterministic/TF32 controls drift")
    require(runtime["exact_python_argv"][-8:] == [
        "--sequences", "3", "--samples-per-sequence", "10",
        "--num-workers", "0", "--chunk-elements", "8388608"],
        "exact one-shot argv drift")
    attacks = attack_suite(contract, producer_text, runner_text)
    result = {
        "schema": "m700_m699_fresh_static_independent_audit_v1",
        "status": "PASS_STATIC_HAMMER__GPU_MODEL_EDA_NOT_EXECUTED",
        "identity": {key: EXPECTED[key] for key in
                     ("producer", "runner", "contract", "tests", "docs359",
                      "m511", "m686")},
        "author_double_seal": author_seal,
        "selected_source_inventory": source_inventory,
        "selection": {"sequences": list(SEQUENCES), "samples_each": 10,
                      "total": 30, "expected_hook_calls": 120,
                      "indices": EXPECTED_INDICES},
        "attacks": attacks,
        "prelaunch_state": {"canonical_output_absent": True,
                            "attempt_absent": True},
        "execution": {"gpu": False, "model": False, "eda": False},
    }
    (REVIEW / "independent_audit_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
