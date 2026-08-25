#!/usr/bin/env python3
"""Frozen-H67 opportunity for a same-cycle authoritative done-fence load."""

import argparse
import functools
import hashlib
import importlib.util
import json
import multiprocessing
from collections import defaultdict
from pathlib import Path


EXPECTED_MANIFEST = "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
EXPECTED_M172 = "ae633daee1e07f16396570c1ef757c11bac7f1f72be108f4411d75f6dcb5f6d9"
EXPECTED_M192 = "39715b64890d75be7c60587d639f3e75b51e5bf38642e6b8640c761fa512f24b"
EXPECTED_M213_ANALYZER = "8130123a32cafd6e8700f2f7d88bd2bc1807a06f5d0e447bf943974997d932f7"
EXPECTED_M213_RESULT = "58e420f071736b5393cc5a47fa2404fe36835498822a9c0f1bf7d9506e1eb650"
EXPECTED_M214_MODEL = "01a870f85ae62208d9d9c145021a476a59b26cb2fc0fad343d2ed51006517b5e"
EXPECTED_DOCS359 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

M213 = None


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path, expected, name):
    require(sha256(path) == expected, name + " identity drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fraction(numerator, denominator):
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def run_task(task):
    return M213.audit_record(task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m192-analyzer", required=True, type=Path)
    parser.add_argument("--m213-analyzer", required=True, type=Path)
    parser.add_argument("--m213-result", required=True, type=Path)
    parser.add_argument("--m214-model", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_start = sha256(Path(__file__).resolve())
    require(sha256(args.manifest) == EXPECTED_MANIFEST, "manifest drift")
    require(sha256(args.m213_result) == EXPECTED_M213_RESULT, "M213 drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359, "docs359 drift")

    global M213
    M213 = load_module(args.m213_analyzer, EXPECTED_M213_ANALYZER,
                       "m213_pinned_m214")
    m172 = load_module(args.m172_analyzer, EXPECTED_M172, "m172_pinned_m214")
    m192 = load_module(args.m192_analyzer, EXPECTED_M192, "m192_pinned_m214")
    m214 = load_module(args.m214_model, EXPECTED_M214_MODEL,
                       "m214_opportunity_model")
    # Reuse M213's frozen-payload auditor without duplicating its identity and
    # popcount code; only substitute the controller recurrence entry point.
    raw_simulate = m214.simulate_m214_bank_loads

    @functools.lru_cache(maxsize=100_000)
    def cached_simulate(load_key, depth, output_blocks):
        loads = [None if value is None else tuple(value)
                 for value in load_key]
        return raw_simulate(loads, depth, output_blocks)

    def compatible_simulate(loads, depth, output_blocks):
        load_key = tuple(None if value is None else tuple(value)
                         for value in loads)
        return cached_simulate(load_key, depth, output_blocks)

    m214.simulate_m212_bank_loads = compatible_simulate
    M213.M172 = m172
    M213.M192 = m192
    M213.M212 = m214
    M213.PAYLOAD_ROOT = args.payload_root
    M213.CHUNK_TOKENS = args.chunk_tokens

    manifest = json.loads(args.manifest.read_text())
    records = [record for record in manifest["records"]
               if record["operator"] == "Linear"
               and ".mlp.fc2" in record["name"]]
    require(len(records) == 120, "FC2 record extent drift")
    tasks = list(enumerate(records))
    if args.workers == 1:
        audited = [run_task(task) for task in tasks]
    else:
        context = multiprocessing.get_context("fork")
        with context.Pool(processes=args.workers) as pool:
            audited = list(pool.imap_unordered(run_task, tasks))
    audited.sort(key=lambda item: item[0])
    aggregate = M213.empty_ledger()
    per_stage = defaultdict(M213.empty_ledger)
    for count, (_ordinal, stage, ledger) in enumerate(audited, start=1):
        M213.merge(aggregate, ledger)
        M213.merge(per_stage[stage], ledger)
        print("[M214] {}/120".format(count), flush=True)

    m213 = json.loads(args.m213_result.read_text())
    exact_m212 = m213["aggregate"]["m212_rtl_semantic_cycles"]
    opportunity = aggregate["m212_rtl_semantic_cycles"]
    per_stage_result = {}
    for stage, ledger in sorted(per_stage.items()):
        old = m213["per_stage"][str(stage)]["m212_rtl_semantic_cycles"]
        new = ledger["m212_rtl_semantic_cycles"]
        per_stage_result[str(stage)] = {
            "m212_cycles": old,
            "m214_opportunity_cycles": new,
            "cycles_saved": old - new,
        }
    result = {
        "schema": "m214_h67_fc2_same_cycle_done_load_opportunity_v1",
        "status": "PASS_EXACT_PAYLOAD_OPPORTUNITY__RTL_VCS_PENDING",
        "identity": {
            "analyzer_start_sha256": script_start,
            "manifest_sha256": EXPECTED_MANIFEST,
            "m213_analyzer_sha256": EXPECTED_M213_ANALYZER,
            "m213_result_sha256": EXPECTED_M213_RESULT,
            "m214_model_sha256": EXPECTED_M214_MODEL,
            "docs359_sha256": EXPECTED_DOCS359,
        },
        "architecture_delta": {
            "base": "M212",
            "same_cycle_authoritative_done_fence_for_candidate_load": True,
            "same_cycle_done_and_terminal_release": False,
            "new_payload_sparsity_assumption": False,
        },
        "aggregate": {
            "tokens": aggregate["tokens"],
            "events": aggregate["events"],
            "m212_cycles": exact_m212,
            "m214_opportunity_cycles": opportunity,
            "cycles_saved": exact_m212 - opportunity,
            "speed_vs_m212": fraction(exact_m212, opportunity),
            "controller_cache": {
                "hits": cached_simulate.cache_info().hits,
                "misses": cached_simulate.cache_info().misses,
                "maxsize": cached_simulate.cache_info().maxsize,
                "current_size": cached_simulate.cache_info().currsize,
            },
        },
        "per_stage": per_stage_result,
        "claim_boundary": {
            "exact_frozen_payload_identity": True,
            "m212_base_vcs_calibrated": True,
            "m214_rule_vcs_calibrated": False,
            "m214_rtl_exists": False,
            "complete_fc2": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    require(aggregate["tokens"] == 5_580_000, "token drift")
    require(aggregate["events"] == 143_894_510, "event drift")
    require(opportunity <= exact_m212, "opportunity regression")
    require(sha256(Path(__file__).resolve()) == script_start,
            "analyzer mutated during run")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
