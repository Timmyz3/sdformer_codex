#!/usr/bin/env python3
"""Replay frozen H67 FC2 with scope-matched M216 K1 and K8 frontends."""

import argparse
import functools
import gc
import hashlib
import importlib.util
import json
import multiprocessing
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


EXPECTED_MANIFEST = "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
EXPECTED_M172 = "ae633daee1e07f16396570c1ef757c11bac7f1f72be108f4411d75f6dcb5f6d9"
EXPECTED_M192 = "39715b64890d75be7c60587d639f3e75b51e5bf38642e6b8640c761fa512f24b"
EXPECTED_M213_ANALYZER = "8130123a32cafd6e8700f2f7d88bd2bc1807a06f5d0e447bf943974997d932f7"
EXPECTED_M214_RESULT = "f48104fcaf6f1f39737e280456aaeb9ec64b0669ca8eff7361767088e2195701"
EXPECTED_M216_MODEL = "1bed35da65287b48bcaee0e5181bcfae01c3dbc41ea1927d9eccbf94dfaf380b"
EXPECTED_M216_K1_VALIDATION = "9bbd26405bcdaf6916d7dd1e197d2bf35737cb221673d5fde5e22d9e4c3ddd7f"
EXPECTED_M216_K8_IDENTITY = "adbbe0cd774aa4fe6b7f264cab546169922f54c88034228f4feb513016afb4d6"
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
    """Fork-safe trampoline for the dynamically loaded frozen auditor."""
    return M213.audit_record(task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m192-analyzer", required=True, type=Path)
    parser.add_argument("--m213-analyzer", required=True, type=Path)
    parser.add_argument("--m214-result", required=True, type=Path)
    parser.add_argument("--m216-model", required=True, type=Path)
    parser.add_argument("--m216-k1-validation", required=True, type=Path)
    parser.add_argument("--m216-k8-identity", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    parser.add_argument("--cache-size", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_start = sha256(Path(__file__).resolve())
    require(sha256(args.manifest) == EXPECTED_MANIFEST, "manifest drift")
    require(sha256(args.m214_result) == EXPECTED_M214_RESULT,
            "M214 result drift")
    require(sha256(args.m216_k1_validation) == EXPECTED_M216_K1_VALIDATION,
            "M216 K1 validation drift")
    require(sha256(args.m216_k8_identity) == EXPECTED_M216_K8_IDENTITY,
            "M216 K8 identity drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359, "docs359 drift")

    k1_validation = json.loads(args.m216_k1_validation.read_text())
    require(k1_validation["status"] == "PASS_EXACT_256_CASE_VCS"
            and k1_validation["cases"] == 256
            and k1_validation["mismatches"] == 0
            and k1_validation["source_caps"] == [1],
            "K1 VCS recurrence not admitted")
    k8_identity = json.loads(args.m216_k8_identity.read_text())
    require(k8_identity["status"] == "PASS_EXACT_256_CASE_MODEL_IDENTITY"
            and k8_identity["cases"] == 256
            and k8_identity["mismatches"] == 0,
            "K8/M214 model identity not admitted")

    global M213
    M213 = load_module(args.m213_analyzer, EXPECTED_M213_ANALYZER,
                       "m213_pinned_m216")
    m172 = load_module(args.m172_analyzer, EXPECTED_M172,
                       "m172_pinned_m216")
    m192 = load_module(args.m192_analyzer, EXPECTED_M192,
                       "m192_pinned_m216")
    m216 = load_module(args.m216_model, EXPECTED_M216_MODEL,
                       "m216_scope_matched_model")
    M213.M172 = m172
    M213.M192 = m192
    M213.PAYLOAD_ROOT = args.payload_root
    M213.CHUNK_TOKENS = args.chunk_tokens

    manifest = json.loads(args.manifest.read_text())
    records = [record for record in manifest["records"]
               if record["operator"] == "Linear"
               and ".mlp.fc2" in record["name"]]
    require(len(records) == 120, "FC2 record extent drift")
    tasks = list(enumerate(records))

    def audit_cap(source_cap):
        raw_simulate = m216.simulate_m216_bank_loads

        @functools.lru_cache(maxsize=args.cache_size)
        def cached_simulate(load_key, depth, output_blocks):
            loads = [None if value is None else tuple(value)
                     for value in load_key]
            return raw_simulate(loads, depth, output_blocks,
                                source_cap=source_cap)

        def compatible_simulate(loads, depth, output_blocks):
            load_key = tuple(None if value is None else tuple(value)
                             for value in loads)
            return cached_simulate(load_key, depth, output_blocks)

        M213.M212 = SimpleNamespace(
            simulate_m212_bank_loads=compatible_simulate)
        if args.workers == 1:
            audited = []
            for count, task in enumerate(tasks, start=1):
                audited.append(run_task(task))
                print("[M216 K{}] {}/120".format(
                    source_cap, count), flush=True)
        else:
            context = multiprocessing.get_context("fork")
            with context.Pool(processes=args.workers) as pool:
                audited = []
                for count, item in enumerate(
                        pool.imap_unordered(run_task, tasks), start=1):
                    audited.append(item)
                    print("[M216 K{}] {}/120".format(
                        source_cap, count), flush=True)
        audited.sort(key=lambda item: item[0])
        aggregate = M213.empty_ledger()
        per_stage = defaultdict(M213.empty_ledger)
        for _ordinal, stage, ledger in audited:
            M213.merge(aggregate, ledger)
            M213.merge(per_stage[stage], ledger)
        # With fork workers the per-process caches are intentionally private;
        # the parent counters remain zero.  This is a performance diagnostic,
        # never part of the cycle result.
        cache_info = cached_simulate.cache_info()
        cache_receipt = {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "current_size": cache_info.currsize,
            "workers": args.workers,
            "per_worker_private_cache": args.workers != 1,
        }
        return aggregate, dict(per_stage), cache_receipt

    # Recompute the K8 denominator in this same script and payload traversal;
    # do not import the M214 number as the denominator by itself.
    k8_aggregate, k8_stage, k8_cache = audit_cap(8)
    gc.collect()
    k1_aggregate, k1_stage, k1_cache = audit_cap(1)

    m214_result = json.loads(args.m214_result.read_text())
    require(k8_aggregate["m212_rtl_semantic_cycles"]
            == m214_result["aggregate"]["m214_opportunity_cycles"],
            "same-script K8 denominator differs from M214 frozen replay")
    require(k8_aggregate["tokens"] == 5580000
            and k8_aggregate["events"] == 143894510,
            "K8 frozen identity drift")
    require(k1_aggregate["tokens"] == k8_aggregate["tokens"]
            and k1_aggregate["events"] == k8_aggregate["events"]
            and k1_aggregate["nonzero96_descriptors"]
                == k8_aggregate["nonzero96_descriptors"],
            "K1/K8 payload identity drift")

    k1_cycles = k1_aggregate["m212_rtl_semantic_cycles"]
    k8_cycles = k8_aggregate["m212_rtl_semantic_cycles"]
    weighted_event_cycles = sum(
        ledger["events"] * (1 << int(stage))
        for stage, ledger in k8_stage.items())
    require(weighted_event_cycles == 412900394,
            "output-block-weighted event identity drift")
    require(k1_cycles >= weighted_event_cycles,
            "K1 cycles below event-service lower bound")
    require(k8_cycles <= k1_cycles, "K8 regressed against K1")

    per_stage = {}
    for stage in sorted(k8_stage):
        k1_value = k1_stage[stage]["m212_rtl_semantic_cycles"]
        k8_value = k8_stage[stage]["m212_rtl_semantic_cycles"]
        require(k8_value
                == m214_result["per_stage"][str(stage)]
                    ["m214_opportunity_cycles"],
                "per-stage K8/M214 drift")
        per_stage[str(stage)] = {
            "tokens": k8_stage[stage]["tokens"],
            "events": k8_stage[stage]["events"],
            "output_blocks": 1 << int(stage),
            "weighted_event_cycles": k8_stage[stage]["events"]
                * (1 << int(stage)),
            "k1_cycles": k1_value,
            "k8_cycles": k8_value,
            "cycles_saved": k1_value - k8_value,
            "k8_speedup_vs_k1": fraction(k1_value, k8_value),
        }

    result = {
        "schema": "m216_h67_fc2_scope_matched_k1_k8_replay_v1",
        "status": "PASS_EXACT_FROZEN_H67_SCOPE_MATCHED_K1_K8_FRONTEND_REPLAY",
        "identity": {
            "analyzer_start_sha256": script_start,
            "manifest_sha256": EXPECTED_MANIFEST,
            "m172_analyzer_sha256": EXPECTED_M172,
            "m192_analyzer_sha256": EXPECTED_M192,
            "m213_analyzer_sha256": EXPECTED_M213_ANALYZER,
            "m214_result_sha256": EXPECTED_M214_RESULT,
            "m216_model_sha256": EXPECTED_M216_MODEL,
            "m216_k1_validation_sha256": EXPECTED_M216_K1_VALIDATION,
            "m216_k8_identity_sha256": EXPECTED_M216_K8_IDENTITY,
            "docs359_sha256": EXPECTED_DOCS359,
        },
        "matched_scope": {
            "raw_scan_width": 4,
            "descriptor_emit_width": 4,
            "descriptor_queue_depth": 8,
            "physical_window_buffers": 2,
            "descriptor_capacity_each": 8,
            "fixed_output_banks": 8,
            "group_interface_lanes": 8,
            "same_terminal_hint_done_and_handoff_control": True,
            "only_variable": "accepted sources per replay group: K1 versus K8",
        },
        "aggregate": {
            "records": k8_aggregate["records"],
            "tokens": k8_aggregate["tokens"],
            "events": k8_aggregate["events"],
            "raw96_beats": k8_aggregate["raw96_beats"],
            "nonzero96_descriptors": k8_aggregate[
                "nonzero96_descriptors"],
            "output_block_weighted_event_cycles": weighted_event_cycles,
            "k1_cycles": k1_cycles,
            "k8_cycles": k8_cycles,
            "cycles_saved": k1_cycles - k8_cycles,
            "k8_speedup_vs_k1": fraction(k1_cycles, k8_cycles),
            "k1_control_overhead_vs_weighted_events":
                k1_cycles - weighted_event_cycles,
            "k8_service_cycle_floor": 70657362,
            "k8_control_and_collision_overhead_vs_service_floor":
                k8_cycles - 70657362,
            "k1_controller_cache": k1_cache,
            "k8_controller_cache": k8_cache,
        },
        "per_stage": per_stage,
        "claim_boundary": {
            "exact_frozen_payload_identity": True,
            "same_script_numerator_and_denominator": True,
            "scope_matched_interface_control_and_storage": True,
            "k1_and_k8_rtl_exist": True,
            "synopsys_vcs_calibrated": True,
            "standalone_sparse_frontend_cycle_speedup": True,
            "complete_fc2": False,
            "complete_ffn": False,
            "weight_sram_response_latency": False,
            "accumulator_and_commit": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    require(sha256(Path(__file__).resolve()) == script_start,
            "analyzer mutated during run")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
