#!/usr/bin/env python3
"""M946 D1/D2/D3 bounded-prefix decoder source candidate.

This additive source candidate extends the frozen M896 selector from one D0
row to one explicitly selected D1/D2/D3 row.  It does not change M785's
transaction generator, resource, addresses, transaction order, scheduling
recurrence or cycle-class priority.  It refuses full-row, production, result
publication, EDA, GPU and remote modes and accepts only 1K/10K/100K prefixes.

D1 is always a common-charged full-shape nonheadline diagnostic.  D2 and D3
are exact-binary support rows.  No output of this file is decoder-complete or
paper-citable until independently hammered and separately released.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M896_PATH = HERE / "analyze_m896_decoder_run_gtls_source_candidate.py"
M896_SHA256 = "c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39"
M785_CONTRACT = HW / "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json"
M785_CONTRACT_SHA256 = "612a2ba39ceecedc351f2f6550347ad50ca9526fd89ed143bc6362c3e5681810"
M785_SOURCE_SHA256 = "7fbd72d27e4733179d1d3037080c69ebc9e6ceb0aa5716cc497d3dfee81070f1"
M942_DIR = HW / "reviews/m942_decoder_d1_exactness_and_four_layer_slice_first_principles_r1_20260829"
M942_IDENTITY = (
    "7f4a290abe4df5de2a98b29bdcb9aadc138fc0c028f85b84ac904a16acfe412f",
    "181e94050c25122b8a1b3463dc28a0ab5eda84ac4d6b836639ab113ea3576fc5",
    "49698663cf41edc26a46ed7e2c6bc311c8ecc2bf7ada44e2d0eb0ef817b9f21b",
)
M939_DIR = HW / "reviews/m939_m925_m896_decoder_gtls_full_first_row_result_hammer_r1_20260829"
M939_IDENTITY = (
    "477b3e97c67985dea77abae4bcb89c22ccc287ac71da8e6a92ea89d098721871",
    "18bab0c0de545be55553c9e28478526e9dcf5095eaf34ddc2319ed4adfe01461",
    "18bf220605b6a919f041e15ee3a64c1224517b5ee23a5fd7860d6ae6f38d2612",
)
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CONTRACT_SCHEMA = "m946_decoder_multilayer_bounded_prefix_source_candidate_v1"
PYTHON_PATH = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
PYTHON_VERSION = (3, 10, 18)
ALLOWED_PREFIXES = (1000, 10000, 100000)
ALLOWED_LAYERS = ("D1", "D2", "D3")
MODULE_BY_LAYER = {"D1": 1, "D2": 2, "D3": 3}
ROUTE_BY_LAYER = {
    "D1": "COMMON_CHARGED_FULL_SHAPE_DIAGNOSTIC_NONHEADLINE",
    "D2": "EXACT_BINARY_SUPPORT",
    "D3": "EXACT_BINARY_SUPPORT",
}
TARGET_REQUEST_PROXY = {
    "D1": 16688570,
    "D2": 151879626,
    "D3": 504012937,
}
TIMEOUT_CAP_SECONDS = 6 * 60 * 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen(path: Path, expected: str, name: str):
    if sha256(path) != expected:
        raise RuntimeError(name + " identity drift")
    spec = importlib.util.spec_from_file_location("m946_" + name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_interpreter() -> Dict[str, object]:
    executable = Path(sys.executable).resolve()
    if executable != PYTHON_PATH or sha256(executable) != PYTHON_SHA256:
        raise RuntimeError("M946 requires the exact frozen M925 Python interpreter")
    if tuple(sys.version_info[:3]) != PYTHON_VERSION:
        raise RuntimeError("M946 Python version identity drift")
    return {
        "path": str(executable),
        "sha256": PYTHON_SHA256,
        "version": list(PYTHON_VERSION),
    }


_validate_interpreter()
M896 = _load_frozen(M896_PATH, M896_SHA256, "frozen_m896")
M890 = M896.M890
M785 = M896.M785
Failure = M896.Failure
require = M896.require
CompressedTransaction = M896.CompressedTransaction


def _verify_sealed(directory: Path, identity: Tuple[str, str, str],
                   name: str) -> Dict[str, object]:
    sealed = M785.verify_sealed_directory(directory)
    require(sha256(directory / "review.json") == identity[0],
            name + " review identity drift")
    require(sealed["manifest_sha256"] == identity[1],
            name + " manifest identity drift")
    require(sealed["outer_seal_file_sha256"] == identity[2],
            name + " outer seal identity drift")
    return sealed


def validate_frozen_authority() -> Dict[str, object]:
    interpreter = _validate_interpreter()
    require(sha256(M785_CONTRACT) == M785_CONTRACT_SHA256,
            "M785 contract identity drift")
    require(sha256(HW / "system_simulator/scripts/analyze_m785_h67_decoder_physical_residency_repair.py") ==
            M785_SOURCE_SHA256, "M785 source identity drift")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    contract = M785.strict_json(M785_CONTRACT)
    entry = contract["inputs"]["primary_m686"]
    payload_root = HW / entry["directory"]
    sealed = M785.verify_sealed_directory(payload_root)
    require(sha256(payload_root / "manifest.json") == entry["manifest_sha256"],
            "M686 manifest identity drift")
    require(sealed["outer_seal_file_sha256"] ==
            entry["outer_seal_file_sha256"], "M686 outer seal drift")
    _verify_sealed(M942_DIR, M942_IDENTITY, "M942")
    _verify_sealed(M939_DIR, M939_IDENTITY, "M939")
    return {
        "status": "PASS_M946_FROZEN_AUTHORITY",
        "interpreter": interpreter,
        "m896_source_sha256": sha256(M896_PATH),
        "m785_contract_sha256": sha256(M785_CONTRACT),
        "m686_manifest_sha256": sha256(payload_root / "manifest.json"),
        "m942_review_sha256": sha256(M942_DIR / "review.json"),
        "m939_review_sha256": sha256(M939_DIR / "review.json"),
        "docs359_sha256": DOCS359_SHA256,
    }


def _context():
    contract = M785.strict_json(M785_CONTRACT)
    entry = contract["inputs"]["primary_m686"]
    payload_root = HW / entry["directory"]
    manifest = M785.strict_json(payload_root / "manifest.json")
    population_id = "M686_ZURICH_CITY_09_A_S10"
    records = M785.normalized_population_records(manifest, population_id)
    mapper_row = contract["inputs"]["m672_mapper"]
    mapper = M785.load_pinned_module(HW / mapper_row["path"],
                                     mapper_row["sha256"], "m946_mapper")
    m712, m722, storage = (contract["inputs"][name] for name in
                           ("m712_oracle", "m722r2_oracle",
                            "m785_storage_oracle"))
    oracles = M785.load_pinned_oracles(
        HW / m712["path"], m712["sha256"],
        HW / m722["path"], m722["sha256"],
        HW / storage["path"], storage["sha256"])
    return contract, payload_root, population_id, records, mapper, oracles


def select_record(records: Sequence[Mapping[str, object]], layer: str,
                  sample_id: int) -> Mapping[str, object]:
    require(layer in ALLOWED_LAYERS, "M946 refuses D0/unknown layer")
    require(0 <= int(sample_id) < 10, "sample_id is outside sealed S10")
    module_index = MODULE_BY_LAYER[layer]
    matches = [row for row in records
               if int(row["module_index"]) == module_index and
               int(row["sample_id"]) == int(sample_id)]
    require(len(matches) == 1, "selected decoder row is not unique")
    row = matches[0]
    shape = tuple(int(value) for value in row["input_shape"])
    require(shape[0] == 10, "selected row is not T10")
    return row


def row_identity(record: Mapping[str, object], layer: str, sample_id: int,
                 config: str, timestep: int) -> Dict[str, object]:
    return {
        "population_id": "M686_ZURICH_CITY_09_A_S10",
        "layer": layer,
        "module_index": int(record["module_index"]),
        "sample_id": int(sample_id),
        "config": config,
        "timestep": int(timestep),
        "input_shape": [int(value) for value in record["input_shape"]],
        "payload_relative_path": str(record["relative_path"]),
        "payload_sha256": str(record["packed_sha256"]),
        "numerical_route": ROUTE_BY_LAYER[layer],
        "headline_eligible": False,
        "decoder_complete": False,
    }


def prefix_transactions(layer: str, sample_id: int, config: str,
                        timestep: int, limit: int) -> Tuple[
                            List[CompressedTransaction], Dict[str, object]]:
    require(int(limit) in ALLOWED_PREFIXES,
            "only 1K/10K/100K bounded prefixes are allowed")
    require(config in M785.CONFIGS, "configuration is not frozen by M785")
    require(0 <= int(timestep) < 10, "timestep is outside frozen T10")
    _, payload_root, population_id, records, mapper, oracles = _context()
    record = select_record(records, layer, int(sample_id))
    stream = M785.iter_record_transactions(
        mapper, record, payload_root, population_id, config,
        int(timestep), oracles)
    transactions = M890.truncate_transactions(stream, int(limit))
    require(sum(int(tx.count) for tx in transactions) == int(limit),
            "bounded prefix request conservation failure")
    require(all(tx.population_id == population_id and tx.config == config
                for tx in transactions), "row isolation drift")
    return transactions, row_identity(record, layer, int(sample_id), config,
                                      int(timestep))


EXACT_FIELDS = (
    "total_cycles", "expanded_request_count",
    "compressed_transaction_count", "scheduled_requests",
    "compressed_schedule", "transaction_address_sha256",
    "commit_sequence_sha256", "population_ids", "configs",
    "cycle_classes", "same_cycle_response_slot_reuse",
    "terminal_readiness", "terminal_readiness_sha256", "port_calendars",
)


def exact_schedule(transactions: Sequence[CompressedTransaction],
                   row: Mapping[str, object], include_old: bool
                   ) -> Dict[str, object]:
    resource_model = M785.resource_from_contract(M785.strict_json(M785_CONTRACT))
    shard = (str(row["population_id"]), str(row["config"]),
             int(row["module_index"]), int(row["sample_id"]),
             int(row["timestep"]))
    new = M896.RUNGTLSScheduler(resource_model).schedule(
        M896.RunGroupIR(transactions, shard), retain_details=True,
        retain_expanded_address_sha=True)
    reference = M890.GTLSScheduler(resource_model).schedule(
        M890.PackedGroupIR(transactions, shard), retain_details=True,
        retain_expanded_address_sha=True)
    for field in EXACT_FIELDS:
        require(new[field] == reference[field],
                "M890/M896 exact miter mismatch: " + field)
    old_status = "NOT_RUN_ABOVE_10K_BOUND"
    if include_old:
        require(sum(int(tx.count) for tx in transactions) <= 10000,
                "old exact reference is limited to 10K")
        prior = M890.exact_miter(transactions, include_old=True)
        require(prior["terminal_readiness_sha256"] ==
                new["terminal_readiness_sha256"],
                "M768/M861/M890/M896 terminal exact miter mismatch")
        old_status = "PASS_M768_M861_M890_M896_EXACT_MITER"
    commit_requests = sum(int(tx.count) for tx in transactions
                          if tx.kind == "commit")
    cycle_identity = {
        "total_cycles": int(new["total_cycles"]),
        "cycle_classes": new["cycle_classes"],
        "same_cycle_response_slot_reuse":
            bool(new["same_cycle_response_slot_reuse"]),
    }
    return {
        "status": ("PASS_M768_M861_M890_M896_EXACT_MITER" if include_old
                   else "PASS_M890_M896_EXACT_MITER"),
        "old_reference_status": old_status,
        "expanded_request_count": int(new["expanded_request_count"]),
        "compressed_transaction_count":
            int(new["compressed_transaction_count"]),
        "total_cycles_diagnostic_only": int(new["total_cycles"]),
        "six_cycle_classes": new["cycle_classes"],
        "compressed_transaction_order_sha256":
            str(new["compressed_group_ir_sha256"]),
        "transaction_address_sha256":
            str(new["transaction_address_sha256"]),
        "commit_sequence_sha256": str(new["commit_sequence_sha256"]),
        "commit_requests_in_prefix": commit_requests,
        "cycle_identity_sha256": M785.canonical_sha256(cycle_identity),
        "terminal_readiness_sha256":
            str(new["terminal_readiness_sha256"]),
        "combined_live_event_state_bytes":
            int(new["combined_live_event_state_bytes"]),
        "event_resident_state_bytes":
            int(new["event_resident_state_bytes"]),
        "compact_control_state_bytes":
            int(new["compact_control_state_bytes"]),
        "liveness_resident_state_bytes":
            int(new["liveness_resident_state_bytes"]),
        "live_token_peak": int(new["live_token_peak"]),
        "event_run_counts": new["event_run_counts"],
        "exact_fields": list(EXACT_FIELDS),
    }


def _meminfo() -> Dict[str, int]:
    rows: Dict[str, int] = {}
    with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            token = value.strip().split()[0]
            rows[key] = int(token) * 1024
    require(all(key in rows for key in
                ("MemAvailable", "CommitLimit", "Committed_AS")),
            "required /proc/meminfo fields missing")
    return rows


def projection(layer: str, prefix: int, elapsed_seconds: float,
               max_rss_kib: int, scheduler_state_bytes: int
               ) -> Dict[str, object]:
    target = TARGET_REQUEST_PROXY[layer]
    projected_state = math.ceil(scheduler_state_bytes * target / prefix)
    measured_peak = int(max_rss_kib) * 1024
    nonscheduler = max(0, measured_peak - int(scheduler_state_bytes))
    projected_peak = nonscheduler + projected_state
    two_x_memory = 2 * projected_peak
    projected_elapsed = float(elapsed_seconds) * target / prefix
    timeout = math.ceil(2.0 * projected_elapsed)
    mem = _meminfo()
    commit_headroom = max(0, mem["CommitLimit"] - mem["Committed_AS"])
    memory_pass = (two_x_memory <= mem["MemAvailable"] and
                   two_x_memory <= commit_headroom)
    timeout_pass = timeout <= TIMEOUT_CAP_SECONDS
    prefix_is_gate = int(prefix) == 100000
    return {
        "target_full_row_request_proxy": target,
        "target_is_exact_closed_form": layer == "D1",
        "target_is_sizing_proxy_only": layer in ("D2", "D3"),
        "measured_prefix_requests": int(prefix),
        "projected_scheduler_state_bytes": projected_state,
        "measured_process_peak_bytes": measured_peak,
        "nonscheduler_peak_bytes": nonscheduler,
        "projected_peak_bytes": projected_peak,
        "two_x_projected_memory_bytes": two_x_memory,
        "mem_available_bytes": mem["MemAvailable"],
        "commit_headroom_bytes": commit_headroom,
        "two_x_memory_gate_pass": memory_pass,
        "projected_elapsed_seconds": projected_elapsed,
        "two_x_timeout_seconds": timeout,
        "timeout_cap_seconds": TIMEOUT_CAP_SECONDS,
        "two_x_timeout_gate_pass": timeout_pass,
        "prefix_is_authoritative_100k_gate": prefix_is_gate,
        "future_full_row_scalability_preflight_pass":
            bool(prefix_is_gate and memory_pass and timeout_pass),
        "full_row_authorized": False,
    }


def run_bounded_prefix(layer: str, sample_id: int, config: str,
                       timestep: int, prefix: int) -> Dict[str, object]:
    validate_frozen_authority()
    started = time.monotonic()
    transactions, row = prefix_transactions(
        layer, sample_id, config, timestep, prefix)
    exact = exact_schedule(transactions, row, include_old=prefix <= 10000)
    elapsed = time.monotonic() - started
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    scaling = projection(layer, prefix, elapsed, max_rss,
                         int(exact["combined_live_event_state_bytes"]))
    return {
        "schema": "m946_decoder_multilayer_bounded_prefix_result_v1",
        "status": "PASS_M946_BOUNDED_PREFIX_EXACT_PREFLIGHT__NO_FULL_ROW",
        "row_identity": row,
        "row_identity_sha256": M785.canonical_sha256(row),
        "prefix": int(prefix),
        "exact_miter": exact,
        "elapsed_seconds": elapsed,
        "process_max_rss_kib": max_rss,
        "scalability_projection": scaling,
        "claim_boundary": {
            "source_candidate_only": True,
            "bounded_prefix_only": True,
            "prefix_cycle_diagnostic_only": True,
            "paper_citable": False,
            "production_row": False,
            "full_row_authorized": False,
            "decoder_complete": False,
            "table_a_row": False,
            "system_speedup": False,
            "eda_gpu_remote_used": False,
        },
    }


def validate_source_candidate(contract_path: Path) -> Dict[str, object]:
    validate_frozen_authority()
    contract = M785.strict_json(Path(contract_path))
    require(contract.get("schema") == CONTRACT_SCHEMA,
            "M946 contract schema drift")
    require(contract.get("status") ==
            "DRAFT_SOURCE_ONLY__INDEPENDENT_FRESH_HAMMER_REQUIRED",
            "M946 draft status drift")
    auth = contract["authorization"]
    require(auth.get("bounded_prefix_up_to_100k") is True and
            auth.get("full_row") is False and
            auth.get("production") is False and
            auth.get("eda_gpu_remote") is False,
            "M946 authorization is not fail closed")
    boundary = contract["claim_boundary"]
    require(all(boundary.get(name) is False for name in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "full_row_authorized")),
            "M946 claim boundary is not fail closed")
    for name, row in contract["source_identity"].items():
        path = HW / row["path"]
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == row["sha256"],
                "source identity drift: " + name)
    return {
        "status": "PASS_M946_DRAFT_SOURCE_IDENTITY__NO_EXECUTION_RELEASE",
        "contract_sha256": sha256(Path(contract_path)),
        "full_row_authorized": False,
        "paper_citable": False,
    }


def source_self_test() -> Dict[str, object]:
    authority = validate_frozen_authority()
    _, _, _, records, _, _ = _context()
    selected = {}
    for layer in ALLOWED_LAYERS:
        row = select_record(records, layer, 0)
        selected[layer] = {
            "module_index": int(row["module_index"]),
            "sample_id": int(row["sample_id"]),
            "route": ROUTE_BY_LAYER[layer],
        }
    synthetic = M896.exact_miter(M890.synthetic_transactions(1000),
                                 include_old=True)
    return {
        "status": "PASS_M946_SOURCE_SELF_TEST__NO_FULL_ROW",
        "authority": authority,
        "selected_rows": selected,
        "synthetic_1k": synthetic,
        "full_row_authorized": False,
        "paper_citable": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-source-candidate", action="store_true")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--bounded-prefix", type=int)
    parser.add_argument("--layer", choices=ALLOWED_LAYERS)
    parser.add_argument("--sample-id", type=int)
    parser.add_argument("--config", choices=tuple(M785.CONFIGS))
    parser.add_argument("--timestep", type=int)
    parser.add_argument("--run-full-row", action="store_true")
    parser.add_argument("--run-production", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    require(not args.run_full_row and not args.run_production,
            "M946 source candidate refuses full-row/production")
    require(args.output is None,
            "M946 source candidate refuses result publication")
    if args.self_test:
        print(json.dumps(source_self_test(), sort_keys=True, allow_nan=False))
        return 0
    if args.validate_source_candidate:
        require(args.contract is not None, "contract is required")
        print(json.dumps(validate_source_candidate(args.contract),
                         sort_keys=True, allow_nan=False))
        return 0
    if args.bounded_prefix is not None:
        require(args.layer is not None and args.sample_id is not None and
                args.config is not None and args.timestep is not None,
                "bounded prefix requires layer/sample/config/timestep")
        print(json.dumps(run_bounded_prefix(
            args.layer, args.sample_id, args.config, args.timestep,
            args.bounded_prefix), sort_keys=True, allow_nan=False))
        return 0
    raise Failure("only bounded source validation/test modes are authorized")


if __name__ == "__main__":
    raise SystemExit(main())
