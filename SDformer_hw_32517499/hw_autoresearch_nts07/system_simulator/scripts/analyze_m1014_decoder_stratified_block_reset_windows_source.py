#!/usr/bin/env python3
"""M1014 source-only block-reset decoder window implementation.

This module implements the M1009 measurement protocol but deliberately does
not expose a production or real-payload execution mode.  Its executable CLI
is limited to frozen-source validation and a small synthetic exact-miter
self-test.  An independent hammer/release must authorize real D0/D2/D3
windows later.
"""

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
CONTRACT = HW / "contracts/m1014_decoder_stratified_block_reset_windows_source_contract_r1_20260829.json"
PLAN = HW / "contracts/m1009_decoder_stratified_window_source_plan_contract_r1_20260829.json"
M946_PATH = HERE / "analyze_m946_decoder_multilayer_bounded_prefix_source_candidate.py"
ESTIMATOR_PATH = HERE / "check_m1009_decoder_stratified_window_source_plan.py"

FROZEN = {
    "m785": (HERE / "analyze_m785_h67_decoder_physical_residency_repair.py",
             "7fbd72d27e4733179d1d3037080c69ebc9e6ceb0aa5716cc497d3dfee81070f1"),
    "m890": (HERE / "analyze_m890_decoder_gtls_source_candidate.py",
             "cacc118ea33616ae4284403ad69656bbeacaa7bc83d227c0d9b5a86c2ead459e"),
    "m896": (HERE / "analyze_m896_decoder_run_gtls_source_candidate.py",
             "c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39"),
    "m946": (M946_PATH,
             "0ffd1ee810f24d1a95b0df33ffe8eae43240920e12a2fccb86c947d2be51b6ac"),
    "m1009_plan": (PLAN,
             "d107c7e7ef1a8a0971a1bd4882e0ff7f46140787c7ad7afa2842366f6e5b6999"),
    "m1009_estimator": (ESTIMATOR_PATH,
             "6819d948b270572cbd5bf6cb88948ddecf1436ef96f14af7a92dcbe5812b925a"),
    "docs359": (HW / "docs/359_DATE终局冻结_20260813.md",
             "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"),
}
SCHEMA = "m1014_decoder_stratified_block_reset_windows_source_v1"
ALLOWED_LAYERS = ("D0", "D2", "D3")
MODULE_BY_LAYER = {"D0": 0, "D2": 2, "D3": 3}
STRATA = ("SOURCE_INIT_CENSUS", "COMPUTE_REGULAR",
          "DEPENDENCY_STRESS", "COMMIT_TAIL")
PILOT_PER_STRATUM = 8
MAX_PER_STRATUM = 32
WINDOW_EXPANDED_REQUEST_CAP = 10000
SELECTION_SEED = "M1009_STRATIFIED_WINDOW_R1_20260829"


def require(value, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()


def load_pinned(path: Path, expected: str, name: str):
    require(path.is_file() and not path.is_symlink(), name + " source absent")
    require(sha256(path) == expected, name + " identity drift")
    spec = importlib.util.spec_from_file_location("m1014_" + name, path)
    require(spec is not None and spec.loader is not None,
            "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Importing M946 also validates its exact frozen Python and transitive sources.
M946 = load_pinned(*FROZEN["m946"], "m946")
M896, M890, M785 = M946.M896, M946.M890, M946.M785
ESTIMATOR = load_pinned(*FROZEN["m1009_estimator"], "m1009_estimator")
CompressedTransaction = M896.CompressedTransaction


@dataclass(frozen=True)
class WindowSpec:
    """Cycle-blind identity frozen before a window is scheduled."""

    block_id: str
    layer: str
    stratum: str
    population_blocks: int
    sample_id: int = 0
    timestep: int = 0

    def validate(self) -> None:
        require(self.block_id, "empty block id")
        require(self.layer in ALLOWED_LAYERS, "D1 is strict common charge")
        require(self.stratum in STRATA, "unknown stratum")
        require(self.population_blocks >= 1, "invalid stratum population")
        require(self.sample_id == 0 and self.timestep == 0,
                "frozen sample/timestep drift")

    @property
    def identity_sha256(self) -> str:
        self.validate()
        return canonical_sha(asdict(self))


def reject_d1(layer: str) -> None:
    require(layer != "D1",
            "D1_STRICT_COMMON_CHARGE_NO_GENERATOR_OR_SCHEDULER_CALL")
    require(layer in ALLOWED_LAYERS, "unsupported decoder layer")


def frozen_route(layer: str) -> Dict[str, object]:
    """Return identity only; this function never opens a real payload."""
    reject_d1(layer)
    return {
        "layer": layer,
        "module_index": MODULE_BY_LAYER[layer],
        "parser": ("M896_D0_FROZEN_EXACT_BINARY" if layer == "D0"
                   else "M946_{}_FROZEN_EXACT_BINARY".format(layer)),
        "sample_id": 0,
        "timestep": 0,
        "real_payload_opened": False,
    }


def classify_stratum(metadata: Mapping[str, object]) -> str:
    """Priority classification using only pre-cycle block metadata."""
    if bool(metadata.get("source_init")):
        return "SOURCE_INIT_CENSUS"
    if int(metadata.get("commit_count", 0)) > 0:
        return "COMMIT_TAIL"
    if (int(metadata.get("psum_external_move_count", 0)) > 0 or
            int(metadata.get("weight_refill_external_count", 0)) > 0 or
            int(metadata.get("max_dependency_fan_in", 0)) >= 3):
        return "DEPENDENCY_STRESS"
    require(int(metadata.get("compute_count", 0)) > 0,
            "regular block requires positive compute metadata")
    return "COMPUTE_REGULAR"


def deterministic_select(index: Sequence[Mapping[str, object]], stratum: str,
                         requested: int) -> List[Mapping[str, object]]:
    """Deterministic SRSWOR surrogate selected before any cycle field exists."""
    require(stratum in STRATA, "unknown selection stratum")
    requested = int(requested)
    limit = 1 if stratum == "SOURCE_INIT_CENSUS" else MAX_PER_STRATUM
    require(1 <= requested <= limit, "selection count exceeds frozen bound")
    candidates = []
    identities = set()
    forbidden = {"cycles", "candidate_cycles", "baseline_cycles", "speedup"}
    for row in index:
        require(not forbidden.intersection(row),
                "cycle-derived field present before selection")
        identity = str(row["block_id"])
        require(identity not in identities, "duplicate block identity")
        identities.add(identity)
        if classify_stratum(row) == stratum:
            key = canonical_sha([SELECTION_SEED, identity])
            candidates.append((key, identity, row))
    require(candidates, "empty stratum")
    count = min(requested, len(candidates))
    return [row for _, _, row in sorted(candidates)[:count]]


def _reset_tx(tx: CompressedTransaction, transaction_id: str,
              dependencies: Tuple[str, ...], produces: str
              ) -> CompressedTransaction:
    # Compute markers use the same population/config and the frozen compute
    # port.  They are charged identically, never hidden as zero-cost metadata.
    return CompressedTransaction(
        transaction_id=transaction_id,
        population_id=tx.population_id,
        config=tx.config,
        kind="compute",
        base_address=1 << 60,
        address_stride_bytes=1,
        count=1,
        bank_pattern=(0,),
        width_bytes=1,
        dependency_tokens=dependencies,
        produces_token_prefix=produces,
        earliest_issue_cycle=0,
    )


def block_reset_transactions(body: Sequence[CompressedTransaction],
                             spec: WindowSpec, side: str
                             ) -> Tuple[List[CompressedTransaction], Dict[str, object]]:
    """Wrap a dependency-closed work block with charged reset/fill/drain."""
    spec.validate()
    require(side in ("candidate", "baseline"), "invalid paired side")
    require(body, "empty block body")
    body_requests = sum(int(tx.count) for tx in body)
    require(body_requests + 3 <= WINDOW_EXPANDED_REQUEST_CAP,
            "window exceeds 10K including reset overhead")
    first = body[0]
    require(all(tx.population_id == first.population_id and
                tx.config == first.config for tx in body),
            "mixed population/config block")
    namespace = "m1014:{}:{}:{}".format(spec.identity_sha256[:16], side,
                                        spec.layer.lower())
    boundary = _reset_tx(first, namespace + ":boundary", (),
                         namespace + ":boundary_ready")
    boundary_token = M890.terminal_token(boundary)
    fill = _reset_tx(first, namespace + ":fill", (boundary_token,),
                     namespace + ":fill_ready")
    fill_token = M890.terminal_token(fill)
    produced = set()
    rewritten: List[CompressedTransaction] = []
    external_dependency_count = 0
    for ordinal, tx in enumerate(body):
        mapped = []
        for dependency in tx.dependency_tokens:
            if dependency in produced:
                mapped.append(dependency)
            else:
                mapped.append(fill_token)
                external_dependency_count += 1
        if not mapped and ordinal == 0:
            mapped.append(fill_token)
        rewritten_tx = replace(tx, dependency_tokens=tuple(dict.fromkeys(mapped)))
        rewritten.append(rewritten_tx)
        terminal = M890.terminal_token(rewritten_tx)
        if terminal:
            produced.add(terminal)
    last_token = next((M890.terminal_token(tx) for tx in reversed(rewritten)
                       if M890.terminal_token(tx)), fill_token)
    drain = _reset_tx(first, namespace + ":drain", (last_token,),
                      namespace + ":drained")
    output = [boundary, fill] + rewritten + [drain]
    require(sum(int(tx.count) for tx in output) == body_requests + 3,
            "block-reset request conservation failure")
    return output, {
        "window_identity_sha256": spec.identity_sha256,
        "original_transaction_id_census_sha256": canonical_sha(
            [tx.transaction_id for tx in body]),
        "reset_transaction_id_census_sha256": canonical_sha(
            [boundary.transaction_id, fill.transaction_id,
             drain.transaction_id]),
        "body_expanded_request_count": body_requests,
        "reset_expanded_request_count": 3,
        "external_dependency_remap_count": external_dependency_count,
        "boundary_ready_token_sha256": canonical_sha(boundary_token),
    }


def _resource_for(transactions: Sequence[CompressedTransaction]):
    if transactions[0].population_id == "M890_SYNTHETIC":
        return M896.M861._synthetic_resource()
    contract = M785.strict_json(
        HW / "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json")
    return M785.resource_from_contract(contract)


def exact_replay(transactions: Sequence[CompressedTransaction],
                 spec: WindowSpec) -> Dict[str, object]:
    """Fresh-scheduler M896/M890 exact replay for one additive block."""
    spec.validate()
    expanded = sum(int(tx.count) for tx in transactions)
    require(1 <= expanded <= WINDOW_EXPANDED_REQUEST_CAP,
            "window expanded-request cap failure")
    resource = _resource_for(transactions)
    shard = (transactions[0].population_id, transactions[0].config,
             MODULE_BY_LAYER[spec.layer], spec.sample_id, spec.timestep)
    new = M896.RUNGTLSScheduler(resource).schedule(
        M896.RunGroupIR(transactions, shard), retain_details=True,
        retain_expanded_address_sha=True)
    reference = M890.GTLSScheduler(resource).schedule(
        M890.PackedGroupIR(transactions, shard), retain_details=True,
        retain_expanded_address_sha=True)
    for field in M946.EXACT_FIELDS:
        require(new[field] == reference[field],
                "M890/M896 exact miter mismatch: " + field)
    prior = M890.exact_miter(transactions, include_old=True)
    require(prior["terminal_readiness_sha256"] ==
            new["terminal_readiness_sha256"],
            "M768/M861/M890/M896 terminal exact miter mismatch")
    commit_count = sum(int(tx.count) for tx in transactions
                       if tx.kind == "commit")
    if spec.stratum == "COMMIT_TAIL":
        require(commit_count > 0, "commit stratum has zero commit requests")
    total = int(new["total_cycles"])
    require(sum(int(value) for value in new["cycle_classes"].values()) == total,
            "cycle-class conservation failure")
    require(int(new["live_token_final"]) == 0,
            "terminal liveness did not drain")
    outstanding_after_total = []
    for row in new["port_calendars"]["outstanding_returns"]:
        outstanding_after_total.extend(
            value for value in row[2] if int(value) >= total)
    require(not outstanding_after_total, "outstanding return after block end")
    return {
        "status": "PASS_M768_M861_M890_M896_BLOCK_RESET_EXACT_MITER",
        "window_identity_sha256": spec.identity_sha256,
        "total_cycles": total,
        "expanded_request_count": int(new["expanded_request_count"]),
        "compressed_transaction_count": int(new["compressed_transaction_count"]),
        "commit_request_count": commit_count,
        "cycle_classes": new["cycle_classes"],
        "transaction_address_sha256": new["transaction_address_sha256"],
        "commit_sequence_sha256": new["commit_sequence_sha256"],
        "terminal_readiness_sha256": new["terminal_readiness_sha256"],
        "port_calendars_sha256": canonical_sha(new["port_calendars"]),
        "live_token_final_zero": True,
        "outstanding_return_final_zero": True,
        "cycle_class_sum_equals_total": True,
        "exact_fields": list(M946.EXACT_FIELDS),
    }


def paired_replay(candidate_body: Sequence[CompressedTransaction],
                  baseline_body: Sequence[CompressedTransaction],
                  spec: WindowSpec) -> Dict[str, object]:
    """Replay paired bodies under identical window identity/reset charges."""
    candidate, cmeta = block_reset_transactions(candidate_body, spec, "candidate")
    baseline, bmeta = block_reset_transactions(baseline_body, spec, "baseline")
    cresult = exact_replay(candidate, spec)
    bresult = exact_replay(baseline, spec)
    require(cmeta["reset_expanded_request_count"] ==
            bmeta["reset_expanded_request_count"] == 3,
            "paired reset charge drift")
    return {
        "window_identity_sha256": spec.identity_sha256,
        "stratum": spec.stratum,
        "candidate_cycles": cresult["total_cycles"],
        "baseline_cycles": bresult["total_cycles"],
        "candidate": cresult,
        "baseline": bresult,
        "candidate_reset": cmeta,
        "baseline_reset": bmeta,
        "transaction_ratio_is_speedup": False,
    }


def estimate_paired_totals(strata: Sequence[Mapping[str, object]],
                           fixed_candidate: float = 0.0,
                           fixed_baseline: float = 0.0) -> Dict[str, object]:
    return ESTIMATOR.estimate_paired_totals(
        strata, fixed_candidate=fixed_candidate,
        fixed_baseline=fixed_baseline)


def validate_source(contract_path: Path = CONTRACT) -> Dict[str, object]:
    contract = M785.strict_json(contract_path)
    require(contract["schema"] == SCHEMA and
            contract["status"] == "SOURCE_ONLY__NO_REAL_WINDOW_EXECUTION" and
            contract["launch_now"] is False, "M1014 contract drift")
    for name, (path, expected) in FROZEN.items():
        require(path.is_file() and sha256(path) == expected,
                "frozen identity drift: " + name)
    require(contract["layers"] == {
        "included_exact": ["D0", "D2", "D3"],
        "D1": "STRICT_COMMON_CHARGE_NO_WINDOW"}, "layer policy drift")
    require(contract["sampling"] == {
        "strata": list(STRATA), "pilot_per_noncensus_stratum": 8,
        "adaptive_max_per_noncensus_stratum": 32,
        "window_expanded_request_cap": 10000,
        "selection_after_index_before_cycles": True}, "sampling drift")
    require(all(contract["claim_boundary"][key] is False for key in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "real_window_execution_authorized",
                 "eda_gpu_remote_used")), "claim boundary expanded")
    return {
        "status": "PASS_M1014_SOURCE_VALIDATION__NO_REAL_EXECUTION",
        "contract_sha256": sha256(contract_path),
        "launch_now": False,
        "real_payload_opened": False,
        "window_execution": False,
        "eda_gpu_remote_used": False,
    }


def self_test() -> Dict[str, object]:
    # Small synthetic only: no M785 payload root, full row, GPU, remote, or EDA.
    index = [
        {"block_id": "source", "source_init": True},
        {"block_id": "regular-a", "compute_count": 2},
        {"block_id": "regular-b", "compute_count": 1},
        {"block_id": "stress", "compute_count": 1,
         "max_dependency_fan_in": 3},
        {"block_id": "commit", "commit_count": 1},
    ]
    require(classify_stratum(index[0]) == "SOURCE_INIT_CENSUS" and
            classify_stratum(index[3]) == "DEPENDENCY_STRESS" and
            classify_stratum(index[4]) == "COMMIT_TAIL",
            "stratum priority self-test failure")
    first = deterministic_select(index, "COMPUTE_REGULAR", PILOT_PER_STRATUM)
    second = deterministic_select(index, "COMPUTE_REGULAR", PILOT_PER_STRATUM)
    require([row["block_id"] for row in first] ==
            [row["block_id"] for row in second], "selection nondeterminism")
    d1_rejected = False
    try:
        frozen_route("D1")
    except RuntimeError as error:
        d1_rejected = "STRICT_COMMON_CHARGE" in str(error)
    require(d1_rejected, "D1 rejection self-test failure")

    # Seven synthetic runs cover every transaction kind, including commit.
    body = M890.synthetic_transactions(448)
    spec = WindowSpec("synthetic-commit", "D0", "COMMIT_TAIL", 1)
    paired = paired_replay(body, body, spec)
    require(paired["candidate_cycles"] == paired["baseline_cycles"] and
            paired["candidate"]["commit_request_count"] > 0,
            "paired synthetic replay drift")
    estimate = estimate_paired_totals([
        {"stratum": "COMMIT_TAIL", "population_blocks": 2,
         "candidate_cycles": [paired["candidate_cycles"]] * 2,
         "baseline_cycles": [paired["baseline_cycles"]] * 2},
    ])
    require(estimate["paired_speedup_estimate"] == 1.0 and
            estimate["paired_speedup_ci95"] == [1.0, 1.0],
            "paired FPC/covariance self-test failure")
    return {
        "status": "PASS_M1014_SMALL_SYNTHETIC_BLOCK_RESET_SELFTEST",
        "strata": list(STRATA),
        "pilot_per_noncensus_stratum": PILOT_PER_STRATUM,
        "adaptive_max_per_noncensus_stratum": MAX_PER_STRATUM,
        "window_expanded_request_cap": WINDOW_EXPANDED_REQUEST_CAP,
        "d1_strict_common_charge_rejected": True,
        "paired_replay": paired,
        "paired_estimator": estimate,
        "real_payload_opened": False,
        "real_window_execution": False,
        "launch_now": False,
        "eda_gpu_remote_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-source", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    require(args.validate_source != args.self_test,
            "select exactly one source-only mode")
    result = validate_source() if args.validate_source else self_test()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
