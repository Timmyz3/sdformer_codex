#!/usr/bin/env python3
"""Audit whether the H67 ATLIF module needs persistent inference state SRAM.

This is a boundary/completeness audit, not a cycle, energy, or PPA experiment.
It intentionally distinguishes tile-local RTL working storage from memory that
must survive a context release or be indexed by spatial position/frame.
"""

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Set, Type


EXPECTED = {
    "algorithm": (
        "neuron_experiments/H9_bipolar_self_attention/overlay/models/"
        "STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py",
        "d9ee7e172f941a53ad1c031b0d5cdbbf7819f521c807e5bc54001a80c41b57f3",
    ),
    "rtl": (
        "hw_autoresearch_nts07/rtl_m273/m273_integrated_rank3_atlif.sv",
        "11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d",
    ),
    "cycle_model": (
        "hw_autoresearch_nts07/results/"
        "m265_atlif_matched_boundary_trace_cycle_r1_20260825/"
        "m265_atlif_matched_boundary_trace_cycle_r1.json",
        "7fa5d46a4676241012d8a7afb32d9d36729170658caf9380171fbeeb9ae4b31d",
    ),
    "macro_audit": (
        "hw_autoresearch_nts07/reviews/tsmc28_sram_macro_audit_r1_20260827/"
        "tsmc28_sram_mapping_r1.json",
        "68017fb51773713dd7dbee9463ec60d1dcdac9dea6e56588463e7f4ded96be4d",
    ),
    "trace": (
        "hw_autoresearch_nts07/results/"
        "h67_ep35_full_network_ordered_trace_s10_20260821/execution_trace.csv",
        "ad8d1f286c0936ce7cf42324068cfd074aeef3cf77af62890e0598b663b91bfd",
    ),
    "docs359": (
        "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md",
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    ),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def self_attributes(node: ast.AST, context: Type[ast.expr_context]) -> Set[str]:
    result = set()  # type: Set[str]
    for item in ast.walk(node):
        if (
            isinstance(item, ast.Attribute)
            and isinstance(item.value, ast.Name)
            and item.value.id == "self"
            and isinstance(item.ctx, context)
        ):
            result.add(item.attr)
    return result


def assigned_names(node: ast.AST) -> Set[str]:
    result = set()  # type: Set[str]
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
            result.add(item.id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()

    identities = {}  # type: Dict[str, Dict[str, str]]
    for name, (relative, expected_sha) in EXPECTED.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"bad input: {relative}")
        actual = digest(path)
        require(actual == expected_sha, f"SHA drift: {relative}: {actual}")
        identities[name] = {"path": relative, "sha256": actual}

    algorithm_path = root / EXPECTED["algorithm"][0]
    algorithm_text = algorithm_path.read_text()
    tree = ast.parse(algorithm_text)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ATLIFTernaryPSN"
    )
    forward = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    # The forward body contains a nested gradient-hook function that returns
    # ``grad``.  Only the direct forward return defines the inference output.
    returns = [node for node in forward.body if isinstance(node, ast.Return)]
    require(len(returns) == 1, "forward must have one return")
    require(
        isinstance(returns[0].value, ast.Name) and returns[0].value.id == "out",
        "forward return identity drift",
    )
    loaded_attrs = self_attributes(forward, ast.Load)
    written_attrs = self_attributes(forward, ast.Store)
    inference_observer_writes = {
        "update_value",
        "r",
        "pos_r",
        "neg_r",
        "positive_trigger_r",
        "negative_trigger_r",
        "act_value",
        "quantile_value",
        "_quantile_initialized",
        "importance_last",
        "importance_ema",
        "_importance_initialized",
    }
    require(written_attrs <= inference_observer_writes, "unexpected forward state write")
    output_variables = {"flattened", "latent", "h_seq", "negative_scale", "spike", "out"}
    output_path_attrs = set()  # type: Set[str]
    for statement in ast.walk(forward):
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            if assigned_names(statement) & output_variables:
                output_path_attrs |= self_attributes(statement, ast.Load)
    # Branch selectors controlling which output-path assignment executes are
    # frozen module configuration, so include them explicitly in the audit.
    output_path_attrs |= {
        "temporal_factor_rank",
        "center_mode",
        "output_mode",
        "threshold_mode",
    }
    output_path_dynamic_state_reads = written_attrs & output_path_attrs
    require(
        not output_path_dynamic_state_reads,
        f"mutable observer state is read by forward: {output_path_dynamic_state_reads}",
    )
    for required_fragment in (
        "flattened = x_seq.flatten(1)",
        "torch.mm(self.temporal_factor_right, flattened)",
        "torch.addmm(self.bias, self.temporal_factor_left, latent)",
        "torch.addmm(self.bias, self.weight, flattened)",
        "out = spike.view(x_seq.shape)",
    ):
        require(required_fragment in algorithm_text, f"algorithm fragment drift: {required_fragment}")

    rtl_text = (root / EXPECTED["rtl"][0]).read_text()
    for required_fragment in (
        "raw_bank0_q",
        "raw_bank1_q",
        "stage1_acc_q",
        "inter_bank0_q",
        "inter_bank1_q",
        "fifo_tag_q",
        "release_ready",
        "&&work_empty&&!raw_valid",
        "if(release_accept)config_loaded_q<=1'b0",
    ):
        require(required_fragment in rtl_text, f"RTL boundary fragment drift: {required_fragment}")
    forbidden_external_state_ports = (
        "previous_membrane",
        "membrane_state_input",
        "membrane_state_output",
        "previous_frame_state",
    )
    for name in forbidden_external_state_ports:
        require(name not in rtl_text, f"external state port appeared: {name}")

    with (root / EXPECTED["trace"][0]).open(newline="") as handle:
        trace = list(csv.DictReader(handle))
    atlif = [row for row in trace if row["kind"] == "atlif"]
    steps = Counter(int(row["temporal_steps"]) for row in atlif)
    samples = Counter(int(row["sample_id"]) for row in atlif)
    require(len(trace) == 1840, "ordered trace record count drift")
    require(len(atlif) == 930, "ATLIF record count drift")
    require(steps == {2: 480, 10: 450}, f"temporal cohort drift: {steps}")
    require(len(samples) == 10 and set(samples.values()) == {93}, "sample balance drift")

    cycle_model = json.loads((root / EXPECTED["cycle_model"][0]).read_text())
    common = cycle_model["matched_boundary"]["common"]
    require(common["contexts_must_drain_before_release"], "context drain contract missing")
    require(not common["cross_context_execution_forbidden"] is False, "cross-context drift")
    require(common["raw_input_banks"] == 2, "raw bank count drift")
    require(common["result_fifo_depth_beats"] == 16, "FIFO depth drift")

    macro = json.loads((root / EXPECTED["macro_audit"][0]).read_text())
    mapping = next(item for item in macro["mappings"] if item["id"] == "C3_M273_ATLIF_WORKING_STATE")
    require(mapping["mapping"] == "current stdcell registers", "C3 mapping drift")
    require(mapping["evidence"] == "STDCELL", "C3 evidence class drift")

    payload_bits = {
        "configuration": 1536,
        "two_raw_banks": 2 * 1280,
        "stage1_accumulators": 48 * 24,
        "two_intermediate_banks": 2 * 384,
        "product_register": 147,
        "result_fifo_payload": 16 * 147,
    }
    payload_total = sum(payload_bits.values())
    require(payload_total == 8515, "working payload arithmetic drift")

    result = {
        "schema": "m515_atlif_state_boundary_audit_v1",
        "status": "PASS_TILE_LOCAL_STATE_ONLY__NO_PERSISTENT_INFERENCE_STATE_SRAM_AT_MODULE_BOUNDARY",
        "identity": identities,
        "algorithm_audit": {
            "forward_return": "out",
            "input_dependency": "current x_seq plus frozen parameters/buffers",
            "mutable_forward_attributes": sorted(written_attrs),
            "output_path_self_attributes": sorted(output_path_attrs),
            "mutable_observer_state_on_output_dependency_path": sorted(output_path_dynamic_state_reads),
            "cross_tile_or_cross_frame_recurrent_state": False,
            "note": "Training/calibration counters and EMA observers are not inference output state.",
        },
        "trace_population": {
            "records": len(trace),
            "atlif_records": len(atlif),
            "t10_records": steps[10],
            "t2_records": steps[2],
            "samples": len(samples),
            "atlif_records_per_sample": sorted(set(samples.values())),
        },
        "rtl_working_state": {
            "payload_breakdown_bits": payload_bits,
            "payload_total_bits_excluding_tags_order_and_control": payload_total,
            "payload_total_bytes_ceiling": (payload_total + 7) // 8,
            "mapping": "synthesized standard-cell registers",
            "context_release_requires_complete_drain": True,
            "state_survives_release": False,
        },
        "admission": {
            "standalone_c3_external_persistent_state_macro_required": False,
            "working_registers_already_charged_in_m289_logic_area": True,
            "full_system_weight_config_memory_closed": False,
            "same_boundary_fixed_rtl_closed": False,
            "matched_saif_ptpx_closed": False,
            "trained_rank3_accuracy_closed": False,
            "cycle_speedup": False,
            "energy": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
        "paper_safe_statement": (
            "The ATLIF tile consumes a complete temporal input tile and drains all raw, "
            "intermediate, product, and FIFO state before release; therefore its standalone "
            "inference boundary has no spatially indexed or cross-frame persistent membrane SRAM. "
            "Its 8,515-bit payload working set (plus tags/control) is already synthesized as "
            "standard-cell state. System weight/config memory remains outside this claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
