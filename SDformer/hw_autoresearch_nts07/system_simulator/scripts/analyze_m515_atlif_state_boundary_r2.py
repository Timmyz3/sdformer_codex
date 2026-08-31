#!/usr/bin/env python3
"""Fail-closed M515r2 audit of the standalone M273 T10 ATLIF state boundary.

This is a static completeness/accounting audit.  It does not measure cycles,
power, accuracy, or system performance.  The algorithm result is explicitly
conditional on a separately pinned frozen-inference deployment contract.
"""

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, MutableMapping, Optional, Set, Tuple


EXPECTED = {
    "deployment_manifest": (
        "hw_autoresearch_nts07/contracts/"
        "m515_atlif_frozen_inference_manifest_r2_20260827.json",
        "7b5bd132567821a2e1690b1544e8d5e9b6303f54dec9ca9a9e1fad617c3691f1",
    ),
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
    "trace": (
        "hw_autoresearch_nts07/results/"
        "h67_ep35_full_network_ordered_trace_s10_20260821/execution_trace.csv",
        "ad8d1f286c0936ce7cf42324068cfd074aeef3cf77af62890e0598b663b91bfd",
    ),
    "m289_contract": (
        "hw_autoresearch_nts07/contracts/"
        "m289_m273r2_protocol_repaired_logic_only_dc_contract_r1_20260825.json",
        "07efe93cbbd2e6998944f9f8c96422e8a489a3627c4d2f1f861c97a9d0675710",
    ),
    "m289_input_identity": (
        "hw_autoresearch_nts07/dc_handoff/runs/"
        "m289_m273r2_protocol_repaired_logic_only_dc_3p000ns_r1_20260825/"
        "input_sha256.txt",
        "2cf634143ef33d6d1ca8b5fcb86858ad45f4f364ff14f620126fd5932de621c2",
    ),
    "m289_evidence_manifest": (
        "hw_autoresearch_nts07/dc_handoff/runs/"
        "m289_m273r2_protocol_repaired_logic_only_dc_3p000ns_r1_20260825/"
        "evidence_manifest.sha256",
        "acbc2063dd3d48730d4c9b970e4f0fdb03d6bf75dea042b4496042dc99d8541f",
    ),
    "m289_evidence_seal": (
        "hw_autoresearch_nts07/dc_handoff/runs/"
        "m289_m273r2_protocol_repaired_logic_only_dc_3p000ns_r1_20260825/"
        "evidence_manifest.seal.sha256",
        "1e71e1dde81f17b93a4c748459370ca022b7d3e9bd94c66e4d62e55d7494f2da",
    ),
    "m289_area": (
        "hw_autoresearch_nts07/dc_handoff/runs/"
        "m289_m273r2_protocol_repaired_logic_only_dc_3p000ns_r1_20260825/"
        "reports/area.rpt",
        "88edb12472a4f6712f25626802fc353d24eb298aab874d95ab96968daf25e825",
    ),
    "m289_netlist": (
        "hw_autoresearch_nts07/dc_handoff/runs/"
        "m289_m273r2_protocol_repaired_logic_only_dc_3p000ns_r1_20260825/"
        "netlist/m273_integrated_rank3_atlif_mapped.v",
        "c4f35a836038388b7d55fe4e810369abda2bfd57da8e0882db6ed99bbc755ddc",
    ),
    "m302_contract": (
        "hw_autoresearch_nts07/contracts/"
        "m302_m289_m273r2_logic_only_dc_independent_hammer_contract_r1_20260825.json",
        "e8ca72d3006197fe9b668c1996dce2ff78b1c10491679edc780603b76982fa4a",
    ),
    "m302_review": (
        "hw_autoresearch_nts07/results/"
        "m302_m289_m273r2_logic_only_dc_independent_hammer_r1_20260825/"
        "m302_m289_independent_hammer_review_r1.json",
        "aed54de63ddabbd669bb2bf57207c160352e3df10d50f4fb1bc9cb84541ba4b7",
    ),
    "m302_manifest": (
        "hw_autoresearch_nts07/results/"
        "m302_m289_m273r2_logic_only_dc_independent_hammer_r1_20260825/"
        "SHA256SUMS",
        "1ec6110e9a349e6425e43c99deb075a3cbf959f72b8a7253ec0735c7649f6996",
    ),
    "m302_seal": (
        "hw_autoresearch_nts07/results/"
        "m302_m289_m273r2_logic_only_dc_independent_hammer_r1_20260825/"
        "SHA256SUMS.seal.sha256",
        "8f8a03f7ed7d6faace7afdf2e2507ce4b4ca23f02d6e4b660516ac436fe4b277",
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


def self_attribute(node: ast.AST) -> Optional[str]:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def target_names(node: ast.AST) -> Iterable[str]:
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            yield from target_names(element)


def dependencies(node: Optional[ast.AST], env: MutableMapping[str, Set[str]]) -> Set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load):
            return set(env.get(node.id, {"local:" + node.id}))
        return set()
    attr = self_attribute(node)
    if attr is not None and isinstance(node.ctx, ast.Load):
        return {"self:" + attr}
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "self"
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        extra = set()
        for item in node.args[2:]:
            extra |= dependencies(item, env)
        return {"dynamic-self:" + node.args[1].value} | extra
    result = set()
    for child in ast.iter_child_nodes(node):
        result |= dependencies(child, env)
    return result


def assigned_self_attributes(node: ast.AST) -> Set[str]:
    result = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Attribute) and isinstance(item.ctx, (ast.Store, ast.Del)):
            attr = self_attribute(item)
            if attr is not None:
                result.add(attr)
    return result


def analyze_statements(
    statements: Iterable[ast.stmt],
    env: MutableMapping[str, Set[str]],
    control: Set[str],
) -> Tuple[MutableMapping[str, Set[str]], Set[str]]:
    returns = set()
    for statement in statements:
        if isinstance(statement, ast.Assign):
            value_deps = dependencies(statement.value, env) | control
            for target in statement.targets:
                for name in target_names(target):
                    env[name] = set(value_deps)
        elif isinstance(statement, ast.AnnAssign):
            value_deps = dependencies(statement.value, env) | control
            for name in target_names(statement.target):
                env[name] = set(value_deps)
        elif isinstance(statement, ast.AugAssign):
            value_deps = dependencies(statement.target, env) | dependencies(statement.value, env) | control
            for name in target_names(statement.target):
                env[name] = set(value_deps)
        elif isinstance(statement, ast.If):
            predicate = dependencies(statement.test, env) | control
            left_env, left_return = analyze_statements(statement.body, dict(env), predicate)
            right_env, right_return = analyze_statements(statement.orelse, dict(env), predicate)
            for name in set(left_env) | set(right_env):
                env[name] = set(left_env.get(name, set())) | set(right_env.get(name, set()))
            returns |= left_return | right_return
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            env, nested_return = analyze_statements(statement.body, env, control)
            returns |= nested_return
        elif isinstance(statement, (ast.For, ast.While)):
            predicate = dependencies(getattr(statement, "test", None), env) | dependencies(
                getattr(statement, "iter", None), env
            ) | control
            body_env, body_return = analyze_statements(statement.body, dict(env), predicate)
            for name in set(env) | set(body_env):
                env[name] = set(env.get(name, set())) | set(body_env.get(name, set()))
            returns |= body_return
        elif isinstance(statement, ast.Return):
            returns |= dependencies(statement.value, env) | control
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # A nested callback is a value.  Its body is not executed at definition.
            env[statement.name] = {"nested-callable:" + statement.name}
    return env, returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()

    identities: Dict[str, Dict[str, str]] = {}
    for name, (relative, expected_sha) in EXPECTED.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"bad input: {relative}")
        actual = digest(path)
        require(expected_sha != "TO_BE_FILLED", "analyzer deployment-manifest SHA not frozen")
        require(actual == expected_sha, f"SHA drift: {relative}: {actual}")
        identities[name] = {"path": relative, "sha256": actual}

    deployment = json.loads((root / EXPECTED["deployment_manifest"][0]).read_text())
    require(
        deployment["status"] == "CONDITIONAL_DEPLOYMENT_CONTRACT__NOT_RUNTIME_MEASUREMENT",
        "deployment status drift",
    )
    required_conditions = {
        "h9_calibration_observer_absent": True,
        "threshold_update_disabled": True,
        "optimizer_updates_disabled": True,
        "autograd_disabled": True,
        "parameters_frozen": True,
        "buffers_frozen": True,
        "complete_temporal_tile_per_forward": True,
    }
    require(
        deployment["required_inference_conditions"] == required_conditions,
        "frozen-inference conditions drift",
    )

    algorithm_path = root / EXPECTED["algorithm"][0]
    algorithm_text = algorithm_path.read_text()
    tree = ast.parse(algorithm_text)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ATLIFTernaryPSN"
    )
    forward = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    env: MutableMapping[str, Set[str]] = {"x_seq": {"input:x_seq"}}
    _, output_dependencies = analyze_statements(forward.body, env, set())
    written_attrs = assigned_self_attributes(forward)
    dynamic_attrs = set()
    for item in ast.walk(forward):
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "getattr"
            and len(item.args) >= 2
            and isinstance(item.args[0], ast.Name)
            and item.args[0].id == "self"
            and isinstance(item.args[1], ast.Str)
        ):
            dynamic_attrs.add(item.args[1].s)
    require(dynamic_attrs == {"_h9_calibration_observer"}, f"dynamic callback drift: {dynamic_attrs}")
    output_self_attrs = {
        item.split(":", 1)[1] for item in output_dependencies if item.startswith("self:")
    }
    output_mutable_writes = output_self_attrs & written_attrs
    require(not output_mutable_writes, f"forward-written state reaches out: {output_mutable_writes}")
    require("input:x_seq" in output_dependencies, "out lost current x_seq dependency")
    require("self:thresh" in output_dependencies, "out lost frozen threshold dependency")
    require(
        "observer = getattr(self, \"_h9_calibration_observer\", None)" in algorithm_text,
        "observer guard source drift",
    )
    require("self.update_value += thre_updates" in algorithm_text, "observer-state update drift")
    require("out = spike.view(x_seq.shape)" in algorithm_text, "out construction drift")

    rtl_text = (root / EXPECTED["rtl"][0]).read_text()
    for fragment in (
        "work_empty=!fill_active_q&&raw_owned_q==0&&!stage1_active_q",
        "&&inter_reserved_q==0&&!stage2_active_q&&!product_valid_q",
        "&&fifo_count_q==0;",
        "release_ready=!rst_core&&config_loaded_q&&!protocol_error_q",
        "&&tiles_loaded_q!=0&&work_empty&&!raw_valid;",
        "if(stage1_selected_phase==0)base_value='0;",
        "if(release_accept)config_loaded_q<=1'b0",
    ):
        require(fragment in rtl_text, f"RTL liveness fragment drift: {fragment}")
    for name in (
        "previous_membrane", "membrane_state_input", "membrane_state_output",
        "previous_frame_state",
    ):
        require(name not in rtl_text, f"external state port appeared: {name}")

    declared_state_bits = {
        "config_frame": 1536,
        "config_control": 5,
        "decoded_config": 1109,
        "raw_banks": 2560,
        "raw_tags": 96,
        "raw_orders": 64,
        "raw_ownership_and_ready": 4,
        "raw_fill_control_and_tag": 53,
        "stage1_control": 6,
        "stage1_accumulators": 1152,
        "intermediate_banks": 768,
        "intermediate_tags": 96,
        "intermediate_orders": 64,
        "intermediate_reserved_and_valid": 4,
        "stage2_control": 5,
        "product_register_including_valid_tag_beat": 148,
        "fifo_entries_including_tag_beat": 2352,
        "fifo_pointers_and_count": 13,
        "debug_counters": 320,
        "context_retire_and_tile_done_state": 115,
    }
    declared_total = sum(declared_state_bits.values())
    require(declared_total == 10470, "declared-state accounting drift")

    netlist_text = (root / EXPECTED["m289_netlist"][0]).read_text()
    dff_counts = {
        "DFKCNQD1BWP35P140": len(re.findall(r"^\s*DFKCNQD1BWP35P140\s+", netlist_text, re.M)),
        "DFKCNQD2BWP35P140": len(re.findall(r"^\s*DFKCNQD2BWP35P140\s+", netlist_text, re.M)),
    }
    sequential_cells = sum(dff_counts.values())
    require(dff_counts == {"DFKCNQD1BWP35P140": 9638, "DFKCNQD2BWP35P140": 1}, "DFF population drift")
    require(sequential_cells == 9639, "sequential-cell total drift")

    area_text = (root / EXPECTED["m289_area"][0]).read_text()
    area_match = re.search(r"Total cell area:\s+([0-9.]+)", area_text)
    noncomb_match = re.search(r"Noncombinational area:\s+([0-9.]+)", area_text)
    require(area_match is not None and noncomb_match is not None, "M289 area parse failed")
    total_area = float(area_match.group(1))
    noncomb_area = float(noncomb_match.group(1))
    require(total_area == 102852.287739 and noncomb_area == 19432.224313, "M289 area drift")
    m302 = json.loads((root / EXPECTED["m302_review"][0]).read_text())
    require(m302["status"] == "PASS_LOGIC_ONLY_DC_EVIDENCE__PHYSICAL_AND_ALGORITHM_ADMISSION_OPEN", "M302 status drift")
    require(m302["dc_recompute"]["sequential_cells"] == sequential_cells, "M302 sequential-cell mismatch")
    require(m302["dc_recompute"]["macro_count"] == 0, "M302 macro-count drift")
    require(not m302["admission"]["paper_ppa_ready"], "M302 claim boundary drift")

    with (root / EXPECTED["trace"][0]).open(newline="") as handle:
        trace = list(csv.DictReader(handle))
    atlif = [row for row in trace if row["kind"] == "atlif"]
    steps = Counter(int(row["temporal_steps"]) for row in atlif)
    samples = Counter(int(row["sample_id"]) for row in atlif)
    require(len(trace) == 1840 and len(atlif) == 930, "trace population drift")
    require(steps == {2: 480, 10: 450}, f"temporal cohort drift: {steps}")
    require(len(samples) == 10 and set(samples.values()) == {93}, "sample balance drift")

    cycle_model = json.loads((root / EXPECTED["cycle_model"][0]).read_text())
    common = cycle_model["matched_boundary"]["common"]
    require(common["contexts_must_drain_before_release"], "context drain contract missing")
    require(common["cross_context_execution_forbidden"], "cross-context rule drift")
    require(common["raw_input_banks"] == 2 and common["result_fifo_depth_beats"] == 16, "boundary dimensions drift")

    result = {
        "schema": "m515_atlif_state_boundary_audit_v2",
        "status": "PASS_CONDITIONAL_FROZEN_INFERENCE__M273_T10_STANDALONE_LIVE_STATE_BOUNDARY_CLOSED",
        "identity": identities,
        "conditional_algorithm_audit": {
            "deployment_contract_is_runtime_measurement": False,
            "required_inference_conditions": required_conditions,
            "recursive_output_dependencies": sorted(output_dependencies),
            "output_path_self_attributes": sorted(output_self_attrs),
            "forward_written_attributes": sorted(written_attrs),
            "forward_written_state_on_output_dependency_path": sorted(output_mutable_writes),
            "dynamic_callback_names_in_forward": sorted(dynamic_attrs),
            "dynamic_callback_absence_required_by_contract": True,
            "intrinsic_cross_tile_or_cross_frame_recurrent_state": False,
        },
        "trace_population": {
            "records": len(trace), "atlif_records": len(atlif),
            "t10_records": steps[10], "t2_records": steps[2],
            "samples": len(samples), "atlif_records_per_sample": sorted(set(samples.values())),
            "t2_rtl_covered_by_this_boundary_audit": False,
        },
        "rtl_live_state_boundary": {
            "scope": "M273 T10 standalone engine",
            "external_persistent_membrane_state_port": False,
            "release_requires_all_live_ownership_pipeline_fifo_state_drained": True,
            "live_or_valid_tile_state_survives_release": False,
            "physical_stale_bits_are_cleared_on_release": False,
            "stale_bits_unobservable_and_overwritten_before_valid_reuse": True,
        },
        "state_accounting": {
            "rtl_declared_sequential_bits_preoptimization_upper_bound": declared_total,
            "rtl_declared_breakdown_bits": declared_state_bits,
            "m289_synthesized_one_bit_sequential_cells": sequential_cells,
            "m289_dff_reference_counts": dff_counts,
            "declared_bits_and_synthesized_cells_are_different_metrics": True,
            "m289_logic_only_total_cell_area_um2": total_area,
            "m289_logic_only_noncombinational_area_um2": noncomb_area,
            "m289_macro_count": 0,
            "ideal_clock_zero_wireload": True,
        },
        "admission": {
            "standalone_m273_t10_persistent_membrane_state_sram_required_under_contract": False,
            "working_registers_charged_in_exact_source_m289_logic_area": True,
            "runtime_instance_compliance_measured": False,
            "t2_rtl_state_closed": False,
            "network_weight_config_backing_closed": False,
            "activation_storage_closed": False,
            "same_boundary_fixed_rtl_closed": False,
            "trained_rank3_accuracy_closed": False,
            "matched_saif_ptpx_closed": False,
            "cycle_speedup": False,
            "energy": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
        "paper_safe_statement": (
            "With calibration hooks absent, autograd and training-time threshold/optimizer updates "
            "disabled, and parameters and buffers frozen, the pinned ATLIF forward maps each complete "
            "temporal tile independently. The exact M273 T10 RTL accepts context release only after all "
            "live ownership, pipeline, and result-FIFO state has drained; stale physical bits remain "
            "unobservable. Thus this standalone boundary requires no spatially indexed or cross-frame "
            "membrane-state SRAM. The exact-source M289 logic-only netlist contains 9,639 one-bit "
            "sequential standard-cell cells and zero macros; network-wide weight/config and activation "
            "storage, T2 RTL, Fixed comparison, trained rank-3 accuracy, power, energy, and system "
            "performance remain outside this claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
