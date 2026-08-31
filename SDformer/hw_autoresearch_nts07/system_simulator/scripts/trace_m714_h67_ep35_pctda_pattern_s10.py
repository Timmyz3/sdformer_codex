#!/usr/bin/env python3
"""Capture exact H67 T10 bit-plane pattern coalescing statistics.

This is a narrow extension of the frozen M366 hook.  It does not retain full
tensors: for every 16-column tile it counts distinct non-zero 5-bit addresses
in the two temporal groups at each signed-INT8 bit plane.  Equal addresses may
share one distributed-arithmetic subset-vector read and broadcast its result
to a lane mask.  The capture reports opportunity and an ideal-resource issue
lower bound; it does not claim executable cycles, RTL, PPA, energy, or system
speedup.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M366_SCRIPT = HW / "system_simulator/scripts/trace_m366_h67_ep35_atlif_remaining_budget_s10.py"
DEFAULT_M366_CONTRACT = HW / "contracts/m366_h67_ep35_atlif_remaining_budget_s10_contract_r1_20260825.json"
PORTS = (1, 2, 4, 8)
LANES = 16
GROUP = 5
BITS = 8
FIXED_ISSUE_CYCLES = 17
MACRO_BITS = 128 * 128
MACRO_AREA_UM2 = 8758.360550
FIXED_PROVISIONAL_CELL_AREA_UM2 = 66778.235814
M714_SCHEMA = "m714_h67_ep35_pctda_pattern_s10_contract_v2"


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
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key {} in {}".format(
                key, path))
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON token {} in {}".format(
            token, path))

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def resolve_hw_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (HW / path).resolve()


def validate_m714_contract(contract_path):
    contract_path = Path(contract_path).resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") == M714_SCHEMA,
            "M714 contract schema drift")
    require(set(contract) == {
        "schema", "milestone", "objective", "identity", "runtime",
        "cycle_accounting", "admission", "claim_boundary"},
        "M714 contract top-level key drift")
    require(contract.get("milestone") == "M714-r2",
            "M714 milestone drift")
    identity = contract.get("identity")
    require(isinstance(identity, dict) and set(identity) == {
        "m714_script", "m366_script", "m366_contract",
        "m716_prerun_review", "protected_docs359"},
        "M714 identity key drift")
    observed = {}
    for key, item in identity.items():
        require(isinstance(item, dict) and set(item) == {"path", "sha256"},
                "bad M714 identity member {}".format(key))
        path = resolve_hw_path(item["path"])
        require(path.is_file() and not path.is_symlink(),
                "missing/non-regular M714 identity {}".format(path))
        actual = sha256(path)
        require(actual == item["sha256"],
                "M714 identity SHA drift {}".format(key))
        observed[key] = {"path": str(path), "sha256": actual}
    require(Path(observed["m714_script"]["path"]) == Path(__file__).resolve(),
            "M714 script path drift")
    require(Path(observed["m366_script"]["path"]) == M366_SCRIPT.resolve(),
            "M366 script path drift")
    require(Path(observed["m366_contract"]["path"]) ==
            DEFAULT_M366_CONTRACT.resolve(), "canonical M366 contract drift")
    runtime = contract.get("runtime")
    require(runtime == {
        "samples": 10,
        "installed_atlif_modules": 105,
        "live_sites": 81,
        "live_t10_sites": 45,
        "live_t2_sites": 36,
        "t10_calls": 450,
        "tile_lanes": 16,
        "gpu_launch_requires_four_consecutive_idle_checks": True,
        "gpu_launch_forbidden_while_any_training_eval_valid_or_profile_process_exists": True,
    }, "M714 runtime contract drift")
    return contract, observed


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pure_python_da_selftest():
    # Deterministic randomized algebra smoke test, including signed MSB.
    # This is deliberately not described as exhaustive: the real capture below
    # is a pattern-population measurement, while this local test only catches
    # signed-bitplane implementation errors before a GPU attempt is consumed.
    import random
    rng = random.Random(714)
    checks = 0
    for _ in range(256):
        weights = [[rng.randint(-128, 127) for _ in range(10)]
                   for _ in range(10)]
        values = [rng.randint(-128, 127) for _ in range(10)]
        dense = [sum(weights[o][t] * values[t] for t in range(10))
                 for o in range(10)]
        da = [0 for _ in range(10)]
        codes = [value & 0xff for value in values]
        for bit in range(8):
            sign = -128 if bit == 7 else (1 << bit)
            for begin in (0, 5):
                address = sum(((codes[begin + offset] >> bit) & 1) << offset
                              for offset in range(5))
                for output in range(10):
                    subset = sum(weights[output][begin + offset]
                                 for offset in range(5)
                                 if (address >> offset) & 1)
                    da[output] += sign * subset
        require(da == dense, "signed DA algebra mismatch")
        checks += 10
    return {
        "classification": "deterministic_randomized_algebra_smoke",
        "seed": 714,
        "vectors": 256,
        "output_checks": checks,
        "mismatches": 0,
        "exhaustive": False,
    }


def new_pattern_counter():
    return {
        "tiles": 0,
        "tile_bitplanes": 0,
        "nonzero_lane_group_addresses": 0,
        "distinct_nonzero_group_addresses": 0,
        "unique_histogram": [0 for _ in range(LANES + 1)],
        "coalesced_cycles": {str(port): 0 for port in PORTS},
        "uncoalesced_cycles": {str(port): 0 for port in PORTS},
        "coalesced_cycles_floor_one_per_bitplane": {
            str(port): 0 for port in PORTS},
    }


def add_pattern_counter(destination, source):
    for key in ("tiles", "tile_bitplanes", "nonzero_lane_group_addresses",
                "distinct_nonzero_group_addresses"):
        destination[key] += int(source[key])
    destination["unique_histogram"] = [a + int(b) for a, b in zip(
        destination["unique_histogram"], source["unique_histogram"])]
    for key in ("coalesced_cycles", "uncoalesced_cycles",
                "coalesced_cycles_floor_one_per_bitplane"):
        for port in PORTS:
            destination[key][str(port)] += int(source[key][str(port)])


def validate_pattern_counter(counter, label):
    tiles = int(counter["tiles"])
    tile_bitplanes = int(counter["tile_bitplanes"])
    histogram = [int(value) for value in counter["unique_histogram"]]
    distinct = int(counter["distinct_nonzero_group_addresses"])
    nonzero = int(counter["nonzero_lane_group_addresses"])
    require(tile_bitplanes == BITS * tiles,
            "{} tile-bitplane conservation".format(label))
    require(sum(histogram) == 2 * tile_bitplanes,
            "{} group histogram population".format(label))
    require(sum(index * value for index, value in enumerate(histogram)) ==
            distinct, "{} distinct-address histogram".format(label))
    require(0 <= distinct <= nonzero <= 2 * LANES * tile_bitplanes,
            "{} address bounds".format(label))
    for port in PORTS:
        key = str(port)
        coalesced = int(counter["coalesced_cycles"][key])
        uncoalesced = int(counter["uncoalesced_cycles"][key])
        floored = int(counter[
            "coalesced_cycles_floor_one_per_bitplane"][key])
        require(0 <= coalesced <= uncoalesced,
                "{} P{} coalescing monotonicity".format(label, port))
        require(floored >= coalesced,
                "{} P{} floor monotonicity".format(label, port))


def require_site_sum_equals_aggregate(aggregate, per_site):
    rebuilt = new_pattern_counter()
    for name, counter in sorted(per_site.items()):
        validate_pattern_counter(counter, name)
        add_pattern_counter(rebuilt, counter)
    require(rebuilt == aggregate,
            "per-site pattern counters do not equal aggregate")


def validate_m366_payload(payload, contract, observed):
    require(payload.get("schema") ==
            "m366_h67_ep35_atlif_remaining_budget_s10_capture_v1",
            "M366 payload schema drift")
    identity = payload.get("identity", {})
    require(identity.get("contract_path") ==
            observed["m366_contract"]["path"],
            "M366 payload contract path drift")
    require(identity.get("contract_sha256") ==
            observed["m366_contract"]["sha256"],
            "M366 payload contract SHA drift")
    require(identity.get("capture_script_sha256") ==
            observed["m366_script"]["sha256"],
            "M366 payload script SHA drift")
    population = payload.get("population", {})
    expected = contract["runtime"]
    require(population.get("samples") == expected["samples"],
            "M366 sample population drift")
    require(len(population.get("sample_keys", [])) == expected["samples"],
            "M366 sample-key population drift")
    require(population.get("installed_atlif_modules") ==
            expected["installed_atlif_modules"],
            "M366 installed ATLIF population drift")
    require(population.get("live_sites") == expected["live_sites"] and
            population.get("live_t10_sites") == expected["live_t10_sites"] and
            population.get("live_t2_sites") == expected["live_t2_sites"],
            "M366 live-site population drift")
    require(population.get("dead_called_sites") == [],
            "M366 dead site called")
    t10 = payload.get("t10_nonattention_main", {})
    require(t10.get("calls") == expected["t10_calls"],
            "M366 T10 call population drift")
    for key in ("signed_q8_range_violations", "input_nonfinite",
                "bound_violations", "integer_early_mismatches"):
        require(t10.get(key) == 0, "M366 numeric gate failed: {}".format(key))
    return True


def finalize(counter, calls):
    tiles = int(counter["tiles"])
    result = json.loads(json.dumps(counter))
    result["calls"] = int(calls)
    result["fixed_issue_cycles"] = tiles * FIXED_ISSUE_CYCLES
    result["fixed_warm_full_service_cycles"] = (
        tiles * FIXED_ISSUE_CYCLES + 12 * int(calls))
    # The sealed M518 17*N+12 accepted-start-to-retire denominator already
    # includes its five configuration beats.  Do not charge those five beats
    # a second time here.
    result["fixed_reference_service_cycles_17n_plus_12"] = (
        result["fixed_warm_full_service_cycles"])
    result["address_coalescing_reduction"] = (
        1.0 - float(counter["distinct_nonzero_group_addresses"]) /
        counter["nonzero_lane_group_addresses"]
        if counter["nonzero_lane_group_addresses"] else 0.0)
    result["points"] = []
    subset_width = 8 + int(math.ceil(math.log2(GROUP)))
    logical_table_bits = 2 * (1 << GROUP) * 10 * subset_width
    acc_bits = LANES * 10 * 25
    for port in PORTS:
        issue_lower_bound = int(counter[
            "coalesced_cycles_floor_one_per_bitplane"][str(port)])
        average = float(issue_lower_bound) / tiles if tiles else 0.0
        macro_count = port  # one 128x128 1RW replica per vector-read port
        macro_capacity_bits = macro_count * MACRO_BITS
        macro_area = macro_count * MACRO_AREA_UM2
        issue_lower_bound_ratio = (
            float(tiles * FIXED_ISSUE_CYCLES) / issue_lower_bound
            if issue_lower_bound else 0.0)
        same_framing_lower_bound_cycles = (
            issue_lower_bound + 12 * int(calls))
        same_framing_lower_bound_ratio = (
            float(result["fixed_warm_full_service_cycles"]) /
            same_framing_lower_bound_cycles
            if same_framing_lower_bound_cycles else 0.0)
        # Build-from-weights keeps the same five-beat M518 configuration and
        # adds 64 ideal table-write cycles.  Direct-table-load instead sends
        # 28 256-bit beats, i.e. 23 beats beyond M518's five, and does not
        # charge table construction.  These remain ideal-resource lower
        # bounds; neither is an executable schedule.
        build_from_weights_lower_bound_cycles = (
            same_framing_lower_bound_cycles + 64 * int(calls))
        direct_table_load_lower_bound_cycles = (
            same_framing_lower_bound_cycles + 23 * int(calls))
        build_from_weights_lower_bound_ratio = (
            float(result["fixed_warm_full_service_cycles"]) /
            build_from_weights_lower_bound_cycles
            if build_from_weights_lower_bound_cycles else 0.0)
        direct_table_load_lower_bound_ratio = (
            float(result["fixed_warm_full_service_cycles"]) /
            direct_table_load_lower_bound_cycles
            if direct_table_load_lower_bound_cycles else 0.0)
        optimistic_tp_area_upper = (
            issue_lower_bound_ratio * FIXED_PROVISIONAL_CELL_AREA_UM2 /
            macro_area
            if macro_area else 0.0)
        max_candidate_area = (
            FIXED_PROVISIONAL_CELL_AREA_UM2 * issue_lower_bound_ratio / 1.25
            if issue_lower_bound_ratio else 0.0)
        all_45_macro_count = port * int(math.ceil(45.0 / 2.0))
        result["points"].append({
            "vector_read_ports": port,
            "ideal_resource_issue_lower_bound_cycles": issue_lower_bound,
            "average_ideal_resource_issue_lower_bound_cycles_per_tile":
                average,
            "ideal_resource_issue_lower_bound_ratio_vs_fixed17":
                issue_lower_bound_ratio,
            "same_12_cycle_framing_lower_bound_cycles":
                same_framing_lower_bound_cycles,
            "same_12_cycle_framing_lower_bound_ratio_vs_fixed":
                same_framing_lower_bound_ratio,
            "build_from_weights_lower_bound_cycles_add_64_per_call":
                build_from_weights_lower_bound_cycles,
            "build_from_weights_lower_bound_ratio_vs_fixed":
                build_from_weights_lower_bound_ratio,
            "direct_table_load_lower_bound_cycles_add_23_beats_per_call":
                direct_table_load_lower_bound_cycles,
            "direct_table_load_lower_bound_ratio_vs_fixed":
                direct_table_load_lower_bound_ratio,
            "logical_table_bits_per_copy": logical_table_bits,
            "replicated_logical_table_bits": logical_table_bits * port,
            "macro_count_128x128_1rw": macro_count,
            "macro_capacity_bits": macro_capacity_bits,
            "accumulator_state_bits": acc_bits,
            "active_macro_plus_acc_state_bytes":
                (macro_capacity_bits + acc_bits) // 8,
            "macro_area_um2": macro_area,
            "all_45_exact_11bit_table_bytes_unreplicated":
                45 * logical_table_bits // 8,
            "all_45_exact_11bit_table_bytes_replicated":
                45 * logical_table_bits * port // 8,
            "all_45_resident_macro_count_128x128_1rw":
                all_45_macro_count,
            "all_45_resident_macro_capacity_bytes":
                all_45_macro_count * MACRO_BITS // 8,
            "all_45_resident_macro_area_um2":
                all_45_macro_count * MACRO_AREA_UM2,
            "active_table_build_cycles_per_call": 64,
            "build_from_weights_external_config_beats_256b_per_call": 5,
            "direct_table_external_config_beats_256b_per_call": 28,
            "direct_table_external_config_bytes_per_call":
                28 * 256 // 8,
            "active_table_internal_replica_write_bytes_per_call":
                logical_table_bits * port // 8,
            "provisional_fixed_cell_area_um2_not_admitted":
                FIXED_PROVISIONAL_CELL_AREA_UM2,
            "optimistic_throughput_per_area_upper_ignoring_candidate_logic":
                optimistic_tp_area_upper,
            "candidate_total_area_budget_for_1p25_tp_per_area_um2":
                max_candidate_area,
            "remaining_logic_area_budget_after_macros_um2":
                max_candidate_area - macro_area,
            "diagnostic_issue_lower_bound_le_10": average <= 10.0,
            "diagnostic_same_framing_lower_bound_ge_1p25":
                same_framing_lower_bound_ratio >= 1.25,
            "diagnostics_are_not_admission_gates": True,
            "active_state_diagnostic_le_24kib":
                (macro_capacity_bits + acc_bits) <= 24 * 1024 * 8,
            "all_45_resident_state_exceeds_active_state_and_is_separately_charged":
                True,
            "tp_per_area_not_proved_until_dc": True,
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "refusing to overwrite M714 output")

    contract, m714_observed = validate_m714_contract(args.contract)
    m366_contract = Path(
        m714_observed["m366_contract"]["path"]).resolve()
    m366 = load_module(M366_SCRIPT, "m714_frozen_m366")
    _, m366_observed = m366.validate_contract(m366_contract)
    original_class = m366.RemainingBudgetCapture

    class PatternCapture(original_class):
        latest = None

        def __init__(self, *capture_args, **capture_kwargs):
            super().__init__(*capture_args, **capture_kwargs)
            self.pattern_aggregate = new_pattern_counter()
            self.pattern_by_site = {}
            PatternCapture.latest = self

        def _capture_witnesses(self, name, sample_id, column_base, x_q,
                               resolve_k, full_hidden, fixed_event, float_event):
            if self.sites[name]["temporal_steps"] == 10:
                torch = self.torch
                local = new_pattern_counter()
                codes = x_q.to(torch.int16).bitwise_and(255)
                columns = int(codes.shape[1])
                require(int(column_base) % LANES == 0,
                        "M714 chunk begins inside a 16-lane tile")
                pad = (-columns) % LANES
                if pad:
                    require(int(column_base) + columns == int(full_hidden),
                            "M714 non-terminal chunk requires padding")
                    codes = torch.cat((codes, torch.zeros(
                        (10, pad), device=codes.device,
                        dtype=codes.dtype)), dim=1)
                tiles = int(codes.shape[1]) // LANES
                codes = codes.reshape(10, tiles, LANES)
                local["tiles"] = tiles
                local["tile_bitplanes"] = tiles * BITS
                for bit in range(BITS):
                    unique_counts = []
                    nonzero_counts = []
                    for begin in (0, 5):
                        pattern = torch.zeros(
                            (tiles, LANES), device=codes.device,
                            dtype=torch.int16)
                        for offset in range(GROUP):
                            pattern |= (((codes[begin + offset] >> bit) & 1)
                                        << offset)
                        ordered = torch.sort(pattern, dim=1)[0]
                        unique = ordered[:, 0].ne(0).to(torch.int32)
                        unique += ((ordered[:, 1:] != ordered[:, :-1]) &
                                   ordered[:, 1:].ne(0)).sum(
                                       dim=1, dtype=torch.int32)
                        nonzero = pattern.ne(0).sum(
                            dim=1, dtype=torch.int32)
                        unique_counts.append(unique)
                        nonzero_counts.append(nonzero)
                        hist = torch.bincount(
                            unique.to(torch.int64), minlength=LANES + 1)
                        hist_values = hist.detach().cpu().tolist()
                        local["unique_histogram"] = [a + int(b) for a, b in
                            zip(local["unique_histogram"], hist_values)]
                        local["distinct_nonzero_group_addresses"] += int(
                            unique.sum().item())
                        local["nonzero_lane_group_addresses"] += int(
                            nonzero.sum().item())
                    for port in PORTS:
                        coalesced = sum((value + port - 1) // port
                                        for value in unique_counts)
                        uncoalesced = sum((value + port - 1) // port
                                          for value in nonzero_counts)
                        local["coalesced_cycles"][str(port)] += int(
                            coalesced.sum().item())
                        local["uncoalesced_cycles"][str(port)] += int(
                            uncoalesced.sum().item())
                        local[
                            "coalesced_cycles_floor_one_per_bitplane"
                        ][str(port)] += int(torch.maximum(
                            torch.ones_like(coalesced), coalesced).sum().item())
                add_pattern_counter(self.pattern_aggregate, local)
                if name not in self.pattern_by_site:
                    self.pattern_by_site[name] = new_pattern_counter()
                add_pattern_counter(self.pattern_by_site[name], local)
            super()._capture_witnesses(
                name, sample_id, column_base, x_q, resolve_k, full_hidden,
                fixed_event, float_event)

    m366.RemainingBudgetCapture = PatternCapture
    m366.execute(m366_contract, output_dir)
    capture = PatternCapture.latest
    require(capture is not None, "M714 capture object missing")
    m366_path = output_dir / "m366_h67_ep35_atlif_remaining_budget_s10_capture.json"
    payload = strict_json(m366_path)
    validate_m366_payload(payload, contract, m714_observed)
    require(int(m366_observed["protected_docs359"]["sha256"] ==
                m714_observed["protected_docs359"]["sha256"]) == 1,
            "M714/M366 protected docs359 identity drift")
    require_site_sum_equals_aggregate(
        capture.pattern_aggregate, capture.pattern_by_site)
    validate_pattern_counter(capture.pattern_aggregate, "aggregate")
    require(len(capture.pattern_by_site) == contract["runtime"][
        "live_t10_sites"], "M714 T10 site population drift")
    total_calls = sum(int(capture.aggregate[name]["calls"])
                      for name in capture.pattern_by_site)
    require(total_calls == contract["runtime"]["t10_calls"],
            "M714 T10 call total drift")
    payload["schema"] = "m714_h67_ep35_pctda_pattern_s10_capture_v1"
    payload["status"] = (
        "PASS_M714_R2_PCTDA_PATTERN_CAPTURE__IDEAL_RESOURCE_LOWER_BOUND_ONLY")
    payload["m714_pctda"] = {
        "algebra_selftest": pure_python_da_selftest(),
        "aggregate": finalize(
            capture.pattern_aggregate, total_calls),
        "per_site": {name: finalize(
            counter, int(capture.aggregate[name]["calls"])) for name, counter in
                     sorted(capture.pattern_by_site.items())},
        "identity": {
            "m714_script_sha256": sha256(Path(__file__).resolve()),
            "frozen_m366_script_sha256": sha256(M366_SCRIPT),
            "m714_contract_path": str(Path(args.contract).resolve()),
            "m714_contract_sha256": sha256(Path(args.contract).resolve()),
            "m714_observed_inputs": m714_observed,
            "m366_contract_sha256": sha256(m366_contract),
            "m366_observed_inputs": m366_observed,
        },
        "schedule_contract": {
            "tile_lanes": LANES,
            "temporal_groups": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            "signed_int8_bitplanes": BITS,
            "zero_address_read_elided": True,
            "equal_address_one_vector_read_plus_lane_mask_broadcast": True,
            "groups_charged_separately_to_avoid_two_updates_per_accumulator": True,
            "at_least_one_scheduler_cycle_per_bitplane": True,
            "warm_full_service_adds_m518_12_cycle_context_tax": True,
            "m518_17n_plus_12_already_includes_five_config_beats": True,
            "build_from_weights_adds_64_table_build_cycles_not_five_more_beats": True,
            "direct_table_load_adds_23_beats_over_m518_five_beat_config": True,
            "45_unique_configs_reported_as_resident_or_per_call_load": True,
            "result_commit_is_in_12_cycle_context_tax": True,
            "reported_cycles_are_ideal_resource_lower_bounds_not_executable_schedules": True,
        },
        "claim_boundary": (
            "Exact quantized-input pattern statistics and ideal-resource DA "
            "issue lower bounds on frozen H67 ep35/no-running S10. The "
            "provisional Fixed cell area comes from an incomplete quarantined "
            "DC run and is used only to compute a budget, never as admitted "
            "PPA. No executable candidate cycle, real-output DA miter, RTL, "
            "macro timing, energy, accuracy, full-network cycle, system "
            "speedup, or headline is admitted."),
    }
    payload["admission"].update({
        "pctda_s10_pattern_capture": True,
        "pctda_ideal_resource_issue_lower_bound": True,
        "pctda_executable_cycle": False,
        "pctda_real_output_miter": False,
        "pctda_rtl": False,
        "pctda_ppa": False,
        "pctda_system_speedup": False,
        "pctda_headline": False,
    })
    m714_path = output_dir / "m714_h67_ep35_pctda_pattern_s10_capture.json"
    m714_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    (output_dir / "M714_PAYLOAD_PATH.txt").write_text(
        m714_path.name + "\n", encoding="utf-8")
    print(m714_path)


if __name__ == "__main__":
    main()
