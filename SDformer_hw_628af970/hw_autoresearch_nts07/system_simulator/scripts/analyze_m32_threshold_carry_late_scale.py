#!/usr/bin/env python3
"""Audit threshold-carrying event consumers and build a closed M32 DSE.

The audit does not treat values that happen to be close to one as binary.  A
consumer is admitted only when every frozen sample places it immediately after
one checkpoint-static, scalar-threshold, official binary ATLIF producer with
identical tensor shape and element count.  The exact real-arithmetic identity
is W(theta*b)+bias = theta*(W*b)+bias.  Fixed-point and cycle claims remain
closed until the wide late-scale operation has a VCS/DC implementation.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = (
    ROOT / "hw_autoresearch_nts07/contracts/"
    "m32_threshold_carry_input_contract_r1_20260822.json"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ceil_div(numerator, denominator):
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_path(raw_path):
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_and_verify_inputs(contract_path):
    contract_path = Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "m32_threshold_carry_input_contract_v1":
        raise ValueError("unexpected M32 input contract schema")
    paths = {}
    hashes = {}
    for name, spec in sorted(contract["inputs"].items()):
        path = resolve_path(spec["path"])
        if not path.is_file():
            raise ValueError("missing M32 input {}: {}".format(name, path))
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(
                "M32 input hash drift for {}: {} != {}".format(
                    name, actual, spec["sha256"]
                )
            )
        paths[name] = path
        hashes[name] = actual
    return contract, paths, hashes


def _class_method(tree, class_name, method_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and item.name == method_name:
                    return item
    raise ValueError("missing {}.{} source proof".format(class_name, method_name))


def prove_scalar_binary_source(source_path):
    text = Path(source_path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    official_forward = _class_method(tree, "OfficialATLIFSurrogate", "forward")
    returns = [node for node in ast.walk(official_forward) if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Tuple):
        raise ValueError("official ATLIF return structure drift")
    first = returns[0].value.elts[0]
    if not (
        isinstance(first, ast.BinOp)
        and isinstance(first.op, ast.Mult)
        and isinstance(first.left, ast.Name)
        and first.left.id == "out"
        and isinstance(first.right, ast.Name)
        and first.right.id == "thre"
    ):
        raise ValueError("official ATLIF no longer returns out * thre")

    init = _class_method(tree, "ATLIFTernaryPSN", "__init__")
    scalar_assignment = False
    for node in ast.walk(init):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "thresh"
        ):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and call.args):
            continue
        tensor_call = call.args[0]
        if not (isinstance(tensor_call, ast.Call) and tensor_call.args):
            continue
        float_call = tensor_call.args[0]
        if (
            isinstance(float_call, ast.Call)
            and isinstance(float_call.func, ast.Name)
            and float_call.func.id == "float"
            and len(float_call.args) == 1
            and isinstance(float_call.args[0], ast.Name)
            and float_call.args[0].id == "thresh"
        ):
            scalar_assignment = True
    if not scalar_assignment:
        raise ValueError("ATLIF threshold is not proven scalar by construction")

    required_fragments = [
        'if threshold_mode == "official_atlif" and output_mode != "binary"',
        "self.act = OfficialATLIFSurrogate.apply",
        "self.thresh = nn.Parameter(torch.tensor(float(thresh))",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise ValueError("ATLIF source contract tokens drift: {}".format(missing))
    return {
        "official_forward_output": "out_times_scalar_threshold",
        "threshold_shape": "scalar_by_torch_tensor_float_construction",
        "official_mode_guard": "binary_only",
    }


def unique_by(rows, field, label):
    result = {}
    for row in rows:
        key = row[field]
        if key in result:
            raise ValueError("duplicate {} identity: {}".format(label, key))
        result[key] = row
    return result


def parse_shape(raw):
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("invalid tensor shape: {}".format(raw))
    return value


def bool_false(raw):
    return str(raw).strip().lower() in ("false", "0", "no")


def audit_bypass_operators(dual_contract, execution_rows, runtime_rows,
                           atlif_rows, expected_samples):
    runtime = unique_by(runtime_rows, "name", "operator runtime")
    atlif = unique_by(atlif_rows, "name", "ATLIF activity")
    by_sample = {}
    for row in execution_rows:
        by_sample.setdefault(row["sample_id"], []).append(row)
    if len(by_sample) != expected_samples:
        raise ValueError("M32 sample population drift")
    for rows in by_sample.values():
        rows.sort(key=lambda row: int(row["call_index"]))
        indexes = [int(row["call_index"]) for row in rows]
        if indexes != list(range(len(rows))):
            raise ValueError("execution call_index is not dense and ordered")

    bypass = [
        row for row in dual_contract["operators"]
        if row["measurement_status"] == "EXPLICIT_BYPASS"
        and int(row["baseline_activity_cycles"]) > 0
    ]
    admitted = []
    continuous = []
    for operator in bypass:
        name = operator["name"]
        if name not in runtime:
            raise ValueError("missing runtime row for {}".format(name))
        hits = []
        predecessors = []
        for sample_id, rows in sorted(by_sample.items(), key=lambda item: int(item[0])):
            positions = [
                index for index, row in enumerate(rows)
                if row["kind"] == "operator" and row["name"] == name
            ]
            if len(positions) != 1:
                raise ValueError(
                    "{} appears {} times in sample {}".format(
                        name, len(positions), sample_id
                    )
                )
            position = positions[0]
            row = rows[position]
            previous = rows[position - 1] if position else None
            hits.append(row)
            predecessors.append(previous)

        immediate_atlif = [
            previous is not None
            and previous["kind"] == "atlif"
            and int(previous["call_index"]) + 1 == int(hit["call_index"])
            for hit, previous in zip(hits, predecessors)
        ]
        if any(immediate_atlif) and not all(immediate_atlif):
            raise ValueError("mixed ATLIF predecessor identity for {}".format(name))
        if not all(immediate_atlif):
            continuous.append({
                "name": name,
                "operator": operator["operator"],
                "category": operator["category"],
                "baseline_activity_cycles": int(operator["baseline_activity_cycles"]),
                "reason": "NO_IMMEDIATE_ATLIF_PRODUCER_IN_FROZEN_ORDERED_TRACE",
            })
            continue

        predecessor_names = set(previous["name"] for previous in predecessors)
        if len(predecessor_names) != 1:
            raise ValueError("producer drift across samples for {}".format(name))
        predecessor_name = next(iter(predecessor_names))
        if predecessor_name not in atlif:
            raise ValueError("missing ATLIF activity row for {}".format(predecessor_name))
        activity = atlif[predecessor_name]
        if activity["output_mode"] != "binary":
            raise ValueError("non-binary ATLIF producer for {}".format(name))
        if activity["threshold_mode"] != "official_atlif":
            raise ValueError("non-official threshold producer for {}".format(name))
        if not bool_false(activity["deployment_dead_result"]):
            raise ValueError("deployment-dead producer for {}".format(name))
        if int(activity["calls"]) != expected_samples:
            raise ValueError("ATLIF producer call population drift for {}".format(name))
        if operator["operator"] not in ("Linear", "Conv2d"):
            raise ValueError("unsupported threshold-factor consumer {}".format(name))

        for hit, previous in zip(hits, predecessors):
            if int(previous["output_elements"]) != int(hit["input_elements"]):
                raise ValueError("producer/consumer element mismatch for {}".format(name))
            if parse_shape(previous["output_shape"]) != parse_shape(hit["input_shape"]):
                raise ValueError("producer/consumer shape mismatch for {}".format(name))

        runtime_row = runtime[name]
        if int(runtime_row["calls"]) != expected_samples:
            raise ValueError("operator runtime call population drift for {}".format(name))
        output_sum = sum(int(hit["output_elements"]) for hit in hits)
        if output_sum != int(runtime_row["output_elements"]):
            raise ValueError("operator output population mismatch for {}".format(name))
        minimum = float(runtime_row["input_sample_value_min"])
        maximum = float(runtime_row["input_sample_value_max"])
        absmax = float(runtime_row["input_sample_value_absmax"])
        finite_ratio = float(runtime_row["input_sample_finite_ratio"])
        density = float(runtime_row["input_sample_density"])
        mean_abs = float(runtime_row["input_sample_value_mean_abs"])
        if not (
            finite_ratio == 1.0 and minimum >= 0.0 and maximum > 0.0
            and abs(maximum - absmax) <= 1.0e-9 and density > 0.0
        ):
            raise ValueError("runtime corroboration drift for {}".format(name))
        observed_amplitude = mean_abs / density
        if abs(observed_amplitude - maximum) > 2.0e-5:
            raise ValueError("runtime scalar-amplitude corroboration failed for {}".format(name))

        per_invocation_outputs = [int(hit["output_elements"]) for hit in hits]
        admitted.append({
            "name": name,
            "operator": operator["operator"],
            "category": operator["category"],
            "producer": predecessor_name,
            "baseline_activity_cycles": int(operator["baseline_activity_cycles"]),
            "calls": expected_samples,
            "output_elements_aggregate": output_sum,
            "output_elements_per_sample": output_sum // expected_samples,
            "per_invocation_outputs": per_invocation_outputs,
            "observed_first_sample_threshold_amplitude": maximum,
            "semantic_status": (
                "REAL_ARITHMETIC_SCALAR_FACTOR_IDENTITY_AND_ORDERED_SHAPE_PROOF"
            ),
        })
    return admitted, continuous


def average_rounded_cycles(admitted, lanes):
    total = sum(
        ceil_div(outputs, lanes)
        for row in admitted for outputs in row["per_invocation_outputs"]
    )
    calls = sum(row["calls"] for row in admitted)
    operators = len(admitted)
    if calls != operators * 10 or total % 10:
        raise ValueError("M32 per-sample late-scale rounding is not integral")
    return total // 10


def build_sensitivity_rows(fixed_cycles, m30, m25, event_cycles,
                           admitted, outputs_per_sample):
    candidates = {
        row["name"]: row for row in m30["port_candidates"]
    }
    base = candidates["dual256b_independent_output_packed24"]
    speeds = m25["effective_m4_speed_rebind"]
    line_points = [
        ("local", int(base["local_cycles"]), float(speeds["local"])),
        ("motion", int(base["motion_cycles"]),
         float(speeds["hybrid_motion_stateful"])),
    ]
    variants = [
        ("byte12_arithmetic_lower_bound", 12,
         "ARITHMETIC_LOWER_BOUND_UNSIGNED_BYTE_AND_RECOMBINATION_NOT_EXECUTABLE"),
        ("radix24_provisional", 24,
         "PROVISIONAL_SIGN_SAFE_ENVELOPE_REQUIRES_VCS_AND_DC"),
        ("stress48", 48,
         "STRESS_ENVELOPE_NOT_AN_IMPLEMENTED_SCHEDULE"),
    ]
    rows = []
    for line, current_cycles, speed in line_points:
        transferred_cycles = int(math.ceil(event_cycles / speed))
        for variant, products_per_output, status in variants:
            outputs_per_cycle = 96 // products_per_output
            if outputs_per_cycle <= 0:
                raise ValueError("invalid late-scale product decomposition")
            late_cycles = average_rounded_cycles(admitted, outputs_per_cycle)
            if late_cycles != ceil_div(outputs_per_sample, outputs_per_cycle):
                raise ValueError("late-scale cycle accounting drift")
            proposal_cycles = (
                current_cycles - event_cycles + transferred_cycles + late_cycles
            )
            rows.append({
                "line": line,
                "variant": variant,
                "status": status,
                "existing_m30_cycles": current_cycles,
                "transferred_fixed_bypass_cycles": event_cycles,
                "borrowed_effective_m4_speed": speed,
                "borrowed_event_accumulation_cycles": transferred_cycles,
                "int8_product_slots": 96,
                "int8_partial_products_per_output": products_per_output,
                "late_scale_outputs_per_cycle": outputs_per_cycle,
                "late_scale_outputs_per_sample": outputs_per_sample,
                "late_scale_cycles": late_cycles,
                "proposal_compute_cycles_sensitivity": proposal_cycles,
                "speedup_vs_fixed_sensitivity": fixed_cycles / float(proposal_cycles),
                "crosses_2p5x": proposal_cycles * 2.5 < fixed_cycles,
                "crosses_2p75x": proposal_cycles * 2.75 < fixed_cycles,
                "crosses_3x": proposal_cycles * 3 < fixed_cycles,
                "cycles_margin_to_3x": fixed_cycles / 3.0 - proposal_cycles,
            })
    return rows


def build_report(contract_path=DEFAULT_CONTRACT):
    contract, paths, hashes = load_and_verify_inputs(contract_path)
    source_proof = prove_scalar_binary_source(paths["atlif_source"])
    profile = json.loads(paths["profile"].read_text(encoding="utf-8"))
    identity = contract["identity"]
    if (
        profile.get("experiment") != identity["experiment"]
        or int(profile.get("samples", -1)) != int(identity["samples"])
        or not profile.get("ordered_trace")
        or not profile.get("dual_line_trace")
        or profile["artifact_identity"]["checkpoint_sha256"]
        != identity["checkpoint_sha256"]
        or profile["checkpoint_load_audit"]["missing_count"] != 0
        or profile["checkpoint_load_audit"]["unexpected_count"] != 0
    ):
        raise ValueError("M32 H67 profile identity drift")

    dual_contract = json.loads(paths["dual_line_contract"].read_text(encoding="utf-8"))
    if (
        dual_contract.get("schema") != "h67_dual_line_full_system_contract_v0"
        or dual_contract.get("status") != "PASS_TRACE_PRESENT_TIMING_PENDING"
    ):
        raise ValueError("unexpected dual-line input contract")
    execution_rows = read_csv(paths["execution_trace"])
    runtime_rows = read_csv(paths["operator_runtime"])
    atlif_rows = read_csv(paths["atlif_activity"])
    dual_rows = read_csv(paths["dual_line_operator_trace"])
    if not dual_rows or len(execution_rows) != 1840:
        raise ValueError("M32 ordered trace population drift")

    admitted, continuous = audit_bypass_operators(
        dual_contract, execution_rows, runtime_rows, atlif_rows,
        int(identity["samples"]),
    )
    event_cycles = sum(row["baseline_activity_cycles"] for row in admitted)
    continuous_cycles = sum(row["baseline_activity_cycles"] for row in continuous)
    outputs_per_sample = sum(row["output_elements_per_sample"] for row in admitted)
    expected = contract["expected"]
    actual_expected = {
        "factorable_bypass_operators": len(admitted),
        "continuous_bypass_operators": len(continuous),
        "factorable_bypass_cycles": event_cycles,
        "continuous_bypass_cycles": continuous_cycles,
        "factorable_outputs_per_sample": outputs_per_sample,
    }
    for key, value in actual_expected.items():
        if int(expected[key]) != int(value):
            raise ValueError("M32 expected census drift for {}".format(key))

    m25 = json.loads(paths["m25_cycle_model"].read_text(encoding="utf-8"))
    m30 = json.loads(paths["m30_cycle_dse"].read_text(encoding="utf-8"))
    if m25.get("schema") != "m25_resource_bounded_tiled_cycle_architecture_v1":
        raise ValueError("unexpected M25 cycle contract")
    if m30.get("schema") != "m30_resident_stream_system_dse_v2":
        raise ValueError("unexpected M30 cycle contract")
    fixed_cycles = int(m30["frozen_resources"]["fixed_compute_cycles"])
    if fixed_cycles != int(expected["fixed_compute_cycles"]):
        raise ValueError("M32 fixed-cycle identity drift")
    sensitivity = build_sensitivity_rows(
        fixed_cycles, m30, m25, event_cycles, admitted, outputs_per_sample
    )

    return {
        "schema": "m32_threshold_carry_late_scale_audit_v1",
        "status": (
            "PASS_REAL_ARITHMETIC_SEMANTIC_CENSUS_"
            "CYCLE_SENSITIVITY_ONLY_NO_HEADLINE_CLAIM"
        ),
        "identity": {
            "input_contract": str(Path(contract_path).resolve()),
            "input_contract_sha256": sha256(contract_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "verified_input_sha256": hashes,
            "experiment": identity["experiment"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "samples": int(identity["samples"]),
        },
        "source_semantic_proof": source_proof,
        "mechanism": {
            "name": "THRESHOLD_CARRYING_EVENT_LATE_SCALE",
            "event_representation": "b in {0,1}; theta carried once per producer context",
            "local_identity": "W*(theta*b)+bias = theta*(W*b)+bias",
            "motion_identity": (
                "for checkpoint-static theta, delta(theta*b) = theta*delta(b)"
            ),
            "required_stage_order": [
                "accumulate selected W columns into signed accumulator S",
                "compute theta*S with explicit deployment Q-format",
                "add bias after scaling; never scale bias by theta",
            ],
            "shared_pool_constraint": (
                "late scale must be decomposed onto the sole 96 signed-INT8 "
                "product slots or charged as additional area"
            ),
        },
        "census": {
            "factorable_bypass_operators": len(admitted),
            "factorable_bypass_cycles": event_cycles,
            "factorable_outputs_per_sample": outputs_per_sample,
            "continuous_bypass_operators": len(continuous),
            "continuous_bypass_cycles": continuous_cycles,
            "factorable": admitted,
            "continuous_preserved": continuous,
        },
        "cycle_sensitivity": {
            "fixed_compute_cycles": fixed_cycles,
            "rows": sensitivity,
            "interpretation": (
                "subtracts only frozen EXPLICIT_BYPASS cycles admitted by the "
                "M32 census, borrows M25's measured effective M4 speed for a "
                "sensitivity, and serially charges late scaling; it is not an "
                "executable full-system cycle result"
            ),
        },
        "claim_boundary": {
            "permitted": [
                "exact real-arithmetic scalar factor identity",
                "frozen ten-sample ordered adjacency/shape census",
                "resource-explicit cycle sensitivity for 12/24/48 products per output",
            ],
            "forbidden": [
                "treating near-one thresholds as exactly one",
                "claiming ordered-hook adjacency proves tensor pointer identity",
                "claiming fixed-point or checkpoint-output bit exactness",
                "claiming any sensitivity row is VCS/DC executable",
                "claiming PPA, energy, FPS, DRAMsim3 timing, or DATE comparison",
                "claiming 2.5x, 2.75x, or 3x as measured system performance",
            ],
            "next_admission_gates": [
                "freeze threshold/weight/bias Q-formats and rounding/saturation order",
                "capture producer bits and consumer tensors for a pointer/value miter",
                "implement signed Acc32-by-Q24 decomposition on the sole 96-slot pool",
                "VCS numeric/SVA verification including bias ordering and saturation",
                "DC/Formality/SAIF/PTPX and address-timed SRAM/DRAM integration",
                "quantized checkpoint accuracy and cross-sequence density stratification",
            ],
        },
        "headline_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite M32 report: {}".format(args.output))
    report = build_report(args.contract.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
