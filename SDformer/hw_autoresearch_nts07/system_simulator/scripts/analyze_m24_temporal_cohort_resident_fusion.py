#!/usr/bin/env python3
"""Audit exact T10 coefficient cohorts and the M4->M21->ATLIF boundary.

M24 is deliberately fail closed.  Frozen P4 tile files are exact for the rows
they contain, but they are samples rather than a full coefficient census.  The
script therefore reports exact sampled work and its exact fraction of the
aggregate dual-line work; it never extrapolates sampled masks to the network.

The M22 ``serialized_service_ticks`` field is copied only as a named logical
transport quantity.  It is never used as a hardware cycle count.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


EXACT_STATUS = "PASS_EXACT_SOURCE_WORK"
M22_STATUS = "PASS_FROZEN_INPUT_PARTIAL_TRANSACTION_LEDGER_NOT_DRAMSIM_OR_SPEEDUP"
M18_STATUS = "PASS_EXACT_PATH_CERTIFICATES_ALL_BN_BLOCKED_M15_PROHIBITED"
M21_STATUS = "PASS_EXACT_M17_ORDERED_ELASTIC_BANKED_MOMENT_DSE_NOT_SYSTEM_SPEEDUP"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def integer(row, field):
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid integer {} in {}".format(field, row.get("name", "row")))
    return value


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError("non-canonical Boolean {!r}".format(value))
    return value == "True"


def ceil_log2(value):
    if value <= 0:
        raise ValueError("ceil_log2 requires a positive value")
    return max(1, int(math.ceil(math.log(value, 2))))


def popcount(value):
    return bin(value).count("1")


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def portable_path(path):
    """Return a repository-style path when the artifact lives below hw root."""
    path = Path(path)
    parts = path.parts
    if "hw_autoresearch_nts07" in parts:
        index = parts.index("hw_autoresearch_nts07")
        return str(Path(*parts[index:]))
    return path.name


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: {}".format(path))
        return list(reader)


def resolve_entry(repo_root, entry):
    relative = Path(entry.get("path", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("contract path is not repository-relative: {}".format(relative))
    path = Path(repo_root) / relative
    if not path.is_file():
        raise ValueError("missing contracted input: {}".format(path))
    actual = sha256(path)
    if actual != entry.get("sha256"):
        raise ValueError(
            "contract SHA mismatch for {}: expected={} actual={}".format(
                relative, entry.get("sha256"), actual
            )
        )
    return path


def load_contract(path, expected_sha256, repo_root):
    path = Path(path)
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            "input contract SHA mismatch: expected={} actual={}".format(
                expected_sha256, actual
            )
        )
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != "m24_temporal_cohort_input_contract_v1"
        or contract.get("status") != "FROZEN_EXPECTED_INPUT_IDENTITY"
    ):
        raise ValueError("M24 input contract schema/status is not admitted")
    paths = {}
    for section_name, section in contract.items():
        if section_name in ("schema", "status", "requested_topology"):
            continue
        if not isinstance(section, dict):
            continue
        for name, value in section.items():
            if isinstance(value, dict) and "path" in value and "sha256" in value:
                paths[section_name + "." + name] = resolve_entry(repo_root, value)
            elif isinstance(value, dict):
                for child_name, child in value.items():
                    if isinstance(child, dict) and "path" in child and "sha256" in child:
                        paths[section_name + "." + name + "." + child_name] = resolve_entry(
                            repo_root, child
                        )
    return contract, actual, paths


def read_npy_u8(archive, member):
    with zipfile.ZipFile(str(archive), "r") as handle:
        raw = handle.read(member)
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("invalid NPY magic: {}".format(member))
    major = raw[6]
    if major == 1:
        header_length = struct.unpack("<H", raw[8:10])[0]
        header_start = 10
    elif major in (2, 3):
        header_length = struct.unpack("<I", raw[8:12])[0]
        header_start = 12
    else:
        raise ValueError("unsupported NPY version {}".format(major))
    header = ast.literal_eval(
        raw[header_start:header_start + header_length].decode("latin1").strip()
    )
    if (
        header.get("descr") != "|u1"
        or header.get("fortran_order") is not False
        or not isinstance(header.get("shape"), tuple)
        or len(header["shape"]) != 2
    ):
        raise ValueError("M24 admits only row-major two-dimensional uint8 NPY")
    rows, columns = [int(value) for value in header["shape"]]
    payload = raw[header_start + header_length:]
    if rows <= 0 or columns <= 0 or len(payload) != rows * columns:
        raise ValueError("NPY payload extent mismatch")
    return [payload[index * columns:(index + 1) * columns] for index in range(rows)]


def bitmap(row_bytes):
    return int.from_bytes(row_bytes, byteorder="little", signed=False)


def physical_chunk_key(row):
    return (
        row["sample_id"], row["sequence_key"], row["name"], row["operator"],
        row["operator_call_index"], row["row_id"], row["chunk_index"],
        row["weight_group"], row["source_base"], row["valid_bits"],
        row["output_channel_fanout"],
    )


def physical_row_key(row):
    return (
        row["sample_id"], row["sequence_key"], row["name"], row["operator"],
        row["operator_call_index"], row["row_id"], row["weight_group"],
        row["output_channel_fanout"],
    )


def empty_line_metrics():
    return {
        "coefficient_read_vectors_step_major": 0,
        "coefficient_read_vectors_cohort": 0,
        "coefficient_scalar_reads_step_major": 0,
        "coefficient_scalar_reads_cohort": 0,
        "destination_scalar_updates": 0,
        "positive_destination_scalar_updates": 0,
        "negative_destination_scalar_updates": 0,
        "positive_mask_source_entries": 0,
        "negative_mask_source_entries": 0,
        "sparse_control_payload_bits": 0,
        "coefficient_cache_peak_bits": 0,
        "resident_accumulator_peak_bits": 0,
        "cohort_resident_peak_bits": 0,
        "selector_control_bits": 0,
        "cohorts": 0,
    }


def merge_line(target, source):
    for field in (
        "coefficient_read_vectors_step_major",
        "coefficient_read_vectors_cohort",
        "coefficient_scalar_reads_step_major",
        "coefficient_scalar_reads_cohort",
        "destination_scalar_updates",
        "positive_destination_scalar_updates",
        "negative_destination_scalar_updates",
        "positive_mask_source_entries",
        "negative_mask_source_entries",
        "sparse_control_payload_bits",
        "selector_control_bits",
        "cohorts",
    ):
        target[field] += source[field]
    for field in (
        "coefficient_cache_peak_bits", "resident_accumulator_peak_bits",
        "cohort_resident_peak_bits",
    ):
        target[field] = max(target[field], source[field])


def finalize_line(metrics):
    result = dict(metrics)
    step = result["coefficient_scalar_reads_step_major"]
    cohort = result["coefficient_scalar_reads_cohort"]
    updates = result["destination_scalar_updates"]
    if min(step, cohort, updates) <= 0 or cohort > step or updates != step:
        raise ValueError("cohort coefficient/update conservation failed")
    if (
        result["positive_destination_scalar_updates"]
        + result["negative_destination_scalar_updates"] != updates
    ):
        raise ValueError("signed destination update conservation failed")
    result["coefficient_read_reduction_fraction"] = 1.0 - float(cohort) / step
    result["serialized_read_plus_update_operation_envelope"] = {
        "fair_step_major": step + updates,
        "cohort": cohort + updates,
        "sampled_component_speedup": float(step + updates) / (cohort + updates),
        "boundary": "exact scalar-operation sum, not measured or simulated cycles",
    }
    result["fully_overlapped_read_update_envelope"] = {
        "fair_step_major": max(step, updates),
        "cohort": max(cohort, updates),
        "sampled_component_speedup": float(max(step, updates)) / max(cohort, updates),
        "boundary": "same one coefficient-read path plus one update path; lower-bound operation stages, not cycles",
    }
    result["equal_resource_baseline"] = {
        "coefficient_read_ports": 1,
        "destination_update_paths": 1,
        "resident_capacity_bits": result["cohort_resident_peak_bits"],
        "coefficient_precision_bits": 8,
        "accumulator_precision_bits": 32,
        "baseline_policy": "step-major may use the identical resident capacity and ports",
    }
    result["strongest_composable_same_resource_baseline"] = {
        "policy": (
            "a generic coefficient-resident cache uses the identical capacity and retains "
            "each source/weight vector across the same T10 row cohort"
        ),
        "coefficient_scalar_reads": cohort,
        "destination_scalar_updates": updates,
        "operation_envelope_speedup_of_cohort_masks": 1.0,
        "traffic_reduction_of_cohort_masks": 0.0,
        "boundary": (
            "the step-major comparison is an implementation ablation; it is not the "
            "strongest composable architecture baseline"
        ),
    }
    return result


def analyze_tiles(label, manifest_path, records_path, packed_path, dual_path, temporal_steps):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "dual_line_real_tile_trace_v1"
        or manifest.get("status")
        != "PASS_REAL_BITMAPS_ROW_SELECTOR_TILE_EXECUTION_NOT_ACC32_ORACLE"
        or int(manifest.get("pairs_per_operator_call", -1)) != 4
        or int(manifest.get("tile_bits", -1)) != 256
    ):
        raise ValueError("{} frozen tile manifest is not the admitted sampled-v1 evidence".format(label))
    if (
        manifest.get("sha256", {}).get("tile_records.csv") != sha256(records_path)
        or manifest.get("sha256", {}).get("packed_tiles.npz") != sha256(packed_path)
    ):
        raise ValueError("{} tile manifest payload SHA mismatch".format(label))
    records = read_csv(records_path)
    current_rows = read_npy_u8(packed_path, "packed_current_bits.npy")
    previous_rows = read_npy_u8(packed_path, "packed_previous_bits.npy")
    if not (len(records) == len(current_rows) == len(previous_rows) == int(manifest["records"])):
        raise ValueError("{} tile payload cardinality mismatch".format(label))

    dual = read_csv(dual_path)
    dual_status = defaultdict(set)
    full_work = {"local_line": 0, "motion_selector_shared_state": 0}
    for row in dual:
        dual_status[row["name"]].add(row["status"])
        if row["status"] == EXACT_STATUS:
            full_work["local_line"] += integer(row, "local_work")
            full_work["motion_selector_shared_state"] += integer(row, "selected_work")
    if len(dual_status) != 79 or any(len(values) != 1 for values in dual_status.values()):
        raise ValueError("{} M22 dual trace does not have one status for each of 79 operators".format(label))
    exact_names = {name for name, values in dual_status.items() if EXACT_STATUS in values}
    tile_names = {row["name"] for row in records}
    if tile_names != exact_names or int(manifest["operators"]) != len(tile_names):
        raise ValueError("{} tile/exact-operator name census mismatch".format(label))

    groups = defaultdict(list)
    for index, row in enumerate(records):
        if integer(row, "record_id") != index:
            raise ValueError("{} tile record_id is not dense".format(label))
        groups[physical_chunk_key(row)].append((row, bitmap(current_rows[index]), bitmap(previous_rows[index])))

    totals = {
        "local_line": empty_line_metrics(),
        "motion_selector_shared_state": empty_line_metrics(),
    }
    per_operator = {
        "local_line": defaultdict(empty_line_metrics),
        "motion_selector_shared_state": defaultdict(empty_line_metrics),
    }
    row_selector_seen = {"local_line": set(), "motion_selector_shared_state": set()}
    for key in sorted(groups):
        values = sorted(groups[key], key=lambda item: integer(item[0], "temporal_step"))
        if [integer(item[0], "temporal_step") for item in values] != list(range(temporal_steps)):
            raise ValueError("{} physical chunk is not exact T{}".format(label, temporal_steps))
        valid_bits = integer(values[0][0], "valid_bits")
        fanout = integer(values[0][0], "output_channel_fanout")
        if not 0 < valid_bits <= 256 or fanout <= 0:
            raise ValueError("invalid tile geometry")
        valid_mask = (1 << valid_bits) - 1
        local_events = 0
        local_union = 0
        motion_events = 0
        motion_union = 0
        motion_positive_events = 0
        motion_negative_events = 0
        motion_positive_union = 0
        motion_negative_union = 0
        previous_expected = 0
        for timestep, (row, current, previous) in enumerate(values):
            if current & ~valid_mask or previous & ~valid_mask:
                raise ValueError("nonzero invalid tail bits")
            state_valid = boolean(row["state_valid"])
            if state_valid != (timestep > 0) or previous != previous_expected:
                raise ValueError("{} previous bitmap temporal identity mismatch".format(label))
            positive = current & ~previous
            negative = previous & ~current
            if (
                popcount(current) != integer(row, "tile_current_count")
                or popcount(positive) != integer(row, "tile_positive_count")
                or popcount(negative) != integer(row, "tile_negative_count")
            ):
                raise ValueError("{} packed bitmap/count mismatch".format(label))
            local_events += popcount(current)
            local_union |= current
            if boolean(row["row_use_motion"]):
                if not state_valid:
                    raise ValueError("Motion cannot be selected without valid previous state")
                selected_positive, selected_negative = positive, negative
            else:
                selected_positive, selected_negative = current, 0
            motion_events += popcount(selected_positive | selected_negative)
            motion_union |= selected_positive | selected_negative
            motion_positive_events += popcount(selected_positive)
            motion_negative_events += popcount(selected_negative)
            motion_positive_union |= selected_positive
            motion_negative_union |= selected_negative
            previous_expected = current

        source_index_bits = ceil_log2(valid_bits)
        accumulator_bits = fanout * 32
        operator = values[0][0]["name"]
        local = empty_line_metrics()
        local["coefficient_read_vectors_step_major"] = local_events
        local["coefficient_read_vectors_cohort"] = popcount(local_union)
        local["coefficient_scalar_reads_step_major"] = local_events * fanout
        local["coefficient_scalar_reads_cohort"] = popcount(local_union) * fanout
        local["destination_scalar_updates"] = local_events * fanout
        local["positive_destination_scalar_updates"] = local_events * fanout
        local["positive_mask_source_entries"] = popcount(local_union)
        local["sparse_control_payload_bits"] = popcount(local_union) * (
            source_index_bits + temporal_steps
        )
        local["coefficient_cache_peak_bits"] = popcount(local_union) * fanout * 8
        local["resident_accumulator_peak_bits"] = accumulator_bits
        local["cohort_resident_peak_bits"] = (
            local["sparse_control_payload_bits"]
            + local["coefficient_cache_peak_bits"] + accumulator_bits
        )
        local["cohorts"] = 1

        motion = empty_line_metrics()
        motion["coefficient_read_vectors_step_major"] = motion_events
        motion["coefficient_read_vectors_cohort"] = popcount(motion_union)
        motion["coefficient_scalar_reads_step_major"] = motion_events * fanout
        motion["coefficient_scalar_reads_cohort"] = popcount(motion_union) * fanout
        motion["destination_scalar_updates"] = motion_events * fanout
        motion["positive_destination_scalar_updates"] = motion_positive_events * fanout
        motion["negative_destination_scalar_updates"] = motion_negative_events * fanout
        motion["positive_mask_source_entries"] = popcount(motion_positive_union)
        motion["negative_mask_source_entries"] = popcount(motion_negative_union)
        motion["sparse_control_payload_bits"] = popcount(motion_union) * (
            source_index_bits + 2 * temporal_steps
        )
        motion["coefficient_cache_peak_bits"] = popcount(motion_union) * fanout * 8
        motion["resident_accumulator_peak_bits"] = accumulator_bits
        motion["cohort_resident_peak_bits"] = (
            motion["sparse_control_payload_bits"]
            + motion["coefficient_cache_peak_bits"] + accumulator_bits
        )
        motion["cohorts"] = 1
        for line, item in (("local_line", local), ("motion_selector_shared_state", motion)):
            row_key = physical_row_key(values[0][0])
            if row_key not in row_selector_seen[line]:
                row_selector_seen[line].add(row_key)
                if line == "motion_selector_shared_state":
                    item["selector_control_bits"] = temporal_steps
                    item["sparse_control_payload_bits"] += temporal_steps
                    item["cohort_resident_peak_bits"] += temporal_steps
            merge_line(totals[line], item)
            merge_line(per_operator[line][operator], item)

    finalized = {}
    operator_rows = []
    for line in ("local_line", "motion_selector_shared_state"):
        metrics = finalize_line(totals[line])
        sampled = metrics["destination_scalar_updates"]
        if sampled > full_work[line] or full_work[line] <= 0:
            raise ValueError("{} sampled/full coefficient conservation failed".format(label))
        metrics["aggregate_exact_coefficient_work"] = full_work[line]
        metrics["sampled_exact_coefficient_work"] = sampled
        metrics["exact_coefficient_coverage_fraction"] = float(sampled) / full_work[line]
        metrics["fallback_coefficient_fraction"] = 1.0 - float(sampled) / full_work[line]
        metrics["headline_coverage_admitted"] = False
        metrics["headline_rejection"] = (
            "frozen P4 masks are exact samples, not an exhaustive coefficient census"
        )
        finalized[line] = metrics
        for operator in sorted(per_operator[line]):
            item = finalize_line(per_operator[line][operator])
            operator_rows.append({
                "identity": label,
                "line": line,
                "operator": operator,
                "sampled_step_major_coefficient_scalar_reads": item["coefficient_scalar_reads_step_major"],
                "sampled_cohort_coefficient_scalar_reads": item["coefficient_scalar_reads_cohort"],
                "sampled_destination_scalar_updates": item["destination_scalar_updates"],
                "sampled_positive_destination_scalar_updates": item["positive_destination_scalar_updates"],
                "sampled_negative_destination_scalar_updates": item["negative_destination_scalar_updates"],
                "sampled_coefficient_read_reduction_fraction": item["coefficient_read_reduction_fraction"],
                "sampled_serialized_operation_speedup": item["serialized_read_plus_update_operation_envelope"]["sampled_component_speedup"],
                "sampled_fully_overlapped_operation_speedup": item["fully_overlapped_read_update_envelope"]["sampled_component_speedup"],
                "cohort_resident_peak_bits": item["cohort_resident_peak_bits"],
            })
    return {
        "identity": label,
        "operator_topology": len(dual_status),
        "exact_operator_names": len(exact_names),
        "sampled_tile_operator_names": len(tile_names),
        "sampled_records": len(records),
        "sampled_physical_chunks": len(groups),
        "pairs_per_operator_call": int(manifest["pairs_per_operator_call"]),
        "line_metrics": finalized,
        "status_counts_by_operator": dict(sorted(Counter(
            next(iter(values)) for values in dual_status.values()
        ).items())),
        "evidence_boundary": (
            "exact T10 packed bitmaps for the frozen sampled rows only; aggregate dual-line "
            "work supplies a denominator but cannot reconstruct missing temporal masks"
        ),
    }, operator_rows


def analyze_resident_fusion(m18_path, m21_path, m7_path, m22):
    m18 = json.loads(Path(m18_path).read_text(encoding="utf-8"))
    m21 = json.loads(Path(m21_path).read_text(encoding="utf-8"))
    m7 = json.loads(Path(m7_path).read_text(encoding="utf-8"))
    if m18.get("status") != M18_STATUS or len(m18.get("rows", [])) != 13:
        raise ValueError("M18 exact BN-blocked boundary census is not admitted")
    if m21.get("status") != M21_STATUS or int(m21.get("summary", {}).get("operators", -1)) != 13:
        raise ValueError("M21 13-operator DSE is not admitted")
    if m7.get("status") != "PREMACRO_CANDIDATE" or m7.get("paper_ppa_admitted") is not False:
        raise ValueError("M7 premacro claim boundary drift")
    m18_names = {row["producer"] for row in m18["rows"]}
    m21_names = {row["operator"] for row in m21.get("selected_operator_rows", [])}
    if len(m18_names) != 13 or m18_names != m21_names:
        raise ValueError("M18/M21 producer census mismatch")
    h67_profile = m22["identities"]["h67_ep35"]["profile_identity"]
    if m18.get("identities", {}).get("checkpoint_sha256") != h67_profile["checkpoint_sha256"]:
        raise ValueError("M18/M22 H67 checkpoint mismatch")
    elements = 0
    for row in m18["rows"]:
        shape = row.get("producer_output_tensor", {}).get("shape", [])
        if len(shape) < 2 or int(row.get("temporal_steps", -1)) != 10:
            raise ValueError("M18 producer tensor extent is invalid")
        elements += product(shape)
    if elements != 552960000:
        raise ValueError("unexpected M18 materialized element population")
    m21_summary = m21["summary"]
    logical_ticks = {}
    for identity, item in sorted(m22["identities"].items()):
        logical_ticks[identity] = {}
        for variant, variant_item in sorted(item["variants"].items()):
            logical_ticks[identity][variant] = variant_item["totals"]["serialized_byte_service_ticks"]
    return {
        "exact_h67_single_sample_edges": 13,
        "exact_h67_single_sample_elements": elements,
        "retained_unnormalized_materialization_bytes_one_way": elements * 4,
        "retained_m22_boundary_movements": [
            "producer unnormalized Acc32/output write",
            "post-global-BN-barrier fused BN+ATLIF replay read",
        ],
        "strict_transactions_deleted_from_canonical_m22": 0,
        "strict_bytes_deleted_from_canonical_m22": 0,
        "reason_no_m22_deletion": (
            "all 13 exact paths cross a no-running global BN barrier; M21 explicitly retains "
            "producer materialization, and M22 already models the strongest two-movement "
            "online-moments plus BN-ATLIF-fusion boundary"
        ),
        "strongest_composable_baseline_speedup": 1.0,
        "liveness": {
            "m18_context_release": m18["hardware_tag_ledger"]["lifecycle_contract"]["context_release"],
            "m21_barrier": m21["architecture_contract"]["barrier"],
            "direct_pre_barrier_atlif_residency_admitted": False,
        },
        "payload_only_capacity_bits": {
            "three_tile_fifo40": int(m21_summary["three_tile_fifo40_required_payload_bits"]),
            "maximum_moment_state": int(m21_summary["maximum_moment_state_bits"]),
            "sum_not_physical_macro_sizing": int(m21_summary["three_tile_fifo40_required_payload_bits"])
            + int(m21_summary["maximum_moment_state_bits"]),
        },
        "m7_premacro_logic": m7["metrics"],
        "m22_serialized_byte_service_ticks_copied_as_logical_ticks_not_cycles": logical_ticks,
        "coverage_boundary": (
            "M18/M21 close one H67 sample and 13 paths. Local ep44 and the remaining network "
            "do not have equivalent dependency/liveness certificates."
        ),
    }


def amdahl_audit(topology, identities):
    legacy_fraction = float(topology["legacy_eligible_system_fraction"])
    legacy_target = float(topology["legacy_eligible_engine_target_for_2x"])
    recomputed_target = legacy_fraction / (0.5 - (1.0 - legacy_fraction))
    implied_fraction = 0.5 / (1.0 - 1.0 / legacy_target)
    sampled = {}
    for label, identity in sorted(identities.items()):
        sampled[label] = {}
        for line, item in sorted(identity["line_metrics"].items()):
            component = item["serialized_read_plus_update_operation_envelope"]["sampled_component_speedup"]
            sampled[label][line] = {
                "sampled_component_operation_speedup": component,
                "hypothetical_full_system_speedup_if_legacy_fraction_applied": 1.0 / (
                    (1.0 - legacy_fraction) + legacy_fraction / component
                ),
                "strongest_same_resource_component_speedup": 1.0,
                "strongest_same_resource_hypothetical_full_system_speedup": 1.0,
                "headline_admitted": False,
            }
    return {
        "legacy_eligible_system_fraction": legacy_fraction,
        "recomputed_eligible_engine_speedup_required_for_2x": recomputed_target,
        "frozen_legacy_target_for_2x": legacy_target,
        "eligible_fraction_implied_by_frozen_target": implied_fraction,
        "coverage_fraction_discrepancy": implied_fraction - legacy_fraction,
        "threshold_consistency_admitted": False,
        "sampled_mask_what_if_not_headline": sampled,
        "two_x_gate": (
            "NO_GO: exact coefficient fallback exceeds 5%, sampled component operations are "
            "not cycles, a generic same-capacity resident cache matches the cohort traffic, "
            "and the two legacy Amdahl constants are mutually inconsistent"
        ),
    }


def write_outputs(output_dir, payload, operator_rows, input_contract, gap_contract):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("refusing to overwrite output directory: {}".format(output_dir))
    output_dir.mkdir(parents=True)
    summary_path = output_dir / "m24_temporal_cohort_resident_fusion.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = output_dir / "m24_sampled_operator_cohort_dse.csv"
    fields = list(operator_rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(operator_rows)
    report_path = output_dir / "m24_REPORT.md"
    report_path.write_text(
        "# M24 temporal-cohort / resident-fusion strict DSE\n\n"
        "Status: **{}**\n\n"
        "Frozen P4 packed tiles are exact only for sampled rows. H67 covers 31 exact "
        "operator names and Local ep44 covers 36, while the requested topology is 57/79. "
        "The output records the coefficient-work fallback fraction and rejects all network "
        "speedup headlines until it is below 5%.\n\n"
        "Local presence masks and Motion signed masks are bit-exact for every admitted T10 "
        "sampled cohort. Coefficient reads, destination updates, sparse-control payload and "
        "resident Acc32 capacity are reported with an equal-resource step-major baseline. "
        "The additive and overlapped figures are operation envelopes, not cycles.\n\n"
        "The 13 M18/M21 H67 paths all cross dynamic BN. Canonical M22 already retains the "
        "two mandatory movements and assumes fused BN+ATLIF replay, so M24 deletes zero M22 "
        "transactions. Its serialized ticks remain logical byte-service ticks only.\n\n"
        "Gap contract: `{}` (`{}`).\n"
        .format(payload["status"], portable_path(gap_contract), sha256(gap_contract)),
        encoding="utf-8",
    )
    artifacts = {}
    for path in (summary_path, csv_path, report_path):
        artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema": "m24_temporal_cohort_output_manifest_v1",
        "status": payload["status"],
        "artifacts": artifacts,
        "input_contract": {"path": portable_path(input_contract), "sha256": sha256(input_contract)},
        "gap_contract": {"path": portable_path(gap_contract), "sha256": sha256(gap_contract)},
        "sources": {
            portable_path(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            "hw_autoresearch_nts07/system_simulator/tests/test_m24_temporal_cohort_resident_fusion.py": sha256(
                Path(__file__).resolve().parents[1]
                / "tests/test_m24_temporal_cohort_resident_fusion.py"
            ),
        },
        "claim": "EXACT_SAMPLED_COHORT_DSE_AND_GAP_CONTRACT_NOT_SYSTEM_CYCLES_OR_SPEEDUP",
    }
    manifest_path = output_dir / "m24_output_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt = []
    for path in (summary_path, csv_path, report_path, manifest_path):
        receipt.append("{}  {}".format(sha256(path), path.name))
    (output_dir / "m24_evidence.sha256").write_text("\n".join(receipt) + "\n", encoding="utf-8")


def build(contract, contract_sha, paths, repo_root, gap_contract):
    topology = contract["requested_topology"]
    if (
        int(topology["operator_count"]) != 79
        or int(topology["eligible_operator_target"]) != 57
        or int(topology["temporal_steps"]) != 10
        or float(topology["maximum_fallback_coefficient_fraction"]) != 0.05
    ):
        raise ValueError("M24 requested topology/coverage gate drift")
    m22 = json.loads(paths["m22.summary"].read_text(encoding="utf-8"))
    m22_output = json.loads(paths["m22.output_manifest"].read_text(encoding="utf-8"))
    m22_input = json.loads(paths["m22.input_manifest"].read_text(encoding="utf-8"))
    if m22.get("status") != M22_STATUS or m22.get("schema") != "m22_ordered_compressed_system_transactions_v2":
        raise ValueError("canonical M22 summary is not admitted")
    if (
        m22_output.get("status") != "FROZEN_REPRODUCIBLE_PARTIAL_LEDGER"
        or m22_output.get("artifacts", {}).get("m22_ordered_transactions.csv", {}).get("sha256")
        != sha256(paths["m22.transactions"])
        or m22_output.get("artifacts", {}).get("m22_summary.json", {}).get("sha256")
        != sha256(paths["m22.summary"])
        or m22_input.get("status") != "FROZEN_EXPECTED_INPUT_IDENTITY"
    ):
        raise ValueError("M22 input/output manifest binding failed")
    identities = {}
    operator_rows = []
    for label in ("h67_ep35", "local_ep44"):
        identity, rows = analyze_tiles(
            label,
            paths["identities.{}.tile_manifest".format(label)],
            paths["identities.{}.tile_records".format(label)],
            paths["identities.{}.packed_tiles".format(label)],
            paths["identities.{}.dual_trace".format(label)],
            int(topology["temporal_steps"]),
        )
        identities[label] = identity
        operator_rows.extend(rows)
    maximum_fallback = max(
        line["fallback_coefficient_fraction"]
        for identity in identities.values()
        for line in identity["line_metrics"].values()
    )
    if maximum_fallback <= float(topology["maximum_fallback_coefficient_fraction"]):
        raise ValueError("frozen sampled inputs unexpectedly satisfy the exhaustive coverage gate")
    resident = analyze_resident_fusion(
        paths["resident_fusion.m18_boundaries"],
        paths["resident_fusion.m21_dse"],
        paths["resident_fusion.m7_candidate"],
        m22,
    )
    contracted_gap = paths.get("gap_contract.exact_temporal_bitmap_gap")
    if contracted_gap is None or Path(gap_contract).resolve() != contracted_gap.resolve():
        raise ValueError("CLI gap contract is not the content-bound M24 gap contract")
    gap = json.loads(contracted_gap.read_text(encoding="utf-8"))
    if gap.get("status") != "BLOCKED_UNTIL_EXHAUSTIVE_STREAMING_COHORT_CENSUS_EXISTS":
        raise ValueError("M24 gap contract is not fail closed")
    payload = {
        "schema": "m24_temporal_cohort_resident_fusion_v1",
        "revision": 1,
        "status": "GAP_EXACT_COEFFICIENT_COVERAGE_LT95_PERCENT_NO_HEADLINE",
        "headline_gate": {
            "requested_eligible_operators": int(topology["eligible_operator_target"]),
            "operator_topology": int(topology["operator_count"]),
            "maximum_fallback_coefficient_fraction": float(topology["maximum_fallback_coefficient_fraction"]),
            "observed_maximum_fallback_coefficient_fraction": maximum_fallback,
            "admitted": False,
        },
        "identities": identities,
        "resident_fusion": resident,
        "amdahl_and_2x": amdahl_audit(topology, identities),
        "input_identity": {
            "input_contract_sha256": contract_sha,
            "m22_summary_sha256": sha256(paths["m22.summary"]),
            "m22_transactions_sha256": sha256(paths["m22.transactions"]),
            "gap_contract_sha256": sha256(gap_contract),
            "source_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "permitted": [
                "exact T10 Local presence and Motion signed-transition mask accounting for frozen sampled rows",
                "sampled coefficient-read/update/control/capacity DSE with same-resource operation envelopes",
                "exact recognition that canonical M22 permits zero additional M4-M21-ATLIF boundary deletion",
                "algebraic Amdahl threshold audit",
            ],
            "forbidden": [
                "full-network, 57/79 eligible, or cross-sequence coefficient reuse",
                "system cycles, latency, FPS, energy, or speedup",
                "using M22 serialized byte-service ticks as cycles",
                "extrapolating four sampled row pairs per call to coefficient populations",
                "claiming resident fusion speedup over the equally composable two-movement baseline",
                "claiming novelty or traffic gain over a generic same-capacity coefficient-resident cache",
                "physical SRAM/macro capacity, timing, energy, or PPA",
            ],
        },
    }
    payload["content_sha256_excluding_this_field"] = canonical_sha256(payload)
    return payload, operator_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--input-contract-sha256", required=True)
    parser.add_argument("--gap-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    contract, contract_sha, paths = load_contract(
        args.input_contract.resolve(), args.input_contract_sha256, repo_root
    )
    gap_contract = args.gap_contract.resolve()
    payload, operator_rows = build(contract, contract_sha, paths, repo_root, gap_contract)
    write_outputs(
        args.output_dir.resolve(), payload, operator_rows,
        args.input_contract.resolve(), gap_contract,
    )
    print(
        "PASS_M24_FAIL_CLOSED h67_exact={} local_exact={} fallback_max={:.9f} "
        "m22_deleted={}".format(
            payload["identities"]["h67_ep35"]["exact_operator_names"],
            payload["identities"]["local_ep44"]["exact_operator_names"],
            payload["headline_gate"]["observed_maximum_fallback_coefficient_fraction"],
            payload["resident_fusion"]["strict_transactions_deleted_from_canonical_m22"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
