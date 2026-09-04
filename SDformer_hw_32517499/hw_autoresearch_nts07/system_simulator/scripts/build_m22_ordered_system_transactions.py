#!/usr/bin/env python3
"""Build an ordered, compressed SRAM/DRAM transaction ledger from real traces.

M22 deliberately stops at an address-assigned transport ledger.  Its logical
timestamps serialize configured byte service and are neither calibrated cycles
nor a DRAMsim3 result.  Product-term reads are compressed repeated-address
transactions; a later adapter must expand them into controller bursts.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


REQUIRED_FILES = (
    "execution_trace.csv",
    "dual_line_operator_trace.csv",
    "operator_runtime.csv",
    "atlif_activity.csv",
    "nts11_hardware_p0_profile.json",
    "dual_line_trace.sha256",
)
EXECUTION_FIELDS = {
    "call_index", "dense_macs", "input_elements", "kind", "name",
    "output_elements", "sample_id", "sample_key", "sequence_key",
}
DUAL_FIELDS = {
    "current_source_count", "local_selected_rows", "local_work",
    "motion_selected_rows", "motion_work", "name",
    "negative_transition_source_count", "operator_call_index",
    "output_channel_fanout", "positive_transition_source_count", "sample_id",
    "selected_work", "selector_rows", "selector_saved_work", "state_valid",
    "status", "temporal_step", "sample_key", "sequence_key", "operator",
    "valid_source_work",
}
ADMITTED_DUAL_STATUS = {
    "PASS_EXACT_SOURCE_WORK", "NON_BINARY_BYPASS", "TEMPORAL_AXIS_UNQUALIFIED",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: {}".format(path))
        return list(reader), set(reader.fieldnames)


def integer(row, key, allow_blank=False):
    value = row.get(key, "")
    if value in ("", None):
        if allow_blank:
            return None
        raise ValueError("missing integer {} in {}".format(key, row.get("name", "record")))
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("invalid integer {}={} in {}".format(key, value, row.get("name", "record")))
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("non-integral {}={} in {}".format(key, value, row.get("name", "record")))
    return int(number)


def truth(value):
    if value not in ("True", "False"):
        raise ValueError("expected canonical Boolean, got {!r}".format(value))
    return value == "True"


def ceil_div(value, divisor):
    if value < 0 or divisor <= 0:
        raise ValueError("invalid ceil_div operands")
    return (value + divisor - 1) // divisor


def parse_sha_receipt(path):
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError("malformed SHA receipt line: {!r}".format(line))
        name = Path(fields[1].strip().lstrip("*")).name
        if name in entries:
            raise ValueError("duplicate SHA receipt basename: {}".format(name))
        entries[name] = fields[0].lower()
    return entries


def load_input_manifest(path, expected_sha256, repo_root):
    path = Path(path)
    actual = sha256(path)
    if len(expected_sha256) != 64 or actual != expected_sha256.lower():
        raise ValueError(
            "input manifest SHA mismatch: expected={} actual={}".format(expected_sha256, actual)
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "m22_ordered_trace_input_manifest_v2"
        or manifest.get("status") != "FROZEN_EXPECTED_INPUT_IDENTITY"
        or not isinstance(manifest.get("identities"), dict)
        or not manifest["identities"]
    ):
        raise ValueError("M22 input manifest schema/status is not admitted")
    producer = manifest.get("producer_source", {})
    producer_path = Path(producer.get("path", ""))
    if (
        producer_path.is_absolute()
        or ".." in producer_path.parts
        or not (Path(repo_root) / producer_path).is_file()
        or sha256(Path(repo_root) / producer_path) != producer.get("sha256")
    ):
        raise ValueError("M22 producer source identity mismatch")
    result = []
    for label, contract in sorted(manifest["identities"].items()):
        relative = Path(contract.get("directory", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("input manifest directory must be repo-relative")
        result.append((label, Path(repo_root) / relative, contract))
    return manifest, actual, result


class AddressArena(object):
    def __init__(self, base, alignment=64):
        self.base = base
        self.cursor = base
        self.alignment = alignment
        self.objects = {}

    def allocate(self, name, size):
        if size <= 0:
            raise ValueError("nonpositive allocation {}={}".format(name, size))
        if name in self.objects:
            if self.objects[name]["bytes"] != size:
                raise ValueError("object size drift: {}".format(name))
            return self.objects[name]["base_address"]
        base = ceil_div(self.cursor, self.alignment) * self.alignment
        self.objects[name] = {"base_address": base, "bytes": size}
        self.cursor = base + size
        return base


def validate_identity(directory, contract):
    directory = Path(directory)
    for name in REQUIRED_FILES:
        if not (directory / name).is_file():
            raise ValueError("missing M22 input: {}".format(directory / name))

    receipt = parse_sha_receipt(directory / "dual_line_trace.sha256")
    expected_files = contract.get("files_sha256", {})
    if set(expected_files) != set(REQUIRED_FILES):
        raise ValueError("input manifest does not bind exactly the required M22 files")
    for name in REQUIRED_FILES:
        actual = sha256(directory / name)
        expected = expected_files.get(name)
        if actual != expected:
            raise ValueError(
                "trusted input manifest mismatch for {}: expected={} actual={}".format(
                    name, expected, actual
                )
            )
    for name in ("execution_trace.csv", "dual_line_operator_trace.csv", "nts11_hardware_p0_profile.json"):
        expected = receipt.get(name)
        if expected is None:
            raise ValueError("SHA receipt does not bind {}".format(name))
        actual = sha256(directory / name)
        if actual != expected:
            raise ValueError("SHA mismatch for {}: expected={} actual={}".format(name, expected, actual))

    profile = json.loads((directory / "nts11_hardware_p0_profile.json").read_text(encoding="utf-8"))
    identity_contract = contract.get("profile_identity", {})
    artifact_identity = profile.get("artifact_identity", {})
    audit = profile.get("checkpoint_load_audit", {})
    audit_zero_fields = (
        "missing_count", "unexpected_count", "overlay_missing_count", "overlay_unexpected_count"
    )
    if (
        profile.get("experiment") != identity_contract.get("experiment")
        or profile.get("samples") != identity_contract.get("samples")
        or profile.get("ordered_trace") is not True
        or profile.get("dual_line_trace") is not True
        or Path(profile.get("checkpoint", "")).name != identity_contract.get("checkpoint_basename")
        or Path(profile.get("config", "")).name != identity_contract.get("config_basename")
        or Path(artifact_identity.get("checkpoint_path", "")).name
        != identity_contract.get("checkpoint_basename")
        or Path(artifact_identity.get("config_path", "")).name
        != identity_contract.get("config_basename")
        or artifact_identity.get("checkpoint_sha256") != identity_contract.get("checkpoint_sha256")
        or artifact_identity.get("config_sha256") != identity_contract.get("config_sha256")
        or any(audit.get(field) != 0 for field in audit_zero_fields)
        or audit.get("checkpoint") != profile.get("checkpoint")
        or audit.get("missing_sample") != []
        or audit.get("unexpected_sample") != []
        or profile.get("module_counts") != identity_contract.get("module_counts")
        or profile.get("eval_protocol") != identity_contract.get("eval_protocol")
    ):
        raise ValueError("profile checkpoint/config/load/eval identity contract mismatch")

    execution, execution_fields = read_csv(directory / "execution_trace.csv")
    dual, dual_fields = read_csv(directory / "dual_line_operator_trace.csv")
    operators, _operator_fields = read_csv(directory / "operator_runtime.csv")
    atlifs, _atlif_fields = read_csv(directory / "atlif_activity.csv")
    if not EXECUTION_FIELDS.issubset(execution_fields):
        raise ValueError("execution trace schema is incomplete")
    if not DUAL_FIELDS.issubset(dual_fields):
        raise ValueError("dual-line trace schema is incomplete")

    sample_ids = sorted(set(integer(row, "sample_id") for row in execution))
    if sample_ids != list(range(len(sample_ids))) or not sample_ids:
        raise ValueError("execution sample ids are not dense from zero")
    samples = len(sample_ids)
    if samples != profile.get("samples"):
        raise ValueError("profile/execution sample count mismatch")
    canonical = None
    execution_by_key = {}
    for sample_id in sample_ids:
        rows = [row for row in execution if integer(row, "sample_id") == sample_id]
        calls = [integer(row, "call_index") for row in rows]
        if calls != list(range(len(rows))):
            raise ValueError("execution call order is not dense for sample {}".format(sample_id))
        sequence = [(row["kind"], row["name"]) for row in rows]
        if canonical is None:
            canonical = sequence
        elif sequence != canonical:
            raise ValueError("execution topology/order changed across samples")
        for row in rows:
            if row["kind"] not in {"operator", "atlif", "attention"}:
                raise ValueError("unsupported execution kind: {}".format(row["kind"]))
            key = (sample_id, row["name"])
            if key in execution_by_key:
                raise ValueError("execution name is not unique within sample: {}".format(key))
            execution_by_key[key] = row
            if row["kind"] != "attention":
                if integer(row, "input_elements") <= 0 or integer(row, "output_elements") <= 0:
                    raise ValueError("nonpositive execution tensor extent")
            elif (
                integer(row, "token_total") <= 0
                or integer(row, "pair_total") <= 0
                or integer(row, "windows") <= 0
            ):
                raise ValueError("attention summary extent is invalid")

    operator_by_name = {}
    for row in operators:
        name = row["name"]
        if name in operator_by_name or integer(row, "calls") != samples:
            raise ValueError("operator runtime identity/call count mismatch: {}".format(name))
        operator_by_name[name] = row
    atlif_by_name = {}
    for row in atlifs:
        name = row["name"]
        if name in atlif_by_name or integer(row, "calls") != samples:
            raise ValueError("ATLIF runtime identity/call count mismatch: {}".format(name))
        if integer(row, "parameter_entries") <= 0 or integer(row, "temporal_steps") <= 0:
            raise ValueError("ATLIF runtime parameter/temporal extent is invalid: {}".format(name))
        atlif_by_name[name] = row

    for kind, runtime, fields in (
        ("operator", operator_by_name, ("dense_macs", "input_elements", "output_elements")),
        ("atlif", atlif_by_name, ("input_elements",)),
    ):
        names = {row["name"] for row in execution if row["kind"] == kind}
        if names != set(runtime):
            raise ValueError("{} execution/runtime name set mismatch".format(kind))
        for name in names:
            rows = [row for row in execution if row["kind"] == kind and row["name"] == name]
            for field in fields:
                runtime_field = "elements" if kind == "atlif" else field
                if sum(integer(row, field) for row in rows) != integer(runtime[name], runtime_field):
                    raise ValueError("{} runtime conservation failed: {}/{}".format(kind, name, field))
            if kind == "operator" and integer(runtime[name], "weight_elements") <= 0:
                raise ValueError("operator has no weight extent: {}".format(name))
            if kind == "atlif":
                for event in rows:
                    if integer(event, "temporal_steps") != integer(runtime[name], "temporal_steps"):
                        raise ValueError("ATLIF execution/runtime temporal extent mismatch: {}".format(name))

    status_counts = Counter(row["status"] for row in dual)
    if not status_counts or not set(status_counts).issubset(ADMITTED_DUAL_STATUS):
        raise ValueError("dual-line status is not admitted")
    dual_samples = sorted(set(integer(row, "sample_id") for row in dual))
    if dual_samples != sample_ids:
        raise ValueError("dual-line sample identity mismatch")
    profile_summary = profile.get("summary", {})
    def profile_record_count(key):
        value = profile_summary.get(key)
        return len(value) if isinstance(value, list) else value
    if (
        len(execution) != profile_record_count("execution_records")
        or len(dual) != profile_record_count("dual_line_records")
        or len(operators) != profile_record_count("operator_rows")
        or len(atlifs) != profile_record_count("atlif_rows")
    ):
        raise ValueError("profile summary trace cardinality mismatch")
    exact_groups = defaultdict(list)
    for row in dual:
        sample_id = integer(row, "sample_id")
        if (sample_id, row["name"]) not in execution_by_key:
            raise ValueError("dual-line row has no execution owner")
        owner = execution_by_key[(sample_id, row["name"])]
        # Producer hook counters are module-local across the profile batch, so
        # the one call from sample N carries operator_call_index=N.
        if (
            owner["kind"] != "operator"
            or integer(row, "operator_call_index") != sample_id
            or row.get("sample_key") != owner.get("sample_key")
            or row.get("sequence_key") != owner.get("sequence_key")
            or row.get("operator") != owner.get("operator")
        ):
            raise ValueError("dual-line row owner/call/identity mismatch")
        if row["status"] != "PASS_EXACT_SOURCE_WORK":
            if integer(row, "temporal_step") != -1:
                raise ValueError("non-exact dual-line row must use temporal_step=-1")
            continue
        fanout = integer(row, "output_channel_fanout")
        current = integer(row, "current_source_count")
        positive = integer(row, "positive_transition_source_count")
        negative = integer(row, "negative_transition_source_count")
        local = integer(row, "local_work")
        motion = integer(row, "motion_work")
        selected = integer(row, "selected_work")
        local_rows = integer(row, "local_selected_rows")
        motion_rows = integer(row, "motion_selected_rows")
        selector_rows = integer(row, "selector_rows")
        saved = integer(row, "selector_saved_work")
        valid_source_work = integer(row, "valid_source_work")
        counts = (
            current, positive, negative, local, motion, selected, local_rows,
            motion_rows, selector_rows, saved, valid_source_work,
        )
        if fanout <= 0 or any(value < 0 for value in counts):
            raise ValueError("dual-line exact counts/fanout are invalid")
        if local != current * fanout or motion != (positive + negative) * fanout:
            raise ValueError("dual-line product work does not conserve")
        if local % fanout or motion % fanout or selected % fanout:
            raise ValueError("dual-line work is not fanout aligned")
        if (
            selected > local
            or local - selected != saved
            or valid_source_work < local
            or valid_source_work % fanout
            or motion > valid_source_work
        ):
            raise ValueError("dual-line selected work is not a valid Local subset")
        if local_rows + motion_rows != selector_rows:
            raise ValueError("dual-line selector rows do not conserve")
        timestep = integer(row, "temporal_step")
        state_valid = truth(row["state_valid"])
        if state_valid != (timestep > 0):
            raise ValueError("dual-line state-valid/timestep mismatch")
        if state_valid:
            if selected > motion:
                raise ValueError("dual-line selected work exceeds Motion candidate")
        elif selected != local or motion_rows != 0:
            raise ValueError("t0/invalid-state row illegally selects Motion")
        exact_groups[(sample_id, row["name"])].append(row)

    for key, rows in exact_groups.items():
        rows.sort(key=lambda row: integer(row, "temporal_step"))
        if [integer(row, "temporal_step") for row in rows] != list(range(len(rows))):
            raise ValueError("dual-line temporal steps are not contiguous: {}".format(key))
        owner = execution_by_key[key]
        output_elements = sum(
            integer(row, "selector_rows") * integer(row, "output_channel_fanout") for row in rows
        )
        if output_elements != integer(owner, "output_elements"):
            raise ValueError("dual-line output geometry does not conserve: {}".format(key))
        if integer(owner, "input_elements") % len(rows):
            raise ValueError("exact operator input cannot be split across temporal steps")
        if (integer(owner, "input_elements") // len(rows)) % 8:
            raise ValueError("exact operator temporal bitmap step is not byte aligned")
        runtime = operator_by_name[key[1]]
        try:
            binary_ratio = Decimal(str(runtime.get("input_sample_binary01_ratio", "")))
        except InvalidOperation:
            binary_ratio = Decimal("NaN")
        sample_elements = integer(runtime, "input_sample_elements")
        # Producer stores this ratio as float32, so a mathematically exact 1.0
        # may arrive one ULP low.  Requiring the implied non-binary population
        # to remain below half an element admits that representation error but
        # rejects even one real non-binary element.
        if (
            not binary_ratio.is_finite()
            or binary_ratio < 0
            or binary_ratio > 1
            or Decimal(1) - binary_ratio > Decimal("0.00000011920928955078125")
            or (Decimal(1) - binary_ratio) * sample_elements >= Decimal("0.5")
        ):
            raise ValueError("exact dual-line operator is not packed-binary qualified")
        geometry = {
            (integer(row, "selector_rows"), integer(row, "output_channel_fanout")) for row in rows
        }
        if len(geometry) != 1:
            raise ValueError("dual-line temporal output geometry changes within operator")

    expected_per_sample = len(exact_groups) // samples
    if not exact_groups or len(exact_groups) != expected_per_sample * samples:
        raise ValueError("dual-line exact group cardinality is not sample-balanced")
    exact_names = None
    for sample_id in sample_ids:
        names = {name for sid, name in exact_groups if sid == sample_id}
        if exact_names is None:
            exact_names = names
        elif names != exact_names:
            raise ValueError("dual-line exact operator coverage changed across samples")

    return {
        "directory": directory,
        "execution": execution,
        "operators": operator_by_name,
        "atlifs": atlif_by_name,
        "exact_groups": exact_groups,
        "sample_count": samples,
        "calls_per_sample": len(canonical),
        "status_counts": dict(status_counts),
        "hashes": {name: sha256(directory / name) for name in REQUIRED_FILES},
        "receipt_entries": receipt,
        "profile_identity": identity_contract,
    }


class Scheduler(object):
    def __init__(self, identity, variant, dram_bytes_per_cycle, sram_bytes_per_cycle,
                 activation_bits, weight_bits, accumulator_bits):
        self.identity = identity
        self.variant = variant
        self.dram_bpc = dram_bytes_per_cycle
        self.sram_bpc = sram_bytes_per_cycle
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits
        self.accumulator_bits = accumulator_bits
        self.dram = AddressArena(0x0000100000000000)
        self.sram = AddressArena(0x0000000080000000)
        self.cycle = 0
        self.rows = []
        self.totals = Counter()

    def emit(self, sample_id, call_index, event_kind, name, temporal_step, tier,
             direction, phase, object_id, object_bytes, byte_count, address_offset=0,
             address_pattern="CONTIGUOUS", evidence_class="TRACE_EXACT"):
        if byte_count <= 0 or object_bytes <= 0 or address_offset < 0:
            raise ValueError("invalid transaction extent")
        arena = self.dram if tier == "DRAM" else self.sram
        base = arena.allocate(object_id, object_bytes)
        if address_offset >= object_bytes:
            raise ValueError("transaction address offset is outside object")
        bandwidth = self.dram_bpc if tier == "DRAM" else self.sram_bpc
        service = ceil_div(byte_count, bandwidth)
        start = self.cycle
        self.cycle += service
        issue_order = len(self.rows)
        self.rows.append({
            "identity": self.identity,
            "variant": self.variant,
            "request_issue_order": issue_order,
            "previous_in_trace_order": issue_order - 1 if issue_order else -1,
            "sample_id": sample_id,
            "call_index": call_index,
            "event_kind": event_kind,
            "name": name,
            "temporal_step": temporal_step,
            "tier": tier,
            "direction": direction,
            "phase": phase,
            "address": "0x{:016x}".format(base + address_offset),
            "object_id": object_id,
            "object_span_bytes": object_bytes,
            "byte_count": byte_count,
            "serialized_service_start": start,
            "serialized_service_end_exclusive": self.cycle,
            "service_bytes_per_cycle": bandwidth,
            "address_pattern": address_pattern,
            "evidence_class": evidence_class,
        })
        self.totals["{}_{}_bytes".format(tier.lower(), direction.lower())] += byte_count
        self.totals["{}_bytes".format(phase)] += byte_count
        self.totals["transactions"] += 1

    def transfer_to_sram(self, sample_id, call_index, event_kind, name, phase,
                         dram_object, sram_object, object_bytes, evidence_class):
        self.emit(sample_id, call_index, event_kind, name, -1, "DRAM", "READ",
                  "dram_{}_read".format(phase), dram_object, object_bytes, object_bytes,
                  evidence_class=evidence_class)
        self.emit(sample_id, call_index, event_kind, name, -1, "SRAM", "WRITE",
                  "sram_{}_fill".format(phase), sram_object, object_bytes, object_bytes,
                  evidence_class=evidence_class)

    def transfer_from_sram(self, sample_id, call_index, event_kind, name, phase,
                           dram_object, sram_object, object_bytes, evidence_class):
        self.emit(sample_id, call_index, event_kind, name, -1, "SRAM", "READ",
                  "sram_{}_drain".format(phase), sram_object, object_bytes, object_bytes,
                  evidence_class=evidence_class)
        self.emit(sample_id, call_index, event_kind, name, -1, "DRAM", "WRITE",
                  "dram_{}_write".format(phase), dram_object, object_bytes, object_bytes,
                  evidence_class=evidence_class)


def schedule_variant(validated, identity, variant, config):
    if variant not in (
        "local_line", "motion_selector_shared_state", "motion_selector_explicit_copy"
    ):
        raise ValueError("unsupported M22 variant")
    scheduler = Scheduler(
        identity, variant, config["dram_bytes_per_cycle"], config["sram_bytes_per_cycle"],
        config["activation_bits"], config["weight_bits"], config["accumulator_bits"],
    )
    weight_bytes_per_element = ceil_div(config["weight_bits"], 8)
    acc_bytes_per_element = ceil_div(config["accumulator_bits"], 8)
    events = sorted(validated["execution"], key=lambda row: (integer(row, "sample_id"), integer(row, "call_index")))
    coverage = Counter()
    for event in events:
        sample_id = integer(event, "sample_id")
        call_index = integer(event, "call_index")
        name = event["name"]
        kind = event["kind"]
        prefix = "{}:s{}:c{}".format(identity, sample_id, call_index)
        if kind == "operator":
            runtime = validated["operators"][name]
            exact_rows = validated["exact_groups"].get((sample_id, name))
            binary_ratio = float(runtime.get("input_sample_binary01_ratio", "-1") or -1)
            input_bits = 1 if binary_ratio >= 0.999 else config["activation_bits"]
            input_bytes = ceil_div(integer(event, "input_elements") * input_bits, 8)
            output_bytes = ceil_div(integer(event, "output_elements") * config["accumulator_bits"], 8)
            weight_bytes = integer(runtime, "weight_elements") * weight_bytes_per_element
            dram_input = "dram:activation_input:{}".format(prefix)
            sram_input = "sram:activation_input:{}".format(prefix)
            dram_weight = "dram:weight:{}".format(name)
            sram_weight = "sram:weight:{}".format(name)
            dram_output = "dram:operator_acc_output:{}".format(prefix)
            sram_output = "sram:operator_acc_output:{}".format(prefix)
            scheduler.transfer_to_sram(sample_id, call_index, kind, name, "activation",
                                       dram_input, sram_input, input_bytes, "EXECUTION_SHAPE")
            scheduler.transfer_to_sram(sample_id, call_index, kind, name, "weight",
                                       dram_weight, sram_weight, weight_bytes, "RUNTIME_WEIGHT_EXTENT")
            if exact_rows:
                coverage["dual_line_exact_operator_events"] += 1
                input_step_bytes = ceil_div(integer(event, "input_elements") // len(exact_rows), 8)
                output_step_bytes_fixed = (
                    integer(exact_rows[0], "selector_rows")
                    * integer(exact_rows[0], "output_channel_fanout") * acc_bytes_per_element
                )
                selector_bytes = ceil_div(
                    integer(exact_rows[0], "selector_rows") * config["selector_bits_per_row"], 8
                )
                motion_enabled = variant != "local_line"
                explicit_copy = variant == "motion_selector_explicit_copy"
                if motion_enabled:
                    scheduler.totals["motion_retained_bitmap_peak_bytes"] = max(
                        scheduler.totals["motion_retained_bitmap_peak_bytes"], input_step_bytes
                    )
                    scheduler.totals["motion_retained_acc_peak_bytes"] = max(
                        scheduler.totals["motion_retained_acc_peak_bytes"], output_step_bytes_fixed
                    )
                    scheduler.totals["motion_selector_peak_bytes"] = max(
                        scheduler.totals["motion_selector_peak_bytes"], selector_bytes
                    )
                    incremental = selector_bytes
                    if explicit_copy:
                        incremental += input_step_bytes + output_step_bytes_fixed
                    scheduler.totals["motion_incremental_state_peak_bytes"] = max(
                        scheduler.totals["motion_incremental_state_peak_bytes"], incremental
                    )
                input_step_offset = 0
                output_step_offset = 0
                for row in exact_rows:
                    timestep = integer(row, "temporal_step")
                    output_step_elements = integer(row, "selector_rows") * integer(row, "output_channel_fanout")
                    output_step_bytes = output_step_elements * acc_bytes_per_element
                    scheduler.emit(sample_id, call_index, kind, name, timestep, "SRAM", "READ",
                                   "source_bitmap_read", sram_input, input_bytes, input_step_bytes,
                                   address_offset=input_step_offset,
                                   evidence_class="DUAL_LINE_TRACE_EXACT")
                    if motion_enabled and timestep > 0:
                        if explicit_copy:
                            previous_bitmap_object = "sram:motion_bitmap_state:{}".format(prefix)
                            previous_bitmap_span = input_step_bytes
                            previous_bitmap_offset = 0
                        else:
                            previous_bitmap_object = sram_input
                            previous_bitmap_span = input_bytes
                            previous_bitmap_offset = input_step_offset - input_step_bytes
                        scheduler.emit(
                            sample_id, call_index, kind, name, timestep, "SRAM", "READ",
                            "motion_previous_bitmap_read", previous_bitmap_object,
                            previous_bitmap_span, input_step_bytes,
                            address_offset=previous_bitmap_offset,
                            evidence_class="DUAL_LINE_TEMPORAL_BITMAP_EXACT",
                        )
                        selector_object = "sram:motion_selector:{}".format(prefix)
                        scheduler.emit(
                            sample_id, call_index, kind, name, timestep, "SRAM", "WRITE",
                            "motion_selector_decision_write", selector_object, selector_bytes,
                            selector_bytes, evidence_class="PACKED_SELECTOR_POLICY",
                        )
                        scheduler.emit(
                            sample_id, call_index, kind, name, timestep, "SRAM", "READ",
                            "motion_selector_decision_read", selector_object, selector_bytes,
                            selector_bytes, evidence_class="PACKED_SELECTOR_POLICY",
                        )
                    work = integer(row, "local_work" if not motion_enabled else "selected_work")
                    if work:
                        scheduler.emit(sample_id, call_index, kind, name, timestep, "SRAM", "READ",
                                       "coefficient_term_read", sram_weight, weight_bytes,
                                       work * weight_bytes_per_element,
                                       address_pattern="CYCLIC_WEIGHT_OBJECT_COMPRESSED",
                                       evidence_class="DUAL_LINE_TRACE_EXACT")
                    if motion_enabled and timestep > 0:
                        previous_bytes = (
                            integer(row, "motion_selected_rows")
                            * integer(row, "output_channel_fanout") * acc_bytes_per_element
                        )
                        if previous_bytes:
                            if explicit_copy:
                                previous_acc_object = "sram:motion_acc_state:{}".format(prefix)
                                previous_acc_span = output_step_bytes_fixed
                                previous_offset = 0
                            else:
                                previous_acc_object = sram_output
                                previous_acc_span = output_bytes
                                previous_offset = output_step_offset - output_step_bytes
                            scheduler.emit(sample_id, call_index, kind, name, timestep, "SRAM", "READ",
                                           "motion_previous_acc_read", previous_acc_object,
                                           previous_acc_span,
                                           previous_bytes, address_offset=previous_offset,
                                           address_pattern="ROW_SELECTED_WITHIN_PREVIOUS_TIMESTEP",
                                           evidence_class="DUAL_LINE_TRACE_EXACT")
                    scheduler.emit(sample_id, call_index, kind, name, timestep, "SRAM", "WRITE",
                                   "operator_acc_write", sram_output, output_bytes, output_step_bytes,
                                   address_offset=output_step_offset,
                                   evidence_class="DUAL_LINE_TRACE_EXACT")
                    if explicit_copy and timestep < len(exact_rows) - 1:
                        bitmap_state = "sram:motion_bitmap_state:{}".format(prefix)
                        acc_state = "sram:motion_acc_state:{}".format(prefix)
                        scheduler.emit(
                            sample_id, call_index, kind, name, timestep, "SRAM", "WRITE",
                            "motion_state_bitmap_copy_write", bitmap_state, input_step_bytes,
                            input_step_bytes, evidence_class="EXPLICIT_COPY_STATE_POLICY",
                        )
                        scheduler.emit(
                            sample_id, call_index, kind, name, timestep, "SRAM", "WRITE",
                            "motion_state_acc_copy_write", acc_state, output_step_bytes_fixed,
                            output_step_bytes_fixed, evidence_class="EXPLICIT_COPY_STATE_POLICY",
                        )
                    input_step_offset += input_step_bytes
                    output_step_offset += output_step_bytes
                if input_step_offset != input_bytes or output_step_offset != output_bytes:
                    raise ValueError("temporal operator extent does not conserve")
            else:
                coverage["dense_fallback_operator_events"] += 1
                scheduler.emit(sample_id, call_index, kind, name, -1, "SRAM", "READ",
                               "activation_dense_read", sram_input, input_bytes, input_bytes,
                               evidence_class="EXECUTION_SHAPE_DENSE_FALLBACK")
                dense_work = integer(event, "dense_macs")
                scheduler.emit(sample_id, call_index, kind, name, -1, "SRAM", "READ",
                               "coefficient_term_read", sram_weight, weight_bytes,
                               dense_work * weight_bytes_per_element,
                               address_pattern="CYCLIC_WEIGHT_OBJECT_COMPRESSED",
                               evidence_class="EXECUTION_DENSE_MAC_FALLBACK")
                scheduler.emit(sample_id, call_index, kind, name, -1, "SRAM", "WRITE",
                               "operator_acc_write", sram_output, output_bytes, output_bytes,
                               evidence_class="EXECUTION_SHAPE_DENSE_FALLBACK")
            scheduler.transfer_from_sram(sample_id, call_index, kind, name, "operator_acc_output",
                                         dram_output, sram_output, output_bytes,
                                         "ALL_BOUNDARIES_MATERIALIZED_POLICY")
        elif kind == "atlif":
            coverage["atlif_events"] += 1
            temporal = integer(event, "temporal_steps")
            if temporal <= 0 or integer(event, "output_elements") % temporal:
                raise ValueError("ATLIF temporal extent is invalid")
            input_bytes = integer(event, "input_elements") * acc_bytes_per_element
            output_bytes = ceil_div(integer(event, "output_elements"), 8)
            state_object_bytes = (
                integer(event, "output_elements") // temporal * acc_bytes_per_element
            )
            state_traffic_bytes = integer(event, "output_elements") * acc_bytes_per_element
            parameter_bytes = integer(validated["atlifs"][name], "parameter_entries") * acc_bytes_per_element
            dram_input = "dram:atlif_acc_input:{}".format(prefix)
            sram_input = "sram:atlif_acc_input:{}".format(prefix)
            sram_state = "sram:atlif_state:{}".format(prefix)
            sram_param = "sram:atlif_param:{}".format(name)
            dram_param = "dram:atlif_param:{}".format(name)
            sram_output = "sram:atlif_bitmap_output:{}".format(prefix)
            dram_output = "dram:atlif_bitmap_output:{}".format(prefix)
            scheduler.transfer_to_sram(sample_id, call_index, kind, name, "atlif_acc_input",
                                       dram_input, sram_input, input_bytes, "EXECUTION_SHAPE")
            scheduler.transfer_to_sram(sample_id, call_index, kind, name, "atlif_parameter",
                                       dram_param, sram_param, parameter_bytes, "ATLIF_RUNTIME_PARAMETER_EXTENT")
            scheduler.emit(sample_id, call_index, kind, name, -1, "SRAM", "READ",
                           "atlif_acc_input_read", sram_input, input_bytes, input_bytes,
                           evidence_class="EXECUTION_SHAPE")
            scheduler.emit(sample_id, call_index, kind, name, -1, "SRAM", "READ",
                           "atlif_state_read", sram_state, state_object_bytes, state_traffic_bytes,
                           address_pattern="TEMPORAL_STATE_REVISIT_COMPRESSED",
                           evidence_class="EXECUTION_TEMPORAL_EXTENT")
            scheduler.emit(sample_id, call_index, kind, name, -1, "SRAM", "WRITE",
                           "atlif_state_write", sram_state, state_object_bytes, state_traffic_bytes,
                           address_pattern="TEMPORAL_STATE_REVISIT_COMPRESSED",
                           evidence_class="EXECUTION_TEMPORAL_EXTENT")
            scheduler.emit(sample_id, call_index, kind, name, -1, "SRAM", "WRITE",
                           "atlif_bitmap_write", sram_output, output_bytes, output_bytes,
                           evidence_class="PACKED1_OUTPUT_POLICY")
            scheduler.transfer_from_sram(sample_id, call_index, kind, name, "atlif_bitmap_output",
                                         dram_output, sram_output, output_bytes,
                                         "ALL_BOUNDARIES_MATERIALIZED_POLICY")
        else:
            coverage["attention_summary_events"] += 1
            token_bytes = ceil_div(integer(event, "token_total") , 8)
            pair_bytes = ceil_div(integer(event, "pair_total"), 8)
            token_object = "sram:attention_token_summary:{}".format(prefix)
            pair_object = "sram:attention_pair_summary:{}".format(prefix)
            scheduler.emit(sample_id, call_index, kind, name, -1, "SRAM", "READ",
                           "attention_token_summary_read", token_object, token_bytes, token_bytes,
                           evidence_class="ABSTRACT_PACKED1_TRACE_COUNTER")
            scheduler.emit(sample_id, call_index, kind, name, -1, "SRAM", "WRITE",
                           "attention_pair_summary_write", pair_object, pair_bytes, pair_bytes,
                           evidence_class="ABSTRACT_PACKED1_TRACE_COUNTER")
    scheduler.totals["serialized_byte_service_ticks"] = scheduler.cycle
    return scheduler.rows, dict(scheduler.totals), coverage, scheduler.dram, scheduler.sram


def build(identities, config, input_manifest_meta):
    payload = {
        "schema": "m22_ordered_compressed_system_transactions_v2",
        "status": "PASS_FROZEN_INPUT_PARTIAL_TRANSACTION_LEDGER_NOT_DRAMSIM_OR_SPEEDUP",
        "generator": {"name": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "input_manifest": input_manifest_meta,
        "config": config,
        "request_order_basis": "TRACE_ORDER_PLUS_PREVIOUS_IN_TRACE_NOT_COMPLETION_DEPENDENCY_OR_ARRIVAL_TIME",
        "service_estimate_basis": "SERIALIZED_BYTE_SERVICE_NO_OVERLAP_NOT_CALIBRATED_CYCLES",
        "address_basis": "DETERMINISTIC_LOGICAL_OBJECT_IMAGE_NOT_PHYSICAL_SRAM_CAPACITY_MAPPING",
        "identities": {},
        "claim_boundary": {
            "permitted": [
                "manifest-frozen ten-sample execution order and strengthened aggregate dual-line conservation",
                "deterministic logical SRAM/DRAM object addresses, directions, and byte extents",
                "Local versus Motion shared/copy state transport ledgers including previous bitmap, previous Acc32, selector bits, and aggregate product-term counts",
            ],
            "forbidden": [
                "system latency, FPS, energy, or speedup",
                "DRAMsim3 readiness/timing, CACTI energy, bank conflicts, cache hits, or burst timing",
                "physical SRAM capacity/placement, tensor liveness, fusion, or cross-call aliasing",
                "selector/popcount compute latency, tags, or control energy",
                "attention physical traffic (H67 rows are packed1 summaries; Local trace has no attention execution rows)",
                "physical bank-address trace for compressed cyclic/row-selected accesses",
                "cross-sequence, event-density/equal-rate, or full-DSEC generalization",
            ],
            "required_next": [
                "derive validated issue/ready dependencies and expand compressed patterns into aligned controller bursts",
                "run DRAMsim3 and calibrate SRAM banks/ports plus compute issue against RTL/Synopsys timing",
                "replace operator/ATLIF-boundary materialization with validated liveness/fusion/residency",
                "bind attention summaries to an RTL-exact physical memory schedule",
            ],
        },
    }
    all_rows = []
    for label, directory, contract in identities:
        validated = validate_identity(directory, contract)
        attention_records = sum(row["kind"] == "attention" for row in validated["execution"])
        identity = {
            "source_directory": str(Path(directory).resolve()),
            "input_hashes": validated["hashes"],
            "sha_receipt_entries": validated["receipt_entries"],
            "profile_identity": validated["profile_identity"],
            "sample_count": validated["sample_count"],
            "calls_per_sample": validated["calls_per_sample"],
            "execution_records": len(validated["execution"]),
            "attention_execution_records": attention_records,
            "attention_coverage_status": (
                "ABSTRACT_PACKED1_COUNTER_SUMMARY_NOT_PHYSICAL_TRAFFIC" if attention_records
                else "MISSING_FROM_EXECUTION_TRACE_NOT_ZERO_COST"
            ),
            "dual_line_status_counts": validated["status_counts"],
            "exact_dual_line_operator_groups": len(validated["exact_groups"]),
            "variants": {},
            "motion_models_vs_local": {},
        }
        variants = (
            "local_line", "motion_selector_shared_state", "motion_selector_explicit_copy"
        )
        for variant in variants:
            rows, totals, coverage, dram, sram = schedule_variant(validated, label, variant, config)
            all_rows.extend(rows)
            identity["variants"][variant] = {
                "totals": totals,
                "coverage": dict(coverage),
                "dram_address_object_count": len(dram.objects),
                "dram_logical_span_bytes": dram.cursor - dram.base,
                "sram_address_object_count": len(sram.objects),
                "sram_logical_span_bytes": sram.cursor - sram.base,
                "dram_address_map_sha256": hashlib.sha256(
                    json.dumps(dram.objects, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "sram_address_map_sha256": hashlib.sha256(
                    json.dumps(sram.objects, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            }
        local = identity["variants"]["local_line"]
        local_totals = local["totals"]
        for variant in variants[1:]:
            motion = identity["variants"][variant]
            motion_totals = motion["totals"]
            for field in ("dram_read_bytes", "dram_write_bytes"):
                if local_totals.get(field, 0) != motion_totals.get(field, 0):
                    raise ValueError("variant DRAM fairness failed for {} {}".format(label, field))
            if local["dram_address_map_sha256"] != motion["dram_address_map_sha256"]:
                raise ValueError("variant DRAM address-object map changed for {}".format(label))
            if local["coverage"] != motion["coverage"]:
                raise ValueError("variant execution coverage changed for {}".format(label))
            coefficient_delta = (
                motion_totals.get("coefficient_term_read_bytes", 0)
                - local_totals.get("coefficient_term_read_bytes", 0)
            )
            read_components = (
                coefficient_delta
                + motion_totals.get("motion_previous_bitmap_read_bytes", 0)
                + motion_totals.get("motion_previous_acc_read_bytes", 0)
                + motion_totals.get("motion_selector_decision_read_bytes", 0)
            )
            write_components = (
                motion_totals.get("motion_selector_decision_write_bytes", 0)
                + motion_totals.get("motion_state_bitmap_copy_write_bytes", 0)
                + motion_totals.get("motion_state_acc_copy_write_bytes", 0)
            )
            sram_read_delta = (
                motion_totals.get("sram_read_bytes", 0) - local_totals.get("sram_read_bytes", 0)
            )
            sram_write_delta = (
                motion_totals.get("sram_write_bytes", 0) - local_totals.get("sram_write_bytes", 0)
            )
            if sram_read_delta != read_components or sram_write_delta != write_components:
                raise ValueError("variant SRAM delta does not reconcile for {} {}".format(label, variant))
            local_ticks = local_totals["serialized_byte_service_ticks"]
            motion_ticks = motion_totals["serialized_byte_service_ticks"]
            identity["motion_models_vs_local"][variant] = {
                "dram_traffic_and_address_map_identical": True,
                "sram_read_byte_delta": sram_read_delta,
                "sram_write_byte_delta": sram_write_delta,
                "coefficient_term_read_byte_delta": coefficient_delta,
                "motion_previous_bitmap_read_bytes": motion_totals.get("motion_previous_bitmap_read_bytes", 0),
                "motion_previous_acc_read_bytes": motion_totals.get("motion_previous_acc_read_bytes", 0),
                "motion_selector_read_write_bytes": (
                    motion_totals.get("motion_selector_decision_read_bytes", 0)
                    + motion_totals.get("motion_selector_decision_write_bytes", 0)
                ),
                "motion_state_copy_write_bytes": (
                    motion_totals.get("motion_state_bitmap_copy_write_bytes", 0)
                    + motion_totals.get("motion_state_acc_copy_write_bytes", 0)
                ),
                "serialized_byte_service_tick_delta": motion_ticks - local_ticks,
                "serialized_byte_service_fractional_change": motion_ticks / local_ticks - 1.0,
                "sram_delta_reconciles": True,
                "claim": "TRANSPORT_LEDGER_DELTA_ONLY_NOT_SYSTEM_SPEEDUP",
            }
        payload["identities"][label] = identity
    for index, row in enumerate(all_rows):
        row["transaction_id"] = index
    return payload, all_rows


def write_outputs(output, payload, rows, input_manifest_path):
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "m22_ordered_transactions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["transaction_id"] + [key for key in rows[0] if key != "transaction_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload["transaction_records"] = len(rows)
    payload["transactions_sha256"] = sha256(csv_path)
    summary_path = output / "m22_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# M22 ordered compressed SRAM/DRAM transaction ledger\n\n",
        "This milestone assigns deterministic logical addresses, trace order, read/write direction, byte extents, and a separate serialized byte-service estimate to manifest-frozen ten-sample traces.\n\n",
        "| identity | attention coverage | variant | records | DRAM read B | DRAM write B | SRAM read B | SRAM write B | serialized service ticks |\n",
        "|---|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    comparisons = []
    for label, identity in payload["identities"].items():
        for variant, result in identity["variants"].items():
            totals = result["totals"]
            report.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                    label, identity["attention_coverage_status"], variant,
                    totals["transactions"], totals.get("dram_read_bytes", 0),
                    totals.get("dram_write_bytes", 0), totals.get("sram_read_bytes", 0),
                    totals.get("sram_write_bytes", 0), totals["serialized_byte_service_ticks"],
                )
            )
        for variant, comparison in identity["motion_models_vs_local"].items():
            comparisons.append(
                "{} {} transport delta: SRAM read {:+d} B, SRAM write {:+d} B; "
                "serialized byte-service {:+d} ({:+.4%}). This is not a speedup.\n\n".format(
                    label, variant, comparison["sram_read_byte_delta"],
                    comparison["sram_write_byte_delta"],
                    comparison["serialized_byte_service_tick_delta"],
                    comparison["serialized_byte_service_fractional_change"],
                )
            )
    report.append("\n")
    report.extend(comparisons)
    report.append(
        "\nThe service ticks are a serialized byte-service estimate, not request arrival or system cycles. "
        "The CSV still requires validated dependencies, physical allocation, burst expansion, DRAMsim3, SRAM bank/port calibration, "
        "selector compute/control, liveness/fusion, and an RTL-exact attention schedule before any latency, energy, FPS, or speedup claim.\n"
    )
    report_path = output / "REPORT.md"
    report_path.write_text("".join(report), encoding="utf-8")
    test_source = Path(__file__).resolve().parents[1] / "tests/test_m22_ordered_system_transactions.py"
    output_manifest = {
        "schema": "m22_ordered_transaction_output_manifest_v2",
        "status": "FROZEN_REPRODUCIBLE_PARTIAL_LEDGER",
        "input_manifest": {
            "path": str(input_manifest_path),
            "sha256": payload["input_manifest"]["sha256"],
        },
        "sources_sha256": {
            Path(__file__).name: sha256(Path(__file__)),
            test_source.name: sha256(test_source),
        },
        "config": payload["config"],
        "artifacts": {
            csv_path.name: {"sha256": sha256(csv_path), "bytes": csv_path.stat().st_size},
            summary_path.name: {"sha256": sha256(summary_path), "bytes": summary_path.stat().st_size},
            report_path.name: {"sha256": sha256(report_path), "bytes": report_path.stat().st_size},
        },
        "transaction_records": len(rows),
        "claim": "REPRODUCIBLE_LOGICAL_LEDGER_NOT_DRAMSIM_OR_SYSTEM_SPEEDUP",
    }
    (output / "m22_output_manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dram-bytes-per-cycle", type=int, default=192)
    parser.add_argument("--sram-bytes-per-cycle", type=int, default=96)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--accumulator-bits", type=int, default=32)
    parser.add_argument("--selector-bits-per-row", type=int, default=1)
    args = parser.parse_args()
    config = {
        "dram_bytes_per_cycle": args.dram_bytes_per_cycle,
        "sram_bytes_per_cycle": args.sram_bytes_per_cycle,
        "activation_bits": args.activation_bits,
        "weight_bits": args.weight_bits,
        "accumulator_bits": args.accumulator_bits,
        "selector_bits_per_row": args.selector_bits_per_row,
        "boundary_policy": "OPERATOR_AND_ATLIF_BOUNDARIES_MATERIALIZED_ATTENTION_EXCLUDED",
        "motion_policy": "AGGREGATE_ROW_SELECTOR_WITH_SHARED_AND_EXPLICIT_COPY_STATE_TRANSPORT",
    }
    if min(config[key] for key in (
        "dram_bytes_per_cycle", "sram_bytes_per_cycle", "activation_bits", "weight_bits",
        "accumulator_bits", "selector_bits_per_row"
    )) <= 0:
        raise ValueError("M22 configuration values must be positive")
    if config["weight_bits"] % 8 or config["accumulator_bits"] % 8:
        raise ValueError("weight and accumulator widths must be byte aligned")
    _manifest, manifest_sha, identities = load_input_manifest(
        args.input_manifest, args.expected_input_manifest_sha256, args.repo_root
    )
    manifest_meta = {
        "path": str(args.input_manifest),
        "sha256": manifest_sha,
        "expected_sha256_from_cli": args.expected_input_manifest_sha256.lower(),
    }
    payload, rows = build(identities, config, manifest_meta)
    write_outputs(args.output, payload, rows, args.input_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
