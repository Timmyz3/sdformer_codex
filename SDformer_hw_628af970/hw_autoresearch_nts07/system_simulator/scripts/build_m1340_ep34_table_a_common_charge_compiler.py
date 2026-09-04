#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Build same-denominator Table-A rows without hiding uncaptured operators.

The final ep34 capture retains exact payloads for only a subset of the network.
This compiler therefore separates every population point into:

* direct branches, whose row-dependent charges come from executable replay; and
* common operators, whose conservative charge is added unchanged to all rows.

The program is deliberately fail closed.  It will not emit a production
candidate unless the final ep34 identity, all 30 population points, all three
direct branches, all nine common operator classes, the 17-SRAM resource tuple,
and a >=95% native mapped-activity energy authority are present and hash bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "m1340.ep34.table_a.common_charge.compiler.r1"
CHARGE_SCHEMA = "m1340.table_a.charge_population.r1"
ENERGY_SCHEMA = "m1340.table_a.energy_authority.r1"
PROTECTED_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
ROWS = ("B0", "B1", "B2", "B3", "C2", "Ours")
DIRECT_BRANCHES = ("c1_bottleneck", "decoder", "attention")
COMMON_CATEGORIES = (
    "patch_embed",
    "other_conv_projection",
    "fc1",
    "dynamic_bn",
    "atlif",
    "attention_projection_completion",
    "fc2",
    "prediction_head",
    "preprocess_completion",
)
SRAM_MACROS = tuple(["weight_%02d" % i for i in range(8)] +
                    ["state_%02d" % i for i in range(8)] + ["parent_00"])
RESOURCE = {
    "process_nm": 28,
    "clock_period_ns": 3.0,
    "source_lanes": 96,
    "accumulator_bits": 24,
    "local_sram_bytes": 240 * 1024,
    "dram_bytes_per_cycle": 192,
    "dram_bandwidth_gbps_decimal": 64,
    "weight_banks": 8,
    "state_banks": 8,
    "parent_banks": 1,
    "sram_port_mode": "1RW_no_same_bank_read_write",
    "external_read_ports_per_bank": 1,
    "external_write_ports_per_bank": 1,
    "group_fifo_depth": 4,
    "outstanding_weight_requests": 8,
    "sram_macro_ids": list(SRAM_MACROS),
}


class CompileError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs_no_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CompileError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=_pairs_no_duplicates,
                          parse_constant=lambda token: (_ for _ in ()).throw(
                              CompileError("non-finite JSON number: %s" % token)))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompileError("cannot read strict JSON %s: %s" % (path, exc))


def exact_object(value: Any, fields: Iterable[str], label: str) -> dict[str, Any]:
    expected = set(fields)
    if not isinstance(value, dict) or set(value) != expected:
        raise CompileError("%s fields differ: expected=%s actual=%s" %
                           (label, sorted(expected),
                            sorted(value) if isinstance(value, dict) else type(value).__name__))
    return value


def finite_nonnegative_integer(value: Any, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompileError("%s must be an integer" % label)
    if value < (1 if positive else 0):
        raise CompileError("%s out of range" % label)
    return value


def finite_nonnegative_number(value: Any, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompileError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result) or result < (0.0 if not positive else 1e-300):
        raise CompileError("%s out of range" % label)
    return result


def resolve_spec(root: Path, spec: Any, label: str,
                 allowed_media: tuple[str, ...] = ("application/json",)) -> tuple[Path, dict[str, Any]]:
    spec = exact_object(spec, ("path", "sha256", "media_type"), label)
    if spec["media_type"] not in allowed_media:
        raise CompileError("%s media_type is not admitted" % label)
    rel = Path(spec["path"])
    if rel.is_absolute() or ".." in rel.parts:
        raise CompileError("%s path escapes workspace" % label)
    path = root / rel
    try:
        path.resolve().relative_to(root)
    except (OSError, ValueError):
        raise CompileError("%s resolves outside workspace" % label)
    try:
        st = path.lstat()
    except OSError as exc:
        raise CompileError("%s missing: %s" % (label, exc))
    if path.is_symlink() or not path.is_file() or st.st_nlink != 1:
        raise CompileError("%s must be a single-link regular file" % label)
    if st.st_mode & 0o222:
        raise CompileError("%s must be read-only" % label)
    if not isinstance(spec["sha256"], str) or len(spec["sha256"]) != 64:
        raise CompileError("%s SHA grammar invalid" % label)
    actual = sha256(path)
    if actual != spec["sha256"]:
        raise CompileError("%s SHA drift" % label)
    return path, spec


def population_key(sequence_id: str, sample_id: int) -> str:
    return "%s::%02d" % (sequence_id, sample_id)


def validate_population(value: Any) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if not isinstance(value, list) or len(value) != 30:
        raise CompileError("population must contain exactly 30 points")
    points: list[dict[str, Any]] = []
    sequences: dict[str, list[int]] = {}
    seen: set[str] = set()
    for index, raw in enumerate(value):
        row = exact_object(raw, ("sequence_id", "sample_id", "density_stratum", "weight"),
                           "population[%d]" % index)
        sequence = row["sequence_id"]
        if not isinstance(sequence, str) or not sequence or sequence.strip() != sequence:
            raise CompileError("population sequence_id invalid")
        sample = finite_nonnegative_integer(row["sample_id"], "sample_id")
        if row["density_stratum"] not in ("low", "medium", "high"):
            raise CompileError("density_stratum invalid")
        weight = finite_nonnegative_number(row["weight"], "population weight", positive=True)
        key = population_key(sequence, sample)
        if key in seen:
            raise CompileError("duplicate population point: %s" % key)
        seen.add(key)
        sequences.setdefault(sequence, []).append(sample)
        points.append({**row, "weight": weight, "key": key})
    if len(sequences) != 3 or any(sorted(samples) != list(range(10))
                                  for samples in sequences.values()):
        raise CompileError("population must be three sequences with sample_id 0..9")
    if {row["density_stratum"] for row in points} != {"low", "medium", "high"}:
        raise CompileError("all density strata must be populated")
    total_weight = sum(row["weight"] for row in points)
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise CompileError("population weights must sum to one")
    return points, tuple(sorted(sequences))


def empty_sram() -> dict[str, dict[str, int]]:
    return {name: {"read_bytes": 0, "write_bytes": 0} for name in SRAM_MACROS}


def validate_charge(raw: Any, label: str) -> dict[str, Any]:
    row = exact_object(raw, ("cycles", "fixed_numerator", "dram_read_bytes",
                             "dram_write_bytes", "sram_bytes"), label)
    cycles = finite_nonnegative_integer(row["cycles"], label + ".cycles", positive=True)
    numerator = finite_nonnegative_integer(row["fixed_numerator"],
                                           label + ".fixed_numerator", positive=True)
    dram_read = finite_nonnegative_integer(row["dram_read_bytes"],
                                           label + ".dram_read_bytes")
    dram_write = finite_nonnegative_integer(row["dram_write_bytes"],
                                            label + ".dram_write_bytes")
    if not isinstance(row["sram_bytes"], dict) or set(row["sram_bytes"]) != set(SRAM_MACROS):
        raise CompileError("%s must account for all 17 SRAM macros" % label)
    sram: dict[str, dict[str, int]] = {}
    for macro in SRAM_MACROS:
        access = exact_object(row["sram_bytes"][macro], ("read_bytes", "write_bytes"),
                              "%s.%s" % (label, macro))
        sram[macro] = {
            "read_bytes": finite_nonnegative_integer(access["read_bytes"],
                                                       label + ".sram.read"),
            "write_bytes": finite_nonnegative_integer(access["write_bytes"],
                                                        label + ".sram.write"),
        }
    return {
        "cycles": cycles,
        "fixed_numerator": numerator,
        "dram_read_bytes": dram_read,
        "dram_write_bytes": dram_write,
        "sram_bytes": sram,
    }


def read_charge_file(root: Path, spec: Any, expected_kind: str,
                     expected_name: str, population_keys: set[str]) -> dict[str, dict[str, Any]]:
    path, _ = resolve_spec(root, spec, "%s charge" % expected_name)
    payload = exact_object(load_json(path),
                           ("schema", "kind", "name", "identity", "population"),
                           "%s payload" % expected_name)
    if payload["schema"] != CHARGE_SCHEMA or payload["kind"] != expected_kind or \
            payload["name"] != expected_name or payload["identity"] != "Motion-C12-ep34-final":
        raise CompileError("%s charge identity mismatch" % expected_name)
    rows = payload["population"]
    if not isinstance(rows, dict) or set(rows) != population_keys:
        raise CompileError("%s population coverage mismatch" % expected_name)
    return {key: validate_charge(rows[key], "%s[%s]" % (expected_name, key))
            for key in sorted(rows)}


def add_charge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for field in ("cycles", "fixed_numerator", "dram_read_bytes", "dram_write_bytes"):
        target[field] += source[field]
    for macro in SRAM_MACROS:
        target["sram_bytes"][macro]["read_bytes"] += source["sram_bytes"][macro]["read_bytes"]
        target["sram_bytes"][macro]["write_bytes"] += source["sram_bytes"][macro]["write_bytes"]


def new_charge() -> dict[str, Any]:
    return {"cycles": 0, "fixed_numerator": 0, "dram_read_bytes": 0,
            "dram_write_bytes": 0, "sram_bytes": empty_sram()}


def validate_energy(root: Path, spec: Any) -> dict[str, Any]:
    path, _ = resolve_spec(root, spec, "energy authority")
    payload = exact_object(load_json(path),
                           ("schema", "identity", "native_mapped_activity_coverage",
                            "logic_pj_per_cycle", "dram_pj_per_byte", "sram_pj_per_byte"),
                           "energy authority payload")
    if payload["schema"] != ENERGY_SCHEMA or payload["identity"] != "Motion-C12-ep34-final":
        raise CompileError("energy identity mismatch")
    coverage = finite_nonnegative_number(payload["native_mapped_activity_coverage"],
                                         "mapped activity coverage")
    if coverage < 0.95 or coverage > 1.0:
        raise CompileError("native mapped activity coverage must be in [0.95,1]")
    if not isinstance(payload["logic_pj_per_cycle"], dict) or \
            set(payload["logic_pj_per_cycle"]) != set(ROWS):
        raise CompileError("energy logic rows differ")
    logic = {row: finite_nonnegative_number(payload["logic_pj_per_cycle"][row],
                                             "logic energy", positive=True)
             for row in ROWS}
    dram = exact_object(payload["dram_pj_per_byte"], ("read", "write"), "dram energy")
    dram = {key: finite_nonnegative_number(value, "dram energy", positive=True)
            for key, value in dram.items()}
    if not isinstance(payload["sram_pj_per_byte"], dict) or \
            set(payload["sram_pj_per_byte"]) != set(SRAM_MACROS):
        raise CompileError("energy authority must cover all SRAM macros")
    sram: dict[str, dict[str, float]] = {}
    for macro in SRAM_MACROS:
        rates = exact_object(payload["sram_pj_per_byte"][macro], ("read", "write"),
                             "sram energy %s" % macro)
        sram[macro] = {key: finite_nonnegative_number(value, "sram energy", positive=True)
                       for key, value in rates.items()}
    return {"coverage": coverage, "logic": logic, "dram": dram, "sram": sram}


def charge_energy(charge: dict[str, Any], row_id: str, authority: dict[str, Any]) -> dict[str, float]:
    logic = charge["cycles"] * authority["logic"][row_id]
    dram = (charge["dram_read_bytes"] * authority["dram"]["read"] +
            charge["dram_write_bytes"] * authority["dram"]["write"])
    sram = 0.0
    for macro in SRAM_MACROS:
        sram += (charge["sram_bytes"][macro]["read_bytes"] * authority["sram"][macro]["read"] +
                 charge["sram_bytes"][macro]["write_bytes"] * authority["sram"][macro]["write"])
    return {"logic_pj": logic, "sram_pj": sram, "dram_pj": dram,
            "total_pj": logic + sram + dram}


def build(config_path: Path, workspace_root: Path) -> dict[str, Any]:
    root = workspace_root.resolve()
    config = exact_object(load_json(config_path),
                          ("schema", "status", "identity", "resource", "population", "rows",
                           "common_operators", "direct_branches", "energy_authority",
                           "claim_boundary", "protected_file"), "compiler config")
    if config["schema"] != SCHEMA or config["status"] not in (
            "SOURCE_FIXTURE", "PRODUCTION_CANDIDATE"):
        raise CompileError("compiler schema/status mismatch")
    identity = exact_object(config["identity"],
                            ("name", "checkpoint", "config", "profile", "capture_result",
                             "capture_result_hammer"), "identity")
    if identity["name"] != "Motion-C12-ep34-final":
        raise CompileError("final identity name mismatch")
    identity_media = {
        "checkpoint": ("application/octet-stream",),
        "config": ("application/yaml", "application/json"),
        "profile": ("application/json", "text/csv"),
        "capture_result": ("application/json",),
        "capture_result_hammer": ("application/json",),
    }
    for field in ("checkpoint", "config", "profile", "capture_result", "capture_result_hammer"):
        resolve_spec(root, identity[field], "identity.%s" % field, identity_media[field])
    if config["resource"] != RESOURCE or config["rows"] != list(ROWS):
        raise CompileError("resource or row tuple drift")
    protected_path, protected_spec = resolve_spec(
        root, config["protected_file"], "protected file", ("text/markdown",))
    if protected_spec["sha256"] != PROTECTED_SHA256 or protected_path.name != \
            "359_DATE终局冻结_20260813.md":
        raise CompileError("protected docs/359 identity mismatch")
    points, sequences = validate_population(config["population"])
    keys = {point["key"] for point in points}
    common_specs = config["common_operators"]
    if not isinstance(common_specs, dict) or set(common_specs) != set(COMMON_CATEGORIES):
        raise CompileError("common operator categories differ")
    common = {name: read_charge_file(root, common_specs[name], "common", name, keys)
              for name in COMMON_CATEGORIES}
    branch_specs = config["direct_branches"]
    if not isinstance(branch_specs, dict) or set(branch_specs) != set(DIRECT_BRANCHES):
        raise CompileError("direct branch set differs")
    branches: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for branch in DIRECT_BRANCHES:
        row_specs = branch_specs[branch]
        if not isinstance(row_specs, dict) or set(row_specs) != set(ROWS):
            raise CompileError("%s must contain all six rows" % branch)
        branches[branch] = {
            row: read_charge_file(root, row_specs[row], "direct", "%s.%s" % (branch, row), keys)
            for row in ROWS
        }
    authority = validate_energy(root, config["energy_authority"])
    if config["claim_boundary"] != {
            "same_denominator": True,
            "common_charge_identical_all_rows": True,
            "component_speedups_not_multiplied": True,
            "external_prosperity_not_ours": True,
            "independent_hammer_required": True,
            "paper_headline_admitted": False}:
        raise CompileError("claim boundary drift")
    output_rows: list[dict[str, Any]] = []
    common_totals: dict[str, dict[str, Any]] = {}
    for point in points:
        key = point["key"]
        total = new_charge()
        for category in COMMON_CATEGORIES:
            add_charge(total, common[category][key])
        common_totals[key] = total
    for row_id in ROWS:
        aggregate = new_charge()
        per_population = []
        weighted_cycles = 0.0
        weighted_numerator = 0.0
        weighted_energy = {"logic_pj": 0.0, "sram_pj": 0.0,
                           "dram_pj": 0.0, "total_pj": 0.0}
        for point in points:
            key = point["key"]
            charge = new_charge()
            add_charge(charge, common_totals[key])
            for branch in DIRECT_BRANCHES:
                add_charge(charge, branches[branch][row_id][key])
            if charge["fixed_numerator"] <= 0 or charge["cycles"] <= 0:
                raise CompileError("zero denominator or cycles after aggregation")
            add_charge(aggregate, charge)
            weighted_cycles += point["weight"] * charge["cycles"]
            weighted_numerator += point["weight"] * charge["fixed_numerator"]
            point_energy = charge_energy(charge, row_id, authority)
            for field in weighted_energy:
                weighted_energy[field] += point["weight"] * point_energy[field]
            per_population.append({"key": key, "sequence_id": point["sequence_id"],
                                   "sample_id": point["sample_id"],
                                   "density_stratum": point["density_stratum"],
                                   "weight": point["weight"], "charge": charge,
                                   "energy": point_energy})
        output_rows.append({"row_id": row_id, "aggregate": aggregate,
                            "weighted_cycles": weighted_cycles,
                            "weighted_fixed_numerator": weighted_numerator,
                            "per_population": per_population,
                            "aggregate_energy": charge_energy(aggregate, row_id, authority),
                            "weighted_energy": weighted_energy})
    baseline_cycles = output_rows[0]["weighted_cycles"]
    baseline_energy = output_rows[0]["weighted_energy"]["total_pj"]
    baseline_numerator = output_rows[0]["weighted_fixed_numerator"]
    for row in output_rows:
        if not math.isclose(row["weighted_fixed_numerator"], baseline_numerator,
                            rel_tol=0.0, abs_tol=1e-9):
            raise CompileError("fixed numerator differs across rows")
        row["speedup_vs_B0"] = baseline_cycles / row["weighted_cycles"]
        row["energy_reduction_vs_B0"] = 1.0 - row["weighted_energy"]["total_pj"] / baseline_energy
    return {
        "schema": "m1340.ep34.table_a.common_charge.output.r1",
        "status": ("PASS_PRODUCTION_CANDIDATE_UNHAMMERED" if
                   config["status"] == "PRODUCTION_CANDIDATE" else
                   "PASS_SOURCE_FIXTURE_NOT_PRODUCTION"),
        "identity": identity["name"],
        "resource": RESOURCE,
        "population": {"points": len(points), "sequences": list(sequences),
                       "density_strata": ["low", "medium", "high"],
                       "weight_sum": sum(point["weight"] for point in points)},
        "common_operator_categories": list(COMMON_CATEGORIES),
        "direct_branches": list(DIRECT_BRANCHES),
        "energy_authority": {"native_mapped_activity_coverage": authority["coverage"],
                             "sram_macro_count": len(SRAM_MACROS), "dram_included": True},
        "rows": output_rows,
        "claim_boundary": {**config["claim_boundary"],
                           "paper_headline_admitted": False,
                           "requires_fresh_independent_bundle_hammer": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(args.config, args.workspace_root)
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                             allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(str(args.output), flags, 0o444)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        print("M1340_TABLE_A_COMMON_CHARGE_PASS status=%s rows=6 population=30 "
              "sram_macros=17 headline=false" % result["status"])
        return 0
    except (CompileError, OSError, ValueError) as exc:
        print("M1340_TABLE_A_COMMON_CHARGE_FAIL: %s" % exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
