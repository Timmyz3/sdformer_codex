#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1158D3 static weight-fit bridge-inclusive CPU fast-kill.

This does not reuse the M1156 accumulator candidate ledger.  It independently
replays the frozen D3 bitpack through the exact M672 inverse mapper, counts
bank-conflict-aware K8 groups, and combines those counts with the already
sealed M712 A1/weight-cache ledger.  D0-D2 are statically fixed to A1-OSG.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, getcontext
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile

import numpy as np

getcontext().prec = 40
RESULT_SCHEMA = "m1158d3_static_weight_fit_bridge_fastkill_result_v1"
CONTRACT_SCHEMA = "m1158d3_static_weight_fit_bridge_fastkill_contract_v1"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
LAYERS = ("D0", "D1", "D2", "D3")


class Failure(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Failure(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON token: " + token)))


def safe_member(name):
    member = PurePosixPath(name)
    require(member.parts and not member.is_absolute() and ".." not in member.parts and
            member.as_posix() == name, "unsafe member: " + name)
    return member


def verify_regular(path, expected=None):
    path = Path(path)
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink(),
            "not a regular non-symlink: " + str(path))
    observed = sha256(path)
    if expected is not None:
        require(observed == expected, "SHA drift: " + str(path))
    return observed


def verify_sealed_directory(path, expected_outer):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "bad sealed directory")
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    verify_regular(outer, expected_outer)
    manifest_sha, name = outer.read_text(encoding="utf-8").split()
    require(name == "SHA256SUMS", "outer seal target")
    verify_regular(manifest, manifest_sha)
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        safe_member(name)
        require(name not in listed, "duplicate sealed member")
        listed[name] = digest
    actual = set()
    for member in path.rglob("*"):
        require(not member.is_symlink(), "symlink in sealed directory")
        if member.is_file() and member.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(member.relative_to(path).as_posix())
    require(actual == set(listed), "sealed member topology drift")
    for name, digest in listed.items():
        verify_regular(path.joinpath(*PurePosixPath(name).parts), digest)
    return manifest_sha


def verify_double(path):
    path = Path(path)
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    contract_sha = verify_regular(path)
    side_sha = verify_regular(side)
    outer_sha = verify_regular(outer)
    require(side.read_text(encoding="utf-8").split() == [contract_sha, path.name] and
            outer.read_text(encoding="utf-8").split() == [side_sha, side.name],
            "contract double seal")
    return contract_sha, side_sha, outer_sha


def ratio(numerator, denominator):
    require(int(denominator) > 0, "zero ratio denominator")
    return format(Decimal(int(numerator)) / Decimal(int(denominator)), ".12f")


def load_module(path, expected_sha, name):
    verify_regular(path, expected_sha)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def aggregate_m712_sample0(rows_path):
    aggregate = defaultdict(lambda: defaultdict(int))
    count = 0
    with Path(rows_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["sequence"] != "interlaken_01_a" or row["sequence_sample_id"] != 0:
                continue
            count += 1
            values = aggregate[row["module"]]
            a1 = row["a1_cycles"]["A1-OSG"]
            pidp = row["pidp_cycles"]
            values["a1_total"] += int(a1["total"])
            values["a1_source_scan"] += int(a1["source_scan"])
            values["contributors"] += int(row["contributors"])
            values["optimistic_groups"] += int(row["optimistic_k8_groups"])
            values["pidp_source_stream"] += int(pidp["source_stream"])
            values["pidp_bitmap_probe"] += int(pidp["bitmap_probe"])
            values["pidp_weight_refill"] += int(pidp["weight_refill"])
            values["pidp_dense_commit"] += int(pidp["dense_output_commit"])
            values["pidp_owner_transition"] += int(pidp["owner_transition"])
            values["a1_terminal_directory"] += int(a1["terminal_directory"])
            values["weight_references"] += int(row["weight_cache"]["pidp_references"])
            values["weight_misses"] += int(row["weight_cache"]["pidp_misses"])
            values["weight_identities"] = max(values["weight_identities"],
                                               int(row["weight_cache"]["active_tile_identities"]))
            values["cache_entries"] = int(row["weight_cache"]["cache_entries"])
    require(count == 40 and set(aggregate) == set(LAYERS), "M712 sample0 population")
    return aggregate


def count_d3_groups(mapper, payload, trusted_root, shape):
    cin = int(shape[2])
    contributors = 0
    groups = 0
    nonempty = 0
    per_timestep_contributors = np.zeros(10, dtype=np.int64)
    per_timestep_groups = np.zeros(10, dtype=np.int64)
    per_timestep_nonempty = np.zeros(10, dtype=np.int64)
    for tile in mapper.iter_polyphase_tiles(
            payload, shape, tile_m=256, trusted_root=trusted_root):
        values = np.asarray(tile["values"], dtype=np.uint8)
        require(values.ndim == 3 and values.shape[0] == 10, "D3 mapper tile shape")
        bank_counts = []
        flat_k = np.arange(values.shape[2], dtype=np.int64)
        banks = np.mod(np.mod(flat_k, cin), 8)
        for bank in range(8):
            bank_counts.append(values[:, :, banks == bank].sum(axis=2, dtype=np.int64))
        counts = np.stack(bank_counts, axis=2)
        group_count = counts.max(axis=2)
        source_count = counts.sum(axis=2)
        per_timestep_contributors += source_count.sum(axis=1, dtype=np.int64)
        per_timestep_groups += group_count.sum(axis=1, dtype=np.int64)
        per_timestep_nonempty += (group_count > 0).sum(axis=1, dtype=np.int64)
    contributors = int(per_timestep_contributors.sum())
    groups = int(per_timestep_groups.sum())
    nonempty = int(per_timestep_nonempty.sum())
    require(contributors > 0 and groups > 0 and nonempty > 0 and
            np.all(groups >= per_timestep_groups), "D3 count sanity")
    return {
        "contributors": contributors,
        "bank_conflict_groups": groups,
        "nonempty_destination_timestep_rows": nonempty,
        "per_timestep_contributors": [int(value) for value in per_timestep_contributors],
        "per_timestep_bank_conflict_groups": [int(value) for value in per_timestep_groups],
        "per_timestep_nonempty_destination_rows": [int(value) for value in per_timestep_nonempty],
    }


def build_result(root, contract):
    docs359 = root / contract["inputs"]["docs359"]
    verify_regular(docs359, DOCS359_SHA)
    m699 = root / contract["inputs"]["m699_directory"]
    m712 = root / contract["inputs"]["m712_directory"]
    m718 = root / contract["inputs"]["m718_directory"]
    m1157 = root / contract["inputs"]["m1157_directory"]
    verify_sealed_directory(m699, contract["inputs"]["m699_outer_file_sha256"])
    verify_sealed_directory(m712, contract["inputs"]["m712_outer_file_sha256"])
    verify_sealed_directory(m718, contract["inputs"]["m718_outer_file_sha256"])
    verify_sealed_directory(m1157, contract["inputs"]["m1157_outer_file_sha256"])
    manifest = strict_json(m699 / "manifest.json")
    require(sha256(m699 / "manifest.json") == contract["inputs"]["m699_manifest_sha256"],
            "M699 manifest drift")
    d3_records = [row for row in manifest["records"] if
                  row["sequence"] == "interlaken_01_a" and
                  int(row["sequence_sample_id"]) == 0 and int(row["module_index"]) == 3]
    require(len(d3_records) == 1, "unique D3 sample0 record")
    record = d3_records[0]
    require(tuple(record["input_shape"]) == (10, 1, 194, 120, 160) and
            record["route"] == "EXACT_BINARY_BITPACK", "D3 identity")
    payload = m699.joinpath(*safe_member(record["relative_path"]).parts)
    verify_regular(payload, contract["inputs"]["d3_payload_sha256"])
    mapper = load_module(root / contract["inputs"]["mapper"],
                         contract["inputs"]["mapper_sha256"], "m1158d3_mapper")
    exact = count_d3_groups(mapper, payload, m699.resolve(), tuple(record["input_shape"]))
    m712_rows = m712 / "rows.jsonl"
    verify_regular(m712_rows, contract["inputs"]["m712_rows_sha256"])
    aggregate = aggregate_m712_sample0(m712_rows)
    require(exact["contributors"] == aggregate["D3"]["contributors"],
            "D3 contributor reconciliation")
    require(aggregate["D3"]["weight_identities"] == 13 and
            aggregate["D3"]["cache_entries"] == 16 and
            aggregate["D3"]["weight_misses"] == 130,
            "D3 static weight-fit/cache identity")

    axes = {}
    geometry = contract["fixed_model"]["geometry"]
    for width in (128, 96):
        baseline = {}
        for layer in LAYERS:
            bits = int(geometry[layer]["cin"]) * int(geometry[layer]["hin"]) * int(geometry[layer]["win"])
            ingress = 10 * math.ceil(bits / width)
            baseline[layer] = aggregate[layer]["a1_total"] - aggregate[layer]["a1_source_scan"] + ingress
        d3 = geometry["D3"]
        cin = int(d3["cin"]); sites = int(d3["hin"]) * int(d3["win"])
        if width == 128:
            source_ingress = aggregate["D3"]["pidp_source_stream"]
            bitmap_probe = aggregate["D3"]["pidp_bitmap_probe"]
            probe_definition = "sealed M712 per-spatial-edge 128b probe (conservative, no cross-tap packing)"
        else:
            source_ingress = 10 * math.ceil(cin * int(d3["hin"]) * int(d3["win"]) / 96)
            bitmap_probe = 10 * sites * sum(math.ceil(taps * cin / 96)
                                            for taps in (4, 2, 2, 1))
            probe_definition = "per-destination fused-tap 96b inverse scan ceil(taps*Cin/96)"
        group_service = exact["bank_conflict_groups"] * 15
        weight_refill = aggregate["D3"]["pidp_weight_refill"]
        dense_commit = aggregate["D3"]["pidp_dense_commit"]
        control = (aggregate["D3"]["pidp_owner_transition"] +
                   aggregate["D3"]["a1_terminal_directory"])
        d3_candidate = source_ingress + bitmap_probe + group_service + weight_refill + dense_commit + control
        mixed = sum(baseline[layer] for layer in ("D0", "D1", "D2")) + d3_candidate
        baseline_sum = sum(baseline.values())
        axes[str(width)] = {
            "width_bits": width,
            "baseline_a1_osg_cycles": baseline,
            "baseline_all_four_sum": baseline_sum,
            "D3_candidate_components": {
                "source_ingress": source_ingress,
                "bitmap_probe": bitmap_probe,
                "bank_conflict_group_service_15_cycles": group_service,
                "weight_refill_13_of_16": weight_refill,
                "dense_commit": dense_commit,
                "owner_and_terminal_control": control,
            },
            "D3_candidate_cycles": d3_candidate,
            "D3_local_a1_over_candidate": ratio(baseline["D3"], d3_candidate),
            "all_four_static_mixed_cycles": mixed,
            "all_four_a1_over_static_mixed": ratio(baseline_sum, mixed),
            "probe_definition": probe_definition,
        }

    gate = (all(Decimal(axes[key]["D3_local_a1_over_candidate"]) >= Decimal("1.20")
                for key in axes) and
            all(Decimal(axes[key]["all_four_a1_over_static_mixed"]) >= Decimal("1.20")
                for key in axes))
    require(exact["bank_conflict_groups"] >= aggregate["D3"]["optimistic_groups"],
            "bank conflict groups below optimistic groups")
    capacity = {
        "weight_tile_bytes": 13_824,
        "static_weight_identities": 13,
        "logical_cache_entries": 16,
        "line_buffer_bytes": 8_064,
        "acc24_plus_metadata_bytes": 290,
        "control_bytes": 8_192,
        "total_13_entries_bytes": 13 * 13_824 + 8_064 + 290 + 8_192,
        "total_16_entries_bytes": 16 * 13_824 + 8_064 + 290 + 8_192,
    }
    capacity["headroom_13_entries_bytes"] = 240 * 1024 - capacity["total_13_entries_bytes"]
    capacity["headroom_16_entries_bytes"] = 240 * 1024 - capacity["total_16_entries_bytes"]
    require(capacity["headroom_13_entries_bytes"] >= 0 and
            capacity["headroom_16_entries_bytes"] >= 0, "logical 240KiB capacity")
    return {
        "schema": RESULT_SCHEMA,
        "date": "2026-08-30",
        "status": ("GO_FRESH_DIFFERENT_AUTHOR_CPU_HAMMER_BEFORE_ANY_RTL" if gate
                   else "NO_GO_RTL__ALL_FOUR_1P20_GATE_FAILED"),
        "policy": {"D0": "A1-OSG", "D1": "A1-OSG", "D2": "A1-OSG",
                   "D3": "STATIC_WEIGHT_FIT_BRIDGE_INCLUSIVE",
                   "runtime_or_sample_or_sequence_oracle": False,
                   "configuration_bits": 4},
        "exact_D3_replay": exact,
        "reconciliation": {
            "M712_D3_contributors": aggregate["D3"]["contributors"],
            "M712_D3_optimistic_groups": aggregate["D3"]["optimistic_groups"],
            "bank_conflict_group_overhead": ratio(exact["bank_conflict_groups"],
                                                  aggregate["D3"]["optimistic_groups"]),
            "weight_references": aggregate["D3"]["weight_references"],
            "weight_misses": aggregate["D3"]["weight_misses"],
            "weight_identities_over_cache": "13/16"},
        "width_axes": axes,
        "capacity": capacity,
        "decision": {
            "minimum_D3_local_ratio": min(axes[key]["D3_local_a1_over_candidate"] for key in axes),
            "minimum_all_four_ratio": min(axes[key]["all_four_a1_over_static_mixed"] for key in axes),
            "D3_gate_1p20_pass": all(Decimal(axes[key]["D3_local_a1_over_candidate"]) >= Decimal("1.20") for key in axes),
            "all_four_gate_1p20_pass": all(Decimal(axes[key]["all_four_a1_over_static_mixed"]) >= Decimal("1.20") for key in axes),
            "overall_gate_pass": gate,
            "rtl_authorized": False,
            "fresh_different_author_hammer_required": True},
        "identity": {
            "contract_sha256": contract["_observed_contract_sha256"],
            "analyzer_sha256": contract["identity"]["analyzer_sha256"],
            "m699_outer_file_sha256": contract["inputs"]["m699_outer_file_sha256"],
            "m712_outer_file_sha256": contract["inputs"]["m712_outer_file_sha256"],
            "m718_outer_file_sha256": contract["inputs"]["m718_outer_file_sha256"],
            "m1157_outer_file_sha256": contract["inputs"]["m1157_outer_file_sha256"],
            "d3_payload_sha256": contract["inputs"]["d3_payload_sha256"],
            "docs359_sha256": DOCS359_SHA},
        "claim_boundary": {
            "one_sequence_one_sample_all_four_calls": True,
            "D1_diagnostic_included": True,
            "decoder_population_complete": False,
            "system_speedup": False,
            "headline": False,
            "rtl_vcs_dc_eda": False,
            "paper_ppa_ready": False},
    }


def write_result(result, output):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "output already exists")
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".staging.", dir=str(output.parent)))
    try:
        (staging / "report.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        with (staging / "d3_timestep_counts.jsonl").open("w", encoding="utf-8") as stream:
            exact = result["exact_D3_replay"]
            for timestep in range(10):
                row = {"timestep": timestep,
                       "contributors": exact["per_timestep_contributors"][timestep],
                       "bank_conflict_groups": exact["per_timestep_bank_conflict_groups"][timestep],
                       "nonempty_destination_rows": exact["per_timestep_nonempty_destination_rows"][timestep]}
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        (staging / "RUN_COMPLETE.txt").write_text(result["status"] + "\n", encoding="utf-8")
        members = sorted(path for path in staging.iterdir() if path.is_file())
        manifest = staging / "SHA256SUMS"
        manifest.write_text("".join("{}  {}\n".format(sha256(path), path.name)
                                    for path in members), encoding="utf-8")
        (staging / "SHA256SUMS.seal.sha256").write_text(
            "{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")
        verify_sealed_directory(staging, sha256(staging / "SHA256SUMS.seal.sha256"))
        os.replace(staging, output)
        verify_sealed_directory(output, sha256(output / "SHA256SUMS.seal.sha256"))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    contract_path = Path(args.contract).resolve()
    contract = strict_json(contract_path)
    require(contract["schema"] == CONTRACT_SCHEMA, "contract schema")
    contract_triple = verify_double(contract_path)
    contract["_observed_contract_sha256"] = contract_triple[0]
    require(sha256(Path(__file__).resolve()) == contract["identity"]["analyzer_sha256"],
            "analyzer identity drift")
    result = build_result(root, contract)
    write_result(result, Path(args.output).resolve())
    print(json.dumps({"status": result["status"],
                      "D3_min": result["decision"]["minimum_D3_local_ratio"],
                      "all_four_min": result["decision"]["minimum_all_four_ratio"]},
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
