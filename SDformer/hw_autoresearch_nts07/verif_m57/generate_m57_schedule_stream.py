#!/usr/bin/env python3
"""Generate a deterministic compressed M53-to-M54 K4-C16 group stream.

The canonical M53 analyzer is loaded read-only.  Its exact temporal M45 source
transformation receives only fail-closed instrumentation edits that expose the
already-selected fusion groups and their signed masks.  The schedule itself is
not replaced.  One gzip member is emitted per sample so VCS can consume it
through a FIFO without holding the workload in memory.
"""

from __future__ import print_function

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[2]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
M53_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m53_adaptive_temporal_parent_k4_ctx16_dse.py")
M53_RESULT = HW_ROOT / (
    "results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/"
    "m53_adaptive_temporal_parent_k4_ctx16_dse.json")
M53_CONTRACT = HW_ROOT / (
    "contracts/m53_adaptive_temporal_parent_k4_ctx16_dse_contract_r1_20260823.json")
M43_TEMPORAL = HW_ROOT / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_spatiotemporal_parent_delta_ablation.json")
MANIFEST = HW_ROOT / (
    "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
    "m40_bottleneck_packed_source_manifest.json")
EXPECTED_SHA256 = {
    "m53_analyzer": "638809bd72ab7f66fc69b51f4cb726f2c0d1c7712f71188066b4ef04cbdda531",
    "m53_result": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    "m53_contract": "e1dd6eb10a4b580115ff8cfe9d28605167256dfe81942ea2e2ea92d5fba88e03",
    "m43_temporal": "995fa9643ab2180d9b1480b4143959275dc3a04b4b346f8d7e22bed5266a639c",
    "manifest": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
}
PARENT_CODE = {"local_zero": 0, "left": 1, "up": 2,
               "previous_timestep": 3}
HEADER = struct.Struct("<8sIIQQQ")
GROUP = struct.Struct("<4sQQBBBBBBH")
DESCRIPTOR = struct.Struct("<HBB32s32s")
TRAILER = struct.Struct("<4sQQQ")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


class HashingWriter(object):
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.bytes = 0

    def write(self, payload):
        self.digest.update(payload)
        self.bytes += len(payload)
        self.handle.write(payload)


def replace_once(source, before, after, label):
    require(source.count(before) == 1,
            "M57 instrumentation source anchor drift: {}".format(label))
    return source.replace(before, after), {
        "name": label, "occurrences": 1,
        "before_sha256": hashlib.sha256(before.encode("utf-8")).hexdigest(),
        "after_sha256": hashlib.sha256(after.encode("utf-8")).hexdigest(),
    }


def load_instrumented_m53():
    spec = importlib.util.spec_from_file_location("m57_pinned_m53", M53_ANALYZER)
    require(spec is not None and spec.loader is not None,
            "cannot load canonical M53 analyzer")
    m53 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m53)
    canonical, source, inherited_edits = m53.transformed_m45_source(True)
    edits = []
    transformations = (
        ("    delta_masks = []\n    selected_parent = []",
         "    delta_masks = []\n    add_masks = []\n    subtract_masks = []\n"
         "    selected_parent = []", "signed_mask_lists"),
        ("        delta_masks.append(delta)\n        selected_parent.append(name)",
         "        delta_masks.append(delta)\n        add_masks.append(add_mask)\n"
         "        subtract_masks.append(subtract_mask)\n"
         "        selected_parent.append(name)", "signed_mask_append"),
        ("    return delta_masks, selected_parent, add_terms, subtract_terms",
         "    return (delta_masks, add_masks, subtract_masks, selected_parent,\n"
         "            add_terms, subtract_terms)", "signed_mask_return"),
        ("def schedule_tile_timestep(m43, delta_masks, selected_parent, fanout_k,\n"
         "                           context_capacity, start_cycle, weight_ready_cycle,\n"
         "                           tile_index):",
         "def schedule_tile_timestep(m43, delta_masks, add_masks, subtract_masks,\n"
         "                           selected_parent, fanout_k, context_capacity,\n"
         "                           start_cycle, weight_ready_cycle, tile_index,\n"
         "                           timestep_index):", "schedule_signature"),
        ("            delta_masks, selected_parent, adds, subtracts = (\n"
         "                build_tile_timestep_tasks(m43, masks, tile, timestep))",
         "            (delta_masks, add_masks, subtract_masks, selected_parent,\n"
         "             adds, subtracts) = build_tile_timestep_tasks(\n"
         "                 m43, masks, tile, timestep)", "signed_mask_call"),
        ("                m43, delta_masks, selected_parent, fanout_k,\n"
         "                context_capacity, tile_start,\n"
         "                weight_ready, tile)",
         "                m43, delta_masks, add_masks, subtract_masks,\n"
         "                selected_parent, fanout_k, context_capacity, tile_start,\n"
         "                weight_ready, tile, timestep)", "schedule_call"),
        ("            group_cycles = cycles(union_mask)\n"
         "        for task in group:",
         "            group_cycles = cycles(union_mask)\n"
         "        if M57_GROUP_CALLBACK is not None:\n"
         "            M57_GROUP_CALLBACK({\n"
         "                'tasks': tuple(group),\n"
         "                'add_masks': tuple(add_masks[task] for task in group),\n"
         "                'subtract_masks': tuple(subtract_masks[task] for task in group),\n"
         "                'parents': tuple(selected_parent[task] for task in group),\n"
         "                'group_cycles': group_cycles, 'start_cycle': now,\n"
         "                'tile': tile_index, 'timestep': timestep_index})\n"
         "        for task in group:", "group_callback"),
    )
    for before, after, name in transformations:
        source, audit = replace_once(source, before, after, name)
        edits.append(audit)
    namespace = {
        "__file__": str(m53.M45_ANALYZER),
        "__name__": "m57_instrumented_m53_temporal_m45",
        "M57_GROUP_CALLBACK": None,
    }
    exec(compile(source, str(m53.M45_ANALYZER) + "#M57", "exec"), namespace)
    m43 = namespace["load_m43_module"]()
    require(m43.ALLOW_TEMPORAL_PARENT is True,
            "M57 temporal parent did not remain enabled")
    return m53, namespace, m43, {
        "canonical_m45_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "m53_temporal_transformed_sha256": hashlib.sha256(
            m53.transformed_m45_source(True)[1].encode("utf-8")).hexdigest(),
        "m57_instrumented_source_sha256": hashlib.sha256(
            source.encode("utf-8")).hexdigest(),
        "inherited_m53_edits": inherited_edits,
        "m57_instrumentation_edits": edits,
        "schedule_semantic_edits": 0,
    }


def int256(value):
    require(0 <= value < (1 << 256), "mask outside 256 bits")
    return int(value).to_bytes(32, byteorder="little", signed=False)


def build(args):
    for name, path in (("m53_analyzer", M53_ANALYZER),
                       ("m53_result", M53_RESULT),
                       ("m53_contract", M53_CONTRACT),
                       ("m43_temporal", M43_TEMPORAL),
                       ("manifest", MANIFEST)):
        require(path.is_file() and sha256_path(path) == EXPECTED_SHA256[name],
                "M57 input SHA drift: {}".format(name))
    require(0 <= args.sample_id < 10, "sample id outside all-ten cohort")
    require(not args.output.exists() and not args.manifest_output.exists(),
            "refusing M57 schedule overwrite")
    m53, r1, m43, edit_audit = load_instrumented_m53()
    r1["validate_contract"]()
    source_manifest = strict_json(MANIFEST)
    m43_reference = strict_json(M43_TEMPORAL)
    references = dict(((row["sample_id"], row["operator"]), row)
                      for row in m43_reference["records"])
    m53_result = strict_json(M53_RESULT)
    temporal = [row for row in m53_result["configuration_ledgers"]
                if row["name"] == "K4_CTX16_TEMPORAL"]
    require(len(temporal) == 1, "M53 K4 temporal result missing")
    temporal = temporal[0]
    expected_sample = [row for row in temporal["per_sample"]
                       if row["sample_id"] == args.sample_id]
    require(len(expected_sample) == 1, "M53 sample ledger missing")
    expected_sample = expected_sample[0]
    records = [row for row in source_manifest["records"]
               if row["sample_id"] == args.sample_id]
    require(len(records) == 4, "M57 sample operator population drift")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_handle = args.output.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", compresslevel=6,
                               fileobj=raw_handle, mtime=0)
    writer = HashingWriter(compressed)
    writer.write(HEADER.pack(b"M57R1BIN", 1, args.sample_id,
                             expected_sample["fusion_groups"],
                             expected_sample["descriptor_commands"],
                             expected_sample["integrated_cycles"]))
    totals = {
        "fusion_groups": 0, "descriptor_commands": 0,
        "source_issue_cycles": 0, "signed_add_updates": 0,
        "signed_subtract_updates": 0,
        "zero_source_groups": 0,
    }
    parent_stream = dict((name, 0) for name in PARENT_CODE)
    operator_summaries = []
    record_base_cycle = 0
    group_id = 0
    for operator_index, record in enumerate(records):
        key = (record["sample_id"], record["operator"])
        require(key in references, "M43 temporal reference missing")
        masks = m43.unpack_record_masks(MANIFEST.parent, record)
        captured = []
        r1["M57_GROUP_CALLBACK"] = captured.append
        row = r1["analyze_record"](m43, masks, references[key], 4, 16)
        r1["M57_GROUP_CALLBACK"] = None
        canonical_row = [item for item in temporal["record_ledger"]["records"]
                         if (item["sample_id"], item["operator"]) == key]
        require(len(canonical_row) == 1, "M53 record ledger missing")
        canonical_row = canonical_row[0]
        for field in ("fusion_groups", "descriptor_commands", "source_only_cycles",
                      "integrated_cycles", "logical_source_updates",
                      "signed_add_updates", "signed_subtract_updates",
                      "zero_source_groups"):
            require(row[field] == canonical_row[field],
                    "M57 record reconstruction mismatch {} {}".format(
                        operator_index, field))
        require(len(captured) * r1["BLOCKS"] == row["fusion_groups"],
                "captured fusion-group population mismatch")
        require(sum(len(group["tasks"]) for group in captured) * r1["BLOCKS"] ==
                row["descriptor_commands"],
                "captured descriptor population mismatch")
        block_cycles = row["integrated_cycles"] // r1["BLOCKS"]
        require(block_cycles * r1["BLOCKS"] == row["integrated_cycles"],
                "record block-cycle divisibility drift")
        op_counts = {"fusion_groups": 0, "descriptor_commands": 0,
                     "source_issue_cycles": 0, "zero_source_groups": 0}
        for block in range(r1["BLOCKS"]):
            for group in captured:
                count = len(group["tasks"])
                require(1 <= count <= 4 and group["group_cycles"] <= 32,
                        "M57 group geometry drift")
                target_cycle = (record_base_cycle + block * block_cycles +
                                group["start_cycle"])
                writer.write(GROUP.pack(
                    b"GRP1", target_cycle, group_id, args.sample_id,
                    operator_index, group["timestep"], group["tile"], block,
                    count, group["group_cycles"]))
                union = 0
                for slot, task in enumerate(group["tasks"]):
                    add_mask = group["add_masks"][slot]
                    subtract_mask = group["subtract_masks"][slot]
                    parent = group["parents"][slot]
                    require(parent in PARENT_CODE and
                            (add_mask & subtract_mask) == 0,
                            "M57 signed/parent descriptor drift")
                    writer.write(DESCRIPTOR.pack(
                        task, PARENT_CODE[parent], 0,
                        int256(add_mask), int256(subtract_mask)))
                    union |= add_mask | subtract_mask
                    parent_stream[parent] += 1
                    totals["signed_add_updates"] += m43.population(add_mask)
                    totals["signed_subtract_updates"] += m43.population(
                        subtract_mask)
                issue = m43.bank_issue_cycles(union)
                require(issue == group["group_cycles"],
                        "M57 union issue-cycle drift")
                totals["fusion_groups"] += 1
                totals["descriptor_commands"] += count
                totals["source_issue_cycles"] += issue
                totals["zero_source_groups"] += int(issue == 0)
                op_counts["fusion_groups"] += 1
                op_counts["descriptor_commands"] += count
                op_counts["source_issue_cycles"] += issue
                op_counts["zero_source_groups"] += int(issue == 0)
                group_id += 1
        require(op_counts["fusion_groups"] == row["fusion_groups"] and
                op_counts["descriptor_commands"] == row["descriptor_commands"] and
                op_counts["source_issue_cycles"] == row["source_only_cycles"] and
                op_counts["zero_source_groups"] == row["zero_source_groups"],
                "M57 emitted operator totals mismatch")
        operator_summaries.append(dict({
            "operator_index": operator_index, "operator": record["operator"],
            "model_integrated_cycles": row["integrated_cycles"],
        }, **op_counts))
        record_base_cycle += row["integrated_cycles"]

    writer.write(TRAILER.pack(b"END1", totals["fusion_groups"],
                             totals["descriptor_commands"],
                             totals["source_issue_cycles"]))
    uncompressed_sha = writer.digest.hexdigest()
    uncompressed_bytes = writer.bytes
    compressed.close()
    raw_handle.close()
    require(totals["fusion_groups"] == expected_sample["fusion_groups"] and
            totals["descriptor_commands"] == expected_sample["descriptor_commands"] and
            totals["source_issue_cycles"] == expected_sample["source_only_cycles"] and
            totals["zero_source_groups"] == expected_sample["zero_source_groups"] and
            totals["signed_add_updates"] == expected_sample["signed_add_updates"] and
            totals["signed_subtract_updates"] == expected_sample["signed_subtract_updates"] and
            record_base_cycle == expected_sample["integrated_cycles"],
            "M57 emitted sample does not reproduce M53")
    normalized_parent = dict((name, count // r1["BLOCKS"])
                             for name, count in parent_stream.items())
    require(all(parent_stream[name] % r1["BLOCKS"] == 0
                for name in parent_stream) and
            normalized_parent == expected_sample["parent_choice_by_tile"],
            "M57 parent-choice normalization mismatch")
    payload = {
        "schema": "m57_m53_k4c16_temporal_schedule_stream_manifest_v1",
        "status": "PASS_M57_SAMPLE_STREAM_EXACT_M53_RECONSTRUCTION",
        "sample_id": args.sample_id,
        "identity": {
            "generator_sha256": sha256_path(Path(__file__).resolve()),
            "inputs_sha256": EXPECTED_SHA256,
            "compressed_stream_path": str(
                args.output.resolve().relative_to(HW_ROOT.resolve())),
            "compressed_stream_sha256": sha256_path(args.output),
            "compressed_stream_bytes": args.output.stat().st_size,
            "uncompressed_stream_sha256": uncompressed_sha,
            "uncompressed_stream_bytes": uncompressed_bytes,
        },
        "format": {
            "endianness": "little",
            "header_struct": HEADER.format.decode("ascii") if isinstance(
                HEADER.format, bytes) else HEADER.format,
            "group_struct": GROUP.format.decode("ascii") if isinstance(
                GROUP.format, bytes) else GROUP.format,
            "descriptor_struct": DESCRIPTOR.format.decode("ascii") if isinstance(
                DESCRIPTOR.format, bytes) else DESCRIPTOR.format,
            "trailer_struct": TRAILER.format.decode("ascii") if isinstance(
                TRAILER.format, bytes) else TRAILER.format,
            "compression": "gzip-level6-mtime0-no-filename",
        },
        "m53_exact_reconstruction": {
            "fusion_groups": totals["fusion_groups"],
            "descriptor_commands": totals["descriptor_commands"],
            "source_issue_cycles": totals["source_issue_cycles"],
            "model_integrated_cycles": expected_sample["integrated_cycles"],
            "zero_source_groups": totals["zero_source_groups"],
            "signed_add_updates": totals["signed_add_updates"],
            "signed_subtract_updates": totals["signed_subtract_updates"],
            "parent_descriptor_count_stream_x8": parent_stream,
            "parent_choice_by_tile_normalized_div8": normalized_parent,
        },
        "operator_summaries": operator_summaries,
        "dynamic_source_edit_audit": edit_audit,
        "claim_boundary": {
            "accepted": "exact compressed workload stream for one frozen M53 sample",
            "not_yet_admitted": [
                "VCS accepted-handshake replay",
                "all-ten completion",
                "M53 transaction cycles as RTL or system cycles",
                "DC, PPA, power, energy or DATE headline"
            ]
        }
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    print("PASS M57 stream sample={} groups={} commands={} source={}".format(
        result["sample_id"],
        result["m53_exact_reconstruction"]["fusion_groups"],
        result["m53_exact_reconstruction"]["descriptor_commands"],
        result["m53_exact_reconstruction"]["source_issue_cycles"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M57 schedule stream: {}".format(error))
        raise SystemExit(1)
