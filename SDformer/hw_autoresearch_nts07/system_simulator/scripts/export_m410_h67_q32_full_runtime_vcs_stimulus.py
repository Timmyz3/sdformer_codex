#!/usr/bin/env python3
"""Export the full ordered H67 q32 runtime population for Synopsys VCS.

The output preserves all 51,840,000 source rows in M401 phase order:
sample, operator, partition, source_row.  Each 32-bit row record is:
  [15:0]  original 16-bit source pattern
  [20:16] contract-visible lowest-ID center (q16 for pop<2, q32 otherwise)
  [25:21] contract-visible Hamming distance
  [26]    strict PWP eligibility (1 + distance < source population)
  [27]    expected registered pass1 task
  [28]    expected exact pass0 zero-distance early stop
  [31:29] zero

Each 768-bit phase configuration is the exact three-beat M405 image:
  [511:0]   32 little-indexed 16-bit centers
  [767:512] center-major narrow bitmap, center_id*8 + output_block
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


SAMPLES = 10
OPERATORS = 4
PARTITIONS = 432
ROWS = 3000
PHASES = SAMPLES * OPERATORS * PARTITIONS
SOURCE_ROWS = PHASES * ROWS


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
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            "cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M410 overwrite")

    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m410r2_h67_q32_full_runtime_vcs_stimulus_export_contract_v2",
            "M410R2 contract schema drift")
    hw_root = args.contract.resolve().parents[1]
    inputs = contract["inputs"]
    paths = {}
    for name, identity in inputs.items():
        path = hw_root / identity["path"]
        require(path.is_file(), "missing M410 input: " + str(path))
        require(sha256(path) == identity["sha256"],
                "M410 input SHA drift: " + name)
        paths[name] = path

    m401_contract = strict_json(paths["m401_contract"])
    require(m401_contract["schema"] ==
            "m401_h67_q32_elastic_pwp_full_replay_contract_v1",
            "M401 contract drift")
    trace = strict_json(paths["m40_trace"])
    catalog = strict_json(paths["m338_catalog"])
    static_manifest = strict_json(paths["m408_static_manifest"])
    require(static_manifest["schema"] ==
            "m408_h67_q32_static_codec_vcs_stimulus_v1" and
            static_manifest["population"]["blocks"] == 442368 and
            static_manifest["population"]["narrow_blocks"] == 112167,
            "M408 static identity drift")
    paper = contract["paper_identity"]
    require(trace["identity"]["checkpoint_sha256"] ==
            paper["checkpoint_sha256"] and
            trace["identity"]["bn_policy"] == "no_running" and
            len(trace["records"]) == 40,
            "M410 H67 runtime identity drift")
    require(catalog["split"]["role"] ==
            "DSEC_TRAIN_ONLY_PAFT_CALIBRATION" and
            catalog["split"]["selected_train_sequences"] == 18 and
            catalog["split"]["test_or_validation_data_used"] is False and
            catalog["admission"]["train_only_catalog"] is True,
            "M410 train-only catalog identity drift")

    static_lines = paths["m408_static_memh"].read_text(
        encoding="ascii").splitlines()
    require(len(static_lines) == 442368,
            "M408 static memh extent drift")
    narrow = np.zeros((OPERATORS, PARTITIONS, 32, 8), dtype=np.bool_)
    for index, line in enumerate(static_lines):
        require(len(line) == 321 and line[0] in "01",
                "M408 static memh record drift")
        operator = index // (PARTITIONS * 32 * 8)
        within_operator = index % (PARTITIONS * 32 * 8)
        partition = within_operator // (32 * 8)
        within_partition = within_operator % (32 * 8)
        center = within_partition // 8
        output_block = within_partition % 8
        narrow[operator, partition, center, output_block] = line[0] == "1"
    require(int(np.count_nonzero(narrow)) == 112167,
            "M410 narrow bitmap population drift")
    del static_lines

    m43 = load_module(paths["m43_unpacker"], "m410_m43")
    require(m43.ROWS == ROWS and m43.TILES == 27,
            "M410 M43 geometry drift")
    trace_dir = paths["m40_trace"].parent
    records = {}
    payload_files = 0
    payload_bytes = 0
    for record in trace["records"]:
        for key, sha_key in (("packed_file", "packed_file_sha256"),
                             ("value_payload_file",
                              "value_payload_sha256")):
            path = trace_dir / record[key]
            require(path.is_file() and sha256(path) == record[sha_key],
                    "M410 M40 payload drift")
            payload_files += 1
            payload_bytes += path.stat().st_size
        identity = (record["sample_id"], record["operator_index"])
        require(identity not in records, "duplicate M410 runtime record")
        records[identity] = record
    require(len(records) == SAMPLES * OPERATORS,
            "M410 runtime record coverage drift")

    pop8 = np.asarray([bin(index).count("1") for index in range(256)],
                      dtype=np.uint8)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_path = args.output_dir / "m410r2_h67_q32_phase_config_768.memh"
    row_path = args.output_dir / "m410r2_h67_q32_runtime_rows_32.memh"
    config_digest = hashlib.sha256()
    row_digest = hashlib.sha256()
    phase_digest = hashlib.sha256()
    phases = source_rows = zero_rows = pop1_rows = 0
    pass1_tasks = early_stops = pwp_rows = 0
    minimum_distance = 31
    maximum_distance = 0
    center_histogram = np.zeros(32, dtype=np.int64)
    distance_histogram = np.zeros(17, dtype=np.int64)

    with config_path.open("wb") as config_output, row_path.open("wb") as row_output:
        for sample in range(SAMPLES):
            for operator in range(OPERATORS):
                record = records[(sample, operator)]
                masks = m43.unpack_record_masks(trace_dir, record)
                require(len(masks) == ROWS * m43.TILES,
                        "M410 unpacked mask extent drift")
                for partition in range(PARTITIONS):
                    centers = np.asarray([
                        int(value, 16) for value in
                        catalog["operators"][operator]["partitions"]
                        [partition]["nested_patterns"][:32]],
                        dtype=np.uint16)
                    require(centers.shape == (32,),
                            "M410 q32 center extent drift")
                    bitmap_bits = narrow[operator, partition].reshape(-1)
                    config_word = 0
                    for center_id, center in enumerate(centers):
                        config_word |= int(center) << (center_id * 16)
                    for bit_index, value in enumerate(bitmap_bits):
                        config_word |= int(value) << (512 + bit_index)
                    config_line = f"{config_word:0192x}\n".encode("ascii")
                    config_output.write(config_line)
                    config_digest.update(config_line)

                    tile = partition // 16
                    subtile = partition % 16
                    original = np.fromiter((
                        (masks[row * m43.TILES + tile] >>
                         (subtile * 16)) & 0xffff
                        for row in range(ROWS)), dtype=np.uint16,
                        count=ROWS)
                    xor = np.bitwise_xor(original[:, None], centers[None, :])
                    distances = (pop8[np.bitwise_and(xor, 0xff)] +
                                 pop8[np.right_shift(xor, 8)])
                    best_id = np.argmin(distances, axis=1).astype(np.uint32)
                    best_distance = distances[
                        np.arange(ROWS), best_id].astype(np.uint32)
                    population = (pop8[np.bitwise_and(original, 0xff)] +
                                  pop8[np.right_shift(original, 8)]).astype(
                                      np.uint32)
                    pass0_id = np.argmin(distances[:, :16],
                                         axis=1).astype(np.uint32)
                    pass0_distance = distances[
                        np.arange(ROWS), pass0_id].astype(np.uint32)
                    pass1 = np.logical_and(population >= 2,
                                           pass0_distance > 0)
                    early = np.logical_and(population >= 2,
                                           pass0_distance == 0)
                    use_pwp = 1 + best_distance < population
                    # M405 deliberately avoids pass1 for zero/pop1 fallback
                    # rows. Their center is not consumed by PWP, so the
                    # contract-visible result is the exact pass0 result.
                    # Eligible rows retain the exact lowest-ID q32 result.
                    visible_id = np.where(population >= 2, best_id,
                                          pass0_id).astype(np.uint32)
                    visible_distance = np.where(
                        population >= 2, best_distance,
                        pass0_distance).astype(np.uint32)
                    packed = (original.astype(np.uint32) |
                              np.left_shift(visible_id, 16) |
                              np.left_shift(visible_distance, 21) |
                              np.left_shift(use_pwp.astype(np.uint32), 26) |
                              np.left_shift(pass1.astype(np.uint32), 27) |
                              np.left_shift(early.astype(np.uint32), 28))
                    require(int(np.count_nonzero(packed >> 29)) == 0,
                            "M410 reserved row bits nonzero")
                    row_lines = "".join(
                        f"{int(value):08x}\n" for value in packed).encode(
                            "ascii")
                    row_output.write(row_lines)
                    row_digest.update(row_lines)
                    phase_digest.update(phases.to_bytes(4, "little"))
                    phase_digest.update(config_word.to_bytes(96, "little"))
                    phase_digest.update(packed.astype("<u4").tobytes())

                    phases += 1
                    source_rows += ROWS
                    zero_rows += int(np.count_nonzero(population == 0))
                    pop1_rows += int(np.count_nonzero(population == 1))
                    pass1_tasks += int(np.count_nonzero(pass1))
                    early_stops += int(np.count_nonzero(early))
                    pwp_rows += int(np.count_nonzero(use_pwp))
                    local_min = int(visible_distance.min())
                    local_max = int(visible_distance.max())
                    minimum_distance = min(minimum_distance, local_min)
                    maximum_distance = max(maximum_distance, local_max)
                    center_histogram += np.bincount(
                        visible_id, minlength=32).astype(np.int64)
                    distance_histogram += np.bincount(
                        visible_distance, minlength=17).astype(np.int64)
                print("[M410 EXPORT] sample={}/10 operator={}/4 phases={}".
                      format(sample + 1, operator + 1, phases), flush=True)

    gates = contract["execution_gates"]
    observed = {
        "phases": phases,
        "config_beats": phases * 3,
        "source_rows": source_rows,
        "zero_rows": zero_rows,
        "pop1_rows": pop1_rows,
        "pass1_tasks": pass1_tasks,
        "early_stops": early_stops,
        "pwp_rows": pwp_rows,
        "matcher_task_cycles": source_rows + pass1_tasks,
        "m401_matcher_cycles_with_two_cycle_phase_overhead":
            source_rows + pass1_tasks + phases * 2,
    }
    for name, expected in gates.items():
        if name in observed:
            require(observed[name] == expected,
                    "M410 frozen runtime gate drift: " + name)
    require(sum(center_histogram) == SOURCE_ROWS and
            sum(distance_histogram) == SOURCE_ROWS and
            minimum_distance == 0 and maximum_distance <= 16,
            "M410 result histogram drift")
    require(config_path.stat().st_size == PHASES * 193 and
            row_path.stat().st_size == SOURCE_ROWS * 9,
            "M410 memh byte extent drift")
    require(config_digest.hexdigest() == sha256(config_path) and
            row_digest.hexdigest() == sha256(row_path),
            "M410 streaming digest drift")

    manifest = {
        "schema": "m410r2_h67_q32_full_runtime_vcs_stimulus_v2",
        "status": "PASS_M410R2_CONTRACT_VISIBLE_FULL_RUNTIME_STIMULUS_EXPORT",
        "identity": {
            "contract": {
                "path": str(args.contract.resolve().relative_to(hw_root)),
                "sha256": sha256(args.contract)},
            "m401_contract": inputs["m401_contract"],
            "m40_trace": inputs["m40_trace"],
            "m338_catalog": inputs["m338_catalog"],
            "m408_static_manifest": inputs["m408_static_manifest"],
            "m408_static_memh": inputs["m408_static_memh"],
            "docs359": inputs["docs359"],
            "payload_files": payload_files,
            "payload_bytes": payload_bytes,
        },
        "layout": {
            "phase_order": "sample,operator,partition",
            "row_order": "source_row_0_to_2999",
            "result_semantics": "population<2 exact q16 pass0 fallback result; population>=2 exact lowest-ID q32 result",
            "config_word_bits": 768,
            "config_hex_digits": 192,
            "row_word_bits": 32,
            "row_hex_digits": 8,
            "row_fields": {
                "original": [0, 15],
                "center_id": [16, 20],
                "distance": [21, 25],
                "use_pwp": 26,
                "pass1": 27,
                "early": 28,
                "reserved_zero": [29, 31],
            },
        },
        "population": observed,
        "result_population": {
            "minimum_distance": minimum_distance,
            "maximum_distance": maximum_distance,
            "center_id_histogram": [int(value) for value in center_histogram],
            "distance_histogram": [int(value) for value in
                                   distance_histogram],
        },
        "output": {
            "config": {
                "path": config_path.name,
                "bytes": config_path.stat().st_size,
                "sha256": config_digest.hexdigest(),
            },
            "rows": {
                "path": row_path.name,
                "bytes": row_path.stat().st_size,
                "sha256": row_digest.hexdigest(),
            },
            "phase_identity_sha256": phase_digest.hexdigest(),
        },
        "claim_boundary": {
            "full_ordered_runtime_stimulus": True,
            "vcs_executed": False,
            "rtl_realtrace_cycle_match": False,
            "rtl_measured_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    manifest_path = args.output_dir / \
        "m410r2_h67_q32_full_runtime_vcs_stimulus_r2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2,
                                        sort_keys=True) + "\n",
                             encoding="utf-8")
    print("M410R2_EXPORT_PASS phases={} rows={} pass1={} early={} pwp={} "
          "task_cycles={} row_sha256={}".format(
              phases, source_rows, pass1_tasks, early_stops, pwp_rows,
              observed["matcher_task_cycles"], row_digest.hexdigest()),
          flush=True)


if __name__ == "__main__":
    main()
