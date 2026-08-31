#!/usr/bin/env python3
"""Freeze the M425 sample-0/equidistant H67 subset for direct RTL SAIF.

This exporter only copies already frozen M410R2 runtime records and M408
static PWP codec records.  It does not estimate activity, cycles, power, or
energy.  Linear-record access is byte-exact and every copied record is parsed
again before the output manifest is sealed.
"""

import argparse
import hashlib
import json
from pathlib import Path


PARTITIONS = 432
OPERATORS = 4
ROWS_PER_PHASE = 3000
CENTERS = 32
BLOCKS_PER_CENTER = 8
SELECTED_PARTITIONS = tuple(range(0, 432, 27))
assert SELECTED_PARTITIONS == (
    0, 27, 54, 81, 108, 135, 162, 189,
    216, 243, 270, 297, 324, 351, 378, 405)


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
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs)


def read_fixed_line(handle, index, bytes_per_line, hex_digits, label):
    handle.seek(index * bytes_per_line)
    line = handle.read(bytes_per_line)
    require(len(line) == bytes_per_line, f"{label} record extent failure")
    require(line[-1:] == b"\n", f"{label} newline drift")
    text = line[:-1]
    require(len(text) == hex_digits, f"{label} hex width drift")
    try:
        int(text, 16)
    except ValueError as exc:
        raise RuntimeError(f"{label} non-hex record") from exc
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M425 subset overwrite")
    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m425_h67_balanced_selected_slice_saif_subset_contract_v1",
            "M425 subset contract schema drift")
    require(contract["selection"]["sample"] == 0, "sample selection drift")
    require(tuple(contract["selection"]["partitions_per_operator"]) ==
            SELECTED_PARTITIONS, "partition selection drift")
    require(contract["selection"]["operators"] == [0, 1, 2, 3],
            "operator selection drift")

    hw_root = args.contract.resolve().parents[1]
    for entry in contract["inputs"].values():
        path = hw_root / entry["path"]
        require(path.is_file(), "missing input: " + str(path))
        require(sha256(path) == entry["sha256"],
                "input SHA drift: " + entry["path"])

    runtime_manifest = strict_json(
        hw_root / contract["inputs"]["runtime_manifest"]["path"])
    static_manifest = strict_json(
        hw_root / contract["inputs"]["static_manifest"]["path"])
    require(runtime_manifest["schema"] ==
            "m410r2_h67_q32_full_runtime_vcs_stimulus_v2",
            "runtime manifest schema drift")
    require(runtime_manifest["layout"]["phase_order"] ==
            "sample,operator,partition", "runtime phase order drift")
    require(runtime_manifest["layout"]["row_order"] ==
            "source_row_0_to_2999", "runtime row order drift")
    require(static_manifest["schema"] ==
            "m408_h67_q32_static_codec_vcs_stimulus_v1",
            "static manifest schema drift")
    require(static_manifest["layout"]["index_order"] ==
            "operator,partition,center_id,global_output_block",
            "static index order drift")

    config_source = hw_root / contract["inputs"]["runtime_configs"]["path"]
    row_source = hw_root / contract["inputs"]["runtime_rows"]["path"]
    pwp_source = hw_root / contract["inputs"]["static_pwp"]["path"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_out = args.output_dir / "m425_h67_phase_config_768.memh"
    row_out = args.output_dir / "m425_h67_runtime_rows_32.memh"
    pwp_out = args.output_dir / "m425_h67_static_pwp_1281.memh"

    phases = []
    total_rows = pwp_rows = pass1 = early = zero_rows = pop1_rows = 0
    pwp_blocks = narrow_blocks = wide_blocks = contributions = 0
    center_hist = [0] * CENTERS
    distance_hist = [0] * 17
    with config_source.open("rb") as src_cfg, \
            row_source.open("rb") as src_rows, \
            pwp_source.open("rb") as src_pwp, \
            config_out.open("wb") as dst_cfg, \
            row_out.open("wb") as dst_rows, \
            pwp_out.open("wb") as dst_pwp:
        local_phase = 0
        for operator in range(OPERATORS):
            for partition in SELECTED_PARTITIONS:
                source_phase = operator * PARTITIONS + partition
                config_line = read_fixed_line(
                    src_cfg, source_phase, 193, 192, "config")
                dst_cfg.write(config_line)
                local_pwp_rows = local_pass1 = local_early = 0
                local_zero = local_pop1 = 0
                for local_row in range(ROWS_PER_PHASE):
                    source_row = source_phase * ROWS_PER_PHASE + local_row
                    row_line = read_fixed_line(
                        src_rows, source_row, 9, 8, "runtime row")
                    dst_rows.write(row_line)
                    record = int(row_line[:-1], 16)
                    require((record >> 29) == 0, "reserved row bits nonzero")
                    require(not (((record >> 27) & 1) and
                                 ((record >> 28) & 1)),
                            "pass1/early flags overlap")
                    original = record & 0xffff
                    center = (record >> 16) & 0x1f
                    distance = (record >> 21) & 0x1f
                    use_pwp = (record >> 26) & 1
                    require(distance <= 16, "distance outside q16 range")
                    population = bin(original).count("1")
                    require(use_pwp == int(distance + 1 < population),
                            "use-PWP equation drift")
                    center_hist[center] += 1
                    distance_hist[distance] += 1
                    local_pwp_rows += use_pwp
                    local_pass1 += (record >> 27) & 1
                    local_early += (record >> 28) & 1
                    local_zero += population == 0
                    local_pop1 += population == 1
                for center in range(CENTERS):
                    for output_block in range(BLOCKS_PER_CENTER):
                        source_block = (((operator * PARTITIONS + partition)
                                         * CENTERS + center)
                                        * BLOCKS_PER_CENTER + output_block)
                        pwp_line = read_fixed_line(
                            src_pwp, source_block, 322, 321, "static PWP")
                        dst_pwp.write(pwp_line)
                        physical = int(pwp_line[:-1], 16)
                        narrow = (physical >> 1280) & 1
                        require(((physical >> 1152) & ((1 << 128)-1)) == 0,
                                "static high padding nonzero")
                        low = physical & ((1 << 768)-1)
                        high = (physical >> 768) & ((1 << 512)-1)
                        if narrow:
                            for lane in range(96):
                                low_byte = (low >> (lane*8)) & 0xff
                                high_nibble = (high >> (lane*4)) & 0xf
                                require(high_nibble ==
                                        (0xf if low_byte & 0x80 else 0),
                                        "narrow sign-extension drift")
                        narrow_blocks += narrow
                        wide_blocks += 1 - narrow
                        pwp_blocks += 1
                phases.append({
                    "local_phase": local_phase,
                    "sample": 0,
                    "operator": operator,
                    "partition": partition,
                    "source_phase": source_phase,
                    "source_row_begin": source_phase * ROWS_PER_PHASE,
                    "source_static_block_begin":
                        source_phase * CENTERS * BLOCKS_PER_CENTER,
                    "pwp_rows": local_pwp_rows,
                    "pass1_tasks": local_pass1,
                    "early_stops": local_early,
                    "zero_rows": local_zero,
                    "pop1_rows": local_pop1,
                })
                total_rows += ROWS_PER_PHASE
                pwp_rows += local_pwp_rows
                pass1 += local_pass1
                early += local_early
                zero_rows += local_zero
                pop1_rows += local_pop1
                local_phase += 1

    require(len(phases) == 64 and total_rows == 192000,
            "M425 selected extent drift")
    require(pwp_blocks == 64 * 32 * 8,
            "M425 static subset extent drift")

    # Count only the blocks actually replayed by selected runtime results.
    actual_narrow = actual_wide = 0
    with row_out.open("rb") as rows, pwp_out.open("rb") as pwp:
        for phase in range(64):
            for local_row in range(ROWS_PER_PHASE):
                record = int(read_fixed_line(
                    rows, phase*ROWS_PER_PHASE+local_row, 9, 8,
                    "output row")[:-1], 16)
                if not ((record >> 26) & 1):
                    continue
                center = (record >> 16) & 0x1f
                for block in range(8):
                    item = int(read_fixed_line(
                        pwp, (phase*32+center)*8+block, 322, 321,
                        "output PWP")[:-1], 16)
                    narrow = (item >> 1280) & 1
                    actual_narrow += narrow
                    actual_wide += 1 - narrow
    require(actual_narrow + actual_wide == pwp_rows * 8,
            "actual replay block ledger drift")
    contributions = actual_narrow + 2 * actual_wide

    outputs = {}
    for name, path, records, bits in (
            ("configs", config_out, 64, 768),
            ("rows", row_out, total_rows, 32),
            ("static_pwp", pwp_out, pwp_blocks, 1281)):
        outputs[name] = {
            "path": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "records": records,
            "word_bits": bits,
        }

    manifest = {
        "schema": "m425_h67_balanced_selected_slice_saif_subset_v1",
        "status": "PASS_M425_FROZEN_PRE_VCS_ACTIVITY_SUBSET_EXPORT",
        "contract": {"path": str(args.contract.resolve().relative_to(hw_root)),
                     "sha256": sha256(args.contract)},
        "selection": contract["selection"],
        "phase_map": phases,
        "population": {
            "phases": len(phases),
            "source_rows": total_rows,
            "pass0_tasks": total_rows,
            "pass1_tasks": pass1,
            "early_stops": early,
            "zero_rows": zero_rows,
            "pop1_rows": pop1_rows,
            "pwp_rows": pwp_rows,
            "pwp_blocks_replayed": pwp_rows * 8,
            "low_accepts_expected": pwp_rows * 8,
            "high_accepts_expected": actual_wide,
            "narrow_blocks_replayed": actual_narrow,
            "wide_blocks_replayed": actual_wide,
            "contributions_expected": contributions,
            "center_id_histogram": center_hist,
            "distance_histogram": distance_hist,
            "exported_static_blocks_all_32_centers": pwp_blocks,
            "exported_static_narrow_blocks": narrow_blocks,
            "exported_static_wide_blocks": wide_blocks,
        },
        "outputs": outputs,
        "claim_boundary": {
            "deterministic_frozen_subset": True,
            "synopsys_vcs_executed": False,
            "saif_generated": False,
            "power_or_energy": False,
            "system_speedup": False,
            "paper_power_eligible": False,
            "headline": False,
        },
    }
    manifest_path = args.output_dir / "m425_h67_saif_subset_manifest_r1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2,
                                        sort_keys=True) + "\n",
                             encoding="utf-8")
    print("M425_SUBSET_EXPORT_PASS phases=64 rows={} pwp_rows={} "
          "blocks={} narrow={} wide={} contributions={}".format(
              total_rows, pwp_rows, pwp_rows*8, actual_narrow,
              actual_wide, contributions), flush=True)


if __name__ == "__main__":
    main()
