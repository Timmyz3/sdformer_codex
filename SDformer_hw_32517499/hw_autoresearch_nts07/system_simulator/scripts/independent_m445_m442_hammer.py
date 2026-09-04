#!/usr/bin/env python3
"""Independent full-population and raw-evidence audit for M442.

This deliberately does not import the M442 stimulus builder or consume either
M442 receipt as an oracle.  It reconstructs every PWP vector from the frozen
catalog and four frozen INT8 weight binaries, then checks every serialized
stimulus field and the raw VCS evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np


EXPECTED_SHA = {
    "catalog": "3ff522ff2296a021b005ca5733d846cc169560c125c8713c814b22a14d372f78",
    "weight_o0": "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "weight_o1": "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "weight_o2": "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "weight_o3": "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
    "stimulus": "6afd66512fc8b6fe2b4a7f759bca1299bd0cd825a51d7a5923ebadb84e4d3c1a",
    "rtl": "75ad462a584ea46bd1043bb6a21d82b5687e7ab392995b28d707c248a5f96046",
    "sva": "e5a645a0e256c7d3a72f07f027ecaf2c1d136b433c45e13248592940aba85501",
    "tb": "1bad0b365a890b7498f9fa3f2c7dc453fc913432cd8dc7ff8e35e0f50ae007cf",
    "filelist": "7a04d06c3b678a515aee548d0a004f22e3b11ae790ecc352ab8b39680974eae5",
    "builder": "9ac9e483a4bc1f0c00a38582e4f8a2158fc3156cb4a5ce511c3969e870d0311e",
    "vcs_contract": "b4f16a8c6342123364f91b9558ba90e8383658ba018486173128d326a05e23f2",
    "m430_result": "6cf413e93d8159d9516ad048eaa26c741e49c2c9a3b330fb1d6dd20ba64dab2a",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXPECTED_CODEC_SHA = "4938438e4bde7c8831deb4ed8661450261ff534113ff73dfb5045fd9612d1ba7"
EXPECTED_BLOCKS = 4 * 432 * 32 * 8
EXPECTED_LANES = EXPECTED_BLOCKS * 96
EXPECTED_NARROW = 70503


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError(f"non-standard JSON token: {token}")

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def pack_hex(values: np.ndarray, width: int, digits: int) -> str:
    packed = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values.tolist()):
        packed |= (int(value) & mask) << (lane * width)
    return f"{packed:0{digits}x}"


def verify_checksum_file(checksum_file: Path, base: Path) -> int:
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("* ")
        path = Path(name)
        if not path.is_absolute():
            path = base / path
        require(path.is_file(), f"sealed file missing: {path}")
        require(sha256(path) == expected, f"sealed file drift: {path}")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.hardware_root.resolve()
    output = args.output.resolve()
    require(not output.exists(), "refusing to overwrite independent M445 audit")
    output.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "catalog": root / "results/m430a_trainonly_dualaware_q32_catalog_r1_20260826/m430_trainonly_dualaware_q32_catalog_r1.json",
        "weight_o0": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o0_weight_i_ky_kx_o_s8.bin",
        "weight_o1": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o1_weight_i_ky_kx_o_s8.bin",
        "weight_o2": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o2_weight_i_ky_kx_o_s8.bin",
        "weight_o3": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o3_weight_i_ky_kx_o_s8.bin",
        "stimulus": root / "results/m442a_m430_full_static_codec_stimulus_r1_20260826/m442_m430_static_codec_population.hex",
        "rtl": root / "rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv",
        "sva": root / "verif_m433/m433_exact_dualbank_coread_pwp_adapter_assertions.sv",
        "tb": root / "tb_m442/tb_m442_m430_full_static_codec_m433.sv",
        "filelist": root / "dc_handoff/filelists/date_m442_m430_full_static_codec_m433_vcs.f",
        "builder": root / "system_simulator/scripts/build_m442_m430_static_codec_vcs_stimulus.py",
        "vcs_contract": root / "contracts/m442_m430_full_static_codec_m433_vcs_contract_r1_20260826.json",
        "m430_result": root / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/m430b_h67_dualaware_q32_heldout_r1.json",
        "docs359": root / "docs/359_DATE终局冻结_20260813.md",
    }
    identities = {}
    for name, path in paths.items():
        actual = sha256(path)
        require(actual == EXPECTED_SHA[name], f"exact-SHA input drift: {name}")
        identities[name] = {"path": str(path.relative_to(root)), "sha256": actual}

    catalog_dir = paths["catalog"].parent
    stimulus_dir = paths["stimulus"].parent
    vcs_dir = root / "results/m442b_m430_full_static_codec_m433_vcs_r1_20260826"
    m430_dir = root / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826"
    seal_checks = {
        "m430a_manifest_entries": verify_checksum_file(catalog_dir / "SHA256SUMS", catalog_dir),
        "m430a_outer_entries": verify_checksum_file(catalog_dir / "SHA256SUMS.seal.sha256", catalog_dir),
        "m442a_manifest_entries": verify_checksum_file(stimulus_dir / "SHA256SUMS", stimulus_dir),
        "m442a_outer_entries": verify_checksum_file(stimulus_dir / "SHA256SUMS.seal.sha256", stimulus_dir),
        "m442b_manifest_entries": verify_checksum_file(vcs_dir / "RUN_MANIFEST.sha256", vcs_dir),
        "m442b_outer_entries": verify_checksum_file(vcs_dir / "RUN_MANIFEST.seal.sha256", vcs_dir),
        "m430_manifest_entries": verify_checksum_file(m430_dir / "SHA256SUMS", m430_dir),
        "m430_outer_entries": verify_checksum_file(m430_dir / "SHA256SUMS.seal.sha256", m430_dir),
    }

    catalog = strict_json(paths["catalog"])
    require(catalog["status"] == "PASS_M430_TRAIN_ONLY_DUALAWARE_Q32_FROZEN_BEFORE_HELDOUT",
            "catalog status drift")
    geometry = catalog["geometry"]
    require(len(catalog["operators"]) == 4 and
            geometry["partitions_per_operator"] == 432 and
            geometry["q_capacity"] == 32 and
            geometry["output_blocks"] == 8 and
            geometry["shared_lanes"] == 96 and
            geometry["partition_bits"] == 16,
            "catalog geometry drift")

    blocks = lanes = narrow_blocks = 0
    signed12_violations = payload_mismatches = metadata_mismatches = 0
    reconstructed_mismatches = narrow_high_nonzero = 0
    malformed_lines = 0
    global_minimum = 1 << 30
    global_maximum = -(1 << 30)
    codec_digest = hashlib.sha256()
    with paths["stimulus"].open("r", encoding="ascii") as handle:
        for operator in range(4):
            raw_weights = np.fromfile(paths[f"weight_o{operator}"], dtype=np.int8)
            require(raw_weights.size == 6912 * 768, f"weight extent drift o{operator}")
            weights = raw_weights.reshape(6912, 768).astype(np.int16)
            for partition in range(432):
                partition_entry = catalog["operators"][operator]["partitions"][partition]
                require(partition_entry["partition"] == partition,
                        f"catalog partition index drift o{operator} p{partition}")
                centers = [int(item, 16) for item in partition_entry["nested_patterns"][:32]]
                require(len(centers) == 32 and len(set(centers)) == 32,
                        f"catalog center population drift o{operator} p{partition}")
                weight_slice = weights[partition * 16:(partition + 1) * 16]
                for center_id, center in enumerate(centers):
                    active_rows = [bit for bit in range(16) if (center >> bit) & 1]
                    if active_rows:
                        full_vector = weight_slice[active_rows].sum(axis=0, dtype=np.int32)
                    else:
                        full_vector = np.zeros(768, dtype=np.int32)
                    for output_block in range(8):
                        line = handle.readline()
                        require(line != "", f"stimulus truncated before block {blocks}")
                        fields = line.rstrip("\n").split()
                        if len(fields) != 8:
                            malformed_lines += 1
                            raise RuntimeError(f"malformed stimulus fields at block {blocks}")
                        tag_s, tile_s, center_s, block_s, narrow_s, low_s, high_s, expected_s = fields
                        require([len(tag_s), len(tile_s), len(center_s), len(block_s),
                                 len(narrow_s), len(low_s), len(high_s), len(expected_s)] ==
                                [6, 1, 2, 1, 1, 192, 128, 288],
                                f"non-canonical field width at block {blocks}")
                        require(all(re.fullmatch(r"[0-9a-f]+", item)
                                    for item in (tag_s, center_s, block_s, low_s,
                                                 high_s, expected_s)),
                                f"non-canonical hex at block {blocks}")

                        vector = full_vector[output_block * 96:(output_block + 1) * 96]
                        minimum, maximum = int(vector.min()), int(vector.max())
                        global_minimum = min(global_minimum, minimum)
                        global_maximum = max(global_maximum, maximum)
                        signed12_violations += int(np.count_nonzero(
                            (vector < -2048) | (vector > 2047)))
                        narrow = minimum >= -128 and maximum <= 127
                        raw12 = vector & 0xfff
                        expected_low = pack_hex(raw12 & 0xff, 8, 192)
                        expected_high_full = pack_hex(raw12 >> 8, 4, 96)
                        expected_high = "0" * 128 if narrow else "0" * 32 + expected_high_full
                        expected_payload = pack_hex(raw12, 12, 288)

                        expected_metadata = (
                            int(tag_s, 16) == blocks and
                            int(tile_s, 10) == (operator & 1) and
                            int(center_s, 16) == center_id and
                            int(block_s, 16) == output_block and
                            int(narrow_s, 10) == int(narrow))
                        metadata_mismatches += int(not expected_metadata)
                        payload_mismatches += int(low_s != expected_low)
                        payload_mismatches += int(high_s != expected_high)
                        payload_mismatches += int(expected_s != expected_payload)
                        narrow_high_nonzero += int(narrow and int(high_s, 16) != 0)

                        low_int = int(low_s, 16)
                        high_int = int(high_s, 16)
                        reconstructed = np.empty(96, dtype=np.int32)
                        for lane in range(96):
                            low8 = (low_int >> (lane * 8)) & 0xff
                            if narrow:
                                reconstructed[lane] = low8 - 256 if low8 & 0x80 else low8
                            else:
                                reconstructed[lane] = (((high_int >> (lane * 4)) & 0xf) << 8) | low8
                                if reconstructed[lane] & 0x800:
                                    reconstructed[lane] -= 4096
                        reconstructed_mismatches += int(np.count_nonzero(reconstructed != vector))

                        header = struct.pack("<HHBBH", operator, partition,
                                             center_id, output_block, center)
                        codec_digest.update(hashlib.sha256(
                            header + bytes.fromhex(expected_low)[::-1] +
                            int(expected_high_full, 16).to_bytes(48, "little") +
                            bytes(16) + bytes([int(narrow)])).digest())
                        blocks += 1
                        lanes += 96
                        narrow_blocks += int(narrow)
        require(handle.readline() == "", "stimulus has trailing population")

    # bytes.fromhex(expected_low)[::-1] is equivalent to little-endian encoding
    # of the 768-bit packed integer, including all leading zeros.
    require(blocks == EXPECTED_BLOCKS and lanes == EXPECTED_LANES,
            "independent population count mismatch")
    require(narrow_blocks == EXPECTED_NARROW,
            "independent narrow population mismatch")
    require(global_minimum == -1089 and global_maximum == 1059,
            "independent min/max mismatch")
    require(signed12_violations == 0 and payload_mismatches == 0 and
            metadata_mismatches == 0 and reconstructed_mismatches == 0 and
            narrow_high_nonzero == 0 and malformed_lines == 0,
            "independent stimulus reconstruction failed")
    require(codec_digest.hexdigest() == EXPECTED_CODEC_SHA,
            "independent codec global SHA mismatch")

    compile_rc = (vcs_dir / "compile.rc").read_text().strip()
    sim_rc = (vcs_dir / "sim.rc").read_text().strip()
    require(compile_rc == "0" and sim_rc == "0", "raw VCS return code nonzero")
    compile_log = (vcs_dir / "compile.log").read_text(errors="replace")
    sim_log = (vcs_dir / "sim.log").read_text(errors="replace")
    assertion_log = (vcs_dir / "assert.report").read_text(errors="replace")
    require("V-2023.12-SP1_Full64" in compile_log and
            "Top Level Modules:\n       tb_m442_m430_full_static_codec_m433" in compile_log,
            "raw compile identity mismatch")
    require(not re.search(r"Warning-\[|Error-\[|^Error", compile_log, re.I | re.M),
            "compile warning/error present")
    pass_re = re.compile(
        r"PASS M442 M430 full static codec through M433 blocks=(\d+) lanes=(\d+) "
        r"narrow=(\d+) wide=(\d+) metadata_mismatches=(\d+) "
        r"arithmetic_mismatches=(\d+) unknown_outputs=(\d+) protocol_faults=(\d+) "
        r"pop_push=(\d+) stall_cycles=(\d+) max_queue=(\d+) "
        r"runtime_issue_population=false cycles=false system_speedup=false "
        r"power=false ppa=false headline=false")
    match = pass_re.search(sim_log)
    require(match is not None, "raw VCS PASS payload missing")
    raw = [int(item) for item in match.groups()]
    require(raw[:4] == [442368, 42467328, 70503, 371865] and
            raw[4:8] == [0, 0, 0, 0] and raw[8] == 442367 and
            raw[9] == 108 and raw[10] == 1,
            "raw VCS counters mismatch")
    require(not re.search(r"failed at|Offending|Fatal:|^Error|watchdog", sim_log, re.I | re.M),
            "raw simulation failure marker present")
    require(not re.search(r"failed at|Offending|Fatal:|^Error", assertion_log, re.I | re.M),
            "raw assertion failure marker present")
    cover_matches = {}
    for name in ("cp_pop_push", "cp_protocol_fault", "cp_narrow", "cp_wide",
                 "cp_ii1_request", "cp_long_stall"):
        found = re.search(rf"\.{name}, (\d+) attempts, (\d+) match", assertion_log)
        require(found is not None, f"missing SVA cover evidence: {name}")
        cover_matches[name] = {"attempts": int(found.group(1)),
                               "matches": int(found.group(2))}
    require(cover_matches["cp_pop_push"]["matches"] == 442367 and
            cover_matches["cp_ii1_request"]["matches"] == 442259 and
            cover_matches["cp_protocol_fault"]["matches"] == 0 and
            cover_matches["cp_long_stall"]["matches"] == 0,
            "SVA coverage counts drift")

    result = {
        "schema": "m445_m442_independent_hammer_result_v1",
        "status": "PASS_M445_INDEPENDENT_FULL_POPULATION_AND_RAW_VCS_AUDIT",
        "identity": identities,
        "seal_checks": seal_checks,
        "independent_population": {
            "formula": "4*432*32*8",
            "blocks": blocks,
            "lanes_per_block": 96,
            "lanes": lanes,
            "narrow_blocks": narrow_blocks,
            "wide_blocks": blocks - narrow_blocks,
            "global_minimum": global_minimum,
            "global_maximum": global_maximum,
            "signed12_violations": signed12_violations,
            "metadata_mismatches": metadata_mismatches,
            "payload_mismatches": payload_mismatches,
            "reconstructed_lane_mismatches": reconstructed_mismatches,
            "narrow_high_nonzero": narrow_high_nonzero,
            "malformed_lines": malformed_lines,
            "codec_global_sha256": codec_digest.hexdigest(),
            "stimulus_sha256": sha256(paths["stimulus"]),
        },
        "raw_vcs": {
            "compile_rc": int(compile_rc),
            "sim_rc": int(sim_rc),
            "blocks": raw[0],
            "lanes": raw[1],
            "narrow_blocks": raw[2],
            "wide_blocks": raw[3],
            "metadata_mismatches": raw[4],
            "arithmetic_mismatches": raw[5],
            "unknown_outputs": raw[6],
            "protocol_faults": raw[7],
            "simultaneous_pop_push": raw[8],
            "stall_cycles": raw[9],
            "maximum_scoreboard_depth": raw[10],
            "sva_failure_markers": 0,
            "cover": cover_matches,
        },
        "claim_boundary": {
            "full_static_codec_population_vcs": True,
            "runtime_issue_population_vcs": False,
            "runtime_issue_population": 127277168,
            "rtl_measured_cycles": False,
            "m430_cycle_speedup_upgraded": False,
            "system_speedup": False,
            "power": False,
            "ppa": False,
            "date_headline": False,
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS_M445_INDEPENDENT_M442_HAMMER "
          f"blocks={blocks} lanes={lanes} narrow={narrow_blocks} "
          f"wide={blocks-narrow_blocks} codec_sha={codec_digest.hexdigest()} "
          "population_mismatches=0 raw_vcs_failures=0")


if __name__ == "__main__":
    main()
