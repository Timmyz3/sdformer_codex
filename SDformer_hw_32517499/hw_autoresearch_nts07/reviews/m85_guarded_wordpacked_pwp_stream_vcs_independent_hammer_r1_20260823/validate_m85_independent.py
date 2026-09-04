#!/usr/bin/env python3
"""Independent M85 record/metadata/mask/address oracle.

This checker deliberately does not import the M83/M84/M85 producer code.  It
parses the frozen binary formats directly, reconstructs the bank rows from the
mathematical word address, masks beats, and compares the resulting signed
vectors with a byte-window reference decoder.
"""

import argparse
import hashlib
import json
import random
import re
import struct
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple


PHASES = 1728
ENTRIES = 128
PATTERNS = 16
BLOCKS = 8
HEADER_BYTES = 48
METADATA_BYTES = 74
BUFFER_WORDS = 3680
ROWS = 460
LANES = 96
WORDS = {0: 24, 1: 27, 2: 30, 3: 33, 4: 0}
FETCH_WORDS = {0: 24, 1: 32, 2: 32, 3: 40, 4: 0}
BEATS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 1}
KEEP_WORDS = {0: 8, 1: 3, 2: 6, 3: 1}
EXPECTED_SHA = {
    "records": "6de1521b2ee91281eadd5945d6f69b45df2cf5f1e2cc0c93834df4ec4c87190d",
    "offsets": "1cddfc800ba18569b9c0d7f4c193f4b07ddf9046fa7688a1859f6c3e448bf30c",
    "metadata": "52b700b1c17172ae5a2d08acacfd9c5bac007893332f9afd9f23c29636e468a0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_fields(raw: bytes, count: int, width: int) -> List[int]:
    value = int.from_bytes(raw, "little")
    mask = (1 << width) - 1
    return [(value >> (index * width)) & mask for index in range(count)]


def pattern_bases(codes: List[int], invalid_as_zero: bool = False) -> List[int]:
    cursor = 0
    bases = []
    for pattern in range(PATTERNS):
        bases.append(cursor)
        for code in codes[pattern * BLOCKS:(pattern + 1) * BLOCKS]:
            if code in WORDS:
                cursor += WORDS[code]
            elif not invalid_as_zero:
                raise ValueError(f"reserved code {code}")
    return bases


def audit_metadata(codes: List[int], bases: List[int]) -> dict:
    reasons = []  # type: List[str]
    cursor = 0
    max_fetch_end = 0
    for pattern in range(PATTERNS):
        if bases[pattern] != cursor:
            reasons.append(f"base[{pattern}]={bases[pattern]} expected={cursor}")
        for block in range(BLOCKS):
            index = pattern * BLOCKS + block
            code = codes[index]
            if code not in WORDS:
                reasons.append(f"reserved_code[{index}]={code}")
                continue
            fetch_end = cursor + FETCH_WORDS[code]
            max_fetch_end = max(max_fetch_end, fetch_end)
            if fetch_end > BUFFER_WORDS:
                reasons.append(f"fetch_overflow[{index}]={fetch_end}")
            cursor += WORDS[code]
            if cursor > BUFFER_WORDS:
                reasons.append(f"cursor_overflow[{index}]={cursor}")
    rounded_terminal = (cursor + 7) & ~7
    if rounded_terminal == 0:
        reasons.append("zero_terminal")
    if cursor > BUFFER_WORDS:
        reasons.append(f"terminal_overflow={cursor}")
    return {
        "accepted": not reasons,
        "cursor_words": cursor,
        "rounded_terminal_words": rounded_terminal,
        "max_fetch_end": max_fetch_end,
        "reason_count": len(reasons),
        "reasons_first_12": reasons[:12],
    }


def payload_word(record: bytes, word_index: int, terminal: int) -> int:
    if word_index >= terminal:
        return 0
    offset = HEADER_BYTES + word_index * 4
    return int.from_bytes(record[offset:offset + 4], "little")


def reference_decode(payload: bytes, width: int) -> List[int]:
    """Decode with small byte windows, independent of beat assembly."""
    decoded = []
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    for lane in range(LANES):
        bit_offset = lane * width
        byte_offset, shift = divmod(bit_offset, 8)
        window = int.from_bytes(payload[byte_offset:byte_offset + 3], "little")
        raw = (window >> shift) & mask
        decoded.append(raw - (1 << width) if raw & sign else raw)
    return decoded


def stream_decode(words: List[int], width: int) -> List[int]:
    """Model M82 extraction from a post-mask word stream with one big integer."""
    packed = 0
    for index, word in enumerate(words):
        packed |= (word & 0xFFFFFFFF) << (32 * index)
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    decoded = []
    for lane in range(LANES):
        raw = (packed >> (lane * width)) & mask
        decoded.append(raw - (1 << width) if raw & sign else raw)
    return decoded


def bank_fetch(record: bytes, logical_base: int, terminal: int) -> Tuple[List[int], List[int]]:
    """Return physical-bank rows and logical words for one 256-bit fetch."""
    base_row, base_bank = divmod(logical_base, 8)
    rows = [base_row + (bank < base_bank) for bank in range(8)]
    bank_words = [payload_word(record, rows[bank] * 8 + bank, terminal)
                  for bank in range(8)]
    logical = [bank_words[(base_bank + index) & 7] for index in range(8)]
    direct = [payload_word(record, logical_base + index, terminal)
              for index in range(8)]
    if logical != direct:
        raise AssertionError("bank rotation did not reconstruct consecutive words")
    return rows, logical


def pack_values_bitwise(values: List[int], width: int) -> bytes:
    """Slow bit-by-bit synthetic packer, intentionally unlike integer decode."""
    bits = [0] * (len(values) * width)
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        raw = value & mask
        for bit in range(width):
            bits[lane * width + bit] = (raw >> bit) & 1
    output = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        output[index // 8] |= bit << (index & 7)
    return bytes(output)


def synthetic_vectors() -> dict:
    rng = random.Random(0x4D85)
    tested = Counter()
    dirty_words = Counter()
    for width in (8, 9, 10, 11):
        lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
        boundary = [lo, lo + 1, -2, -1, 0, 1, 2, hi - 1, hi]
        vectors = []
        for vector_index in range(40):
            values = []
            for lane in range(LANES):
                if vector_index < len(boundary):
                    values.append(boundary[(lane + vector_index) % len(boundary)])
                else:
                    values.append(rng.randint(lo, hi))
            vectors.append(values)
        code = width - 8
        for vector_index, values in enumerate(vectors):
            raw = pack_values_bitwise(values, width)
            raw += bytes((-len(raw)) % 4)
            source_words = [int.from_bytes(raw[index:index + 4], "little")
                            for index in range(0, len(raw), 4)]
            beats = BEATS[code]
            post_mask = []
            for beat in range(beats):
                fetched = source_words[beat * 8:(beat + 1) * 8]
                fetched += [0xA5000000 | (vector_index << 8) | word
                            for word in range(8 - len(fetched))]
                if beat == beats - 1:
                    keep = KEEP_WORDS[code]
                    dirty_words[width] += sum(word != 0 for word in fetched[keep:])
                    fetched[keep:] = [0] * (8 - keep)
                post_mask.extend(fetched)
            got = stream_decode(post_mask, width)
            if got != values:
                raise AssertionError(f"synthetic signed decode mismatch width={width}")
            tested[width] += 1
    return {
        "vectors_per_width": {str(width): tested[width] for width in tested},
        "total_vectors": sum(tested.values()),
        "dirty_padding_words_masked": {
            str(width): dirty_words[width] for width in dirty_words
        },
        "boundary_values_include_signed_min_max": True,
        "random_seed": "0x4d85",
    }


def parse_vcs_log(path: Optional[Path]) -> dict:
    if path is None:
        return {"provided": False}
    text = path.read_text()
    match = re.search(
        r"PASS M85 actual-record integration phases=(\d+) entries=(\d+) "
        r"outputs=(\d+) escape=(\d+) beats=(\d+) "
        r"masked_nonzero_words=(\d+) ii_checks=(\d+) metadata_poison_attacks=(\d+)",
        text,
    )
    covers = ["cp_phase_load", "cp_escape", "cp_width9", "cp_width10",
              "cp_width11", "cp_metadata_error"]
    return {
        "provided": True,
        "pass_line_found": match is not None,
        "pass_numbers": [int(value) for value in match.groups()] if match else [],
        "cover_text_found": {cover: cover in text for cover in covers},
        "failure_signature": bool(re.search(
            r"failed at|Offending|^Error|^Fatal|watchdog timeout", text,
            flags=re.IGNORECASE | re.MULTILINE)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sim-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observed_sha = {
        "records": sha256(args.records),
        "offsets": sha256(args.offsets),
        "metadata": sha256(args.metadata),
    }
    if observed_sha != EXPECTED_SHA:
        raise AssertionError(f"input SHA mismatch: {observed_sha}")
    records = args.records.read_bytes()
    offsets_raw = args.offsets.read_bytes()
    metadata_raw = args.metadata.read_bytes()
    if len(offsets_raw) != (PHASES + 1) * 4:
        raise AssertionError("offset table length mismatch")
    if len(metadata_raw) != PHASES * METADATA_BYTES:
        raise AssertionError("metadata length mismatch")
    offsets = list(struct.unpack(f"<{PHASES + 1}I", offsets_raw))
    if offsets[0] != 0 or offsets[-1] != len(records):
        raise AssertionError("offset endpoints mismatch")

    width_counts = Counter()
    beats_by_width = Counter()
    valid_words = Counter()
    masked_nonzero_words = 0
    output_count = 0
    all_vectors_checked = 0
    max_row = -1
    max_terminal = -1
    max_terminal_phase = -1
    metadata_audits = []
    escape_locations = []
    last_entry_boundaries = []
    address_reconstruction_checks = 0

    for phase in range(PHASES):
        record = records[offsets[phase]:offsets[phase + 1]]
        meta = metadata_raw[phase * METADATA_BYTES:(phase + 1) * METADATA_BYTES]
        if meta[:HEADER_BYTES] != record[:HEADER_BYTES]:
            raise AssertionError(f"header mismatch phase={phase}")
        codes = unpack_fields(meta[:HEADER_BYTES], ENTRIES, 3)
        bases = unpack_fields(meta[HEADER_BYTES:], PATTERNS, 13)
        if bases != pattern_bases(codes):
            raise AssertionError(f"pattern base mismatch phase={phase}")
        audit = audit_metadata(codes, bases)
        if not audit["accepted"]:
            raise AssertionError(f"metadata rejected phase={phase}: {audit}")
        metadata_audits.append(audit)
        terminal = sum(WORDS[code] for code in codes)
        if terminal > max_terminal:
            max_terminal, max_terminal_phase = terminal, phase
        required_record = HEADER_BYTES + terminal * 4
        if required_record > len(record):
            raise AssertionError(f"truncated record phase={phase}")

        cursor = 0
        for entry, code in enumerate(codes):
            width_counts[12 if code == 4 else code + 8] += 1
            output_count += 1
            if code == 4:
                escape_locations.append({
                    "phase": phase,
                    "entry": entry,
                    "pattern": entry // BLOCKS,
                    "block": entry % BLOCKS,
                    "cursor_before": cursor,
                    "cursor_after": cursor,
                    "previous_code": codes[entry - 1] if entry else None,
                    "next_code": codes[entry + 1] if entry + 1 < ENTRIES else None,
                })
                beats_by_width[12] += 1
                continue
            width = code + 8
            word_count = WORDS[code]
            beat_count = BEATS[code]
            beats_by_width[width] += beat_count
            valid_words[width] = KEEP_WORDS[code]
            payload = record[HEADER_BYTES + cursor * 4:
                             HEADER_BYTES + (cursor + word_count) * 4]
            expected = reference_decode(payload, width)
            post_mask_words = []
            for beat in range(beat_count):
                rows, logical = bank_fetch(record, cursor + beat * 8, terminal)
                max_row = max(max_row, *rows)
                if max(rows) >= ROWS:
                    raise AssertionError(f"row overflow phase={phase} entry={entry}")
                address_reconstruction_checks += 1
                if beat == beat_count - 1:
                    keep = KEEP_WORDS[code]
                    masked_nonzero_words += sum(word != 0 for word in logical[keep:])
                    logical[keep:] = [0] * (8 - keep)
                post_mask_words.extend(logical)
            got = stream_decode(post_mask_words, width)
            if got != expected:
                raise AssertionError(
                    f"signed output mismatch phase={phase} entry={entry} width={width}")
            all_vectors_checked += 1
            cursor += word_count
            if entry == ENTRIES - 1:
                last_entry_boundaries.append({
                    "phase": phase,
                    "code": code,
                    "start_word": cursor - word_count,
                    "terminal_word": cursor,
                    "last_fetch_end": cursor - word_count + FETCH_WORDS[code],
                })
        if cursor != terminal:
            raise AssertionError(f"cursor mismatch phase={phase}")

    if len(escape_locations) != 1:
        raise AssertionError(f"escape count {len(escape_locations)}")
    if masked_nonzero_words != 733459:
        raise AssertionError(f"pollution count {masked_nonzero_words}")
    if output_count != 221184 or all_vectors_checked != 221183:
        raise AssertionError("output/vector count mismatch")
    if sum(beats_by_width.values()) != 835383:
        raise AssertionError("beat count mismatch")

    # Parser attacks are generated independently of the producer metadata.
    zero_codes = [0] * ENTRIES
    parser_attacks = {}
    for reserved in (5, 6, 7):
        codes = zero_codes.copy()
        codes[0] = reserved
        result = audit_metadata(codes, pattern_bases(codes, invalid_as_zero=True))
        if result["accepted"] or not any(
                f"reserved_code[0]={reserved}" in reason
                for reason in result["reasons_first_12"]):
            raise AssertionError(f"reserved code {reserved} was not rejected")
        parser_attacks[f"reserved_predecessor_code_{reserved}"] = result
    wrong_bases = pattern_bases(zero_codes)
    wrong_bases[4] += 1
    parser_attacks["wrong_pattern4_base"] = audit_metadata(zero_codes, wrong_bases)
    wrong_last_base = pattern_bases(zero_codes)
    wrong_last_base[15] = 8191
    parser_attacks["pattern15_base_8191"] = audit_metadata(zero_codes, wrong_last_base)
    overflow_codes = [3] * ENTRIES
    parser_attacks["fetch_and_terminal_over_460x8"] = audit_metadata(
        overflow_codes, pattern_bases(overflow_codes))
    for name, result in parser_attacks.items():
        if result["accepted"]:
            raise AssertionError(f"parser attack accepted: {name}")

    max_boundary = max(last_entry_boundaries,
                       key=lambda row: row["terminal_word"])
    if max_boundary["terminal_word"] != max_terminal:
        raise AssertionError("maximum terminal boundary mismatch")
    if max_boundary["last_fetch_end"] > BUFFER_WORDS:
        raise AssertionError("legal maximum terminal fetch overflows")

    report = {
        "schema": "m85_guarded_wordpacked_independent_oracle_v1",
        "status": "PASS_INDEPENDENT_BINARY_MASK_ADDRESS_AND_SIGNED_ORACLE",
        "identity": {
            "input_sha256": observed_sha,
            "records_bytes": len(records),
            "offsets_bytes": len(offsets_raw),
            "metadata_bytes": len(metadata_raw),
        },
        "full_replay": {
            "phases": PHASES,
            "entries": output_count,
            "regular_vectors_checked": all_vectors_checked,
            "escape_controls": len(escape_locations),
            "outputs_expected": output_count,
            "beats_including_escape": sum(beats_by_width.values()),
            "within_phase_start_ii_checks_expected": output_count - PHASES,
            "width_counts": {str(key): width_counts[key]
                             for key in sorted(width_counts)},
            "beats_by_width": {str(key): beats_by_width[key]
                               for key in sorted(beats_by_width)},
        },
        "final_mask": {
            "valid_words_on_last_beat": {
                str(width): valid_words[width] for width in sorted(valid_words)
            },
            "nonzero_successor_words_that_require_mask": masked_nonzero_words,
            "mask_direction": "keep low logical words; zero higher logical words",
        },
        "address_oracle": {
            "bank_fetches_reconstructed": address_reconstruction_checks,
            "maximum_row_address": max_row,
            "rows_per_bank": ROWS,
            "all_addresses_in_range": max_row < ROWS,
            "method": "derive each physical word as row*8+bank, then rotate banks back into consecutive logical words",
        },
        "signed_output_oracle": {
            "actual_vectors_bit_exact": all_vectors_checked,
            "reference_method": "three-byte window per lane",
            "stream_method": "post-mask 32-bit-word integer extraction",
            "synthetic": synthetic_vectors(),
        },
        "metadata": {
            "all_actual_phases_accepted": all(audit["accepted"] for audit in metadata_audits),
            "maximum_terminal_words": max_terminal,
            "maximum_terminal_phase": max_terminal_phase,
            "maximum_fetch_end": max(audit["max_fetch_end"] for audit in metadata_audits),
            "pattern15_block7_max_terminal_boundary": max_boundary,
            "unique_escape": escape_locations[0],
            "parser_attacks": parser_attacks,
            "invalid_lookup_cases_required": [
                "regular lookup_beat equal to descriptor beat count",
                "escape lookup_beat other than zero",
                "lookup before phase load",
            ],
        },
        "vcs_rerun_log": parse_vcs_log(args.sim_log),
        "interpretation_limits": [
            "Python oracle is not a cycle-accurate SRAM or backpressure model",
            "VCS PASS counters come from the reviewed TB and are not an independent waveform trace",
            "address outputs are reconstructed here but the sealed TB does not consume DUT bank_row_addresses",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "outputs": output_count,
        "beats": sum(beats_by_width.values()),
        "pollution_words": masked_nonzero_words,
        "maximum_row": max_row,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
