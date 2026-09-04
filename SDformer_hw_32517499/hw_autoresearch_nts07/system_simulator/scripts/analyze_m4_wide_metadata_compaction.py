#!/usr/bin/env python3
"""Size the bank-coherent metadata optimization used by the M4 wide state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(
    contexts: int = 4,
    base_tiles: int = 32,
    banks: int = 6,
    lanes_per_bank: int = 16,
    acc_bits: int = 32,
    epoch_bits: int = 16,
    domain_bits: int = 32,
    step_bits: int = 4,
    length_bits: int = 4,
    tag_bits: int = 32,
) -> dict[str, Any]:
    values = (
        contexts, base_tiles, banks, lanes_per_bank, acc_bits, epoch_bits,
        domain_bits, step_bits, length_bits, tag_bits,
    )
    if any(value <= 0 for value in values):
        raise ValueError("metadata-compaction geometry must be positive")
    fields = {
        "epoch": epoch_bits,
        "domain": domain_bits,
        "next_step": step_bits,
        "sequence_length": length_bits,
        "sequence_tag": tag_bits,
        "epoch_initialized": 1,
        "state_valid": 1,
        "sequence_open": 1,
    }
    metadata_bits_per_row = sum(fields.values())
    rows = contexts * base_tiles
    legacy_metadata = rows * banks * metadata_bits_per_row
    shared_metadata = rows * metadata_bits_per_row
    data_bits = rows * banks * lanes_per_bank * acc_bits
    saving = legacy_metadata - shared_metadata
    return {
        "geometry": {
            "contexts": contexts,
            "base_tiles": base_tiles,
            "banks": banks,
            "lanes_per_bank": lanes_per_bank,
            "acc_bits": acc_bits,
            "rows": rows,
        },
        "metadata_fields": fields,
        "metadata_bits_per_row": metadata_bits_per_row,
        "legacy_per_bank_metadata_bits": legacy_metadata,
        "shared_wide_metadata_bits": shared_metadata,
        "metadata_bit_reduction": saving,
        "metadata_reduction_fraction": saving / legacy_metadata,
        "state_data_bits_unchanged": data_bits,
        "legacy_data_plus_metadata_bits": data_bits + legacy_metadata,
        "shared_data_plus_metadata_bits": data_bits + shared_metadata,
        "persistent_destination_data_plus_metadata_reduction_fraction": saving /
            (data_bits + legacy_metadata),
        "rtl_source_admission_comparison_replication": {
            "legacy": banks,
            "shared_wide": 1,
            "reduction_fraction": (banks - 1) / banks,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "m4_wide_metadata_compaction_audit_v1",
        "status": "PASS_M4_WIDE_METADATA_BIT_AUDIT_PRE_DC",
        "claim_boundary": (
            "Exact RTL logical storage inventory for the default parameters. "
            "It assumes M4's admitted contract that all six banks share one "
            "temporal identity. The 12.576% denominator contains only persistent "
            "destination-bank data plus temporal metadata; it is not total "
            "accelerator state. Comparison replication is a source-level RTL "
            "inventory until DC. Bit counts are not standard-cell area, SRAM "
            "area, energy, timing, or full-system speedup."
        ),
        **analyze(),
        "script_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"PASS M4 wide metadata audit -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
