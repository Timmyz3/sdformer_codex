#!/usr/bin/env python3
"""Generate sample-traceable H67 multisample T450 row vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

try:
    from scripts.generate_h67_checkpoint_row_vectors import (
        EXPECTED_BLOCKS,
        EXPECTED_HEADS,
        file_sha256,
        parse_record,
        validate_run_context,
    )
except ModuleNotFoundError:
    from generate_h67_checkpoint_row_vectors import (
        EXPECTED_BLOCKS,
        EXPECTED_HEADS,
        file_sha256,
        parse_record,
        validate_run_context,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_GENERATOR = SCRIPT_DIR / "generate_h67_checkpoint_row_vectors.py"


def expected_names() -> list[str]:
    return [
        f"S{stage}.B{block}.attn"
        for stage, depth in EXPECTED_BLOCKS.items()
        for block in range(depth)
    ]


def validate_record_sequence(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("records")
    names = expected_names()
    if not isinstance(records, list) or not records:
        raise ValueError("manifest records must be a non-empty list")
    if len(records) % len(names):
        raise ValueError("record count is not an integer number of all12 samples")
    sample_count = len(records) // len(names)
    if manifest.get("sample_limit") != sample_count:
        raise ValueError("sample_limit does not match complete sample count")
    if manifest.get("windows_per_call") != 1:
        raise ValueError("multisample evidence requires windows_per_call=1")
    if manifest.get("first_block_only") is not False:
        raise ValueError("multisample evidence requires first_block_only=false")

    samples: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for sample_id in range(sample_count):
        group = records[sample_id * len(names):(sample_id + 1) * len(names)]
        actual_names = [str(record.get("name", "")) for record in group]
        if actual_names != names:
            raise ValueError(
                f"sample {sample_id} all12 coverage/order mismatch: {actual_names}"
            )
        if any(record.get("sample_id") != sample_id for record in group):
            raise ValueError(f"sample {sample_id} has a discontinuous sample_id")
        if any(record.get("windows_captured") != 1 for record in group):
            raise ValueError(f"sample {sample_id} does not have one captured window")
        keys = {record.get("sample_key") for record in group}
        if len(keys) != 1:
            raise ValueError(f"sample {sample_id} has inconsistent sample_key values")
        sample_key = next(iter(keys))
        if not isinstance(sample_key, str) or not sample_key:
            raise ValueError(f"sample {sample_id} has an empty sample_key")
        if sample_key in seen_keys:
            raise ValueError(f"sample_key is reused across samples: {sample_key}")
        seen_keys.add(sample_key)
        samples.append(
            {"sample_id": sample_id, "sample_key": sample_key, "records": group}
        )
    return samples


def generate_vectors(
    manifest_path: Path,
    output_dir: Path,
    *,
    expected_tokens: int = 450,
    context_validator: Callable[[dict[str, Any], int], dict[str, Any]] =
        validate_run_context,
    record_parser: Callable[
        [dict[str, Any], int], tuple[dict[str, Any], list[dict[str, Any]]]
    ] = parse_record,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = validate_record_sequence(manifest)
    context = context_validator(manifest, expected_tokens)

    rows: list[dict[str, Any]] = []
    record_summaries: list[dict[str, Any]] = []
    for sample in samples:
        for record_order, record in enumerate(sample["records"]):
            summary, parsed_rows = record_parser(record, expected_tokens)
            stage = int(summary["stage"])
            block = int(summary["block"])
            expected_heads = EXPECTED_HEADS[stage]
            if (summary.get("name") != record["name"]
                    or int(summary.get("heads", -1)) != expected_heads
                    or len(parsed_rows) != expected_heads):
                raise ValueError(
                    f"parsed row count mismatch for sample {sample['sample_id']} "
                    f"{record['name']}"
                )
            enriched_summary = dict(summary)
            enriched_summary.update(
                {
                    "sample_id": sample["sample_id"],
                    "sample_key": sample["sample_key"],
                    "record_order": record_order,
                }
            )
            record_summaries.append(enriched_summary)
            for expected_head, row in enumerate(parsed_rows):
                if (int(row["stage"]) != stage
                        or int(row["block"]) != block
                        or int(row["head"]) != expected_head):
                    raise ValueError(
                        f"parsed head order mismatch for {record['name']} "
                        f"head {expected_head}"
                    )
                rows.append(
                    {
                        **row,
                        "sample_id": sample["sample_id"],
                        "sample_key": sample["sample_key"],
                        "record_order": record_order,
                    }
                )

    expected_rows = len(samples) * sum(
        EXPECTED_BLOCKS[stage] * heads
        for stage, heads in EXPECTED_HEADS.items()
    )
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / "h67_multisample_checkpoint_rows.txt"
    row_index_path = output_dir / "row_index.jsonl"
    with vector_path.open("w", encoding="ascii") as vector_handle, \
            row_index_path.open("w", encoding="ascii") as index_handle:
        vector_handle.write(f"{len(rows)} {expected_tokens}\n")
        for row_tag, row in enumerate(rows):
            vectors = row["vectors"]
            if len(vectors) != expected_tokens:
                raise ValueError(f"row {row_tag} token vector count mismatch")
            vector_handle.write(
                f"{row_tag} {row['stage']} {row['block']} {row['head']} "
                f"{row['expected_outputs']} {row['expected_folded']}\n"
            )
            for vector in vectors:
                vector_handle.write(
                    f"{int(vector['q']):08x} {int(vector['current_k']):08x} "
                    f"{int(vector['peer_k']):08x} {int(vector['gate'])}\n"
                )
            index_handle.write(
                json.dumps(
                    {
                        "row_tag": row_tag,
                        "sample_id": row["sample_id"],
                        "sample_key": row["sample_key"],
                        "stage": int(row["stage"]),
                        "block": int(row["block"]),
                        "head": int(row["head"]),
                        "record_order": row["record_order"],
                        "expected_outputs": int(row["expected_outputs"]),
                        "expected_folded": int(row["expected_folded"]),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ) + "\n"
            )

    output = {
        "schema": "h67_multisample_checkpoint_t450_vectors_v1",
        "status": "PASS",
        "evidence_boundary": (
            "vectors_only_not_rtl_result; source trace was produced by GPU, "
            "this transformation is CPU-only"
        ),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "run_context": context,
        "sample_count": len(samples),
        "sample_keys": [sample["sample_key"] for sample in samples],
        "tokens_per_row": expected_tokens,
        "rows_per_sample": expected_rows // len(samples),
        "row_count": len(rows),
        "token_vector_count": len(rows) * expected_tokens,
        "expected_active_outputs": sum(int(row["expected_outputs"]) for row in rows),
        "expected_folded_tokens": sum(int(row["expected_folded"]) for row in rows),
        "records": record_summaries,
        "artifacts": {
            "vector_file": str(vector_path),
            "vector_sha256": file_sha256(vector_path),
            "row_index": str(row_index_path),
            "row_index_sha256": file_sha256(row_index_path),
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": file_sha256(Path(__file__).resolve()),
            "legacy_semantic_generator": str(LEGACY_GENERATOR),
            "legacy_semantic_generator_sha256": file_sha256(LEGACY_GENERATOR),
        },
        "independent_reference_matches_trace": True,
    }
    output_path = output_dir / "manifest.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-tokens", type=int, default=450)
    args = parser.parse_args()
    result = generate_vectors(
        args.manifest, args.output_dir, expected_tokens=args.expected_tokens
    )
    print(Path(args.output_dir).resolve() / "manifest.json")
    print(
        f"PASS H67 multisample vectors samples={result['sample_count']} "
        f"rows={result['row_count']} tokens={result['tokens_per_row']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
