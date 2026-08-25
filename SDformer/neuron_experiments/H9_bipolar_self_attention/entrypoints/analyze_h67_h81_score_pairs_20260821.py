#!/usr/bin/env python3
"""Compare H67 and H81 score equality on matched real Q/K bit traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(payload: Any, key: str) -> np.ndarray:
    shape = tuple(int(value) for value in payload[f"{key}_shape"])
    count = int(np.prod(shape))
    return np.unpackbits(
        payload[f"{key}_bits_packed"], bitorder="little", count=count
    ).reshape(shape).astype(bool)


def rne_div_pow2(values: np.ndarray, denominator: int) -> np.ndarray:
    quotient = values // denominator
    remainder = values % denominator
    half = denominator // 2
    return quotient + (
        (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
    )


def counters(q: np.ndarray, k: np.ndarray, motion: bool) -> dict[str, int]:
    q_count = q.sum(axis=-1, dtype=np.int64)
    k_count = k.sum(axis=-1, dtype=np.int64)
    overlap = (q & k).sum(axis=-1, dtype=np.int64)
    same_zero = q.shape[-1] - q_count - k_count + overlap
    numerator = 64 * overlap + same_zero
    if motion:
        motion_count = (k[0] ^ k[1]).sum(axis=-1, dtype=np.int64)
        numerator = numerator + 16 * motion_count[None, ...]
    score = rne_div_pow2(numerator, 16)
    equal = score[0] == score[1]
    non_empty = (q_count + k_count).sum(axis=0) != 0
    both_k_active = (k_count[0] != 0) & (k_count[1] != 0)
    categories = {
        "all": np.ones_like(equal, dtype=bool),
        "non_empty": non_empty,
        "both_k_active": both_k_active,
    }
    result: dict[str, int] = {}
    for category, mask in categories.items():
        total = int(mask.sum())
        same = int((equal & mask).sum())
        result[f"{category}_pairs"] = total
        result[f"{category}_equal"] = same
    return result


def add(target: dict[str, int], values: dict[str, int]) -> None:
    for key, value in values.items():
        target[key] += int(value)


def finalize(values: dict[str, int]) -> dict[str, float | int]:
    result: dict[str, float | int] = dict(values)
    for category in ("all", "non_empty", "both_k_active"):
        total = values[f"{category}_pairs"]
        equal = values[f"{category}_equal"]
        result[f"{category}_equal_ratio"] = equal / total if total else 0.0
        result[f"{category}_ideal_slot_reduction"] = (
            equal / (2 * total) if total else 0.0
        )
    return result


def parse_manifest(trace_dir: Path, sample_limit: int) -> dict[str, Any]:
    manifest_path = trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for record in manifest.get("records", []):
        if int(record["sample_id"]) >= sample_limit:
            continue
        path = Path(record["file"])
        if not path.is_absolute():
            path = trace_dir / path
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"trace SHA mismatch: {path}")
        records.append((record, path))
    return {"manifest": manifest, "manifest_path": manifest_path, "records": records}


def analyze_route(route: str, parsed: dict[str, Any], motion: bool) -> dict[str, Any]:
    overall: dict[str, int] = defaultdict(int)
    by_stage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    samples: set[int] = set()
    names: set[str] = set()
    for record, path in parsed["records"]:
        with np.load(path, allow_pickle=False) as payload:
            q = unpack(payload, "q")
            k = unpack(payload, "k")
        values = counters(q, k, motion)
        stage = str(record["name"]).split(".")[0]
        add(overall, values)
        add(by_stage[stage], values)
        samples.add(int(record["sample_id"]))
        names.add(str(record["name"]))
    return {
        "route": route,
        "score_operator": "H67 Motion-TTX Q7" if motion else "H81 TTX Q7",
        "sample_ids": sorted(samples),
        "record_count": len(parsed["records"]),
        "attention_blocks": sorted(names),
        "overall": finalize(overall),
        "by_stage": {key: finalize(by_stage[key]) for key in sorted(by_stage)},
        "trace_manifest": str(parsed["manifest_path"].resolve()),
        "trace_manifest_sha256": sha256(parsed["manifest_path"]),
        "run_context": parsed["manifest"].get("run_context"),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# H67/H81 matched score-pair profile",
        "",
        "This profile uses the first captured window of every attention block. "
        "It is real-Q/K trace evidence, not a full-valid825 population estimate.",
        "",
        "| route | scope | pairs | equal | equality | ideal dual-slot reduction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for route in result["routes"]:
        for scope in ("all", "non_empty", "both_k_active"):
            row = route["overall"]
            lines.append(
                f"| {route['route']} | {scope} | {row[f'{scope}_pairs']} | "
                f"{row[f'{scope}_equal']} | {row[f'{scope}_equal_ratio']:.4%} | "
                f"{row[f'{scope}_ideal_slot_reduction']:.4%} |"
            )
    lines += ["", "## Stage breakdown", ""]
    lines += [
        "| route | stage | all equality | non-empty equality | both-K-active equality |",
        "|---|---|---:|---:|---:|",
    ]
    for route in result["routes"]:
        for stage, row in route["by_stage"].items():
            lines.append(
                f"| {route['route']} | {stage} | {row['all_equal_ratio']:.4%} | "
                f"{row['non_empty_equal_ratio']:.4%} | "
                f"{row['both_k_active_equal_ratio']:.4%} |"
            )
    lines += [
        "",
        "`both_k_active` means both temporal K slices contain at least one active lane. "
        "H81 uses the TTX score; H67 uses the Motion-TTX score. Cross-route differences "
        "are descriptive because the checkpoints are recipe-level matched, not step-paired.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h67-trace-dir", required=True, type=Path)
    parser.add_argument("--h81-trace-dir", required=True, type=Path)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    h67 = parse_manifest(args.h67_trace_dir.resolve(), args.sample_limit)
    h81 = parse_manifest(args.h81_trace_dir.resolve(), args.sample_limit)
    h67_route = analyze_route("H67", h67, motion=True)
    h81_route = analyze_route("H81", h81, motion=False)
    if h67_route["sample_ids"] != h81_route["sample_ids"]:
        raise RuntimeError("H67/H81 sample IDs are not matched")
    if h67_route["attention_blocks"] != h81_route["attention_blocks"]:
        raise RuntimeError("H67/H81 attention block coverage is not matched")
    result = {
        "schema": "h67_h81_score_pair_profile_v1",
        "sample_limit": args.sample_limit,
        "matched_sample_ids": h67_route["sample_ids"],
        "routes": [h67_route, h81_route],
        "claim_boundary": (
            "first-window-per-block real-QK trace; recipe-level checkpoint comparison; "
            "not full-valid825 and not step-paired causality"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), result)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
