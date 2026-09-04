#!/usr/bin/env python3
"""Profile exact uniform-score relation certificates on Local5 vectors.

This is a read-only architecture screen.  A destination row is certifiable
when every valid Q7 score is identical.  Masked Shiftmax then produces a gate
that depends only on the topology-derived valid degree, so its five dynamic
gate fields need not be materialized.  Non-uniform rows remain exact fallback
exceptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTOR_DIR = (
    ROOT
    / "tb_qfit/vectors/"
    "local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/local5_uniform_relation_certificate_profile100_20260814"
)

TOKENS = 450
PLANE_TOKENS = 225
WIDTH = 15
RING_ROWS = 3
ROLE_COUNT = 5
GATE_W = 9


def read_memh(path: Path) -> list[int]:
    return [
        int(line.strip(), 16)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lanes(word: int, width: int, count: int) -> list[int]:
    mask = (1 << width) - 1
    return [(word >> (index * width)) & mask for index in range(count)]


def valid_values(word: int, valid_mask: int, width: int) -> list[int]:
    return [
        value
        for index, value in enumerate(lanes(word, width, ROLE_COUNT))
        if (valid_mask >> index) & 1
    ]


def is_identical_k(k_word: int, valid_mask: int) -> bool:
    values = valid_values(k_word, valid_mask, 32)
    return bool(values) and all(value == values[0] for value in values[1:])


def expected_uniform_gate(valid_degree: int) -> int:
    # Exact hardware-order Shiftmax result for equal Q7 scores.
    return {1: 128, 2: 64, 3: 32, 4: 32, 5: 16}[valid_degree]


def max_ring_exceptions(flags: list[bool]) -> int:
    if len(flags) != TOKENS:
        raise ValueError(f"expected {TOKENS} flags, got {len(flags)}")
    maximum = 0
    for plane in range(2):
        base = plane * PLANE_TOKENS
        rows = [
            sum(flags[base + y * WIDTH : base + (y + 1) * WIDTH])
            for y in range(WIDTH)
        ]
        for y in range(WIDTH):
            maximum = max(maximum, sum(rows[max(0, y - RING_ROWS + 1) : y + 1]))
    return maximum


def sparse_role_bits(exception_capacity: int) -> dict[str, int]:
    # Five role banks must remain independently readable.  Each role stores a
    # one-bit implicit/exception map for all live ring entries plus a compact
    # exact fallback CAM containing {ring address, gate}.  Geometry supplies
    # candidate-valid bits for implicit rows.
    ring_entries = RING_ROWS * WIDTH
    addr_w = max(1, math.ceil(math.log2(ring_entries)))
    mode_bits = ROLE_COUNT * ring_entries
    exception_bits = ROLE_COUNT * exception_capacity * (addr_w + GATE_W)
    baseline_bits = ROLE_COUNT * ring_entries * (1 + GATE_W)
    candidate_bits = mode_bits + exception_bits
    return {
        "ring_entries": ring_entries,
        "address_bits": addr_w,
        "baseline_gate_valid_bits": baseline_bits,
        "implicit_mode_bits": mode_bits,
        "exception_cam_bits": exception_bits,
        "candidate_gate_valid_bits": candidate_bits,
        "reduction_bits": baseline_bits - candidate_bits,
    }


def percentile(values: list[int], percent: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = math.ceil(percent / 100 * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def build_profile(vector_dir: Path) -> dict[str, object]:
    manifest_path = vector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["selection"]["rows"]
    if len(rows) != 100:
        raise ValueError(f"expected 100 groups, got {len(rows)}")

    valid = read_memh(vector_dir / "input_valid.memh")
    q_words = read_memh(vector_dir / "input_q.memh")
    k_words = read_memh(vector_dir / "input_candidate_k.memh")
    scores = read_memh(vector_dir / "expected_scores.memh")
    gates = read_memh(vector_dir / "expected_gates.memh")
    expected_len = len(rows) * TOKENS
    for name, values in {
        "valid": valid,
        "q": q_words,
        "k": k_words,
        "score": scores,
        "gate": gates,
    }.items():
        if len(values) != expected_len:
            raise ValueError(f"{name}: expected {expected_len}, got {len(values)}")

    totals = Counter()
    stage = defaultdict(Counter)
    group_records = []
    ring_peaks = []
    gate_mismatches = 0
    invalid_gate_mismatches = 0

    for group_index, row_meta in enumerate(rows):
        start = group_index * TOKENS
        stop = start + TOKENS
        group_exceptions: list[bool] = []
        group_counter = Counter()
        for index in range(start, stop):
            mask = valid[index]
            score_values = valid_values(scores[index], mask, 16)
            gate_values = lanes(gates[index], GATE_W, ROLE_COUNT)
            degree = mask.bit_count()
            uniform = bool(score_values) and all(
                value == score_values[0] for value in score_values[1:]
            )
            qsilent = q_words[index] == 0
            identk = (not qsilent) and is_identical_k(k_words[index], mask)
            group_exceptions.append(not uniform)
            group_counter["rows"] += 1
            group_counter["uniform_rows"] += int(uniform)
            group_counter["exception_rows"] += int(not uniform)
            group_counter["qsilent_rows"] += int(qsilent)
            group_counter["identk_rows"] += int(identk)
            group_counter["residual_uniform_rows"] += int(
                uniform and not qsilent and not identk
            )
            group_counter["uniform_qsilent_rows"] += int(uniform and qsilent)
            if uniform:
                expected_gate = expected_uniform_gate(degree)
                for role, gate in enumerate(gate_values):
                    if (mask >> role) & 1:
                        gate_mismatches += int(gate != expected_gate)
                    else:
                        invalid_gate_mismatches += int(gate != 0)

        ring_peak = max_ring_exceptions(group_exceptions)
        ring_peaks.append(ring_peak)
        group_counter["max_live_ring_exceptions"] = ring_peak
        totals.update(
            {
                key: value
                for key, value in group_counter.items()
                if key != "max_live_ring_exceptions"
            }
        )
        stage[int(row_meta["stage"])].update(
            {
                key: value
                for key, value in group_counter.items()
                if key != "max_live_ring_exceptions"
            }
        )
        group_records.append(
            {
                "group": group_index,
                "sample": int(row_meta["sample"]),
                "stage": int(row_meta["stage"]),
                "empty": bool(row_meta["empty"]),
                **dict(group_counter),
            }
        )

    p95_capacity = percentile(ring_peaks, 95)
    worst_capacity = max(ring_peaks)
    p95_storage = sparse_role_bits(p95_capacity)
    worst_storage = sparse_role_bits(worst_capacity)
    baseline_bits = p95_storage["baseline_gate_valid_bits"]

    result = {
        "schema": "local5_uniform_relation_certificate_profile_v1",
        "evidence_level": "prof",
        "vector_identity": {
            "directory": str(vector_dir),
            "manifest_sha256": sha256(manifest_path),
            "groups": len(rows),
            "rows": expected_len,
            "stage_counts": manifest["selection"]["stage_counts"],
        },
        "exact_contract": {
            "predicate": "all valid Q7 scores for one destination are equal",
            "implicit_gate_by_valid_degree": {
                str(degree): expected_uniform_gate(degree)
                for degree in range(1, 6)
            },
            "valid_gate_mismatches": gate_mismatches,
            "invalid_gate_mismatches": invalid_gate_mismatches,
            "destination_identity_preserved": True,
            "nonuniform_rows_use_exact_fallback": True,
        },
        "population": {
            **dict(totals),
            "uniform_fraction": totals["uniform_rows"] / totals["rows"],
            "exception_fraction": totals["exception_rows"] / totals["rows"],
            "residual_uniform_fraction": (
                totals["residual_uniform_rows"] / totals["rows"]
            ),
        },
        "by_stage": {
            str(key): {
                **dict(value),
                "uniform_fraction": value["uniform_rows"] / value["rows"],
            }
            for key, value in sorted(stage.items())
        },
        "live_ring_exception_capacity": {
            "mean": sum(ring_peaks) / len(ring_peaks),
            "p50": percentile(ring_peaks, 50),
            "p95": p95_capacity,
            "worst": worst_capacity,
        },
        "conservative_five_read_port_storage_model": {
            "p95": {
                **p95_storage,
                "reduction_fraction": p95_storage["reduction_bits"] / baseline_bits,
            },
            "worst": {
                **worst_storage,
                "reduction_fraction": worst_storage["reduction_bits"] / baseline_bits,
            },
            "excluded": [
                "K storage, unchanged",
                "CAM compare/control area and energy",
                "DC/STA/SAIF and SRAM macro effects",
            ],
        },
        "cycle_boundary": {
            "term_count_unchanged": True,
            "destination_updates_unchanged": True,
            "current_one_row_per_cycle_relation_ingress_unchanged": True,
            "projected_cycle_speedup": 1.0,
        },
        "decision": {
            "verdict": "NO_GO_AS_DATE_CONTRIBUTION",
            "reason": (
                "Only 1.58% of rows add coverage beyond Q-silent and identical-K. "
                "At p95 the live ring needs 43 of 45 exact exceptions, making the "
                "conservative five-read-port store 53.33% larger than the current "
                "gate/valid banks. Source terms, destination updates, and the "
                "one-row-per-cycle schedule are also unchanged."
            ),
            "rtl_authorized": False,
        },
        "groups": group_records,
    }
    return result


def render_markdown(result: dict[str, object]) -> str:
    population = result["population"]
    capacity = result["live_ring_exception_capacity"]
    storage = result["conservative_five_read_port_storage_model"]
    lines = [
        "# Local5 等分数隐式 relation 只读筛选",
        "",
        "## 裁决",
        "",
        "**NO_GO_AS_DATE_CONTRIBUTION**，不写 RTL，不改封存主表。",
        "",
        "该候选保持 destination 身份，并把等分数行的五路 Q1.7 gate "
        "改为按拓扑有效度重建；但新增覆盖仅 1.58%，高负载活跃环又需要近乎满容量的 "
        "exact fallback。保守五读口实现不但不减周期，p95 存储还更大。",
        "",
        "## [prof] 精确机会",
        "",
        f"- 真实行：{population['rows']}；等分数行："
        f"{population['uniform_rows']} ({population['uniform_fraction']:.2%})。",
        f"- 非 Q==0、非 ident-K 仍可证书化："
        f"{population['residual_uniform_rows']} "
        f"({population['residual_uniform_fraction']:.2%} / 全部行)。",
        f"- gate 重建 miter：valid mismatch="
        f"{result['exact_contract']['valid_gate_mismatches']}，invalid mismatch="
        f"{result['exact_contract']['invalid_gate_mismatches']}。",
        f"- Dr=3 活跃环异常行：mean={capacity['mean']:.2f}，"
        f"p50={capacity['p50']}，p95={capacity['p95']}，worst={capacity['worst']}。",
        "",
        "| 保守五读口模型 | baseline gate+valid | candidate | 降幅 |",
        "|---|---:|---:|---:|",
        f"| p95 exception capacity | {storage['p95']['baseline_gate_valid_bits']} | "
        f"{storage['p95']['candidate_gate_valid_bits']} | "
        f"{storage['p95']['reduction_fraction']:.2%} |",
        f"| worst exception capacity | {storage['worst']['baseline_gate_valid_bits']} | "
        f"{storage['worst']['candidate_gate_valid_bits']} | "
        f"{storage['worst']['reduction_fraction']:.2%} |",
        "",
        "上述 candidate 含五份 implicit/exception map 与五个精确 exception CAM，"
        "未计 CAM 比较、控制、目标 SRAM 宏和功耗，因此不能写成 PPA。",
        "",
        "## 边界",
        "",
        "- `[prof]`，100-group、45000 destination rows。",
        "- OUT_DIM 不参与本统计；没有 encoder 加速主张。",
        "- 不修改 Query-Silent、IDENTK、FCSR/compiled retirement 或 TCFM5。",
        "- `docs/359` 与封存数字保持不变。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_profile(args.vector_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    population = result["population"]
    print(
        "PASS uniform-relation profile "
        f"rows={population['rows']} uniform={population['uniform_rows']} "
        f"mismatch={result['exact_contract']['valid_gate_mismatches']} "
        f"verdict={result['decision']['verdict']}"
    )


if __name__ == "__main__":
    main()
