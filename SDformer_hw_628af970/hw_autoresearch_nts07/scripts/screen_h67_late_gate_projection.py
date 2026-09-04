#!/usr/bin/env python3
"""Screen exact late-gate projection against direct and gate-resident baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.generate_h67_real_weight_projection2_vectors import parse_base_vectors
except ModuleNotFoundError:
    from generate_h67_real_weight_projection2_vectors import parse_base_vectors


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT / "results/h67_fcip_ep35_fullres_t450_profile_20260813/report.json"
)
DEFAULT_REALW = (
    ROOT
    / "tb_h67/vectors/h67_ep35_real_weight_projection2_20260813/"
    "h67_real_weight_projection2.txt"
)
DEFAULT_BASE = (
    ROOT
    / "tb_h67/vectors/h67_fullres_ep35_postconvergence_t450_20260805/"
    "h67_checkpoint_rows.txt"
)
DEFAULT_OUT = ROOT / "results/h67_late_gate_projection_screen_20260814"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_real_weight_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="ascii") as handle:
        header = handle.readline().split()
        if header != ["138", "2"]:
            raise ValueError(f"unexpected projection2 header: {header}")
        for expected_row in range(138):
            fields = handle.readline().split()
            if len(fields) != 6 or int(fields[0]) != expected_row:
                raise ValueError(f"invalid projection2 row header {expected_row}")
            weights = [[], []]
            for lane in range(32):
                values = handle.readline().split()
                if len(values) != 2:
                    raise ValueError(
                        f"invalid projection2 weights row={expected_row} lane={lane}"
                    )
                weights[0].append(int(values[0]))
                weights[1].append(int(values[1]))
            rows.append(
                {
                    "row": int(fields[0]),
                    "stage": int(fields[1]),
                    "block": int(fields[2]),
                    "head": int(fields[3]),
                    "expected": [int(fields[4]), int(fields[5])],
                    "weights": weights,
                }
            )
        if handle.read().strip():
            raise ValueError("projection2 vector has trailing data")
    return rows


def project_direct(vectors: list[dict[str, int]], weights: list[int]) -> int:
    total = 0
    for token in vectors:
        for lane in range(32):
            if (token["k"] >> lane) & 1:
                total += int(token["gate"]) * int(weights[lane])
    return total


def project_late_gate(vectors: list[dict[str, int]], weights: list[int]) -> int:
    total = 0
    for token in vectors:
        selected_weight_sum = sum(
            int(weights[lane])
            for lane in range(32)
            if (token["k"] >> lane) & 1
        )
        total += int(token["gate"]) * selected_weight_sum
    return total


def row_counts(vectors: list[dict[str, int]]) -> dict[str, int]:
    active_tokens = 0
    active_lane_events = 0
    gate_lane_keys: set[tuple[int, int]] = set()
    for token in vectors:
        k_bits = int(token["k"])
        if k_bits == 0:
            continue
        active_tokens += 1
        for lane in range(32):
            if (k_bits >> lane) & 1:
                active_lane_events += 1
                gate_lane_keys.add((int(token["gate"]), lane))
    return {
        "active_tokens": active_tokens,
        "active_lane_events": active_lane_events,
        "final_gate_lane_terms": len(gate_lane_keys),
    }


def traffic_model(
    *, active_tokens: int, active_lane_events: int, gate_lane_terms: int, out_dim: int
) -> dict[str, Any]:
    weight_w = 8
    gate_w = 9
    product_w = weight_w + gate_w
    sum_w = 14
    direct_mul = active_lane_events * out_dim
    late_mul = active_tokens * out_dim
    resident_mul = gate_lane_terms * out_dim
    direct_weight_bits = active_lane_events * weight_w * out_dim
    late_weight_bits = direct_weight_bits
    resident_weight_bits = gate_lane_terms * weight_w * out_dim
    resident_product_write_bits = gate_lane_terms * product_w * out_dim
    resident_product_read_bits = active_lane_events * product_w * out_dim
    resident_array_bits = (
        resident_weight_bits
        + resident_product_write_bits
        + resident_product_read_bits
    )
    return {
        "out_dim": out_dim,
        "direct": {
            "multiply_starts": direct_mul,
            "weight_read_bits": direct_weight_bits,
        },
        "late_gate": {
            "multiply_starts": late_mul,
            "selected_weight_adds": direct_mul,
            "weight_read_bits": late_weight_bits,
            "streaming_partial_state_bits": sum_w * out_dim,
        },
        "row_resident_gate_weight": {
            "multiply_starts": resident_mul,
            "weight_read_bits": resident_weight_bits,
            "product_write_bits": resident_product_write_bits,
            "product_read_bits": resident_product_read_bits,
            "modeled_data_array_bits": resident_array_bits,
        },
        "ratios": {
            "late_multiply_reduction_vs_direct": 1.0 - late_mul / direct_mul,
            "late_multiply_ratio_vs_resident": late_mul / resident_mul,
            "late_weight_read_ratio_vs_resident": (
                late_weight_bits / resident_weight_bits
            ),
            "late_data_array_reduction_vs_materialized_resident": (
                1.0 - late_weight_bits / resident_array_bits
            ),
        },
    }


def run(profile_path: Path, realw_path: Path, base_path: Path) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_rows = profile.get("rows", [])
    base_rows = parse_base_vectors(base_path)
    realw_rows = parse_real_weight_rows(realw_path)
    if not (len(profile_rows) == len(base_rows) == len(realw_rows) == 138):
        raise ValueError("expected 138 aligned rows")

    totals = defaultdict(int)
    stage_totals: dict[int, defaultdict[str, int]] = {
        stage: defaultdict(int) for stage in range(4)
    }
    checked = 0
    for profile_row, base_row, realw_row in zip(
        profile_rows, base_rows, realw_rows, strict=True
    ):
        identity = (base_row["row"], base_row["stage"], base_row["block"], base_row["head"])
        realw_identity = (
            realw_row["row"],
            realw_row["stage"],
            realw_row["block"],
            realw_row["head"],
        )
        expected_name = f"S{base_row['stage']}.B{base_row['block']}.attn"
        if identity != realw_identity or profile_row.get("name") != expected_name:
            raise ValueError(f"row identity mismatch: {identity}")
        if int(profile_row.get("head", -1)) != base_row["head"]:
            raise ValueError(f"head identity mismatch: {identity}")

        counts = row_counts(base_row["vectors"])
        for key, value in counts.items():
            if int(profile_row.get(key, -1)) != value:
                raise ValueError(f"profile count mismatch row={identity} key={key}")
            totals[key] += value
            stage_totals[base_row["stage"]][key] += value

        for channel in range(2):
            direct = project_direct(base_row["vectors"], realw_row["weights"][channel])
            late = project_late_gate(
                base_row["vectors"], realw_row["weights"][channel]
            )
            expected = int(realw_row["expected"][channel])
            if direct != late or late != expected:
                raise ValueError(
                    f"projection mismatch row={identity} channel={channel}: "
                    f"direct={direct} late={late} expected={expected}"
                )
            checked += 1

    models = {
        str(out_dim): traffic_model(
            active_tokens=totals["active_tokens"],
            active_lane_events=totals["active_lane_events"],
            gate_lane_terms=totals["final_gate_lane_terms"],
            out_dim=out_dim,
        )
        for out_dim in (2, 32)
    }
    return {
        "schema": "h67_late_gate_projection_screen_v1",
        "status": "NO_GO_AS_NEW_DATE_CANDIDATE_KEEP_AS_IMPLEMENTATION_DSE",
        "evidence": "[prof]+[real-checkpoint-int8 software miter]+[model]",
        "scope": (
            "ep35 sample0/window0 all12, 138 T450 head-row; not multisample RTL, "
            "cycle, energy, DC, or full encoder"
        ),
        "exact_contract": (
            "sum_l gate*K_l*W_l,o == gate*sum_{l:K_l=1} W_l,o under integer "
            "Acc32 with no intermediate rounding/saturation"
        ),
        "coverage": {
            "rows": 138,
            "real_weight_acc32_values": checked,
            "mismatches": 0,
        },
        "totals": dict(totals),
        "per_stage": {str(stage): dict(values) for stage, values in stage_totals.items()},
        "models": models,
        "decision": {
            "positive": [
                "Late-gate changes the arithmetic object to one selected-weight partial per token.",
                "It removes most direct lane-wise gate multiplications without a product SRAM.",
            ],
            "blocking": [
                "The same integer distributive law and ttx_late_gate_accum leaf already exist in the repository.",
                "When executed only after gate availability it does not move the cross-stage boundary; it is datapath strength reduction.",
                "A row-resident gate-weight/FCIP baseline uses fewer weight reads and multiply starts, at the price of bitmap/product state and scatter control.",
                "No matched cycle, SAIF, target-memory, or full-output Pareto evidence exists.",
            ],
            "reopen_only_if": [
                "same-port RTL beats direct and row-resident/FCIP baselines by at least 15% EDP in every stage mean and p95",
                "total state is no more than 1.10x the strongest baseline",
                "no extra K reread and real-weight Acc32 remains bit-exact",
            ],
        },
        "inputs": {
            "profile": str(profile_path.resolve()),
            "profile_sha256": sha256_file(profile_path),
            "real_weight_vector": str(realw_path.resolve()),
            "real_weight_vector_sha256": sha256_file(realw_path),
            "base_vector": str(base_path.resolve()),
            "base_vector_sha256": sha256_file(base_path),
        },
    }


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    model = report["models"]["2"]
    ratios = model["ratios"]
    lines = [
        "# Motion late-gate projection go/no-go",
        "",
        f"裁决：`{report['status']}`。",
        "",
        "## 证据",
        "",
        f"- 范围：{report['scope']}；",
        f"- 真实 INT8 Acc32：{report['coverage']['real_weight_acc32_values']} 个，0 mismatch；",
        f"- active token / lane event / unique gate-lane："
        f"{report['totals']['active_tokens']:,} / "
        f"{report['totals']['active_lane_events']:,} / "
        f"{report['totals']['final_gate_lane_terms']:,}。",
        "",
        "## OUT2 模型",
        "",
        "| 对象 | direct | late-gate | row-resident gate-weight |",
        "|---|---:|---:|---:|",
        f"| multiply start | {model['direct']['multiply_starts']:,} | "
        f"{model['late_gate']['multiply_starts']:,} | "
        f"{model['row_resident_gate_weight']['multiply_starts']:,} |",
        f"| weight read bit | {model['direct']['weight_read_bits']:,} | "
        f"{model['late_gate']['weight_read_bits']:,} | "
        f"{model['row_resident_gate_weight']['weight_read_bits']:,} |",
        "",
        f"late-gate 相对 direct 的 multiply start 降低 "
        f"`{ratios['late_multiply_reduction_vs_direct']:.2%}`，但相对 row-resident "
        f"强下界仍多 `{ratios['late_multiply_ratio_vs_resident']:.3f}x` multiply 和 "
        f"`{ratios['late_weight_read_ratio_vs_resident']:.3f}x` weight read。",
        "",
        "## 为什么不晋级",
        "",
        "该等式与仓库既有 `ttx_late_gate_accum` 相同；gate 得到后再换序只属于"
        "datapath strength reduction，并未前移 normalization-to-projection 边界。"
        "若在 gate 前预计算并保存 token projection，则状态随输出通道膨胀且仍需 K fallback。",
        "",
        "因此只保留为实现 DSE，不新增贡献名、不写 RTL、不改 `docs/359`。",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--real-weight-vector", type=Path, default=DEFAULT_REALW)
    parser.add_argument("--base-vector", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = run(args.profile, args.real_weight_vector, args.base_vector)
    write_report(report, args.output_dir)
    print(
        "PASS H67 late-gate screen "
        f"rows={report['coverage']['rows']} acc32={report['coverage']['real_weight_acc32_values']} "
        f"status={report['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
