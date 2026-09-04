#!/usr/bin/env python3
"""Independently bind Local5 source-owned gate quotienting to production RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


HEIGHT = 15
WIDTH = 15
PLANES = 2
SOURCES = HEIGHT * WIDTH * PLANES
HEAD_DIM = 32
ROLES = 5
GATE_W = 9
OUT_DIM = 2
PLANE_TOKENS = HEIGHT * WIDTH

# Candidate order is self, up, down, left, right from destination to source.
DEST_TO_SOURCE = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
GROUP_RE = re.compile(
    r"^GROUP .* group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"score_rows=(?P<score_rows>\d+) score_service=(?P<score_service>\d+) "
    r"score_direct_rows=(?P<score_direct>\d+) qsilent_rows=(?P<qsilent>\d+) "
    r"identk_rows=(?P<identk>\d+) overlap=(?P<overlap>\d+) "
    r"active=(?P<active>\d+).* terms=(?P<terms>\d+) updates=(?P<updates>\d+)"
)
PASS_RE = re.compile(
    r"^PASS Local5 score-to-projection .* groups=100 total_cycles=(?P<cycles>\d+)"
)
BAD_RE = re.compile(r"%Error|Assertion failed|MISMATCH|\$fatal|\bFAIL\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_memh(path: Path) -> list[int]:
    return [
        int(line.strip(), 16)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]


def signed(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    return value - (1 << width) if value & (1 << (width - 1)) else value


def source_for(destination: int, role: int) -> int | None:
    plane, within = divmod(destination, PLANE_TOKENS)
    y, x = divmod(within, WIDTH)
    dy, dx = DEST_TO_SOURCE[role]
    sy, sx = y + dy, x + dx
    if not (0 <= sy < HEIGHT and 0 <= sx < WIDTH):
        return None
    return plane * PLANE_TOKENS + sy * WIDTH + sx


def destination_for(source: int, role: int) -> int | None:
    plane, within = divmod(source, PLANE_TOKENS)
    y, x = divmod(within, WIDTH)
    dy, dx = DEST_TO_SOURCE[role]
    destination_y, destination_x = y - dy, x - dx
    if not (
        0 <= destination_y < HEIGHT and 0 <= destination_x < WIDTH
    ):
        return None
    return plane * PLANE_TOKENS + destination_y * WIDTH + destination_x


def analyze_group(
    *,
    candidate_k: list[int],
    valid_mask: list[int],
    packed_gates: list[int],
    weights: list[list[int]],
) -> dict[str, Any]:
    if not (
        len(candidate_k) == len(valid_mask) == len(packed_gates) == SOURCES
    ):
        raise ValueError("group does not contain 450 destination rows")
    if len(weights) != HEAD_DIM or any(len(row) != OUT_DIM for row in weights):
        raise ValueError("weight shape is not 32x2")

    source_k: list[int | None] = [None] * SOURCES
    source_roles: list[list[tuple[int, int]]] = [[] for _ in range(SOURCES)]
    valid_edges = 0
    nonzero_gate_edges = 0
    relation_lane_delivery = 0
    destination_mfep_terms = 0
    invalid_nonzero_gates = 0
    edge_k_mismatches = 0

    for destination in range(SOURCES):
        destination_lane_gates: list[set[int]] = [set() for _ in range(HEAD_DIM)]
        for role in range(ROLES):
            valid = bool((valid_mask[destination] >> role) & 1)
            gate = (packed_gates[destination] >> (role * GATE_W)) & (
                (1 << GATE_W) - 1
            )
            mapped_source = source_for(destination, role)
            if valid != (mapped_source is not None):
                raise AssertionError(
                    f"topology-valid mismatch destination={destination} role={role}"
                )
            if not valid:
                invalid_nonzero_gates += int(gate != 0)
                continue
            assert mapped_source is not None
            valid_edges += 1
            k_value = (candidate_k[destination] >> (role * HEAD_DIM)) & 0xFFFFFFFF
            if source_k[mapped_source] is None:
                source_k[mapped_source] = k_value
            elif source_k[mapped_source] != k_value:
                edge_k_mismatches += 1
            if gate != 0:
                nonzero_gate_edges += 1
                source_roles[mapped_source].append((role, gate))
                lane_bitmap = int(k_value)
                relation_lane_delivery += lane_bitmap.bit_count()
                for lane in range(HEAD_DIM):
                    if (lane_bitmap >> lane) & 1:
                        destination_lane_gates[lane].add(gate)
        destination_mfep_terms += sum(
            len(gate_set) for gate_set in destination_lane_gates
        )

    active_sources = 0
    terms = 0
    updates = 0
    active_unique_gate_instances = 0
    all_source_unique_gate_instances = 0
    acc = [[0 for _ in range(OUT_DIM)] for _ in range(SOURCES)]
    multiplicity_histogram: defaultdict[int, int] = defaultdict(int)
    term_multiset_size = 0

    for source in range(SOURCES):
        k_value = source_k[source]
        if k_value is None:
            raise AssertionError(f"source {source} has no legal self/neighbor K binding")
        groups: dict[int, int] = {}
        for role, gate in source_roles[source]:
            destination = destination_for(source, role)
            if destination is None:
                raise AssertionError(
                    f"source role escapes topology source={source} role={role}"
                )
            groups[gate] = groups.get(gate, 0) | (1 << role)
        lane_count = int(k_value).bit_count()
        if lane_count and groups:
            active_sources += 1
        all_source_unique_gate_instances += len(groups)
        if lane_count:
            active_unique_gate_instances += len(groups)
        terms += lane_count * len(groups)
        updates += lane_count * sum(mask.bit_count() for mask in groups.values())
        for destination_mask in groups.values():
            multiplicity_histogram[destination_mask.bit_count()] += lane_count
        for lane in range(HEAD_DIM):
            if not ((k_value >> lane) & 1):
                continue
            for gate, destination_mask in groups.items():
                term_multiset_size += 1
                for role in range(ROLES):
                    if not ((destination_mask >> role) & 1):
                        continue
                    destination = destination_for(source, role)
                    assert destination is not None
                    for out_index in range(OUT_DIM):
                        acc[destination][out_index] += (
                            gate * weights[lane][out_index]
                        )

    if term_multiset_size != terms:
        raise AssertionError("constructed term multiset does not match term count")
    return {
        "active": active_sources,
        "terms": terms,
        "updates": updates,
        "valid_edges": valid_edges,
        "nonzero_gate_edges": nonzero_gate_edges,
        "relation_lane_delivery": relation_lane_delivery,
        "destination_mfep_terms": destination_mfep_terms,
        "active_unique_gate_instances": active_unique_gate_instances,
        "all_source_unique_gate_instances": all_source_unique_gate_instances,
        "invalid_nonzero_gates": invalid_nonzero_gates,
        "edge_k_mismatches": edge_k_mismatches,
        "multiplicity_histogram": dict(sorted(multiplicity_histogram.items())),
        "acc": acc,
    }


def parse_rtl_log(path: Path) -> dict[int, dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    if BAD_RE.search(text):
        raise ValueError(f"bad marker in {path}")
    rows: dict[int, dict[str, int]] = {}
    passes: list[int] = []
    for line in text.splitlines():
        match = GROUP_RE.match(line)
        if match:
            row = {key: int(value) for key, value in match.groupdict().items()}
            group = row.pop("group")
            if group in rows:
                raise ValueError(f"duplicate group {group} in {path}")
            rows[group] = row
        match = PASS_RE.match(line)
        if match:
            passes.append(int(match.group("cycles")))
    if sorted(rows) != list(range(100)):
        raise ValueError("RTL log is not the sealed 100-group population")
    if passes != [sum(row["cycles"] for row in rows.values())]:
        raise ValueError("RTL PASS total does not match per-group cycles")
    return rows


def validate_artifact(vector_dir: Path, manifest: dict[str, Any], name: str) -> Path:
    item = manifest.get("artifacts", {}).get(name)
    if not isinstance(item, dict):
        raise ValueError(f"manifest missing artifact {name}")
    path = vector_dir / str(item.get("file", ""))
    if not path.is_file() or sha256(path) != item.get("sha256"):
        raise ValueError(f"artifact {name} missing or SHA mismatch")
    return path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=(
            root
            / "tb_qfit/vectors/local5_joint_ep29_score_projection_realw_"
            "sample100_population_v1_20260813"
        ),
    )
    parser.add_argument(
        "--rtl-log",
        type=Path,
        default=(
            root
            / "results/local5_qsilent_rolling_composition_20260814/"
            "rolling_q1_g100_verilator_assert.log"
        ),
    )
    parser.add_argument(
        "--rtl-log-icarus",
        type=Path,
        default=(
            root
            / "results/local5_qsilent_rolling_composition_20260814/"
            "rolling_q1_g100_iverilog.log"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/local5_source_owned_gate_quotient_rtl_miter_v2_20260814",
    )
    args = parser.parse_args()

    manifest_path = args.vector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "local5_score_projection_vectors_v1"
        or manifest.get("selection", {}).get("groups") != 100
        or manifest.get("shape", {}).get("out_dim") != OUT_DIM
    ):
        raise ValueError("vector manifest is not the sealed 100-group OUT_DIM=2 cohort")
    rows_meta = manifest["selection"]["rows"]
    if len(rows_meta) != 100:
        raise ValueError("manifest selection.rows is not 100 groups")

    artifact_paths = {
        name: validate_artifact(args.vector_dir, manifest, name)
        for name in (
            "input_candidate_k",
            "input_valid",
            "expected_gates",
            "input_weights",
            "expected_active",
            "expected_terms",
            "expected_updates",
            "expected_acc",
        )
    }
    candidate_k = read_memh(artifact_paths["input_candidate_k"])
    valid_mask = read_memh(artifact_paths["input_valid"])
    packed_gates = read_memh(artifact_paths["expected_gates"])
    weight_values = read_memh(artifact_paths["input_weights"])
    expected_active = read_memh(artifact_paths["expected_active"])
    expected_terms = read_memh(artifact_paths["expected_terms"])
    expected_updates = read_memh(artifact_paths["expected_updates"])
    expected_acc = read_memh(artifact_paths["expected_acc"])
    expected_lengths = {
        "input_candidate_k": 100 * SOURCES,
        "input_valid": 100 * SOURCES,
        "expected_gates": 100 * SOURCES,
        "input_weights": 100 * HEAD_DIM * OUT_DIM,
        "expected_active": 100,
        "expected_terms": 100,
        "expected_updates": 100,
        "expected_acc": 100 * SOURCES * OUT_DIM,
    }
    actual_values = {
        "input_candidate_k": candidate_k,
        "input_valid": valid_mask,
        "expected_gates": packed_gates,
        "input_weights": weight_values,
        "expected_active": expected_active,
        "expected_terms": expected_terms,
        "expected_updates": expected_updates,
        "expected_acc": expected_acc,
    }
    for name, expected_length in expected_lengths.items():
        if len(actual_values[name]) != expected_length:
            raise ValueError(f"{name} length mismatch")

    rtl = parse_rtl_log(args.rtl_log)
    rtl_icarus = parse_rtl_log(args.rtl_log_icarus)
    if rtl != rtl_icarus:
        raise AssertionError("Icarus and Verilator ledgers differ")

    mismatch = defaultdict(int)
    totals = defaultdict(int)
    multiplicity = defaultdict(int)
    per_group: list[dict[str, Any]] = []
    for group in range(100):
        row_base = group * SOURCES
        weight_base = group * HEAD_DIM * OUT_DIM
        group_weights = [
            [
                signed(weight_values[weight_base + lane * OUT_DIM + out_index], 8)
                for out_index in range(OUT_DIM)
            ]
            for lane in range(HEAD_DIM)
        ]
        observed = analyze_group(
            candidate_k=candidate_k[row_base : row_base + SOURCES],
            valid_mask=valid_mask[row_base : row_base + SOURCES],
            packed_gates=packed_gates[row_base : row_base + SOURCES],
            weights=group_weights,
        )
        manifest_row = rows_meta[group]
        references = {
            "active": int(expected_active[group]),
            "terms": int(expected_terms[group]),
            "updates": int(expected_updates[group]),
        }
        for field in ("active", "terms", "updates"):
            mismatch[f"independent_vs_memh_{field}"] += int(
                observed[field] != references[field]
            )
            mismatch[f"independent_vs_manifest_{field}"] += int(
                observed[field] != int(manifest_row[field if field != "active" else "active_sources"])
            )
            mismatch[f"independent_vs_rtl_{field}"] += int(
                observed[field] != rtl[group][field]
            )
            totals[field] += int(observed[field])
        mismatch["invalid_nonzero_gate_groups"] += int(
            observed["invalid_nonzero_gates"] != 0
        )
        mismatch["edge_k_mismatch_groups"] += int(observed["edge_k_mismatches"] != 0)
        totals["valid_edges"] += int(observed["valid_edges"])
        totals["nonzero_gate_edges"] += int(observed["nonzero_gate_edges"])
        totals["relation_lane_delivery"] += int(observed["relation_lane_delivery"])
        totals["destination_mfep_terms"] += int(observed["destination_mfep_terms"])
        totals["active_unique_gate_instances"] += int(
            observed["active_unique_gate_instances"]
        )
        totals["all_source_unique_gate_instances"] += int(
            observed["all_source_unique_gate_instances"]
        )
        for key, value in observed["multiplicity_histogram"].items():
            multiplicity[int(key)] += int(value)

        acc_mismatch = 0
        acc_base = group * SOURCES * OUT_DIM
        for destination in range(SOURCES):
            for out_index in range(OUT_DIM):
                expected_value = signed(
                    expected_acc[
                        acc_base + destination * OUT_DIM + out_index
                    ],
                    32,
                )
                if observed["acc"][destination][out_index] != expected_value:
                    acc_mismatch += 1
        mismatch["acc32_values"] += acc_mismatch
        per_group.append(
            {
                "group": group,
                "sample": int(manifest_row["sample"]),
                "stage": int(manifest_row["stage"]),
                "active": int(observed["active"]),
                "terms": int(observed["terms"]),
                "updates": int(observed["updates"]),
                "acc32_mismatch": acc_mismatch,
            }
        )

    if any(mismatch.values()):
        raise AssertionError(f"source-owned production miter failed: {dict(mismatch)}")
    if (
        totals["active"] != 11_245
        or totals["terms"] != 74_131
        or totals["updates"] != 222_649
        or totals["relation_lane_delivery"] != totals["updates"]
    ):
        raise AssertionError("sealed production term ledger drift")
    if totals["active_unique_gate_instances"] > 5 * totals["active"]:
        raise AssertionError("active unique-gate count exceeds Local5 fanout")

    report = {
        "schema": "local5_source_owned_gate_quotient_rtl_miter_v2",
        "status": "PASS_EXISTING_EXECUTION_OBJECT_BOUND_TO_PRODUCTION_RTL",
        "evidence": "[rtl]",
        "scope": (
            "100 sample-disjoint population-stage-weighted real raw-Q/K and checkpoint-weight groups; "
            "OUT_DIM=2 score-to-Acc32 tile; not encoder"
        ),
        "execution_object": (
            "one term per source-owned active K lane and unique nonzero incoming gate; "
            "the five-bit destination mask preserves every consumer"
        ),
        "independent_reconstruction": {
            "groups": 100,
            "acc32_values": 100 * SOURCES * OUT_DIM,
            "totals": dict(totals),
            "destination_mask_multiplicity_by_term": dict(sorted(multiplicity.items())),
            "mismatch": dict(mismatch),
        },
        "same_trace_strong_baseline": {
            "raw_relation_lane_delivery": totals["relation_lane_delivery"],
            "destination_local_mfep_terms": totals["destination_mfep_terms"],
            "source_owned_terms": totals["terms"],
            "source_reduction_vs_relation_lane": (
                1.0 - totals["terms"] / totals["relation_lane_delivery"]
            ),
            "source_reduction_vs_destination_mfep": (
                1.0 - totals["terms"] / totals["destination_mfep_terms"]
            ),
            "destination_mfep_over_source_ratio": (
                totals["destination_mfep_terms"] / totals["terms"]
            ),
        },
        "four_way_miter": {
            "independent_destination_memh_reconstruction": True,
            "manifest_rows": True,
            "expected_memh": True,
            "icarus_verilator_rtl_ledger": True,
        },
        "production_rtl": {
            "cycles": sum(row["cycles"] for row in rtl.values()),
            "active": sum(row["active"] for row in rtl.values()),
            "terms": sum(row["terms"] for row in rtl.values()),
            "updates": sum(row["updates"] for row in rtl.values()),
            "acc32_mismatch": 0,
        },
        "claim_boundary": [
            "This binds an existing source-multicast execution object; it is not newly added RTL.",
            "Term reduction is not itself cycle, energy, full-encoder speedup, or ASIC PPA.",
            "OUT_DIM=2 tile excludes cross-head reduction, bias, BN, requantization, residual, decoder, and IO.",
            "The result does not modify docs/359 frozen columns.",
        ],
        "sha256": {
            "vector_manifest": sha256(manifest_path),
            "verilator_log": sha256(args.rtl_log),
            "icarus_log": sha256(args.rtl_log_icarus),
            **{
                name: sha256(path) for name, path in artifact_paths.items()
            },
            "source_multicast_rtl": sha256(
                root / "rtl_qfit/qfit_source_multicast_term_builder.sv"
            ),
            "score_projection_wrapper_rtl": sha256(
                root / "rtl_qfit/qfit_local5_score_active_projection_tile.sv"
            ),
            "active_projection_tile_rtl": sha256(
                root / "rtl_qfit/qfit_local5_active_projection_tile.sv"
            ),
            "rolling_frontier_rtl": sha256(
                root / "rtl_qfit/sidecar/qfit_dual_color_relation_frontier_sync.sv"
            ),
            "relation_transpose_rtl": sha256(
                root / "rtl_qfit/qfit_relation_transpose_leaf.sv"
            ),
            "retirement_scheduler_rtl": sha256(
                root / "rtl_qfit/qfit_retirement_scheduler.sv"
            ),
            "tcfm5_rtl": sha256(
                root / "rtl_qfit/qfit_tcfm5_projection_top.sv"
            ),
            "tcfm5_acc_bank_rtl": sha256(
                root / "rtl_qfit/qfit_tcfm5_acc_bank.sv"
            ),
            "term_conservation_sva": sha256(
                root
                / "verif_qfit/qfit_local5_source_owned_term_conservation_assertions.sv"
            ),
        },
        "per_group": per_group,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = f"""# Local5 源所有 gate 商生产 RTL 四方 miter

- 裁决：`{report['status']}`，证据 `[rtl]`。
- 边界：{report['scope']}。
- 独立重构：从 destination-major raw K/valid/gate memh 反演 source descriptor，不调用向量生成器；按 `source x active lane x unique nonzero gate` 构造 term，并以 5-bit destination mask 保留全部 consumer。
- 四方一致：独立重构、manifest row、expected memh、Icarus/Verilator RTL ledger 在 100/100 组逐组一致。
- 生产账本：cycle `{report['production_rtl']['cycles']}`，active source `{report['production_rtl']['active']}`，term `{report['production_rtl']['terms']}`，destination update `{report['production_rtl']['updates']}`。
- 同 trace 强基线：raw relation-lane `{report['same_trace_strong_baseline']['raw_relation_lane_delivery']}`，destination-local MFEP `{report['same_trace_strong_baseline']['destination_local_mfep_terms']}`，source-owned `{report['same_trace_strong_baseline']['source_owned_terms']}`；source-owned 相对 MFEP 再减少 `{report['same_trace_strong_baseline']['source_reduction_vs_destination_mfep']:.2%}` product term。
- 数值：真实 checkpoint INT8 权重下 `{100 * SOURCES * OUT_DIM}` 个 Acc32 值 mismatch `0`。
- 守恒：term 只折叠同一 source/lane/gate 的重复 product；destination mask 保留每个目标，因此 update 数不被伪删。

## 主张边界

这是对现有 `qfit_source_multicast_term_builder` 执行对象的生产绑定，不是新增 RTL，也不是新的独立贡献名。它使 Local5 的统一叙事有了生产级证据：固定拓扑在线转置产生 source descriptor，source-owned gate 商减少 product term，TCFM5 对 destination mask 精确散播。该包仍是 `OUT_DIM=2` tile，不是 encoder；term 削减不能直接写成周期、能量或 ASIC PPA，且不修改 `docs/359`。
"""
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": report["status"], "totals": dict(totals)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
