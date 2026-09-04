#!/usr/bin/env python3
"""Screen exact quantization-phase residual scores for Local5 Shiftmax5.

The deployed score is RNE(raw16 / 16).  If A is the self-candidate raw score,
then A = 32*m + phase.  Adding 32*m changes every quantized candidate by the
same even integer, including ties-to-even cases, and therefore cancels before
Shiftmax.  Only phase=A mod 32 and the four raw score deltas are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TOKENS = 450
HEIGHT = 15
WIDTH = 15
PLANES = 2
RAW_MAX = 32 * 64
LUT = np.asarray(
    [256, 245, 234, 224, 215, 205, 196, 188,
     181, 173, 165, 158, 152, 145, 139, 133],
    dtype=np.int64,
)
CANDIDATE_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
INCOMING_ROLES = (0, 1, 2, 3, 4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rne_div16(value: int) -> int:
    quotient, remainder = divmod(int(value), 16)
    if remainder > 8 or (remainder == 8 and (quotient & 1)):
        quotient += 1
    return quotient


def raw16(q_bits: int, k_bits: int) -> int:
    mask = (1 << 32) - 1
    same_one = (q_bits & k_bits).bit_count()
    same_zero = ((~q_bits & ~k_bits) & mask).bit_count()
    return 64 * same_one + same_zero


def shiftmax5_gate(scores: list[int], valid: list[bool]) -> list[int]:
    row_max = max(score for score, keep in zip(scores, valid, strict=True) if keep)
    exp = []
    for score, keep in zip(scores, valid, strict=True):
        if not keep:
            exp.append(0)
            continue
        distance = row_max - score
        integer_shift = min(8, distance >> 7)
        frac_index = min(15, ((distance & 127) + 7) >> 3)
        exp.append(int(LUT[frac_index]) >> integer_shift)
    row_sum = sum(exp)
    den_shift = max(0, (row_sum - 1).bit_length())
    result = []
    for value, keep in zip(exp, valid, strict=True):
        if not keep:
            result.append(0)
            continue
        scaled = value << 7
        quotient = scaled >> den_shift
        remainder = scaled - (quotient << den_shift)
        if den_shift:
            half = 1 << (den_shift - 1)
            if remainder > half or (remainder == half and (quotient & 1)):
                quotient += 1
        result.append(min(256, quotient))
    return result


def exhaustive_phase_proof() -> dict:
    mismatches = 0
    checked = 0
    translated_min = 1 << 30
    translated_max = -(1 << 30)
    for anchor in range(RAW_MAX + 1):
        phase = anchor & 31
        common = 2 * (anchor >> 5)
        anchor_quant = rne_div16(anchor)
        candidates = np.arange(RAW_MAX + 1, dtype=np.int64)
        baseline = np.rint(candidates / 16.0).astype(np.int64)
        translated = np.rint((phase + candidates - anchor) / 16.0).astype(np.int64)
        mismatch = baseline - anchor_quant != translated - rne_div16(phase)
        mismatches += int(np.count_nonzero(mismatch))
        checked += int(candidates.size)
        translated_min = min(translated_min, int(translated.min()))
        translated_max = max(translated_max, int(translated.max()))
        if rne_div16(anchor) - rne_div16(phase) != common:
            mismatches += 1
    return {
        "pairs_checked": checked,
        "mismatches": mismatches,
        "translated_score_min": translated_min,
        "translated_score_max": translated_max,
        "signed_score_bits_required": 9,
    }


def candidate_source(destination: int, role: int) -> tuple[int, bool]:
    plane = destination // (HEIGHT * WIDTH)
    spatial = destination % (HEIGHT * WIDTH)
    y, x = divmod(spatial, WIDTH)
    dy, dx = CANDIDATE_OFFSETS[role]
    sy, sx = y + dy, x + dx
    valid = 0 <= sy < HEIGHT and 0 <= sx < WIDTH
    source = plane * HEIGHT * WIDTH + sy * WIDTH + sx if valid else 0
    return source, valid


def profile_population(vector_manifest: Path, ordered_payload: Path) -> dict:
    manifest = json.loads(vector_manifest.read_text())
    rows = manifest["selection"]["rows"]
    if len(rows) != 100:
        raise ValueError("expected the locked Local5 100-group population")

    score_translation_mismatches = 0
    gate_mismatches = 0
    trace_gate_mismatches = 0
    valid_candidates = 0
    qsilent_rows = 0
    identk_rows = 0
    normal_rows = 0
    translated_min = 1 << 30
    translated_max = -(1 << 30)

    with np.load(ordered_payload, allow_pickle=False) as payload:
        offsets = np.asarray(payload["descriptor_group_offsets"])
        source_ids = np.asarray(payload["descriptor_source_id"])
        q_values = np.asarray(payload["descriptor_q_bitmap"])
        k_values = np.asarray(payload["descriptor_k_bitmap"])
        incoming = np.asarray(payload["descriptor_incoming_gates"])
        incoming_valid = np.asarray(payload["descriptor_valid_mask"])

        for metadata in rows:
            group = int(metadata["input_group_index"])
            start, stop = int(offsets[group]), int(offsets[group + 1])
            if stop - start != TOKENS:
                raise ValueError(f"group {group} is not T450")
            q_by_source = np.zeros(TOKENS, dtype=np.uint64)
            k_by_source = np.zeros(TOKENS, dtype=np.uint64)
            gate_by_source = np.zeros((TOKENS, 5), dtype=np.uint16)
            valid_by_source = np.zeros(TOKENS, dtype=np.uint8)
            for index in range(start, stop):
                source = int(source_ids[index])
                q_by_source[source] = q_values[index]
                k_by_source[source] = k_values[index]
                gate_by_source[source] = incoming[index]
                valid_by_source[source] = incoming_valid[index]

            for destination in range(TOKENS):
                q_bits = int(q_by_source[destination])
                sources = []
                valid = []
                raw = []
                trace_gate = []
                valid_k = []
                for role in range(5):
                    source, keep = candidate_source(destination, role)
                    sources.append(source)
                    valid.append(keep)
                    k_bits = int(k_by_source[source]) if keep else 0
                    valid_k.append(k_bits)
                    raw.append(raw16(q_bits, k_bits) if keep else 0)
                    trace_gate.append(
                        int(gate_by_source[source, INCOMING_ROLES[role]]) if keep else 0
                    )
                    if keep and not ((int(valid_by_source[source]) >> INCOMING_ROLES[role]) & 1):
                        raise ValueError(
                            f"missing incoming relation group={group} destination={destination} role={role}"
                        )

                if q_bits == 0:
                    qsilent_rows += 1
                elif len(set(valid_k)) == 1:
                    identk_rows += 1
                else:
                    normal_rows += 1

                baseline_scores = [rne_div16(value) if keep else -256 for value, keep in zip(raw, valid, strict=True)]
                anchor = raw[0]
                phase = anchor & 31
                translated_scores = [
                    rne_div16(phase + value - anchor) if keep else -256
                    for value, keep in zip(raw, valid, strict=True)
                ]
                common = baseline_scores[0] - translated_scores[0]
                for base, translated, keep in zip(
                    baseline_scores, translated_scores, valid, strict=True
                ):
                    if keep:
                        valid_candidates += 1
                        score_translation_mismatches += int(base - translated != common)
                        translated_min = min(translated_min, translated)
                        translated_max = max(translated_max, translated)

                baseline_gate = shiftmax5_gate(baseline_scores, valid)
                translated_gate = shiftmax5_gate(translated_scores, valid)
                gate_mismatches += sum(
                    left != right
                    for left, right in zip(baseline_gate, translated_gate, strict=True)
                )
                trace_gate_mismatches += sum(
                    left != right
                    for left, right in zip(baseline_gate, trace_gate, strict=True)
                )

    return {
        "groups": len(rows),
        "rows": len(rows) * TOKENS,
        "valid_candidates": valid_candidates,
        "score_translation_mismatches": score_translation_mismatches,
        "gate_mismatches": gate_mismatches,
        "trace_gate_mismatches": trace_gate_mismatches,
        "translated_score_min": translated_min,
        "translated_score_max": translated_max,
        "existing_path_split": {
            "qsilent_rows": qsilent_rows,
            "identk_rows": identk_rows,
            "normal_rows": normal_rows,
            "normal_fraction": normal_rows / (len(rows) * TOKENS),
        },
        "provenance": {
            "vector_manifest": str(vector_manifest.resolve()),
            "vector_manifest_sha256": sha256(vector_manifest),
            "ordered_payload": str(ordered_payload.resolve()),
            "ordered_payload_sha256": sha256(ordered_payload),
        },
    }


def build_report(vector_manifest: Path, ordered_payload: Path) -> dict:
    exhaustive = exhaustive_phase_proof()
    population = profile_population(vector_manifest, ordered_payload)
    baseline_bits = 5 * 13 + 5 * 16
    candidate_bits = 5 + 4 * 13 + 5 * 9
    structural = {
        "baseline_state_bits": baseline_bits,
        "candidate_state_bits": candidate_bits,
        "state_reduction": baseline_bits / candidate_bits,
        "state_bit_reduction": 1.0 - candidate_bits / baseline_bits,
        "shiftmax_score_width_baseline": 16,
        "shiftmax_score_width_candidate": 9,
        "cycle_reduction_proven": 0.0,
    }
    gates = {
        "universal_integer_equivalence": exhaustive["mismatches"] == 0,
        "population_score_equivalence": population["score_translation_mismatches"] == 0,
        "population_gate_equivalence": population["gate_mismatches"] == 0,
        "existing_trace_gate_equivalence": population["trace_gate_mismatches"] == 0,
        "state_bit_reduction_ge_20pct": structural["state_bit_reduction"] >= 0.20,
        "incremental_normal_rows_ge_15pct": population["existing_path_split"]["normal_fraction"] >= 0.15,
        "rtl_and_same_sdc_proxy_complete": False,
    }
    status = (
        "CONDITIONAL_PROFILE_REQUIRES_LEAF_PPA"
        if all(
            gates[key]
            for key in (
                "universal_integer_equivalence",
                "population_score_equivalence",
                "population_gate_equivalence",
                "existing_trace_gate_equivalence",
                "state_bit_reduction_ge_20pct",
            )
        )
        else "NO_GO"
    )
    return {
        "schema": "local5_quantization_phase_residual_v1",
        "status": status,
        "evidence": "[derivation]+[exhaustive-integer]+[prof]",
        "exact_contract": {
            "identity": "RNE((A+d)/16) = 2*floor(A/32) + RNE(((A mod 32)+d)/16)",
            "normalization": "Shiftmax removes the shared even-integer translation before LUT evaluation",
            "metadata": "5-bit anchor phase plus four signed raw deltas; no saturation, clipping, or changed gate order",
        },
        "exhaustive": exhaustive,
        "population": population,
        "structural_model": structural,
        "gates": gates,
        "claim_boundary": [
            "No production RTL, cycle speedup, DC, STA, SAIF, SRAM macro, or PTPX result is claimed.",
            "The 29.7% state-bit reduction is leaf-local and excludes control, K/Q registers, compaction, and projection.",
            "Existing Query-Silent/ident-K paths already bypass most rows; only the normal-row fraction is incremental dynamic work.",
            "Softmax/Shiftmax translation invariance is known; novelty can only come from preserving ties-to-even with the 5-bit quantization phase and proving useful hardware Pareto.",
            "The candidate remains outside docs/359 and outside the DATE contribution list until a same-SDC leaf comparison passes.",
        ],
    }


def render(report: dict) -> str:
    pop = report["population"]
    split = pop["existing_path_split"]
    structural = report["structural_model"]
    return "\n".join(
        [
            "# Local5 quantization-phase residual Shiftmax screen",
            "",
            f"Status: **{report['status']}** (`{report['evidence']}`)",
            "",
            "## Exact result",
            "",
            f"- Exhaustive raw-score pairs: {report['exhaustive']['pairs_checked']}, mismatches {report['exhaustive']['mismatches']}.",
            f"- Population: {pop['rows']} rows, score-translation mismatch {pop['score_translation_mismatches']}, transformed-gate mismatch {pop['gate_mismatches']}, existing-trace gate mismatch {pop['trace_gate_mismatches']}.",
            f"- Translated score range is [{pop['translated_score_min']}, {pop['translated_score_max']}]; signed 9-bit is sufficient.",
            "",
            "## Structural screen",
            "",
            f"- Modeled score state: {structural['baseline_state_bits']} -> {structural['candidate_state_bits']} bit = {structural['state_reduction']:.3f}x ({100*structural['state_bit_reduction']:.2f}% reduction).",
            f"- Shiftmax score datapath width: 16 -> 9 bit.",
            f"- Existing path split: Q-silent {split['qsilent_rows']}, ident-K {split['identk_rows']}, normal {split['normal_rows']} ({100*split['normal_fraction']:.2f}%).",
            "- No cycle reduction is established; the next gate is a current-leaf versus phase-residual-leaf mapping under the same SDC.",
            "",
            "## Boundaries",
            "",
            *[f"- {item}" for item in report["claim_boundary"]],
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vector-manifest",
        type=Path,
        default=Path("tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_out32_v1_20260814/manifest.json"),
    )
    parser.add_argument(
        "--ordered-payload",
        type=Path,
        default=Path("results/local5_fullres_bb1e4_joint_heads_profile100_20260809/ordered_term_items.npz"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/local5_quantization_phase_residual_20260814"),
    )
    args = parser.parse_args()
    report = build_report(args.vector_manifest, args.ordered_payload)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out_dir / "report.md").write_text(render(report))
    print(json.dumps({"status": report["status"], "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
