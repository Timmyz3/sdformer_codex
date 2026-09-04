#!/usr/bin/env python3
"""ACQN-163 与逐 candidate Shiftmax 的独立整数等价参考。

本文件只验证 normalization/member replay 数值合同，不模拟 RTL 周期。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "acqn163_reference_20260730"
EXP2_LUT = (
    256,
    245,
    234,
    224,
    215,
    205,
    196,
    188,
    181,
    173,
    165,
    158,
    152,
    145,
    139,
    133,
)
MAX_SCORE_CLASS = 162


@dataclass(frozen=True)
class Candidate:
    member_id: int
    score_q7: int
    k_nonzero: bool
    valid: bool = True
    multiplicity: int = 1
    context: int = 0
    destination: int = 0
    k_bits: int = 1
    last: bool = False

    def validate(self) -> None:
        if not 0 <= self.score_q7 <= MAX_SCORE_CLASS:
            raise ValueError("score class 超出 0..162")
        if self.multiplicity <= 0:
            raise ValueError("multiplicity 必须为正数")
        if not 0 <= self.member_id < 450:
            raise ValueError("member_id 超出 T450 合法域")
        if not 0 <= self.context < 2:
            raise ValueError("context 超出双 context 合法域")
        if not 0 <= self.destination < 450:
            raise ValueError("destination 超出 T450 合法域")
        if not 0 <= self.k_bits < (1 << 32):
            raise ValueError("K payload 超出 32 bit")
        if self.k_nonzero != (self.k_bits != 0):
            raise ValueError("k_nonzero 必须与 K payload 一致")
        if self.k_nonzero and self.multiplicity != 1:
            raise ValueError("active member 必须逐 identity 保存，multiplicity 只能为 1")


def ceil_log2(value: int) -> int:
    return max(0, (max(1, value) - 1).bit_length())


def rne_pow2(value: int, shift: int) -> int:
    if shift == 0:
        return value
    quotient = value >> shift
    remainder = value - (quotient << shift)
    half = 1 << (shift - 1)
    return quotient + int(
        remainder > half
        or (remainder == half and bool(quotient & 1))
    )


def exp2_q8(delta_q7: int) -> int:
    if delta_q7 >= 0:
        return 256
    abs_delta = -delta_q7
    integer_shift = min(8, abs_delta >> 7)
    fraction_index = min(((abs_delta & 127) + 7) >> 3, 15)
    return EXP2_LUT[fraction_index] >> integer_shift


def acqn_exp2_q8(delta_q7: int) -> int:
    """与 expanded reference 独立书写的 class-path LUT 模型。"""

    if delta_q7 >= 0:
        return 0x100
    magnitude = -delta_q7
    coarse = magnitude // 128
    fractional = magnitude % 128
    lut_address = min(15, (fractional + 7) // 8)
    return EXP2_LUT[lut_address] >> min(8, coarse)


def gate_q17(
    exp_q8: int,
    row_sum_q8: int,
    n_tokens: int,
    preserve_mean: bool,
) -> int:
    if row_sum_q8 == 0:
        return 0
    token_scale = n_tokens if preserve_mean else 1
    scaled = exp_q8 * token_scale * 128
    rounded = rne_pow2(scaled, ceil_log2(row_sum_q8))
    return min(rounded, 256)


def acqn_gate_q17(
    exp_value: int,
    denominator: int,
    token_count: int,
    preserve_mean: bool,
) -> int:
    if denominator == 0:
        return 0
    scale = token_count if preserve_mean else 1
    numerator = exp_value * scale << 7
    shift = (denominator - 1).bit_length()
    if shift == 0:
        rounded = numerator
    else:
        base = numerator >> shift
        residue = numerator & ((1 << shift) - 1)
        midpoint = 1 << (shift - 1)
        rounded = base + int(
            residue > midpoint
            or (residue == midpoint and bool(base & 1))
        )
    return min(rounded, 0x100)


def member_record(
    candidate: Candidate,
    gate: int,
    output_last: bool,
) -> tuple[int, ...]:
    return (
        candidate.context,
        candidate.member_id,
        candidate.destination,
        candidate.k_bits,
        gate,
        int(output_last),
    )


def active_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.valid and candidate.k_nonzero
    ]


def validate_row_context(candidates: list[Candidate]) -> None:
    contexts = {candidate.context for candidate in candidates}
    if len(contexts) > 1:
        raise ValueError("同一 normalization row 不得混合 context")


def expanded_reference(
    candidates: list[Candidate],
    preserve_mean: bool,
) -> dict:
    for candidate in candidates:
        candidate.validate()
    validate_row_context(candidates)
    valid = [candidate for candidate in candidates if candidate.valid]
    if not valid:
        return {
            "row_sum_q8": 0,
            "n_tokens": 0,
            "active_member_sequence": [],
            "exp_evals": 0,
        }
    n_tokens = sum(candidate.multiplicity for candidate in valid)
    row_max = max(candidate.score_q7 for candidate in valid)
    row_sum = sum(
        candidate.multiplicity
        * exp2_q8(candidate.score_q7 - row_max)
        for candidate in valid
    )
    active = []
    active_members = active_candidates(candidates)
    for index, candidate in enumerate(active_members):
        exp_value = exp2_q8(candidate.score_q7 - row_max)
        gate = gate_q17(
            exp_value,
            row_sum,
            n_tokens,
            preserve_mean,
        )
        active.append(
            member_record(
                candidate,
                gate,
                index == len(active_members) - 1,
            )
        )
    return {
        "row_sum_q8": row_sum,
        "n_tokens": n_tokens,
        "active_member_sequence": active,
        "exp_evals": n_tokens,
    }


def acqn_reference(
    candidates: list[Candidate],
    preserve_mean: bool,
) -> dict:
    for candidate in candidates:
        candidate.validate()
    validate_row_context(candidates)
    valid = [candidate for candidate in candidates if candidate.valid]
    if not valid:
        return {
            "row_sum_q8": 0,
            "n_tokens": 0,
            "active_member_sequence": [],
            "class_count": {},
            "class_gate": {},
            "exp_evals": 0,
        }

    class_count: dict[int, int] = {}
    for candidate in valid:
        class_count[candidate.score_q7] = (
            class_count.get(candidate.score_q7, 0)
            + candidate.multiplicity
        )
    n_tokens = sum(class_count.values())
    row_max = max(class_count)
    class_exp = {
        score: acqn_exp2_q8(score - row_max)
        for score in class_count
    }
    row_sum = sum(
        class_count[score] * class_exp[score]
        for score in class_count
    )
    class_gate = {
        score: acqn_gate_q17(
            class_exp[score],
            row_sum,
            n_tokens,
            preserve_mean,
        )
        for score in class_count
    }
    active = []
    active_members = active_candidates(candidates)
    for index, candidate in enumerate(active_members):
        active.append(
            member_record(
                candidate,
                class_gate[candidate.score_q7],
                index == len(active_members) - 1,
            )
        )
    return {
        "row_sum_q8": row_sum,
        "n_tokens": n_tokens,
        "active_member_sequence": active,
        "class_count": class_count,
        "class_gate": class_gate,
        "exp_evals": len(class_count),
    }


def verify_row(
    candidates: list[Candidate],
    preserve_mean: bool,
) -> tuple[int, int]:
    expanded = expanded_reference(candidates, preserve_mean)
    acqn = acqn_reference(candidates, preserve_mean)
    if expanded["row_sum_q8"] != acqn["row_sum_q8"]:
        raise AssertionError("ACQN denominator 与 expanded 不一致")
    if expanded["n_tokens"] != acqn["n_tokens"]:
        raise AssertionError("ACQN multiplicity 总数不一致")
    if expanded["active_member_sequence"] != acqn["active_member_sequence"]:
        raise AssertionError("ACQN 原序 member replay 记录不一致")
    return int(expanded["exp_evals"]), int(acqn["exp_evals"])


def directed_rows() -> list[list[Candidate]]:
    return [
        [],
        [Candidate(0, 0, False, False, 1, k_bits=0, last=True)],
        [Candidate(0, 7, True, True, 1, k_bits=1, last=True)],
        [
            Candidate(0, 7, False, True, 1, k_bits=0),
            Candidate(1, 8, True, True, 1, k_bits=3, last=True),
        ],
        [
            Candidate(index, index, bool(index & 1), k_bits=int(bool(index & 1)))
            for index in range(5)
        ],
        [
            Candidate(
                score,
                score,
                bool(score & 1),
                k_bits=int(bool(score & 1)),
                last=score == 162,
            )
            for score in range(163)
        ],
        [
            Candidate(
                index,
                index % 163,
                bool(index & 1),
                destination=index,
                k_bits=int(bool(index & 1)),
                last=index == 161,
            )
            for index in range(162)
        ],
        [
            Candidate(0, 32, False, True, 449, k_bits=0),
            Candidate(1, 32, True, True, 1, k_bits=1, last=True),
        ],
        [
            Candidate(0, 0, False, True, 449, k_bits=0),
            Candidate(1, 162, True, True, 1, k_bits=1, last=True),
        ],
        [
            Candidate(0, 37, True, True, 1, destination=7, k_bits=3),
            Candidate(1, 37, True, True, 1, destination=9, k_bits=5),
            Candidate(2, 36, False, True, 5, k_bits=0),
            Candidate(3, 20, True, False, 1, k_bits=7, last=True),
        ],
    ]


def random_row(rng: random.Random, max_members: int) -> list[Candidate]:
    member_count = rng.randint(0, max_members)
    remaining = 450
    row = []
    context = rng.randrange(2)
    for member_id in range(member_count):
        k_bits = (
            0
            if rng.random() < 0.40
            else max(1, rng.getrandbits(32))
        )
        k_nonzero = bool(k_bits)
        multiplicity = (
            1
            if k_nonzero
            else rng.randint(1, min(16, remaining))
        )
        remaining -= multiplicity
        row.append(
            Candidate(
                member_id=member_id,
                score_q7=rng.randint(0, MAX_SCORE_CLASS),
                k_nonzero=k_nonzero,
                valid=rng.random() >= 0.15,
                multiplicity=multiplicity,
                context=context,
                destination=rng.randrange(450),
                k_bits=k_bits,
                last=member_id == member_count - 1,
            )
        )
        if remaining == 0:
            break
    return row


def run_reference(rows: int, seed: int) -> dict:
    rng = random.Random(seed)
    vectors = directed_rows()
    vectors.extend(random_row(rng, 32) for _ in range(rows))
    expanded_evals = 0
    acqn_evals = 0
    checks = 0
    for candidates in vectors:
        for preserve_mean in (False, True):
            expanded, acqn = verify_row(candidates, preserve_mean)
            expanded_evals += expanded
            acqn_evals += acqn
            checks += 1
    return {
        "schema": "acqn163_ordered_integer_equivalence_v2",
        "seed": seed,
        "random_rows": rows,
        "directed_rows": len(directed_rows()),
        "mode_row_checks": checks,
        "mismatches": 0,
        "expanded_exp_evals": expanded_evals,
        "acqn_exp_evals": acqn_evals,
        "synthetic_exp_eval_reduction": (
            1.0 - acqn_evals / expanded_evals
            if expanded_evals
            else 0.0
        ),
        "coverage": [
            "score class 0..162",
            "preserve_mean on/off",
            "multiplicity up to 450",
            "explicit T=1/2/5/162/163/450",
            "zero-K and active-K",
            "invalid member",
            "empty row",
            "equal-score active member",
            "wide score span",
            "single context per normalization row",
            "K-zero flag equals 32-bit payload zero",
            "active identity multiplicity fixed to one",
            "output last regenerated after zero-K filtering",
            "ordered {context,member,destination,K,gate,last} record",
        ],
        "evidence": (
            "[integer-golden]，只证明原序整数记录；"
            "不是逐拍 RTL、ready/valid 或 PPA"
        ),
    }


def render_markdown(result: dict) -> str:
    return "\n".join(
        [
            "# ACQN-163 整数等价金参考",
            "",
            "## 结论",
            "",
            f"- directed rows：{result['directed_rows']}",
            f"- random rows：{result['random_rows']}",
            f"- preserve-mean on/off row checks：{result['mode_row_checks']}",
            f"- mismatch：{result['mismatches']}",
            (
                "- synthetic exp-eval reduction："
                f"{100.0 * result['synthetic_exp_eval_reduction']:.2f}%"
            ),
            "",
            "逐 candidate 整数参考与 ACQN class-count/member-replay 在",
            "`row_sum_q8`、`n_tokens` 和原序 active member 记录上完全一致。",
            "",
            "该结果只证明原序 replay 的整数算法合同，不包含 ready/valid、",
            "背压、清空、RAW、双 context 或逐拍 RTL。",
            "",
            "## 覆盖",
            "",
            *[f"- {item}" for item in result["coverage"]],
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0xAC0A163)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = run_reference(args.rows, args.seed)
    (args.out / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out / "report.md").write_text(render_markdown(result) + "\n")
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
