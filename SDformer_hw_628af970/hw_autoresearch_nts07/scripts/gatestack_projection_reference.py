#!/usr/bin/env python3
"""GateStack完整多head、多output-tile投影的整数金参考。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _validate_inputs(
    k_event: np.ndarray,
    gate_code: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    output_tile: int,
) -> tuple[int, int, int, int]:
    if k_event.ndim != 3:
        raise ValueError("k_event必须为[heads,tokens,head_dim]")
    heads, tokens, head_dim = map(int, k_event.shape)
    if gate_code.shape != (heads, tokens):
        raise ValueError("gate_code必须为[heads,tokens]")
    if weight.ndim != 3 or weight.shape[:2] != (heads, head_dim):
        raise ValueError("weight必须为[heads,head_dim,outputs]")
    outputs = int(weight.shape[2])
    if bias.shape != (outputs,):
        raise ValueError("bias必须为[outputs]")
    if output_tile <= 0 or outputs % output_tile != 0:
        raise ValueError("outputs必须能被output_tile整除")
    if np.any(gate_code < 0) or np.any(gate_code > 256):
        raise ValueError("gate_code必须位于0..256")
    return heads, tokens, head_dim, outputs


def requantize_signed(
    values: np.ndarray,
    *,
    right_shift: int = 0,
    output_bits: int = 32,
) -> np.ndarray:
    """确定性有符号重标定：绝对值RHAZ右移后饱和。"""

    if right_shift < 0:
        raise ValueError("right_shift不能为负")
    if output_bits < 2 or output_bits > 63:
        raise ValueError("output_bits必须位于2..63")
    work = values.astype(np.int64, copy=False)
    if right_shift:
        magnitude = np.abs(work)
        magnitude = (magnitude + (1 << (right_shift - 1))) >> right_shift
        work = np.where(work < 0, -magnitude, magnitude)
    lower = -(1 << (output_bits - 1))
    upper = (1 << (output_bits - 1)) - 1
    return np.clip(work, lower, upper).astype(np.int64)


def dense_full_projection(
    k_event: np.ndarray,
    gate_code: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    output_tile: int,
    right_shift: int = 0,
    output_bits: int = 32,
) -> np.ndarray:
    """按全局input channel展开的直接投影。"""

    heads, tokens, head_dim, _ = _validate_inputs(
        k_event, gate_code, weight, bias, output_tile
    )
    gated_k = (
        k_event.astype(np.int64) * gate_code.astype(np.int64)[:, :, None]
    ).transpose(1, 0, 2).reshape(tokens, heads * head_dim)
    flat_weight = weight.astype(np.int64).reshape(heads * head_dim, -1)
    accum = gated_k @ flat_weight
    accum += bias.astype(np.int64)[None, :]
    return requantize_signed(
        accum, right_shift=right_shift, output_bits=output_bits
    )


def build_head_representation(
    k_head: np.ndarray,
    gate_head: np.ndarray,
    *,
    class_slots: int,
) -> dict[str, Any]:
    """构建单head目录；类数溢出时整head切换为DIRECT格式。"""

    if k_head.ndim != 2 or gate_head.shape != (k_head.shape[0],):
        raise ValueError("单head输入形状不匹配")
    if class_slots <= 0:
        raise ValueError("class_slots必须为正")
    active_tokens = np.flatnonzero(k_head.any(axis=1))
    active_gates = np.unique(gate_head[active_tokens])
    if active_gates.size > class_slots:
        return {
            "mode": "DIRECT",
            "k_event": k_head.astype(bool, copy=True),
            "gate_code": gate_head.astype(np.int64, copy=True),
            "active_classes": int(active_gates.size),
            "occupied_entries": int(k_head.sum()),
        }

    entries = []
    for gate in active_gates:
        token_ids = np.flatnonzero((gate_head == gate) & k_head.any(axis=1))
        for lane in np.flatnonzero(k_head[token_ids].any(axis=0)):
            destinations = token_ids[np.flatnonzero(k_head[token_ids, lane])]
            entries.append(
                {
                    "gate_code": int(gate),
                    "lane": int(lane),
                    "destinations": destinations.astype(np.int64),
                }
            )
    return {
        "mode": "DIRECTORY",
        "entries": entries,
        "active_classes": int(active_gates.size),
        "occupied_entries": len(entries),
    }


def gatestack_full_projection(
    k_event: np.ndarray,
    gate_code: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    class_slots: int,
    output_tile: int,
    right_shift: int = 0,
    output_bits: int = 32,
) -> tuple[np.ndarray, dict[str, int]]:
    """目录先按head构建，再跨全部output tile重放并持久累加。"""

    heads, tokens, _, outputs = _validate_inputs(
        k_event, gate_code, weight, bias, output_tile
    )
    representations = [
        build_head_representation(
            k_event[head], gate_code[head], class_slots=class_slots
        )
        for head in range(heads)
    ]
    accum = np.zeros((tokens, outputs), dtype=np.int64)
    directory_terms = 0
    direct_terms = 0
    directory_heads = 0
    direct_heads = 0

    for output_base in range(0, outputs, output_tile):
        output_slice = slice(output_base, output_base + output_tile)
        for head, representation in enumerate(representations):
            if representation["mode"] == "DIRECTORY":
                directory_heads += 1
                for entry in representation["entries"]:
                    product = (
                        np.int64(entry["gate_code"])
                        * weight[head, entry["lane"], output_slice].astype(np.int64)
                    )
                    accum[entry["destinations"], output_slice] += product
                    directory_terms += 1
            else:
                direct_heads += 1
                k_head = representation["k_event"]
                gate_head = representation["gate_code"]
                for token in np.flatnonzero(k_head.any(axis=1)):
                    for lane in np.flatnonzero(k_head[token]):
                        accum[token, output_slice] += (
                            np.int64(gate_head[token])
                            * weight[head, lane, output_slice].astype(np.int64)
                        )
                        direct_terms += 1

    accum += bias.astype(np.int64)[None, :]
    int32_min = -(1 << 31)
    int32_max = (1 << 31) - 1
    if int(accum.min(initial=0)) < int32_min or int(accum.max(initial=0)) > int32_max:
        raise OverflowError("投影累加结果超出signed int32")
    output = requantize_signed(
        accum, right_shift=right_shift, output_bits=output_bits
    )
    output_tiles = outputs // output_tile
    return output, {
        "heads": heads,
        "tokens": tokens,
        "outputs": outputs,
        "output_tiles": output_tiles,
        "directory_heads": directory_heads // output_tiles,
        "direct_heads": direct_heads // output_tiles,
        "directory_terms_all_tiles": directory_terms,
        "direct_terms_all_tiles": direct_terms,
        "baseline_active_terms_all_tiles": int(k_event.sum()) * output_tiles,
        "bias_commits": tokens * outputs,
    }


def run_trials(seed: int = 20260715, trials: int = 80) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mismatches = 0
    compared = 0
    fallback_trials = 0
    for trial in range(trials):
        heads = (3, 6, 12, 24)[trial % 4]
        tokens = 19
        head_dim = 8
        outputs = heads * head_dim
        output_tile = 8
        k_event = rng.random((heads, tokens, head_dim)) < rng.uniform(0.02, 0.30)
        gate_code = rng.integers(0, 17, size=(heads, tokens), dtype=np.int16)
        if trial % 5:
            gate_code %= 4
        weight = rng.integers(-128, 128, size=(heads, head_dim, outputs), dtype=np.int16)
        bias = rng.integers(-4096, 4096, size=outputs, dtype=np.int32)
        expected = dense_full_projection(
            k_event,
            gate_code,
            weight,
            bias,
            output_tile=output_tile,
            right_shift=3,
            output_bits=16,
        )
        actual, counters = gatestack_full_projection(
            k_event,
            gate_code,
            weight,
            bias,
            class_slots=4,
            output_tile=output_tile,
            right_shift=3,
            output_bits=16,
        )
        mismatches += int(np.count_nonzero(expected != actual))
        compared += int(expected.size)
        fallback_trials += int(counters["direct_heads"] > 0)
    return {
        "seed": seed,
        "trials": trials,
        "compared_outputs": compared,
        "mismatches": mismatches,
        "trials_with_direct_fallback": fallback_trials,
        "requant_contract": "RHAZ右移后有符号饱和；仅作为RTL金参考，网络量化合同未冻结",
    }


def main() -> int:
    result = run_trials()
    output = ROOT / "results/gatestack_projection_reference_20260715.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    print("PASS" if result["mismatches"] == 0 else "FAIL")
    return int(result["mismatches"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
