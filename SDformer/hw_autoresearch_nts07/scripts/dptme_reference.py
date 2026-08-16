#!/usr/bin/env python3
"""DP-TME的整数golden、双模式映射和周期利用率验证。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHANNELS = 32
TEMPORAL_SLOTS = 10
PAIR_GROUPS = TEMPORAL_SLOTS // 2
POSITIONS = 81


def direct_temporal_matrix(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """x为[T,P,C]，返回[输出T,P,C]，使用int64避免参考模型溢出。"""

    return np.einsum("oi,ipc->opc", weight.astype(np.int64), x.astype(np.int64)) + bias[:, None, None]


def dptme_t10(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    if x.shape[0] != 10 or weight.shape != (10, 10):
        raise ValueError("T10模式要求x[10,P,32]和weight[10,10]")
    positions = x.shape[1]
    result = np.broadcast_to(bias[:, None, None], (10, positions, CHANNELS)).astype(np.int64).copy()
    for position in range(positions):
        accum = result[:, position, :]
        for input_time in range(10):
            # 32个通道与10个输出时间槽形成32x10外积。
            accum += weight[:, input_time, None].astype(np.int64) * x[input_time, position, None, :]
    return result


def dptme_t2_five_way(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    if x.shape[0] != 2 or weight.shape != (2, 2):
        raise ValueError("T2模式要求x[2,P,32]和weight[2,2]")
    positions = x.shape[1]
    result = np.broadcast_to(bias[:, None, None], (2, positions, CHANNELS)).astype(np.int64).copy()
    for base in range(0, positions, PAIR_GROUPS):
        valid = min(PAIR_GROUPS, positions - base)
        for input_time in range(2):
            for group in range(valid):
                position = base + group
                result[:, position, :] += (
                    weight[:, input_time, None].astype(np.int64)
                    * x[input_time, position, None, :]
                )
    return result


def event_bits(hidden: np.ndarray, threshold: int) -> np.ndarray:
    return hidden >= int(threshold)


def run_trials(seed: int = 20260713, trials: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    mismatch_t10 = 0
    mismatch_t2 = 0
    event_mismatch_t10 = 0
    event_mismatch_t2 = 0
    compared_t10 = 0
    compared_t2 = 0
    for _ in range(trials):
        x10 = rng.integers(-128, 128, size=(10, POSITIONS, CHANNELS), dtype=np.int16)
        w10 = rng.integers(-128, 128, size=(10, 10), dtype=np.int16)
        b10 = rng.integers(-32768, 32768, size=(10,), dtype=np.int32)
        direct10 = direct_temporal_matrix(x10, w10, b10)
        mapped10 = dptme_t10(x10, w10, b10)
        mismatch_t10 += int(np.count_nonzero(direct10 != mapped10))
        event_mismatch_t10 += int(np.count_nonzero(event_bits(direct10, 64) != event_bits(mapped10, 64)))
        compared_t10 += int(direct10.size)

        x2 = rng.integers(-128, 128, size=(2, POSITIONS, CHANNELS), dtype=np.int16)
        w2 = rng.integers(-128, 128, size=(2, 2), dtype=np.int16)
        b2 = rng.integers(-32768, 32768, size=(2,), dtype=np.int32)
        direct2 = direct_temporal_matrix(x2, w2, b2)
        mapped2 = dptme_t2_five_way(x2, w2, b2)
        mismatch_t2 += int(np.count_nonzero(direct2 != mapped2))
        event_mismatch_t2 += int(np.count_nonzero(event_bits(direct2, 64) != event_bits(mapped2, 64)))
        compared_t2 += int(direct2.size)

    t10_cycles = POSITIONS * 10
    t2_cycles = ((POSITIONS + PAIR_GROUPS - 1) // PAIR_GROUPS) * 2
    t2_unpacked_cycles = POSITIONS * 2
    used_t2_slots = POSITIONS * 2
    allocated_t2_slots = ((POSITIONS + PAIR_GROUPS - 1) // PAIR_GROUPS) * TEMPORAL_SLOTS
    return {
        "seed": seed,
        "trials": trials,
        "geometry": {"channels": CHANNELS, "temporal_slots": TEMPORAL_SLOTS, "T2_groups": PAIR_GROUPS},
        "T10": {
            "compared_hidden_values": compared_t10,
            "hidden_mismatches": mismatch_t10,
            "event_mismatches": event_mismatch_t10,
            "cycles_per_81_position_head_tile": t10_cycles,
            "slot_utilization": 1.0,
        },
        "T2": {
            "compared_hidden_values": compared_t2,
            "hidden_mismatches": mismatch_t2,
            "event_mismatches": event_mismatch_t2,
            "cycles_per_81_position_head_tile": t2_cycles,
            "unpacked_cycles": t2_unpacked_cycles,
            "cycle_reduction_vs_unpacked": 1.0 - t2_cycles / t2_unpacked_cycles,
            "slot_utilization": used_t2_slots / allocated_t2_slots,
        },
        "限制": "该验证证明整数矩阵映射，不冻结输入/权重位宽、舍入、饱和或PyTorch浮点量化语义。",
    }


def write_markdown(result: dict, path: Path) -> None:
    t10 = result["T10"]
    t2 = result["T2"]
    lines = [
        "# DP-TME整数双模式映射验证",
        "",
        f"- 随机种子：`{result['seed']}`；试验：`{result['trials']}`组。",
        f"- T10比较 `{t10['compared_hidden_values']:,}` 个hidden值，hidden/event均为 `{t10['hidden_mismatches']}/{t10['event_mismatches']}` mismatch。",
        f"- T2五路打包比较 `{t2['compared_hidden_values']:,}` 个hidden值，hidden/event均为 `{t2['hidden_mismatches']}/{t2['event_mismatches']}` mismatch。",
        "",
        "## 81位置head tile周期",
        "",
        "| 模式 | 周期 | temporal slot利用率 | 对照 |",
        "|---|---:|---:|---|",
        f"| T10 | {t10['cycles_per_81_position_head_tile']} | {t10['slot_utilization']:.2%} | 32通道×10输出槽 |",
        f"| T2五路打包 | {t2['cycles_per_81_position_head_tile']} | {t2['slot_utilization']:.2%} | 相对不打包{t2['unpacked_cycles']}周期下降{t2['cycle_reduction_vs_unpacked']:.2%} |",
        "",
        "T2尾组只含1个空位置组，因此81个位置占85个可分配位置，slot利用率为95.29%。该结果只证明调度和整数累加等价；正式RTL还要冻结定点输入、权重、bias、threshold、累加宽度和溢出规则。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = run_trials()
    output = ROOT / "results/dptme_integer_mapping.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, output.with_suffix(".md"))
    if any(result[mode][key] for mode in ("T10", "T2") for key in ("hidden_mismatches", "event_mismatches")):
        return 1
    print(output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
