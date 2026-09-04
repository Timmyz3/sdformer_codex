#!/usr/bin/env python3
"""评估 Motion 可逆商编码 Token-Time Bundle 的流量与后端事务。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

try:
    from .evidence_provenance import (
        sha256_file,
        validate_motion_tesc_provenance,
    )
except ImportError:
    from evidence_provenance import (
        sha256_file,
        validate_motion_tesc_provenance,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPACT = ROOT / "results/profile100_compact_arch_stats_20260714.json"
DEFAULT_TESC = ROOT / "results/motion_temporal_equivalence_20260803/report.json"
DEFAULT_OUT = ROOT / "results/motion_rqtb_screen_20260806"


def file_binding(path: Path) -> dict:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"provenance文件不存在: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("比例分母必须为正数")
    return numerator / denominator


def _extract_stats(profile: dict) -> tuple[dict, int | None]:
    if "models" in profile:
        return profile["models"]["H67"]["binary_temporal_pairs"], None
    if "summary" in profile:
        records = profile["summary"].get("h60_records") or []
        tokens = int(records[0]["tokens"]) if records else None
        return profile["summary"]["binary_temporal_pairs"], tokens
    raise ValueError("不支持的 profile schema")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_identity(tesc: dict, temporal_tokens: int | None) -> tuple[dict, bool]:
    source = dict(tesc.get("source") or {})
    profile_path = str(tesc.get("profile", "未记录"))
    if source:
        if str(source.get("profile", "")) != profile_path:
            raise ValueError("TESC source与profile路径不一致")
        if int(source.get("temporal_tokens", -1)) != temporal_tokens:
            raise ValueError("TESC source与profile temporal token不一致")
    else:
        source = {"profile": profile_path, "temporal_tokens": temporal_tokens}

    fullres = (
        source.get("resolution") == [480, 640]
        and source.get("crop") is None
        and source.get("window_size") == [2, 15, 15]
        and int(source.get("temporal_tokens", -1)) == 450
        and int(source.get("samples", -1)) == 100
        and int(source.get("h60_records", -1)) == 1200
        and source.get("bn_policy") == "no_running"
        and _is_sha256(source.get("config_sha256"))
        and _is_sha256(source.get("checkpoint_sha256"))
    )
    return source, fullres


def build_model(compact: dict, tesc: dict) -> dict:
    stats, temporal_tokens = _extract_stats(compact)
    analysis = tesc["analysis"]
    source, fullres_identity = _source_identity(tesc, temporal_tokens)

    pair_total = int(stats["pair_total"])
    both_zero = int(stats["pair_kzero_both"])
    one_active = int(stats["pair_kzero_one"])
    both_active = int(stats["pair_both_active"])
    equal_total = int(stats["pair_score_equal_h67"])
    both_zero_equal = int(stats["pair_kzero_same_class_h67"])
    both_active_equal = int(analysis["active_pair_detail"]["pair_both_active_equal"])
    one_active_equal = equal_total - both_zero_equal - both_active_equal

    modes = {
        "zz_equal": both_zero_equal,
        "zz_split": both_zero - both_zero_equal,
        "one_equal": one_active_equal,
        "one_split": one_active - one_active_equal,
        "both_equal": both_active_equal,
        "both_split": both_active - both_active_equal,
    }
    if sum(modes.values()) != pair_total:
        raise ValueError("六种 bundle 模式不能覆盖全部 temporal pair")
    if any(value < 0 for value in modes.values()):
        raise ValueError("bundle 模式计数出现负数")

    unequal_total = pair_total - equal_total
    token_slots = pair_total * 2
    fixed_ttb_slots = pair_total * 2
    quotient_slots = equal_total + unequal_total * 2
    slot_bits = 16

    active_k_tokens = one_active + 2 * both_active
    pair_wide_k_reads = one_active + both_active
    pair_wide_k_bits = pair_wide_k_reads * 64
    bank_gated_k_bits = active_k_tokens * 32

    baseline_active_entries = int(analysis["baseline_scs_active_entries"])
    quotient_active_entries = int(analysis["quotient_scs_active_entries"])
    baseline_exp = int(analysis["baseline_scs_exp_transactions_model"])
    quotient_exp = int(analysis["quotient_scs_exp_transactions_model"])

    if baseline_active_entries != active_k_tokens:
        raise ValueError("active K token 与基线 SCS active entry 不一致")

    token_plus_k_bits = token_slots * slot_bits + bank_gated_k_bits
    quotient_plus_k_bits = quotient_slots * slot_bits + bank_gated_k_bits

    mode_rows = {
        name: {"pairs": value, "ratio": _ratio(value, pair_total)}
        for name, value in modes.items()
    }
    return {
        "schema": "motion_reversible_quotient_bundle_v2",
        "status": "BOUNDED_MODEL_COMPLETE",
        "evidence": "[prof-ordered]+[模型]；不是RTL cycle、SAIF或PPA",
        "source": {
            **source,
        },
        "contract": {
            "slot_bits": slot_bits,
            "common_slot": "{last, split=0, active_k_mask[1:0], score_class0[7:0], reserved}",
            "split_slots": "首slot携带score0与split=1；扩展slot携带score1；pair计数只在完整packet退休后递增",
            "normalization": "common packet向class histogram加入multiplicity=2；split packet各加入1",
            "reversibility": "Shiftmax后按active_k_mask读取K0/K1并恢复原token顺序；零K仍保留分母贡献",
        },
        "counts": {
            "pair_total": pair_total,
            "score_equal": equal_total,
            "score_split": unequal_total,
            "modes": mode_rows,
            "active_k_tokens": active_k_tokens,
        },
        "stream": {
            "token_slots_16b": token_slots,
            "fixed_ttb_slots_16b": fixed_ttb_slots,
            "quotient_slots_16b": quotient_slots,
            "quotient_slot_reduction_vs_token": 1.0 - _ratio(quotient_slots, token_slots),
            "quotient_slot_reduction_vs_fixed_ttb": 1.0
            - _ratio(quotient_slots, fixed_ttb_slots),
            "token_plus_bank_gated_k_bits": token_plus_k_bits,
            "quotient_plus_bank_gated_k_bits": quotient_plus_k_bits,
            "combined_bit_reduction": 1.0
            - _ratio(quotient_plus_k_bits, token_plus_k_bits),
        },
        "k_store": {
            "pair_wide_64b_read_bits": pair_wide_k_bits,
            "two_bank_active_mask_read_bits": bank_gated_k_bits,
            "read_bit_reduction": 1.0 - _ratio(bank_gated_k_bits, pair_wide_k_bits),
            "warning": "active-mask bank gating不是RQTB独占贡献，公平token/TTB基线也可采用",
        },
        "backend": {
            "baseline_scs_active_entries": baseline_active_entries,
            "quotient_scs_active_entries": quotient_active_entries,
            "active_entry_reduction": 1.0
            - _ratio(quotient_active_entries, baseline_active_entries),
            "baseline_exp_transactions_model": baseline_exp,
            "quotient_exp_transactions_model": quotient_exp,
            "exp_transaction_reduction_model": 1.0 - _ratio(quotient_exp, baseline_exp),
        },
        "admission": {
            "fullres_t450_profile_present": fullres_identity,
            "fullres_identity_contract": (
                "resolution=480x640,crop=null,window=2x15x15,T=450,samples=100,"
                "h60_records=1200,bn_policy=no_running,config/checkpoint SHA256"
            ),
            "minimum_slot_reduction": 0.25,
            "minimum_active_entry_reduction": 0.10,
            "minimum_exp_reduction": 0.10,
            "rtl_condition": "16-bit slot FIFO/SRAM与固定32-bit TTB同延迟、同反压、同K-store接口比较",
            "physical_condition": "同宏规则下面积归一吞吐非负，且score+SCS+K-store动态能量至少降低15%",
        },
        "limits": [
            "RQTB是TESC-WD的流接口与存储实现，不得与TESC重复计为两条贡献。",
            "若实现仍使用固定32-bit单packet接口，slot降低不会自动转化为周期收益。",
            "K-store bank gating可被强基线采用，不得归因于RQTB独占创新。",
            "是否属于fullres/W15/T450必须由profile身份与temporal_tokens审计，不能只看输出目录名。",
        ],
    }


def write_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    counts = result["counts"]
    stream = result["stream"]
    backend = result["backend"]
    k_store = result["k_store"]
    source = result["source"]
    scope = (
        f"T={source['temporal_tokens']}"
        if source["temporal_tokens"] is not None
        else "compact profile（T未内嵌）"
    )
    lines = [
        "# Motion 可逆商编码 Bundle 架构筛选",
        "",
        "## 结论",
        "",
        f"H67 ordered profile100（{scope}）表明，RQTB 可把 score→SCS 的 16-bit slot 数相对逐 token 或普通 fixed-TTB 均减少 "
        f"`{stream['quotient_slot_reduction_vs_fixed_ttb']:.2%}`，SCS active entry 减少 "
        f"`{backend['active_entry_reduction']:.2%}`，指数事务模型减少 "
        f"`{backend['exp_transaction_reduction_model']:.2%}` `[prof]+[模型]`。",
        "",
        "RQTB 不是 TESC 之外的新数学优化，而是把 TESC 的 exact temporal quotient 固化为可变长 16-bit slot 流、weighted directory 和可逆 gated-K 展开的架构接口。若物理实现仍固定每 pair 32 bit，则流量收益不能成立。",
        "",
        "## 六种精确模式",
        "",
        "| 模式 | pair | 比例 |",
        "|---|---:|---:|",
    ]
    labels = {
        "zz_equal": "双K零、同class",
        "zz_split": "双K零、双class",
        "one_equal": "单K有效、同class",
        "one_split": "单K有效、双class",
        "both_equal": "双K有效、同class",
        "both_split": "双K有效、双class",
    }
    for key, row in counts["modes"].items():
        lines.append(f"| {labels[key]} | {row['pairs']} | {row['ratio']:.2%} |")

    source_status = (
        "- 当前输入通过 fullres T450 完整身份合同 `[prof]`。"
        if result["admission"]["fullres_t450_profile_present"]
        else "- 当前输入未通过 fullres T450 完整身份合同，Motion T450 结论保持 `[待验证]`。"
    )
    lines.extend(
        [
            "",
            "## 流量与事务",
            "",
            "| 指标 | 基线 | RQTB | 变化 |",
            "|---|---:|---:|---:|",
            f"| 16-bit score流 slot | {stream['token_slots_16b']} | {stream['quotient_slots_16b']} | -{stream['quotient_slot_reduction_vs_token']:.2%} |",
            f"| score流+有效K payload bit | {stream['token_plus_bank_gated_k_bits']} | {stream['quotient_plus_bank_gated_k_bits']} | -{stream['combined_bit_reduction']:.2%} |",
            f"| SCS active entry | {backend['baseline_scs_active_entries']} | {backend['quotient_scs_active_entries']} | -{backend['active_entry_reduction']:.2%} |",
            f"| SCS exp事务模型 | {backend['baseline_exp_transactions_model']} | {backend['quotient_exp_transactions_model']} | -{backend['exp_transaction_reduction_model']:.2%} |",
            f"| K读取bit：64-bit pair读 vs 双bank mask读 | {k_store['pair_wide_64b_read_bits']} | {k_store['two_bank_active_mask_read_bits']} | -{k_store['read_bit_reduction']:.2%} |",
            "",
            "K 双bank按 mask 读取必须同时加入强基线，因此只作为存储实现要求，不列为 RQTB 独占收益。",
            "",
            "## 数据流",
            "",
            "```text",
            "temporal pair score",
            "  -> score0==score1 ? common 16b slot : split 2x16b slots",
            "  -> weighted SCS: multiplicity=2 或 1+1",
            "  -> Shiftmax",
            "  -> active_k_mask选择K0/K1 bank",
            "  -> 恢复原token顺序的gated-K",
            "```",
            "",
            "## DATE 创新边界",
            "",
            "可辩护的是“归一化域内取商、投影边界可逆展开的 token-time bundle 数据流”；不可把普通 TTB、16-bit FIFO、双bank K SRAM或 TESC/RQTB 两个名称分别计为贡献。",
            "",
            "本土化来源：Bishop 的 bundle 原子、Phi 的 common/residual 分层、Prosperity 的 exact reuse 成本纪律、FireFly-T 的时间维物理映射。差分是所有合并由整数 Q7 class 等价严格触发，Shiftmax 分母保留 multiplicity，gated-K 端不删除任何有效 K。",
            "",
            "## 晋级门槛",
            "",
            "1. fullres/W15/T450 的 slot 降低至少 25%、active entry 与 exp 事务均至少降低 10%；",
            "2. 16-bit slot 实现与固定 32-bit TTB 在同 K-store 延迟和随机反压下逐 gated-K 等价；",
            "3. 同宏规则物理比较中面积归一吞吐不退化，score+SCS+K-store 动态能量至少降低 15%；",
            "4. 未达到门槛则继续作为 TESC-WD 的内部编码，不新增论文贡献。",
            "",
            "## 证据边界",
            "",
            f"- pair 数：`{counts['pair_total']}`，来源 profile：`{source['profile']}` `[prof]`；",
            "- slot、bit 与 exp 数是模型，不是 RTL cycle、SAIF 或 ASIC PPA `[模型]`；",
            source_status,
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--tesc", type=Path, default=DEFAULT_TESC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--watcher", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    args = parser.parse_args()

    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    tesc = json.loads(args.tesc.read_text(encoding="utf-8"))
    validate_motion_tesc_provenance(tesc)
    result = build_model(compact, tesc)
    config_binding = file_binding(Path(result["source"]["config_path"]))
    checkpoint_binding = file_binding(Path(result["source"]["checkpoint_path"]))
    if config_binding["sha256"] != result["source"]["config_sha256"]:
        raise ValueError("TESC记录的config SHA与文件不一致")
    if checkpoint_binding["sha256"] != result["source"]["checkpoint_sha256"]:
        raise ValueError("TESC记录的checkpoint SHA与文件不一致")
    result["provenance"] = {
        "profile": file_binding(args.compact),
        "tesc_report": file_binding(args.tesc),
        "config": config_binding,
        "checkpoint": checkpoint_binding,
        "model": file_binding(Path(__file__)),
        "validator": file_binding(ROOT / "scripts/evidence_provenance.py"),
        "watcher": file_binding(args.watcher),
        "test_log": file_binding(args.test_log),
        "tests": [
            file_binding(ROOT / "tests/test_new_dual_line_architecture_models.py"),
            file_binding(ROOT / "tests/test_model_motion_reversible_quotient_bundle.py"),
            file_binding(ROOT / "tests/test_motion_model_provenance.py"),
        ],
    }
    write_report(result, args.out)
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
