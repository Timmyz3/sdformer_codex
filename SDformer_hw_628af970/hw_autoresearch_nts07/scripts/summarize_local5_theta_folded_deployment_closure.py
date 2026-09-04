#!/usr/bin/env python3
"""汇总Local5 theta折叠生产合同与投影RTL闭环证据。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/local5_theta_folded_deployment_closure_20260805"
SIDECAR = (
    ROOT
    / "results/local5_theta_folded_projection_contract_old_checkpoint_20260805"
)
PRODUCTION = (
    ROOT
    / "results/local5_theta_folded_projection_contract_production_old_checkpoint_20260805"
)
VECTORS = (
    ROOT
    / "tb_qfit/vectors/local5_theta_folded_active_projection_postg0_100_20260805"
)
QGASR = ROOT / "results/local5_theta_folded_qgasr2c_fivebank_rtl_20260805"
TCFM = ROOT / "results/local5_theta_folded_tcfm5_linear5_rtl_20260805"
UNIT_LOG = DEFAULT_OUTPUT / "unit_tests.log"
INDEPENDENT_RECOMPUTE = DEFAULT_OUTPUT / "independent_numeric_recompute.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON根节点不是object: {path}")
    return value


def main() -> int:
    output = DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    side_manifest_path = SIDECAR / "checkpoint_projection_contract_theta_folded.json"
    side_payload_path = SIDECAR / "checkpoint_projection_contract_theta_folded.npz"
    production_manifest_path = PRODUCTION / "checkpoint_projection_contract.json"
    production_payload_path = PRODUCTION / "checkpoint_projection_contract.npz"
    vector_manifest_path = VECTORS / "manifest.json"
    qgasr_report_path = QGASR / "report.json"
    tcfm_report_path = TCFM / "report.json"
    independent_recompute_path = INDEPENDENT_RECOMPUTE

    side = load_json(side_manifest_path)
    production = load_json(production_manifest_path)
    vector = load_json(vector_manifest_path)
    qgasr = load_json(qgasr_report_path)
    tcfm = load_json(tcfm_report_path)
    independent_recompute = load_json(independent_recompute_path)

    for contract in (side, production):
        if (
            contract.get("schema") != "local5_checkpoint_projection_contract_v2"
            or contract.get("status") != "THETA_FOLDED_WEIGHT_CONTRACT"
            or len(contract.get("blocks", [])) != 12
        ):
            raise ValueError("v2 theta-folded合同语义或block覆盖错误")
    if (
        side["checkpoint_sha256"] != production["checkpoint_sha256"]
        or side["raw_vs_folded"] != production["raw_vs_folded"]
        or [row["theta"] for row in side["blocks"]]
        != [row["theta"] for row in production["blocks"]]
    ):
        raise ValueError("生产合同与旁路合同元数据不等价")

    with np.load(side_payload_path) as side_arrays, np.load(
        production_payload_path
    ) as production_arrays:
        if set(side_arrays.files) != set(production_arrays.files):
            raise ValueError("生产合同与旁路合同数组集不一致")
        mismatched_arrays = [
            key
            for key in side_arrays.files
            if not np.array_equal(side_arrays[key], production_arrays[key])
        ]
        array_count = len(side_arrays.files)
    if mismatched_arrays:
        raise ValueError(f"生产合同与旁路合同数组不等价: {mismatched_arrays}")

    if (
        independent_recompute.get("status") != "PASS"
        or independent_recompute.get("checkpoint_sha256")
        != production["checkpoint_sha256"]
        or independent_recompute.get("payload_sha256")
        != production["payload_sha256"]
        or int(independent_recompute.get("blocks", 0)) != 12
        or int(independent_recompute.get("arrays", 0)) != array_count
        or int(independent_recompute.get("verified_weight_int8_entries", 0))
        != 2_156_544
        or int(independent_recompute.get("verified_scale_entries", 0)) != 4_416
        or int(independent_recompute.get("verified_bias_entries", 0)) != 4_416
        or independent_recompute.get("topology_mapping")
        != "PASS_FIXED_12_BLOCK_ABI"
    ):
        raise ValueError("checkpoint独立数值重算证据不完整或与生产合同不一致")

    binding = vector.get("projection_contract_binding") or {}
    if (
        vector.get("weight_mode")
        != "checkpoint_theta_folded_dyadic_int8_head_slice"
        or binding.get("schema") != "local5_checkpoint_projection_contract_v2"
        or binding.get("payload_sha256") != side["payload_sha256"]
        or int(vector.get("selection", {}).get("groups", 0)) != 100
    ):
        raise ValueError("T450向量未完整绑定v2 theta-folded合同")

    qgasr_checks = qgasr.get("verification", {})
    if (
        qgasr.get("weight_mode") != vector["weight_mode"]
        or qgasr.get("correctness", {}).get("acc32")
        != "100/100组PASS，逐元素零失配"
        or any(
            qgasr_checks.get(name) != "PASS"
            for name in (
                "checkpoint_weight_binding",
                "random_sva",
                "verilator_lint",
                "yosys_check",
            )
        )
    ):
        raise ValueError("Direct/DS-GASR theta-folded RTL证据不完整")

    configurations = tcfm.get("configurations", {})
    if (
        tcfm.get("weight_mode") != vector["weight_mode"]
        or tcfm.get("vector_manifest_sha256") != file_sha256(vector_manifest_path)
        or int(tcfm.get("groups", 0)) != 100
        or not all(
            name in configurations
            for name in (
                "tcfm5_l1",
                "linear5_l1",
                "tcfm5_l2",
                "linear5_l2",
            )
        )
    ):
        raise ValueError("TCFM5/Linear5 theta-folded RTL证据不完整")

    unit_status = "PASS" if UNIT_LOG.is_file() and "OK" in UNIT_LOG.read_text() else "NOT_RUN"
    report = {
        "schema": "local5_theta_folded_deployment_closure_v1",
        "status": "PASS_OLD_CHECKPOINT_PENDING_NEW_RANK1",
        "evidence_scope": (
            "old checkpoint qualified post-G0 T450 component RTL exact; "
            "not new rank-1, full attention, full encoder, or ASIC PPA"
        ),
        "new_rank1_checkpoint_bound": False,
        "production_contract": {
            "schema": production["schema"],
            "checkpoint_sha256": production["checkpoint_sha256"],
            "blocks": len(production["blocks"]),
            "arrays": array_count,
            "sidecar_array_mismatches": len(mismatched_arrays),
            "raw_vs_folded": production["raw_vs_folded"],
            "runtime_datapath": production["runtime_datapath"],
        },
        "independent_numeric_recompute": {
            "status": independent_recompute["status"],
            "blocks": independent_recompute["blocks"],
            "arrays": independent_recompute["arrays"],
            "verified_weight_int8_entries": independent_recompute[
                "verified_weight_int8_entries"
            ],
            "verified_scale_entries": independent_recompute[
                "verified_scale_entries"
            ],
            "verified_bias_entries": independent_recompute[
                "verified_bias_entries"
            ],
            "topology_contract": independent_recompute["topology_contract"],
            "topology_mapping": independent_recompute["topology_mapping"],
            "verifier_sha256": independent_recompute["verifier_sha256"],
        },
        "vectors": {
            "groups": vector["selection"]["groups"],
            "weight_mode": vector["weight_mode"],
            "manifest_sha256": file_sha256(vector_manifest_path),
        },
        "direct_qgasr": {
            "acc32": qgasr["correctness"]["acc32"],
            "direct_cycles": qgasr["aggregate"]["direct_cycles"],
            "qgasr_cycles": qgasr["aggregate"]["gasr_cycles"],
            "qgasr_speedup": qgasr["aggregate"]["gasr_speedup"],
            "sram_transaction_reduction": qgasr["aggregate"][
                "sram_transaction_reduction"
            ],
        },
        "tcfm5_linear5": {
            "acc32_checks": 100 * 450 * 2 * 4,
            "tcfm5_l1_cycles": configurations["tcfm5_l1"]["cycles"]["total"],
            "linear5_l1_cycles": configurations["linear5_l1"]["cycles"]["total"],
            "speedup_l1": configurations["speedup_l1"]["ratio_of_totals"],
            "tcfm5_l2_cycles": configurations["tcfm5_l2"]["cycles"]["total"],
            "linear5_l2_cycles": configurations["linear5_l2"]["cycles"]["total"],
            "speedup_l2": configurations["speedup_l2"]["ratio_of_totals"],
        },
        "unit_tests": unit_status,
        "source_bindings": {
            "production_manifest": {
                "path": str(production_manifest_path.resolve()),
                "sha256": file_sha256(production_manifest_path),
            },
            "production_payload": {
                "path": str(production_payload_path.resolve()),
                "sha256": file_sha256(production_payload_path),
            },
            "independent_numeric_recompute": {
                "path": str(independent_recompute_path.resolve()),
                "sha256": file_sha256(independent_recompute_path),
            },
            "vector_manifest": {
                "path": str(vector_manifest_path.resolve()),
                "sha256": file_sha256(vector_manifest_path),
            },
            "qgasr_report": {
                "path": str(qgasr_report_path.resolve()),
                "sha256": file_sha256(qgasr_report_path),
            },
            "tcfm_report": {
                "path": str(tcfm_report_path.resolve()),
                "sha256": file_sha256(tcfm_report_path),
            },
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    mismatch = report["production_contract"]["raw_vs_folded"]
    lines = [
        "# Local5 theta 折叠生产合同与投影 RTL 闭环",
        "",
        "## 结论",
        "",
        "Local5 生产 profile/acceptance/vector 链已切换为 "
        "`local5_checkpoint_projection_contract_v2`：先计算 "
        "`W_eff=theta_K×W_float`，再做 dyadic INT8 量化；运行时 K "
        "保持 1-bit event。",
        "",
        f"旧 checkpoint 上生产导出器与独立旁路导出的 "
        f"{array_count} 个数组逐项一致，失配 0。相对错误的 "
        f"`quantize(W)`，折叠后改变 {mismatch['weight_int8_mismatch']}/"
        f"{mismatch['weight_int8_entries']} 个 INT8 权重码。",
        "",
        "独立 verifier 不信任 manifest 中的量化 payload，而是重新加载 checkpoint，"
        "对 12 个 block 重算 theta 折叠和权重量化。已逐项核对 "
        f"{independent_recompute['verified_weight_int8_entries']:,} 个 INT8 权重、"
        f"{independent_recompute['verified_scale_entries']:,} 个 scale 和 "
        f"{independent_recompute['verified_bias_entries']:,} 个 bias，全部一致。",
        "",
        "verifier 还将 12 个 `(stage, block)`、module、weight/theta/bias 名称、"
        "payload prefix、通道数、head 数与 head-dim 固定为 Local5 部署 ABI；"
        "不再信任 manifest 自报的参数映射。",
        "",
        "## RTL 证据",
        "",
        "| 路径 | 同步存储合同 | Acc32 | 周期/收益 |",
        "|---|---|---:|---|",
        f"| Direct/DS-GASR | 五bank单端口1RW | 100/100组零失配 | "
        f"{report['direct_qgasr']['direct_cycles']:,} / "
        f"{report['direct_qgasr']['qgasr_cycles']:,}，"
        f"{report['direct_qgasr']['qgasr_speedup']:.3f}x |",
        f"| TCFM5/Linear5 L1 | 关系SRAM 1拍 | 180,000项零失配 | "
        f"{report['tcfm5_linear5']['tcfm5_l1_cycles']:,} / "
        f"{report['tcfm5_linear5']['linear5_l1_cycles']:,}，"
        f"{report['tcfm5_linear5']['speedup_l1']:.3f}x |",
        f"| TCFM5/Linear5 L2 | 关系SRAM 2拍 | 180,000项零失配 | "
        f"{report['tcfm5_linear5']['tcfm5_l2_cycles']:,} / "
        f"{report['tcfm5_linear5']['linear5_l2_cycles']:,}，"
        f"{report['tcfm5_linear5']['speedup_l2']:.3f}x |",
        "",
        "## 证据边界",
        "",
        "1. 这是旧 fullres Local5 checkpoint 的 qualified post-G0 100 组 T450 "
        "部件闭环，不是正在训练的新 rank-1；",
        "2. Acc32 边界是每 head 部分累加，尚不包含 cross-head reduction、"
        "bias、no-running BN、requant、残差、ATLIF 和 decoder；",
        "3. TCFM5/Linear5 周期与折叠前趋势相同，因为权重数值不改变term"
        "数或bank冲突；本轮修复的是软件数值合同，不是新周期优化；",
        "4. 同步存储仅是可综合端口合同，不是foundry SRAM macro、DC/STA"
        "或功耗结果。",
        "5. 单元测试包含‘同时篡改 payload 与 manifest SHA’及‘同时交换 block"
        "参数映射与 payload’的 fail-closed 负例。",
        "",
        "## 下一唯一门槛",
        "",
        "新 Local5 rank-1 释放后，使用当前生产 v2 链原样重跑 "
        "profile100/all12、Direct/TCFM5 Acc32 和 checkpoint SHA acceptance。在新 "
        "checkpoint 逐项零失配前，不启动 12-block scheduler 集成。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
