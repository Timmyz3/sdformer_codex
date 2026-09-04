#!/usr/bin/env python3
"""运行 Prosperity 官方 CPU simulator，并冻结 SDformer 输入适配合同。"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import io
import json
import subprocess
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PROSPERITY = ROOT / "third_party" / "Prosperity"
SIM_DIR = PROSPERITY / "simulator"
DEFAULT_OUT = ROOT / "results" / "prosperity_official_reference_probe_20260729"
DEFAULT_DATA = PROSPERITY / "data" / "spikformer_cifar100.pkl"
DEFAULT_LAYERS = ("fc_q_enc_0", "fc_o_enc_0", "fc_2_enc_0")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def load_official_api():
    """载入未修改的官方 CPU 路径。

    官方 simulator.py 无条件导入 CUDA extension，但 run_fc CPU 路径不调用它。
    这里注入空模块只为通过 import，不替换任何 CPU 周期函数。
    """

    module_names = (
        "utils",
        "configs",
        "accelerator",
        "networks",
        "baselines",
        "energy",
        "simulator",
        "prosparsity_engine",
    )
    saved = {
        name: sys.modules.pop(name)
        for name in module_names
        if name in sys.modules
    }
    saved_path = list(sys.path)
    sys.path[:] = [str(SIM_DIR)] + [
        item for item in sys.path if item != str(SIM_DIR)
    ]
    try:
        sys.modules["prosparsity_engine"] = types.ModuleType(
            "prosparsity_engine"
        )
        accelerator_module = importlib.import_module("accelerator")
        networks_module = importlib.import_module("networks")
        simulator_module = importlib.import_module("simulator")
        result = (
            accelerator_module.Accelerator,
            networks_module.FC,
            simulator_module.Simulator,
            networks_module.create_network,
        )
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path
    return result


@dataclass(frozen=True)
class OfficialRun:
    layer: str
    mode: str
    activation_shape: list[int]
    activation_density: float
    total_cycles: int
    compute_cycles: int
    preprocess_stall_cycles: int
    memory_stall_cycles: int
    num_ops: int
    dram_reads: int
    dram_writes: int
    g_act_reads: int
    g_wgt_reads: int
    g_psum_reads: int
    g_psum_writes: int
    official_stdout: str


def run_official_fc(
    operator,
    product_sparsity: bool,
    *,
    spike_stored_in_buffer: bool = False,
    weight_stored_in_buffer: bool = False,
) -> OfficialRun:
    Accelerator, _, Simulator, _ = load_official_api()
    accelerator = Accelerator(
        type="Prosperity",
        adder_array_size=128,
        LIF_array_size=32,
        tile_size_M=256,
        tile_size_K=16,
        product_sparsity=product_sparsity,
        dense=False,
        issue_type=2,
        mem_if_width=1024,
    )
    op = copy.deepcopy(operator)
    density = float((op.activation_tensor.sparse_map != 0).float().mean())
    shape = list(op.activation_tensor.sparse_map.shape)
    simulator = Simulator(
        accelerator=accelerator,
        network=[op],
        benchmark_name="spikformer_cifar100",
        use_cuda=False,
    )
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        stats = simulator.run_fc(
            op,
            spike_stored_in_buffer=spike_stored_in_buffer,
            weight_stored_in_buffer=weight_stored_in_buffer,
        )
    return OfficialRun(
        layer=op.name,
        mode="product_sparsity" if product_sparsity else "bit_sparsity",
        activation_shape=shape,
        activation_density=density,
        total_cycles=int(stats.total_cycles),
        compute_cycles=int(stats.compute_cycles),
        preprocess_stall_cycles=int(stats.preprocess_stall_cycles),
        memory_stall_cycles=int(stats.mem_stall_cycles),
        num_ops=int(stats.num_ops),
        dram_reads=int(stats.reads["dram"]),
        dram_writes=int(stats.writes["dram"]),
        g_act_reads=int(stats.reads["g_act"]),
        g_wgt_reads=int(stats.reads["g_wgt"]),
        g_psum_reads=int(stats.reads["g_psum"]),
        g_psum_writes=int(stats.writes["g_psum"]),
        official_stdout=captured.getvalue(),
    )


def validate_sdformer_matrix_contract(
    activation: torch.Tensor,
    metadata: dict,
) -> None:
    required = {
        "schema",
        "line",
        "sample_id",
        "block",
        "head",
        "time_steps",
        "sequence_length",
        "input_dim",
        "output_dim",
        "semantic",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"缺少矩阵元数据: {missing}")
    if metadata["schema"] != "sdformer_binary_matrix_v1":
        raise ValueError("不支持的矩阵 schema")
    if metadata["line"] not in {"Motion", "Local5"}:
        raise ValueError("line 必须为 Motion 或 Local5")
    if activation.dtype not in {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("activation 必须是整数/布尔二值张量")
    if not bool(torch.all((activation == 0) | (activation == 1))):
        raise ValueError("activation 只能包含 0/1")
    expected = (
        int(metadata["time_steps"]),
        int(metadata["sequence_length"]),
        int(metadata["input_dim"]),
    )
    if tuple(activation.shape) != expected:
        raise ValueError(
            f"activation shape={tuple(activation.shape)}，期望={expected}"
        )
    if metadata["semantic"] not in {
        "projection_activation",
        "attention_q",
        "attention_k",
    }:
        raise ValueError("semantic 不在允许集合")


def build_report(layer_names: tuple[str, ...]) -> dict:
    _, FC, _, create_network = load_official_api()
    old_cwd = Path.cwd()
    try:
        # 官方 create_network 以相对路径解析 dataset 名和 pickle。
        import os

        os.chdir(PROSPERITY)
        network = create_network(
            "spikformer",
            "data/spikformer_cifar100.pkl",
        )
    finally:
        os.chdir(old_cwd)

    by_name = {
        op.name: op
        for op in network
        if isinstance(op, FC)
    }
    missing = sorted(set(layer_names) - by_name.keys())
    if missing:
        raise ValueError(f"官方 workload 缺少层: {missing}")

    runs = []
    for layer in layer_names:
        runs.append(run_official_fc(by_name[layer], True))
        runs.append(run_official_fc(by_name[layer], False))

    comparisons = {}
    for layer in layer_names:
        product = next(
            run
            for run in runs
            if run.layer == layer and run.mode == "product_sparsity"
        )
        bit = next(
            run
            for run in runs
            if run.layer == layer and run.mode == "bit_sparsity"
        )
        comparisons[layer] = {
            "official_cycle_speedup_product_vs_bit": (
                bit.total_cycles / product.total_cycles
            ),
            "official_g_wgt_read_reduction": (
                1.0 - product.g_wgt_reads / max(1, bit.g_wgt_reads)
            ),
        }

    return {
        "schema": "prosperity_official_reference_probe_v1",
        "generated_date": "2026-07-29",
        "official_repo": {
            "path": str(PROSPERITY.relative_to(ROOT)),
            "commit": git_commit(PROSPERITY),
            "simulator_sha256": sha256_file(SIM_DIR / "simulator.py"),
            "accelerator_sha256": sha256_file(SIM_DIR / "accelerator.py"),
            "networks_sha256": sha256_file(SIM_DIR / "networks.py"),
            "input": str(DEFAULT_DATA.relative_to(ROOT)),
            "input_sha256": sha256_file(DEFAULT_DATA),
        },
        "configuration": {
            "adder_array_size": 128,
            "tile_size_M": 256,
            "tile_size_K": 16,
            "mem_if_width": 1024,
            "issue_type": 2,
            "device": "cpu",
        },
        "runs": [asdict(run) for run in runs],
        "comparisons": comparisons,
        "sdformer_adapter_contract": {
            "schema": "sdformer_binary_matrix_v1",
            "activation_shape": "[time_steps, sequence_length, input_dim]",
            "activation_dtype": "bool/uint8，值域仅 0/1",
            "required_metadata": [
                "schema",
                "line",
                "sample_id",
                "block",
                "head",
                "time_steps",
                "sequence_length",
                "input_dim",
                "output_dim",
                "semantic",
            ],
            "semantic": [
                "projection_activation",
                "attention_q",
                "attention_k",
            ],
            "status": "合同和校验器已完成；真实 Motion/Local5 矩阵待 fullres follower 导出",
        },
        "evidence_boundary": [
            "本报告真实调用 Prosperity 官方 Simulator.run_fc CPU 周期路径",
            "官方 reference 数字只验证工具链，不代表 Motion/Local5",
            "本网络 profile 只有计数/直方图，不能重构 Prosperity 所需逐元素二值矩阵",
            "Phi 未发现公开官方 simulator，不能用本探针冒充 Phi 结果",
        ],
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Prosperity 官方 Simulator 复现与 SDformer 适配合同\n\n",
        "## 1. 结论\n\n",
        "本报告真实调用官方 `Simulator.run_fc` CPU 路径，不再只导入 `Stats`。"
        "结果用于验证工具链和确定适配字段，不能当作 Motion/Local5 性能。\n\n",
        "## 2. 官方 reference 结果\n\n",
        "| 层 | density | product cycles | bit-sparse cycles | 官方周期加速 | g_wgt 读取降低 |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    runs = report["runs"]
    for layer, comp in report["comparisons"].items():
        product = next(
            run
            for run in runs
            if run["layer"] == layer and run["mode"] == "product_sparsity"
        )
        bit = next(
            run
            for run in runs
            if run["layer"] == layer and run["mode"] == "bit_sparsity"
        )
        lines.append(
            f"| {layer} | {product['activation_density']:.4f} | "
            f"{product['total_cycles']} | {bit['total_cycles']} | "
            f"{comp['official_cycle_speedup_product_vs_bit']:.3f}× | "
            f"{100*comp['official_g_wgt_read_reduction']:.2f}% |\n"
        )
    lines.extend(
        [
            "\n## 3. CPU import 处理\n\n",
            "官方 `simulator.py` 无条件导入 CUDA extension，但 `run_fc` CPU 路径不调用该扩展。"
            "本探针只在 `sys.modules` 注入空的 import shim；未修改官方仓库，且周期、"
            "product-sparsity 搜索、存储分账均执行官方 Python 源码。\n\n",
            "## 4. SDformer 矩阵合同\n\n",
            "真实适配输入必须是 `[time_steps, sequence_length, input_dim]` 的 0/1 张量，"
            "并绑定主线、sample、block、head、输出维度和语义。计数、密度或 histogram "
            "不能替代逐元素矩阵。\n\n",
            "当前状态：合同与校验器已完成；Motion/Local5 真实矩阵等待 fullres follower "
            "增加导出。导出前，旧 `online-matcher oracle` 只能保留为本地解析下界。\n\n",
            "## 5. 证据边界\n\n",
        ]
    )
    for item in report["evidence_boundary"]:
        lines.append(f"- {item}。\n")
    lines.extend(
        [
            "\n## 6. 复现\n\n",
            "```bash\n",
            "/opt/conda/envs/sdformerflow/bin/python "
            "scripts/run_prosperity_official_probe.py\n",
            "```\n",
        ]
    )
    path.write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--layers",
        default=",".join(DEFAULT_LAYERS),
    )
    args = parser.parse_args()
    layers = tuple(item for item in args.layers.split(",") if item)
    report = build_report(layers)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    write_markdown(report, args.out / "report.md")
    print(args.out / "report.md")
    for layer, comp in report["comparisons"].items():
        print(
            layer,
            f"official_product_vs_bit="
            f"{comp['official_cycle_speedup_product_vs_bit']:.3f}x",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
