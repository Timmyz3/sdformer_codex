#!/usr/bin/env python3
"""M472: official Prosperity replay of frozen H67 Conv original16 matrices."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "contracts"
    / "m472_h67_official_prosperity_iso_workload_preflight_contract_r1_20260826.json"
)
DEFAULT_OUT = (
    ROOT
    / "results"
    / "m472_h67_official_prosperity_iso_workload_r1_20260826"
)

_ROWS_FD: int | None = None
_FC = None
_SIMULATOR = None
_ACCELERATOR = None
_ACCELS: dict[bool, Any] = {}
_ROWS_PER_PHASE = 0
_BYTES_PER_LINE = 9
_OUTPUT_DIM = 768
_N_TILE = 128
_N_TILES = 6
_MEM_IF_WIDTH = 1024
_OPERATORS: list[str] = []
_PARTITIONS = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def git_stdout(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def preflight(contract: dict[str, Any]) -> dict[str, Any]:
    frozen = contract["frozen_inputs"]
    checked: dict[str, Any] = {}
    for key in ("m410r2_manifest", "m410r2_rows", "docs359"):
        item = frozen[key]
        path = ROOT / item["path"]
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(
                f"{key} SHA mismatch: expected {item['sha256']} got {actual}"
            )
        checked[key] = {
            "path": item["path"],
            "sha256": actual,
            "bytes": path.stat().st_size,
        }

    manifest = json.loads(
        (ROOT / frozen["m410r2_manifest"]["path"]).read_text()
    )
    population = contract["population"]
    if manifest["layout"]["phase_order"] != "sample,operator,partition":
        raise RuntimeError("M410R2 phase order changed")
    if manifest["layout"]["row_order"] != "source_row_0_to_2999":
        raise RuntimeError("M410R2 row order changed")
    if manifest["layout"]["row_fields"]["original"] != [0, 15]:
        raise RuntimeError("M410R2 original field changed")
    if manifest["population"]["source_rows"] != population["source_rows"]:
        raise RuntimeError("M410R2 source-row population changed")

    rows_path = ROOT / frozen["m410r2_rows"]["path"]
    expected_bytes = population["source_rows"] * _BYTES_PER_LINE
    if rows_path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"fixed-width rows mismatch: expected {expected_bytes} bytes"
        )
    with rows_path.open("rb") as handle:
        first = handle.readline()
        handle.seek(-_BYTES_PER_LINE, os.SEEK_END)
        last = handle.read(_BYTES_PER_LINE)
    for label, row in (("first", first), ("last", last)):
        if len(row) != _BYTES_PER_LINE or row[-1:] != b"\n":
            raise RuntimeError(f"{label} row is not fixed-width 8-hex+newline")
        int(row[:8], 16)

    repo = ROOT / frozen["prosperity_repo"]["path"]
    commit = git_stdout(repo, "rev-parse", "HEAD")
    dirty = git_stdout(repo, "status", "--porcelain")
    if commit != frozen["prosperity_repo"]["commit"]:
        raise RuntimeError(f"Prosperity commit mismatch: {commit}")
    if frozen["prosperity_repo"]["must_be_clean"] and dirty:
        raise RuntimeError("Prosperity repository is dirty")
    checked["prosperity_repo"] = {
        "path": frozen["prosperity_repo"]["path"],
        "commit": commit,
        "clean": not bool(dirty),
        "simulator_sha256": sha256_file(repo / "simulator" / "simulator.py"),
        "accelerator_sha256": sha256_file(repo / "simulator" / "accelerator.py"),
        "networks_sha256": sha256_file(repo / "simulator" / "networks.py"),
    }
    checked["contract"] = {
        "path": str(CONTRACT.relative_to(ROOT)),
        "sha256": sha256_file(CONTRACT),
    }
    return checked


def worker_init(
    rows_path: str,
    rows_per_phase: int,
    partitions: int,
    operators: list[str],
) -> None:
    global _ROWS_FD, _FC, _SIMULATOR, _ACCELERATOR, _ACCELS
    global _ROWS_PER_PHASE, _PARTITIONS, _OPERATORS

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_prosperity_official_probe import load_official_api

    _ACCELERATOR, _FC, _SIMULATOR, _ = load_official_api()
    _ACCELS = {
        product: _ACCELERATOR(
            type="Prosperity",
            adder_array_size=_N_TILE,
            LIF_array_size=32,
            tile_size_M=256,
            tile_size_K=16,
            product_sparsity=product,
            dense=False,
            issue_type=2,
            mem_if_width=_MEM_IF_WIDTH,
        )
        for product in (False, True)
    }
    _ROWS_FD = os.open(rows_path, os.O_RDONLY)
    _ROWS_PER_PHASE = rows_per_phase
    _PARTITIONS = partitions
    _OPERATORS = operators


def decode_phase(phase_index: int) -> torch.Tensor:
    if _ROWS_FD is None:
        raise RuntimeError("worker not initialized")
    phase_bytes = _ROWS_PER_PHASE * _BYTES_PER_LINE
    raw = os.pread(_ROWS_FD, phase_bytes, phase_index * phase_bytes)
    if len(raw) != phase_bytes:
        raise RuntimeError(f"short read at phase {phase_index}")
    lines = raw.splitlines()
    if len(lines) != _ROWS_PER_PHASE:
        raise RuntimeError(f"row-count mismatch at phase {phase_index}")
    words = [int(line, 16) for line in lines]
    if any(word >> 32 for word in words):
        raise RuntimeError(f"non-32-bit word at phase {phase_index}")
    masks = torch.tensor(
        [word & 0xFFFF for word in words], dtype=torch.int64
    )
    shifts = torch.arange(16, dtype=torch.int64)
    return ((masks[:, None] >> shifts[None, :]) & 1).to(torch.uint8)


def decode_identity(phase_index: int) -> tuple[int, int, int]:
    sample_operator, partition = divmod(phase_index, _PARTITIONS)
    sample, operator = divmod(sample_operator, len(_OPERATORS))
    return sample, operator, partition


def run_official_n_tile(
    activation: torch.Tensor,
    *,
    phase_index: int,
    product: bool,
    output_dim: int,
) -> dict[str, int]:
    sample, operator_index, partition = decode_identity(phase_index)
    name = (
        f"h67_s{sample:02d}_o{operator_index}_p{partition:03d}_kernel3_fc"
    )
    op = _FC(name, 16, output_dim, _ROWS_PER_PHASE, 1, 1)
    op.activation_tensor.sparse_map = activation
    accelerator = _ACCELS[product]
    if output_dim != _N_TILE:
        accelerator = _ACCELERATOR(
            type="Prosperity",
            adder_array_size=_N_TILE,
            LIF_array_size=32,
            tile_size_M=256,
            tile_size_K=16,
            product_sparsity=product,
            dense=False,
            issue_type=2,
            mem_if_width=_MEM_IF_WIDTH,
        )
    sim = _SIMULATOR(
        accelerator=accelerator,
        network=[op],
        benchmark_name="h67_ep35_s10_four_bottleneck_conv3x3",
        use_cuda=False,
    )
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        stats = sim.run_fc(
            op,
            spike_stored_in_buffer=False,
            weight_stored_in_buffer=False,
        )
    return {
        "total_cycles": int(stats.total_cycles),
        "compute_cycles": int(stats.compute_cycles),
        "preprocess_stall_cycles": int(stats.preprocess_stall_cycles),
        "memory_stall_cycles": int(stats.mem_stall_cycles),
        "num_ops": int(stats.num_ops),
        "dram_reads": int(stats.reads["dram"]),
        "dram_writes": int(stats.writes["dram"]),
        "g_act_reads": int(stats.reads["g_act"]),
        "g_act_writes": int(stats.writes["g_act"]),
        "g_wgt_reads": int(stats.reads["g_wgt"]),
        "g_wgt_writes": int(stats.writes["g_wgt"]),
        "g_psum_reads": int(stats.reads["g_psum"]),
        "g_psum_writes": int(stats.writes["g_psum"]),
    }


def expand_n_tiles(one_tile: dict[str, int]) -> dict[str, int]:
    scaled_fields = (
        "compute_cycles",
        "preprocess_stall_cycles",
        "num_ops",
        "dram_reads",
        "dram_writes",
        "g_act_reads",
        "g_act_writes",
        "g_wgt_reads",
        "g_wgt_writes",
        "g_psum_reads",
        "g_psum_writes",
    )
    expanded = {
        field: one_tile[field] * _N_TILES for field in scaled_fields
    }
    initial_bits = (
        min(16, 16) * min(_N_TILE, _OUTPUT_DIM) * 8
        + min(16, 16) * min(256, _ROWS_PER_PHASE)
    )
    total_dram = expanded["dram_reads"] + expanded["dram_writes"]
    middle_bits = total_dram - initial_bits
    initial_latency = initial_bits // _MEM_IF_WIDTH
    middle_latency = middle_bits // _MEM_IF_WIDTH
    expanded["memory_stall_cycles"] = initial_latency + max(
        0, middle_latency - expanded["compute_cycles"]
    )
    expanded["total_cycles"] = (
        expanded["compute_cycles"] + expanded["memory_stall_cycles"]
    )
    return expanded


def worker_run(phase_index: int) -> dict[str, Any]:
    activation = decode_phase(phase_index)
    nnz = int(torch.count_nonzero(activation).item())
    modes: dict[str, Any] = {}
    for product in (False, True):
        primitive = run_official_n_tile(
            activation,
            phase_index=phase_index,
            product=product,
            output_dim=_N_TILE,
        )
        modes["product" if product else "bit"] = {
            "official_n128": primitive,
            "expanded_n768": expand_n_tiles(primitive),
        }
    sample, operator_index, partition = decode_identity(phase_index)
    bit_cycles = modes["bit"]["expanded_n768"]["total_cycles"]
    product_cycles = modes["product"]["expanded_n768"]["total_cycles"]
    return {
        "phase_index": phase_index,
        "sample": sample,
        "operator_index": operator_index,
        "operator": _OPERATORS[operator_index],
        "partition": partition,
        "input_nnz": nnz,
        "input_density": nnz / (_ROWS_PER_PHASE * 16),
        "modes": modes,
        "product_vs_bit_speedup": bit_cycles / max(1, product_cycles),
    }


COUNTER_FIELDS = (
    "total_cycles",
    "compute_cycles",
    "preprocess_stall_cycles",
    "memory_stall_cycles",
    "num_ops",
    "dram_reads",
    "dram_writes",
    "g_act_reads",
    "g_act_writes",
    "g_wgt_reads",
    "g_wgt_writes",
    "g_psum_reads",
    "g_psum_writes",
)


def aggregate(phases: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}

    def add(key: str, phase: dict[str, Any]) -> None:
        bucket = buckets.setdefault(
            key,
            {
                "phases": 0,
                "input_nnz": 0,
                "input_elements": 0,
                "bit": defaultdict(int),
                "product": defaultdict(int),
            },
        )
        bucket["phases"] += 1
        bucket["input_nnz"] += phase["input_nnz"]
        bucket["input_elements"] += _ROWS_PER_PHASE * 16
        for mode in ("bit", "product"):
            counters = phase["modes"][mode]["expanded_n768"]
            for field in COUNTER_FIELDS:
                bucket[mode][field] += counters[field]

    for phase in phases:
        add("overall", phase)
        add(f"sample:{phase['sample']:02d}", phase)
        add(f"operator:{phase['operator_index']}", phase)
        add(
            f"sample_operator:{phase['sample']:02d}:{phase['operator_index']}",
            phase,
        )

    result: dict[str, Any] = {}
    for key, bucket in buckets.items():
        bit = dict(bucket["bit"])
        product = dict(bucket["product"])
        result[key] = {
            "phases": bucket["phases"],
            "input_nnz": bucket["input_nnz"],
            "input_elements": bucket["input_elements"],
            "input_density": bucket["input_nnz"]
            / max(1, bucket["input_elements"]),
            "bit": bit,
            "product": product,
            "product_vs_bit_speedup": bit["total_cycles"]
            / max(1, product["total_cycles"]),
            "product_num_ops_reduction": 1.0
            - product["num_ops"] / max(1, bit["num_ops"]),
            "product_g_wgt_read_reduction": 1.0
            - product["g_wgt_reads"] / max(1, bit["g_wgt_reads"]),
        }
    return result


def validate_direct(
    indices: list[int],
    rows_path: Path,
    rows_per_phase: int,
    partitions: int,
    operators: list[str],
) -> list[dict[str, Any]]:
    worker_init(str(rows_path), rows_per_phase, partitions, operators)
    checks = []
    for phase_index in indices:
        activation = decode_phase(phase_index)
        for product in (False, True):
            n128 = run_official_n_tile(
                activation,
                phase_index=phase_index,
                product=product,
                output_dim=_N_TILE,
            )
            expanded = expand_n_tiles(n128)
            direct = run_official_n_tile(
                activation,
                phase_index=phase_index,
                product=product,
                output_dim=_OUTPUT_DIM,
            )
            mismatches = {
                field: {"expanded": expanded[field], "direct": direct[field]}
                for field in COUNTER_FIELDS
                if expanded[field] != direct[field]
            }
            checks.append(
                {
                    "phase_index": phase_index,
                    "mode": "product" if product else "bit",
                    "expanded": expanded,
                    "direct": direct,
                    "mismatches": mismatches,
                    "pass": not bool(mismatches),
                }
            )
    return checks


def write_markdown(report: dict[str, Any], path: Path) -> None:
    overall = report["aggregates"]["overall"]
    lines = [
        "# M472 H67 官方 Prosperity 同负载周期重放\n\n",
        "## 结论\n\n",
        f"状态：`{report['status']}`。在冻结 H67 ep35 S10 四层 bottleneck "
        "Conv3x3 的 original16 二值矩阵上，官方 Prosperity product-sparsity "
        f"相对其官方 bit-sparsity 模式为 **{overall['product_vs_bit_speedup']:.6f}×**。"
        "该数字只属于官方 Prosperity 配置和四层 Conv，不是 H67 全网或本项目硬件倍速。\n\n",
        "| 范围 | density | bit cycles | product cycles | product/bit speedup | product ops 降低 | g_wgt 读取降低 |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for index, operator in enumerate(report["operator_order"]):
        item = report["aggregates"].get(f"operator:{index}")
        if item is None:
            continue
        lines.append(
            f"| `{operator}` | {item['input_density']:.6f} | "
            f"{item['bit']['total_cycles']:,} | "
            f"{item['product']['total_cycles']:,} | "
            f"{item['product_vs_bit_speedup']:.6f}× | "
            f"{100*item['product_num_ops_reduction']:.3f}% | "
            f"{100*item['product_g_wgt_read_reduction']:.3f}% |\n"
        )
    lines.append(
        f"| **overall** | {overall['input_density']:.6f} | "
        f"**{overall['bit']['total_cycles']:,}** | "
        f"**{overall['product']['total_cycles']:,}** | "
        f"**{overall['product_vs_bit_speedup']:.6f}×** | "
        f"**{100*overall['product_num_ops_reduction']:.3f}%** | "
        f"**{100*overall['product_g_wgt_read_reduction']:.3f}%** |\n"
    )
    lines.extend(
        [
            "\n## 等价批量化证明\n\n",
            "每个 phase 先真实调用未修改的官方 `Simulator.run_fc` CPU 路径，"
            "输出维度取一个完整的 128-lane N tile；随后按官方源码方程展开为 "
            "N=768 的六个相同 N tile，并重新计算只发生一次的初始 DRAM 延迟。"
            f"{len(report['direct_validation'])} 个 mode-phase 直接 N=768 对照均为 "
            "0 mismatch。\n\n",
            "## 证据边界\n\n",
        ]
    )
    for item in report["claim_boundary_notes"]:
        lines.append(f"- {item}\n")
    lines.extend(
        [
            "\n## 复现\n\n",
            "```bash\n",
            "/opt/anaconda3/envs/pytorch310/bin/python "
            "scripts/run_m472_h67_official_prosperity_iso_workload.py\n",
            "```\n",
        ]
    )
    path.write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--workers", type=int, default=min(20, max(1, os.cpu_count() or 1))
    )
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--limit-phases", type=int, default=None)
    parser.add_argument("--validation-indices", default="0,431,17279")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.contract = args.contract.resolve()

    contract = json.loads(args.contract.read_text())
    if args.contract.resolve() != CONTRACT.resolve():
        raise RuntimeError("M472 accepts only the frozen r1 contract")
    checked = preflight(contract)
    population = contract["population"]
    total_phases = population["phases"]
    if args.limit_phases is not None:
        total_phases = min(total_phases, args.limit_phases)
    rows_path = ROOT / contract["frozen_inputs"]["m410r2_rows"]["path"]
    operators = contract["operator_order"]

    phases: list[dict[str, Any]] = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=worker_init,
        initargs=(
            str(rows_path),
            population["rows_per_partition"],
            population["partitions_per_operator"],
            operators,
        ),
    ) as executor:
        for completed, result in enumerate(
            executor.map(worker_run, range(total_phases), chunksize=args.chunksize),
            start=1,
        ):
            phases.append(result)
            if completed % 1000 == 0 or completed == total_phases:
                print(f"M472 phases {completed}/{total_phases}", flush=True)

    validation_indices = [
        int(item) for item in args.validation_indices.split(",") if item
    ]
    if args.limit_phases is not None:
        validation_indices = [
            item for item in validation_indices if item < total_phases
        ]
        if not validation_indices:
            validation_indices = [0]
    direct_validation = validate_direct(
        validation_indices,
        rows_path,
        population["rows_per_partition"],
        population["partitions_per_operator"],
        operators,
    )
    mismatch_count = sum(
        len(item["mismatches"]) for item in direct_validation
    )
    expected_checks = len(validation_indices) * 2
    if args.limit_phases is None:
        required = contract["exact_batching_proof_obligation"]
        if len(validation_indices) < required["required_direct_checks"]:
            raise RuntimeError("insufficient direct N=768 checks")
        if mismatch_count != required["required_mismatches"]:
            raise RuntimeError("N-tile expansion validation failed")
    if len(direct_validation) != expected_checks:
        raise RuntimeError("direct-validation population mismatch")

    aggregates = aggregate(phases)
    full_run = total_phases == population["phases"]
    status = (
        "PASS_M472_OFFICIAL_PROSPERITY_H67_ISO_WORKLOAD"
        if full_run and mismatch_count == 0
        else "PASS_M472_DEVELOPMENT_SUBSET"
    )
    report = {
        "schema": "m472_h67_official_prosperity_iso_workload_v1",
        "date": "2026-08-26",
        "status": status,
        "identity": checked,
        "configuration": contract["official_configuration"],
        "population": {
            **population,
            "executed_phases": total_phases,
            "full_run": full_run,
        },
        "operator_order": operators,
        "method": contract["exact_batching_proof_obligation"],
        "direct_validation": direct_validation,
        "direct_validation_mismatch_count": mismatch_count,
        "aggregates": aggregates,
        "phases": phases,
        "claim_boundary": contract["claim_boundary"],
        "claim_boundary_notes": [
            "真实调用 Prosperity 官方未修改 CPU `Simulator.run_fc`；CUDA 仅使用 import shim，周期函数未替换。",
            "输入是冻结 H67 ep35 S10 四层 bottleneck Conv3x3 的 original16 0/1 im2col 矩阵。",
            "product-vs-bit 是同一官方 Prosperity 配置内部的同负载对比。",
            "本结果不含 ATLIF、动态 BN、attention、FC、patch embed 或全网调度。",
            "官方 Prosperity 配置与 M430/M467 的资源和 cycle boundary 不同，禁止直接相除。",
            "没有由本周期重放推导能量、PPA、FPS、精度或系统倍速。",
        ],
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "m472_h67_official_prosperity_iso_workload_r1.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_markdown(report, args.out / "m472_h67_official_prosperity_iso_workload_r1.md")
    receipt = {
        "schema": "m472_h67_official_prosperity_iso_workload_receipt_v1",
        "status": status,
        "result": {
            "path": str(json_path.relative_to(ROOT)),
            "sha256": sha256_file(json_path),
        },
        "payload_sha256": report["payload_sha256"],
        "direct_validation_mismatch_count": mismatch_count,
        "overall": aggregates["overall"],
        "claim_boundary": contract["claim_boundary"],
    }
    receipt_path = args.out / "m472_h67_official_prosperity_iso_workload_receipt_r1.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    )
    print(json_path)
    print(
        "M472 official product-vs-bit",
        f"{aggregates['overall']['product_vs_bit_speedup']:.6f}x",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
