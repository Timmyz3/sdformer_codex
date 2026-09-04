#!/usr/bin/env python3
"""M618: frozen H67 binary-FC1 replay on official Prosperity CPU run_fc."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import types
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "contracts"
    / "m618_h67_ep35_binary_fc1_official_prosperity_iso_workload_contract_r1_20260828.json"
)
DEFAULT_OUT = (
    ROOT
    / "results"
    / "m618_h67_ep35_binary_fc1_official_prosperity_dev_r1_20260828"
)

_FC = None
_SIMULATOR = None
_ACCELERATOR = None
_SIM_MODULE = None
_ACCELS: dict[bool, Any] = {}
_PAYLOAD_ROOT: Path | None = None
_DIRECT_BASENAMES: set[str] = set()

N_TILE = 128
M_TILE = 256
K_TILE = 16
MEM_IF_WIDTH = 1024

COUNTER_FIELDS = (
    "total_cycles",
    "compute_cycles",
    "raw_issue_cycles",
    "raw_preprocess_cycles",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise RuntimeError(f"non-standard JSON constant: {value}")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject,
    )


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"invalid shape dimension: {value!r}")
        result *= value
    return result


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def subset_collection_sha(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            (
                record["relative_path"]
                + "\0"
                + record["file_sha256"]
                + "\0"
                + str(record["packed_bytes"])
                + "\0"
                + str(record["active_elements"])
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def preflight(contract: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen = contract["frozen_inputs"]
    checked: dict[str, Any] = {}
    for key in ("m51_manifest", "m51_validation_receipt", "docs359"):
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

    manifest_path = ROOT / frozen["m51_manifest"]["path"]
    manifest = strict_json(manifest_path)
    if manifest.get("schema") != "m51_h67_ep35_binary_input_trace_manifest_v1":
        raise RuntimeError("M51 manifest schema mismatch")
    receipt = strict_json(ROOT / frozen["m51_validation_receipt"]["path"])
    if not str(receipt.get("status", "")).startswith("PASS_REAL_GPU"):
        raise RuntimeError("M51 independent GPU validation receipt is not PASS")

    records = sorted(
        [row for row in manifest["records"] if row["name"].endswith(".mlp.fc1")],
        key=lambda row: (row["sample_id"], row["module_index"]),
    )
    population = contract["population"]
    module_names = sorted({row["name"] for row in records})
    if len(records) != population["records"] or len(module_names) != population["binary_fc1_modules"]:
        raise RuntimeError("binary FC1 record/module population mismatch")
    if sorted({row["sample_id"] for row in records}) != list(range(population["samples"])):
        raise RuntimeError("binary FC1 sample population mismatch")
    if any(".layers.3." in name for name in module_names):
        raise RuntimeError("stage-3 FC1 illegally entered exact-binary population")
    stage_counts = {
        f"stage{stage}": sum(f".layers.{stage}." in name for name in module_names)
        for stage in range(4)
    }
    if stage_counts != population["stages"]:
        raise RuntimeError(f"FC1 stage population mismatch: {stage_counts}")

    payload_root = manifest_path.parent
    popcount_table = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    totals = defaultdict(int)
    for row in records:
        shape = row["input_shape"]
        if len(shape) != 5 or shape[1] != 1:
            raise RuntimeError(f"unsupported non-[T,1,H,W,C] shape: {shape}")
        if product(shape) != row["input_elements"]:
            raise RuntimeError("M51 FC1 input shape arithmetic mismatch")
        if (row["input_elements"] + 7) // 8 != row["packed_bytes"]:
            raise RuntimeError("M51 FC1 packed-byte arithmetic mismatch")
        identity = manifest["module_identities"][row["name"]]["weight"]
        output_shape = row["output_shape"]
        if len(output_shape) != 5 or output_shape[:4] != shape[:4]:
            raise RuntimeError("FC1 input/output row-domain mismatch")
        n_dim, k_dim = identity["shape"]
        if k_dim != shape[-1] or n_dim != output_shape[-1]:
            raise RuntimeError(f"FC1 weight K/N identity mismatch: {row['name']}")
        if k_dim % K_TILE or n_dim % N_TILE:
            raise RuntimeError("M618 exact N expansion requires K%16=N%128=0")

        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("unsafe M51 relative payload path")
        path = payload_root / relative
        if not path.is_file():
            raise RuntimeError(f"selected FC1 payload missing: {relative}")
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size != row["packed_bytes"]:
            raise RuntimeError(f"FC1 payload size mismatch: {relative}")
        if sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"FC1 payload SHA mismatch: {relative}")
        active = int(popcount_table[raw].sum(dtype=np.uint64))
        if active != row["active_elements"]:
            raise RuntimeError(f"FC1 payload popcount mismatch: {relative}")
        used = row["input_elements"] % 8 or 8
        if used < 8 and int(raw[-1]) & ~((1 << used) - 1):
            raise RuntimeError(f"FC1 packed tail is nonzero: {relative}")
        totals["input_elements"] += row["input_elements"]
        totals["packed_bytes"] += row["packed_bytes"]
        totals["active_elements"] += row["active_elements"]

    for field in ("input_elements", "packed_bytes", "active_elements"):
        if totals[field] != population[field]:
            raise RuntimeError(f"FC1 population {field} mismatch")
    collection = subset_collection_sha(records)
    expected_collection = frozen["fc1_subset_collection"]["sha256"]
    if collection != expected_collection:
        raise RuntimeError("FC1 subset collection SHA mismatch")
    checked["fc1_subset_collection"] = {
        "sha256": collection,
        "records": len(records),
        **dict(totals),
    }

    repo_item = frozen["official_prosperity_repo"]
    repo = Path(repo_item["path"])
    commit = git_text(repo, "rev-parse", "HEAD")
    dirty = git_text(repo, "status", "--porcelain", "--untracked-files=all")
    if commit != repo_item["commit"]:
        raise RuntimeError(f"official Prosperity commit mismatch: {commit}")
    if repo_item["must_be_clean"] and dirty:
        raise RuntimeError(f"official Prosperity repository is dirty: {dirty}")
    official_files: dict[str, str] = {}
    for relative, expected in repo_item["files"].items():
        actual = sha256_file(repo / relative)
        if actual != expected:
            raise RuntimeError(f"official source SHA mismatch: {relative}")
        official_files[relative] = actual
    checked["official_prosperity_repo"] = {
        "path": str(repo),
        "commit": commit,
        "clean": not bool(dirty),
        "files": official_files,
    }
    checked["contract"] = {
        "path": str(CONTRACT.relative_to(ROOT)),
        "sha256": sha256_file(CONTRACT),
    }
    return checked, records


def load_official_api(repo: Path) -> tuple[Any, Any, Any, Any]:
    """Import the official CPU path without writing or replacing its functions."""

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
    saved = {name: sys.modules.pop(name) for name in module_names if name in sys.modules}
    saved_path = list(sys.path)
    sim_dir = repo / "simulator"
    sys.path[:] = [str(sim_dir)] + [item for item in sys.path if item != str(sim_dir)]
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.modules["prosparsity_engine"] = types.ModuleType("prosparsity_engine")
        accelerator_module = importlib.import_module("accelerator")
        networks_module = importlib.import_module("networks")
        simulator_module = importlib.import_module("simulator")
        result = (
            accelerator_module.Accelerator,
            networks_module.FC,
            simulator_module.Simulator,
            simulator_module,
        )
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path
        sys.dont_write_bytecode = old_dont_write
    return result


def worker_init(payload_root: str, repo: str, direct_basenames: list[str]) -> None:
    global _FC, _SIMULATOR, _ACCELERATOR, _SIM_MODULE, _ACCELS
    global _PAYLOAD_ROOT, _DIRECT_BASENAMES
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _ACCELERATOR, _FC, _SIMULATOR, _SIM_MODULE = load_official_api(Path(repo))
    _ACCELS = {
        product_mode: _ACCELERATOR(
            type="Prosperity",
            adder_array_size=N_TILE,
            LIF_array_size=32,
            tile_size_M=M_TILE,
            tile_size_K=K_TILE,
            product_sparsity=product_mode,
            dense=False,
            issue_type=2,
            mem_if_width=MEM_IF_WIDTH,
        )
        for product_mode in (False, True)
    }
    _PAYLOAD_ROOT = Path(payload_root)
    _DIRECT_BASENAMES = set(direct_basenames)


def decode_activation(record: dict[str, Any]) -> torch.Tensor:
    if _PAYLOAD_ROOT is None:
        raise RuntimeError("worker payload root is not initialized")
    raw = np.fromfile(_PAYLOAD_ROOT / record["relative_path"], dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder="little", count=record["input_elements"])
    array = bits.reshape(record["input_shape"])
    # The official path performs [T,sequence,K] -> [sequence,T,K].
    t_dim, b_dim, h_dim, w_dim, k_dim = record["input_shape"]
    if b_dim != 1:
        raise RuntimeError("official run_fc CPU path supports B=1 only")
    return torch.from_numpy(array.reshape(t_dim, b_dim * h_dim * w_dim, k_dim).copy())


def parse_official_stdout(raw: str) -> dict[str, int]:
    patterns = {
        "raw_issue_cycles": r"^compute cycles:\s+([0-9]+)\s*$",
        "raw_preprocess_cycles": r"^preprocess cycles:\s+([0-9]+)\s*$",
    }
    result: dict[str, int] = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, raw, flags=re.MULTILINE)
        if match is None:
            raise RuntimeError(f"official stdout lacks {field}")
        result[field] = int(match.group(1))
    return result


def run_official(
    activation: torch.Tensor,
    record: dict[str, Any],
    product_mode: bool,
    output_dim: int,
) -> dict[str, int]:
    if _FC is None or _SIMULATOR is None or _ACCELERATOR is None:
        raise RuntimeError("official API is not initialized")
    t_dim, _, h_dim, w_dim, k_dim = record["input_shape"]
    op = _FC(record["name"], k_dim, output_dim, h_dim * w_dim, 1, t_dim)
    op.activation_tensor.sparse_map = activation
    accelerator = _ACCELS[product_mode]
    if output_dim != N_TILE:
        accelerator = _ACCELERATOR(
            type="Prosperity",
            adder_array_size=N_TILE,
            LIF_array_size=32,
            tile_size_M=M_TILE,
            tile_size_K=K_TILE,
            product_sparsity=product_mode,
            dense=False,
            issue_type=2,
            mem_if_width=MEM_IF_WIDTH,
        )
    if hasattr(_SIM_MODULE, "clear_global_stats"):
        _SIM_MODULE.clear_global_stats()
    sim = _SIMULATOR(
        accelerator=accelerator,
        network=[op],
        benchmark_name="h67_ep35_binary_fc1_s10",
        use_cuda=False,
    )
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        stats = sim.run_fc(
            op,
            spike_stored_in_buffer=False,
            weight_stored_in_buffer=False,
        )
    parsed = parse_official_stdout(captured.getvalue())
    return {
        "total_cycles": int(stats.total_cycles),
        "compute_cycles": int(stats.compute_cycles),
        "raw_issue_cycles": parsed["raw_issue_cycles"],
        "raw_preprocess_cycles": parsed["raw_preprocess_cycles"],
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


def expand_n_tiles(one_tile: dict[str, int], *, m_dim: int, k_dim: int, n_dim: int) -> dict[str, int]:
    if n_dim % N_TILE:
        raise RuntimeError("N dimension is not an exact official N-tile multiple")
    n_tiles = n_dim // N_TILE
    expanded = {
        field: one_tile[field] * n_tiles
        for field in COUNTER_FIELDS
        if field not in ("total_cycles", "memory_stall_cycles")
    }
    initial_bits = min(K_TILE, k_dim) * min(N_TILE, n_dim) * 8
    initial_bits += min(K_TILE, k_dim) * min(M_TILE, m_dim)
    total_dram_bits = expanded["dram_reads"] + expanded["dram_writes"]
    middle_bits = total_dram_bits - initial_bits
    if middle_bits < 0:
        raise RuntimeError("official memory expansion produced negative middle transfer")
    initial_latency = initial_bits // MEM_IF_WIDTH
    middle_latency = middle_bits // MEM_IF_WIDTH
    expanded["memory_stall_cycles"] = initial_latency + max(
        0, middle_latency - expanded["compute_cycles"]
    )
    expanded["total_cycles"] = (
        expanded["compute_cycles"] + expanded["memory_stall_cycles"]
    )
    return expanded


def derived_counters(counters: dict[str, int], n_dim: int) -> dict[str, Any]:
    return {
        **counters,
        "dram_bits": counters["dram_reads"] + counters["dram_writes"],
        "global_buffer_bits": sum(
            counters[field]
            for field in (
                "g_act_reads",
                "g_act_writes",
                "g_wgt_reads",
                "g_wgt_writes",
                "g_psum_reads",
                "g_psum_writes",
            )
        ),
        "support_nnz": counters["num_ops"] // n_dim,
        "support_nnz_divisible_by_N": counters["num_ops"] % n_dim == 0,
    }


def worker_run(record: dict[str, Any]) -> dict[str, Any]:
    activation = decode_activation(record)
    t_dim, b_dim, h_dim, w_dim, k_dim = record["input_shape"]
    n_dim = record["output_shape"][-1]
    m_dim = t_dim * b_dim * h_dim * w_dim
    observed_nnz = int(torch.count_nonzero(activation).item())
    if observed_nnz != record["active_elements"]:
        raise RuntimeError(f"decoded FC1 nnz mismatch: {record['relative_path']}")

    modes: dict[str, Any] = {}
    direct_checks: list[dict[str, Any]] = []
    for product_mode in (False, True):
        label = "product" if product_mode else "bit"
        primitive = run_official(activation, record, product_mode, N_TILE)
        expanded = expand_n_tiles(primitive, m_dim=m_dim, k_dim=k_dim, n_dim=n_dim)
        modes[label] = {
            "official_n128": derived_counters(primitive, N_TILE),
            "expanded_full_n": derived_counters(expanded, n_dim),
        }
        if Path(record["relative_path"]).name in _DIRECT_BASENAMES:
            direct = run_official(activation, record, product_mode, n_dim)
            mismatches = {
                field: {"expanded": expanded[field], "direct": direct[field]}
                for field in COUNTER_FIELDS
                if expanded[field] != direct[field]
            }
            direct_checks.append(
                {
                    "mode": label,
                    "direct_full_n": derived_counters(direct, n_dim),
                    "mismatches": mismatches,
                    "pass": not mismatches,
                }
            )

    bit = modes["bit"]["expanded_full_n"]
    product_result = modes["product"]["expanded_full_n"]
    if not bit["support_nnz_divisible_by_N"] or not product_result["support_nnz_divisible_by_N"]:
        raise RuntimeError("official num_ops is not divisible by output N")
    module_stage = int(re.search(r"\.layers\.(\d+)\.", record["name"]).group(1))
    return {
        "sample_id": record["sample_id"],
        "sample_key": record["sample_key"],
        "module_index": record["module_index"],
        "module": record["name"],
        "stage": module_stage,
        "payload": {
            "relative_path": record["relative_path"],
            "sha256": record["file_sha256"],
            "packed_bytes": record["packed_bytes"],
        },
        "shape": {
            "T": t_dim,
            "B": b_dim,
            "H": h_dim,
            "W": w_dim,
            "M": m_dim,
            "K": k_dim,
            "N": n_dim,
        },
        "tiles": {
            "M": math.ceil(m_dim / M_TILE),
            "K": math.ceil(k_dim / K_TILE),
            "N": math.ceil(n_dim / N_TILE),
            "M_padding_rows": math.ceil(m_dim / M_TILE) * M_TILE - m_dim,
            "K_padding_channels": math.ceil(k_dim / K_TILE) * K_TILE - k_dim,
            "N_padding_channels": math.ceil(n_dim / N_TILE) * N_TILE - n_dim,
            "official_partial_tile_policy": "charge cur tile dimensions; no synthetic active padding",
        },
        "input_nnz": observed_nnz,
        "input_elements": record["input_elements"],
        "input_density": observed_nnz / record["input_elements"],
        "modes": modes,
        "product_vs_bit_speedup": bit["total_cycles"] / max(1, product_result["total_cycles"]),
        "product_support_reduction": 1.0
        - product_result["support_nnz"] / max(1, bit["support_nnz"]),
        "direct_validation": direct_checks,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}

    def add(key: str, row: dict[str, Any]) -> None:
        bucket = buckets.setdefault(
            key,
            {
                "records": 0,
                "input_nnz": 0,
                "input_elements": 0,
                "bit": defaultdict(int),
                "product": defaultdict(int),
                "speedups": [],
            },
        )
        bucket["records"] += 1
        bucket["input_nnz"] += row["input_nnz"]
        bucket["input_elements"] += row["input_elements"]
        bucket["speedups"].append(row["product_vs_bit_speedup"])
        for mode in ("bit", "product"):
            counters = row["modes"][mode]["expanded_full_n"]
            for field in COUNTER_FIELDS + ("dram_bits", "global_buffer_bits", "support_nnz"):
                bucket[mode][field] += counters[field]

    for row in records:
        add("overall", row)
        add(f"sample:{row['sample_id']:02d}", row)
        add(f"module:{row['module_index']:02d}", row)

    result: dict[str, Any] = {}
    for key, bucket in buckets.items():
        speedups = bucket.pop("speedups")
        bit = dict(bucket.pop("bit"))
        product_result = dict(bucket.pop("product"))
        result[key] = {
            **bucket,
            "input_density": bucket["input_nnz"] / max(1, bucket["input_elements"]),
            "bit": bit,
            "product": product_result,
            "aggregate_cycle_ratio_speedup": bit["total_cycles"]
            / max(1, product_result["total_cycles"]),
            "per_record_speedup_distribution": {
                "geometric_mean": math.exp(sum(math.log(x) for x in speedups) / len(speedups)),
                "minimum": min(speedups),
                "maximum": max(speedups),
                "arithmetic_mean": sum(speedups) / len(speedups),
            },
            "product_support_reduction": 1.0
            - product_result["support_nnz"] / max(1, bit["support_nnz"]),
        }
    return result


def write_markdown(report: dict[str, Any], path: Path) -> None:
    overall = report["aggregates"]["overall"]
    dist = overall["per_record_speedup_distribution"]
    lines = [
        "# M618 H67 ep35 binary FC1 × 官方 Prosperity CPU 重放（开发结果）\n\n",
        "## 结论\n\n",
        f"状态：`{report['status']}`。在冻结的 100 组 exact-binary FC1 输入上，",
        "官方 Prosperity product-sparsity 相对其同配置 bit-sparsity 的聚合周期比为 ",
        f"**{overall['aggregate_cycle_ratio_speedup']:.6f}×**；逐记录倍率 geomean/min/max 为 ",
        f"**{dist['geometric_mean']:.6f}× / {dist['minimum']:.6f}× / {dist['maximum']:.6f}×**。",
        "这是 external official-artifact 结果，不是本项目 RTL、全网或系统倍速。\n\n",
        "| FC1 module | stage | density | bit cycles | product cycles | cycle ratio | support reduction |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for index, name in enumerate(report["module_order"]):
        item = report["aggregates"][f"module:{report['module_indices'][index]:02d}"]
        stage = next(row["stage"] for row in report["records"] if row["module"] == name)
        lines.append(
            f"| `{name}` | {stage} | {item['input_density']:.6f} | "
            f"{item['bit']['total_cycles']:,} | {item['product']['total_cycles']:,} | "
            f"{item['aggregate_cycle_ratio_speedup']:.6f}× | "
            f"{100*item['product_support_reduction']:.3f}% |\n"
        )
    lines.append(
        f"| **overall** | — | {overall['input_density']:.6f} | "
        f"**{overall['bit']['total_cycles']:,}** | **{overall['product']['total_cycles']:,}** | "
        f"**{overall['aggregate_cycle_ratio_speedup']:.6f}×** | "
        f"**{100*overall['product_support_reduction']:.3f}%** |\n"
    )
    lines.extend(
        [
            "\n## 映射与边界\n\n",
            "- 输入按冻结 `[T,B,H,W,C]` C-order little-bit 解包；官方 `run_fc` 再执行 "
            "`[T,BHW,K] -> [BHW,T,K]`，所以有效 M 行为 `b,h,w,t`，K 保持输入通道顺序。\n",
            "- K=16、M=256、N=128；K/N 均整 tile，M 尾 tile 按官方 `cur_tile_size_M` 收费，"
            "没有补造激活。\n",
            "- stage-3 FC1 输入非二值，未进入 M51 exact-binary 集合，故明确排除。\n",
            "- 真实权重只用于 M51 SHA 身份及 K/N shape 核对；官方 product/bit CPU 路径不读取权重值，"
            "只建模 8-bit 权重流量。\n",
            "- 禁止与 M481 或其他自研周期相除；禁止称 ours、full-network、PPA、energy 或 system speedup。\n",
            "- 本结果在 M619 fresh hammer 前保持 development / headline_admitted=false。\n\n",
            "## 复跑\n\n",
            "```bash\n",
            "PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/pytorch310/bin/python "
            "scripts/run_m618_h67_ep35_binary_fc1_official_prosperity_iso_workload.py --workers 3\n",
            "```\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--skip-direct-validation", action="store_true")
    args = parser.parse_args()
    args.contract = args.contract.resolve()
    args.out = args.out.resolve()
    if args.contract != CONTRACT.resolve():
        raise RuntimeError("M618 accepts only the frozen r1 contract")
    if not 1 <= args.workers <= 3:
        raise RuntimeError("M618 CPU worker count must be in [1,3]")

    contract = strict_json(args.contract)
    checked, all_records = preflight(contract)
    records = all_records
    if args.limit_records is not None:
        if args.limit_records <= 0:
            raise RuntimeError("--limit-records must be positive")
        records = records[: args.limit_records]
    direct_basenames = [] if args.skip_direct_validation else contract["exact_n_tile_expansion"]["direct_validation_records"]

    manifest_path = ROOT / contract["frozen_inputs"]["m51_manifest"]["path"]
    repo_path = contract["frozen_inputs"]["official_prosperity_repo"]["path"]
    completed: list[dict[str, Any]] = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=worker_init,
        initargs=(str(manifest_path.parent), repo_path, direct_basenames),
    ) as executor:
        for count, result in enumerate(
            executor.map(worker_run, records, chunksize=args.chunksize), start=1
        ):
            completed.append(result)
            if count % 5 == 0 or count == len(records):
                print(f"M618 records {count}/{len(records)}", flush=True)

    completed.sort(key=lambda row: (row["sample_id"], row["module_index"]))
    direct_checks = [
        {
            "relative_path": row["payload"]["relative_path"],
            "module": row["module"],
            "sample_id": row["sample_id"],
            **check,
        }
        for row in completed
        for check in row["direct_validation"]
    ]
    mismatch_count = sum(len(check["mismatches"]) for check in direct_checks)
    full_run = len(completed) == contract["population"]["records"]
    if full_run and not args.skip_direct_validation:
        required = contract["exact_n_tile_expansion"]
        expected_checks = len(required["direct_validation_records"]) * required["required_modes_per_record"]
        if len(direct_checks) != expected_checks:
            raise RuntimeError(
                f"direct validation population mismatch: {len(direct_checks)} != {expected_checks}"
            )
        if mismatch_count != required["required_counter_mismatches"]:
            raise RuntimeError(f"exact N-tile expansion failed: {mismatch_count} mismatches")

    aggregates = aggregate(completed)
    status = (
        "PASS_M618_DEV_FULL100_OFFICIAL_PROSPERITY_FC1_NOT_ADMITTED"
        if full_run and not args.skip_direct_validation and mismatch_count == 0
        else "PASS_M618_DEVELOPMENT_SUBSET_NOT_ADMITTED"
    )
    module_rows = sorted(
        {(row["module_index"], row["module"]) for row in completed}
    )
    report = {
        "schema": "m618_h67_ep35_binary_fc1_official_prosperity_iso_workload_v1",
        "date": "2026-08-28",
        "status": status,
        "identity": checked,
        "configuration": contract["official_configuration"],
        "population": {
            **contract["population"],
            "executed_records": len(completed),
            "full_run": full_run,
        },
        "mapping": contract["mapping"],
        "n_tile_expansion": contract["exact_n_tile_expansion"],
        "direct_validation": direct_checks,
        "direct_validation_counter_mismatch_count": mismatch_count,
        "module_indices": [row[0] for row in module_rows],
        "module_order": [row[1] for row in module_rows],
        "aggregates": aggregates,
        "records": completed,
        "claim_boundary": contract["claim_boundary"],
        "claim_boundary_notes": contract["fail_closed"],
    }
    report["payload_sha256"] = canonical_sha(report)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "m618_h67_ep35_binary_fc1_official_prosperity_dev_r1.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path = args.out / "m618_h67_ep35_binary_fc1_official_prosperity_dev_r1.md"
    write_markdown(report, markdown_path)
    receipt = {
        "schema": "m618_h67_ep35_binary_fc1_official_prosperity_dev_receipt_v1",
        "status": status,
        "result": {
            "path": display_path(json_path),
            "sha256": sha256_file(json_path),
        },
        "markdown": {
            "path": display_path(markdown_path),
            "sha256": sha256_file(markdown_path),
        },
        "payload_sha256": report["payload_sha256"],
        "direct_validation_counter_mismatch_count": mismatch_count,
        "overall": aggregates["overall"],
        "claim_boundary": contract["claim_boundary"],
    }
    receipt_path = args.out / "m618_h67_ep35_binary_fc1_official_prosperity_dev_receipt_r1.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json_path)
    print(
        "M618 external official product-vs-bit",
        f"{aggregates['overall']['aggregate_cycle_ratio_speedup']:.6f}x",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
