#!/usr/bin/env python3
"""汇总Local5同步关系SRAM双后端真实post-G0 RTL回放。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SOURCE = ROOT / "sim_new_arch/run_local5_active_projection_postg0_checks.sh"
VECTOR_GENERATOR_SOURCE = (
    ROOT / "scripts/generate_local5_active_projection_postg0_vectors.py"
)


GROUP_RE = re.compile(
    r"GROUP backend=(?P<backend>\d+) "
    r"(?:new1rw=\d+ mode=\d+ )?latency=(?P<latency>\d+) "
    r"group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"active=(?P<active>\d+) avoided=(?P<avoided>\d+) "
    r"memory_wait=(?P<memory_wait>\d+) terms=(?P<terms>\d+) "
    r"updates=(?P<updates>\d+)"
)


def parse_log(path: Path) -> list[dict[str, int]]:
    rows = [
        {name: int(value) for name, value in match.groupdict().items()}
        for match in GROUP_RE.finditer(path.read_text(encoding="utf-8"))
    ]
    if not rows or [row["group"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"{path}的group记录不完整")
    return rows


def summary(values: np.ndarray) -> dict[str, float | int]:
    integer_values = np.issubdtype(values.dtype, np.integer)
    return {
        "total": int(values.sum()) if integer_values else float(values.sum()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()) if integer_values else float(values.max()),
    }


def vector(rows: list[dict[str, int]], field: str) -> np.ndarray:
    return np.asarray([row[field] for row in rows], dtype=np.int64)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tool_version(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout + completed.stderr).strip().splitlines()
    return output[0] if output else ""


def load_bound_inputs(
    vector_dir: Path,
) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    """Fail closed on the selected vectors and their full-profile source trace."""

    vector_manifest_path = (vector_dir / "manifest.json").resolve()
    manifest = json.loads(vector_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "local5_active_projection_postg0_vectors_v1":
        raise ValueError("Local5向量manifest schema不合法")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Local5向量manifest缺少artifacts")
    vector_root = vector_dir.resolve()
    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            raise ValueError(f"向量artifact {name}格式不合法")
        path = (vector_root / str(artifact.get("file", ""))).resolve()
        if path.parent != vector_root or not path.is_file():
            raise ValueError(f"向量artifact {name}路径不合法")
        if file_sha256(path) != artifact.get("sha256"):
            raise ValueError(f"向量artifact {name} SHA失配")

    source_manifest_path = Path(str(manifest.get("source_manifest", ""))).resolve()
    source_payload_path = Path(str(manifest.get("source_payload", ""))).resolve()
    for label, path, expected in (
        ("source manifest", source_manifest_path, manifest.get("source_manifest_sha256")),
        ("source payload", source_payload_path, manifest.get("source_payload_sha256")),
    ):
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"{label}绑定失效")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("schema") != "et3_ordered_term_trace_v2"
        or not source_manifest.get("qualification", {}).get("qualified")
        or source_manifest.get("payload_sha256") != file_sha256(source_payload_path)
    ):
        raise ValueError("来源ordered trace未通过绑定或qualification")

    source_groups = source_manifest.get("groups")
    selected_rows = manifest.get("selection", {}).get("rows")
    if not isinstance(source_groups, list) or not isinstance(selected_rows, list):
        raise ValueError("来源或向量selection缺少group")
    observed: set[int] = set()
    identity_fields = (
        "sample",
        "stage",
        "block",
        "window",
        "head",
        "ordered_item_sha256",
    )
    for vector_index, row in enumerate(selected_rows):
        source_index = row.get("input_group_index")
        if (
            type(source_index) is not int
            or not 0 <= source_index < len(source_groups)
            or source_index in observed
        ):
            raise ValueError(f"向量selection {vector_index}来源索引不合法或重复")
        observed.add(source_index)
        source_row = source_groups[source_index]
        if any(row.get(field) != source_row.get(field) for field in identity_fields):
            raise ValueError(f"向量selection {vector_index}与来源group身份不一致")

    return manifest, source_manifest, source_manifest_path, source_payload_path


def seal_package(
    result_dir: Path,
    *,
    vector_manifest_path: Path,
    vector_manifest: dict[str, object],
    source_manifest_path: Path,
    source_payload_path: Path,
) -> None:
    source_dir = result_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, str] = {}
    for source in (Path(__file__).resolve(), RUNNER_SOURCE, VECTOR_GENERATOR_SOURCE):
        destination = source_dir / source.name
        shutil.copyfile(source, destination)
        snapshots[str(destination.relative_to(result_dir))] = file_sha256(destination)
    receipt_path = result_dir / "source_sha256.txt"
    bound_dir = source_dir / "bound"
    bound_dir.mkdir(exist_ok=True)
    seen_names: set[str] = set()
    for source in verify_source_receipt(receipt_path):
        if source.name in seen_names:
            continue
        seen_names.add(source.name)
        destination = bound_dir / source.name
        shutil.copyfile(source, destination)
        snapshots[str(destination.relative_to(result_dir))] = file_sha256(destination)

    package_files: dict[str, str] = {}
    for path in sorted(result_dir.iterdir()):
        if path.is_file() and path.name != "complete.json":
            package_files[path.name] = file_sha256(path)
    package_files.update(snapshots)
    complete = {
        "schema": "local5_tcfm5_linear5_rtl_package_v1",
        "status": "SEALED",
        "evidence": "[rtl]+[profile-qualified-trace]+[sync-1r1w-contract]",
        "package_files": package_files,
        "vector_manifest": str(vector_manifest_path.resolve()),
        "vector_manifest_sha256": file_sha256(vector_manifest_path),
        "vector_artifacts": vector_manifest["artifacts"],
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_payload": str(source_payload_path),
        "source_payload_sha256": file_sha256(source_payload_path),
        "report_sha256": file_sha256(result_dir / "report.json"),
    }
    (result_dir / "complete.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_source_receipt(receipt_path: Path) -> list[Path]:
    sources: list[Path] = []
    for line in receipt_path.read_text(encoding="utf-8").splitlines():
        expected_digest, raw_path = line.split(maxsplit=1)
        source = Path(raw_path).resolve()
        if file_sha256(source) != expected_digest:
            raise ValueError(f"source receipt SHA失配: {source}")
        sources.append(source)
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(
            "results/local5_active_projection_sync_sram_postg0_rtl_20260803"
        ),
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=Path("tb_qfit/vectors/local5_active_projection_postg0_100"),
    )
    args = parser.parse_args()
    vector_manifest_path = args.vector_dir / "manifest.json"
    (
        manifest,
        source_manifest,
        source_manifest_path,
        source_payload_path,
    ) = load_bound_inputs(args.vector_dir)
    weight_mode = str(manifest.get("weight_mode", "synthetic"))
    projection_binding = manifest.get("projection_contract_binding")
    if weight_mode == "checkpoint_theta_folded_dyadic_int8_head_slice":
        if os.environ.get("CHECKPOINT_WEIGHTS") != "1":
            raise ValueError("checkpoint权重向量必须由CHECKPOINT_WEIGHTS=1运行")
        if not isinstance(projection_binding, dict):
            raise ValueError("theta-folded向量缺少projection contract binding")
        if (
            projection_binding.get("schema")
            != "local5_checkpoint_projection_contract_v2"
            or projection_binding.get("status") != "THETA_FOLDED_WEIGHT_CONTRACT"
        ):
            raise ValueError("theta-folded向量的v2合同语义错误")
        for key in ("manifest", "payload"):
            path = Path(str(projection_binding.get(key, "")))
            if not path.is_file() or file_sha256(path) != projection_binding.get(
                f"{key}_sha256"
            ):
                raise ValueError(f"theta-folded projection {key}绑定失效")
    configurations: dict[str, dict[str, object]] = {}
    row_sets: dict[tuple[str, int], list[dict[str, int]]] = {}
    for latency in (1, 2):
        for backend, backend_id in (("tcfm5", 0), ("linear5", 1)):
            rows = parse_log(
                args.result_dir / f"{backend}_l{latency}_verilator.log"
            )
            if any(
                row["backend"] != backend_id or row["latency"] != latency
                for row in rows
            ):
                raise ValueError("日志backend/latency标签不一致")
            row_sets[(backend, latency)] = rows

    reference = row_sets[("tcfm5", 1)]
    if len(manifest.get("selection", {}).get("rows", [])) != len(reference):
        raise ValueError("向量manifest与RTL日志group数不一致")
    for rows in row_sets.values():
        if len(rows) != len(reference):
            raise ValueError("四配置group数量不一致")
        for left, right in zip(reference, rows, strict=True):
            for field in ("group", "active", "avoided", "terms", "updates"):
                if left[field] != right[field]:
                    raise ValueError(f"四配置{field}不一致")

    active = vector(reference, "active")
    stages = np.asarray(
        [row["stage"] for row in manifest["selection"]["rows"]],
        dtype=np.int64,
    )
    stage_results = []
    for latency in (1, 2):
        tcfm = vector(row_sets[("tcfm5", latency)], "cycles")
        linear = vector(row_sets[("linear5", latency)], "cycles")
        speedup = linear / tcfm
        configurations[f"tcfm5_l{latency}"] = {
            "cycles": summary(tcfm),
            "memory_wait": summary(
                vector(row_sets[("tcfm5", latency)], "memory_wait")
            ),
        }
        configurations[f"linear5_l{latency}"] = {
            "cycles": summary(linear),
            "memory_wait": summary(
                vector(row_sets[("linear5", latency)], "memory_wait")
            ),
        }
        configurations[f"speedup_l{latency}"] = {
            "ratio_of_totals": float(linear.sum() / tcfm.sum()),
            "per_group": summary(speedup),
        }
        for stage in range(4):
            mask = stages == stage
            stage_results.append(
                {
                    "latency": latency,
                    "stage": stage,
                    "groups": int(mask.sum()),
                    "tcfm5_cycles": int(tcfm[mask].sum()),
                    "linear5_cycles": int(linear[mask].sum()),
                    "speedup": float(linear[mask].sum() / tcfm[mask].sum()),
                }
            )

    l1_tcfm = vector(row_sets[("tcfm5", 1)], "cycles")
    l2_tcfm = vector(row_sets[("tcfm5", 2)], "cycles")
    l1_linear = vector(row_sets[("linear5", 1)], "cycles")
    l2_linear = vector(row_sets[("linear5", 2)], "cycles")
    actual_acc32: dict[str, dict[str, object]] = {}
    for latency in (1, 2):
        for backend in ("tcfm5", "linear5"):
            path = args.result_dir / f"{backend}_l{latency}_actual_acc32.memh"
            if not path.is_file():
                raise ValueError(f"缺少actual Acc32归档: {path.name}")
            lines = sum(1 for _ in path.open("r", encoding="ascii"))
            expected_lines = len(reference) * int(manifest["shape"]["sources"]) * int(
                manifest["shape"]["out_dim"]
            )
            if lines != expected_lines:
                raise ValueError(f"{path.name}的Acc32数量不守恒")
            actual_acc32[f"{backend}_l{latency}"] = {
                "file": path.name,
                "values": lines,
                "sha256": file_sha256(path),
            }
    if len({row["sha256"] for row in actual_acc32.values()}) != 1:
        raise ValueError("TCFM5/Linear5或L1/L2 actual Acc32不一致")

    stress_logs: dict[str, dict[str, object]] = {}
    for latency in (1, 2):
        for backend in ("tcfm5", "linear5"):
            stress_log = (
                args.result_dir
                / f"{backend}_l{latency}_random_stress_verilator.log"
            )
            if (
                not stress_log.is_file()
                or "PASS post-G0 active projection"
                not in stress_log.read_text(encoding="utf-8")
            ):
                raise ValueError(
                    f"{backend}-L{latency}随机输入/读回gap压力回归未通过"
                )
            stress_logs[f"{backend}_l{latency}"] = {
                "backend": backend,
                "relation_read_latency": latency,
                "random_input_gaps": 1,
                "random_read_gaps": 1,
                "headline_cycles": False,
                "log": stress_log.name,
                "log_sha256": file_sha256(stress_log),
            }
    represented_samples = len(
        {int(row["sample"]) for row in manifest["selection"]["rows"]}
    )
    population_stage_weighted = "population-stage-weighted" in str(
        manifest["selection"].get("method")
    )
    result = {
        "schema": "local5_active_projection_sync_sram_postg0_rtl_v2",
        "status": "RTL_REPLAY_COMPLETE",
        "evidence": "[rtl]+[profile-qualified-trace]+[sync-1r1w-contract]",
        "scope": "post-score relation write -> synchronous relation banks -> active source -> term builder -> Acc",
        "groups": len(reference),
        "source_cohort": {
            "groups": len(source_manifest["groups"]),
            "processed_samples": source_manifest["qualification"]["processed_samples"],
            "checkpoint": source_manifest.get("checkpoint"),
            "checkpoint_sha256": source_manifest.get("checkpoint_sha256"),
            "config": source_manifest.get("config"),
            "config_sha256": source_manifest.get("config_sha256"),
            "manifest": str(source_manifest_path),
            "manifest_sha256": file_sha256(source_manifest_path),
            "payload": str(source_payload_path),
            "payload_sha256": file_sha256(source_payload_path),
        },
        "selection": {
            "method": manifest["selection"].get("method"),
            "generator_per_stage_parameter": manifest["selection"].get("per_stage"),
            "stage_counts": {
                str(stage): int((stages == stage).sum()) for stage in range(4)
            },
            "represented_samples": represented_samples,
            "population_stage_weighted": population_stage_weighted,
        },
        "vector_manifest": str(vector_manifest_path.resolve()),
        "vector_manifest_sha256": file_sha256(vector_manifest_path),
        "weight_mode": weight_mode,
        "weight_contract": manifest.get("weight_contract"),
        "projection_contract_binding": projection_binding,
        "actual_acc32": actual_acc32,
        "execution_receipt": {
            "checkpoint_weights_env": os.environ.get("CHECKPOINT_WEIGHTS"),
            "vector_pregenerated_env": os.environ.get("VECTOR_PREGENERATED"),
            "random_stress": stress_logs,
            "python": sys.version.splitlines()[0],
            "python_executable": str(Path(sys.executable).resolve()),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "iverilog": tool_version(["iverilog", "-V"]),
            "verilator": tool_version(["verilator", "--version"]),
            "yosys": tool_version(["yosys", "-V"]),
        },
        "memory_contract": {
            "banks": "1xK + 5x(direction gate,valid)",
            "ports": "logical write/read channels, build/read phases exclusive; target 1RW macro",
            "read_latency_sweep": [1, 2],
            "descriptor_assembly": "one registered output buffer; frontier-visible latency is L+1",
        },
        "configurations": configurations,
        "latency_penalty": {
            "tcfm5_l2_over_l1": float(l2_tcfm.sum() / l1_tcfm.sum()),
            "linear5_l2_over_l1": float(l2_linear.sum() / l1_linear.sum()),
        },
        "active_sources": summary(active),
        "zero_active_groups": int((active == 0).sum()),
        "stage_results": stage_results,
        "fairness": [
            "相同同步关系SRAM、frontier、term builder、权重、五个单写Acc bank和读回合同。",
            "公平对照合同仅改变destination到Acc bank的映射与冲突replay策略；两后端是独立RTL实现。",
            "四配置均逐值校验100x450x2个32-bit Acc输出。",
            "四配置actual Acc32逐项一致并归档；四配置均跑随机输入/读回gap压力。",
        ],
        "limits": [
            "同步1R1W为可综合端口合同，尚未绑定具体foundry SRAM macro和Liberty/LEF。",
            f"{len(reference)}组覆盖{represented_samples}个sample；stage配额按{len(source_manifest['groups']):,}个来源group比例确定，但每sample仅一组，仍不是完整总体。",
            "无目标工艺频率、面积、功耗和EDP结论。",
            "slice从post-score开始，不含Local5 score/Shiftmax与full encoder。",
            "内部formal_g0仍为DENY；本包是trace-driven assertion RTL，不是全输入formal proof。",
        ],
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Local5同步关系SRAM真实Post-G0 RTL回放",
        "",
        "## 结论",
        "",
        f"在`{len(reference)}`个qualified T450 group上，1-cycle同步关系SRAM下",
        f"TCFM5为`{int(l1_tcfm.sum()):,}`周期，Linear5为",
        f"`{int(l1_linear.sum()):,}`周期，加速",
        f"`{l1_linear.sum() / l1_tcfm.sum():.3f}x`；2-cycle SRAM下分别为",
        f"`{int(l2_tcfm.sum()):,}`与`{int(l2_linear.sum()):,}`，加速",
        f"`{l2_linear.sum() / l2_tcfm.sum():.3f}x`。",
        "",
        f"四配置每个均通过`{len(reference) * 450 * 2:,}`个Acc32比较，合计",
        f"`{len(reference) * 450 * 2 * 4:,}`个。证据等级为",
        "**[rtl]+[profile-qualified-trace]+[sync-1r1w-contract]**。",
        "",
        *(
            [
                "本次向量绑定 `local5_checkpoint_projection_contract_v2`：",
                "`theta_K` 先折叠到 projection 权重再做 dyadic INT8 量化，",
                "运行时 K 仍是 1-bit event，不增加 theta 乘法器。",
                "",
            ]
            if weight_mode
            == "checkpoint_theta_folded_dyadic_int8_head_slice"
            else []
        ),
        "## 周期分布",
        "",
        "| 配置 | total | mean | p50 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, values in (
        ("TCFM5-L1", l1_tcfm),
        ("Linear5-L1", l1_linear),
        ("TCFM5-L2", l2_tcfm),
        ("Linear5-L2", l2_linear),
    ):
        row = summary(values)
        lines.append(
            f"| {label} | {row['total']:,} | {row['mean']:.2f} | "
            f"{row['p50']:.1f} | {row['p95']:.1f} | {row['p99']:.1f} | "
            f"{row['max']:,} |"
        )
    lines += [
        "",
        f"关系SRAM从1拍变2拍后，TCFM5总周期增加",
        f"`{l2_tcfm.sum() / l1_tcfm.sum() - 1:.2%}`，Linear5增加",
        f"`{l2_linear.sum() / l1_linear.sum() - 1:.2%}`。",
        "",
        "| SRAM延迟 | aggregate加速 | 逐group mean | p50 | p95 | p99 |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| 1 | {l1_linear.sum() / l1_tcfm.sum():.3f}x | "
        f"{np.mean(l1_linear / l1_tcfm):.3f}x | "
        f"{np.percentile(l1_linear / l1_tcfm, 50):.3f}x | "
        f"{np.percentile(l1_linear / l1_tcfm, 95):.3f}x | "
        f"{np.percentile(l1_linear / l1_tcfm, 99):.3f}x |",
        f"| 2 | {l2_linear.sum() / l2_tcfm.sum():.3f}x | "
        f"{np.mean(l2_linear / l2_tcfm):.3f}x | "
        f"{np.percentile(l2_linear / l2_tcfm, 50):.3f}x | "
        f"{np.percentile(l2_linear / l2_tcfm, 95):.3f}x | "
        f"{np.percentile(l2_linear / l2_tcfm, 99):.3f}x |",
        "",
        "## Stage分账",
        "",
        "| SRAM延迟 | Stage | group | TCFM5 | Linear5 | 加速 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage_results:
        lines.append(
            f"| {row['latency']} | S{row['stage']} | {row['groups']} | "
            f"{row['tcfm5_cycles']:,} | {row['linear5_cycles']:,} | "
            f"{row['speedup']:.3f}x |"
        )
    lines += [
        "",
        "## 存储与公平性",
        "",
        "- 关系存储为一个K bank和五个方向gate/valid bank；每个bank有逻辑写/读通道且相序互斥，目标映射单端口1RW宏。",
        "- build与read相序互斥；五个方向可并行读不同地址，descriptor经一个保持buffer承受下游反压。",
        "- SRAM读延迟为L拍，六路响应再进入一拍descriptor保持寄存器，因此前沿可见延迟为L+1拍。",
        "- 公平对照合同仅改变Acc bank映射/replay策略；两后端是独立RTL实现。",
        "- TCFM5/Linear5与L1/L2四配置均通过随机输入gap与随机读回gap的Verilator+SVA压力回归；压力周期不进入headline。",
        "",
        "## 证据边界",
        "",
        "- 这是可综合同步memory合同，不是已绑定foundry SRAM macro的PPA。",
        f"- {len(reference)}组覆盖{represented_samples}个sample，stage配额按{len(source_manifest['groups']):,}个来源group比例确定；每sample仅一组，不是完整总体或完整帧全部window。",
        "- 本slice仍不含Local5 score/Shiftmax、full encoder和目标工艺功耗。",
        "- 内部formal_g0仍为DENY；本包不构成全输入formal proof。",
    ]
    (args.result_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    seal_package(
        args.result_dir,
        vector_manifest_path=vector_manifest_path,
        vector_manifest=manifest,
        source_manifest_path=source_manifest_path,
        source_payload_path=source_payload_path,
    )
    print(
        json.dumps(
            {
                "speedup_l1": configurations["speedup_l1"]["ratio_of_totals"],
                "speedup_l2": configurations["speedup_l2"]["ratio_of_totals"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
