#!/usr/bin/env python3
"""Motion/Local5 双线分层稀疏性能模型。

模型边界：
1. 只导入校验 Prosperity 官方开源仓库的 Stats 字段；正式
   compute/memory/preprocess 周期由本地兼容模型计算，不调用官方 Simulator。
2. Phi 未公开官方模拟器；这里按论文的 L1 pattern/PWP + L2 residual
   机制建立显式命中率扫描，结果只能标为 [模型]。
3. Motion 的 T=162 与 T=450 分别读取各自 100 个 DSEC 样本的 ordered profile。
4. Local5 T=450 在联合 fullres profile 完成前仍按 T=162 几何比例外推。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import struct
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
PROSPERITY_SIM = ROOT / "third_party" / "Prosperity" / "simulator"
DEFAULT_OUT = ROOT / "results" / "phi_prosperity_dual_line_sim_20260729"
LOCAL_PROFILE = (
    ROOT
    / "results"
    / "local5_hardware_profile_preG0_profile100_20260726"
    / "local5_hardware_features.json"
)
MOTION_PROFILE_T162 = (
    ROOT.parent
    / "neuron_experiments"
    / "H9_bipolar_self_attention"
    / "results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
MOTION_PROFILE_T450 = (
    ROOT
    / "results"
    / "h67_fullres_ep30_t450_profile100_20260805"
    / "nts11_hardware_p0_profile.json"
)

_OFFICIAL_STATS = None
if PROSPERITY_SIM.is_dir():
    try:
        spec = importlib.util.spec_from_file_location(
            "_prosperity_official_utils",
            PROSPERITY_SIM / "utils.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError("无法创建 Prosperity utils 模块规格")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _OFFICIAL_STATS = module.Stats
    except Exception:
        _OFFICIAL_STATS = None


def ceil_div(a: float, b: float) -> int:
    if b <= 0:
        raise ValueError("除数必须为正数")
    return int(math.ceil(float(a) / float(b)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def percentile(values: Iterable[float], q: float) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def distribution(values: Iterable[float]) -> dict:
    xs = [float(v) for v in values]
    if not xs:
        return {k: 0.0 for k in ("mean", "p50", "p95", "p99", "max", "cv")}
    mean = statistics.fmean(xs)
    return {
        "mean": mean,
        "p50": percentile(xs, 0.50),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
        "max": max(xs),
        "cv": statistics.pstdev(xs) / mean if mean else 0.0,
    }


@dataclass(frozen=True)
class ArchConfig:
    score_lanes: int = 32
    residual_lanes: int = 4
    matcher_lanes: int = 32
    metadata_issue_width: int = 1
    term_issue_width: int = 1
    projection_banks: int = 3
    weight_bank_bits_per_cycle: int = 256
    output_lanes: int = 96
    weight_bits: int = 8
    activation_sram_bits_per_cycle: int = 128
    descriptor_bits: int = 64
    compact_source_bits: int = 24
    compact_header_bits: int = 40
    compact_continuation_flags: int = 4
    compact_delta_continuation_bits: int = 6
    pattern_index_bits: int = 5
    pattern_table_entries: int = 16
    pwp_bits_per_pattern: int = 96 * 16
    ttb_tokens: int = 4
    freq_mhz: float = 500.0


@dataclass
class SampleWorkload:
    line: str
    sample_id: int
    tokens_per_window: int
    vector_count: int
    pattern_vectors: int
    direct_score_lane_work: int
    anchor_score_lane_work: int
    residual_lane_work: int
    online_match_lane_work: int
    direct_k_bits: int
    anchor_k_bits: int
    direct_projection_products: int
    exact_projection_terms: int
    exact_destination_count: int
    packed_delivery_commands: int
    term_scan_entries: int
    static_score_cycles_w2: int
    static_score_cycles_w4: int
    static_score_cycles_w8: int
    bundle_total: int
    bundle_empty: int
    profile_source: str
    evidence: str

    def __post_init__(self) -> None:
        if self.exact_projection_terms > self.exact_destination_count:
            raise ValueError("unique projection term不能超过destination数量")
        if self.packed_delivery_commands > self.exact_destination_count:
            raise ValueError("packed command不能超过标量destination数量")
        if self.packed_delivery_commands < self.exact_projection_terms:
            raise ValueError("每个非空term至少需要一条packed command")


@dataclass
class Component:
    preprocess_cycles: int = 0
    compute_cycles: int = 0
    memory_cycles: int = 0
    payload_bits: int = 0
    metadata_bits: int = 0
    fabric_bits: int = 0

    @property
    def total_cycles(self) -> int:
        return self.preprocess_cycles + max(self.compute_cycles, self.memory_cycles)


def _sum_fields(records: list[dict], names: Iterable[str]) -> dict[str, int]:
    return {name: int(sum(int(r.get(name, 0)) for r in records)) for name in names}


def decode_count_trace(encoded: dict) -> list[int]:
    formats = {
        "int16_le": ("h", 2),
        "int32_le": ("i", 4),
    }
    dtype = encoded.get("dtype")
    if dtype not in formats or encoded.get("codec") != "zlib_base64":
        raise ValueError("不支持的 ordered trace 编码")
    raw = zlib.decompress(base64.b64decode(encoded["data"]))
    code, item_bytes = formats[dtype]
    if len(raw) % item_bytes:
        raise ValueError("ordered trace payload字节未对齐")
    count = len(raw) // item_bytes
    values = list(struct.unpack(f"<{count}{code}", raw))
    expected = math.prod(int(v) for v in encoded["shape"])
    if count != expected:
        raise ValueError("ordered trace shape 与 payload 不一致")
    return values


def ordered_decoupled_cycles(delta_counts: Iterable[int], width: int) -> int:
    backlog = 0
    count = 0
    for delta in delta_counts:
        service = ceil_div(delta, width) if delta > 0 else 0
        backlog = max(0, backlog + service - 1)
        count += 1
    return count + backlog


def local_hist_service(records: list[dict], width: int) -> int:
    service = 0
    for record in records:
        for direction in ("up", "down", "left", "right"):
            histogram = record[f"{direction}_delta_histogram"]
            service += sum(
                int(freq) * (ceil_div(delta, width) if delta else 0)
                for delta, freq in enumerate(histogram)
            )
    return service


def load_local_samples() -> list[SampleWorkload]:
    data = json.loads(LOCAL_PROFILE.read_text())
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in data["records"]:
        grouped[int(record["sample_id"])].append(record)

    out = []
    fields = (
        "token_heads",
        "valid_edges",
        "directional_valid_edges",
        "directional_delta_lane_sum",
        "query_major_k_lane_reads",
        "source_resident_k_lane_reads",
        "naive_active_edge_products",
        "destination_gate_lane_groups",
        "mfep_multicast_terms",
        "valid_gate_entries",
        "batch_windows",
    )
    for sample_id, records in sorted(grouped.items()):
        s = _sum_fields(records, fields)
        static_cycles = {
            width: max(
                s["token_heads"],
                local_hist_service(records, width),
            )
            for width in (2, 4, 8)
        }
        out.append(
            SampleWorkload(
                line="Local5",
                sample_id=sample_id,
                tokens_per_window=162,
                vector_count=s["token_heads"],
                pattern_vectors=s["valid_edges"],
                direct_score_lane_work=s["valid_edges"] * 32,
                anchor_score_lane_work=s["token_heads"] * 32,
                residual_lane_work=s["directional_delta_lane_sum"],
                online_match_lane_work=s["directional_valid_edges"] * 32,
                direct_k_bits=s["query_major_k_lane_reads"],
                anchor_k_bits=s["source_resident_k_lane_reads"]
                + s["directional_delta_lane_sum"],
                direct_projection_products=s["naive_active_edge_products"],
                exact_projection_terms=s["mfep_multicast_terms"],
                exact_destination_count=s["destination_gate_lane_groups"],
                # pre-G0 Local5 尚无物理端口打包证据，保守按一目的/命令。
                packed_delivery_commands=s["destination_gate_lane_groups"],
                term_scan_entries=s["valid_gate_entries"],
                static_score_cycles_w2=static_cycles[2],
                static_score_cycles_w4=static_cycles[4],
                static_score_cycles_w8=static_cycles[8],
                # Local5 尚无 ordered STT-empty profile，不能凭空跳过 bundle。
                bundle_total=ceil_div(s["token_heads"], 4),
                bundle_empty=0,
                profile_source=str(LOCAL_PROFILE.relative_to(ROOT)),
                evidence="[prof-preG0]",
            )
        )
    return out


def resolve_motion_temporal_tokens(protocol: dict, records: list[dict]) -> int:
    record_tokens = {int(record.get("tokens", -1)) for record in records}
    if len(record_tokens) != 1 or next(iter(record_tokens), -1) <= 0:
        raise ValueError("Motion h60_records的T不唯一或非法")
    record_temporal_tokens = next(iter(record_tokens))
    protocol_temporal_tokens = int(protocol.get("tokens_per_window", -1))
    if protocol_temporal_tokens > 0 and protocol_temporal_tokens != record_temporal_tokens:
        raise ValueError("Motion profile的T与h60_records不一致")
    return record_temporal_tokens


def load_motion_samples(profile_path: Path) -> list[SampleWorkload]:
    data = json.loads(profile_path.read_text())
    records = data["summary"]["h60_records"]
    protocol = data.get("eval_protocol") or {}
    temporal_tokens = resolve_motion_temporal_tokens(protocol, records)
    samples = int(data.get("samples", -1))
    if samples <= 0:
        raise ValueError("Motion profile samples必须为正数")
    sample_ids = {int(record.get("sample_id", -1)) for record in records}
    if len(sample_ids) != samples or sample_ids != set(range(samples)):
        raise ValueError("Motion profile sample_id覆盖与samples不一致")
    if data.get("ordered_trace") is not True:
        raise ValueError("Motion profile不是ordered trace")
    artifact = data.get("artifact_identity") or {}
    audit = data.get("checkpoint_load_audit") or {}
    fullres_identity = (
        protocol.get("resolution") == [480, 640]
        and protocol.get("crop") is None
        and protocol.get("window_size") == [2, 15, 15]
        and temporal_tokens == 450
        and samples == 100
        and len(records) == 1200
        and protocol.get("bn_policy") == "no_running"
        and data.get("module_counts")
        == {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12}
        and int(audit.get("checkpoint_overlay_keys", -1)) == 210
        and int(audit.get("model_overlay_keys", -1)) == 210
        and int(audit.get("missing_count", -1)) == 0
        and int(audit.get("unexpected_count", -1)) == 0
        and re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("config_sha256", "")))
        is not None
        and re.fullmatch(
            r"[0-9a-f]{64}", str(artifact.get("checkpoint_sha256", ""))
        )
        is not None
    )
    evidence = "[prof-ordered-fullres]" if fullres_identity else "[prof-ordered]"
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["sample_id"])].append(record)

    out = []
    fields = (
        "pair_total",
        "pair_empty",
        "qk_temporal_update_elements",
        "k_temporal_baseline_reads",
        "k_temporal_union_reads",
        "projection_baseline_active_lanes",
        "projection_gate_class_channel_terms_deploy",
        "projection_gate_multicast_delivery_m1",
        "projection_gate_multicast_delivery_m4",
        "row_total",
        "ttb_tok4_total",
        "ttb_tok4_empty",
    )
    for sample_id, sample_records in sorted(grouped.items()):
        s = _sum_fields(sample_records, fields)
        static_cycles = {}
        for width in (2, 4, 8):
            static_cycles[width] = sum(
                ordered_decoupled_cycles(
                    decode_count_trace(record["delta_update_ordered_trace"]),
                    width,
                )
                for record in sample_records
            )
        out.append(
            SampleWorkload(
                line="Motion",
                sample_id=sample_id,
                tokens_per_window=temporal_tokens,
                vector_count=s["pair_total"],
                pattern_vectors=s["pair_total"],
                direct_score_lane_work=s["pair_total"] * 64,
                anchor_score_lane_work=s["pair_total"] * 32,
                residual_lane_work=s["qk_temporal_update_elements"],
                online_match_lane_work=s["pair_total"] * 64,
                direct_k_bits=s["k_temporal_baseline_reads"],
                anchor_k_bits=s["k_temporal_union_reads"],
                direct_projection_products=s["projection_baseline_active_lanes"],
                exact_projection_terms=s["projection_gate_class_channel_terms_deploy"],
                exact_destination_count=s["projection_gate_multicast_delivery_m1"],
                packed_delivery_commands=s["projection_gate_multicast_delivery_m4"],
                term_scan_entries=s["row_total"] * 35,
                static_score_cycles_w2=static_cycles[2],
                static_score_cycles_w4=static_cycles[4],
                static_score_cycles_w8=static_cycles[8],
                bundle_total=s["ttb_tok4_total"],
                bundle_empty=s["ttb_tok4_empty"],
                profile_source=str(profile_path.resolve()),
                evidence=evidence,
            )
        )
    return out


def scale_workload(w: SampleWorkload, target_tokens: int) -> SampleWorkload:
    """按窗口面积比例外推 T=162 -> T=450。

    480x640/window15 与 288x384/window9 的窗口数近似相同，窗口内 token
    数按 450/162 增长。非线性的 gate cardinality 和边界效应留待实测修正。
    """

    factor = target_tokens / w.tokens_per_window

    def sc(v: int) -> int:
        return int(round(v * factor))

    return SampleWorkload(
        **{
            **asdict(w),
            "tokens_per_window": target_tokens,
            "vector_count": sc(w.vector_count),
            "pattern_vectors": sc(w.pattern_vectors),
            "direct_score_lane_work": sc(w.direct_score_lane_work),
            "anchor_score_lane_work": sc(w.anchor_score_lane_work),
            "residual_lane_work": sc(w.residual_lane_work),
            "online_match_lane_work": sc(w.online_match_lane_work),
            "direct_k_bits": sc(w.direct_k_bits),
            "anchor_k_bits": sc(w.anchor_k_bits),
            "direct_projection_products": sc(w.direct_projection_products),
            "exact_projection_terms": sc(w.exact_projection_terms),
            "exact_destination_count": sc(w.exact_destination_count),
            "packed_delivery_commands": sc(w.packed_delivery_commands),
            "term_scan_entries": sc(w.term_scan_entries),
            "static_score_cycles_w2": sc(w.static_score_cycles_w2),
            "static_score_cycles_w4": sc(w.static_score_cycles_w4),
            "static_score_cycles_w8": sc(w.static_score_cycles_w8),
            "bundle_total": sc(w.bundle_total),
            "bundle_empty": sc(w.bundle_empty),
            "evidence": "[外推-25/9]",
        }
    )


def workloads_at_tokens(
    samples: list[SampleWorkload],
    target_tokens: int,
) -> list[SampleWorkload]:
    """保留实测 T=162 证据标签，仅对其他规模执行外推。"""

    if not samples:
        return []
    if target_tokens == samples[0].tokens_per_window:
        return samples
    return [scale_workload(w, target_tokens) for w in samples]


def make_official_stats_probe() -> dict:
    if _OFFICIAL_STATS is None:
        return {
            "imported": False,
            "reason": "当前 Python 环境无法导入 Prosperity Stats；模型使用兼容字段",
        }
    stats = _OFFICIAL_STATS()
    stats.compute_cycles = 7
    stats.mem_stall_cycles = 3
    stats.preprocess_stall_cycles = 2
    return {
        "imported": True,
        "class": f"{stats.__class__.__module__}.{stats.__class__.__name__}",
        "field_probe": {
            "compute_cycles": stats.compute_cycles,
            "mem_stall_cycles": stats.mem_stall_cycles,
            "preprocess_stall_cycles": stats.preprocess_stall_cycles,
            "mem_namespace": list(stats.mem_namespace),
        },
    }


def _score_component(
    w: SampleWorkload,
    cfg: ArchConfig,
    scheme: str,
    phi_hit_rate: float,
) -> Component:
    if scheme == "direct":
        lane_work = w.direct_score_lane_work
        preprocess = 0
        k_bits = w.direct_k_bits
        metadata = 0
    elif scheme == "prosperity_online":
        # 给 Prosperity 风格 baseline 最有利的 exact/partial residual compute，
        # 但保留在线 relation matcher 的比较与预处理成本。
        lane_work = (
            w.anchor_score_lane_work
            + w.residual_lane_work
            + w.online_match_lane_work
        )
        preprocess = ceil_div(w.pattern_vectors, cfg.matcher_lanes)
        # matcher 与 score 共享同一 K 读取，避免对基线重复计 SRAM 流量。
        k_bits = w.direct_k_bits
        metadata = w.pattern_vectors * cfg.pattern_index_bits
    elif scheme == "phi_pattern_residual":
        miss = 1.0 - phi_hit_rate
        # 未命中必须回退完整 direct；命中才由 PWP 加 residual 修正。
        lane_work = int(
            round(
                w.direct_score_lane_work * miss
                + w.residual_lane_work * phi_hit_rate
            )
        )
        preprocess = ceil_div(w.pattern_vectors, cfg.matcher_lanes)
        # L1 命中读取 PWP，L2 读取 residual；miss 读取完整 K。
        pwp_reads = ceil_div(w.pattern_vectors * phi_hit_rate, cfg.score_lanes)
        k_bits = (
            int(round(w.direct_k_bits * miss))
            + int(round(w.residual_lane_work * phi_hit_rate))
            + pwp_reads * cfg.pwp_bits_per_pattern
        )
        metadata = w.pattern_vectors * cfg.pattern_index_bits
    elif scheme in ("static_anchor_term", "static_anchor_hierdesc"):
        lane_work = w.anchor_score_lane_work + w.residual_lane_work
        preprocess = 0
        k_bits = w.anchor_k_bits
        active_bundles = w.bundle_total - w.bundle_empty
        metadata = max(0, active_bundles) * cfg.descriptor_bits
    else:
        raise ValueError(f"未知方案: {scheme}")

    # 静态锚点方案按真实 32-lane anchor + W-lane residual 结构建模。
    # Motion 使用 ordered backlog；Local5 使用逐向量 histogram 服务下界。
    if scheme in ("static_anchor_term", "static_anchor_hierdesc"):
        cycle_field = f"static_score_cycles_w{cfg.residual_lanes}"
        if not hasattr(w, cycle_field):
            raise ValueError(
                f"residual_lanes={cfg.residual_lanes} 缺少 ordered/hist 周期字段"
            )
        compute = int(getattr(w, cycle_field))
    else:
        compute = ceil_div(lane_work, cfg.score_lanes)
    memory = ceil_div(k_bits, cfg.activation_sram_bits_per_cycle)
    return Component(
        preprocess_cycles=preprocess,
        compute_cycles=compute,
        memory_cycles=memory,
        payload_bits=k_bits,
        metadata_bits=metadata,
        fabric_bits=(
            metadata
            if scheme in ("prosperity_online", "phi_pattern_residual")
            else 0
        ),
    )


def _term_component(w: SampleWorkload, cfg: ArchConfig, scheme: str) -> Component:
    if scheme not in ("static_anchor_term", "static_anchor_hierdesc"):
        return Component()
    scan = ceil_div(w.term_scan_entries, cfg.score_lanes)
    source_beats = w.exact_destination_count
    issue = ceil_div(source_beats, cfg.term_issue_width)
    if scheme == "static_anchor_term":
        metadata = source_beats * cfg.descriptor_bits
    else:
        # Builder 与 directory 共驻留；入口只保留 key/destination 紧凑字段。
        metadata = source_beats * cfg.compact_source_bits
    return Component(
        preprocess_cycles=scan,
        compute_cycles=issue,
        memory_cycles=ceil_div(metadata, cfg.activation_sram_bits_per_cycle),
        metadata_bits=metadata,
        # builder 与 directory 共驻留，source metadata 不进入全局 fabric。
        fabric_bits=0,
    )


def _projection_component(
    w: SampleWorkload,
    cfg: ArchConfig,
    scheme: str,
    phi_hit_rate: float,
) -> Component:
    direct = w.direct_projection_products
    if scheme == "direct":
        weight_fetches = direct
        delivery_commands = direct
        metadata = 0
    elif scheme == "prosperity_online":
        # Oracle-friendly：允许在线 matcher 达到与 exact term 流相同的
        # product reuse；差异只剩在线关系发现和描述符成本。
        weight_fetches = w.exact_projection_terms
        delivery_commands = w.exact_destination_count
        metadata = w.exact_destination_count * cfg.pattern_index_bits
    elif scheme == "phi_pattern_residual":
        miss = 1.0 - phi_hit_rate
        # 一次 PWP 供 32 个 lane 共享；miss 完整执行，hit 再按实测
        # score residual density 支付 L2 修正。
        residual_ratio = min(
            1.0,
            w.residual_lane_work / max(1, w.direct_score_lane_work),
        )
        pwp_fetches = ceil_div(direct * phi_hit_rate, cfg.score_lanes)
        miss_fetches = int(round(direct * miss))
        residual_fetches = int(round(direct * phi_hit_rate * residual_ratio))
        weight_fetches = pwp_fetches + miss_fetches + residual_fetches
        delivery_commands = pwp_fetches + miss_fetches + residual_fetches
        metadata = w.pattern_vectors * cfg.pattern_index_bits
    elif scheme in ("static_anchor_term", "static_anchor_hierdesc"):
        # term 数决定 product/weight fetch；destination group 数决定写回。
        # 两者必须独立，不能把跨 destination fanout 免费消掉。
        weight_fetches = w.exact_projection_terms
        if scheme == "static_anchor_term":
            # 现有定长接口可在一条命令中打包最多四个 destination。
            delivery_commands = w.packed_delivery_commands
            metadata = delivery_commands * cfg.descriptor_bits
        else:
            # 当前 ISHD 定义是一条 header + 每个额外 destination 一条
            # continuation；不能把 M4 已打包命令数当 destination 数。
            delivery_commands = w.exact_destination_count
            continuations = max(0, w.exact_destination_count - weight_fetches)
            metadata = (
                weight_fetches * cfg.compact_header_bits
                + continuations
                * (
                    cfg.compact_delta_continuation_bits
                    + cfg.compact_continuation_flags
                )
            )
    else:
        raise ValueError(f"未知方案: {scheme}")

    weight_bits = weight_fetches * cfg.output_lanes * cfg.weight_bits
    bank_bw = cfg.projection_banks * cfg.weight_bank_bits_per_cycle
    weight_cycles = ceil_div(weight_bits, bank_bw)
    compute = max(weight_fetches, delivery_commands)
    return Component(
        compute_cycles=compute,
        memory_cycles=weight_cycles,
        payload_bits=weight_bits,
        metadata_bits=metadata,
        fabric_bits=metadata,
    )


def simulate_sample(
    w: SampleWorkload,
    cfg: ArchConfig,
    scheme: str,
    phi_hit_rate: float = 0.75,
) -> dict:
    score = _score_component(w, cfg, scheme, phi_hit_rate)
    term = _term_component(w, cfg, scheme)
    projection = _projection_component(w, cfg, scheme, phi_hit_rate)
    components = {"score": score, "term": term, "projection": projection}
    total_cycles = sum(c.total_cycles for c in components.values())
    payload_bits = sum(c.payload_bits for c in components.values())
    metadata_bits = sum(c.metadata_bits for c in components.values())
    fabric_bits = sum(c.fabric_bits for c in components.values())
    return {
        "sample_id": w.sample_id,
        "total_cycles": total_cycles,
        "payload_bits": payload_bits,
        "metadata_bits": metadata_bits,
        "fabric_bits": fabric_bits,
        "metadata_ratio": metadata_bits / max(1, metadata_bits + payload_bits),
        "components": {
            name: {
                **asdict(component),
                "total_cycles": component.total_cycles,
            }
            for name, component in components.items()
        },
    }


def summarize_scheme(samples: list[dict], cfg: ArchConfig) -> dict:
    cycle_dist = distribution(s["total_cycles"] for s in samples)
    payload_dist = distribution(s["payload_bits"] for s in samples)
    metadata_dist = distribution(s["metadata_bits"] for s in samples)
    fabric_dist = distribution(s["fabric_bits"] for s in samples)
    ratio_dist = distribution(s["metadata_ratio"] for s in samples)
    stage = {}
    for name in ("score", "term", "projection"):
        stage[name] = distribution(
            s["components"][name]["total_cycles"] for s in samples
        )
    return {
        "cycles": cycle_dist,
        "latency_ms_at_freq": {
            key: value / (cfg.freq_mhz * 1e3)
            for key, value in cycle_dist.items()
            if key != "cv"
        },
        "payload_bits": payload_dist,
        "metadata_bits": metadata_dist,
        "fabric_bits": fabric_dist,
        "metadata_ratio": ratio_dist,
        "component_cycles": stage,
    }


def evaluate_line(
    samples: list[SampleWorkload],
    cfg: ArchConfig,
    target_tokens: int,
    phi_hits: list[float],
) -> dict:
    scaled = workloads_at_tokens(samples, target_tokens)
    schemes = {}
    for scheme in (
        "direct",
        "prosperity_online",
        "static_anchor_term",
        "static_anchor_hierdesc",
    ):
        recs = [simulate_sample(w, cfg, scheme) for w in scaled]
        schemes[scheme] = summarize_scheme(recs, cfg)

    phi_sweep = {}
    for hit in phi_hits:
        recs = [
            simulate_sample(w, cfg, "phi_pattern_residual", phi_hit_rate=hit)
            for w in scaled
        ]
        phi_sweep[f"{hit:.2f}"] = summarize_scheme(recs, cfg)

    direct_cycles = schemes["direct"]["cycles"]["mean"]
    for rec in schemes.values():
        rec["speedup_vs_direct_mean"] = direct_cycles / rec["cycles"]["mean"]
    for rec in phi_sweep.values():
        rec["speedup_vs_direct_mean"] = direct_cycles / rec["cycles"]["mean"]

    ours = schemes["static_anchor_hierdesc"]
    fixed64 = schemes["static_anchor_term"]
    fabric_bit_reduction = 1.0 - (
        ours["fabric_bits"]["mean"] / max(1.0, fixed64["fabric_bits"]["mean"])
    )
    score_lane_equivalent = cfg.score_lanes + cfg.residual_lanes
    lane_normalized_speedup = (
        ours["speedup_vs_direct_mean"]
        * cfg.score_lanes
        / score_lane_equivalent
    )
    phi_break_even = None
    for step in range(101):
        hit = step / 100.0
        phi_cycles = statistics.fmean(
            simulate_sample(
                w,
                cfg,
                "phi_pattern_residual",
                phi_hit_rate=hit,
            )["total_cycles"]
            for w in scaled
        )
        if phi_cycles <= ours["cycles"]["mean"]:
            phi_break_even = hit
            break
    return {
        "line": scaled[0].line,
        "tokens_per_window": target_tokens,
        "samples": len(scaled),
        "evidence": scaled[0].evidence,
        "schemes": schemes,
        "phi_pattern_hit_sweep": phi_sweep,
        "phi_break_even_hit_vs_static_anchor_hierdesc": phi_break_even,
        "decision_metrics": {
            "static_anchor_hierdesc_speedup": ours["speedup_vs_direct_mean"],
            "score_lane_equivalent": score_lane_equivalent,
            "score_lane_normalized_speedup_proxy": lane_normalized_speedup,
            "static_anchor_hierdesc_metadata_ratio_mean": ours["metadata_ratio"]["mean"],
            "static_anchor_fixed64_fabric_bits_mean": fixed64["fabric_bits"]["mean"],
            "static_anchor_hierdesc_fabric_bits_mean": ours["fabric_bits"]["mean"],
            "static_anchor_hierdesc_fabric_bit_reduction_vs_fixed64": fabric_bit_reduction,
            "static_anchor_hierdesc_p99_over_mean": (
                ours["cycles"]["p99"] / ours["cycles"]["mean"]
                if ours["cycles"]["mean"]
                else 0.0
            ),
            "passes_speedup_1p15": ours["speedup_vs_direct_mean"] >= 1.15,
            "passes_score_lane_normalized_1p15": lane_normalized_speedup >= 1.15,
            "passes_fabric_bit_reduction_ge_15pct": fabric_bit_reduction >= 0.15,
            "passes_p99_le_1p25mean": (
                ours["cycles"]["p99"] <= 1.25 * ours["cycles"]["mean"]
            ),
        },
    }


def architecture_dse(
    samples: list[SampleWorkload],
    base_cfg: ArchConfig,
    target_tokens: int,
) -> dict:
    scaled = workloads_at_tokens(samples, target_tokens)
    direct_mean = statistics.fmean(
        simulate_sample(w, base_cfg, "direct")["total_cycles"] for w in scaled
    )
    records = []
    for residual_lanes in (2, 4, 8):
        for bank_bits in (128, 256):
            for continuation_bits in (4, 6, 10):
                cfg = replace(
                    base_cfg,
                    residual_lanes=residual_lanes,
                    weight_bank_bits_per_cycle=bank_bits,
                    compact_delta_continuation_bits=continuation_bits,
                )
                rows = [
                    simulate_sample(w, cfg, "static_anchor_hierdesc")
                    for w in scaled
                ]
                summary = summarize_scheme(rows, cfg)
                speedup = direct_mean / summary["cycles"]["mean"]
                lane_equivalent = cfg.score_lanes + residual_lanes
                lane_normalized = speedup * cfg.score_lanes / lane_equivalent
                fixed_rows = [
                    simulate_sample(w, cfg, "static_anchor_term")
                    for w in scaled
                ]
                fixed_summary = summarize_scheme(fixed_rows, cfg)
                fabric_reduction = 1.0 - (
                    summary["fabric_bits"]["mean"]
                    / max(1.0, fixed_summary["fabric_bits"]["mean"])
                )
                tail = summary["cycles"]["p99"] / summary["cycles"]["mean"]
                # Local5 只有 histogram 服务下界；样本间 p99 不是 ordered
                # FIFO tail，因此在 ordered trace 到达前必须阻断正式晋级。
                tail_available = (
                    scaled[0].line == "Motion"
                    and all(w.evidence.startswith("[prof-ordered") for w in scaled)
                )
                tail_pass = tail <= 1.25 if tail_available else False
                preordered_pass = (
                    speedup >= 1.15
                    and lane_normalized >= 1.15
                    and fabric_reduction >= 0.15
                )
                records.append(
                    {
                        "residual_lanes": residual_lanes,
                        "score_lane_equivalent": lane_equivalent,
                        "weight_bank_bits_per_cycle": bank_bits,
                        "delta_continuation_bits": continuation_bits,
                        "mean_cycles": summary["cycles"]["mean"],
                        "p99_cycles": summary["cycles"]["p99"],
                        "speedup_vs_direct": speedup,
                        "score_lane_normalized_speedup_proxy": lane_normalized,
                        "fabric_bit_reduction_vs_fixed64": fabric_reduction,
                        "p99_over_mean": tail,
                        "tail_evidence": (
                            "[prof-ordered]"
                            if tail_available
                            else "[缺失-ordered]"
                        ),
                        "passes_preordered_screen": preordered_pass,
                        "passes_tail_contract": (
                            tail_pass if tail_available else None
                        ),
                        "passes": preordered_pass and tail_pass,
                        "candidate_status": (
                            "eligible"
                            if preordered_pass and tail_pass
                            else "blocked_missing_ordered_tail"
                            if preordered_pass and not tail_available
                            else "rejected"
                        ),
                    }
                )
    passing = [r for r in records if r["passes"]]
    preordered = [r for r in records if r["passes_preordered_screen"]]
    passing.sort(
        key=lambda r: (
            -r["speedup_vs_direct"],
            -r["fabric_bit_reduction_vs_fixed64"],
            r["score_lane_equivalent"],
        )
    )
    balanced_pool = [
        r
        for r in passing
        if r["delta_continuation_bits"] == 6
        and r["weight_bank_bits_per_cycle"] == 128
    ]
    balanced_pool.sort(
        key=lambda r: (
            -r["speedup_vs_direct"],
            r["score_lane_equivalent"],
        )
    )
    return {
        "line": scaled[0].line,
        "tokens_per_window": target_tokens,
        "candidate_count": len(records),
        "preordered_screen_count": len(preordered),
        "passing_count": len(passing),
        "top_passing": passing[:12],
        "recommended_balanced": balanced_pool[0] if balanced_pool else None,
        "all_candidates": records,
        "warning": (
            "Local5 使用逐向量 histogram 服务下界，不代表 ordered FIFO p99；"
            "缺失 ordered tail 时正式 passing_count 强制为 0；"
            "delta continuation 仍需 post-G0 ordered trace 测 escape rate。"
        ),
    }


def build_report(
    cfg: ArchConfig,
    phi_hits: list[float],
    motion_profile_t162: Path = MOTION_PROFILE_T162,
    motion_profile_t450: Path = MOTION_PROFILE_T450,
) -> dict:
    local = load_local_samples()
    motion_t162 = load_motion_samples(motion_profile_t162)
    motion_t450 = load_motion_samples(motion_profile_t450)
    evaluations = {
        "Local5_T162": evaluate_line(local, cfg, 162, phi_hits),
        "Local5_T450": evaluate_line(local, cfg, 450, phi_hits),
        "Motion_T162": evaluate_line(motion_t162, cfg, 162, phi_hits),
        "Motion_T450": evaluate_line(motion_t450, cfg, 450, phi_hits),
    }

    return {
        "schema": "phi_prosperity_dual_line_sim_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_policy": {
            "T162": "[模型]+真实 profile100；周期不是 RTL/DC",
            "Motion_T450": "[模型]+真实 fullres ordered profile100；周期不是RTL/DC",
            "Local5_T450": "[外推]，等待同窗全head fullres profile替换",
            "Phi": "[模型] 命中率扫描；不是官方 Phi artifact",
            "Prosperity": (
                "仅导入并校验官方Stats字段；正式周期由本地兼容模型计算，"
                "未调用官方Simulator，也不复用论文PPA常数"
            ),
        },
        "sources": {
            "phi_paper": "https://arxiv.org/abs/2505.10909",
            "phi_official_public_simulator": None,
            "prosperity_paper": "https://arxiv.org/abs/2503.03379",
            "prosperity_repo": "https://github.com/dubcyfor3/Prosperity",
            "prosperity_local": {
                "path": str(PROSPERITY_SIM.relative_to(ROOT)),
                "commit": git_commit(PROSPERITY_SIM.parent),
                "utils_sha256": sha256_file(PROSPERITY_SIM / "utils.py"),
                "simulator_sha256": sha256_file(PROSPERITY_SIM / "simulator.py"),
            },
            "local5_profile": {
                "path": str(LOCAL_PROFILE.relative_to(ROOT)),
                "sha256": sha256_file(LOCAL_PROFILE),
            },
            "motion_profile_t162": {
                "path": str(motion_profile_t162.resolve()),
                "sha256": sha256_file(motion_profile_t162),
            },
            "motion_profile_t450": {
                "path": str(motion_profile_t450.resolve()),
                "sha256": sha256_file(motion_profile_t450),
            },
            "model_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "prosperity_official_stats_probe": make_official_stats_probe(),
        "arch_config": asdict(cfg),
        "evaluations": evaluations,
        "architecture_dse": {
            "Local5_T162": architecture_dse(local, cfg, 162),
            "Motion_T162": architecture_dse(motion_t162, cfg, 162),
            "Local5_T450": architecture_dse(local, cfg, 450),
            "Motion_T450": architecture_dse(motion_t450, cfg, 450),
        },
        "kill_thresholds": {
            "speedup_vs_equal_direct": ">=1.15",
            "score_lane_normalized_speedup_proxy": ">=1.15，按 Direct32 对 32+W lane 归一",
            "fabric_bits": "相对同配置 fixed64 M4 接口至少减少15%",
            "p99_cycles": "<=1.25x mean",
            "accuracy": "必须通过各自主线 fullres hardware-order/RTL-exact；模型不代替准确率",
            "ppa": "进入主贡献前必须补同 SDC/DC/STA/SAIF",
        },
        "non_claims": [
            "不声称存在公开 Phi 官方模拟器",
            "不把 Phi 命中率 sweep 当作实测",
            "不把 Prosperity 的固定功耗常数当作本设计功耗",
            "不把官方Stats字段probe写成调用官方Simulator或官方周期结果",
            "不把 online-matcher oracle 写成 Prosperity 官方 simulator 结果",
            "不把 Local5 T450 线性外推或 Motion旧T162外推当作fullres profile",
            "不把本模型周期当作 RTL cycle-accurate 或 DC PPA",
        ],
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Phi/Prosperity 双线分层稀疏性能评估\n\n",
        "## 1. 证据边界\n\n",
        "- Prosperity：仅导入并校验官方仓库 `Stats` 字段；正式周期由本地兼容模型计算，未调用官方 Simulator，也不使用其论文功耗常数。\n",
        "- Phi：论文公开了两级 pattern/residual 机制，但未找到官方公开模拟器。本文件是机制复刻和命中率敏感性分析，不是官方 Phi 结果。\n",
        "- T=162：来自同 cohort 的 100 样本 profile；周期仍是模型。\n",
        "- Motion T=450：读取最终 H67 fullres ordered profile100；Local5 T=450 仍为 `450/162=25/9` 外推。\n\n",
        "## 2. 同约束设置\n\n",
        f"- Direct score 为 `{report['arch_config']['score_lanes']}` lane；静态锚点硬件为 "
        f"`{report['arch_config']['score_lanes']}+{report['arch_config']['residual_lanes']}` lane。"
        "周期按真实结构建模，并额外报告 lane 归一吞吐代理；这不是 DC 面积。\n",
        f"- projection：`{report['arch_config']['projection_banks']}×{report['arch_config']['weight_bank_bits_per_cycle']} bit/cycle` bank。\n",
        f"- activation SRAM：`{report['arch_config']['activation_sram_bits_per_cycle']} bit/cycle`；描述符：`{report['arch_config']['descriptor_bits']} bit`。\n",
        f"- Prosperity 官方 `Stats` 字段 probe：`{report['prosperity_official_stats_probe']['imported']}`；该 probe 不参与正式周期计算。\n\n",
        "## 3. 结果总表\n\n",
        "| 主线/规模 | 方案 | mean cycle | p95 | p99 | 相对 direct | fabric bits | 全部 metadata占比 |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for name, result in report["evaluations"].items():
        for scheme, rec in result["schemes"].items():
            lines.append(
                f"| {name} | {scheme} | {rec['cycles']['mean']:.0f} | "
                f"{rec['cycles']['p95']:.0f} | {rec['cycles']['p99']:.0f} | "
                f"{rec['speedup_vs_direct_mean']:.3f}× | "
                f"{rec['fabric_bits']['mean']:.0f} | "
                f"{100*rec['metadata_ratio']['mean']:.2f}% |\n"
            )
        for hit, rec in result["phi_pattern_hit_sweep"].items():
            lines.append(
                f"| {name} | Phi-like hit={hit} | {rec['cycles']['mean']:.0f} | "
                f"{rec['cycles']['p95']:.0f} | {rec['cycles']['p99']:.0f} | "
                f"{rec['speedup_vs_direct_mean']:.3f}× | "
                f"{rec['fabric_bits']['mean']:.0f} | "
                f"{100*rec['metadata_ratio']['mean']:.2f}% |\n"
            )

    lines.extend(
        [
            "\n## 4. 淘汰门槛\n\n",
            "| 门槛 | 要求 |\n|---|---|\n",
            "| 周期收益 | 同 lane/direct 基线至少 `1.15×` |\n",
            "| lane 归一代理 | `raw speedup × 32/(32+W)` 至少 `1.15×` |\n",
            "| 互连命令位数 | ISHD 相对同配置 fixed64/M4 接口至少减少 `15%` |\n",
            "| 尾延迟 | p99 不超过 mean 的 `1.25×` |\n",
            "| 数值 | fullres hardware-order/RTL-exact 通过 |\n",
            "| PPA | 同 SDC、同 SRAM macro 规则下 DC/STA/SAIF |\n\n",
            "## 5. 解释限制\n\n",
            "- `prosperity_online` 只是有利于对手的 online-matcher oracle：允许其使用相同 residual work、相同 exact term 投影复用，并共享 K 读取；它不是 Prosperity 官方 simulator 结果。\n",
            "- `Phi-like` 使用 50%/75%/90%/95% 命中率扫描。只有未来从真实 Q/K/投影 trace 学到 codebook 后，某一行才可升级为 `[prof]`。\n",
            "- `static_anchor_term` 按现有 64-bit/M4 打包命令；`static_anchor_hierdesc` 按一条 term header + 每个额外 destination 一条窄 continuation。两者都守恒同一目的集合。\n",
            "- 因此 `static_anchor_hierdesc` 的负结果只适用于当前 scalar-continuation 实现假设，不能外推为所有可打包 hierarchical descriptor 都无收益。\n",
            "- 层级描述符按一条 term header 加 destination continuation 建模；continuation 已计 4-bit flags，但 delta escape rate 仍须 post-G0 ordered trace 验证。\n",
            "- Local5 明确区分唯一 `mfep_multicast_terms` 与实际 `destination_gate_lane_groups`；不再按假设 fanout 二次缩减 product 或 delivery。\n",
            "- Motion 的 M1 是真实目的数量、M4 是打包命令数；二者不再混用。Local5 pre-G0 无打包证据，保守按 M1。\n",
            "- Motion score tail 来自真实 ordered delta trace；Local5 只有逐向量 histogram 服务下界，样本 p99 不能解释为 FIFO p99。\n",
            "- 本表只比较 attention-to-projection 子系统，不是 full encoder FPS。\n\n",
            "## 6. Phi break-even 与架构 DSE\n\n",
        ]
    )
    for name, result in report["evaluations"].items():
        hit = result["phi_break_even_hit_vs_static_anchor_hierdesc"]
        lines.append(
            f"- `{name}`：Phi-like 相对层级 term 的 break-even L1 hit = "
            f"`{'未达到' if hit is None else f'{100*hit:.0f}%'}`。\n"
        )
    lines.extend(
        [
            "\n| 主线/规模 | DSE 候选 | 三项预筛 | 正式过门槛 | 最优配置（仅模型） |\n",
            "|---|---:|---:|---:|---|\n",
        ]
    )
    for name, dse in report["architecture_dse"].items():
        best = dse["recommended_balanced"]
        if best:
            desc = (
                f"W{best['residual_lanes']}, SRAM{best['weight_bank_bits_per_cycle']}, "
                f"delta{best['delta_continuation_bits']}; "
                f"raw {best['speedup_vs_direct']:.3f}×, "
                f"lane归一 {best['score_lane_normalized_speedup_proxy']:.3f}×"
            )
        else:
            desc = "无"
        lines.append(
            f"| {name} | {dse['candidate_count']} | "
            f"{dse['preordered_screen_count']} | {dse['passing_count']} | {desc} |\n"
        )
    lines.extend(
        [
            "\n- DSE 过门槛只表示模型可继续，不表示可投稿；W8、fanout8 等激进点仍需 RTL/面积淘汰。\n",
            "- Local5 的 delta 位宽、escape 与 ordered tail 必须由 post-G0 trace 替换；证据到达前正式过门槛数固定为 0。\n\n",
            "## 7. 复现\n\n",
            "```bash\n",
            "/opt/conda/envs/sdformerflow/bin/python scripts/phi_prosperity_dual_line_simulator.py\n",
            "/opt/conda/envs/sdformerflow/bin/python -m unittest tests.test_phi_prosperity_dual_line_simulator -v\n",
            "```\n",
        ]
    )
    path.write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--motion-profile-t162", type=Path, default=MOTION_PROFILE_T162)
    parser.add_argument("--motion-profile-t450", type=Path, default=MOTION_PROFILE_T450)
    parser.add_argument(
        "--phi-hits",
        default="0.50,0.75,0.90,0.95",
        help="Phi-like L1 pattern 命中率列表",
    )
    args = parser.parse_args()
    phi_hits = [float(x) for x in args.phi_hits.split(",")]
    if any(x < 0.0 or x > 1.0 for x in phi_hits):
        raise SystemExit("Phi 命中率必须位于 [0,1]")

    report = build_report(
        ArchConfig(),
        phi_hits,
        motion_profile_t162=args.motion_profile_t162,
        motion_profile_t450=args.motion_profile_t450,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.out / "report.md")
    print(args.out / "report.md")
    for name, result in report["evaluations"].items():
        d = result["decision_metrics"]
        print(
            name,
            f"hierdesc_speedup={d['static_anchor_hierdesc_speedup']:.3f}",
            f"lane_norm={d['score_lane_normalized_speedup_proxy']:.3f}",
            f"fabric_bit_reduction={100*d['static_anchor_hierdesc_fabric_bit_reduction_vs_fixed64']:.2f}%",
            f"all_metadata={100*d['static_anchor_hierdesc_metadata_ratio_mean']:.2f}%",
            f"p99/mean={d['static_anchor_hierdesc_p99_over_mean']:.3f}",
        )
    print("prosperity_stats_import", report["prosperity_official_stats_probe"]["imported"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
