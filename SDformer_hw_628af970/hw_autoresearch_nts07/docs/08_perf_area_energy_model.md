# 性能 / 面积 / 能耗模型

## 1. 周期模型

基于 `scripts/nts07_perf_model.py`，NTS-07b 默认参数 @ 288×384。

### 1.1 符号

| 符号 | 含义 | 默认值 |
|------|------|--------|
| T | PSN steps | 10 |
| W_s | stage s 的 window 数 | 见下表 |
| N | tokens/window | 98 (2×7×7) |
| D_s | head_dim | 32 (S2) |
| H_s | heads | 12 (S2) |
| f | global firing rate | 0.079 |
| r_win | active window ratio | 0.85 |
| r_tok | token keep ratio | 0.80 |

### 1.2 各 Stage Window 数（288×384 crop）

| Stage | H×W | windows | blocks | attn type |
|-------|-----|---------|--------|-----------|
| S0 | 240×320 | 800 | 2 | Legacy |
| S1 | 120×160 | 200 | 2 | Legacy |
| S2 | 60×80 | 50 | 6 | **H60** |
| S3 | 30×40 | 13 | 2 | Legacy |

### 1.3 周期公式

```text
# Sparse MAC (MLP dominant)
mac_cycles_s = T × W_s × blocks × N × (4 × dim_s × mlp_ratio) × (1-f) / PE_MAC

# H60 (S2 only)
h60_cycles = T × W_s × blocks × H_s × (
    2 × N × D ×/popcount_par   # TX+SC
  + shiftmax_lat(N)            # ~20
  + N × D / gate_par           # K-gate
) × r_win × r_tok

# Legacy attn
legacy_cycles = T × W_s × blocks × H × (3 × N) × r_win

total_cycles = scatter + patch + Σ mac_cycles + h60_cycles + legacy + decode
```

### 1.4 Baseline 估算（默认 PE=128, 500MHz）

| 子系统 | Mcycles | 占比 |
|--------|---------|------|
| Event scatter | 0.10 | 2% |
| Patch embed | 1.20 | 24% |
| S0/S1/S3 MAC+attn | 2.10 | 42% |
| **S2 MAC+H60** | **1.40** | **28%** |
| Decoder | 0.20 | 4% |
| **Total** | **~5.0** | 100% |

→ **~10 ms/frame → 100 FPS** 理论上限（单引擎）；实际 30 FPS 目标留 3× margin 给 DRAM。

---

## 2. 能耗模型

沿用 SDformerFlow Table V：

```text
E_snn = FLOPS_eff × R_spike × T × E_AC
E_ann = FLOPS × E_MAC
```

| 常数 @ 45nm | 值 |
|-------------|-----|
| E_MAC | 4.6 pJ |
| E_AC | 0.9 pJ |
| E_AND | 0.05 pJ (估算) |

### NTS-07b 估算

| 引擎 | 有效操作 | 能耗 mJ |
|------|----------|---------|
| Sparse MAC | 91G eff AC | ~25 |
| H60 Binary | 0.8G AND+pop | **<0.5** |
| Dense MAC | 15G MAC | ~70 |
| SRAM | 12MB access | ~8 |
| **Total** | | **~35 mJ/frame** |

对比 NB0 profile energy ~37.6 mJ → **−7%**（与 SOPs 降幅一致）

---

## 3. 面积模型（28nm 估算）

| 模块 | LUT/KGE | SRAM KB | 备注 |
|------|---------|---------|------|
| H60 engine | 45 | 32 | TX+SC+Shiftmax+gate |
| Legacy attn | 15 | 16 | 每实例可复用 |
| Sparse MAC 128PE | 180 | 0 | bit-serial |
| Dense MAC 32×32 | 220 | 64 | systolic |
| Controller+DMA | 40 | 4 | metadata |
| **Total logic** | **~500** | **512+256** | <2MB ✓ |

Yosys 综合后需用 `hw/scripts/run_synth.sh` 更新本表。

---

## 4. Autoresearch 实测结果（11 轮，2026-06-09）

| 方案 | 能耗(mJ) | FPS | SRAM(KB) |
|------|---------|-----|----------|
| 基线 | 25.94 | 92.8 | 772 |
| PE 256 | 12.97 | 101.3 | 772 |
| **终极组合** | **12.97** | **101.3** | **388** |

详见 `docs/10_autoresearch实验结果.md`。推荐配置：`scripts/configs/best_config.json`。

## 5. Autoresearch 指标输出格式

`autoresearch.sh` 输出 `METRIC name=value` 行，记录于 `autoresearch.jsonl`。