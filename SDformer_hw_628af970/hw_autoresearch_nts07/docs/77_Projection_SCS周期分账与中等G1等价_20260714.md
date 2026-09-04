# Projection / SCS 周期分账与中等 G1 等价（2026-07-14）

**状态**：本轮 **不需要 GPU**（训练可继续占卡）。  
**约束**：不删改 GPT 已有 RTL / TB / 脚本；全部新增文件。  
**证据**：周期 = [模型]；work-item = [prof] compact；RTL 等价 = [rtl] 仿真。

---

## 0. 资源声明

| 资源 | 本轮 |
|------|------|
| GPU | **不需要**；`nvidia-smi` 显示 0 MiB，留给训练 |
| CPU | cycle ledger + iverilog medium TB |
| 修改 GPT 代码 | **无** |

---

## 1. 本轮新增文件

| 路径 | 作用 |
|------|------|
| `scripts/projection_scs_cycle_ledger.py` | SCS + projection 周期/工作量分账（读 compact + GCM-P DSE） |
| `scripts/test_projection_scs_cycle_ledger.py` | 单元测试（5/5 PASS） |
| `results/projection_scs_cycle_ledger_20260714.{json,md}` | 分账产物 |
| `tb_hitflow/tb_hitflow_g1_projection_top_medium.sv` | 中等规模 G1 direct/NMF 整数等价 TB |
| `sim_hitflow/run_projection_g1_medium_checks.sh` | 仅跑 medium TB（不改 GPT `run_projection_g1_checks.sh`） |

---

## 2. SCS 行核周期分账

公式与既有 `score_class_scan_cycle_model` 一致，输入改为 compact profile100 stage 均值：

```text
row = N_TOKENS + max(active,1) + class_cycles + active + CONTROL
fixed: class_cycles = 35
SCS:   class_cycles = 2 × fold_classes_mean
```

| 指标 | 数值 | 等级 |
|------|------|------|
| 固定扫描周期/帧 | 1,590,541 | [模型] |
| SCS 周期/帧 | 1,385,838 | [模型] |
| **周期下降** | **12.87%** | [模型] |
| 500MHz 行核帧率（SCS） | 360.79 | [模型] 非端到端 |

| stage | active/row | fold | 固定 cyc/row | SCS cyc/row | 下降 |
|------:|-----------:|-----:|-------------:|------------:|-----:|
| 0 | 31.46 | 2.74 | 262.91 | 233.39 | 11.2% |
| 1 | 3.63 | 1.36 | 207.27 | 174.98 | **15.6%** |
| 2 | 10.79 | 2.33 | 221.58 | 191.24 | 13.7% |
| 3 | 24.36 | 2.13 | 248.72 | 217.98 | 12.4% |

**解读**：SCS 在行核上的周期收益中等（~13%），主因 active 项与 token 扫描仍占主导；但与 K-zero 88.7% 的 **动态/存储** 收益是正交的，不能只用周期下降衡量 SCS。

---

## 3. Projection work-item 与 GCM-P 周期

### 3.1 Work-items [prof]

| 项 | 数值 |
|----|------|
| baseline active lanes | 40,560,225 |
| final-gate NMF terms | 7,101,034 |
| **乘积 work 减少** | **82.49%** |
| gate 相对 score-class 额外合并 | 61.03% |
| delivery M=1 / M=4 / M=8 | 100% / 34.2% / 24.0% |

### 3.2 DSE 锁表（S=4，overflow≈0.014%）

瓶颈在全部锁配置中均为 **delivery**（candidate = max(product, delivery)）。

| 推荐关注配置 | ideal speedup | p95 row cyc | 备注 |
|--------------|--------------:|------------:|------|
| **M=4, L=32, P=1** | 2.72× | 696 | 论文默认候选：多播中等、P 不堆 |
| M=8, L=32, P=1 | 3.67× | 504 | 更高 M，需看 bank 冲突 |
| M=1, L=32, P=1 | 1.00× | 2052 | 无多播时 NMF 周期不显形（delivery=baseline） |

**关键结构结论**：

1. **NMF 的 82% 是乘积 work 减少**；在 M=1 时 delivery 仍等于 active lanes，**ideal cycle speedup≈1**。  
2. 周期显形依赖 **multicast 宽度 M≥4** 压 delivery。  
3. 提高 P（product engines）在 delivery 瓶颈下 **ideal speedup 反而下降**（direct 变快、candidate 几乎不变）——不要盲目堆乘积引擎。

### 3.3 与架构锁对齐

> **锁：H67 + SCS + NMF(G=1,S=4) + exact pair/K-zero**  
> 投影后端默认推向 **S=4, M=4, L=32, P=1** 作 DATE 叙述与后续 RTL 参数表；M=8 作上限对照。  
> G≥2 / PHEA / 蝶形互连：仍暂缓（docs/76）。

---

## 4. 中等规模 G1 等价 TB

### 4.1 参数

| 参数 | 小 TB（GPT） | **中等 TB（本轮）** | 全量 deploy 目标 |
|------|-------------|---------------------|------------------|
| TOKENS | 6 | **24** | 162 |
| LANES | 4 | **8** | 32 |
| OUT_TILE | 2 | **4** | 8 |
| BANKS | 2 | **4** | 2–8 |
| SEGMENT | 2 | **8** | 18 |
| SLOTS | 4 | **4** | 4 |

### 4.2 结果

```text
./sim_hitflow/run_projection_g1_medium_checks.sh
CASE1 shared-gate + filters     cycles≈145  terms=12
CASE2 dense K + slot reuse      cycles≈427  terms=32
CASE3 edge gate/weight          cycles≈113  terms=8
CASE4 PRNG within SLOTS         cycles≈247  terms=28
PASS: medium G1 projection top direct/NMF integer equivalence
```

4 组用例全部 **direct dense 软件金参考 == NMF RTL 路径** 整数一致；无 overflow、无 protocol_error。

### 4.3 发现：TOKENS 为 2 的幂时 GPT 累加器挂死（**未改 GPT 代码**）

调试中等 TB 时定位：

```systemverilog
// hitflow_banked_accumulator.sv
assign input_token_in_range = input_token < TOKEN_ID_W'(TOKENS);
```

当 `TOKENS` 为 2 的幂时，`TOKEN_ID_W = $clog2(TOKENS)`，`TOKEN_ID_W'(TOKENS)` **截断为 0**，`update_ready` 永假 → multicast 卡在 `active_q`，顶层停在 `ST_RUN` / NMF `ST_DRAIN_DIRECTORY`。

| TOKENS | 现象 |
|-------:|------|
| 6, 12, **24**, 162 | 正常 |
| 16, 32, 64, 128 | 挂死（仿真复现） |

**处置（本轮）**：

- medium TB 使用 **TOKENS=24**（非 2 幂），避免改 GPT RTL；  
- 全量 **162 安全**；  
- 建议后续由 GPT/主线用更宽比较修复，例如  
  `input_token < (TOKEN_ID_W+1)'(TOKENS)` 或 `32'(input_token) < 32'(TOKENS)`。  
  **本 agent 按用户要求未改 GPT 文件。**

---

## 5. 架构建议（更新）

| 优先级 | 项 | 依据 |
|--------|----|------|
| 保持 | SCS + NMF(G=1,S=4) | work −82.5%；S=4 overflow 0.014% |
| 投影参数表 | **M=4, L=32, P=1** 优先 | DSE speedup 2.72× 且不堆 P |
| 仿真 | medium 24×8 PASS；全量 162 时注意非 2 幂或先修 range 比较 | [rtl] |
| 暂缓 | G≥2 / PHEA / 蝶形 | docs/76 门槛 |
| 不做本轮 | GPU profile 重跑、DC/SAIF、改 GPT RTL | 用户训练优先 |

---

## 6. 复现（CPU）

```bash
cd hw_autoresearch_nts07

# 周期分账
PYTHONPATH=scripts python3 scripts/projection_scs_cycle_ledger.py
PYTHONPATH=scripts python3 -m unittest scripts.test_projection_scs_cycle_ledger -v

# 中等 G1 等价
./sim_hitflow/run_projection_g1_medium_checks.sh
```

---

## 7. 未完成 / 下一步（仍可不占 GPU）

1. **可选**：新建旁路补丁文件修复 `TOKEN_ID_W'(TOKENS)`（或等 GPT 修），再开 32/64 维 TB。  
2. 中等 → 更大 **TOKENS=48/96/160**（非 2 幂）步进；全量 162×32 前建议先修 range。  
3. 把 bias-commit / NMF 建表 串行开销并入 cycle ledger（现 DSE 不含）。  
4. storage budget schema 适配（docs/76 §6）。  
5. DC / SAIF 仍无。

---

## 8. 一句话

> **GPU 可全部留给训练。** 本轮用 CPU 完成 SCS/projection 分账（NMF work −82.5%，周期显形靠 M≥4）与 **TOKENS=24 中等 G1 整数等价 PASS**；并记录 GPT 累加器在 **TOKENS=2ⁿ 时的 range 截断挂死**，未改 GPT 代码。
