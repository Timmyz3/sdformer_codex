# POW2-Safe 补丁、相序分账与 Storage Schema 适配（2026-07-15）

> **2026-07-15 后续审计**：本文的 `1.51x` 是旧相序模型点，不是 RTL 实测加速。该口径采用 buffered direct 基线并未显式计入固定 128 项目录扫描；统一为可流式 direct 后，当前 G1 扫描下界为 `1.176x`，双 context 候选模型为 `1.556x`。以 `results/projection_phase_model_audit_20260715.md` 和 `docs/80_Grok接手成果审计与主线修复_20260715.md` 为准。

**状态**：三项全做完；**GPU 不需要**（训练可继续）。  
**约束**：不删改 GPT 原 RTL / 原 TB / 原 budget 脚本；全部 **新文件**。  
**前置**：docs/76（架构锁）、docs/77（中等 TB + 首版分账）。

---

## 0. 资源与一句话

| 项 | 结论 |
|----|------|
| GPU | **不需要** |
| 一句话 | 旁路修了 `TOKENS=2ⁿ` 累加器截断并跑通 **T=32/64** 等价；cycle ledger **并入 NMF 建表 + bias-commit**；storage ablation **自动重定向**到 contract 后 budget 可跑 |

---

## 1. 新增文件清单

### 1.1 POW2-safe RTL 旁路（不改 GPT）

| 路径 | 说明 |
|------|------|
| `rtl_hitflow_patch/hitflow_banked_accumulator_pow2safe.sv` | 自 GPT 累加器复制；**唯一语义修改**见下 |
| `rtl_hitflow_patch/hitflow_g1_projection_top_pow2safe.sv` | G1 顶层副本，实例化 pow2safe 累加器 |

**修复点**（原 GPT 挂死根因）：

```systemverilog
// GPT 原式（TOKENS=2^n 时 TOKEN_ID_W'(TOKENS)==0 → update_ready 永假）
// assign input_token_in_range = input_token < TOKEN_ID_W'(TOKENS);

// POW2SAFE
assign input_token_in_range = 32'(input_token) < 32'(TOKENS);
```

| TOKENS | GPT 原累加器 | pow2safe |
|-------:|:------------:|:--------:|
| 6 / 12 / 24 / 162 | OK | OK |
| **16 / 32 / 64 / 128** | **挂死** | **OK** |

### 1.2 规模 TB + 仿真入口

| 路径 | 规模 | 结果 |
|------|------|------|
| `tb_hitflow/tb_hitflow_g1_projection_top_pow2_32.sv` | T=32,L=8,O=4,B=4,SEG=8 | **PASS** 3 用例 |
| `tb_hitflow/tb_hitflow_g1_projection_top_pow2_64.sv` | T=64,L=8,O=4,B=4,SEG=8 | **PASS** 3 用例 |
| `sim_hitflow/run_projection_g1_pow2safe_checks.sh` | 串跑 32+64 | **ALL PASS** |

仿真摘录：

```text
POW2SAFE T=32 CASE1/2/3 → PASS
POW2SAFE T=64 CASE1/2/3 → PASS
[pow2safe] ALL PASS (T=32 and T=64)
```

| T | CASE1 cycles | CASE2 cycles | CASE3 cycles |
|--:|-------------:|-------------:|-------------:|
| 32 | 189 | 557 | 281 |
| 64 | 365 | 1069 | 517 |

### 1.3 周期分账 v2（含相序）

| 路径 | 说明 |
|------|------|
| `scripts/projection_scs_cycle_ledger.py` | **schema_version=2**，并入 NMF build / bias / finish |
| `scripts/test_projection_scs_cycle_ledger.py` | 9 测中含相序用例 |
| `results/projection_scs_cycle_ledger_20260715.{json,md}` | 本轮产物 |

### 1.4 Storage schema 适配 + budget

| 路径 | 说明 |
|------|------|
| `scripts/adapt_storage_and_run_encoder_budget.py` | 识别/重定向 + 调 GPT `build_model` |
| `scripts/test_adapt_storage_and_run_encoder_budget.py` | schema 单测 |
| `results/hit_flow_full_encoder_budget_adapted_20260715.{json,md}` | 适配后 budget |
| `results/storage_schema_adapter_map_20260715.md` | schema 映射表 |

---

## 2. 相序分账：NMF 建表 + bias-commit

### 2.1 口径（对齐 G1 顶层 RTL）

```text
总串行 ≈ NMF_build(token 流) + max(product, delivery)[DSE] + bias_commit + finish
```

| 相 | 模型 | 是否加 wall |
|----|------|:-----------:|
| NMF BUILD | `rows × T`，T=162，1 token/cyc | **是** |
| 目录 empty scan | `max(0, S×K_lanes − terms/row)` | 否（假定被 product 吸收，只报告） |
| product \|\| delivery | GCM-P DSE candidate | **是** |
| bias-commit | `rows × ceil(T/BANKS)×2`（BANKS 默认 2） | **是** |
| finish/done | `rows × 2` | **是** |

### 2.2 关键数字（H67，S=4 锁表，672000 rows）

公共开销（与 M/L/P 无关）：

| 相 | 总周期 |
|----|-------:|
| NMF build | **108,864,000** |
| bias-commit | **108,864,000** |
| finish | 1,344,000 |
| **additive 合计** | **≈219.07M** |

| M | L | P | DSE candidate | DSE ideal speedup | **总串行** | 开销占比 | **有效 speedup** |
|--:|--:|--:|---:|---:|---:|---:|---:|
| 4 | 32 | 1 | 92.2M | **2.721×** | 311.3M | 70.4% | **1.510×** |
| 8 | 32 | 1 | 68.4M | **3.668×** | 287.5M | 76.2% | **1.635×** |
| 4 | 32 | 2 | 91.7M | 1.375× | 310.8M | 70.5% | 1.161× |
| 1 | 32 | 1 | 251.0M | 1.000× | 470.1M | 46.6% | 1.000× |

### 2.3 架构含义（比纯 DSE 更硬）

1. **只报 DSE ideal 会高估**：M=4/L=32 从 2.72× 掉到 **1.51×** 有效串行加速。  
2. **NMF 建表与 bias 已与 delivery 同量级**，甚至更大 → 下一代 RTL 必须：  
   - token 流与前窗 drain **重叠**；  
   - bias **多 bank 并行发射**（提高 BANKS 或双发）；  
   - 或 bias 与 multicast 尾部重叠。  
3. **仍不改变锁**：NMF work-item −82.5% [prof] 仍成立；但 DATE 周期表必须写 **有效 speedup**，不能只写 DSE ideal。  
4. 堆 product engines（P=2）在 delivery 瓶颈下仍不划算。

---

## 3. Storage schema 适配

### 3.1 失败根因（docs/76）

| 文件 | 顶层键 | budget 能否直读 |
|------|--------|:---------------:|
| `h67_h68_encoder_storage_contract.json` | `models.H67.atlif_execution_graph` … | **能** |
| `h67_h68_storage_ablation.json` | `状态` / `结果`（Yosys 面积消融） | **不能** |

GPT `model_hit_flow_full_encoder_budget.py` 写死：

```python
h67_storage = storage["models"]["H67"]
graph = h67_storage["atlif_execution_graph"]
```

把 ablation 喂进去 → `KeyError: 'models'`。

### 3.2 适配器行为

```text
输入 storage
  ├─ encoder_storage_contract → 直通
  └─ storage_ablation_yosys   → 重定向到 contract + 记 notes
然后校验 profile.port_aware_pipeline_dse / sops
调用 GPT build_model（不改其源码）
写出 adapted budget + schema map
```

本轮演示命令：

```bash
PYTHONPATH=scripts python3 scripts/adapt_storage_and_run_encoder_budget.py --try-ablation-first
# schema=redirected_ablation_to_contract configs=192 pass30fps=0
```

### 3.3 Budget 结果摘要（ordered 空间代理）

| 项 | 值 |
|----|-----|
| 配置数 | 192 |
| 过 30FPS（guarded serial） | **0** |
| 最优 guarded FPS | **≈23.09**（4×DP, 1024 spatial, 512b, bypass=100%, PCCC 理想上界） |
| 空间代理来源 | H67 ordered profile100 活动率加权 MAC |
| live ATLIF temporal MAC/帧 | 4,424,388,480 |
| 证据 | **[模型]** 非 RTL/DC |

**解读**：在 **串行 + 1.25 保护** 且用 ordered 高空间代理时，系统 **够不着 30FPS**；这与 docs/59 旧代理下部分配置“擦边通过”不矛盾——空间代理更真实后更紧。架构含义：

- 必须依赖 **阶段重叠**（模型 perfect-overlap 下界仍低于预算时才有戏）；  
- HTT bypass / 宽总线 / 更大 spatial lane 是系统级旋钮；  
- **不能**用本表宣称 ASIC 达 30FPS。

---

## 4. 与架构锁的关系

| 决策 | 更新 |
|------|------|
| 锁 H67+SCS+NMF(G=1,S=4) | **维持** |
| 投影参数表 M=4,L=32,P=1 | **维持**，但周期宣传用 **有效 1.51×** 而非 2.72× |
| POW2 TOKENS TB | 用 **pow2safe 旁路**；GPT 原文件未动，建议主线合入同一行修复 |
| G≥2 / PHEA | 仍缓；先做 **建表/bias 重叠** 比开 G=2 更划算 |
| 全 encoder 30FPS | 串行模型 **未过**；重叠与 bypass 证据未闭合 |

---

## 5. 复现（全 CPU）

```bash
cd hw_autoresearch_nts07

# 单测
PYTHONPATH=scripts python3 -m unittest \
  scripts.test_projection_scs_cycle_ledger \
  scripts.test_adapt_storage_and_run_encoder_budget -v

# 相序分账
PYTHONPATH=scripts python3 scripts/projection_scs_cycle_ledger.py

# storage 适配 + budget（故意从 ablation 进入）
PYTHONPATH=scripts python3 scripts/adapt_storage_and_run_encoder_budget.py \
  --try-ablation-first

# 或直通 contract
PYTHONPATH=scripts python3 scripts/adapt_storage_and_run_encoder_budget.py

# POW2-safe 32/64 等价
./sim_hitflow/run_projection_g1_pow2safe_checks.sh
```

---

## 6. 证据等级

| 声明 | 等级 |
|------|------|
| T=32/64 pow2safe 整数等价 PASS | **[rtl]** iverilog |
| GPT 累加器 2ⁿ 截断根因 | **[rtl]** 仿真复现 + 代码审查 |
| NMF work −82.49% | **[prof]** |
| DSE / 相序 / 有效 speedup | **[模型]** |
| Budget FPS / 带宽 | **[模型]** |
| DC/SAIF/mW | **无** |

---

## 7. 未完成（下轮仍可不占 GPU）

1. 把 pow2safe 一行修复合入 GPT 主线（需允许改 GPT 文件时）。  
2. G1 顶层 **build/bias 重叠** 微架构（新 RTL 文件）。  
3. 全量 162×32 pow2safe TB（时间较长）。  
4. budget 加入 projection 相序 / 重叠因子，与 attention DSE 对齐。  
5. DC / SAIF。

---

## 8. 一句话

> **GPU 仍可全给训练。** 本轮用新文件完成：① pow2safe 旁路 + **T=32/64 PASS**；② cycle ledger 并入 **NMF 建表+bias**，有效加速从 2.72× 校正到 **≈1.51×**；③ storage ablation **自动适配**到 contract 并产出 budget（串行模型最优约 **23 FPS**，不过 30）。
