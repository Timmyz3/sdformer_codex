# Local5 独立模块确认、Codex Motion 推进仿写、Prosperity 评估接入

**日期**：2026-07-27  
**回答用户三点**：  
1) 是否只给 Local5 单开模块、不改 Codex Motion；  
2) Codex 如何把 Motion 做到现在、Local5 如何仿写；  
3) Prosperity / Phi 开源仿真器如何用。

---

## 0. 直接确认（隔离合同）

### 0.1 结论

**是的：Local5 为独立新增模块树，没有改写 Codex 已完成的 Motion 线源码。**

| 目录 | 角色 | 本轮是否修改 Codex 文件 |
|---|---|---|
| `rtl_hitflow/` | GateStack / DCTF / NMF / G1 | **否**（最新 mtime 仍为 07-22 及更早） |
| `rtl_h67/` | SCS row engine / Motion-XOR | **否** |
| `rtl_delta/` | TARE-4 dual-mode / classifier | **否**（07-26 Codex/会话已有；本轮未再改） |
| `rtl_local5/` 叶 | `axnor` / `shiftmax5` / `stencil` | **否**（07-25 Codex 原件只读复用） |
| `rtl_local5/` 新文件 | row_context / MFEP / adapter / banklocal / score_gate_term top | **新增** |
| `tb_local5/` / `sim_local5/` | Local5 专用 TB 与回归 | **新增** |
| `scripts/local5_*` | 双线完成度 / Prosperity 风格仿真 | **新增** |
| `third_party/Prosperity` | 官方克隆 | **新增依赖**（只读使用） |

### 0.2 允许的复用方式

```text
Local5 新 top
  ├── 实例化 rtl_local5 叶（Codex 原件，不改源）
  ├── （可选）实例化 rtl_delta/tare4_* 或 dual_mode_*（不改源）
  └── （可选，后续）对接 rtl_hitflow/gatestack_dctf* 端口（不改源，只适配器）
```

当前 `local5_score_gate_term_top` **尚未**实例化 TARE dual-mode（邻居分走与 leaf 相同的 α-XNOR，保证与 `stencil_token` bit 对齐）。  
TARE 仍作为**共享 score substrate** 在 `rtl_delta` 中由既有 TB 验证；Local5 需要 residual 路径时用**新 wrapper** 去 `调用`，而不是改 dual-mode 本体。

### 0.3 明确禁止

- 不在 `rtl_hitflow/*`、`rtl_h67/*` 上“顺手优化”  
- 不把 Local5 multiplicity 语义直接改写进 NMF set-bitmap  
- 不把 Prosperity 论文里的 mW / 28nm 数字写成我们的 DATE PPA  

---

## 1. Codex 如何把 Motion 做到现在（步骤梯子）

从 `docs/81` → `docs/149` 与 `sim_hitflow/`（约 85 个回归脚本）还原的工程节奏：

```text
L0  算法冻结 / profile100 / ordered workload
L1  叶模块 RTL + pyref/golden + iverilog/verilator
L2  行/组引擎（SCS row engine）+ 周期账本
L3  term IR（NMF/G1 builder）+ 自检 TB
L4  投影后端（decoder → bank-local → DCTF）
L5  真实 trace 向量生成 + equal-lane 对照（Central96 / Ind32 / DCTF）
L6  控制面（lifecycle / abort / residency / serializer）
L7  消融 + DATE 独立审稿文档循环
L8  开放映射代理（Yosys/Nangate）— 明确 ≠ DC
L9  候选增量（PPDI 等）单独叶模块晋级合同，不污染主 top
```

典型证据产物形态（Motion 已有）：

| 产物 | 例子 |
|---|---|
| 叶 TB | `sim_hitflow/run_gatestack_*_checks.sh` |
| 金参考 | `scripts/gatestack_*_reference.py` |
| 实迹向量 | `results/gatestack_*_real_trace_*` |
| equal-lane 周期 | `gatestack_equal96_dctf2c_20260722`（Central 59853 / 1C 62264 / 2C 53910） |
| 中文签核/审稿 | `docs/97`–`149` 多轮 |

**可抄的不是文件名，而是节奏：叶 → 链 → 实迹 → 公平对照 → 文档证据档。**

---

## 2. Local5 仿写进度（对齐 L0–L9）

| 阶 | Motion 对应 | Local5 状态 | 本轮 |
|---|---|---|---|
| L0 profile | profile100 | preG0 profile100 有；**缺 post-G0** | 沿用既有 |
| L1 叶 RTL | Motion-XOR 等 | axnor + Shiftmax5 + stencil（Codex） | 只读 |
| L2 行引擎 | SCS row | **`local5_row_context_engine` ANCHOR+PROBE** | **新增 PASS** |
| L3 term | NMF/G1 | **`local5_mfep_term_builder` multiset** | **新增 PASS** |
| L4 投影 | DCTF | **adapter + 轻量 banklocal**（非完整 DCTF-96） | **新增** |
| L5 实迹 equal | equal96 | **尚未** | 下一优先 |
| L6 控制面 | lifecycle | 仅 stencil_done 握手 | 薄 |
| L7 消融/审稿 | 多轮 DATE | doc 150–158 + **159** | 进行中 |
| L8 开放映射 | Yosys | Verilator 主路径；Yosys 旧前端 partial | 诚实记录 |
| L9 候选 | PPDI | STT/line-buffer 仍为候选 | 未做 |

Verilator 回归：

```bash
./sim_local5/run_local5_parity_checks.sh
# row_context PASS / mfep PASS / score_gate_term_top PASS
```

---

## 3. Idea 包装（可照搬 + 按 Local5 特点改）

| 来源 | Motion 落地 | Local5 落地 | 包装句（可写） | 不可写 |
|---|---|---|---|---|
| Prosperity | TARE 时间 anchor residual | 拓扑 self anchor + PROBE | 静态网络语义锚点消除在线关系搜索 | Prosperity 架构复现 / 其 PPA |
| Bishop | ZERO/SPARSE/DENSE + TTB 动机 | 同 classifier；STT 候选 | exact density routing | Bishop 双核原样 |
| FireFly-T | LIST4 extract | 同 TARE leaf | 有界多 lane 更新 | 原创 decoder |
| FLAT/FuseMax | SCS-NMF-DCTF 融合原则 | Shiftmax5-MFEP-Acc | 不物化 A 的边项投影 | “首次融合” |
| LoAS | T=2 组织 | source-stationary / 三行缓冲（目标） | 驻留策略分叉 | LoAS speedup 数字 |
| SpAtten | exact skip 条件 | zero-gate product skip | cascade 思想局部化 | 独立 scheduler 已实现 |
| Phi | 未采用 codebook | pattern+residual **仅基线 idea** | 两级表示对照 | Phi 仿真器结果（无开源） |
| ECTP 建议 | 差异表/消融结构 | 双线 diff + 五级证据门 | 评估包装 | idea-launder PADE/SATA |
| DCTF | set multicast | **multiplicity 1..5** | multiset reduction | 已完整 2C fabric |

可辩护贡献方向（仍须 cycle/PPA）：

> 同一 exact residual 底座上，Motion 用时间 peer + set term，Local5 用拓扑 stencil + **有界多重集 term**，投影后端共享 bank-local 语义。

---

## 4. Prosperity 开源仿真器

### 4.1 官方仓库

- URL：<https://github.com/dubcyfor3/Prosperity>  
- 本地：`hw_autoresearch_nts07/third_party/Prosperity`  
- 许可：MIT  
- 内容：cycle-accurate Python/CUDA 仿真、Eyeriss/PTB/SATO/MINT/LoAS 基线、CACTI buffer、DSE 脚本  

### 4.2 我们实际用法

| 借用 | 做法 |
|---|---|
| `Stats` 字段 | compute / mem_stall / preprocess / reads / writes |
| 周期合成 | `total ≈ preprocess + compute + max(0, mem−compute)`（可切换 sum） |
| 多基线表 | naive / offset / unsafe-set / MFEP |
| 输出形态 | `time` 与 component 分账 JSON/MD |

| 不借用 | 原因 |
|---|---|
| CUDA product-sparsity kernel | 面向通用 SNN 激活矩阵，不是 Local5 stencil score |
| 论文 on-chip mW 常数 | 是 Prosperity 工艺结果，不是我们的 DC |
| 官方 500 MHz / 28 nm | 只能作时间换算假设，不能作签核 |

运行（Local5 适配仿真，**不改 Prosperity 源**）：

```bash
python3 scripts/local5_prosperity_style_simulator.py
# -> results/local5_prosperity_style_sim_20260727/
```

### 4.3 Phi

检索与既有审计一致：**没有可用的官方开源 Phi 仿真器**。  
Phi 只作为 “pattern + residual 两级格式” 的**对照 idea / 基线描述**，不产生可引用 cycle 数字。

### 4.4 其他可借鉴评估器（候选，未全部接入）

| 工具 | 可借鉴 | 限制 |
|---|---|---|
| STONNE | sparse GEMM 投影对照 | 不表达 Shiftmax/MFEP multiplicity |
| Sparseloop | 稀疏数据流分析 | 需单独映射 IR |
| DRAMsim3 / CACTI | 与 Prosperity 相同的外存/buffer 接口习惯 | 需 workload trace |

---

## 5. Local5 下一阶段（严格仿 Codex 节奏）

**P0（对齐 Motion L5）**

1. 从软件导出 Local5 ordered hardware-order 向量（post-G0 合同冻结后）  
2. `tb` 回放 row→MFEP→cmd 与 pyref bit-exact  
3. 与 Motion equal96 **同 sample keys** 出 cycle/term 表  

**P1**

4. 新文件 `local5_dctf_multiset_bridge.sv`：**只适配**、实例化既有 `gatestack_dctf*`，不改其内部  
5. 三行 line-buffer 叶模块 + STT descriptor（仍在 `rtl_local5`）  
6. Prosperity 风格 simulator 加入 SRAM bank conflict / FIFO 深度敏感性  

**P2**

7. 开放 Nangate 代理（标明 ≠ DC）  
8. 目标工艺 DC/STA/SAIF 有资源再做  

---

## 6. 文件清单（本确认相关）

**只读复用（Codex）**

- `rtl_local5/local5_axnor_score_q7.sv`  
- `rtl_local5/local5_shiftmax5_q17.sv`  
- `rtl_local5/local5_stencil_token.sv`  
- `rtl_delta/*`、`rtl_hitflow/*`、`rtl_h67/*`  

**本侧新增（Local5 / 评估）**

- `rtl_local5/local5_row_context_engine.sv`  
- `rtl_local5/local5_mfep_term_builder.sv`  
- `rtl_local5/local5_mfep_dctf_cmd_adapter.sv`  
- `rtl_local5/local5_banklocal_projection_top.sv`  
- `rtl_local5/local5_score_gate_term_top.sv`  
- `tb_local5/tb_local5_{row_context,mfep_term_builder,score_gate_term_top}.sv`  
- `sim_local5/run_local5_parity_checks.sh`  
- `scripts/local5_motion_parity_cycle_model.py`  
- `scripts/local5_prosperity_style_simulator.py`  
- `third_party/Prosperity/`（官方 clone）  
- `docs/158_*`、`docs/159_*`  
- `results/local5_motion_parity_20260727/`  
- `results/local5_prosperity_style_sim_20260727/`  

---

## 7. 一句话

> **Local5 单开模块 + 只读复用 Codex；评估学 Prosperity 的分账与消融结构，不搬其 PPA 数字；推进节奏对齐 Motion 的叶→链→实迹→公平表，而不是一次性改写主线。**
