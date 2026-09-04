# Local5 功能链追平 Motion 与 Idea 融合

**日期**：2026-07-27  
**范围**：将 Local5 硬件从「叶模块 + TARE 单边」推进到与 Motion 同构的  
`score → gate → term → projection` 功能链，并融合既有论文机制 / ECTP 包装口径。  
**机器可读结果**：`results/local5_motion_parity_20260727/parity_report.json`  
**复现**：`./sim_local5/run_local5_parity_checks.sh`  
**硬约束**：不删除既有 GPT/Codex 代码；开放 Yosys ≠ DC；preG0 profile ≠ post-G0 实迹。

---

## 0. 一句话结论

| 问题 | 答案 |
|---|---|
| Local5 功能链是否已闭合到 Motion 同构？ | **是**（Verilator bit-check PASS） |
| 证据深度是否已追平 Motion equal96 / DCTF-2C？ | **否** |
| 自评完成度 | Local5 **0.35 → 0.78**；Motion 仍约 **0.95** |
| 能否写 accelerator speedup / DC PPA？ | **不能** |

---

## 1. 之前差在哪里

| 环节 | Motion（H67 GateStack） | Local5 本轮之前 |
|---|---|---|
| Score | Motion-XOR + SCS row engine + 实迹 | 叶 axnor；TARE 单 edge 合成 |
| Gate / 归一化 | SCS class fold + Shiftmax | 叶 Shiftmax5 comb |
| Term | NMF/G1 destination bitmap | 仅 profile 统计 MFEP |
| Projection | DCTF-1C/2C bank-local，equal96 周期 | **缺失** |
| 共享残差 | TARE-4 dual-mode | 同左（Local5 mode 已有） |

因此「完成度」差距不是公式，而是 **row-context 协议、多重集 term IR、投影对接** 三条硬件链。

---

## 2. 本轮新增 RTL（功能链）

```text
ANCHOR_LOAD(q,k_self,valid_mask)
    -> PROBE(dir,k_neighbor)*
    -> local5_row_context_engine
         (5x alpha-XNOR + Shiftmax5)
    -> gated edge stream {dest,dir,gate,k,score}
    -> local5_mfep_term_builder
         multiset m[lane,gate] in 1..5
    -> local5_mfep_dctf_cmd_adapter
         cmd + multiplicity sideband
    -> local5_banklocal_projection_top
         Acc[dest,out] += mult * gate * W[lane,out]
```

| 文件 | 角色 |
|---|---|
| `rtl_local5/local5_row_context_engine.sv` | ANCHOR_LOAD / PROBE / RETIRE |
| `rtl_local5/local5_mfep_term_builder.sv` | MFEP 多重集 term |
| `rtl_local5/local5_mfep_dctf_cmd_adapter.sv` | term→DCTF-style cmd |
| `rtl_local5/local5_banklocal_projection_top.sv` | 轻量 bank-local Acc |
| `rtl_local5/local5_score_gate_term_top.sv` | 整链 glue top |
| `tb_local5/tb_local5_*.sv` | 自检 TB |
| `sim_local5/run_local5_parity_checks.sh` | Verilator 回归 |

### 2.1 仿真结果（本轮实测）

| TB | 结果 |
|---|---|
| `tb_local5_row_context` | **PASS** 96 edges / 24 rows，gate/score 对齐 `local5_stencil_token` golden |
| `tb_local5_mfep_term_builder` | **PASS** 44 terms；含 mult=2 重叠 lane 与 zero-gate skip |
| `tb_local5_score_gate_term_top` | **PASS** 515 cmds / 8 dest |

Yosys 0.33 旧 SV 前端仍难以完整吃掉本仓库风格（unpacked array / 块内 `int`）；  
**功能正确性以 Verilator 为准**。开放映射不是目标工艺 PPA。

---

## 3. 与 Motion 的同构 / 异构

| 维度 | 共享 | Local5 特有 |
|---|---|---|
| 残差核 | `tare4_residual_composite_core` / dual-mode top | topology anchor，Q 固定，bias=0 |
| 归一化 | Q7 score + Q1.7 gate 合同 | **5 候选 Shiftmax5**，非 162-token SCS |
| Term IR | (gate, lane, dest…) | **multiplicity 1..5**，禁止 set-OR 去重 |
| 投影后端 | 可对接 DCTF bank-local 语义 | 当前为轻量 Acc + mult 侧带；完整 96-lane fabric 未接 |
| 存储 | — | 目标三行 line-buffer / STT（本轮未做 SRAM） |

---

## 4. Idea 融合表（诚实证据档）

| 来源 / Idea | 如何进 Local5 | 证据档 | 不声称 |
|---|---|---|---|
| Prosperity exact residual | ANCHOR_LOAD + TARE 拓扑 anchor | `[RTL partial]` | 在线 TCAM / product-reuse 原创 |
| Bishop density stratifier | TARE ZERO/SPARSE/DENSE | `[RTL]` classifier | Bishop 全架构 / TTB 原样 |
| FireFly-T multi-lane | TARE LIST4 抽取 | `[RTL]` leaf | decoder/overlay 原创 |
| FLAT / FuseMax fusion | 不物化 A，MFEP→Acc | `[RTL prototype]` | “首次融合 attention” |
| LoAS source-stationary | 数据流方向已写；line-buffer 未做 | `[设计]` | temporal FTP 复现 |
| SpAtten cascade skip | zero-gate 不计 product | `[RTL]` 局部 | 独立 cascade scheduler |
| Phi pattern+residual | 仅作 residual 两级思想 | `[灵感]` | codebook |
| ECTP 包装建议 | 双线 diff + 消融口径 + 不洗 PADE/SATA | `[文档]` | ECTP 加速器已实现 |
| DCTF multiset planes | `cmd_multiplicity` x1..x5 | `[RTL adapter]` | 完整 2C fabric + equal96 |

可辩护表述（仍须 PPA 才能做 DATE 主贡献）：

> 事件光流网络给出的**静态拓扑锚点**驱动 exact residual score，经 Shiftmax5  
> 后以**有界多重集 term** 进入 bank-local 投影；与 Motion 时间锚点共享 TARE  
> 底座，但 term 语义从 set multicast 变为 multiset reduction。

---

## 5. 工作量对照（[prof] preG0，非 RTL cycle）

来自 `local5_hardware_features` profile100：

| 指标 | 数值 |
|---:|---:|
| valid edges | 495,936,000 |
| naive active edge products | 188,373,405 |
| MFEP multicast terms | 13,732,741 |
| MFEP term / naive product | 7.2902% |
| term-count reduction | **92.7098%** |
| topology K-read reduction | 78.0488% |

Motion equal96 **已有** RTL 周期（对照，非本轮 Local5 新测）：

| 架构 | cycles |
|---|---:|
| Central96 | 59,853 |
| DCTF-1C | 62,264 |
| DCTF-2C | 53,910 |
| acc32 match | 233,280 |

**禁止**把 92.7% term 压缩直接写成端到端加速比。

---

## 6. 完成度自评（功能 vs 证据）

```text
功能链（score→gate→term→proj 是否存在可仿真 RTL）
  Motion  ████████████████████  0.95
  Local5  ███████████████░      0.78   ← 本轮

证据深度（实迹 / equal-lane cycle / DC）
  Motion  ████████████████████  0.90+
  Local5  ██████░░░░░░░░░░░░░░  0.35   ← 仍远
```

本轮追平的是 **功能链形状与 idea 落点**，不是 DATE 投稿证据包的全部厚度。

---

## 7. 仍未完成（按优先级）

1. **P0** Local5 post-G0 / G1 后 ordered real-trace profile（替换 preG0 数字）  
2. **P0** MFEP multiplicity plane 接入完整 `gatestack_dctf96_*`（非轻量 Acc）  
3. **P1** 三行 K line-buffer + STT descriptor FSM  
4. **P1** 与 Motion equal96 同 sample 的 cycle / term 对照表  
5. **P2** 目标工艺 DC/STA/SAIF；Yosys 仅代理  
6. **P2** SATF 统一顶层（Motion TAB + Local5 STT）  

---

## 8. 对 DATE / 双线策略的影响

- **主线仍是 Motion**：equal96 + DCTF-2C 证据更完整。  
- **Local5 从「切线草稿」升级为「功能链候选」**：row-context + MFEP 已不再只是文档。  
- 投稿叙事应写 **Semantic-Anchor 双前端 + 共享 TARE + 分叉 term（set vs multiset）+ 可共享 DCTF 后端**，并逐条标证据档。  
- ECTP/PADE 只吸收 **对照实验结构与差异表**，不做 idea-laundering。

---

## 9. 复现命令

```bash
cd hw_autoresearch_nts07
./sim_local5/run_local5_parity_checks.sh
python3 scripts/local5_motion_parity_cycle_model.py
# 既有双线 profile / TARE
# ./sim_delta/run_delta_bounded_classifier_checks.sh
```

---

## 10. 变更文件清单

**新增**

- `rtl_local5/local5_row_context_engine.sv`
- `rtl_local5/local5_mfep_term_builder.sv`
- `rtl_local5/local5_mfep_dctf_cmd_adapter.sv`
- `rtl_local5/local5_banklocal_projection_top.sv`
- `rtl_local5/local5_score_gate_term_top.sv`
- `tb_local5/tb_local5_row_context.sv`
- `tb_local5/tb_local5_mfep_term_builder.sv`
- `tb_local5/tb_local5_score_gate_term_top.sv`
- `sim_local5/run_local5_parity_checks.sh`
- `scripts/local5_motion_parity_cycle_model.py`
- `results/local5_motion_parity_20260727/*`
- `docs/158_Local5功能链追平Motion与Idea融合_20260727.md`

**更新**

- `rtl_local5/filelist.f`

**未修改**既有 Motion GateStack / TARE 主路径文件（只复用）。
