# Codex Next Plan — 2026-09-11 Overnight

**Session:** `01a07507`  
**主线唯一：** Stage B — same-port / same-state / same-backpressure 下，ordinary dense-source/raw vs lifting40 `fast_raw_diagonal` 的 schedule compare  
**路径锚点：** `…/fast_temporal_recovery_lifting40/schedule_compare_same_port/`  
**已备输入：** `implementation_inputs.md`、`net_cost_one_page.md`  
**当前数字：** Absolute AEE ≈ **OK (≈1.233)**；Relative **+0.005 FAIL（实测 +0.013）**；service% **UNKNOWN**  
**硬约束：** 不要轻易杀掉 lifting 家族；只停 layouts。Prosperity/Gustav = 第二队列。在 Stage B 裁决前：**禁止** train / quant / RTL / `main.tex`。用户另有 BOX 上 fusion RTL rough probes（iverilog/yosys）——**与 Codex Stage B 分离**；Codex **不得**被 divert 进 Prosperity 长文或 full-chip RTL。

---

## 中文执行纲要（Stage B）

### 步骤 A — 锁定对照条件
1. 确认 compare 目录与入口脚本存在且可复现：`schedule_compare_same_port/`。
2. 核对两边（ordinary dense-source/raw vs lifting40 `fast_raw_diagonal`）满足：
   - **same-port**
   - **same-state**
   - **same-backpressure**
3. 以 `implementation_inputs.md` + `net_cost_one_page.md` 为唯一输入源；不要另开实验维度。
4. 明确度量：Absolute AEE、Relative（相对基线 Δ）、service%（若当前 UNKNOWN，必须在本轮跑出或写明不可得原因）。

### 时间线表（建议顺序，可直接勾选）

| 序号 | 动作 | 产出 | 判据 |
|------|------|------|------|
| A1 | 核对 same-port/state/backpressure 对照是否干净 | 对照 checklist 笔记 | 不一致 → 先修对照，不改算法 |
| A2 | 跑/复跑 schedule compare（ordinary raw vs lifting40 fast_raw_diagonal） | 日志 + 汇总表 | 数字可复现 |
| A3 | 填 Absolute AEE / Relative / service% | 一行结论表 | Abs≈1.233 量级；Rel 相对 +0.005 门槛 |
| A4 | 应用 PASS / FAIL / 灰区规则（见下） | Stage B 裁决 | 见判据 |
| A5 | 若进入 ONE paired recovery（仅一次） | 恢复后二次数字 | 仍 FAIL → 停 layouts，保留 lifting 家族 |
| A6 | 写 Stage B 一页结论；**不得**进入 train/quant/RTL/main.tex/Prosperity 正文 | `STAGE_B_DECISION.md`（或等价） | 主线闭环 |

### PASS / FAIL / 灰区规则
- **PASS：** Absolute AEE 保持可接受（≈1.233 量级 OK）；Relative 相对门槛 **≤ +0.005**；service% 可得且不恶化到不可接受（若仍 UNKNOWN，不得宣称 PASS）。
- **FAIL：** Relative **>+0.005**（当前已报 +0.013）且无正当对照瑕疵；或 Absolute 崩坏。
- **灰区（grey zone）10–15%：** 若 Relative 略超门槛、但落在约 **10–15%** 的“接近可解释/噪声/对照微差”灰带，**且** Absolute 仍 OK、对照条件已核干净——允许进入 **ONE** paired recovery（见下）。超出灰区或对照仍脏 → 直接 FAIL，不恢复。
- **service% UNKNOWN：** 必须补测或文档化“为何测不到”；不得用 Absolute 单独宣布全面 PASS。

### 何时允许 ONE paired recovery（仅一次）
同时满足才允许：
1. 对照条件（same-port/state/backpressure）已确认干净；
2. Absolute AEE 仍 OK；
3. Relative 落在灰区（约 10–15% 可解释带）或存在**单一、明确、可配对**的实现/调度瑕疵（非“再调一堆超参”）；
4. Recovery 是 **一对**（paired）：ordinary 与 lifting40 **同一改动原则**下各跑一次，禁止只偏袒一侧；
5. **全局仅一次**；恢复后仍 FAIL → **停止 layouts**，**保留 lifting 家族**，输出失败分析，转入第二队列排队（Prosperity/Gustav），**不**开 train/quant/RTL/`main.tex`。

### 明确不要做（NOT to do）
- 不要轻易杀掉整个 **lifting 家族**；最多 **stop layouts**。
- 不要把 Codex 主线 divert 到 Prosperity / Gustav 长文、调研综述、full-chip RTL、`nts07/main.tex`。
- Stage B 裁决前：**禁止** train / quant / 生产树 RTL / 改 `main.tex`。
- 用户 BOX 上的 fusion RTL rough probes（iverilog/yosys）是 **独立线**；Codex **不要**接手或展开成芯片级实现。
- 不要为了“好看数字”改对照条件、换端口、换状态、换 backpressure。
- 不要连续多次 recovery / 网格搜索式调参。

### 第二队列提醒
Prosperity / Gustav = **second queue only**。Stage B 未裁决前不启动。

---

## Pasteable English work order (for Codex session 01a07507)

```text
LOCKED CONTEXT — Codex session 01a07507
Main line ONLY = Stage B: same-port / same-state / same-backpressure schedule compare of
  (A) ordinary dense-source/raw
  vs (B) lifting40 fast_raw_diagonal
under …/fast_temporal_recovery_lifting40/schedule_compare_same_port/

Inputs already available (do not invent new ones):
  - implementation_inputs.md
  - net_cost_one_page.md

Current numbers:
  - Absolute AEE ≈ OK (~1.233)
  - Relative vs +0.005 gate: FAIL (reported +0.013)
  - service% = UNKNOWN (must measure or document why unavailable)

Hard rules:
  - Do NOT kill the lifting family lightly; STOP LAYOUTS only if needed.
  - Prosperity/Gustav = second queue ONLY.
  - NO train / quant / RTL / main.tex until Stage B decides.
  - User's fusion RTL rough probes on BOX (iverilog/yosys) are SEPARATE —
    do NOT divert this session into Prosperity essays or full-chip RTL.

STAGE B STEPS
1) Checklist: verify same-port, same-state, same-backpressure for both arms.
   If compare is dirty, fix the compare harness first — do not retune algorithms.
2) Run/re-run schedule compare; produce reproducible Absolute AEE, Relative Δ, service%.
3) Timeline table (fill as you go):
   | Step | Action | Artifact | Gate |
   | A1 | Clean same-port/state/backpressure checklist | notes | dirty→fix harness |
   | A2 | Schedule compare ordinary raw vs lifting40 fast_raw_diagonal | logs+table | reproducible |
   | A3 | Record Abs AEE / Relative / service% | one-line scorecard | Abs~1.233; Rel vs +0.005 |
   | A4 | Apply PASS/FAIL/grey-zone | Stage B verdict | rules below |
   | A5 | At most ONE paired recovery if allowed | second scorecard | still FAIL→stop layouts |
   | A6 | Write Stage B one-pager; do NOT start train/quant/RTL/main.tex/Prosperity | STAGE_B_DECISION | close main line |

PASS / FAIL / GREY ZONE
- PASS: Absolute AEE still acceptable (~1.233); Relative ≤ +0.005; service% available and not unacceptable.
- FAIL: Relative > +0.005 (currently +0.013) with clean compare, or Absolute collapses.
- Grey zone ~10–15%: Relative slightly over gate but within a ~10–15% explainable/noise band,
  Absolute still OK, compare verified clean → may allow ONE paired recovery.
  Outside grey zone or dirty compare → FAIL, no recovery.
- service% UNKNOWN: measure it or document why; do not declare full PASS on Absolute alone.

ONE PAIRED RECOVERY — allowed only if ALL hold:
  (i) compare harness clean;
  (ii) Absolute still OK;
  (iii) Relative in grey zone OR a single clear paired implementation/schedule defect (not a hyperparam sweep);
  (iv) recovery is PAIRED: same change principle applied to ordinary and lifting40;
  (v) globally once only. If still FAIL → stop layouts, KEEP lifting family, write failure note,
      queue Prosperity/Gustav as second line; still NO train/quant/RTL/main.tex.

DO NOT
- Kill lifting family; stop layouts only.
- Divert into Prosperity/Gustav essays, survey digressions, full-chip RTL, or nts07/main.tex.
- Start train/quant/production RTL before Stage B verdict.
- Own the BOX iverilog/yosys fusion probes — that is a separate workstream.
- Change port/state/backpressure to chase nicer numbers.
- Run multiple recoveries or grid-search tuning.

Deliverable: Stage B scorecard + PASS/FAIL/grey decision (+ optional one paired recovery result) + short STAGE_B_DECISION note.
Stay on Stage B until closed.
```

---

## 文件与拷贝
- BOX 原文：`/workspace/overnight_20260911/CODEX_NEXT_PLAN.md`
- 远端副本目标（via zmd SSH → ic.hs.ismd-nemo.xyz as zhumd）：  
  `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/CODEX_NEXT_PLAN.md`

*Generated overnight 2026-09-11 (Asia/Shanghai). Do not wait on 调研. Do not modify nts07/main.tex.*
