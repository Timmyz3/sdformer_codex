# Motion/H67 quotient-file 侧车：⑤ SAIF/PPA 最终锚定实验 runbook（新思机器）

日期：2026-08-18。证据分级沿用项目惯例：`[rtl]` / `[prof]` / `[模型]`；模型数字不冒充周期。
红线：本 runbook 为纯 CPU 文档（本机不跑新思、不碰 GPU）；不修改任何现有文件；
`docs/359`、selector、生产 RTL、194436Z 包不动。GPU 实验只排入队列，不执行（GPU 被 D1 short 训练占用）。

冻结输入（写侧车 RTL 之前的前置 ⑤，全部只读引用）：

| 输入 | 来源 |
|---|---|
| 设计文档 | `docs/CLAUDE_MOTION_SIDECAR_DESIGN_20260818.md`（证据 ③④ 的母文档） |
| 容量/spill 冻结 | `docs/CLAUDE_MOTION_SIDECAR_CAPACITY_PORTS_20260818.md`（证据 ③：count 文件 163×9=1,467 bit 必须计入两侧） |
| 端口契约冻结 | `docs/CLAUDE_MOTION_SIDECAR_PORT_CONTRACT_20260818.md`（证据 ④：目录信号级契约、8 项 perf 单源、`SIDECAR_DIRECTORY` 参数点） |
| golden 证据 ①+② | `results/motion_sidecar_golden_evidence_20260818/`（golden.py / report.json / report.md） |
| 现网新思流程 | `dc_handoff/SERVER_RUN.md`、`dc_handoff/run_dc.sh`、`run_formality.sh`、`run_ptpx.sh`、`run_ptsta.sh`、`PPA_REVIEW_CHECKLIST.md` |

实验身份：**组件动态能量（同端口 SAIF）≥15% / 逻辑+宏面积 ≤+10% / Fmax ≥−5%**，
corner TT/SS/FF，fakeram45 + Nangate45，时钟按 `date_dual_core.sdc`（默认 3.0 ns，可经
`CLOCK_PERIOD_NS` 覆盖；与 docs/264 的 5 ns 口径对比时用 5.0）。结果标签按
`PPA_REVIEW_CHECKLIST.md`：无宏 db = `pre-macro logic DC/STA/PTPX`；有宏 db 无 P&R =
`post-synthesis macro-aware estimate`；有同 run P&R 网表/SPEF = `post-layout estimate`。

---

## 1. 实验身份与两侧对象

### 1.1 两侧顶层（design name 冻结）

| 侧 | 顶层 | 配置 | 说明 |
|---|---|---|---|
| 基线 | `h67_rqtb2s_mssb5_dc_top`（现网，已有 filelist/端口封装） | `QUOTIENT_ENABLE=1, MSSB5_SCORE_FRONT=1, MEMORY_IMPL=1`（PPA_ADMISSION 时） | 现网 C7 目录（含 `class_hist` = 现网版 per-class count 对象，163×9 同宽） |
| 侧车 | `h67_sidecar2s_mssb5_dc_top`（**预留名**，RTL 按证据 ③④ 冻结合同落位后替换为实际名字） | `SIDECAR_DIRECTORY=1`（Mode A，default-on 实例），其余参数同基线 | 侧车目录 = occupancy bitmap(163 bit flop RF) + descriptor 流(450×11→fakeram45_256x32 单宏 2 条/字) + denominator certificate(14 bit flop) + **count 文件(163×9=1,467 bit)** |

- 侧车 wrapper 放 `dc_handoff/rtl/date_sidecar_dc_top.sv`，filelist 为
  `dc_handoff/filelists/date_motion_sidecar.f`（= `date_motion_2s.f` 全文件 +
  sidecar 目录 RTL + wrapper）。**未写 RTL 前不得跑本 runbook 的 Phase 2–5 侧车侧**；
  本 runbook 的 Phase 1（SAIF 准备）与基线侧不依赖侧车 RTL。
- 对照规格硬约束：两侧**同标准单元库、同宏库、同时钟、同角点、同测试向量、同反压 seed、
  同 SAIF 测量窗**（`SERVER_RUN.md` §2 同身份要求 + 证据 ③ §3.4 记账表）。

### 1.2 组件能量边界（冻结，两侧同口径）

计入两侧（SAIF 组件边界）：
- score 前端（`h67_mssb5_temporal_slot_encoder` / `h67_motionxor_score_q7`）；
- 目录层（基线：`h67_temporal_weighted_scs_directory_2s`；侧车：`h67_quotient_sidecar_directory`
  + 四个新对象 `occupancy_bitmap` / `descriptor_stream` / `denominator_certificate` / `count_file`）；
- **count 对象两侧都计入**：基线侧 = 现网 `class_hist`（163×9，每 descriptor 1 次 RMW）；
  侧车侧 = count 文件（163×9，同 RMW 语义）。两对象同构对消，但**必须显式列在
  `ptpx_power_hierarchy.rpt` 的读表项里，不许只报目录差分**（证据 ③ §2.4 硬约束）；
- descriptor 存储（基线：`h67_banked_active_descriptor_store` MEMORY_IMPL=1 的 2×
  fakeram45_256x32，28-bit banked；侧车：1× fakeram45_256x32 单宏 2 条/字，32-bit
  整字口径含 padding 写活动——两侧写活动按 **32-bit 整字口径**一致记账，证据 ③ §2.2/§3.4）。

共享、两侧同一 netlist 块、比较时排除（排除清单写进 PTPX 汇总脚本，两侧同一份清单）：
`h67_sync_dual_bank_k_store`、slot FIFO、Shiftmax/emit 后端、out 边界、perf 计数（top 层）。

主张口径：**组件动态能量 = (E_基线组件 − E_侧车组件) / E_基线组件 ≥ 15%**（同 corner TT 25°C 标称、
SS/FF 敏感复核）；同时必须给出组件/整行边界占比（动态能量/总 row 动态能量），诚实披露
组件腿不是整行腿、不是 encoder 腿（设计文档 §4.2，docs/262 口径）。

### 1.3 面积的记账对象（两侧同端口模型）

builder metadata、descriptor SRAM（侧车 1 宏 vs 基线 2 宏）、denominator certificate、
occupancy bitmap、count 文件（基线 `class_hist` 同款）——**全部计入，不许只报目录差分**
（设计文档 §4.3 / 证据 ③ §3.4）。K store / FIFO / emit 与基线同一实例，两边都有，面积对比
随总面积自然抵消，但总面积必须含。

---

## 2. 前置检查

### 2.1 拷包内容清单（sidecar supplement 包）

基础包（现成，SHA 已冻结在 `packs/COPY_THIS.md`）：
`date_dual_synopsys_handoff_20260814T194436Z.tar(.sha256)`，SHA256 =
`ff986c74070e39f2effe24494f911490dbc896036b798599ebf525779a1f6ebc`。
解包到 `hw_autoresearch_nts07/` 后，两道门必须 PASS（`SERVER_RUN.md` §0）：
`audit_date_dual_handoff.py`、`audit_three_line_predc_gate.py`；`docs/359` SHA 必须仍是
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

sidecar supplement 包（本 runbook 生成时列出，执行打包时新建 tar，**不改基础包**）：

| 文件 | 用途 | 本机 SHA256（身份锚） |
|---|---|---|
| `dc_handoff/CLAUDE_SIDECAR_SAIF_PPA_RUNBOOK_20260818.md` | 本 runbook | 打包时记录 |
| `docs/CLAUDE_MOTION_SIDECAR_DESIGN_20260818.md` + ③④ 两文档 | 冻结合同 | 打包时记录 |
| `results/motion_sidecar_golden_evidence_20260818/golden.py` | 证据 ①+② 重建脚本 | `dbd875765a79c8d739b8daa0468b85b553097d169ed852852174b9de191e686d` |
| `results/motion_sidecar_golden_evidence_20260818/report.json` / `report.md` | 证据 ①+② 结果 | `a43f8c69c521028f304b9df098b22ae63fb3b1ceefff8a7d1a7f93381837799e` |
| `dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/`（VCD + activity_contract.json） | 基线 SAIF 输入 | VCD `544c56d25242e3fd…`（contract 内全 SHA） |
| `dc_handoff/runs/motion_fixed_dc_activity_population138_fair/`（同上） | 对照（只读上下文，不参与两侧对比主判） | VCD `ff4c934792f56065…` |
| `tb_h67/vectors/h67_fullres_ep35_postconvergence_t450_20260805/h67_checkpoint_rows.txt` | 公平包向量 | `0ad20f73bfaa821b…` |
| 侧车活动 VCD + contract（Phase 0 产出，含 138 行公平包侧车侧与 100 样本分层窗口两侧） | 侧车 SAIF 输入 | Phase 0 产出时记录 |

打包动作 = 新 filelist + `tar`（参照 `pack_synopsys_handoff.sh` 结构，不修改基础包与既有脚本）。

### 2.2 本机生成物 SHA（服务器解包后逐项校验）

| 冻结输入 | SHA256 |
|---|---|
| `nts11_hardware_p0_profile.json`（ep35 profile100） | `7b77f666188e09f6ed30620dad301032d92b62d7c450d10a0465dbc4eb4b3d72` |
| `h67_checkpoint_rows.txt` | `0ad20f73bfaa821b4e7b9e4048ae1b25a564d860eb00efa1efac97641010adc2` |
| `ep35_fair_merge.log` | `6a7bf926440afcfff102609b63d74380249a37f45ca5f39add4a3fd81cde6b41` |
| `checkpoint_epoch35.pth`（GPU 实验用） | `4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158` |
| `docs/359_DATE终局冻结_20260813.md` | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |
| 基线 RQTB VCD | `544c56d25242e3fde76e4f19cd0865613cee0eace82f4d00fdeb83f281a0a8ef` |

校验命令：`sha256sum -c <archive>.tar.sha256`（tar 与 sha 文件同目录执行）→ 解包 →
对 2.2 表逐项 `sha256sum`。

### 2.3 新思机器环境要求（服务器自备，本包不提供）

| 需求 | 用途 | 缺失时的行为（真失败，不许造假） |
|---|---|---|
| `dc_shell`（DC Ultra 许可）+ Nangate45 标准单元 `.db` | 综合 | `run_dc.sh` 非零退出（exit 3/4） |
| `fm_shell` | Formality | `run_formality.sh` 退出；无 PASS 不可进下一阶段 |
| `pt_shell`（含 PX 许可） | STA / PTPX | `run_ptpx.sh`/`run_ptsta.sh` 退出（exit 3） |
| `vcd2saif` | VCD→SAIF | 走 §10.1 回退（宏模型 [模型]），**不生成假 SAIF** |
| 可选：`fakeram45_256x32.db` + adapter（`MACRO_DBS`+`EXPECTED_MACRO_REFS`） | PPA_ADMISSION=1 | 无则保持 pre-macro，**不许猜宏名** |
| 可选：布局布线工具（ICC2 或 Innovus）+ StarRC | post-layout | 无则按 `SERVER_RUN.md` 现有 pre-macro 流程，标签如实 |
| 磁盘/运行时间估计 | — | 每顶层每角 DC 约 20–60 min、PTPX 约 10–30 min；VCD/SAIF 数个 GB；两侧 3 角合计预留 ≥30 GB、≥8 CPU 核时 |

运行目录约定：基线 `dc_handoff/runs/h67_rqtb2s_mssb5_dc_top/`；侧车
`dc_handoff/runs/h67_sidecar2s_mssb5_dc_top/`；**禁止复用另一侧的 DC_RUN_DIR/PT_RUN_DIR**。

---

## 3. Phase 0（本机 CPU，打包前）：测试向量与活动 VCD

两侧共用同一组测试向量、同一反压机制（`tb_h67_motion_dc_activity.sv` 的 LFSR
`backpressure_lfsr`，seed `16'h1d3f`，`descriptor_issue_enable/out_ready` 同现网；
现网 VCD 契约见 `SERVER_RUN.md` §2 与 `report_activity_vcd.py`）。向量分两层：

### 3.1 向量层 1：138 行公平包（主向量，现成）

- 源：`h67_checkpoint_rows.txt`（138 行 × 450 token 的 q/k/peer/gate 冻结向量，锚定
  1.1865× 的同一包，golden `[rtl]` 账本已 0-mismatch）。
- 基线侧 VCD 已存在：`dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/`，
  `strip_path=TOP/tb_h67_motion_dc_activity/g_rqtb/dut`，busy=94891（contract PASS）。
- 侧车侧 VCD（**新增**，侧车 RTL 就绪后）：TB 增加 `g_sidecar` 分支
  （实例化 `h67_sidecar2s_mssb5_dc_top`，`SIDECAR_DIRECTORY=1`），同向量、同 LFSR、
  同 `DUMP_ROWS` 参数回放；`strip_path=TOP/tb_h67_motion_dc_activity/g_sidecar/dut`。
  终态行 `PASS Motion wrapper activity mode=sidecar rows=138 …` 且 `protocol_error` 恒 0。
- 检查点：`report_activity_vcd.py` 产出 activity_contract.json，`status=PASS`，
  8 项 perf 与基线 VCD 同口径（`perf_quotient_descriptors=34099` 等逐项相等）。

### 3.2 向量层 2：100 样本分层窗口（增强向量，规格）

- 源：`results/h67_ep35_multisample100_t450_real_rtl_bit_trace/`（1200 个 npz
  `sample{N}_S{stage}_B{block}_attn.npz` + manifest.json，记录级真实 q/k bit trace）。
- 抽样规格（固定 seed=20260818，纯统计重放 `[prof]`）：按 stage 行数占比
  （S0 39.3% / S1 21.4% / S2 32.1% / S3 7.1%，frozen canonical）分层抽 **100 个
  (window,head) 行**：S0 39 / S1 21 / S2 33 / S3 7；强制覆盖 S3 深尾（D>324）≥5 行、
  C≥16 密行 ≥10 行（深尾 E/P=0.4458 是最坏 work 点，必须进 SAIF 窗）。
- 产出：`h67_sidecar_window100_vectors.txt`（与 `h67_checkpoint_rows.txt` 同格式，
  行头 + 450×(q,k,peer,gate)）；gate 列用 golden Mode A 全量 0-mismatch 链重算值。
- 回放：`tb_h67_motion_dc_activity`（`MAX_ROWS=100` 参数档）两侧各跑一遍
  （RQTB/侧车），每行单独 dump，VCD 区间 = 100 个 row 区间的并（activity contract
  的 `start_group/groups` 语义不变）。
- 自检门（不过不出包）：`activity_contract.json status=PASS`；重放的 D/C/E–P 统计与
  frozen canonical 对照：D mean 230.66 ± 2%、D p95=265 ± 5%、C p95=16、E mean 219.34
  （证据 ② 表）；终态 `PASS` 且两侧 `protocol_error` 恒 0。
- 磁盘估计：VCD 每侧约 5–15 GB（100 行 × 平均 ~700 cycle × 层次信号），打包前确认。

### 3.3 VCD 身份链

每份 VCD 配一份 `activity_contract.json`（`report_activity_vcd.py` 生成，含
`source_vcd_sha256`、`measured_cycles`、`busy_cycles`、`paper_population_totals`、
`strip_path`、`clock_period_ps`）。两侧 contract 必须同 `trace_sha256`、同
`paper_population_totals`（slots/equal 与 frozen 列一致）、同 `strip_path` 前缀仅
`g_rqtb`/`g_sidecar` 分支名不同。

---

## 4. Phase 1（服务器）：VCD → SAIF

两侧每份 VCD 转一份 SAIF（本机不转，`vcd2saif` 在服务器）：

```bash
# 基线（RQTB，C7 目录）
vcd2saif \
  -input dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/h67_rqtb2s_mssb5_dc_top.vcd \
  -output dc_handoff/runs/h67_rqtb2s_mssb5_dc_top/c7_rqtb_138row.saif
python3 dc_handoff/scripts/make_saif_manifest.py \
  --root . \
  --activity-contract dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/activity_contract.json \
  --saif dc_handoff/runs/h67_rqtb2s_mssb5_dc_top/c7_rqtb_138row.saif \
  --output dc_handoff/runs/h67_rqtb2s_mssb5_dc_top/c7_rqtb_138row_saif_manifest.json
# 侧车（Mode A default-on，RTL 就绪后）
vcd2saif \
  -input dc_handoff/runs/motion_sidecar_dc_activity_population138_fair/h67_sidecar2s_mssb5_dc_top.vcd \
  -output dc_handoff/runs/h67_sidecar2s_mssb5_dc_top/sidecar_138row.saif
python3 dc_handoff/scripts/make_saif_manifest.py \
  --root . \
  --activity-contract dc_handoff/runs/motion_sidecar_dc_activity_population138_fair/activity_contract.json \
  --saif dc_handoff/runs/h67_sidecar2s_mssb5_dc_top/sidecar_138row.saif \
  --output dc_handoff/runs/h67_sidecar2s_mssb5_dc_top/sidecar_138row_saif_manifest.json
```

100 样本分层窗口层同样处理（后缀 `_window100`）。SAIF 身份由 manifest 绑定
（`audit_saif_manifest.py --require-paper-power-eligible` 在 run_ptpx.sh 内强制）。
检查点：manifest 里 `source_vcd_sha256` 与 2.2 表一致；SAIF 覆盖的层次前缀 = strip_path。

---

## 5. Phase 2（服务器）：DC 综合 + Formality（两侧，TT 25°C 标称）

两侧各自独立 run 目录。基线：

```bash
export LIB_DB=/path/to/tt_0p9v_25c.db
export OPERATING_CONDITION=tt_0p9v_25c          # 以服务器库实际 OC 名为准并记录
export CLOCK_PERIOD_NS=3.0                       # 探索默认；与 docs/264 对比时改 5.0
export DESIGN_NAME=h67_rqtb2s_mssb5_dc_top
export DC_RUN_DIR=dc_handoff/runs/h67_rqtb2s_mssb5_dc_top
dc_handoff/run_dc.sh
dc_handoff/run_formality.sh
```

侧车（RTL 就绪后，`DESIGN_NAME=h67_sidecar2s_mssb5_dc_top`、独立 `DC_RUN_DIR`、filelist 换
`date_motion_sidecar.f`、其余环境变量同）。有宏 db 且宏名审计通过时才加：

```bash
export PPA_ADMISSION=1
export MACRO_DBS=/path/to/fakeram45_256x32.db
export EXPECTED_MACRO_REFS=u_sidecar_descriptor_stream,u_baseline_active_store_a,u_baseline_active_store_b
# EXPECTED_MACRO_REFS 按两侧实际实例名填，不许猜；无 adapter 保持 pre-macro
```

检查点（DC）：`audit_dc_artifacts.py` 通过；`check_design_postcompile.rpt` 无
unresolved reference/multiple driver/comb loop；`check_timing_postcompile.rpt` 无
unconstrained；`references.rpt` 宏实例与 `EXPECTED_MACRO_REFS` 一致（PPA_ADMISSION=1 时）；
`reports/area.rpt`（面积基线）、`reports/qor.rpt`、`reports/timing_setup.rpt`（WNS）。
检查点（Formality）：`reports/formality_status.txt` 精确 `PASS`。
侧车 `SIDECAR_DIRECTORY=0` 实例化一次作 FM 锚：与现网 netlist 逐叶一致（构造性质，
FM PASS 复核默认关闭路径的 bit-exact）。

---

## 6. Phase 3（服务器）：布局布线 + StarRC 寄生抽取

有 ICC2 或 Innovus 许可时执行（无则跳过本节，结果标 pre-macro，进 §8 用无 SPEF 的
PTPX，标签 `pre-macro logic DC/STA/PTPX`）。以 ICC2 为例（Innovus 命令等价替换）：

```bash
# ICC2：读入 DC 网表 + 约束 + 宏物理库（fakeram45 LEF），place → clock tree → route
icc2_shell -f scripts/sidecar_pr.tcl    # 每侧一份：
#   create_lib/read_verilog mapped.v/read_sdc mapped.sdc → place_opt → create_clock_tree
#   → route_opt → write_verilog <run>/pr/<design>_pr.v → write_sdc → write_parasitics -output <run>/pr/<design>.spef
# StarRC（同一版图，同 netlist 同层）：
starRC -cmd_file scripts/sidecar_starrc.cmd   # extract_rc → write_parasitics(SPEF, NOM/TT)
```

检查点：P&R 后时序报告（WNS/TNS，post-route）；StarRC 抽取报告与 P&R 网表
**同一 run 同顶层**（`audit_synopsys_postrun.py` / `ptsta_scope.rpt` 必须
`extracted_spef`）；SPEF 与产生它的 netlist 的 SHA 绑定（`SPEF_FILE` 与
`NETLIST_FILE` 同时传给 PT，禁止混配，`run_ptsta.sh`/`run_ptpx.sh` 已强制）。

---

## 7. Phase 4（服务器）：PTSTA（TT/SS/FF，setup + hold 独立 run 目录）

每侧每个角独立 `PT_RUN_DIR`，禁止覆盖。基线示例：

```bash
# TT setup（post-layout 用 PR 网表+SPEF；pre-macro 用 DC 网表无 SPEF）
LIB_DB=/path/to/tt_0p9v_25c.db OPERATING_CONDITION=tt_0p9v_25c \
CORNER_ROLE=setup PT_RUN_DIR=dc_handoff/runs/h67_rqtb2s_mssb5_dc_top/pt_tt_setup \
NETLIST_FILE=<pr 网表或 DC 网表> SPEF_FILE=<pr spef，P&R 后> \
dc_handoff/run_ptsta.sh
# SS setup（最坏）：ss_0p9v_125c；FF hold（最坏）：ff_0p9v_neg40c
# TT hold / SS hold / FF setup 按需各立 run 目录
```

角点表（OC 名以服务器库为准并写进报告）：**TT 25°C（能量标称 + 主 Fmax）**、
**SS 0p9v 125°C（setup 最坏）**、**FF 0p9v −40°C（hold 最坏）**。
检查点：每角 `ptsta_check_timing.rpt` 无 unconstrained/error；setup 角 WNS/TNS ≥0；
hold 角 hold WNS/TNS ≥0；`ptsta_scope.rpt` 注明 extracted_spef 或 zero-annotated。

---

## 8. Phase 5（服务器）：PTPX 能量（三 corner，SAIF 反标）

每侧三角各一次，SAIF 用该侧自己的 VCD（同向量、同窗）。基线示例：

```bash
SAIF_FILE=dc_handoff/runs/h67_rqtb2s_mssb5_dc_top/c7_rqtb_138row.saif \
SAIF_INSTANCE=TOP/tb_h67_motion_dc_activity/g_rqtb/dut \
SAIF_MANIFEST=dc_handoff/runs/h67_rqtb2s_mssb5_dc_top/c7_rqtb_138row_saif_manifest.json \
CORNER_ROLE=power MIN_SAIF_COVERAGE_PCT=95.0 \
LIB_DB=/path/to/tt_0p9v_25c.db OPERATING_CONDITION=tt_0p9v_25c \
DC_RUN_DIR=dc_handoff/runs/h67_rqtb2s_mssb5_dc_top \
dc_handoff/run_ptpx.sh
# SS / FF 角：换 LIB_DB/OPERATING_CONDITION 重复，SAIF/DC_RUN_DIR 不变
# 侧车：SAIF_INSTANCE=TOP/tb_h67_motion_dc_activity/g_sidecar/dut，DC_RUN_DIR 独立
```

检查点：`ptpx_check_power.rpt` 无 error；`ptpx_unannotated.rpt`（
`report_switching_activity -list_not_annotated`）中 DUT 未注释对象为 0 且覆盖率 ≥95%；
`ptpx_power_hierarchy.rpt` 按实例子树列出 dynamic power（switching+internal 分项）。

---

## 9. 判定表与结果读取

三项门全部基于**同向量（138 行公平包主判 + 100 样本分层窗口复核）、同角、同宏、
同反压**的两侧对比。`E_基线`/`E_侧车` 记为同角组件动态能量（TT 为主判角，SS/FF 复核
符号与幅度），面积记 TT 综合 + P&R 后数字，Fmax 记各角 WNS 导出。

| 门 | 读哪个文件、哪个数 | 判定 | 不达标动作 |
|---|---|---|---|
| 组件动态能量 ≥15% | 两侧 `ptpx_power_hierarchy.rpt` 组件边界实例子树 dynamic power 求和（含 count 对象；共享块清单两侧同款排除）；`E_reduction = (E_基线−E_侧车)/E_基线` | `E_reduction ≥ 0.15`（主判 138 行包；window100 层 `≥ 0.10` 为弱化提示，不后验改门） | §9.4 降级 |
| 面积 ≤+10% | `reports/area.rpt`（DC total cell area）+ `references.rpt`/P&R 宏面积，两侧同端口模型（含 builder metadata/SRAM/certificate/occupancy/count） | `(A_侧车−A_基线)/A_基线 ≤ 0.10`（TT） | §9.4 降级 |
| Fmax ≥−5% | 同角 `timing_setup.rpt` WNS：`Fmax = 1000/(CLOCK_PERIOD_NS − WNS)`（WNS<0 时用 `1000/CLOCK_PERIOD_NS` 作上界），两端口径同 docs/264 RQTB2S 5 ns post-route WNS +0.0686 ns | `(Fmax_侧车/Fmax_基线 − 1) ≥ −0.05`（TT 主判；SS 复核同号） | §9.4 降级 |

辅助（如实报告，不合成 EDP）：周期腿 +0.7%（450+C+desc=696 vs 450+C+PAIRS=691，
证据 ① 口径）作 "持平（non-negative）" 陈述，**不与 energy 合写 EDP**（设计文档 §2.3/
§5 杀 2；证据 ④ §5）。

### 9.4 降级路径（预注册，不后验改门）

1. **HOLD_AS_IMPLEMENTATION**（energy 或 Fmax 任一不过）：侧车 Mode A 仍保留
   "default-off、bit-exact by construction" 的实现身份（`SIDECAR_DIRECTORY=0` 与现网
   逐叶一致，构造性质不依赖本实验），但论文**不主张能量腿**；SAIF/PPA 结果照实落档
   为组件能量实测值（可作表格上下文列），措辞回到证据 ③ §2.4 的诚实重述。
2. **只报相对 C7 物化 −41%**（存储对象口径）：p95 保守点 (16,265) 下含 count 文件的
   侧车存储账（qg'=4,689 / do'=4,559）相对现网 C7 物化 7,957 = **−41.1%**（证据 ③
   §2.4 表），作为上下文列写进论文；`[模型]` 分档，不冒充周期/能量。
3. **门不过时的口径纪律**：`desc/token p95>0.60` 未来样本触发时，能量主张自动降为仅
   storage bit 账（设计文档 §5 杀 3(c) 预注册规则）；一切降级不回头改门。

---

## 10. 风险与回退

### 10.1 SAIF 不可用 → 宏模型能量估算（[模型]，不作 ⑤ 门证据）

触发条件：服务器无 `vcd2saif`、或 VCD→SAIF 许可缺失、或 SAIF 覆盖率 <95% 且无法修复。
替代口径（结果必须显式标注 **[模型]**，与 SAIF 实测分开成表）：

- 对象级写活动账（本机 CPU 可做，冻结输入已齐）：每窗写位宽差 =
  侧车「occupancy 置位（C≤163 位）+ descriptor 流 1 次宏写/对（32-bit 整字，225 字
  ≤1 事务/对）+ count 文件 RMW（每 descriptor 1 次）」vs 基线「450 带分名单 +
  token-gate 物化写（450×9）+ active pair store 28-bit×≤2 写/对（56 bit）+ class_hist
  RMW」；exp 事务用现网 −22.04% VCD 观测（docs/263 §3.3）。
- 能量系数：fakeram45 宏单位写能量 × 写位宽差 + 逻辑部分按现有 pre-macro DC
  power 报告反推 per-gate 系数 × 门计数差。**系数本身也是 [模型]**。
- 结论语义：方向性支撑（写活动位宽差为能量腿的机制载体），**不构成 ≥15% 的门证据**；
  论文只能写"SAIF 不可用，能量主张未实测，以存储位账 + 宏模型方向支撑"。

### 10.2 其他失败与纪律

- 缺 `dc_shell`/库/宏 db：脚本非零退出（`SERVER_RUN.md` §4 真失败原则），**禁止**
  用默认翻转率凑功耗、禁止伪造 SAIF。
- 无 P&R/StarRC：跳过 §6–7 的 SPEF 部分，PTPX 无寄生跑，标签
  `pre-macro logic DC/STA/PTPX` 或 `post-synthesis macro-aware estimate`（按 2.3 表）。
- 侧车 RTL 未按时落位：本 runbook 的基线侧（§4–§9 中 `h67_rqtb2s_mssb5_dc_top` 部分）
  可先行；侧车侧全部阶段以 RTL 冻结为前置，不空跑。
- 任一审计脚本非 PASS、`docs/359` SHA 漂移：停止并把结果作废，不允许带病进论文。

---

## 11. Mode B（one-vote）精度评估实验规格（GPU，只排队列不执行）

### 11.1 为什么必须过完整网络

golden.py 的 Mode B 路径只重建 **672,000 行 Q7 分数上的门级差异**（纯 Python/CPU）：
100% 窗口 Z_B≠Z_C7、Δdenom_shift 全负（p50=−7，range [−9,−3]）、gate 差 Δ=0 占
96.86%（条目级 p99=148/max=180）、发射翻转（截断模型 `[模型]`）t=64 零行、
t=128 全行全 token（302.4M）。**这些是分母/门对象层面的差异，不是 AEE**——AEE 影响
必须把 Mode B 门值送进完整 Swin 网络做 valid825 推理，本机是 GPU 任务（评估走
`eval_DSEC_flow_SNN.py`，batch=1，A800 80GB），排入队列、当前不执行。

### 11.2 实验规格

| 项 | 规格 |
|---|---|
| 目的 | 评估 one-vote 数值合同（Mode B）在完整网络上的 AEE 影响，裁决 Mode B 能否与 H82/H86 算法线接线（侧车不自行裁决，设计文档 §3/§6） |
| checkpoint | `results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth`（SHA `4f33e086…`，AEE 1.3297 float / 1.3287 hardware-order，P0-1 已落档） |
| 评测协议 | 与 P0-1 完全一致：`eval_DSEC_flow_SNN.py --config <hardware_order_q7q17_deploy.yml> --checkpoint ep35 --path_results <out> --mode valid`，batch=1，825 帧/18 序列；config = `configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml`（执行时记录 config SHA） |
| 改动面（唯一差异） | hardware-order 量化路径中**分母仅改为 one-vote**：`Z_B = Σ_{occupied class} exp_q8(c − row_max)`（禁乘 multiplicity），denom_shift=ceil_log2(Z_B)，gate 取整位置/exp LUT/row_max 与现网逐位相同；其余（score Q7、K、gated-K、threshold 截断配置、输出）一概不动 |
| 纪律 | 不 overlay `bsa_attention.py`（磁盘 SHA 66d0a339… 校验；若已变化走 shadow 字节拷贝 + SHA 记录，参照 2026-08-18 P1-4 先例）；新增执行文件按 `run_h67_fullres_ep35_ep40_deploy_q7q17_20260818.py` 模板（该文件只读引用），GPU 执行时新建，不预先写盘 |
| 输出 | `deploy_valid825/mode_b_onevote_q7q17/epoch35/`（spike_profile.json：AEE/AAE/Fl/spikes_G/energy_uj）+ 汇总 json/md（对比 C7 hardware-order 1.3287 与 float 1.3297） |
| 判定阈值 | **AEE_ModeB ≤ 1.3297 × 1.01 = 1.3430**（退化 <1%）⇒ Mode B 可用；同时记录 spikes_G 与 energy_uj 变化（gate 变大可能扩大发射/spikes，如实披露，不作为门） |
| 预计时长 | ~17–20 min GPU（P0-1 单 run 先例）+ 汇总后处理 ~5 min，共 <0.5 h |
| 证据分档 | GPU 数值 = `[模型]`+`[prof]`（非 RTL）；Mode B 硬件 gate 边界 bit-exact 属写 RTL 后的 miter 任务（设计文档 §4.4），本实验不裁决 |

### 11.3 队列位置（只登记，不执行）

GPU 当前被 **D1 short 训练**占用（CLAUDE_SCORE_20260818_1830）。本实验入队位置 =
**D1 short → D3 short → 本实验（Mode B one-vote valid825）**；遵守
`CLAUDE_ALGORITHM_CONTRACT_QUEUE_20260818.md` 纪律：一次一个 GPU 任务、与 DATE 缺口
审计 agent 协调、长任务先写状态文件（本实验 <0.5 h 无需）。前置不满足（D1/D3 未完成）
时本实验顺延，不插队。

### 11.4 结果解读与降级

- AEE ≤ 1.3430：Mode B 数值合同可用 → 侧车论文可补一句 Mode B 精度锚（差异包 +
  GPU 数字双证据）；Mode B 仍**不构成侧车主主张**（主主张 = Mode A energy 腿）。
- AEE > 1.3430：Mode B 不可挂主线；论文只报 Mode A（C7-exact）+ 差异统计包（证据 ① 的
  Mode B 节已归档），措辞 = "one-vote 数值差异客观存在且已量化，精度证据未过门，不主张"。
- t=64/t=128 两档发射翻转事实（golden `[模型]`）随 GPU 数字一起呈现为上下文，不合成
  AEE 之外的新指标。

---

## 12. 归档与论文标签

- 服务器产物回拷目录：`results/motion_sidecar_saif_ppa_20260818/`（两侧、三角、
  全部 rpt + manifest + 汇总 json/md），每份结果带两侧 `SIDECAR_DIRECTORY` 参数、
  LIB/OC 名、SAIF 身份（SHA）与标签。
- 论文标签：按 `PPA_REVIEW_CHECKLIST.md` —— 无目标宏 `pre-macro logic DC/STA/PTPX`；
  有宏无 P&R `post-synthesis macro-aware estimate`；有同 run P&R/SPEF `post-layout
  estimate`。均不等于流片/硅测。
- 签收：按 `PPA_REVIEW_CHECKLIST.md` 逐项人工核（Formality PASS、SAIF coverage、
  独立 PT_RUN_DIR、SPEF 同 run 绑定、标签正确），本 runbook 不作为签收替代。

本文件不修改任何现有文件；只新增本 md。证据分档：runbook 本身 `[模型]`；VCD/contract
`[prof]`；DC/STA/PTPX 结果 `[待验证]`（新思机器）；Mode B GPU 结果 `[模型]`。
