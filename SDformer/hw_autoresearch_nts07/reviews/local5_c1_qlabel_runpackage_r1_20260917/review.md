# Local5 C1：QLABEL 即刻可跑包 + 预注册裁决记录 + CPU 复核 + 新 idea 搜索（r1）

日期：2026-09-17。作者：Local5 算法线 worker（CPU-only）。
产出目录：`hw_autoresearch_nts07/reviews/local5_c1_qlabel_runpackage_r1_20260917/`（本文所在目录，唯一被写入位置）。
上游任务：`docs/CLAUDE_SCORE_20260917_1709.md` §本轮推进 表 B。

**红线遵守声明**：本任务未运行任何 GPU 命令；未执行 QLABEL dump 本身；未删除文件；
未改任何编号文档（含 docs/359、362、366）；H82/H86 未重启；除本目录外无任何写入。
证据分档沿用 [rtl] / [prof] / [模型]；本轮新增数字除特别标注外均为 [模型]（CPU 只读重放）。

---

## 0. 摘要

1. **QLABEL dump 的 run-card 已按脚本实读冻结**（§1）：精确命令行、GPU 空闲判定、
   10 样本预检、产物 schema、通过/失败判定、超时/abort、SHA 校验点齐备；
   **本次不执行**。
2. **C1 一闸裁决记录（§2）**：按预注册门 G1–G7 给出当前证据与预期裁决；
   预期仍为**一闸 FAIL → C1 降 3.1（implementation-only）**。本轮为决策树补上了
   一条**真实 Q 侧证据链**（见第 3 条），使 FAIL 预期不再只依赖代理口径。
3. **CPU 侧复核（§3）完成且全部逐位复现**：ep29 79.3604% / ep44 78.5976%
   （与 `SEALED_EP44_FULL_MATCH` 0.7859762898038378 完全一致），双零占比
   95.402%/94.917%，非零-非零 20.646%/22.519%。**同时发现：ep44 封存 npz 内含
   真实 Q 侧位图（`descriptor_q_bitmap`，2,160,000 条），spec §4.3 所称
   “封存 trace 从未存储真实 q_event”只对 ep29 成立。** 用真实 Q 侧重放得到
   G3=72.17%、G4=23.42%（37,059 条非零-非零边界）、G5 非零-非零仅 0.50%、
   G6（同 token 且带活动）仅 1.44%——FAIL 预期在真实口径下也成立。
4. **新 idea 搜索（§4）**：提出 2 个满足 docs/433 双腿、且不在 ROUND2 C3–C6
   否决清单中的候选（LQ4 时间四元组 stencil 合同；DRC 双角色构造合同），
   另记一条 implementation-only 补充（K-silent 对称叶，不单列 DATE 贡献）。
   A3S 已 NO_GO（docs/448），未作为候选。

---

## 1. QLABEL 即刻可跑包（run-card）

> 依据：`hw_autoresearch_nts07/scripts/dump_local5_qlabel_rank1_20260818.py`（实读，
> 30,211 B，SHA256 `377f73d6e237f5852a6073b3063f15db14c2e2c160e33634f4cedded862d85e0`）
> 与 `hw_autoresearch_nts07/docs/CLAUDE_LOCAL5_QLABEL_DUMP_SPEC_20260818.md`。
> 脚本与规格不一致处以下文 §1.7 为准（**run-card 以脚本为真**）。

### 1.1 冻结身份与 SHA 校验点（本轮已用 CPU 逐项核验）

| 物件 | 路径（相对仓库根） | 期望 SHA256 | 本轮核验 |
|---|---|---|---|
| deploy config | `neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_hardware_order_q7q17_deploy.yml` | `078bb517e2479c95719bd2eb88a08ee935d7f05bdf6733b39d2d4846f01f514d` | ✅ 一致 |
| checkpoint ep44 | `neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth` | `19820bec07cc3bf3da7e9e2e31e2af0b36bda89e636b0d273c0257b368c34f57` | ✅ 一致（591,167,684 B） |
| dump 脚本 | `hw_autoresearch_nts07/scripts/dump_local5_qlabel_rank1_20260818.py` | `377f73d6…d85e0`（30,211 B） | ✅ 未改动 |
| 封存 npz payload（自校验参照） | `hw_autoresearch_nts07/results/local5_ep44_hardware_rebind_20260815_profile100/ordered_term_items.npz` | `b48651dbd2ddb803a5ac55a97a3cc8e4cde68156074874feb24132fb078e05dc`（docs/448 口径） | ✅ 一致（22,724,822 B） |
| 封存 run-identity（脚本身份闸） | `…/local5_ep44_hardware_rebind_20260815_profile100/post_g0_run_identity.json` | 内含上述两组 SHA | ✅ 文件存在 |

脚本启动时会自行用 `file_sha256()` 对照 run-identity 中的 checkpoint/config SHA：
**不匹配即 `ValueError` 中止**；run-identity 缺失时只打印警告后继续（本次核验其存在，警告分支不会走）。

### 1.2 精确命令（全量 100 样本）

```bash
cd /root/private_data/work/sdformer_codex/SDformer
/opt/conda/envs/sdformerflow/bin/python \
  hw_autoresearch_nts07/scripts/dump_local5_qlabel_rank1_20260818.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_hardware_order_q7q17_deploy.yml \
  --checkpoint neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth \
  --output-dir hw_autoresearch_nts07/results/local5_qlabel_rank1_ep44_20260818 \
  --samples 100
```

说明：`--adjacency` 默认即为 True（`action="store_true", default=True`，见 §1.7-D2），
`--groups-per-block-sample=4`、`--num-workers=0` 已是冻结值，无需显式传；
`--sealed-npz`/`--sealed-run-identity` 默认指向封存 ep44 路径，无需显式传。
输出目录 `results/local5_qlabel_rank1_ep44_20260818/` 当前**不存在**（本轮已核），
脚本会 `mkdir(parents=True, exist_ok=True)` 创建。

### 1.3 10 样本预检命令（GPU 释放后的第一件事）

```bash
/opt/conda/envs/sdformerflow/bin/python \
  hw_autoresearch_nts07/scripts/dump_local5_qlabel_rank1_20260818.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_hardware_order_q7q17_deploy.yml \
  --checkpoint neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth \
  --output-dir hw_autoresearch_nts07/results/local5_qlabel_rank1_ep44_20260818_precheck10 \
  --samples 10
```

预检判据（**不要**拿 78.60% 去卡预检）：
- 10 样本走的是 `stratified_dataset_indices(files, 10)`，是**独立于封存 100 样本的
  新分层抽样**（不是封存前 10 个），因此 proxy_full_match 与 78.60% 不可比；
  预检只判：身份闸通过 + 无异常 + 记录数符合 §1.4 的计数公式 + 运行时长合理。
- 预期基础记录 10×12×4×450 = **216,000**；adj 对齐组最多 ~384 组 → 总计约
  **≤ 388,800 条**（随抽中窗口 d 的分布浮动）。

### 1.4 GPU 前置检查清单（空闲判定；由执行者运行，本轮未运行）

1. `ps -p 721081`（t49_ternary_ws4_ep5 训练）**已不存在**；与 GPU 队列纪律一致
   （一次一个任务，与 DATE 缺口审计 agent 协调）。
2. `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
   → 目标卡上无其它 compute 进程。
3. `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv`
   → 间隔 150 s 连续两次采样：`memory.used < 1500 MiB` 且 `utilization.gpu == 0%`。
4. 环境自检：`/opt/conda/envs/sdformerflow/bin/python -c "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())"`
   → `True ≥1`。
5. 资源：results 所在卷 ≥ 2 GB 空闲（npz 含 adj 后可能 >120 MB，压缩耗时数分钟）；
   主机 RAM ≥ 8 GB 空闲（sink 在内存中累积全部组）。

### 1.5 预期产物与 schema

输出目录三件套（脚本 `main()` 落盘）：

| 文件 | 内容 |
|---|---|
| `qlabel_records.npz` | `np.savez_compressed`，键：`sample(u16) / stage(u8) / block(u8) / call(u32) / flat_window(u16) / head(u8) / adj(u8) / plane(u8) / pos(u16) / qv(u32) / kv(u32) / q1(u8) / k1(u8)` + `nbr[N,5]u16 / ev[N,5]u8 / sc[N,5]i16 / ga[N,5]i16` |
| `qlabel_report.json` | 身份（config/checkpoint SHA）、`verdict`（G1–G7 全量 + m_hist + 跨窗口口径）、`sealed_check`、`final_verdict` |
| `qlabel_summary.md` | 裁决摘要表 |

记录计数口径（100 样本）：基础 4800 组 × 450 = **2,160,000**（与封存 npz 逐位对齐）；
adj=1 对齐组约 **3,840 组 ≈ 1,728,000**；总上限约 **3.9M**。
`sc`/`ga` 是 hook 传入的归一化分数/门值 ×128 取整（已核 overlay：L3895-3916 之后
`score_q7=scores`（float，`round(score*128)` 即 Q7 码 −256..256）、`gate` 经
`_apply_hardware_gate_quant` 后为 Q1.7 值域 [0,2] → `round(gate*128)` 0..256，
与规格 §2.1 的列语义一致）。

### 1.6 通过/失败判定（关键：**不看 exit code**）

脚本 `main()` 末尾为 `return 0 if final_verdict["ruling"] == "GATE1_PASS" else 0`
——**任何裁决下退出码都是 0**。判定必须读 `qlabel_report.json`：

| `final_verdict.ruling` | 含义 | 后续动作 |
|---|---|---|
| `GATE1_FAIL` | G1==0 或 G3<0.95 或 G4<0.60 或 G5<0.90 | 按 §2.3 落 C1→3.1 记录 |
| `GATE1_PASS` | G3≥0.95 且 G4≥0.60 且 G5≥0.90 | 先与 §3 封存真实口径对账（§2.3 翻盘路径），对账通过才维持 3.5 |
| `INVALID_DUMP` | 封存复现失配 | 当前**不可达**（§1.7-D2），出现即人工复核 |

人工补充校验（脚本不自动做）：
1. `records_base == 2,160,000`（100 样本）且 `records_total` 在 ~3.6–3.9M 区间；
2. `verdict.proxy_full_match` 与封存 `0.7859762898038378` 之差 ≤ **0.005**（脚本内
   `SEALED_TOL`；规格 §3.3 文字写 ±0.3%，从紧取 **0.003**）；**超差即视为
   INVALID，不得据该 dump 抬分**（脚本自身不会拦，见 §1.7-D2）；
3. `report.config_sha256`/`checkpoint_sha256` 与 §1.1 表一致；
4. 运行前后 dump 脚本 SHA 不变（`377f73d6…`）。

### 1.7 脚本-规格偏差与操作注意（实读发现，必须在执行时携带）

| 编号 | 事实（以脚本为准） | 影响与操作 |
|---|---|---|
| D1 | 退出码恒为 0（L687） | 判定只读 JSON；CI/脚本包装不得依赖 exit code |
| D2 | `--adjacency` 为 `store_true, default=True` → 无法关闭；`sealed_check` 分支条件 `args.sealed_npz.exists() and not args.adjacency` 恒为 False → `reproduced` 恒 True、`INVALID_DUMP` **不可达** | 封存 78.60% 复现校验改为**人工**（§1.6-2）；脚本自校验只剩 `q1==pc(qv)`、`k1==pc(kv)` 与形状/身份闸 |
| D3 | 脚本 `SEALED_TOL=0.005`，规格 §3.3 文字为 ±0.3% | 取从紧 0.003 执行人工判定 |
| D4 | `finalize_verdict()` 不复用 `gate1_pass`：规格 §4.2 的 PASS 路径 A（G1>0 且 G2≥99%）在最终裁决中被压过（只要 G3/G4/G5 任一不达标即 FAIL；脚本的 `gate1_pass` 里 G2 用的是 G6-pop 替代量） | 读到 `G1>0 且 G2≥99% 但 ruling=GATE1_FAIL` 时按“规格-脚本冲突”上报复核，不自行解释 |
| D5 | `T_G7=0.01` 只写进 thresholds 报告，不参与裁决 | G7 按报告量处理（预期 ~0 才符合 389 防御叙事） |
| D6 | 10 样本预检的样本集与封存 100 不同（§1.3） | 预检不做 78.60% 复现判据 |
| D7 | `--samples 1` 走 `list(range(min(1,len(dataset))))`，不触发分层抽样 | 单样本冒烟可用，但只作管线连通性检查 |

### 1.8 超时与 abort 条件

| 阶段 | 预期时长 | 看门狗 | 立即 abort 的条件 |
|---|---|---|---|
| 10 样本预检 | ~30 min | 90 min 硬超时 | 身份闸报错；attach 报错（block 集合/mode 不匹配）；descriptor 形状非 450×32；adjudicate 自校验 `q1!=pc(qv)`；`样本不足`；CUDA OOM；单样本均时 > 4 min |
| 100 样本全量 | 4.5–5.5 h | 8 h 硬超时 | 同上；已跑样本 ≥20 且外推总时 > 10 h；日志出现连续相同 `processed k/100` 停滞 > 20 min |

**abort 时不得删除已写产物**；记录 `processed` 数与 stdout 最后 50 行，随后按
“dump 运行失败 = 不自动维持 3.5，修复后重跑”（规格 §4.2）处理。

### 1.9 本次不执行

**本任务不执行 dump（含 10 样本预检）**：GPU 正被 t49_ternary_ws4_ep5（PID 721081）
占用；本节仅为 GPU 空闲后的即刻可跑包。

---

## 2. C1 预注册裁决记录（G1–G7）

预注册来源：`docs/CLAUDE_LOCAL5_QLABEL_DUMP_SPEC_20260818.md` §4。**运行前冻结，不得
因结果改阈值。** 下表“现有证据”栏中：`[模型]` = 本轮 CPU 只读重放（§3），
`[prof]` = 封存 profiler 口径，`预期` = 规格 §4.1 预先声明的期望值。

### 2.1 逐门记录

| 闸 | 量 | PASS 阈值 | 规格预期 | 现有证据（运行前） | 预期裁决 |
|---|---|---|---|---|---|
| **G1 存在性** | 同一物理 token 双角色实例数（同一 block 内 plane1 帧 f 与 plane0 帧 f） | >0 才可裁决同 token 口径 | **0** | 结构性论证：偶数 block pair 枚举 (0,1),(2,3),(4,5),(6,7),(8,9)、移位 block (1,2),(3,4),(5,6),(7,8),(9,0)，**同一 block 内 pair 不重叠** → 双角色对象不存在 [模型-结构] | **0 → 路径 A 关闭** |
| **G2 同 token 统计一致** | 双角色 token 上 q1_true==k1 一致率 | ≥99%（仅 G1>0 时适用） | 不适用 | G1=0 → 不适用；此外封存真实 Q 侧显示**同 token 的 Q/K 事件向量在有活动时几乎从不同源**（G6-pop 条件一致率 1.44%，§3.2） | **不适用** |
| **G3 全口径统计保持** | `q1_true[p+1]==k1[p]` | ≥95% | ~70–90% | 封存真实 Q 侧 [模型]：**72.17%**（1,558,948/2,159,999 命中）；代理口径 78.60%（复现） | **FAIL**（<95%） |
| **G4 非零-非零保持** | 排除双零后的保持率 | ≥60% | **~20–25%** | 封存真实 Q 侧 [模型]：**23.42%**（命中 8,679 / 非零-非零 37,059）；代理口径 22.52% | **FAIL**（<60%，且在预期带内） |
| **G5 向量恒等** | `qv[p+1]==kv[p]`（32-bit 全等） | ≥90% | **~0** | 封存真实 Q 侧 [模型]：总体 71.78% 全为双零退化；**非零-非零边界上仅 0.50%**（命中边界上 2.15%） | **FAIL**（有效口径 ~0.5%，与预期一致） |
| **G6 同 pair 双角色对照**（降级备选） | 同 token q_event==k_event 全等率 / popcount 一致率 | 记录不设闸 | 待测 | 封存真实 Q 侧 [模型]：pop 一致 72.18% 总体，**带活动 token 上仅 1.44%**；vec 全等 0.034% | 记录；**远低于 90%**，不构成 Direct5-CSE 基线对象 |
| **G7 389 防御** | 统计保持边界上 score 五元组相等率 | <1% | **~0** | 近似重放 [模型]（无 clamp/mask，非权威）：命中边界 99.33%（退化主导）、**非零-非零命中边界 62.90%**（n=8,679）→ 与“~0”预期不符，见 §3.3 警示 | 报告量；**必须由 dump 的逐边 sc 列重算后才可引用** |

补充事实（[模型]，非闸，供证据链）：q1_true 零率 **91.63%**（k1 零率 78.43%）——
Q 侧比 K 侧更静默；有效 lane 边（10,224,000 条，边界 clamp 已排除）上邻居 K 全零占
**78.42%**、目的地 Q 全零占 **91.61%**。

### 2.2 总裁决映射（规格 §4.2，运行前冻结）

- **PASS**：G1>0 且 G2≥99%；**或** G3≥95% 且 G4≥60% 且 G5≥90% → C1 维持 3.5，进第二闸（同端口 miter）。
- **FAIL（预期路径）**：G1=0，或 G4<60%，或 G5<90%，或 G3<95% → C1 降 **3.1**；
  C1/C2 转 `NO_GO_AS_DATE_CONTRIBUTION / HOLD_AS_IMPLEMENTATION_OPTION`；
  保留工程项：m-bit 门控 AND 残差（`n11=pc(Q_m&K_m)`，m=pc(Q)）+ stat-add，不单列 DATE 贡献。
- **G6≥90%（意外）**：归入 docs/150 §9 的 Direct5-CSE 公平基线，C1 维持 3.1，不抬分。
- **dump 失败/自校验失配**：不自动维持 3.5，修复重跑。

### 2.3 dump 跑完后的裁决决策树

```
读 qlabel_report.json: final_verdict.ruling
├─ INVALID_DUMP（脚本当前不可达；若人工判定 proxy_full_match 超差即视为 INVALID）
│   → 不裁决；修环境/重跑；C1 状态不变（不自动维持 3.5）
├─ GATE1_FAIL
│   → 正式记录：C1 降 3.1，转 implementation-only（m-bit 门控残差 + stat-add）
│     （若 G4 落在 15–35% 预期带内、G3≈70–90%，与封存真实口径一致 → 按预期路径收口）
└─ GATE1_PASS（翻盘情形，必须走完下列对账才允许抬分）
    ├─ 冲突检查：dump 的 G3/G4/G5 与封存真实 Q 侧重放（72.17/23.42/0.50）不一致
    │   → 强制对账三步：
    │     ① 基础流一致性：records_base==2,160,000、组元数据与封存 100 样本同抽样、
    │        proxy_full_match≈78.60%；② 分 stage/head 复算 G3/G4 并逐项对齐；
    │     ③ 身份漂移检查：config/checkpoint SHA + overlay/hook 版本（spec §8 的 L4074-4086 /
    │        L2508-2509 路径）是否与 8/15 封存一致。
    │   对账不过 → 不得抬分，按“dump 异常”重跑或升级复核。
    │   对账通过（即封存 npz 的 Q 列被证明不是运行时真值）→ C1 维持 3.5 → 第二闸：
    │     同端口 miter（stat 平面 + m-bit 残差 vs 32-bit XNOR 叶，逐 (Q,K) 整数零失配，
    │     对齐 docs/150 §4.5 门槛 5 与 425 的 360,000 Acc32 零失配口径）+ SAIF/PTPX EDP 对照
    │     （EDP≤0.85x）；G7 必须先落到 <1%（389 防御叙事的前提）。
    ├─ 仅路径 A 成立（G1>0 且 G2≥99% 而 G3/G4/G5 不达标）
    │   → 在 (2,15,15) 非重叠分区下双角色对象不应存在（§2.1 G1）；
    │     视为**分区/配置身份异常**或脚本 D4 冲突 → 升级复核，默认不抬分。
    └─ G6≥90% 意外 → 归 Direct5-CSE 基线，C1 维持 3.1，不抬分。
```

**翻盘现实性评估**：路径 A 需要 pair 重叠（合同级改动，不是 dump 能改变的）；路径 B 需要
G3≥95% 且 G4≥60% 且 G5≥90%，而封存真实 Q 侧给出 72.17/23.42/0.50——dump 的样本、
部署路径、分区与封存一致，**预期不会翻盘**。唯一有意义的翻盘触发点是“封存 npz 的
`descriptor_q_bitmap` 被证明不是运行时真实 q_event”，这正是对账三步要排除的。

**替代正式化路径（供决策，不自行执行）**：封存 ep44 已含真实 Q 侧且身份可锁
（config/checkpoint/npz payload SHA 全部核验通过，§1.1）。若 GPU 长期被训练/审计占用，
可选择以**封存重放 + 本轮脚本复现**作为一闸的 [模型] 级正式证据（记录偏离预注册之处：
无 adj 对齐组、无 G1 显式检查、无逐边 sc/ga 列、G7 仅近似），并把 dump 降为可选确认。
该方案需用户/评分方批准后才可写进裁决记录；否则按预注册等 GPU 缺口跑 10 样本预检。

---

## 3. CPU 侧可复核项（只读重放回执）

执行方式：`cpu_replay_local5_c1_statistics_20260917.py`（本目录），只读加载封存 npz
（`np.load(..., allow_pickle=True)`），不改原数据、不写外部目录；证据分档 **[模型]**。
产物：`cpu_replay_local5_c1_statistics_20260917.json`。

### 3.1 代理口径复现（判决：**完全一致**）

| 指标 | 文档声称（spec §4.3 / 任务背景） | 本轮重放 | 判定 |
|---|---:|---:|---|
| ep29 全口径 `k1[p]==pc(bitmap[p+1])` | 79.36% | **0.7936041**（命中 1,714,184） | ✅ |
| ep29 命中中双零占比 | 95.40% | **0.9540160** | ✅ |
| ep29 非零-非零保持率 | 20.65% | **0.2064633**（n=381,787） | ✅ |
| ep29 `k1==0` 比例 | 79.02% | **0.7901787** | ✅ |
| ep44 全口径（脚本常量 `SEALED_EP44_FULL_MATCH`） | 0.7859762898038378 | **0.7859762898038378**（命中 1,697,708） | ✅ 逐位 |
| ep44 命中中双零占比 | 94.92% | **0.9491697** | ✅ |
| ep44 非零-非零保持率 | 22.52% | **0.2251916**（n=383,207） | ✅ |
| ep44 `k1==0` 比例 | 78.43% | **0.7843069** | ✅ |
| 恒等 `source_k_popcount == pc(descriptor_k_bitmap)` | 隐含口径 | **全量成立** | ✅ |
| 组边界口径 vs 平坦口径 | — | **两者逐位相同**（4800 组连续铺满 2,160,000 条：4,799 条组间边界 ⊂ 平坦相邻对） | ✅（澄清口径，无差异） |

### 3.2 新发现：ep44 封存 npz 含真实 Q 侧，真实口径数字首次落盘 [模型]

事实链：ep29 npz 无 `descriptor_q_bitmap`；**ep44 npz 有**（uint64 × 2,160,000）；
`profile_local5_hardware_features.py` L2021-2022 规定 post_g0 轨迹缺 `q_event` 直接
报错，L2040-2044 用 `q_event[window, head]` 落盘，L2222-2230 打包成
`descriptor_q_bitmap`；ep44 manifest `evidence_level=post_g0`、100 样本、
4800 组、`local5_post_g0_run_identity_v3`。**结论：spec §4.3 “封存 trace 从未存储
真实 q_event（Q 侧缺失）”只对 ep29 成立；ep44（rank-1 部署身份）的 Q 侧在盘上。**

用真实 Q 侧重放 ep44 得到的“一闸真值”（[模型]，与 dump 预期口径差：无 adj 组、
无 G1 显式检查、sc/ga 为近似）：

| 量 | 值 | 对应闸与阈值 |
|---|---:|---|
| `q1_true[p+1]==k1[p]` 全口径 | **0.7217355**（1,558,948 命中） | G3，需 ≥0.95 → **FAIL** |
| 命中中双零占比 | **0.9944328** | QS 覆盖分解证据（比代理口径更极端） |
| 非零-非零保持率 | **0.2341941**（n=37,059，命中 8,679） | G4，需 ≥0.60 → **FAIL**（落 15–35% 预期带） |
| 向量全等 `qv[p+1]==kv[p]` 总体 | 0.7178040（双零退化） | — |
| 向量全等（非零-非零边界） | **0.0050460**；命中边界上 0.0215463 | G5，需 ≥0.90 → **FAIL**（符合“~0”预期） |
| 同 token q_event vs k_event（G6-pop）总体 | 0.7218444 | 记录 |
| **G6-pop（带活动 token, n=609,612）** | **0.0144288**；G6-vec 0.0003363 | 「同源事件共享」在真实数据上不存在 |
| `q1_true` 零率 / `k1` 零率 | 0.9162532 / 0.7843069 | Q 侧更静默 |
| 有效 lane 边（10,224,000）邻居 K 全零占比 | 0.7842211 | 供 §4.4 实现级候选 |
| 有效 lane 边目的地 Q 全零占比 | 0.9161191 | 同上 |

对 C1 的直接影响：**FAIL 预期在真实 Q 侧同样成立**（G3/G4/G5 全不过），且 G6 的条件
数字（1.44%）说明“同一 token 的 Q/K 事件可共享”这条最后的结构性退路也不存在
（因此不构成 docs/150 §9 的 Direct5-CSE 基线对象）。

### 3.3 G7 近似重放（**警示口径，非权威**）

按 overlay L3856-3874 的 stencil 几何（self,N,S,E,W + clamp，15×15 平面）与
Q7 码公式 `RNE((65·n11+32−q1−k1)/16)` 在封存真实 Q/K 位图上重算 5-lane 码元组：
命中边界上元组全等率 99.33%；**非零-非零命中边界上 62.90%（n=8,679）**。
与规格 §4.1 的 G7 预期（~0/<1%）不符。三点限制必须先声明，**不得据此外推**：
1. 本轮省略 `mask-after-quant + invalid_fill=hardware_score_min`（L2557-2566）与
   边界 valid mask，且沿用了原始量纲假定；dump 的 sc 列来自 hook 真值，才是权威；
2. 结果被双零/退化结构主导（99.33% 一栏）；条件到非零-非零后样本仅 8,679 条；
3. 无论数据如何，**跨边界 score 值/商复用在 docs/389 §5、docs/445 是类别级封杀**
   （score-front CSE），不因该 G7 数字而解禁——C1 的合同对象钉死在 descriptor 侧
   6-bit 输入统计量，本近似不作为任何分数复用主张的证据。

### 3.4 不可复核 / 未复核项

- **G1 显式存在性**：封存 npz 无 (window, head) 元数据（`group_tags` 只是顺序计数），
  group→(flat_window,head) 的映射不可从 npz 直接反演；G1 只能由 dump 的
  `flat_window/head/plane/pos` + 移位表裁决。本轮只给结构性论证（§2.1）。
- **adj=1 跨窗口对齐组**口径：封存无此结构，必须由 dump 产生。
- **逐边 sc/ga（G7 权威列）**：封存 npz 未存 per-edge score 码（只有 gate 与
  score 路径的 cycle 计数），必须由 dump 的 hook 列产生。
- **hook/overlay 版本漂移**：封存 NPZ 产生于 8/15，若 overlay 之后被改（spec §8 声称
  未改），dump 的复现闸（proxy 78.60%）能覆盖 K 侧但不能覆盖 Q 侧——对账三步（§2.3）
  必须包含 overlay/hook 版本核对。
- 本轮**未**复核：ep29 的真实 Q 侧（不存在）、valid825 全量（非 100 样本口径）、
  以及任何 RTL/周期/能量主张（超出 [模型] 档）。

### 3.5 复现命令

```bash
cd /root/private_data/work/sdformer_codex/SDformer
PYTHONDONTWRITEBYTECODE=1 python3 \
  hw_autoresearch_nts07/reviews/local5_c1_qlabel_runpackage_r1_20260917/cpu_replay_local5_c1_statistics_20260917.py
```

---

## 4. Local5 新 idea 搜索（满足 docs/433 双腿，且不在 ROUND2 C3–C6 否决清单）

判据：新算法算子合同 + 改变硬件存储/执行对象；排除 C3（gate 5 元组码本）、
C4（K-delta 字母表闭包）、C5（跨 tile gate 商保持）、C6（跨 plane gate 模式复用）；
A3S 已 NO_GO（docs/448），不作为候选；不触碰 docs/433 的 cache/码本/2-wide/第三
stencil/近似剪枝/分配律换序封杀线；不把 Direct5-CSE（共享 q-count，docs/150 §9 公平基线）
包装成创新。新证据（§3.2）给两条硬约束：**Q/K 事件同源在带活动 token 上仅 1.44%**、
**非零-非零跨 descriptor 统计保持仅 23.4%** —— 任何“复用现有统计量”的路线必须绕开这两点。

### 4.1 候选 L1（主推）：LQ4 —— Local5 时间四元组 stencil 合同

- **合同一句话**：把 Local5 的 (2,15,15) 时间 pair 窗改为 **(4,15,15) 四元组窗**
  （每次 call 内两条 pair：(f,f+1),(f+2,f+3)），并规定每个窗位置物化
  **4 槽统计记录** `(q_t, k_t, o_t)`（q/k=事件 popcount，o=自相关 n11），
  分数由槽位记录精确重建，同分槽位沿时间做 run-length 广播。
- **新存储对象**：4-slot per-position 统计记录（4×(q,k,o)），替代逐边 2×32-bit popcount 的重算。
- **新执行对象**：槽位链 stencil 合成器 + 同分槽位广播（不引入新乘加路径）。
- **不在 C3–C6**：无码本、无 delta 字母表、无跨 tile/plane 的 gate 商、无模式表；
  与 C1 的关系：C1 是“跨 pair 保留统计量”（数据已判死），LQ4 是**同一 call 内多槽位的精确记录**
  （不依赖跨边界身份），前提与结论完全不同。
- **诚实风险**：与队列 P0-1/D1（Motion T>2 时间商）同族；Local5 与 Motion 共享 Swin
  窗口合同 → 必须双线协调，若 D1/B2 训练线判负，本候选同步降级；四元组窗对
  SNN 时间步语义的影响需训练裁决。
- **最小验证路径**（由便宜到贵）：
  1. [模型] CPU：用封存 ep44 npz 复算 4 帧窗内的槽位统计与同分率上界（现有观测：
     q1 零率 91.6%、k1 零率 78.4%、非零-非零跨相邻保持 23.4% → 先算槽间同分率与
     广播率，判定位账是否 ≥ 阈值，成本 <1 h）；
  2. [模型] 合同数学检查（槽位分解唯一性、重建逐位可逆，仿 D1 附录 A 的 I1-I7 形式）；
  3. [prof] seed 0 short 训练（loss 不塌）→ fullres → valid825 对比 Local5 锚点
     **1.281893×1.01**（DATE_PAPER_RESULT_TABLES_20260901 表 A5）；
  4. [rtl] 同端口 miter + SAIF/PTPX（服务器，双线排队）。
- **证据分档现状**：[模型]（数据面）；训练/PPA 未做。

### 4.2 候选 L2（备选）：DRC —— 双角色构造合同（跨 block 时间链 + 单事件向量）

- **合同一句话**：把“帧 f 在偶数 block 作 T1(K)、在奇数 block 作 T0(Q)”这一现有
  时间链写进算子合同，并**绑定事件向量**（每 token 单一 32-bit 事件位图，Q/K 同源），
  使 descriptor 侧统计量跨 block 成为**由构造保证**的一次写两次读；新存储 =
  物理坐标索引的**有界统计环**；新执行 = 跨 block stat 消费 + K 侧独立事件位图路径取消。
- **新存储对象**：物理 (frame, pixel) 索引的统计环（生命周期 = 相邻 block 的 pair 跨度，
  按 SRAM 预算裁剪；初估 2 帧全分辨率环 ~194 KB，**必须先做位账**，可能超 388 KB 预算）。
- **新执行对象**：跨 block 的 stat 读/写通路与单事件位图 K 路径。
- **为什么不是 C5/C6**：C5/C6 是 **gate 商/模式** 的跨 tile/plane 复用（已判死）；
  DRC 的对象是 **descriptor 输入侧事件统计**，且绑定条件来自算子合同（事件同源），
  不是查表/缓存（措辞必须逐条对照 docs/389 §5、docs/445 后定稿）。
- **致命风险（必须先回答）**：本轮 [模型] 证据显示同 token 的 Q/K 事件在带活动时
  仅 1.44% 同源——DRC 要求训练把 Q/K 事件**绑成同源**，等于改 attention 语义，
  精度风险最高；且统计环的 SRAM 账很可能过不了 433 的存储预算线。
- **最小验证路径**：
  1. [模型] CPU：以封存 ep44 复算“若 k_event:=q_event”下的 gate/分数变化率与
     silent/ident-K 结构变化（可直接在本目录脚本上扩展）；同时出统计环位账
     （能否 <10% SRAM 预算）；
  2. 若两者都过线 → 合同草案（新算子合同一句话 + 存储/执行对象）→ seed 0 short 训练；
  3. [prof]/[rtl] 同 L1 步骤 3–4。
- **证据分档现状**：[模型]（部分）；建议列为 L1 之后的第二轮候选。

### 4.3 实现级补充（**不满足 433 双腿，不列入 DATE 贡献**）

**K-silent 对称叶**：QS 的对称退化叶——邻域 K 全零时 `score = 32 − q1`，无需 32-bit
K 读。本轮 [模型] 量：有效 lane 边上邻居 K 全零占 **78.42%**、目的地 Q 全零占
**91.61%**（增量覆盖需补算 joint，未做）。它只降 K 读位、不新增物化对象，属
docs/433 的“现有算子下工程敏感度”，与 C1 降级后保留的 m-bit 门控残差 + stat-add
同档（3.1 implementation-only），不得单列 DATE 贡献。

---

## 5. 边界与纪律

- 未改任何编号文档；docs/359/362/366 未触碰；Local5 sealed `1.6957x/1.770x` 未动；
  A3S 维持 `NO_GO_AS_HARDWARE_ACCELERATOR`（docs/448）；H82/H86 未重启。
- 本目录为唯一写入点；封存 npz/checkpoint/config 均只读访问（`np.load` / sha256 读）。
- 所有新数字为 [模型] 档，不构成 RTL/周期/能量/PPA 主张；一闸正式数字仍以 dump
  （或经批准的封存重放替代方案）为准。

## 附录 A：本目录产物

| 文件 | 说明 |
|---|---|
| `review.md` | 本文 |
| `cpu_replay_local5_c1_statistics_20260917.py` | CPU-only 只读重放脚本（§3 全部数字） |
| `cpu_replay_local5_c1_statistics_20260917.json` | 重放回执（机器可读） |
| `review_artifact_sha256.json` | 上述三件套的 SHA256 回执 |
