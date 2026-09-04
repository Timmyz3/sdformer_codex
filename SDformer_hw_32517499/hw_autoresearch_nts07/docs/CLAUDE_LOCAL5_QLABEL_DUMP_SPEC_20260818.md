# Local5 C1 第一闸：rank-1 带 Q 标签 dump 完整规格

日期：2026-08-18。本规格只为 GPU 空闲后的一个只读 dump 运行服务：不改任何现有文件
（overlay、RTL、封存 trace、冻结报告均只读），CPU-only 阶段已完成的预标定见 §4.3。

关联文档：`CLAUDE_INNOVATION_ATTACK_ROUND2_LOCAL5_20260818.md`（C1 候选，§2.6
第一闸：带 Q 标签 dump 裁决统计平面 exact 身份）、docs/150（RCSD 原始规格与被弃置
原因，§9 公平基线含 Direct5-CSE）、docs/389 §5 / docs/445（score-front CSE 封杀
类别）、docs/407（4.0 判定标准）、docs/425（ep44 身份重绑）。

## 1. 裁决对象与一句话

C1 主张："五路 stencil 分数路径上，每 descriptor 6-bit popcount 统计量跨相邻
pair 边界保持（q1[p+1]≡k1[p]，一次生成两次消费）+ pc(Q) 位门控 AND 残差"。
第一闸要裁决的是前一半：**带 Q 标签的 per-pair 数据上，该统计平面身份的 exact
身份是否成立、以何种语义成立**。裁决输出直接决定 C1 维持 3.5、降为 3.1、或转
implementation-only。

## 2. dump 对象：每条记录的字段

dump 的原子对象是**一个 (window, head) 组内的一个 token（descriptor）在本次
attention call 中的双事件向量**。每组 450 token（T2×15×15），逐组完整记录。

### 2.1 记录字段（schema `local5_qlabel_rank1_v1`）

| 字段 | 类型 | 含义 | 用途 |
|---|---|---|---|
| sample_id | u16 | 样本序号（分层抽样序号，与封存 profile100 同口径） | 流对齐 |
| stage / block | u8 | 12 个 Local5 attention block 坐标（POST_G0_BLOCK_PAIRS） | 分 stage 裁决 |
| call_index | u32 | 该 module 内的 attention call 计数（本部署每 sample 每 block 恰 1 call） | 流对齐 |
| flat_window | u16 | call 内 batch 位置（= 窗口枚举号，(d,h,w) 时间主序） | 窗口/物理帧推导 |
| head | u8 | head 序号 | 分 head 裁决 |
| adj | u8 | 0=基础抽样组；1=为跨窗口边界口径附加的对齐组 | 边界口径区分 |
| plane | u8 | 0=对 (f,f+1) 的 T0（作 Q 的帧），1=对 p 的 T1（作 K 的帧） | pair 角色 |
| pos | u16 | 窗内位置 0..224（plane 内行主序） | 几何标签 |
| qv | u32 | q_event 的 32-bit 位图（**真实 Q 事件向量，封存 trace 从未存储**） | 统计平面真值 |
| kv | u32 | k_event 的 32-bit 位图（与封存 descriptor_k_bitmap 同物） | 封存口径复现 |
| q1 | u8 | pc(qv)，6-bit 统计量（真实 q1） | 裁决主量 |
| k1 | u8 | pc(kv) | 裁决主量 |
| nbr0..4 | u16×5 | 五路候选的邻居 token id（self/N/S/E/W） | 几何一致性 |
| ev0..4 | u8×5 | 五路 valid mask | 边界 mask 口径 |
| sc0..4 | i16×5 | 五路 score_q7 码（round(score×128)，Q7 码 −256..256） | 389 防御 + 一致率 |
| ga0..4 | i16×5 | 五路 gate 码（Q1.7 码 0..256） | 389 防御 + 一致率 |

事件向量定义（overlay，只读）：
`q_event = (_qkformer_token_q(q_orig) > 0)`（L2508）、`k_event = (k_orig > 0)`
（L2509），head_dim=32，与 docs/150 §4.1 的 binary Q/K 合同一致。

### 2.2 Q 标签定义（裁决"统计平面身份"用）

对每个 token 附加派生标签（后处理计算，写入报告）：

1. **pair 坐标**：`pair_id = (sample, stage, block, head, d, h, w)`，其中
   `d = flat_window // spatial_windows`（时间窗口号）、`(h, w)` 为空间窗口号；
   窗口枚举为时间主序（`window_partition_v2` 的 permute 顺序，已核实）。
2. **物理帧**：未移位 block（block 内序号为偶）：`frame = 2d + plane`；
   移位 block（奇数号，shift (1,7,7)）：`frame = (2d + 1 + plane) mod 10`。
3. **同 token 双角色对象**：同一物理 token 同时作为 pair p 的 T1（K 角色，
   plane=1）与 pair p+1 的 T0（Q 角色，plane=0）——即
   `(sample,stage,block,head,h,w,py,px,frame)` 在两个相邻 pair 中都出现。
4. **同 pair 双角色对照**：同一 token 在**同一次 call 内**的 q_event 与 k_event
   （G6，降级备选合同用）。

## 3. 采集方式

### 3.1 hook 点（当前磁盘 overlay，H83-H86 叠代版，未改动）

- 入口：`overlay/models/STSwinNet_SNN/bsa_attention.py` L4074-4086，
  mode `binary_axnor_local5_shiftmax` / `lr_ttx` / `h66_lr` →
  `_binary_alpha_xnor_stencil_attention(temporal_pair=False, spatial_cross=True,
  motion_xor_alpha=0.0, profile_module=self)`。
- 事件向量：L2508-2509（q_event / k_event 定义点）。
- 采集钩子：L2608-2623 `trace_collector(module=..., q_event=..., k_event=...,
  k_orig=..., neighbor_index=..., valid=..., score_q7=..., gate=...)` ——
  与封存 profile100 同一钩子（`profile_local5_hardware_features.py` 的
  `Local5Collector.attach` 用 `_h9_local5_trace_collector` 属性挂接，本 dump
  复用同一挂接方式，但换成 QLabelSink）。
- hook 存在性确认：§8。

### 3.2 样本规格（与封存 ep44 profile100 同口径，保证可比）

- checkpoint：`neuron_experiments/H9_bipolar_self_attention/results/
  dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/
  checkpoint_epoch44.pth`（SHA256 `19820bec...c34f57`，docs/425 锁定）；
- config：`configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_
  hardware_order_q7q17_deploy.yml`（SHA256 `078bb517...`，与 ep44 profile100
  run-identity 一致；mode=binary_axnor_local5_shiftmax，window [2,15,15]，
  Q7 score step 1/128，RTL Shiftmax，Q1.7 gate）；
- 100 samples（valid 分层抽样 `sequence_proportional_temporal_midpoint_v1`）、
  12 blocks、每 block-sample 抽 4 组（`coprime_rotating_flat_window_head_v1`，
  与封存完全同一抽样函数与参数）→ **基础记录 4800 组 × 450 = 2,160,000**，
  与封存 npz 的 2,160,000 descriptor 逐位对齐；
- 附加"跨窗口边界对齐组"：对每个抽中组，若时间窗口 d < 4，同 (h,w,head)
  补记窗口 d+1 的完整 450-token（adj=1，预期 +约 3,840 组 / 1,728,000 记录）；
- 记录总数上限约 3.9M（npz ~120 MB，含每边 score/gate）；
- GPU 时长：模型 forward 主导。封存 ep44 profile100 实测 14:31:05→19:22:30
  （约 4h51m，含逐边 item 循环，本 dump 更轻）→ **预估 4.5–5.5 h**；
  `--samples 10` 预检约 30 min；
- CPU-only 后处理：同脚本内 numpy 计算，<10 min。

### 3.3 执行方式

```bash
/opt/conda/envs/sdformerflow/bin/python \
  scripts/dump_local5_qlabel_rank1_20260818.py \
  --config <ep44 deploy yml> --checkpoint <ep44 pth> \
  --output-dir results/local5_qlabel_rank1_ep44_20260818 \
  --samples 100
```

脚本自校验：q1==pc(qv)、k1==pc(kv) 全量一致；记录数 == 2,160,000（基础）；
kv 代理口径应复现封存 ep44 全口径 78.60%（±0.3% 容差，否则中止）；输出
`qlabel_records.npz` + `qlabel_report.json` + `qlabel_summary.md`。

## 4. 裁决标准

### 4.1 裁决层级（预注册，运行前冻结）

| 闸 | 量 | PASS 阈值 | 预期 | 裁决含义 |
|---|---|---|---|---|
| **G1 存在性** | 同一物理 token 双角色实例数（§2.2.3） | ==0 时语义不成立；>0 才可裁决同 token 口径 | **0** | "同一 descriptor 跨 pair 两次消费"在冻结部署不可实例化 → 该语义主张撤销，转入流局部口径 |
| **G2 同 token 统计一致**（仅当 G1>0） | 双角色 token 上 q1_true==k1 一致率 | ≥99% | 不适用 | exact 恒等成立才过 |
| **G3 全口径统计保持** | 流相邻 descriptor：`q1_true[p+1]==k1[p]`（真实 q_event 侧） | ≥95% | ~70–90% | exact 合同主张线；含双零分解报告 |
| **G4 非零-非零保持** | 排除双零后的保持率 | ≥60% | **~20–25%** | 统计平面在非平凡区可主张 |
| **G5 向量恒等** | `qv[p+1]==kv[p]`（32-bit 全等）比例 | ≥90% | **~0** | 代数恒等成立 |
| **G6 同 pair 双角色对照**（降级备选） | 同 token q_event==k_event 向量全等率 / popcount 一致率 | 记录不设闸 | 待测 | 若≥90% 也只等于 docs/150 §9 的 Direct5-CSE 基线（共享 q-count），不构成新对象 |
| **G7 389 防御** | 统计保持边界上 score 五元组相等率 | <1% | **~0** | 统计复用 ≠ score 值复用 |

### 4.2 总裁决映射

- **第一闸 PASS**：G1>0 且 G2≥99%；或 G3≥95% 且 G4≥60% 且 G5≥90%。
  → C1 维持 3.5，进入第二闸（同端口 miter，RTL 侧执行，本 dump 只提供
  Q/K/score 整数参考列）。
- **第一闸 FAIL（预期路径）**：G1=0（双角色对象不存在）或 G4<60% 或 G5<90%。
  → C1 不成立 4.0 第一闸：**降为 3.1（Local5 封顶，docs/407）**；C1、C2 转
  `NO_GO_AS_DATE_CONTRIBUTION / HOLD_AS_IMPLEMENTATION_OPTION`（记法与
  docs/389 §1 的 dyadic gate 同级）：唯一保留的工程项是 m-bit 门控 AND 残差
  （`n11=pc(Q_m & K_m)`，m=pc(Q)）与 stat-add，作为实现优化候选，不单列
  DATE 贡献；不做"跨 pair 统计平面"主张。
- **全口径 <95%（无论分解）**：exact 合同线不达标，同上 FAIL。
- **dump 运行失败**（脚本/环境/自校验失配）：不自动维持 3.5，修复后重跑。
- **G6 意外 ≥90%**：记录为 Direct5-CSE 基线对象（docs/150 §9 早已列为公平
  基线），C1 维持 3.1，不抬分。

### 4.3 CPU-only 预标定（封存 trace 可复算上限，[模型] 级，运行前已做）

对封存 ep29 trace（`local5_fullres_postg0_qfsa_profile100_20260730`）与 ep44
trace（`local5_ep44_hardware_rebind_20260815_profile100`）逐位复算：

| 指标 | ep29（候选文档 79.36% 来源） | ep44（rank-1） |
|---|---:|---:|
| 全口径 k1[p]==pc(bitmap[p+1])（封存代理口径） | 79.36% | 78.60% |
| 命中中双零占比（k1==0 且 proxy q1==0） | **95.40%** | **94.92%** |
| 非零-非零保持率 | **20.65%** | **22.52%** |
| k1==0（K 全零）比例 | 79.02% | 78.43% |

含义：封存 79.36% 中约 95% 由 QS 已覆盖的静默区（Q==0/K==0 → stat=0）贡献，
非零区保持率 ≈ 随机水平；且封存 trace **从未存储真实 q_event**（Q 侧缺失）。
因此 C1 第一闸的**预期裁决为 FAIL**，dump 的作用是把该裁决做成不可辩驳的
数据（真实 q_event 侧 + 双角色存在性 + 389 防御三件套），并给"转 implementation"
提供精确数字。预标定不构成裁决，正式数字以 dump 为准。

预期值区间（dump 后对照）：
- G1 实例数：0（软件时间窗口 (0,1),(2,3),(4,5),(6,7),(8,9) 非重叠；移位 block
  覆盖 (1,2),(3,4),(5,6),(7,8),(9,0)，同样非重叠）；
- G3 全口径：真实 q1 侧零率预期 ≥78%（上限 QS 静默率 89.5%），全口径一致率
  区间 70–90%，双零占比 ≥90%；
- G4 非零-非零：15–35%；
- G5 向量全等：<5%（q/k 来自不同投影 linear_q vs linear_k+pos_enc）；
- G7 score 相等率：<1%。

## 5. "帧间分数复用"攻击（389/445）防御口径

1. **合同对象钉死**：C1 共享的是 descriptor 侧 6-bit **输入统计量**（q1/k1），
   不是 score 值、不是 quotient、不发生在 slot 侧（对照 docs/389 §5 类别定义、
   docs/445 §2 的 score-front CSE 封杀措辞）。dump 逐边记录 score_q7/gate，
   提供逐边界证据：**统计保持边界上 score 五元组相等率（G7，预期 <1%）**——
   统计量保持与 score 值共享在数据上统计独立。
2. **语义排除**：G1=0 时，连"同一 token 跨 pair 共享"对象都不存在，score 值
   跨 pair 更不可能共享；每边 score 均为逐 pair 独立计算（软件模型本就不共享），
   dump 的逐边记录可证。
3. **无 Q 标签漏洞封闭**：封存 npz 只有 source 侧 K bitmap；dump 同时输出
   真实 q_event 口径与 kv 代理口径，两个口径的差异本身即裁决证据。
4. 若 C1 因此转 implementation-only，与 389 dyadic 同口径声明
   `HOLD_AS_IMPLEMENTATION_OPTION`，不与 DATE 创新分挂钩。

## 6. 降级规则汇总

| 情形 | C1 裁决 | 创新分 |
|---|---|---|
| 第一闸 PASS（§4.2） | 维持主候选，进第二闸 | 3.5 |
| G1=0 或 G4<60% 或 G5<90% | 转 implementation-only（m-bit 残差 + stat-add） | **3.1** |
| G3<95% | exact 合同不成立，同上 | **3.1** |
| dump 失败 | 不自动维持，修复重跑 | — |
| G6≥90% 意外 | 归入 Direct5-CSE 基线，不抬分 | 3.1 |

## 7. 执行脚本

`hw_autoresearch_nts07/scripts/dump_local5_qlabel_rank1_20260818.py`（新文件，
本次写入，未运行）。职责：挂 QLabelSink（复用封存 attach 机制）→ 100 samples
forward 采集 → 就地 numpy 裁决（G1–G7 全量）→ 输出 npz + report + summary。
依赖仅 torch/numpy 与项目只读模块（`profile_nts11_hardware_p0`、
`profile_local5_hardware_features` 的纯函数）。GPU 空闲后直接按 §3.3 执行。

## 8. hook 存在性确认（H83-H86 叠代版 overlay，本次已核实）

1. mode 分支 L4074-4086：`binary_axnor_local5_shiftmax`/`lr_ttx`/`h66_lr` →
   纯五路 stencil（temporal_pair=False, spatial_cross=True），与 docs/425 的
   Local5 身份、docs/150 的固定五邻域合同一致；motion 路径（h66g）与 TP 路径
   （h66f）为独立分支，不受影响。
2. 事件向量 L2508-2509 与 Q7 公式代数一致：score_raw=(n11+n00/64)/32，
   Q7 step 1/128 → code=RNE((65·n11+32−q1−k1)/16)，即 docs/150 §4.1 公式。
3. mask-after-quant + invalid_fill=hardware_score_min（L2557-2566）与 G0 合同
   （docs/150 §0.2 P0-A 冻结口径）一致。
4. trace_collector 调用点 L2608-2623 存在且参数齐备；`source_descriptor_trace`
   的 strict_local5_geometry 检查（T2×15×15、self 恒等、N/S/E/W clamp）与
   overlay stencil 构造逐位一致（profile_local5_hardware_features.py
   L1503-1552）。
5. docs/407 的 Local5 数据流（QS→inverse-stencil/FCSR→source-owned→TCFM5）
   与 docs/445 的跨时间平面合并 NO-GO 均不触碰 score 路径；本 dump 只读，
   不改变任何执行路径。

## 9. 红线

本次只产出规格文档 + 脚本草案，未运行任何 GPU 任务，未修改/删除任何现有文件。
