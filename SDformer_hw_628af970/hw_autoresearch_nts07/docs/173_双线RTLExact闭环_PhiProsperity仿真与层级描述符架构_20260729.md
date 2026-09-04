# Motion/Local5 双线 RTL-exact 闭环、Phi/Prosperity 仿真与层级描述符架构

## 0. 本轮结论

本轮不冻结唯一算法主线。当前策略是：

- **Motion**：真实 ordered profile、crop RTL-exact 和 term/delivery 统计更完整，仍是证据领先线。
- **Local5**：必须并行保留。其 crop dyadic/RTL-exact 已有较好准确率，fullres hardware-order 已排队；硬件侧不得因 post-G0 profile 尚未返回而停止。
- **T=450**：两条线都只能做控制、地址、存储和线性容量外推；性能声明必须等待 fullres profile/ordered trace。

新增的硬件编码候选为：

> **ISHD：Identity-Stationary Hierarchical Descriptor，身份驻留层级描述符。**

它不是简单缩窄总线，而是把 term 的不变量与变量分层：

1. L1 header 驻留 `{gate, lane, multiplicity, group/tag}`；
2. L2 continuation 只传单调递增的 destination delta；
3. builder 与 directory 共驻留，入口 key/destination 只在局部寄存器移动；
4. bank-local executor 只在 `term_first` 取权重和生成 product，后续 continuation 复用；
5. escape beat 保证任意 destination 间隔仍然 exact。

这对应 Phi 的“公共 pattern + residual”思想，但本地化对象不是激活近似，而是 **exact term identity + destination residual**，因此不引入准确率损失。需要强调：在独立审稿后，ISHD 已从“独立 DATE 架构贡献”降级为 C1 数据流中的编码候选；只有完成专用 RTL、真实 escape 统计和同约束 PPA 后才重新评估。

### 0.1 2026-07-29 严格审稿纠错

旧版报告中的下列数字全部作废，不得进入论文：

- Local5 ISHD `4.088×`；
- Motion ISHD `1.728×`；
- Motion 在 profile100 M4 delivery 上再次叠加 PPDI `30.27%`；
- “Prosperity 官方仿真结果”这一可能引起误解的表述。

原因有三项：

1. Local5 错把唯一 `mfep_multicast_terms` 同时当成 destination delivery，免费消掉约 `11.2×` fanout；
2. Local5 ISHD 又按假设 fanout 二次缩减 weight fetch；
3. Motion 的 `projection_gate_multicast_delivery_m4` 已是压缩后工作，不能再套 sample0/window0 的标量 PPDI 比例。

修正后增加了语义守恒单测，分别约束：

```text
product/weight fetch = exact unique term
destination delivery = destination gate-lane group 或已统计 M4 delivery
```

并禁止两者相互替代。

---

## 1. 软件/数值主线状态

### 1.1 已有证据

| 主线 | 已完成 | 尚缺 |
|---|---|---|
| Motion window9/T162 | AEE `1.462688`、AAE `9.403994`；attention-row RTL 回归通过 | fullres T450 整行/整窗口 SV exact |
| Local5 window9/T162 | dyadic AEE `1.4475`、AAE `9.3860`；RTL-exact AEE `1.4486`、AAE `9.4210` | post-G0 ordered profile 和 fullres exact |
| Motion window15/T450 | 配置、fullres 浮点队列和 hardware-order follower 已建立 | 训练/valid825、Q7/Q1.7、T450 RTL 控制/地址/存储 |
| Local5 window15/T450 | 配置、fullres 浮点队列和 hardware-order follower 已建立 | 训练/valid825、真实 invalid mask、post-G0 term/ordered trace |

四个部署配置均为：

- 分辨率 `480×640`
- `window=[2,15,15]`
- `crop=null`
- `scale_factor=1`
- BN 不使用 running statistics

截至本轮检查，主 GPU 仍由 paper-window15 队列占用；exact follower 仅轮询，不额外占 GPU。本轮硬件工作均为 CPU-only。

### 1.2 为什么 T=450 可以先做容量外推但不能报性能

crop 配置为 `288×384/window9`，fullres 为 `480×640/window15`：

```text
窗口 token 比 = 450 / 162 = 25 / 9
图像面积比   = (480×640) / (288×384) = 25 / 9
```

窗口数近似不变，而每窗 token 数增加 `25/9`。因此可以提前检查：

- `DEST_W`: T162 需要 8 bit，T450 需要 9 bit；
- row/line-buffer 深度和读端口；
- accumulator 深度；
- descriptor continuation 的 token 编码；
- counter、loop bound 和 power-of-two 截断。

但 gate cardinality、term fanout、边界 invalid 比例、stall 和 p99 不保证线性，必须等待真实 fullres profile。

---

## 2. 本轮证据闭环

### 2.1 Local5 provenance 门禁

`post_g0` replay 现在实际重开并校验：

- config 文件及 SHA256；
- checkpoint 文件及 SHA256；
- `ordered_cohort.json` 及 SHA256；
- producer RTL 源文件及 SHA256；
- 样本 key/sequence key；
- attention mode、Q7/Q1.7、RTL Shiftmax、真实 invalid mask；
- crop、resolution、scale factor、BN 和 window size。

伪造标签、伪 hash、manifest 生成后篡改配置均会被拒绝。只有真实 provenance 全部通过，`performance_claim_allowed` 才能为真。

### 2.2 Producer 顺序闭环

软件 sink 已从候选优先 `torch.nonzero` 顺序改为与 RTL 完全一致：

```text
destination ascending
  -> lane ascending
    -> gate first-valid-candidate occurrence order
```

`tb_local5_mfep_term_builder.sv` 不再只检查 term 数量，而是检查：

- gate
- lane
- multiplicity
- destination
- `term_last`

直接 RTL 结果：

```text
PASS ... checked_terms=44
```

### 2.3 ET3/native-m 阻塞闭环

native-m baseline 新增真实 FIFO 深度 2 场景：

1. executor 阻塞；
2. FIFO 填满；
3. 第三个 item 保持 valid；
4. 解除阻塞；
5. 必须命中同拍 full-pop + push。

结果：

```text
PASS native-m baseline:
items=6 products=6 explode=15 groups=3 stalls=9 full_pop_push=1
```

完整 ET3 流程：

- Python evidence/replay：`13/13 PASS`
- Icarus ET3：PASS
- Icarus native-m：PASS
- Verilator lint：PASS
- Verilator + SVA：PASS
- Yosys synthesis-readiness：PASS

这关闭了第五轮 DATE 评审指出的 provenance 和极端握手覆盖 P0/P1。

---

## 3. Phi/Prosperity 仿真器使用边界

### 3.1 实际复用了什么

本地已有 Prosperity 官方开源仓库：

```text
third_party/Prosperity
```

新模拟器：

```text
scripts/phi_prosperity_dual_line_simulator.py
```

实际通过 SDformer Python 环境导入官方 `simulator/utils.py::Stats`，导入结果为 `True`。这只证明分账字段兼容，不代表调用了 Prosperity 官方 cycle-accurate `Simulator`。当前复用内容为：

- `compute_cycles`
- `mem_stall_cycles`
- `preprocess_stall_cycles`
- `dram/g_act/g_wgt/g_psum` 分账范式
- compute/memory overlap 的建模习惯

`report.json` 同时冻结：

- Prosperity commit；
- `utils.py` / `simulator.py` SHA256；
- Local5/Motion 输入 profile SHA256；
- 当前模型脚本 SHA256。

双线解析模型尚未复用：

- Prosperity 的 `Simulator.run_fc/run_attention` 周期路径；
- Prosperity 论文的固定功耗常数；
- Prosperity 的 CUDA product-sparsity kernel 结果；
- Prosperity 的 speedup/PPA 数字；
- 不同 SRAM、频率或工艺下的能耗结果。

审稿后新增 `scripts/run_prosperity_official_probe.py`，已真实调用官方 CPU `Simulator.run_fc`。在官方 `spikformer_cifar100` reference 上：

| 层 | product-sparsity cycles | bit-sparsity cycles | 官方周期比 |
|---|---:|---:|---:|
| fc_q_enc_0 | 41,429 | 73,883 | 1.783× |
| fc_o_enc_0 | 18,590 | 106,832 | 5.747× |
| fc_2_enc_0 | 33,599 | 44,939 | 1.338× |

这些结果只证明官方工具链已跑通，不能转写为 Motion/Local5 性能。双线必须先导出逐元素 0/1 矩阵，再接入相同官方路径。

### 3.2 Phi 为什么是 clean-room 复刻

Phi 论文提出 L1 pattern/PWP 与 L2 residual 的两级稀疏。2026-07-29 再次检查 arXiv、代码关联页和 GitHub，仍未发现公开官方模拟器。因此本轮只做：

- 50%/75%/90%/95% L1 hit 敏感性；
- miss 必须回退完整 direct；
- hit 支付 PWP 读取、pattern index 和 residual；
- pattern matcher、PWP SRAM、residual 和 projection 分账。

Phi-like 行全部为 `[模型]`，不是官方 Phi artifact 结果。

---

## 4. 双线同约束结果

统一配置：

- direct score 为 32 lane；静态锚点为现有真实的 `32+W` 结构，默认 W4；
- 额外报告 `raw speedup × 32/(32+W)` 的 score-lane 归一代理；它不是 DC 面积归一；
- projection 3 bank；
- 每 bank 256 bit/cycle；
- activation SRAM 128 bit/cycle；
- 500 MHz 只用于换算延迟，不是 STA 结果；
- 100 个同 cohort 样本，报告 mean/p95/p99。

### 4.1 主要结果

| 主线 | 方案 | mean cycle | p99 | 相对 direct | fabric metadata |
|---|---|---:|---:|---:|---:|
| Local5 T162 | direct | 6,843,094 | 7,529,013 | 1.000× | 0% |
| Local5 T162 | online-matcher oracle | 6,725,042 | 7,321,226 | 1.018× | 10.96% |
| Local5 T162 | fixed64 static anchor+term | 4,322,832 | 5,546,063 | 1.583× | 40.64% |
| Local5 T162 | ISHD 编码候选 | 4,322,832 | 5,546,063 | 1.583× | 11.96% |
| Motion T162 | direct | 1,494,242 | 1,667,598 | 1.000× | 0% |
| Motion T162 | Prosperity-online oracle | 1,802,126 | 1,868,318 | 0.829× | 5.95% |
| Motion T162 | fixed64 static anchor+term | 832,495 | 948,810 | 1.795× | 13.91% |
| Motion T162 | ISHD 编码候选 | 832,495 | 948,810 | 1.795× | 6.02% |

资源归一代理：

| 主线 | raw speedup | score lane 等效数 | lane 归一代理 | 当前门槛 |
|---|---:|---:|---:|---|
| Local5 T162 | 1.583× | 36 | 1.407× | 周期通过；metadata `11.96%` 不通过 |
| Motion T162 | 1.795× | 36 | 1.595× | 模型门槛通过 |

解释：

- `Prosperity-online` 已改名为 online-matcher oracle。它被允许达到相同 exact-term 投影复用并共享 K 读取，但并未调用 Prosperity 官方 simulator。
- Local5 明确区分 `13,732,741` 个唯一 term 和 `153,748,435` 个 destination group；修正后只剩 `1.583×` 模型收益。
- ISHD 不改变当前单发射周期，只降低 fabric bit；Local5 仍为 `11.96%`，没有通过 10% 门槛。
- Motion score 使用真实 ordered delta backlog，projection 使用 profile100 M4 delivery，并已禁用额外 PPDI；证据强于 Local5，但仍不是 cycle-accurate RTL。
- Local5 的 p99 只是样本间 histogram 服务下界分布，不是 ordered FIFO p99；不能用于尾延迟宣传。

### 4.2 Phi-like break-even

| 主线 | Phi-like 超过 ISHD 所需 L1 hit |
|---|---:|
| Local5 | 约 41% |
| Motion | 约 47% |

结论：

- **Local5**：在当前理想模型下 Phi-like 达到约 41% 命中即可越过 ISHD，但这尚未计真实 codebook 训练、test split、matcher RTL 与 PWP SRAM，不能解读为 Phi 已胜出。
- **Motion**：Phi-like 值得保留为竞争候选，但必须从真实 temporal Q/K/投影 trace 学 codebook；未得到 ≥47% test-cohort hit 且未计入 PWP 面积前不做 RTL。

### 4.3 DSE 推荐点

联合扫描：

- residual lane：2/4/8；
- weight SRAM：128/256 bit/bank/cycle；
- continuation：4/6/10 bit。

| 主线 | 候选数 | 过模型门槛 | 平衡推荐点 |
|---|---:|---:|---|
| Local5 T162 | 18 | 0 | 无：fabric metadata 最低仍为 `10.43%` |
| Motion T162 | 18 | 18 | W8、SRAM128、delta6；raw `1.791×`、lane 归一 `1.433×` |
| Local5 T450 外推 | 18 | 0 | 无，等待真实 profile |
| Motion T450 外推 | 18 | 18 | 同上，等待真实 profile |

选择平衡点而不是模型最激进点：

- Motion W8 raw 周期略优，但 lane 归一后并非压倒性优于 W2/W4；最终点必须由 DC/STA/SAIF 选择。
- Local5 W2/W4/W8 均因 fabric metadata 超过 10% 而不晋级；p99 也不是 ordered 证据。
- 不再扫描假设 fanout，避免把无法观测的参数当成设计空间收益。
- delta4/6/10 目前只是位宽敏感性；真实 escape rate 尚未统计。

---

## 5. 可写成 DATE 贡献的架构主张

### C1：静态拓扑锚定的 exact residual execution

借鉴：

- Prosperity：exact/partial product reuse；
- Phi：公共成分 + residual 两级执行；
- FireFly-T：稀疏 lane compaction。

本土化：

- Motion 使用同一 token 的 T0/T1 作为时间锚点；
- Local5 使用中心 self-K 作为空间锚点；
- 不运行在线 matcher，不做近似 pattern 选择；
- residual 超阈值直接走 full fallback，保持 bit-exact。

当前证据：

- Motion profile/ordered：强；
- Local5 pre-G0 profile + TARE RTL：中，等待 post-G0。

### C2 候选：ISHD 身份驻留层级描述符

借鉴：

- Phi 的层级表示；
- StreamTensor 的迭代空间描述；
- Prosperity 的 product identity reuse；
- NoC header/body packetization。

本土化新意：

- L1 驻留的是精确 `{gate,lane,multiplicity}`，不是近似 activation pattern；
- L2 是有序 destination delta，不是数值 residual；
- product 只在 header 首拍计算，continuation 只更新目的 Acc；
- exact escape 保证任意间隔；
- T162/T450 只改变 destination 解码，不改变 term 语义。

独立审稿认为 header/continuation/delta 本身属于常规编码，而且旧 DCTF 已有 term identity 驻留。因此当前不能单列 DATE 贡献，只能并入 C1，并通过以下证据争取晋级：

- 独立 ISHD header/continuation/escape RTL；
- 真实 destination delta 与 escape rate；
- fixed64/ISHD 同周期、流量、DC/STA/SAIF；
- 在已计 continuation flags 和 fallback 后仍有至少 15% EDP 或面积归一吞吐收益。

### C3：TTB/STT 统一拓扑描述符

借鉴 Bishop 的 TTB，但不照搬异构双核：

- Motion TTB：打包相邻 token 的两时间片；
- Local5 STT：打包中心 token、有效方向 mask 和 halo 状态；
- descriptor 在进入 score engine 前就携带 exact invalid/empty 信息；
- empty 只跳 payload，不删除 Shiftmax 分母语义。

当前边界：

- Motion ordered TTB 可进入 RTL；
- Local5 STT-empty 必须由 post-G0 ordered trace 统计，暂不声称 skip 收益。

### C4：SCS 到 term 的 exact algebraic handoff

- SCS 保留为共享归一化后端，不单独冒充系统架构；
- score class、gate code、term key 形成有限离散域；
- K-zero 通过闭式 class-count 注入，不能简单删除；
- 输出直接形成 term header，避免物化 gated-K 张量。

该贡献必须与 C1/C2 共同叙述，单独的 SCS-Shiftmax 仍只是算子微架构。

---

## 6. 两条线的下一步

### 6.1 Motion

1. 完成 fullres hardware-order 数值；
2. T450 `DEST_W=9`、row loop、SRAM depth、accumulator depth 回归；
3. 从真实 trace 统计 Phi pattern hit、PWP 容量和 residual；
4. 实现 W2/W4/W8 `32+W` TARE，对 raw 与 lane/PPA 归一结果分开；
5. ISHD header/continuation/escape RTL；
6. fixed64 与 ISHD 同 SDC/DC/STA/SAIF。

Phi-like 只有满足以下条件才晋级：

- test/profile cohort L1 hit ≥47%；
- 相对 Motion ISHD，traffic 或 EDP 至少改善 15%；
- pattern/PWP SRAM 不使面积增加超过收益；
- exact oracle 零失配。

### 6.2 Local5

1. fullres Q7/Q1.7、整数 LUT Shiftmax、真实 invalid mask；
2. post-G0 ordered trace，绑定真实 config/checkpoint/cohort/producer；
3. 统计 unique term、destination group、destination delta、escape、目录 overflow；
4. 用真实结果替换 delta6 假设；
5. T450 line-buffer、halo、地址与 accumulator 回归；
6. fixed64、ISHD、native-m 三者同约束 PPA。

Local5 不因当前证据较弱而停止；其准确率和固定 stencil 拓扑都可能形成最终主线。

---

## 7. 淘汰与投稿门槛

| 类别 | 门槛 |
|---|---|
| 正确性 | fullres hardware-order + RTL-exact；逐元素/逐 term 零失配 |
| 性能 | raw 相对 Direct32 至少 1.15×，且 `raw×32/(32+W)` 至少 1.15× |
| 尾延迟 | p99 ≤ 1.25× mean |
| fabric 元数据 | ≤ 总 fabric payload 的 10% |
| 模型校准 | microtrace 上模型与 RTL cycle 误差 <10% |
| PPA | 同 SRAM macro、同 SDC、同 PVT 的 DC/STA/SAIF |
| 系统外推 | full-encoder Amdahl、FPS、energy/frame |
| 写作 | 所有收益按 `[prof]/[模型]/[RTL]/[DC]` 分档 |

当前结论仍是：

> **架构可继续，论文尚未 sign-off。**

尚未关闭的硬门槛是 fullres 双线 exact、Local5 post-G0 ordered trace、ISHD RTL、同约束 PPA 和 full-encoder Amdahl。

---

## 8. 复现

```bash
cd /root/private_data/work/sdformer_codex/SDformer/hw_autoresearch_nts07

/opt/conda/envs/sdformerflow/bin/python \
  scripts/phi_prosperity_dual_line_simulator.py

/opt/conda/envs/sdformerflow/bin/python -m unittest \
  tests.test_phi_prosperity_dual_line_simulator -v

bash sim_local5/run_local5_parity_checks.sh
bash sim_et3/run_et3_native_slice_checks.sh
```

产物：

- `results/phi_prosperity_dual_line_sim_20260729/report.json`
- `results/phi_prosperity_dual_line_sim_20260729/report.md`
