# M1534｜Motion ep34 无损稀疏、跳过与节能候选独立挖掘

日期：2026-08-31（Asia/Shanghai）  
性质：只读、第一性原理、prior-art-aware 独立审阅  
对象：Motion H67 ep34 / M1458 40-sample capture  
裁决：**保留 3 条 CPU/trace 快杀候选；不授权新 RTL、EDA、GPU 或 SSH。**

本审阅先扫描仓库内 M/G/H/N 已做、已杀和仅支撑机制，再核对公开论文/官方 artifact 的机制边界。没有修改旧文件、论文、`ucli.key` 或 `docs/359_DATE终局冻结_20260813.md`；`docs/359` SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 1. 结论先行

### 1.1 第一性原理判断

H67 当前 exact sparse path 已经按非零 source 发事件。于是，“发现更多零值”只有在同时少付下列至少一项时才有硬件意义：

1. 不发 descriptor / NoC payload；
2. 不读 weight row/tile；
3. 不占用 source issue / Acc24 update；
4. 不读写中间 activation / psum；
5. 不让未选 bank、decoder、adder tree 切换。

若只把 dense MAC 计数改小，而 source issue、weight fetch、psum update、commit 和 metadata 仍相同，本审阅强制 `KILL`。

在当前 C1/C2 边界下，每个有效 source 至少要产生一次语义贡献。无损路径继续减少计算的合法方法只剩三类：

- **跨 token 复用同一 weight row**，不删除贡献；
- **压缩 source/weight 传输格式**，不改变贡献；
- **证明某个 weight block 精确为零**，同时抑制 weight fetch 与 update。

第一类最值得测；第二类更可能给 energy/traffic，而不是周期；第三类高度依赖尚未交付的 ep34 INT8 权重，预期快杀概率高。

### 1.2 最终候选

| 优先级 | 候选 | 无损对象 | 直接 prior | 当前裁决 |
|---|---|---|---|---|
| **P0** | **TSBG：Typed-Signed Bundle-Gustavson** | 相邻 token/window 中同 input-channel 的 weight row 只取一次，向多个 typed signed source/destination context 广播 | ELSA bundled AER + mini-batch spiking Gustavson；SpikeX 跨时空 weight reuse；SCNN/SNE 家族 | **GO_CPU_FASTKILL_ONLY** |
| **P1** | **ACES：Adaptive Codeword/Event Stream** | 每个 96-channel group 在 bitmap、index-list、layer-codeword bitmap、typed escape 中选最短 exact 格式 | FireFly-T multi-lane bitmap decoder；event-DNN bitmask/CSR；ELSA BAER；ESDA sparse token interface | **GO_TRACE_FORMAT_FASTKILL_ONLY** |
| **P2** | **LBWC：Lossless Block-Width/Zero Compression** | INT8 weight tile 用最小有符号位宽或全零标签传输，解码后逐位恢复原 INT8 | CICC bit-width-aware compression；SpikeX activation-induced weight tailoring；sparse-weight accelerators | **BLOCKED_EP34_INT8_THEN_FASTKILL** |

不保留第四、第五条。bank clock gating、BN materialization-elision、FC1 held-weight、C1 parent capture 已是既有实现/支撑；把它们改名不是新候选。

## 2. 当前收费项与 ep34 证据

M1458 `operator_runtime.json` 含 79 个 Conv2d/Linear module aggregate。它不是周期表，但足以确定 profiling 优先级：

| 类别 | module 数 | ep34 input activity | dense MAC proxy | activity-weighted MAC proxy | 解释 |
|---|---:|---:|---:|---:|---|
| patch family | 8 | `17.3591%` | `12.2801 T` | `0.6666 T` | 计算量大；部分输入不是 binary，必须 typed escape |
| FC1 | 12 | `12.3097%` | `3.3974 T` | `0.3843 T` | 高份额、相同权重跨 token 重用，最适合 TSBG |
| FC2 | 12 | `3.1536%` | `3.3974 T` | `0.1134 T` | 极稀疏，最适合 ACES；但 C2 已有 bit-packed ingress，必须对强 baseline |
| bottleneck/other Conv | 8 | `9.5419%` | `2.5565 T` | `0.2576 T` | C1 已吃 product reuse；不得再做通用 matcher |
| Q/K projections | 24 | `6.1022%` | `1.9907 T` | `0.1182 T` | 可作 transport 对照；attention 不作系统主点 |

这些 activity 数只是输入非零比例，不是 skip rate、cycle speedup 或能量。M1458 对 FC/patch 只留 ordered statistics，没有逐 token/channel payload，因此不能直接计算 bundle occupancy、index-list 字节或 bank activity。

## 3. 仓库机制扫描：哪些不能改名复活

| 机制 | 冻结证据或第一性原理 | 裁决 |
|---|---|---|
| C1 finite-capacity product capture | 旧 pre-rebind CPU same-ledger `1.759x`；240-KiB-class 单口 parent/PWP、dead-write/completion 已是主线 | 保留 C1；不再开第四个 Conv matcher |
| C2 typed signed K8 | 等带宽 K8/K1x8 周期 `1.0167x`，吞吐/mm2 `4.541x`，logic area `-77.61%` | 保留 C2；新候选只能成为它的 memory/format 子机制 |
| M24 temporal bitmap/cohort fusion | 需要完整 T10 cohort census；现有 aggregate 不能重建全 temporal mask | 不把 ACES 写成 M24；ACES 只改单 group 传输格式 |
| M231 / NS-INJECT | ep35 已证明 sn2→FC2 是 binary；bridge 已做 raw4 transpose。相对 fused-direct 强基线性能严格接近 `1.0x` | 仅 integration/traffic 支撑，不是新 novelty |
| M501 / APEC / ExSpike | validation event work `1.3796x`，理想 envelope `1.0366x`；当前 trace 是 positive two-codeword，机制直接撞 ExSpike | 不复活；只作 TSBG 的负/对照点 |
| G7 amplitude gate | bottleneck 输入近似 `{0, layer-constant}`，没有中间幅值 Pareto | KILL |
| G8 whole-FFN/token skip | 冻结可执行 tau 网格 skip 为 0；大 tau 是 post-hoc oracle、无 AEE | KILL |
| G10 empty tile | 空 output-site 约 `0.1117%`；task 空率不足以形成主性能 | KILL as speedup |
| G11 cumulative source budget | 对 binary 值域与已有 zero-source skip 重合，收费端口后未过门 | KILL |
| G12 ATLIF early stop | term `-6.58%`，实际 32-lane issue 仅 `-0.0676%` | KILL |
| G15 / lazy-PWP / H1 payload residency | 同资源约 `1.07x / 1.037x / 0.83x` | KILL as performance |
| M523 decoder parent packing | 真实约 `1.007--1.034x` | decoder support only |
| decoder temporal XOR / N2 | delta/full 加权 `1.352x`，更密 | KILL |
| FC2 dense/sparse split | 冻结 tile 上 dense winner 为 0 | KILL |
| FC1 held-context / M229--M262 | 已有 VCS/DC/CPU DSE，属于 C2/FC1 既有机制，不是 fresh idea | 继续收口；不改名 |
| dynamic-BN no-materialization / M160/M475 | exact 代数与 DSE 已立项；current-batch barrier/raw retention 必须收费 | 既有 C3/C2 支撑，不重复提案 |
| generic multi-lane sparse decoder / OOO bank dispatch | FireFly-T 直接 prior，C2 K8 已覆盖主要对象 | 不新开 RTL |
| submanifold sparse Conv | ESDA 的网络/算子语义；H67 标准 Conv 输出几乎非空，不能硬件自行改语义 | NO-GO without algorithm retraining/rebind |
| zero descriptor → zero weight fetch | ExSpike/SNE 已覆盖，C2 已有 source descriptor 基建 | 补 assertion/transaction counter，不算新机制 |

## 4. P0｜TSBG：Typed-Signed Bundle-Gustavson

### 4.1 直接 prior

- [ELSA](https://arxiv.org/abs/2605.20802)：bundled AER 与 mini-batch spiking Gustavson product，减少通信和 memory access。
- [SpikeX](https://arxiv.org/abs/2505.12292)：NTWU 调度、跨 time-window / postsynaptic-neuron 的多级 weight reuse、activation-induced weight tailoring。
- [SNE](https://arxiv.org/abs/2204.10687)：event-proportional sparse convolution 与 resident neuron state。

因此不能宣称发明 Gustavson、bundle、AER 或跨 token weight sharing。

### 4.2 H67 对象/协议差

H67 可主张的差异是：

1. bundle 成员不是纯 binary spike，而是 C2 的 `{source_index, signed_value/codeword, token/context, terminal}`；
2. 一个 weight row 读取后，广播给多个独立 Acc24 destination context；相同 source index 但不同 signed value 仍复用 weight fetch，不能错误复用 product；
3. 与现有 K8、96-output-lane、240-KiB、相同 SRAM bank/port/BW、有限 context 数和 exact completion 绑定；
4. 若成员 value/codeword 相同，可选择 product multicast；不同则只复用 weight row。

这不是 C1 重复：C1 在同一 Conv task 内捕获 activation-parent/product；TSBG 在 FC1/FC2/patch 的多个 token/context 间复用静态 weight row。也不是 M501：不做相邻 feature-map overlap partial sum。

### 4.3 当前能测与缺失

M1458 只能给 module/call activity aggregate，不能给真实 bundle overlap。需要一次轻量增量 capture：

- FC1/FC2/patch 的 per-token、per-96-channel-group support bitmap；
- 非零 INT code/sign/non-unit 标记；
- token/window/spatial order、consumer output-tile identity；
- weight row address/bank key；
- strong baseline row-buffer hit/miss 或可重建地址序列。

无需重训，也不应重做 M1458；绑定相同 ep34 checkpoint、40 sample order 和 outer provenance。

### 4.4 CPU 快杀门

扫描 bundle `B={2,4,8}`，两臂使用相同 K8/K1x8、96 lane、bank/port/BW、240 KiB 与普通 row buffer。

强制计费：bundle builder/search、metadata、row-buffer、多个 destination context、bank conflict、tail、update、commit、queue/backpressure。

晋级条件：

1. contributor multiset、Acc24、output 0 mismatch；
2. FC1+FC2 ratio-of-sums cycle `>=1.15x`，每 sequence `>=1.05x`；**或**周期退化不超过 5%、weight bytes `>=30%` 下降且 memory-energy proxy `>=20%`；
3. signed/non-unit source 占 admitted bundle `>=5%`，否则只可称 ELSA-style binary workload mapping；
4. ordinary row buffer 必须是 baseline common resource，禁止用无 cache 弱 baseline。

论文位置：**C2 memory specialization**，不新增第四条贡献。

## 5. P1｜ACES：Adaptive Codeword/Event Stream

### 5.1 直接 prior

- [FireFly-T](https://arxiv.org/abs/2505.12771)：multi-lane bitmap decoder 与 index reuse。
- event-based DNN flexible encoding：bitmask / CSR 根据密度选择。
- ELSA BAER：多个事件共享 header。
- [ESDA](https://arxiv.org/abs/2401.05626)：统一 sparse token-feature interface。

所以“bitmap/CSR 双格式”“bundle header”本身没有 novelty。

### 5.2 exact 格式

对一个 96-channel source group，写端 sidecar 在四种格式中选**实际物理字节最短**者：

1. `BITMAP_UNIT`：96-bit bitmap，value 由 layer contract 隐含为 `+1`；
2. `BITMAP_CODEWORD`：96-bit bitmap + 一个 signed INT codeword；
3. `INDEX_TYPED`：count + active index + 每 source typed value/code；
4. `RAW_ESCAPE`：原始既有 C2/raw4 格式。

header 固定携带 format、token/context、terminal 和 payload length；decoder 输出完全相同的 C2 source tuple。任何非二值/多 codeword group 自动 escape，不能按 sampled binary ratio 猜测。

H67 的对象差是“layer-codeword + typed signed escape + C2 atomic terminal”，不是发明 sparse encoding。它不重复 M24：M24 试图跨 T10 temporal cohort 复用 coefficient；ACES 只压当前 group 的 source transport。它也不重复 M231：M231 只有固定 binary raw4 transpose，没有按 packet 选择 physical encoding。

### 5.3 第一性原理上限

FC2 aggregate activity `3.15%`、FC1 `12.31%`，说明 index-list 可能在 FC2 有流量价值；但固定 96-bit bitmap 已经很紧凑：

- unit-binary list 每 source 至少需要 7-bit index，未计 count/header；仅当 `k` 明显小于约 13 时才可能胜过 96-bit bitmap；
- typed list 还要 value bits，crossover 更低；
- bitmap decoder 已存在，多格式 mux/length parser 可能吃掉节能。

因此 ACES 默认是 traffic/energy 候选，不预设周期加速。

### 5.4 快杀门

需要与 TSBG 相同的 per-group bitset/value 增量 capture。报告每 module/sequence 的四格式占比、payload/header/padding/ECC、decoder work、source tuple 守恒与实际 SRAM/NoC transactions。

晋级条件：

1. 逐 tuple exact 0 mismatch，escape coverage 非零；
2. 相对现有 bit-packed/raw4 **强 baseline**，FC1+FC2+patch transport bytes `>=30%` 减少；
3. 计入 sidecar/format header/padding 后 SRAM/NoC energy proxy `>=20%`；
4. exposed cycle 不得回退超过 5%；若 ingress 是 max() 且同资源周期 `>=1.10x`，才允许写局部 latency；
5. 若收益仅来自 FP32 弱 baseline，立即 KILL。

论文位置：**C2 descriptor/transport ablation**。即使通过，也不单列 novelty。

## 6. P2｜LBWC：Lossless Block-Width/Zero Compression

### 6.1 直接 prior

- CICC 2026 optical-flow accelerator：bit-width-aware compression 是直接先例；这里只借 lossless weight transport 思路。
- SpikeX：activation tag 在 memory hierarchy 抑制无用 weight fetch。
- sparse-weight accelerator / SCNN family：全零 weight block 不加载、不执行。

因此“按位宽压权重”“全零 block skip”不能称首创。

### 6.2 H67 对象/协议差

对 ep34 **正式 INT8 权重**按 C1/C2 真实 service tile（候选 `1x96`、`8x16`、`16x16`）计算：

- 最小 two's-complement bit width `w in [1,8]`；
- `w=0` 表示 exact all-zero block；
- 物理 payload 为 `{width, length, packed signed weights, ECC/padding}`；
- synchronous decoder 恢复逐位相同 INT8 row，再送现有 256-bit/96-lane service；
- all-zero block 同时 suppress weight fetch、source issue 和 psum update；非零窄 block 只省 bytes/energy，不伪称少算。

它不重复 C1：C1 复用运行时 product/parent；LBWC 只改变静态 weight payload。它不重复 C2：C2 当前比较的是共享 endpoint/Acc24，未给 compressed weight memory。它也不重复 M102：M102 baseline service 按固定 INT8 三 beat 读 96 B，LBWC 正是对该固定 payload 的后续 matched compression。

### 6.3 当前阻塞与快杀门

M1526 已证明 ep34 selected config 的 `hardware_quant_enabled=false`；旧 M61/ep35 量化不能继承。必须等 ep34 Q1--Q4 INT8 authority、四层 deterministic PTQ、S40 kernel/ATLIF miter 和 Acc24 proof。当前不得从 FP32 checkpoint 猜 INT8 width。

权重一到，CPU 审计非常便宜。晋级门：

1. decode 后 INT8 bytes、Acc24 和 output 0 mismatch；
2. 含 width header、block directory、burst padding、ECC 后，weight SRAM/DRAM bytes `>=20%` 减少；`>=30%` 才可进主消融；
3. all-zero block 若不足 `5%`，zero-block skip 自动 KILL，只保留 width compression；
4. decoder/gearbox 使 exposed cycle 回退不得超过 5%；
5. memory-energy proxy `>=15%`，否则不做 RTL；
6. 若 96-weight row 中每行都有 7/8-bit 极值，按真实 service tile无法压缩，立即 KILL，不切换到 post-hoc 极小 tile 逃门。

论文位置：**C1/C2 common weight-store support**，不是第四贡献。

## 7. 明确拒绝的“看起来像新 idea”

### 7.1 SpikeX hierarchical activity tags

H67 source stream 已经按 active source 发出，M468 strong-zero 也给 empty task 相同 weight-fetch policy。若 tag 只在 source decoder 后出现，它不再减少 weight read；若在写入时生成 sidecar，则本质并入 ACES/TSBG。单独立项会重复 zero-source skip。

### 7.2 FireFly-T K8/OOO/load balance

C2 已有 typed K8 和等带宽 K1x8。新 OOO scheduler 只能在真实 bank conflict 是 max() 时有用；当前同带宽周期已接近 parity，主收益是 `4.541x throughput/mm2` 和 `-77.61%` area。继续做通用 OOO 是 prior-art 重复。

### 7.3 ESDA submanifold sparse Conv

submanifold Conv 会改变网络的 active-coordinate/output support 语义。H67 decoder destination 几乎全非零，bottleneck/patch 也没有“硬件可以自行保持 sparse support”的等价定理。没有重训、checkpoint rebind 和 AEE，就不是无损硬件优化。

### 7.4 SNE-style event-only state

普通 LIF 的 event-driven state 更新不能直接替代 H67 full-rank Fixed-T10 temporal transform。C3 已有 exact service；若不读取全部 T10 输入便输出，必须先证明矩阵结构，不可套用 resident LIF state。

### 7.5 Phi pattern/PWP 再包装

M70--M76 已深入测试 Phi-like catalog/PWP。nominal pattern-op、pattern table、matcher、端口和 residual/correction 的 gap 已有证据；不能用新 catalog 名称复活。Phi 继续作为外部/消融对标。

### 7.6 Prosperity / ExSpike 再包装

C1 已占 Prosperity product capture 对象；M501 已是 ExSpike APEC 直接 workload audit。新的 exact sparse 候选必须减少 C1 之外的物理 bytes/energy，不能再次用 overlap/event count 取名。

## 8. 48 小时筛选顺序

### 0--8 小时：只做 schema 与现有数据上限

1. 定义 TSBG/ACES 共用 ep34 增量 capture：per-token/channel support、INT code、order、weight address/bank key；不重做 M1458。
2. 用 M1458 aggregate 给 bundle/format 上限图，明确标 `independent-density proxy`，不写 speedup。
3. 准备 LBWC 权重审计器，但保持 fail-closed，直到 ep34 INT8 authority 到位。

### 8--24 小时：一次增量 capture 后先快杀

1. ACES 先做 bit-exact format byte census；过不了 `30% bytes` 立即 KILL。
2. TSBG 做 B2/B4/B8，baseline 带同容量 ordinary row buffer；过不了 cycle/bytes 二选一门立即 KILL。
3. 两条都只跑 CPU/address recurrence，不碰 RTL/EDA。

### 24--48 小时：最多升一条

1. TSBG 若 cycle `>=1.15x`，优先升为 C2 memory specialization；否则若只省 bytes/energy，降为支撑。
2. ACES 若只相对 FP32 有收益，永久 KILL；相对 bit-packed/raw4 仍 `>=30%` 才保留。
3. ep34 INT8 到位后跑 LBWC；只需几分钟静态扫描，未过门直接封负结果。
4. 三条最多一条进入最小 RTL；不得阻塞 ep34 decoder-complete Table-A、C1 rebind 或生产 power。

## 9. 论文写法与最终裁决

推荐贡献结构不变：

1. **C1 constrained product capture**：性能主点；
2. **C2 typed signed source service**：等带宽面积效率；TSBG/ACES 若过门，只作为其 memory/transport specialization；
3. **C3 exact temporal/system closure**：Fixed-T10、BN、decoder 完整性。

LBWC 若通过，放 C1/C2 common memory ablation。三条候选都不得变成第四、第五个并列 novelty。

独立评分：

| 候选 | 新颖性上限 /5 | 预计意义 /5 | 数据就绪度 /5 | 判断 |
|---|---:|---:|---:|---|
| TSBG | 3.2 | 3.8 | 2.5 | 最值得测；差异必须由 typed signed bundle 覆盖与同资源 bytes/cycle 证明 |
| ACES | 2.7 | 3.2 | 2.4 | 可能节能；编码 prior 很直接，不能单列贡献 |
| LBWC | 2.5 | 3.1 | 1.5 | 便宜快杀；INT8 identity 未到，且量化极值可能让位宽压缩失效 |

最终裁决：**这三条是“可筛选机制”，不是已经成立的新创新。真正有机会增强 DATE 的只有 TSBG；ACES/LBWC 更像让 C2 的 energy/traffic 表完整。任何点若只减 operations 而不减同资源 cycle、physical bytes 或 energy，立即封死。**

## 10. 证据与公开来源

仓库主要证据：

- M1458 ep34 capture `operator_runtime.json`；
- M1529 ep34 novelty/skip first-principles review；
- M475 non-Conv sparse opportunity audit；
- M501/M507 ExSpike-APEC reviews；
- M735 NS-INJECT/M231/C2 audit；
- M102 bit-sparse physical baseline review；
- open-RTL port/schedule mining review；
- M160/M231/M375/M386/M462/M468/M501/M523/M709 evidence chain。

公开来源：

- Prosperity, HPCA 2025: https://arxiv.org/abs/2503.03379 ; official https://github.com/dubcyfor3/Prosperity
- Phi, ISCA 2025: https://arxiv.org/abs/2505.10909
- FireFly-T: https://arxiv.org/abs/2505.12771
- ELSA: https://arxiv.org/abs/2605.20802
- SpikeX: https://arxiv.org/abs/2505.12292
- SNE, DATE 2022: https://arxiv.org/abs/2204.10687
- ESDA, FPGA 2024: https://arxiv.org/abs/2401.05626
- ExSpike, FPL 2026: https://arxiv.org/abs/2606.20414 ; official https://github.com/xiaoyuehai/ExSpike

