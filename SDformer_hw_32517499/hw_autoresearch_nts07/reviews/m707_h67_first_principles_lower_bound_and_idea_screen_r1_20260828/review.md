# M707｜H67/Motion 硬件第一性原理下界与新 idea 筛选

日期：2026-08-28  
模式：独立、只读证据审阅；未运行 GPU/EDA/remote，未写 RTL/论文正文  
裁决：**现状 3.45/5，Borderline Weak Accept/Weak Reject，不是 Strong Accept。保留 C1/C2/C3+RQTB，只开两个 CPU fast-kill，不授权新 RTL。**

## 1. Answer-first

当前硬件并非“每层一个加速器再相乘”，而是三条主贡献和一个次贡献：

1. **C1：exact parent/product capture。** 四个 bottleneck Conv 的同账本 CPU 候选为
   `435,293,339` cycles，相对 strong-zero/same-bit 为 `1.746753x/1.741232x`。
   这是当前唯一同时具备较大局部倍率、公平内部基线和完整守恒的性能主点；但仍是
   E2 CPU 候选，不是 RTL/PPA/system。
2. **C2：shared typed K8 signed-source fabric。** `4.7642x` 只相对单 K1 的低服务
   分母；公平 K1x8 的 directed VCS 中 K8 只有 `1.0122--1.0392x`。C2 应卖共享
   state、协议和未来面积/能量，不应继续卖“等资源 4.76x”。
3. **C3：phase-decoupled ATLIF service。** M518 r11 证明 Fixed-T10 directed 行为和
   `17 cycles/tile` issue anchor；当前没有 admitted speedup，rank3 也未在 ep35 真负载
   上准入。它暂时是协议创新，不是性能 headline。
4. **RQTB：次贡献。** `1.186509x` local、`+1.2559%` logic area、
   `1.171792x` throughput/area；旧 envelope 仅 `1.000911x`。可写完整性和能量动机，
   不能扛系统性能。

DATE 模拟评分为 Novelty `3.6`、Soundness `4.1`、Validation `3.1`、Significance
`3.5`、Presentation readiness `2.4`，综合 **3.45/5**。不足不主要是“idea 还少”，
而是 C1/C2/C3 没有在同一 decoder-complete、macro-aware、same-resource 表里闭合；
M706 注册表虽然已 fail-closed，但 production Table-A 行仍是 0。

## 2. 下界方法

每个算子统一使用：

`cycles >= max(source issue, weight port, state RMW, dependency, completion, memory)`。

“跳过”只有在删除了这个 max 中的一个实际收费项时才是加速。减少 logical work、
descriptor 或 component energy 不自动减少 total cycles。所有最终 commit 字节若未冻结
输出位宽，只保留为 `N_destination × payload_bytes`，不从 128/192 B/cycle 总线宽度反推。

证据级别：E1=分析/外部机会，E2=exact CPU ledger，E3=VCS，E4=logic-only 或
macro component，E5=集成宏同资源 PPA/能量，E6=decoder-complete 多序列系统结果。
完整机器可读表见 `lower_bound_ledger.json`。

## 3. 八类算子的不可消除下界

| 算子 | source/weight 下界 | state/psum/commit 下界 | 端口与依赖 | 当前机制突破了什么 | 当前裁决 |
|---|---|---|---|---|---|
| bottleneck Conv | 每个 live source identity；每个 live product 的选中权重。M528 实际收费 source SRAM 103.68 MB、weight DRAM 9.069 GB | live parent 必须 store/forward；Acc24 到完成；960k vector commits | 1RW 144 B parent access、8 output banks；classify→parent read/forward→product→Acc24→commit | C1 去掉重复 product 和 dead write；不去掉 source、live read、weight、psum、commit | **KEEP PRIMARY，E2+component E4** |
| FC1 | 829.44M elements、112.214M active、103.68 MB packed；每 active source 的 weight response | held context Acc24、descriptor/credit；BN barrier 前结果语义存在 | L96/F2/C16/B2 candidate；bank response/terminal bound | context factorization 摊销 weight request/lifecycle | **M481 E1；未冻结性能** |
| FC2 | 每个 signed descriptor 和 selected weight response | 18,432-bit context plan、Acc24、atomic terminal、result commit | 8-source/cycle、8 banks；K1x8 是公平 service bound | C2 相对 K1 提峰值；对 K1x8 主要剩共享 state/area/energy | **KEEP PRIMARY，E3** |
| ATLIF/PSN | 所有能改变结果的 temporal term/coefficient | phase/context state、temporal Acc、result frame | 96 multiplier slots；17 issue cycles/tile directed | phase/context overlap；未删除 Fixed-T10 terms | **KEEP PRIMARY AS PROTOCOL，E3** |
| decoder | active source × legal tap × weight vector | dense output Acc24；跨时 delta 还要 prior input/output or downstream state；每 destination commit | phase taps 4/2/2/1，96 lane 已达 exact product lower bound | M522/M523 免 inserted-zero materialization、复用 source descriptor，不减 product | **COMPLETENESS ONLY** |
| attention | 每 unique score class、每独立 live K/value | class/FIFO/K state、attention Acc、row output | score→class→Shiftmax→K read→output | RQTB 去 equal-score duplicate class/Shiftmax service | **KEEP SECONDARY，E3/E4** |
| dynamic BN | 所有元素进 moments；barrier 后 raw replay 或 deterministic recompute | moments、22,080 coeff pairs、140.6--281.3 MiB peak raw retention | producer/raw→global barrier→coeff→replay→consumer | fair fused baseline 已去 normalized write/read；不能再算 novelty | **BASELINE ONLY** |
| patch embed | 每 live RF source/weight | 几乎所有 site 的 psum/commit | scan→weight→psum→commit | bit baseline 已跳 exact zero；whole-T zero site 仅 0.00387% | **NO-GO** |

### Decoder 的关键纠偏

M705 的 `23.3188%` 是三序列选定 S3x10 的 input density，不是周期。旧分析显示 decoder
约占 corrected candidate envelope 的 `21.57--22.83%`，但这同样不是 admitted system
share。ConvTranspose 的 Cout 都是 96 的倍数，现有 exact A1 在 product issue 上已贴住
96-lane 算术下界；新 decoder 机制必须打 source/tap 生成、state bytes、weight residence
或 commit，不能再发明第四种 generic Conv matcher。

### BN 的关键纠偏

M480 的 `1.499892x` 是相对显式 normalized tensor write/read 的弱基线。公平强基线本来
就是 inline moments、raw store、barrier、raw replay、直接喂 consumer，因此公平倍率为
`1.0x`。真正不可消除的是 current-batch barrier 两侧的信息：要么存 raw，要么重算 producer。

## 4. 现有 idea 的合法分母、可组合性和税

| idea | 合法分母 | 不得使用的分母 | 资源税 | 组合规则 |
|---|---|---|---|---|
| C1 M528 | M468 strong-zero、same-coordinate bit，同 S10/row64/B8/128 B/cyc/CAM64 | M473 未物理化 1R1W ceiling 作唯一生死门；系统 headline | 213,376 B；9 macro 78,825 um²；integrated PPA open | 与 C2 必须 joint replay，不乘 `1.74×4.76` |
| C2 K8 | fair cycle 用 K1x8；area/energy 也必须 matched 8 endpoint | 单 K1 的 4.764x 冒充稀疏收益 | shared context/Acc24 + 8-bank interface | FC1/decoder 都复用同 fabric，重叠 source/weight/state |
| C3 | matched Fixed-T10 vs trained/admitted rank path | directed 17 cycles/tile 当 speedup | FIFO/phase/rank state | 与 memory/commit overlap 要统一重放 |
| RQTB | matched Fixed/RQTB component | 全 attention/全网；与别的倍率相乘 | +1.2559% logic；macro/power open | 与 C1/C2 基本算子正交，但系统只 joint replay |
| FC1 M481 | full-width same lane/bank/port bit baseline | official Prosperity 2.373x 当 ours；M481×M619 | 2×1R1W、192 adders、2816-bit descriptor | 需要 FC1→BN→ATLIF→FC2 统一地址账 |
| decoder mapper/bundler | same polyphase workload and same C2 service | density直接当speedup；PDR旧预测 | mapper/descriptor + dense psum | arithmetic不减，只能报 frontend/traffic增量 |

C1 的 parent scratch component 从 all-write `3.301889` 降到 dead-write-only
`2.039633 mJ/frozen sampled inference`，省 `38.2283%`。这证明 dead-write 的能量价值，
但不是 C1 总能量、camera frame 或系统能量；同样不能和 cycle ratio相乘成 EDP headline。

## 5. 只保留两个新机制 fast-kill

### N1｜稀疏 FC1 重算替代 dynamic-BN1 raw retention

- **已有工作与对象差：** checkpoint/recompute 是已有思想；对象改成 H67 高稀疏 FC1
  后的 exact current-batch BN1。第一遍 FC1 只形成 moments，第二遍 deterministic
  recompute 后直接 affine→ATLIF，以额外 FC1 work 换掉 BN1 raw retention。BN2 的
  producer/residual 边界不同，不混入第一轮 fast-kill。
- **依据：** 固定 FC1 和 moment accumulation 顺序时，第二遍 `x` bit-identical，
  `BN(x;μ(x),σ(x))` 不变；global barrier 仍保留。
- **公平 baseline：** M480 strong fused raw replay + 同 M481/K8 lanes/banks/Acc24/bandwidth。
- **一天 CPU 门：** 32/64/128 B/cyc address timing；第二遍 FC1、weight、moment、barrier
  全收费。zero mismatch 且 total FC1+BN1 `>=1.15x`，或 DRAM `-30%` 且 EDP proxy
  `>=1.20x`，才准最小 RTL；否则 KILL。
- **最小 RTL：** 一个 FC1 tile 的 moments-only/recompute selector + direct affine→ATLIF。

这是合理但很可能被杀的 storage/computation trade；不预写倍率。

### N2｜exact decoder temporal-delta state bridge（优先）

- **已有工作与对象差：** 明确引用 DeltaCNN/DeltaRNN。对象差是 exact binary/scaled-
  binary ConvTranspose polyphase 的 `{-1,0,+1}` delta，直接走 M522/M523/C2 更新 analog
  ATLIF/residual 的具名 state；不是改名的 Conv matcher。
- **依据：** 在线性 core 内 `y_t=y_(t-1)+W(x_t-x_(t-1))`。bias 只在初始/重同步
  应用；D1 必须保留 runtime-theta identity。任何 BN/nonlinearity/concat 边界若不能给出
  bit-exact state mapping，立即失败。
- **公平 baseline：** M672 exact polyphase A1/K8，96 lanes、8 source/cyc、Acc24、相同
  weight port、240 KiB；prior input、prior output/downstream state、resync 和 dense commit
  全收费，禁止免费上一帧 output SRAM。
- **一天 CPU 门：** 先证明 downstream state identity，再从 M705 bitpack 统计 delta tap/
  product。需要 `delta_product/A1_product <0.70`，且加入 state/memory 后总周期
  `>=1.20x`。若仍需 20.215/30.322 MiB prior-output state或 D1 mismatch，RTL 前 KILL。
- **最小 RTL：** XOR edge descriptor + signed polyphase tap bundler + epoch/resync checker，
  复用 C2 协议，不写新 scheduler/matcher。

N2 排第一，因为 decoder 是唯一尚未进入原 620M 分母、却在 corrected analytical envelope
中约 22% 的高份额算子，而且它攻击 source/tap work；但 density stability 不等于 delta
sparsity，state bridge也可能被 nonlinear boundary直接杀死。

不提第三个机制：FC1/C1/C2/C3/RQTB 当前缺的是 address timing、matched physical A/B和
系统闭合，不是未占用的算术下界；patch/PDR已有负结果。第三条只会变成机制膨胀。

## 6. Strong Accept 路径

强接收不是再多一个局部 1.1x，而是把现有三主点拉到 E5/E6：

1. C1 唯一 bounded RTL，trace recurrence 在 435,293,339 的 1% 内；九 macro 集成后
   VCS/Formality/DC/STA/PTPX，局部仍 `>=1.50x`，throughput/area `>=1.10x`。
2. M519 matched K1/K8/K1x8 DC/STA，若 equal-service 没有面积/能量优势，就把 C2 从
   性能贡献降为统一执行 fabric。
3. M518 matched Fixed/rank3 A/B；若 rank3没有真实 checkpoint/trace，C3只写 exact
   phase protocol，不写 rank speedup。
4. D0--D3 exact address-timed decoder row，D1 数值桥闭合；至少三序列统一重放。
5. 同一 B0/B1/K1x8/K8/Ours resource manifest 下直接产生 cycle/byte/energy 表；不把
   M472/M619 official opportunity填成 ours。
6. M706 registry 接入真实 native Synopsys evidence；当前 canonical production rows=0。

若上述 1--5 闭合且直接 unified same-resource decoder-complete gain `>=1.10x`（优选
`>=1.15x`），当前 3.45 才有机会进入 `3.9--4.1` 的 Strong Accept 边界。N2 只是可选
增益，不得阻塞这条主线。

## 7. 证据边界

- 未修改 `docs/359`；复核 SHA 为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- 未运行 GPU、EDA、remote、training、官方模拟器或大 CPU job。
- 未写 RTL、论文正文或 performance result。
- 本评审不授权 N1/N2 RTL；只允许 root 后续另建 CPU fast-kill 合同并再次独立打铁。
