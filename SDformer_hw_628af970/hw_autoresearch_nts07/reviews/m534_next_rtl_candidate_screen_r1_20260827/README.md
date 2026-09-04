# M534｜下一条 DATE-Accept 级 RTL 候选独立筛选 r1

日期：2026-08-27  
模式：只读证据与一手文献审计；零 VCS/DC/PT/PTPX/Formality，零 CPU 大任务，零远端执行  
审计结论：`RECOMMEND_A_CONDITIONALLY__RUN_EXACT_DECODER_FASTKILL_BEFORE_RTL__DEFER_B_AS_C2_PHYSICAL_ABLATION`  
审计置信评分：**97/100**；P0/P1/P2：**0/3/2**

## 结论

唯一推荐候选是 **A：phase-banked destination rendezvous（PDR）**，但推荐的是
“先过 exact decoder 快杀、再 author RTL”，不是立即写 RTL。它把公开工作的机制迁到一个
不同的计算对象和完成协议上：对 H67 的 K3/S2 ConvTranspose，按
`(tag,time,destination_y,destination_x,phase)` 汇合来自不同 sparse source 的 signed
`(source_channel,kernel_index)` 贡献；利用确定性的 source-order completion frontier 关闭
destination；再把最多八个不同 weight-bank 的贡献交给既有 C2 K8/Acc24 服务。它不是一般的
polyphase、零插入消除、bundled AER 或 OOO dispatch。

候选 B（FC2 central owner + tag-elided typed ticket）在协议差异上成立，规格也更成熟；但它的
目标被正确限定为周期不变的面积/动态功耗消融。当前只有 `27.5346%` local metadata movement
有利上界，且 M519 K1/K8/K1x8 三轴 DC 尚未封存。B 保留为 C2 子机制，**不是下一条性能 RTL**。

| 候选 | 候选评分 | 当前裁决 | 能进入论文的角色 |
|---|---:|---|---|
| A｜decoder PDR | **82/100** | `CONDITIONAL_NEXT_RTL_AFTER_M511_M513_FASTKILL` | 过门后并入 C2，作为 decoder-complete signed-source destination join；可争取局部周期与 psum 能量 |
| B｜FC2 typed-ticket tag elision | **74/100** | `DEFER_UNTIL_M519_THREE_AXIS_DC` | 只作 C2 protocol/physical-efficiency 子机制；不增加第四项贡献 |

不能把 A 的 dense-to-bit `约 4.48--4.81x` 分析机会写成我方创新；也不能把 B 的
`27.53%` movement 上界写成 area/power/energy 或 cycle speedup。

## 1. 公开机制与合法迁移边界

### 1.1 直接前作

| 原工作 | 原机制 | 对本项目的边界 |
|---|---|---|
| [Prosperity, HPCA 2025](https://arxiv.org/html/2503.03379)；[官方 artifact](https://github.com/dubcyfor3/Prosperity) | 在线发现 spike row 的 subset/similarity，复用 inner product；TCAM 加速匹配 | PDR 不做 subset matcher，不复用近似/相似 row；只合并**同一 exact destination** 的线性贡献。若最后只是 product grouping，必须引用 Prosperity，不能称全新 sparsity |
| [Phi, ISCA 2025](https://arxiv.org/html/2505.10909) | 离线 pattern/PWP 加在线 `{+1,-1}` residual；PAFT 提高 L2 sparsity | PDR 不引入 pattern catalog、PWP 或 PAFT；signed source 只是 exact weight contribution 的代数符号。不能把 Phi 的 signed residual 迁名为我方 novelty |
| [FireFly-T](https://arxiv.org/html/2505.12771) | multi-NZ decode、weight dispatch、OOO worker 调度以缓解 bank conflict | PDR 的差异必须是 **destination-keyed rendezvous + ConvTranspose completion frontier + phase-psum banking**；一般 bank-aware dispatch/OOO 已被占位 |
| [ELSA](https://arxiv.org/html/2605.20802) | BAER 按 row 分摊公共 header；mini-batch Gustavson-product 将同 row spike 合批，降低 membrane 访问 | PDR 不是 NoC packet bundling：它按 exact destination 收集跨 source/tap 的贡献，并用几何 frontier 证明 partial group 可关闭。相同之处必须明示引用 |
| [OpenEye paper](https://arxiv.org/abs/2606.01450)；[RTL](https://github.com/Learning-Chips-Lab/OpenEye) | sparse stream constructor、variable-length FIFO、row-stationary dataflow 与参数化路由 | M514/M523 已覆盖 stream construction/FIFO；PDR 的 claim 不能落在“生成 sparse stream”或“可变长 FIFO”上 |
| [Transposed-conv decomposition](https://arxiv.org/abs/2205.02103) | 将 stride transposed convolution 分成 polyphase 子问题，避免零插入 | K3/S2 `4/2/2/1` phase 分解和不物化插零只作强基线/完整性，不是创新 |
| [SNE](https://arxiv.org/abs/2203.12437)、[ESDA](https://arxiv.org/abs/2401.05626) | event-proportional sparse convolution、event sparse token-feature interface | binary event descriptor 和 event-proportional work 是基线；只有 H67 的 destination completion/phase-bank/C2 typed-source 协议差异可写 |

### 1.2 为什么 A 不是“换名字”

A 的可投稿对象差、协议差、资源差必须同时存在：

1. **对象差**：原始 C2 为 FC2 的同一 output vector 聚合 signed sources；PDR 将其迁到
   ConvTranspose scatter 后的同一 destination/output-channel vector，source identity 变为
   flattened `(source_channel,kernel_index)`。
2. **协议差**：sparse input 没有显式零事件，partial destination 不能因“暂时没有 descriptor”就
   提前提交；PDR 必须由 canonical `(time,y,x,channel)` frontier 或 stream fence 证明该
   destination 的最后可能 contributor 已经过界。
3. **资源差**：四个 output-parity psum banks、八个 128-bit weight banks、同一 96-lane Acc24
   pool、同一有限 join capacity；收益来自减少 destination RMW/commit 和空闲 bank，而不是增加
   lane、端口或 SRAM。

缺少任一条时，A 只能降为 M514/M523 的 decoder completeness，不是新贡献。

## 2. 候选 A｜decoder phase-banked destination rendezvous

### 2.1 已冻结事实

- [M510](/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/docs/510_H67反卷积覆盖缺口与EPD立项裁决_20260827.md)
  证明旧 `620,302,905 cycle/frame` 漏掉四层 ConvTranspose；分析界给出 decoder share
  `21.57--22.83%`，但 exact decoder trace 尚缺。
- M514 exact mapper 已有独立 VCS 与 standalone 3 ns DC：mapper cell area
  `383.670001 um2`、442 cells、setup/hold slack `+1.4266/+0.0106 ns`；它只证明地址控制成本。
- [M523 descriptor VCS](/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/reviews/m523_c2d_k8_polyphase_tap_bundler_vcs_receipt_blind_hammer_r1_20260827/m523_c2d_k8_polyphase_tap_bundler_vcs_receipt_blind_hammer_r1_20260827.md)
  为 `100/100, P0/P1/P2=0`，证明 6 events/43 taps/8 transport bundles 的 descriptor-only
  行为；它**没有** flattened weight identity、bank-conflict deferral 或 stored-weight proof。
- M513 静态数学审计已说明：简单 PGPR 在强 96-wide A1 上 product-issue ceiling 为 `1.0x`。
  因而 A 必须以 destination RMW/commit 与 bank utilization 取胜，不能再报“同 product 更快”。

### 2.2 同资源 baseline

`A1-ISO8` 必须与 PDR 同 top、同 trace、同 8 weight banks、同每 bank 1R1W/L4、同 96 Acc24
lanes、同四个 phase-psum banks、同 output commit、同 join-buffer bit budget：

- `A1-ISO8`：M514 exact taps；八 lane source-order intake；只允许当前 head/相邻 exact-destination
  合并；相同 flattened-key bank 冲突顺序延期；不得物化插零；不得多给 psum 端口。
- `PDR`：在同容量 window 中按 destination/phase 做 bounded join；每次最多选择八个不同
  weight-bank source；frontier-close 后 partial group 必须提交；不得丢、重、跨 tag/time。

辅助分母 `A0 dense polyphase` 可展示 binary activation sparsity opportunity，但 PDR 的我方倍率只报
`PDR/A1-ISO8`。不得用单 bank K1 或 naive zero-insertion baseline 做唯一 headline。

### 2.3 一天内快杀门

先用 M511 verified S10 payload + M513 exact scatter reference，**不写 RTL**，固定四层逐 sample
重放：

- functional：destination contributor multiset、flattened weight key、Acc24 final state、commit count
  全部 0 mismatch；canonical close 不得提前；所有 partial 尾部必须由 frontier/fence 解释；
- resource：同 8 bank、同 L4/1R1W、同 96 lanes、同 4 phase psum ports、同 join bytes；
- traffic：weight reads 不得增加；分别报 psum read、write、commit 和 descriptor movement；
- GO-to-RTL：以 `speedup=A1-ISO8_cycles/PDR_cycles` 定义，四层 ratio-of-sums `>=1.30x`，且每个
  sample `>=1.10x`，分析 added
  logic/state `<=20% A1`，decoder exact share `>=15%`；
- support-only：cycle `<1.30x` 但 psum read+write 或 modeled dynamic energy reduction `>=30%`；只作
  decoder energy/completeness，不作下一条性能 RTL；
- KILL：cycle `<1.20x` 且 psum traffic reduction `<30%`，或任一层/sample 回退，或需额外 psum
  端口/weight bandwidth 才过门。

旧 M510 的 `EPD/A1 <1.30x` 停止新 scheduler 门保持有效；本审计没有降低它。

### 2.4 最小 RTL leaf

唯一允许的 leaf 名义边界为 `c2d_phase_banked_destination_join_leaf`：

- 输入：M523 风格 8-lane exact tap descriptors，并新增冻结的 canonical-frontier/fence；
- 四个 phase banks，每 bank 两个 bounded destination contexts；每 context 保存 exact destination key、
  up to-8 flattened signed source keys、valid/bank mask 和 close state；
- `bank = flattened_source_key mod 8`，同 bank contributor 延期，不增 weight port；
- 输出：既有 C2/M519 风格 `{destination,phase,source_mask,source_key[0:7],sign[0:7],last}` group
  command；partial close 合法、full8 优先但不等待未来未知事件；
- 不包含 DRAM、全网 scheduler、BN、完整 decoder、weight data array 或 psum data array。

任何“直接把 M523 八 transport lanes 接成 M218 八 weight banks”的实现都应静态拒绝。

### 2.5 必须的验证/物理链

1. source-only static hammer：frontier closure、slot alias、phase/destination key、same-edge retire/replace、
   fault drain；P0/P1=0 后才可运行；
2. Synopsys VCS + SVA：PyTorch/integer scatter miter、ordered multiset/Acc24、full8/partial、四 phase、
   same-bank conflict、frontier close、tag/time boundary、backpressure、stale/duplicate/malformed/fault drain；
3. paired 3 ns DC：PDR 与 A1-ISO8 同 hierarchy/normalization shell；setup/hold、五类 constraint、
   `TIM-209/OPT-150` clean；added area `<=20%`；
4. Formality：若 DC 使用 sequential inversion/retime 或源码有 signed-cast warning，必须以 SVF 闭合；
5. mapped-gate VCS SAIF + PTPX：同 contiguous window、annotation coverage、internal/switching/leakage；
6. SRAM/energy：四 phase psum 与八 weight bank 同 macro/CACTI 点收费；禁止 0-macro 当 memory free。

## 3. 候选 B｜FC2 central owner + tag-elided typed ticket

### 3.1 迁移差异成立，但不是周期创新

ELSA 的 BAER 是公共 header 摊销，FireFly-T 是 bank-aware dispatch；H67 的合法变化是：在
single-token FC2 内，语义 tag 由中央 `token_tag_q` 唯一拥有，八个独立 backpressure/OOO leaf
仅携带 `{implicit_bank,epoch16,generation32,slot3}` ticket，删除每 leaf 复制的 `tag24`，同时保留
wrong epoch/generation、slot reuse、duplicate、reorder 和 R5 channel-local fault 的 fail-closed 语义。

这属于协议/安全边界变化，不是改名。但 K8 对 equal-service K1x8 的 directed cycle 仅约
`1.01--1.04x`；tag-elision 合同要求 cycle/traffic 不变，所以它不能提供新的纸面 cycle 倍率。

### 3.2 同资源 baseline 与已知上界

- baseline：冻结 tagged R5-precedence reference；同 L4/O8/FIFO4、八 bank 1R1W、128-bit data、
  Acc24、scheduler/queue/fault precedence、macro boundary；
- candidate：只删除 request/response/leaf/central-slot 的 `tag24` state/echo/compare/mux；地址18、
  epoch16、generation32、slot3、expected/arrived mask 全保留；
- 冻结平均 occupancy `n=5.6268169` 下 selected metadata movement
  `2136.1323 -> 1547.9579 bit/transaction`，即 `-27.5346%`；这是有利静态上界。

完整规格与 99/100 静态评审见
[r2 spec](/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/reviews/c2_ticket_elision_prertl_spec_r2_20260827/c2_ticket_elision_prertl_spec_r2_20260827.md)
和
[r2 hammer](/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/reviews/c2_ticket_elision_prertl_spec_static_hammer_r2_20260827/c2_ticket_elision_prertl_spec_static_hammer_r2_20260827.md)。

### 3.3 一天内门、最小 leaf 与验证链

硬前置：M519 R5 K1/K8/K1x8 三轴 DC 必须先有独立双封 canonical。此前不得 author B RTL。

- 最小 leaf：matched tagged/elided `typed_ticket_weight_leaf_shell` + central adapter clone；data SRAM
  black box 完全相同；candidate 不得保留 assertion-only/dead tag state；
- VCS/SVA：ghost-tag scoreboard、A01--A18 非零 cover、wrong-tag tagged-only negative、cycle/bank
  traffic/Acc24/result/done 0 mismatch；
- DC：paired local transport 与 full-K8，same top/constraints；
- PTPX：无论 area 是否过门，clean DC 后都必须做一次 matched mapped-gate SAIF/PTPX；
- PROMOTE：full-K8 sequential area `>=10%` reduction，且 total area `>=15%` **或** dynamic power
  `>=20%`；
- KEEP：total area `>=8%` 或 dynamic power `>=10%`；
- KILL：area `<8%` 且 dynamic power `<10%`；任何 cycle/traffic/timing/functional mismatch 直接 P0。

B 只写进 C2 实现消融，不新增 C4；K1x8 未对称实现相同优化前，不更新 throughput/mm2 主表。

## 4. P0/P1/P2 与唯一执行建议

### P0 = 0

没有把公开机制改名成新意，没有把分析/静态上界升级为 RTL/能量/系统数字，没有启动工具或
远端任务，也没有修改任何冻结输入。

### P1 = 3

1. M511 exact S10 decoder payload/verifier 与 M513 production fast-kill 尚无完成结果；A 当前不得
   author RTL。
2. A 还缺 flattened weight identity、同 bank deferral、stored-weight identity 和 destination-close
   VCS reference；M523 descriptor PASS 不能替代它们。
3. M519 R5 三轴 DC canonical/独立 receipt 尚未完成；B 当前不得 author RTL。

### P2 = 2

1. A 的 exact group richness 可能被 output parity、bank conflict 和 sparse source order显著削弱；
   当前 22% decoder share 不能替代 PDR/A1-ISO8 实测。
2. B 的 27.53% movement 是局部有利上界；计入不变 scoreboard/mask/clock 后，full-K8 area/power
   改善很可能低于 promote 门。

### 唯一建议

**下一候选只保留 A/PDR。** 先让现有 M511/M513 链产出 exact S10，随后只跑一次同资源 CPU
快杀；过 `>=1.30x` 门才创建 RTL author 合同。B 不取消，但排到 M519 三轴 DC 完成之后，按
既有 r2 规格只做一次 clone-only physical A/B。两者都不得与当前 M528/M519 EDA 并发。

## 5. 身份与不可引用边界

- 输入审计依据 SHA：M510 `9406211a...`；M523 receipt hammer `4919b0af...`；C2 r2 spec
  `a647e80b...`；C2 r2 hammer `580d4573...`；M532 method audit `b1ac5e02...`。
- `docs/359_DATE终局冻结_20260813.md` 本轮未修改，SHA256 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- 本审计没有生成 cycle、area、power、energy、accuracy、system speedup 或 paper-ready PPA。
- 当前唯一 admissible 结果是候选排序、合同边界和未来 kill gate；不是实现完成或 DATE headline。
