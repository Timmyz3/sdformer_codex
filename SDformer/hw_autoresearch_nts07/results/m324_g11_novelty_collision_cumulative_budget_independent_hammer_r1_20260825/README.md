# M324：G11 novelty-collision 独立打铁

结论：**当前 G11 的 `|w*x|` 阈值、固定 top-m 和 beta-count ledger 与 M287/M293/M300 完全撞车，不能再算一项新贡献。** 唯一值得做一次 CPU fast-kill 的变化，是按 token active subset 消耗固定累计误差预算 `B`；它有算法差异，但执行成本风险很高，现阶段不能进入论文贡献、GPU、RTL 或 Synopsys。

## 为什么现有 G11 完全同构

对二值 source，令 `a_ti=x_ti∈{0,1}`，并令：

`beta_ig = max_{j in destination group g} |Wq_ji|`

则：

`max_j |Wq_ji*x_ti| = a_ti*beta_ig`

零 source 已由普通 bit-sparse frontend 跳过；对所有被 issue 的 source 都有 `a_ti=1`，所以 `|w*x|<=tau` 与 `beta_ig<=tau` 完全等价。最终 issue mask 就是 token activity bitmap 与 M287/M293 静态 beta mask 的 AND。

- M287 已对 10 个二值 FC1 做该 mask 和 `beta*omitted_count` ledger。
- M293 已对六个 patch/residual Conv 做同一机制。
- M300 已将二者合并成共享 mask/compactor 机会研究。
- M301 已实际将 group4/beta48 选中的 weight pair 清零并做 modified forward。

固定 top-m 只是换一种方式生成静态 mask；给 ledger 换名字也不构成创新。

## M301 已经给出的负面事实

M300 的 group4/beta48 只是 `1.1841687x` 理想 full-envelope sensitivity。修复后的 M301 paired S10 中：

- baseline AEE：`0.9602408795`
- beta48 AEE：`1.0707721690`
- 增量：`+0.1105312895`
- 允许增量：`0.02`

失败幅度为预算的 `5.53x`，因此该静态 mask 已 NO-GO。beta32 的理想 sensitivity 又只有 `1.0503984x`，没有硬件晋级价值。

## 真正不同的候选：token-dynamic cumulative B

为每个 source/destination group 预计算 `beta_ig`，并按 `(beta, source_id)` 固定升序。每个 token 只对当前 active source 消耗预算：若累计 `sum beta_ig` 加上下一项仍不超过 `B`，则 drop；第一次超预算后保留该项及后续项。

这样可证明每个 covered destination accumulator 的绝对整数误差不超过 `B`。`B=0` 必须 bit-exact；零权重任务可以作为 exact elision。

它与静态 mask 真正不同的必要证据是：同一个 `(source, destination group)` 在两个 token 上，因为更早 active source 消耗的预算不同，一个被 drop、另一个被 keep。若找不到这种 witness，候选实际仍退化为静态 mask，不能算新机制。

## 现有数据能测什么

- FC1：本机有 100/100 个 M51 bitpack，可做完整 S10 CPU replay。
- 六个 patch/residual Conv：本机有 60/60 bitpack，可重建 3x3 im2col source key。
- checkpoint：可复现 M287/M293 的逐行对称 INT8 quantization。
- FC2：本机 0/120 payload，不能做新的 token-level budget replay；M216 聚合回执不能替代原始 payload。

CPU 可严格计算 dynamic drop set、`B` ledger、每个 destination 的真实整数误差、novelty witness、group-task reduction，以及保留 source 的理想 K8 issue。后者仍只是 proxy，不是硬件 cycles。

## 最大风险是先筛选反而更贵

候选有三种实现，均不免费：

1. 离线固定 permutation：无需 runtime sorter，但每个 token/group 要按 group-specific 顺序扫描 source 并查 activity。
2. 先读 active list 再查 beta：少扫零 source，却需要对最多 207 个 FC1 active source、448 个 Conv active source排序或分桶。
3. 静态 beta bucket：可省 sorter，但要存 source-ID 列表并做 membership check；桶过粗又会退化回静态 beta mask。

若先读原权重计算 beta，weight traffic 已发生，跳过收益大幅消失。beta/order 必须走独立 metadata SRAM，并把其访问计入同资源周期。

以 M301 group4 的 1,090,944 个 source-group pair 为例：

- 原一位静态 mask：136,368 bytes；
- 朴素 `uint8 beta + uint16 source ID` 顺序表：3,272,832 bytes；
- 是逻辑 metadata 的 `24x`，尚未计 pointer、bank map 和 runtime state。

这不是 SRAM PPA 结果，但足以说明不能只报 MAC drop。

## 最快测量与晋级门

先不用 GPU：

1. 用一个高工作量 FC1 record 和一个高 fan-in Conv record，预提交 group `{4,16}`、`B={0,16,32,64,128,256,512,1024}`。
2. 同时统计 drop、bound、dynamic witness、ideal K8 issue、全域 scan 数、metadata 读和明确 scan width 下的周期敏感性。
3. 任一条件触发即杀：`B=0` 非 exact；无 dynamic witness；scan/metadata 后不快于相同 K8 baseline；唯一获胜实现退化成 static beta。
4. fast-kill 通过才跑本地 FC1+六 Conv 全 S10；再只挑一个 B/group 做 paired modified-forward S10。
5. S10 正确性与 executable same-resource cycles 同时通过后，才允许 valid825、RTL 和 Synopsys。

## 与 M306/M307 的关系

M306/M307 的 Conv near-match 是 PAFT 16-source pattern/codeword/PWP/correction-elision，和 destination-group weight mask 不同。其 tau1011 S10 增量 AEE `0.014362` 通过局部门槛；相对 bit-sparse 是 `1.6446x/1.2924x`，但相对 tau0 exact 的新增收益只有 `1.0675x/1.0482x`，且仍没有 executable SRAM/control RTL、valid825 或系统倍速。

PAFT ep4/running-BN 数字不能与 H67 ep35/no_running 的 G11 population 合并。

最终决定：static G11 **NO-GO 新贡献**；token-dynamic cumulative B **GO CPU fast-kill only**。本审计未改源/RTL、未跑 GPU，`docs/359` SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
