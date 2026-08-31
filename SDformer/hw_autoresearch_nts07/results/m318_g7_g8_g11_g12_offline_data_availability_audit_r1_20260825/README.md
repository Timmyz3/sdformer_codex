# M318：G7/G8/G11/G12 离线数据可得性审计

结论先行：当前最值得继续的是 **G11 的 FC1 + 六个 patch/residual Conv 离线筛选**，以及恢复缺失的 M51 FC2/downsample/pred bitpack。G7 在官方二值事件源上没有新增机会；G8 和 G12 均缺少代表性原始 tensor，不能从现有聚合数据宣称性能或正确性。

## 严格结果

- M51 合同覆盖 31 个模块、310 次 hook、10,506,240,000 个 source，其中 712,894,209 个非零值，数值严格为 `0/1`。
- 对 `theta={0, 2^-6, 2^-5, 2^-4, 2^-3}`，正距离集合 `0 < |x| <= theta` 始终为空。因此新增 source-drop 率为 0，且被删集合的 `sum |w_i x_i|` 上界严格为 0。这个证明是严谨但无机会的结果。
- 官方 ATLIF 输出是 `0` 或约 `1` 的逐站点阈值；最小阈值为 `0.9999649524688721`。同一 theta 网格不会删除非零 ATLIF 输出。该结论不能外推到 ATLIF 的模拟输入或 pre-threshold `h`。
- ordered S10 trace 有 79 个 operator 聚合行、1840 个 execution record、93 个 runtime ATLIF 行。它能给逐层 denominator，却没有 token/bank 或模拟幅值分布，不能生成新的 source issue/cycle 结果。

## 本机 payload 现状

| 类别 | manifest 记录 | 本机 payload | 结论 |
|---|---:|---:|---|
| FC1 | 100 | 100 | 可做 CPU-only K8 issue proxy 与更紧的 source-local bound |
| FC2 | 120 | 0 | 只能引用既有 M216 聚合回执；新逐层 replay 必须先传回 payload |
| downsample linear | 20 | 0 | 先传回 payload |
| 六个 patch/residual Conv | 60 | 60 | 可重建 3x3 im2col source key，做离线 proxy/bound |
| pred Conv | 10 | 0 | 先传回 payload |

本机实际存在 160/310 个 payload，共 748,800,000 bytes；缺 150 个，共 564,480,000 bytes。优先从已有交接源按 manifest SHA 传回，不建议为此重新占用 GPU。

## G7 与 G11 不是两个贡献

G7 的 mask 对每个 destination group 都相同：`keep(i,g)=1{|x_i|>theta}`。G11 则按 destination group 使用贡献界：`keep(i,g)=1{bound(i,g)>budget}`。因此 G7 是 G11 的 destination-independent 特例，两者共享 mask/compactor、K8 executor、selected/skipped ledger 和误差预算。

论文中宜写成一个贡献：**bounded-contribution gate + destination-group sparse executor**。G7 只作为退化模式或 ablation；当前二值源上的 G7 机会为零，不能单列创新点。

## G8 为什么现在不能测

M233 保存的是逐通道/逐样本动态 BN 参数、统计量和范围，不是真实 token tensor。ordered activation record 也只有块级聚合。要证明 FFN residual bypass，至少需要同一 token 的：

1. FFN input/residual；
2. FC1 pre-BN1、BN1 output、sn1；
3. FC2 pre-BN2、BN2 output、完整 branch output；
4. residual-add output 和当前 batch 的 BN statistics。

在 frozen `no_running/current-batch` 语义下，空输入本身不授权跳过整条 branch。必须先做 paired S10 modified forward，再决定是否跑 valid825。

## G12 为什么只能先做机制原型

官方 ATLIF 是 PSN 式 temporal affine + threshold，并不存在传统 recurrent LIF membrane。G12 真正需要跟踪的是逐 output-time 的 temporal partial accumulator，以及剩余正/负贡献的保守界。

现有 DP-TME 向量覆盖 81 个 live site、每站点 320 个定向选点，共 25,920 个事件，但只来自 sample 0，且按 ordinary/near-threshold/max-amplitude 策略偏置采样。它适合验证 early-stop 证明机制，不适合给 S10 opportunity rate。

实现时还必须避免“先算完未来乘积再证明它们可跳过”的伪优化；issue order、bound metadata 及其硬件代价必须显式建模。

## 推荐推进顺序

1. 立即用现有 FC1/六 Conv bitpack 做 CPU-only 理想 K8 issue proxy 与 checkpoint-bound `sum |w_i x_i|` 筛选，明确标成 proxy。
2. 从交接源传回 120 个 FC2、20 个 downsample、10 个 pred payload，逐文件核对 manifest SHA。
3. 用 DP-TME 定向向量原型化 G12 exact remaining-bound 机制，不给代表性 rate。
4. GPU 空闲时只加最小 frozen S10 hook：ATLIF temporal partial/bound + FFN residual boundaries。
5. 先 paired S10，只有正确且机会足够大的候选再跑 valid825；之后才接 executable cycle adapter/RTL/Synopsys。

## 当前禁止宣称

不能从 M318 宣称 accuracy、modified-forward equivalence、完整层/全网 cycle speedup、RTL/VCS/DC/PTPX/PPA/energy、代表性 G12 rate、安全 G8 bypass 或 DATE headline。理想 bank issue、operator weighted-MAC 和 system cycle 必须保持三个不同 denominator。

审计未改网络、RTL 或 `docs/359`，未运行 GPU。`docs/359` SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
