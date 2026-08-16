# Match-Code 跨时间注意力硬件边界

## 软件接口冻结

H73 DE9 与 H74 MC49 都保持 ATLIF105、all12 同构、T=2、head_dim32。attention 不读取动态
`gate*K` carrier，而是产生固定顺序的跨时间位移描述子，再乘静态 per-head codebook。二者不会
与 H60 共存于不同 stage；最终若胜出，12个 block 使用同一计算模板。

## DE9

- K halo：另一时刻 3x3，共9个固定 offset。
- evidence：event-event 与 silence-silence 两组32-lane匹配/popcount。
- normalization：两组 Shiftmax9，输出18项 descriptor。
- projection：每 head 静态 `18x32` 权重；全网新增参数79,488。

## MC49

- K halo：固定49个跨时 offset，边界无效项必须硬 mask。
- score：每 offset 一个32-lane alpha-XNOR/popcount score。
- normalization：Shiftmax49，输出49项 descriptor。
- projection：每 head 静态 `49x32` 权重；全网新增参数216,384。

## PPA 约束

操作审计把 DE9 的18路 lane compare/popcount与`18xD`静态投影、MC49 的49路 compare/popcount与
`49xD`静态投影单列。现有 spike-energy proxy 不含这些操作。RTL/PPA 还必须加入 halo SRAM、
offset address、边界 mask、Shiftmax reduction、codebook SRAM 和输出累加。只有完整成本下仍优于
H60，Match-Code 才能作为硬件主线。

可直接复用的 exact 优化包括64-bit temporal-pair packing、固定 offset K reuse、weight-stationary
codebook 和数学零值 clock gating。offset prune/early-exit/codebook 近似会改变输出，必须重新走
软件 full30，不作为免费硬件优化。

## AX17

- offset：另一时刻横轴9点与纵轴9点，中心共享，共17点。
- score：17个32-lane alpha-XNOR/popcount，随后 Shiftmax17。
- projection：每 head 静态`17x32` codebook，全网新增参数75,072。
- layout：固定轴向地址可分别流式读取 row/column；与 MC49 共用 offset engine，但 halo/投影规模更小。

Flow1D 原硬件负载包含动态一维 attention、softmax、value aggregation 和另一轴 correlation；AX17
只采用其正交搜索动机，不能把 Flow1D 的精度或复杂度数字直接归因给本实现。DATE 2023
RAWAtten 的 stage-dependent window reuse 与可重构 w-core 可用于安排 SDformer 不同 stage 的
window/head 并行度；其近似 LR-Softmax 不替换本项目 Shiftmax，除非另做数值验证。
