# M1762：M1754/M1756 TSBG 执行失败独立诊断

## 结论

本次失败是 **M1721/M1747 分析器的跨层 S2 witness 聚合假阴性**，不是 M1707 capture 身份、顺序、维度、CRC 或数值异常。M1754 固定解释器 preflight 已通过，唯一 attempt 已消费；M1747 result 和 work 均不存在，未发布部分结果。M1747/M1754/M1756 不得直接重跑。

## 第一性原理定位

`DecisionAccumulator.consume_pair` 对每个 FC1 pair 得到长度为 `ceil(input_channels/16)` 的 `drop_seen`/`keep_seen`。但是代码以 `(epsilon, "all", "FC1")` 和 `(epsilon, "sequence", sequence)` 为 key 保存数组，随后强制所有层 shape 相等。

canonical FC1 层的 G16 依次为：`6, 6, 12, 12, 24, 24, 24, 24, 24, 24, 48, 48`。sample 0 的 layer 8 首次建立 shape 6，layer 10 仍为 6，layer 12 变成 12，因此 line 718 必然报 `S2 witness group shape drift`。这与远端 traceback 完全一致。

简单把数组 pad 到 48 不是合法修复。不同层的 source-group 0 没有共同身份，而且 `output_blocks` 也随层为 `24/48/96/192`；当前 all/sequence state 还会错误复用首层 multiplier。

## Capture 排除证据

- 远端对 canonical capture 的 `SHA256SUMS` 逐成员检查全部 PASS；`fc_frames.bin` SHA 为 `dceb6c0c...18b1`。
- M1744 已独立检查 11,040 个 frame 的 header/order/extent、zlib EOF、raw length、CRC、support/sign/nonunit/nnz/code semantics，结论 PASS。
- `layers.json` 明确声明上述异构 FC1 宽度；分析器先验证 payload 的逐层 channel/token 维度，再在报告聚合阶段触发 shape assertion。
- capture 文件时间早于 M1754 attempt，结果/work 目录在失败后均不存在。

## 最小 successor 规范

1. 新建 additive source；不得修改或重跑 M1747/M1754/M1756，也不得重做 capture。
2. TSBG 算法、ordinary-LRU baseline、B2/B4/B8、S2 keep/drop、epsilon、checkpoint、capture 和 claim boundary 全部不变。
3. S2 witness 的基本身份改为 `(epsilon, layer_id, source_group_id)`；每层按自己的 `output_blocks` 计算 witness count。
4. all/sequence 报告行只对层内 witness count 做整数求和，禁止跨层 OR 同下标、禁止沿用首层 `output_blocks`。
5. 新增异构层回归：同一 sequence 内 G16=6/outblocks=24 后接 G16=12/outblocks=48；直接 reference 必须逐层及汇总 0 mismatch。
6. 保持 fail-closed：新 source 独立 hammer、fresh release、fresh namespace 后才允许唯一一次分析。

## 论文边界

M1762 只证明失效原因和 capture 可复用。它不产生 TSBG/S2 周期、流量、能量、AEE、RTL 或论文结果。
