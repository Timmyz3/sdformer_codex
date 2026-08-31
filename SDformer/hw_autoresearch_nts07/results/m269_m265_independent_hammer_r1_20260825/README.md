# M269：M265 ATLIF matched-boundary 独立打铁评审

结论：`88/100`，`P0=0 / P1=2 / P2=5`。M265 可以作为冻结 H67 T10 ATLIF population 上的 **独立模块分析周期模型**继续推进；`3.399935×` 不能写成 RTL、面积归一化吞吐、DC/PPA、能耗、系统或 headline 指标。

## 独立复算

- 从 ordered trace 重建 `1840 execution records / 450 T10 records / 10 samples × 45 contexts`。十个 sample 的 context 名称、次序和 shape 均一致；每 inference 为 `7,318,350 tiles`、`36,591,750 raw beats`、`36,591,750 result beats`，最大 tag 为 `738,658,303`。
- 对 13 个 config/ingress/result pressure 场景全部从独立状态机重算，作者 JSON/CSV 均为 0 mismatch。
- ideal matched-boundary：Fixed `124,412,490 cycles`，rank3 `36,592,605 cycles`，模块周期比 `3.3999353148×`。
- 全部 13 个压力点的模块周期比范围为约 `2.549956×–3.399936×`；所有点仍是 isolated module cycles。

## Fixed 17-cycle 公平性

Fixed 每 tile 有 1,600 个 INT8 product，96 个 multiplier slot，`ceil(1600/96)=17`。独立给出可执行 allocation witness：cycle 13 前五个 result group 已完成 `[319,300,250,200,51]` 个 product；cycle 13–17 分别分配

`[1,19,69,7,0] / [0,1,0,95,0] / [0,0,1,17,78] / [0,0,0,1,95] / [0,0,0,0,96]`。

每拍不超过 96 个 slot，并恰好在 cycle 13–17 每拍关闭一个 320-product result beat；总空闲 slot 为 32。因此 tile-closed 17-cycle baseline 在算术层面合法，不是用 16.667 fractional cycle 冒充 tile 调度。

M25 exact96 T10 crosscheck 为 `121,972,500 cycles`；tile-closed Fixed core 为 `124,411,950 cycles`，差值严格为 `51/50=1.02×`。对 rank3 steady core `36,591,750 cycles` 的独立 crosscheck 严格为 `10/3=3.333333×`。

## Candidate 与边界位宽

Candidate 明确收费五拍 stage1 和五拍 M37-class stage2，并允许两者在独立资源上重叠。M37 的单 entry product register 支持“旧 product 入 FIFO、新 product 同拍替换”。257-tile ideal 攻击中：开启替换为 `1304 cycles`，发生 `1284` 次同拍替换；关闭替换退化为 `2588 cycles`，说明该语义没有被漏计或凭空假设。

配置位推导：

- Fixed：`100×8 + 10×24 + 24 = 1064 bits`，同一 256-bit bus 上为 5 beats/context。
- M37 stage2：`30×8 left factor + 30×4 valid + 30×4 sign + 30×4×3 shift + 10×24 bias + 24 threshold = 1104 bits`。
- 完整 candidate 再加 `30×8 right factor + 5-bit stage1 requant shift`，合计 `1349 bits`，为 6 beats/context。

## Fast-forward、重放与攻击

- 对 13 场景、Fixed/rank3、8 种 tile count、多个周期起始相位执行了 608 组“逐拍 direct vs periodic fast-forward”比较，所有控制状态、计数器和最终周期 0 mismatch。
- Clean replay 的 JSON/CSV/correction overlay/README 与作者封存逐字节同 SHA。
- Relocation 后除预期的绝对 contract path 外，数值 payload 完全一致。
- 污染 relocated execution trace 后，analyzer 返回非零且不创建输出，输入 wrong-SHA gate 有效。
- 仅给 contract 增加一个换行时 analyzer 返回 0 并生成结果：作者目录的 seal 没有损坏，但缺少一个从外部固定 contract SHA 的 exact runner。这是 P1 证据链问题。

## 打铁问题

P1：

1. 直接 analyzer 不固定 contract 自身 SHA；contract byte drift 可成功执行。需要 exact-SHA runner 把 contract 作为 trust root 固定。
2. Candidate 使用独立 96-multiplier stage1 加 M37 shift/add stage2，而 Fixed 只在 96 个 INT8 multiplier 上匹配；尚未面积匹配，因此 `3.399935×` 只是模块周期设计点。

P2：

1. 17-cycle Fixed witness 仍是分析调度，没有实现 accumulator banking、compare/writeback 和 Fixed RTL 时序。
2. stage1-to-M37 组合尚无 integrated RTL；requant 和 intermediate bank timing 仍是模型合同。
3. 压力测试只有四类周期 mask 和固定全局相位，没有任意有限 backpressure 或低于 0.75 的服务率。
4. 每个 ATLIF context 都重新加载配置，尚未给 layer-static 参数驻留/复用敏感性。
5. 十个样本共享相同 shape-derived context map，模型不包含数据相关 ATLIF activity 变化。

决策：`GO_FOR_ISOLATED_ANALYTICAL_MODULE_CYCLE_EVIDENCE_ONLY`。未运行 DC，未修改 `docs/359`。

