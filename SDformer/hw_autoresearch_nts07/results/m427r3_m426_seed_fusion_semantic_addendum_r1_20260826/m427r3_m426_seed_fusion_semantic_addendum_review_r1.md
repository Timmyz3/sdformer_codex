# M427r3：M426 seed fusion 跨 partition 累加语义补充打铁评审

结论：**P0 确认。M426 seed fusion 按现有规格不可执行，437,640,532 cycles / 1.695794× 必须撤销、不得引用。** M427 r2 的原始行/周期复算仍成立，但只能说明算术账本可复现，不能证明其数据通路实现了正确递推。本补充评审不修改已双封的 M427 r2，只覆盖其中 seed-fusion 的语义结论。

评分由 86 降为 **62/100**；严重度由 P0/P1/P2 = 0/3/3 改为 **1/3/3**。

## P0 的直接证据

冻结语义不是每个 partition 从零开始：

- M108 的循环顺序是 `sample, operator, W64 raster window, partition`，到 partition 431 后才 flush/commit。
- M118 只在 `window_start_accept` 清 `row_valid`；每次更新读取旧值并执行 `new_psum = old_psum + update_delta`。
- M120 把 mapper 的 `update_delta` 直接送入 M118；没有 partition-local accumulator、merge 或三操作数路径。
- M401 只在每个 sample 末收费 96,000 commit cycles；M405 是明确标注的 non-contribution shell，M412 又明确把 compute backend 排除在外。

因此 M426 所写的“mux selects reconstructed PWP instead of prior accumulator”实际是：

```text
claimed: new = PWP + correction
required: new = old_psum + PWP + correction
```

它漏掉了 `old_psum`。要在一拍内做对，必须增加已收费模型中不存在的 96-lane pre-adder/compressor/三操作数通路；或者建立局部累加器并显式收费一次向全局 PSUM 的 merge。

## 51.84M 行、432 partitions 全扫

独立脚本没有 import/执行 M401 或 M426 analyzer，而是直接解码冻结 M410R2 运输：

| 指标 | 数值 |
|---|---:|
| runtime rows | 51,840,000 |
| positive-residual PWP rows | 11,620,766 |
| 已有更早 active partition 的 positive PWP | 11,575,447（99.6100%） |
| base 仍为空的 positive PWP | 45,319（0.3900%） |
| 覆盖 sample-operator | 40/40 |
| 出现 prior-base 的 partition | 1–431，431/431 |

第一个直接 witness 是 sample 0 / operator 0 / partition 1 / source row 6，`original=131, distance=1`；该 row 在 partition 0 已 active，所以第一拍必须保留旧 PSUM。

## 周期后果

M426 声称从 dual 的 530,606,660 cycles 再省 92,966,128 cycles，得到 437,640,532。几乎全部这部分都需要第三操作数：

- 用现有 two-input adder，只对 base 为空的 45,319 行融合，修正后是 **530,244,108 cycles / 1.399635× vs strong zero**。
- 相对 dual 仅剩 362,552 cycles，即 **0.0683%** 周期下降。
- 若统一采用“局部算 PWP+corr，再串行 merge global old_psum”，要把全部 92,966,128 cycles 加回去，正好回到 dual 的 **530,606,660 cycles / 1.398679×**。

所以 1.695794× 不是“有一点 caveat”，而是 **non-executable opportunity**。

## dual co-read 的独立判断

dual co-read 的语义仍成立：low8/high4 并读重构一个 signed12 `update_delta`，M118 仍执行 `old_psum + reconstructed_PWP`，不需要第三操作数。4096 个 signed12 编码的穷举重构是 0 mismatch。

但它只允许按 **standalone throughput-area Pareto** 推 RTL：每拍 144 logical PWP bytes（现有 padded input signal 160 bytes），没有与 one-port strong-zero 做资源归一；M412 也未包含 SRAM macro、memory wires、clock tree 和 compute backend。因此 530,606,660 / 1.398679× 不是 DATE headline，更不是系统倍速。

## 决策与下一关

1. 立即停止按 M426 规格实现 seed-fusion RTL，并从候选/headline 表移除 1.695794×。
2. 保留 dual co-read，下一步只做 standalone RTL + VCS value/protocol miter；同时实现 common-resource K2/reference。
3. dual 和 reference 都要过 DC/STA、Formality、真实 SRAM 端口、SAIF/PTPX，之后才画 throughput-area/power Pareto。
4. 若仍要尝试 generic fusion，必须新开里程碑和 contract，显式收费 96-lane pre-adder/compressor 或 local-accumulator merge；不能沿用 M426 的 mux-only 身份。

边界保持：四个冻结 H67 bottleneck Conv3x3、非 RTL measured、非 macro/PPA、非 power/energy、非 full-network/system、非 DATE headline。`docs/359` 未修改。
