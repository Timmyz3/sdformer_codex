# M183 独立打铁评审

结论：**85/100；standalone K8 算术岛通过，但不能取代 K4 主点。**

M183 的 RTL、sealed VCS/SVA、fresh-seed VCS、独立数值边界 miter 和同约束 28 nm 3 ns DC 均通过。8 路 signed INT8 的范围是 `[-1024,1016]`，与 signed24 accumulator 相加后的完整范围 `[-8389632,8389623]` 被 signed25 完整覆盖；`extended_sum[24] != extended_sum[23]` 是正确的 signed24 越界判据。

独立 fresh seed `183997` 重跑得到 481/481 issue/result、1--8 源 histogram `60/60/60/60/60/60/60/61`、159 个连续 II=1 hit、478 次同拍 replace、110 个 stall，12/12 coverpoint 非零且 0 assertion failure。评审专用边界 miter 又直接覆盖 `-128/127`、signed24 正负精确边界、正负 overflow，以及分离的 non-prefix/duplicate-bank 攻击，全部 0 mismatch。

## 性能裁决

| 指标 | K1 / M170 | K4 / M169 | K8 / M183 |
|---|---:|---:|---:|
| 解析 schedule cycles | 424,060,394 | 127,581,198 | 97,607,807 |
| 3 ns logic-only area | 11,940.011991 um2 | 18,522.881882 um2 | 27,031.031773 um2 |
| Ports | 5,548 | 8,152 | 11,625 |
| Sequential cells | 2,341 | 2,343 | 2,344 |
| Logic levels | 9 | 38 | 42 |
| Setup / hold slack | +1.6146 / +0.0221 ns | +0.8670 / +0.0224 ns | +0.7691 / +0.0224 ns |

K8 相对 K4 的 schedule ratio 是 `1.307079853x`，但面积是 `1.459331866x`，所以固定 3 ns 下的解析 schedule throughput / logic area 只有 K4 的 **`0.895670055x`**。换言之，K8 降低 23.493580% cycles，却增加 45.933187% logic area，吞吐密度反而降低 10.432994%。K8 要降到 **24,210.885723 um2** 以下才追平 K4 的这项密度。

相对 K1，K8 的 `4.344533568x` schedule boundary 除以 `2.263903235x` logic area，仍有 `1.919045611x` 的条件性面积归一化机会。因此合理定位是：

- K4 是吞吐密度点；
- K8 是相同 3 ns、面积预算充足时的绝对延迟点；
- `4.3445x` 与 `1.3071x` 都只是 exact-payload analytic schedule ratio，不是 physical、complete-FC2、system 或 headline speedup。

DC 的五类 constraint 全 clean，setup/hold 均 MET，0 macro、mapped multiplier 为 0；但 ideal clock、ZeroWireload、11,625 个 port bits、6,144-bit/cycle weight payload 和 high-fanout 近似意味着它仍不是 routed/macro PPA。

## 最重要的架构修正

M182 的自然结构是“每个 bank 固定一个 selector”，输出应是任意 `bank_valid[7:0]`。M183 却把事件压成 prefix slots，再携带 bank ID 并做 28 组两两唯一性比较。这会额外引入 bank-to-prefix packing/routing，而且这部分还没有进入 DC。

下一步应做 fixed-bank K8：直接接任意 8-bit bank-valid，固定 bank-to-weight lane，`bank_mask=valid`，删除 bank ID、prefix 约束与 28 个比较器。若它仍无法低于 24,210.885723 um2，就不应把 K8 宣布为 K4 的严格替代；应补 K5/K6/K7 的预提交 schedule+DC Pareto sweep。

## 阻塞项

P0 有 3 项：可执行 bounded-K8 frontend/八 bank weight response、accumulator context 到 BN2/residual 的完整 FC2 commit、PAFT 后 sn2 threshold-one 或 folded-weight 数值桥。P1 有 5 项：K8/K4 面积效率、fixed-bank 接口、cycle lineage 命名、macro/route/power/fmax、Formality。P2 有 4 项：M160 census SHA 补链、canonical TB 极值覆盖、VER-318 cast、关系型 SVA。

冻结 ep35 的 M160 census 确认 12 个 `sn2` threshold 都精确等于 1；PAFT 尚无发布 checkpoint/valid825 admission。一个更广泛 threshold census 中出现非 1，只能作为警告，最终必须对 12 个 `sn2` 做 checkpoint-bound 复核；任何非 1 的 `sn2` 都会阻断 M183 当前 multiplier-free identity。

机器可读裁决见 `m183_independent_hammer_review.json`，独立复算与 VCS 记录见 `fresh_recompute.json`。`docs/359` 未修改，SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
