# M380：M377 active-descriptor controller 只读 pre-RTL 打铁

日期：2026-08-25  
结论：`CONDITIONAL_GO_VCS_MINIMAL_ACTIVE_DESCRIPTOR_CONTROLLER_ONLY__NO_GO_IF_BLOCKING_OR_SHARED_PORT`

M380 没有写 RTL，也没有修改 M377、M373 或 `docs/359`。它从仓库根验证 M377/M373 的内容清单和 seal 两层 SHA，再对冻结 S10 源数据重新解包，独立检查 descriptor 信息是否足够、pop1 是否丢失、SERIAL16 是否可能一拍退休多个 descriptor，并攻击 M377 的 1.081x 周期余量。

## 结论先行

- 无 P0，但有 4 个必须由 VCS 关闭的 P1。M377 原来的无条件 `GO_VCS` 应收紧为“只允许按 M380 硬边界实现最小 controller”。
- `original16 + center_id` 足够重构 exact residual：`plus=original&~center`，`minus=center&~original`；实际冻结源人口逐项重放为 0 mismatch。
- 6,762,595 个 pop1 全部保留并走 exact bit-sparse fallback。只有 `original16==0` 的 30,368,111 行可以省略。
- SERIAL16 抽象 recurrence 在 51,840,000 行上证明最大退休率是 1 row/cycle，因此 compact write 上界也是 1 descriptor/cycle。
- 真正的性能危险不是固定 SRAM latency，而是“每 descriptor 串行付费”。L2 streaming SRAM 加每 phase 一次 active-count seal 后仍为 1.080935x；若每次 replay 每 descriptor 额外串行 1 cycle，则只剩 0.995920x（不加 seal 的 fast-kill 为 0.996014x）。
- 保住 1.05x 时，L2+count-seal 之后最多还能承受平均 0.345136 个非重叠 stall cycle / replayed descriptor。这是 VCS cycle replay 的硬门，不能用平均带宽或理想 SRAM 抹掉。

## 冻结人口与守恒

| 项 | 数值 |
|---|---:|
| source rows | 51,840,000 |
| exact zero rows | 30,368,111 |
| active descriptors | 21,471,889 |
| two-replay descriptor accepts | 42,943,778 |
| PWP descriptors | 12,709,384 |
| fallback descriptors | 8,762,505 |
| pop1 fallback descriptors | 6,762,595 |
| max active descriptors/phase | 2,400 / 3,000 |
| max center ID | 31 |
| max serialized retirements/cycle | 1 |
| min phase-average exact service/descriptor/replay | 4.858369 cycles |

M377 的 17,280 个 phase 都有且只有两次 replay；write event 与两次 replay 的 `active_count` 全相等。两次 replay 的 exact compute 总和为 422,285,576 cycles。

## 最小 controller 硬边界

48-bit descriptor 的 r1 唯一布局按 LSB0 冻结：

| 位 | 字段 |
|---|---|
| `[11:0]` | phase-local `row_id12`, 合法 0..2999 |
| `[27:12]` | `original16`, 必须非零 |
| `[34:28]` | `center_id7`, q32 下必须 0..31 |
| `[39:35]` | `distance5`, 必须等于 `popcount(original XOR center)` |
| `[40]` | `use_pwp1` |
| `[47:41]` | reserved flags，r1 必须全零 |

sample/operator/partition/tile 不放进 descriptor，而由 phase registers 提供并在 replay 中保持不变。写端只能在 accepted row retirement 且 `original!=0` 时写一次；`active_count` 在 row2999 退休后一拍 seal 一次。

descriptor SRAM 边界为两 bank、每 bank 3000x48、所选 bank phase-exclusive 1RW。read wrapper 必须是 in-order valid/ready、完整 48-bit response、steady-state req/rsp II1，固定 latency 支持 L1..L8；它不能与 DMA0、tile/pattern SRAM 或 SHARED96 compute 共用未计费仲裁端口。D8 FIFO 使用 `occupancy + accepted_unreturned <= 8` 的 reservation credit。

每个 phase 只能 replay 两次，顺序 tile0、tile1；两次都必须精确接受 sealed `active_count` 个 descriptor。第三次 replay、计数不符、row/center/distance/flags 错误、孤儿 response、FIFO 溢出/下溢、context 改变或提前 reload 都进入 sticky fail-close，直到 reset；reset 后 SRAM 内容无效，必须重新全行 match/write。

## 周期攻击

| 场景 | speedup vs bit-sparse |
|---|---:|
| M377 理想 finite-event | 1.081047x |
| streaming L2 + 1 count seal/phase | 1.080935x |
| streaming L8 + 1 count seal/phase | 1.080490x |
| L2 + 0.125 blocking cycle/replayed descriptor | 1.069523x |
| L2 + 0.250 blocking cycle/replayed descriptor | 1.058349x |
| L2 + 0.500 blocking cycle/replayed descriptor | 1.036687x |
| L2 + 1.000 blocking cycle/replayed descriptor | 0.995920x |

因此固定 latency 只是每次 replay 的启动项；只要预取流水化，未来 descriptor 的 SRAM response 可隐藏在当前 descriptor 至少 4 个 compute cycles 内。相反，blocking read、共享端口、错误的 credit/FIFO 实现或把 active-count 逐 descriptor 提交，都会按 42,943,778 次 replay descriptor 放大。

## VCS 与 DC 门

VCS 必须做软件 pack/unpack + exact plus/minus/fallback cycle miter，覆盖 active_count 0/1/2400/3000、全 pop1、exact-center、plus-only、minus-only、mixed residual、非法 row/center/distance/flags、L1/2/4/8、req/rsp/backpressure、FIFO 0..8、所有状态 reset/reload 和第三 replay 攻击。断言至少包括 `retire<=1/cycle`、write iff active retire、count 守恒、req/rsp/outstanding 守恒、两次 replay 精确同 count、sticky fault 无副作用。

只有 VCS 通过且冻结人口的 post-VCS cycle replay 在所有实际 stall 后仍不低于 1.05x，才进入 DC。DC 要报告 3.0 ns TSMC28 controller+FIFO/wrapper 的 setup/hold、unconstrained paths、增量逻辑面积，并给出两个 3000x48 SRAM macro 的面积/时延/能耗或明确标为非 paper-ready envelope。logic-only DC 不能产生 PPA/energy claim。

## 打铁评分

- 总分：88/100
- evidence integrity：98
- exactness argument：96
- controller boundary completeness：94
- cycle-model realism：78
- RTL readiness：72
- P0/P1/P2：0 / 4 / 4

P1 是：未物化 packed descriptor bytes；未建 SRAM req/resp/backpressure recurrence；未 timestamp active-count seal 和 dual-replay count；未冻结独立 descriptor SRAM 端口。这些都不是现在可以忽略的“实现细节”，必须在 VCS 关闭。

## 复跑

从仓库根执行：

```bash
/opt/anaconda3/bin/python \
  hw_autoresearch_nts07/system_simulator/scripts/analyze_m380_m377_active_descriptor_controller_prertl_hammer.py \
  --contract hw_autoresearch_nts07/contracts/m380_m377_active_descriptor_controller_prertl_hammer_contract_r1_20260825.json \
  --output-dir <new-empty-output-dir>
```

结构化证据见 `m380_m377_active_descriptor_controller_prertl_hammer_r1.json`，完整敏感性表见 `cycle_sensitivity.csv`。
