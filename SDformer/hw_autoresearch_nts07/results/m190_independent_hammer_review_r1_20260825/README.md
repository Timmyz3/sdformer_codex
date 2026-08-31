# M190 独立打铁评审

结论：**90/100，`PASS_STANDALONE_SUM_POSITIVE_SCREEN_ONLY_FLAT_GATE_REQUIRED`**。M190 的 single-hole-elision K7 steering 与七输入 signed-INT8 Acc24 在模块内成立，且第一次让 `M188+M190` 的 standalone 面积和跨过 K8 的吞吐/面积门槛；但优势只有 **0.592082%**，必须经过 matched flattened composition 才能决定是否真的替代 K8。

## 独立功能审计

- sealed VCS input/output manifest 全通过：254 个合法 mask、24,768 次 numeric lane check、八个 lowest-hole 位置、stall、II=1 replacement、overflow/full/empty attack 均有证据；21/21 SVA cover 非零，最少命中 1 次，0 assertion failure signature。
- 独立 Python proof 不 import 生产 analyzer：穷举 mask `0x01..0xfe`，得到 lowest-hole histogram `127,64,32,16,8,4,2,1`；逐 mask/96 lane 做 24,384 次数值 miter、170,688 次相邻 steering 检查，28 个 two-source mask 全覆盖，0 mismatch。
- 独立 VCS bench 再次跑完 254 mask，并通过层次信号逐 slot 核对 selected weight 与 invalid-zero：24,864 次 numeric miter、172,704 次 adjacent steering check；empty、full、overflow 在各自 reset 后独立 fail-close；stall 两拍稳定、连续两组 II=1 replacement 通过。
- 映射理由很简单：最低空 bank 为 `h` 时，slot `s<h` 取 bank `s`，否则取 bank `s+1`。每个 slot 只有 `{s,s+1}` 两种相邻来源，七个 slot 恰好覆盖除 `h` 外的七个结构 bank；额外空 bank 保持 invalid 且精确贡献零。
- 七个 signed INT8 的和界为 `[-896,889]`，11-bit signed tree 足够；再与 Acc24 用 25-bit 扩展相加并检查 bit24/bit23，独立边界审阅未发现 standalone P0 数值缺陷。

## DC 与严格门槛重算

同为 TSMC 28HPC+、3.0 ns、flattened-before-mapping、ideal clock、ZeroWireload、0 macro：

| datapath | area (µm²) | cells | seq | levels | critical | setup | hold | multiplier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M185 K8 Acc24 | 27,129.815772 | 28,266 | 2,344 | 40 | 1.66 ns | +0.8771 ns | +0.0233 ns | 0 |
| M190 K7 elision+Acc24 | 26,487.467754 | 28,549 | 2,343 | 44 | 2.11 ns | +0.4269 ns | +0.0241 ns | 0 |

M190 比 M185 少 **642.348018 µm² / 2.367683%**，但 cell 数反而多 283、logic level 多 4、critical path 长 27.1%；因此它是面积小胜，不是所有维度都胜。

standalone 求和后的严格结果：

| screen | K8: M184+M185 | K7: M188+M190 |
|---|---:|---:|
| logic area | 37,156.643801 µm² | 36,905.147786 µm² |
| exact wall cycles | 97,607,807 | 97,694,539 |

K7 面积少 **251.496015 µm² / 0.676853%**，但 cycle 多 **0.088857646%**，即吞吐只保留 **0.999112212×**。两者合并后，K7 conditional throughput/area 是 **1.005920822×**，优势 **0.592082%**。对应 M190 break-even 是 **26,705.976561 µm²**，本次只多出 **218.508807 µm²** 裕量。这个 margin 小于常见的跨模块 flatten/shared-logic 波动，standalone-sum 只能通过筛选，不能完成架构 admission。

## 打铁优先级

### P0

- 立即做 matched flat `M188+M190` 与 matched flat `M184+M185`，保持 3.0 ns、一拍一组与完全相同 I/O 边界；用 flat K8 面积重新算门槛。若 K7 不再赢，退回 K8。
- 接真实 weight-SRAM response。M190 的物理端口仍输入八个完整 bank bus；当前只证明内部 8→7 elision arithmetic，没有证明 SRAM 容量、响应位宽、响应能耗或 memory timing 下降。
- response composition 必须关闭 M186 的 reset 后 delayed untagged-response alias，加入 epoch/identity 或 flush-ack quarantine。

### P1

- 将 M187 的 120 个冻结 FC2 payload 逐事务 replay 到 composed RTL，端到端检查 M188 mask、SRAM weight、M190 sum、stall、token 边界和 II=1。
- PAFT 训练完成不等于算法 admission；`valid825=false` 仍须补齐。M190 的 multiplier-free identity 依赖 `sn2 threshold=1`，但 checkpoint accuracy 尚未验证。
- flat K7 仍胜后再做 Formality、PT/SAIF/PTPX。**0.592082% 只能称 conditional standalone logic throughput/area**，不能写成 FC2、FFN、physical、system 或 headline speedup。

`docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。机器可读判定见 `m190_independent_hammer_review_r1.json`。
