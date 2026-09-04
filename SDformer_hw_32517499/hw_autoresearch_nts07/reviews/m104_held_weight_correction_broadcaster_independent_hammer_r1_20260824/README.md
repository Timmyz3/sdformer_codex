# M104 held-weight correction broadcaster 独立打铁评审 r1

日期：2026-08-24  
评分：**78/100**  
严重度：**P0=1，P1=4，P2=2**

## 结论

M104 的独立功能结论是 **GO**：冻结 RTL 通过原 sealed 商用 VCS/SVA，也通过本评审新增的 exact-RTL-SHA 商用 VCS/SVA 反例集。96 lanes 的 signed INT8→signed12 正向与取负均 bit-exact；同拍非法请求与 `output_ready` 释放不会让旧 stalled output 泄漏；sticky fault 只能 reset 清除；三拍 load、`last_for_key`、backpressure turnover 和连续 event II=1 均未发现功能反例。

M104 的当前性能账本结论是 **P0/NO-GO**。冻结 M103 的 `190,360,330` correction tokens、`416,582,585` combined tokens 和 `2.6750597075` baseline ratio 可以保留，但只能称为“perfect phase/key batching 加上每 key 一次免费融合/重叠”的条件模型。当前 RTL 明确要求每 key 三个独立 load accept、每 event 一个独立 descriptor accept，并将 simultaneous load/event 判为 fault，因此按当前端口协议串行记账是 `E+3G`，不是 `E+2G`。

允许现在启动 **production-only logic-only DC**。ordered transpose 和 accumulator-bank trace 不是测这个孤立逻辑块面积/时序的前置条件，但它们是任何 scheduled-cycle、physical、system 或 headline 性能结论的前置条件。

## 证据身份

| 证据 | 独立结果 |
|---|---:|
| production RTL SHA256 | `37f86144563d45ea96f594847828a00c7d872602419d81a070738f12b4417f6a` |
| production SVA SHA256 | `ad63c0317b64b5e53aecd037d401669c42f5b4b40409563ed216e4eb776e2f98` |
| production TB SHA256 | `7ed7fcf389c49dcc152a002416f6af9198fdb7c770373b6d711c828984529916` |
| directed VCS filelist SHA256 | `a04e09b3029ee030f53e2cac6146ae13ed6c22bd96e57d86cbfae0adafbe6cbe` |
| production-only DC filelist SHA256 | `4507f6af3f41cae8c1c26f6779f3c33803d30e03dcbaeef36348ee905f99fd36` |
| M104 contract SHA256 | `bbd086a36719f3682216d39450dfc86db46c9373fc508f65657cfac2277dbdd5` |
| M103 audit SHA256 | `935119fab809e15f49089926550f89b3c84c2b13c0be58c96b0ea8709ed683fe` |
| M104 published result SHA256 | `8b00f57d368afe3c80633b0bfdd0770b9200090085204d0ab47c39c36aaaf205` |

原 sealed run 的 input manifest 9/9、output manifest 4/4 校验通过，compile/sim rc 均为 0，VCS 版本为 V-2023.12-SP1。原定向回归 PASS：6 groups、21 load beats、9 events、5 个 II=1 pairs、3 stalled cycles、10 类 protocol attack；八个 frozen cover 数均命中。

独立反例 run 位于 `vcs_adversarial_run_r3/`。它仅引用上述 exact-SHA production RTL，SVA/TB 都在本 review 目录，input manifest 3/3、output manifest 4/4 校验通过。

## 独立 VCS/SVA 结果

独立 PASS 行：

```text
PASS M104 independent adversarial VCS signed_codes=256 lanes=96 signs=2 ready_release_fault=1 sticky_cycles=3 reset_recovery=1 ii1_turnovers=4 load_gap=1 last_wait=1
```

覆盖与结论如下：

1. 三组 payload 的 96 lanes 合计覆盖全部 256 个 INT8 bit pattern，每个 pattern 都检查正向 sign-extension 和负向 two's-complement 结果。`-128 → +128`、`+127 → -127` 也包含在穷举中。
2. 旧结果 stalled 时，在同一拍同时释放 `output_ready` 并送入 wrong-source event，`protocol_error` 当拍拉高，`output_valid/output_accept` 均为 0，内部旧结果不 retire；随后连续三拍仍被 sticky fault quarantine。
3. 只有 reset 清除 fault、held key 和旧 output；reset 后重新三拍 load 与合法 event 成功。
4. 合法 `last_for_key` 在 output backpressure 下保持未 accept 时不会释放 key；`output_ready` 释放的同拍，旧 output retire、新 event accept、新 output 替换，未出现 bubble，accepted last 随后释放 key。
5. 三拍 load 中间插入两个 idle cycles，`collecting/expected_load_beat` 均保持，最终正常完成。
6. 连续合法 event 在 output ready 时达到 II=1。该结论仅针对这个一项输出 elastic buffer 的模块端口，不包含 transpose、SRAM、accumulator 或下游 bank conflict。

未找到上述 frozen directed contract 内的 ready/valid 功能反例。需要写入集成合同的一点是：accepted `last_for_key` 之后若 producer 继续保持同一个 `event_valid`，它会被解释为“没有 held key 的下一笔请求”并触发 fail-closed；producer 必须像标准注册 ready/valid source 一样在 accepted edge 后撤销或换成合法请求。

## P0：token ledger 与当前 RTL 协议不一致

令：

- `E = 188,148,490` correction/fallback destination events；
- `G = 1,105,920` phase-local `(source, block)` weight groups；
- `P = 226,222,255` frozen PWP tokens；
- `B = 1,114,383,288` fixed8 baseline tokens。

独立复算：

| 模型 | correction tokens | combined tokens | `B / combined` |
|---|---:|---:|---:|
| frozen M103 perfect batching，`E+2G` | 190,360,330 | 416,582,585 | 2.6750597075 |
| 当前 M104 互斥串行端口，`E+3G` | 191,466,250 | 417,688,505 | 2.6679769126 |

M104 published JSON 同时写了 `weight_load_tokens_per_group=3`、`destination_tokens_per_event=1`，但 `correction_tokens=190,360,330`。按它自己的字段直接计算：

```text
E × 1 + G × 3 = 191,466,250
```

声明值少算 `G = 1,105,920` tokens。源码也证明当前接口不能免费重叠：

- event 只有在 `held_valid_q` 已经为 1 时才合法；
- 三个 load beat 都需要独立 `load_accept`；
- 每个 destination 都需要独立 `event_accept`；
- `load_valid && event_valid` 被判为 protocol collision；
- 一个 held vector，last accept 后才能开始下一个 key。

因此 `2.6750597075` 不是当前 M104 RTL 的 literal token envelope。它可以作为显式的理想模型保留，前提是增加并验证以下至少一种机制：第三 load beat 融合首 destination、上一 key 的 last event 与下一 key preload 重叠，或等价的 ping-pong held buffer。否则当前模块对应的是同样未调度的 `2.6679769126` 条件 token ratio。

两个比值都不是 scheduled cycles、物理加速、全网加速或摘要 headline。

## production-only DC 准入

`date_m104_held_weight_correction_broadcaster_logic_only_dc.f` 只含 production RTL，作为 source inventory 是正确且足够的。M104 可以在 ordered schedule 前先做 logic-only DC，因为这一动作回答的是“当前模块在同一库/同一约束下有多大、能跑多快”，不是“全模型能否以该顺序供数”。

正式 DC run 仍必须 fail-closed 固定：

- exact RTL/filelist SHA，top=`m104_held_weight_correction_broadcaster`，`TAG_W=32`；
- `SYNTHESIS` define；
- 与对照组相同的 library、PVT corners、SDC、clock、uncertainty、I/O delay 和 fanout；
- 明确 0 macro、ideal clock/ZeroWireload 等 pre-macro 限制；
- 保存 DC log、mapped netlist、mapped SDC、area/QoR/timing reports 与 output manifest。

DC 可准入的结论仅是 production RTL 的 logic-only area/timing。它不能修复 token P0，也不能准入 transpose 可实现性、accumulator throughput、physical/system speedup 或 paper PPA。

## Findings

### P0

- **M104-P0-01-TOKEN-LEDGER-RTL-PROTOCOL-MISMATCH**：published JSON 的字段算术为 `E+3G`，结果却冻结成 `E+2G`；当前 RTL 也没有实现所需的一次/key 免费融合或重叠。修复 analyzer/result，或实现并 VCS 证明融合/双 buffer 后，才能把 `2.6750597075` 关联到具体 RTL。

### P1

- **M104-P1-01-PERFECT-KEY-BATCHING-NOT-SCHEDULED**：M103 只有 order-independent groups，没有 bounded transpose queue、phase drain、backpressure 或实际 record replay。
- **M104-P1-02-ACCUMULATOR-SEMANTICS-PORT-CUT**：destination tag→bank/address/port、PWP dependency 和有限位宽 overflow/saturation 均未接入；重排后的 bit-exact final state 未证明。
- **M104-P1-03-E-PLUS-2G-NEEDS-NEW-MICROARCHITECTURE**：若坚持理想 envelope，需要 load/first-event fusion 或 ping-pong preload；当前单 held vector 与 collision policy 只能串行 `E+3G`。
- **M104-P1-04-DC-RUN-NOT-YET-SEALED**：RTL-only filelist 足以启动，但尚无 M104 专用 exact-SHA DC runner、SDC 和产出收据；当前不能报面积、Fmax 或 PPA。

### P2

- **M104-P2-01-LAST-VALID-RELEASE-CONTRACT**：last handshake 后 lingering valid 会成为下一笔非法请求；应将 producer release timing 写进接口合同，并在集成 SVA 中绑定。
- **M104-P2-02-DIRECTED-NOT-FORMAL-EXHAUSTIVE**：signed data 已穷举 256 codes，但控制状态序列仍是 directed VCS，而不是 formal 全状态证明；进入更复杂 transpose 前应增加 bounded formal 或随机长回归。

## 下一步

1. 先修 token P0：若不改 RTL，将 M104 literal model 改为 `E+3G`；若要保住 `E+2G`，新增第三 load beat/首 event fusion 或双 held-buffer prefetch，并用 same-cycle fault 合同重新 VCS。
2. 可并行跑 production-only common-period DC，严格标注 logic-only、0 macro、not paper-PPA。
3. 生成 ordered semantic trace，建立 bounded phase/key transpose，记录 queue occupancy、spill/fallback、key transition 和 accepted/retired。
4. 接 destination accumulator bank/port，并冻结有限位宽更新语义；做原顺序与重排顺序 final-state bit-exact miter。
5. 最后才运行 scheduled token/cycle simulator；在这之前所有 `2.6751x/2.6680x` 仅为条件 token ratio。

本评审没有修改 production RTL/SVA/TB/contracts/results，也没有触碰 `docs/359`。
