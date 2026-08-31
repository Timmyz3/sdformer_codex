# M204 + M205 独立打铁评审 r1

结论：**84/100**。M204 功能和 standalone logic-only DC 通过；M205 的直接 ready/valid
组合在本次独立压力范围内通过。但主线程提出的 fixed `full final +1 / partial final +2 /
zero +0` 修正规则必须撤销，`90,107,277 cycles / 1.272322434x` 不能升级成 M205 RTL
周期或 RTL speedup。

## 功能与 DC

独立 M204 VCS 用四个 stage0 window 把两个 window buffer 填满，让第三、第四个合法
descriptor packet 持续 hold。观察到 4 个 full-hold cycle、2 次 `pair_release &&
descriptor_accept` 同拍 release/refill，4 个 group 严格按 tag/beat/bitmap 顺序输出，零丢失、
重复和乱序；合法 fullness 没触发 `protocol_error`。另做 tag、window-last 和 done-count 三个
fail-closed attack，全部拒绝并 sticky fault。

独立 M205 VCS 覆盖 stage3 dense 四窗、stage3 两满窗加 partial tail、stage0、stage2 sparse 和
zero token：23 个 raw packet、66 个 group、5 次完成，99 个 full-buffer descriptor hold、2 次
同拍 release/refill。header、descriptor、compact-done 三个 accept 边界逐拍检查 213 次，
3408 次 unused-entry/release scrub 检查，零丢重乱；非法 composite header 和非法 raw prefix
均 fail closed。

M204 sealed DC 的全部 input/evidence SHA 通过。报告精确为 14,910.336072 um2、23,113 cells、
1,901 sequential cells、86 logic levels、2.19 ns critical path、setup +0.5881 ns、hold
0.0000 ns；五类 constraint clean，零 macro，mapped netlist/resource 中无 `DW_mult` 或
multiplier cell。resource report 的 `mult_arch: and` 是常数地址算式映射标签，不是可引用的
乘法器。该结果只有 ideal-clock、ZeroWireload、zero-macro standalone logic-only DC 资格；
hold 恰为零，绝不是 physical 或 paper PPA。M205 本身尚未 flat DC。

## selector stale attack

合法 stage0 轮换、stage3 四窗和 partial tail 中未观察到 stale bitmap，release 前 bitmap 都
自然 drain 到零。但对 partial refill 后 `entry_count` 之外的 unused entry 注入一个 stale
bit，当前无 entry guard 的 selector 会把它作为真实 bank source 输出，且不报协议错。故
当前设计依赖“所有释放 entry 永远 scrub 为零”的单点不变量；这不是合法 traffic P0，但属于
P1 硬化缺口。建议加 registered per-entry-valid vector 作 selector guard，避免把 count decode
重新放回长路径，并用 SVA/Formality 证明 unused-entry-zero 和 release-all-zero。

## fixed tail rule 反例

独立 continuous-source VCS 扫了 256 个 token，并用另一条 Python 路径重算 M203 wall：

- full final 88 例：`+1:62, +2:22, +3:2, +4:1, +8:1`；
- partial final 96 例：全部 `+2`；
- zero 72 例：全部 `+0`。

full `+1` 明确不普适。stage0 W1 暴露每个 window 的 registered candidate-load/release 边界；
pair 模式下，一个 odd full tail 必须等 `upstream_done_seen` 才能 drain。若 full window 后还有
零 raw beat，M202 仍需扫完才发 done，而 M203 finite-wall 已在 window close 时让 odd job
立即可执行。blocks=8、仅一个 full window 的反例因此达到 `+8`。

partial `+2` 和 zero `+0` 只是在本次 96/72 例中成立，尚不能代替 frozen-H67 全 payload
证明。下一准入必须把 M202 queue/backpressure、M204 group register、pair eligibility、done
和 token completion 全部写进 RTL-semantic recurrence，再重放冻结 H67。

## 引用边界与下一步

`90,107,277 / 1.272322434x` 仍可称为 M203 exact-payload analytic schedule opportunity；
禁止称 M205 measured/RTL/physical、complete-FC2、FFN、system 或 headline speedup。

最值钱的性能改进是 stage0 next-window first-group prefetch：stage0 有 4,490,102 个 compact
window，消除每窗 registered 边界比继续加 scanner width 更可能产生系统性收益。第二优先是
由 ATLIF/producer 提供 final-window parity 或 descriptor/window count，使 pair-mode odd full
tail 不必等 trailing raw scan。随后再做 flattened M205 DC，并考虑融合 M202 queue 与 M204
window store。

`docs/359` 未修改，SHA-256 保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
