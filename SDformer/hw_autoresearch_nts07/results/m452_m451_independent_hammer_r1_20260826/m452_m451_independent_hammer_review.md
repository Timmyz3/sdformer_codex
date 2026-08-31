# M452 独立打铁：M451 exact K1 fused adapter

评分 **91/100**，P0=0、P1=3、P2=1。结论是：功能 RTL 可以进入 standalone DC → Formality → PT，但 `1.2019906647x` 仍只能叫四层 Conv trace opportunity。

## 独立验证

M452 没有复用 M451 的 TB、scoreboard 或 SVA，只复用了冻结 DUT：

- wide signed12 × signed8 加减全域：2,097,152 对，0 mismatch；
- narrow signed8 × signed8 加减全域：131,072 对，0 mismatch；
- 23,260 accept / 23,259 retire，差值正好是显式 reload attack 隔离的一个 buffered output；
- 13 类协议攻击、13-cycle stall、23,257 个 pop-push、23,257 个 II=1 pair；
- arithmetic / metadata / unknown / fail-closed leak / SVA failure 全为 0；
- full-interface 输出范围 `[-2176,2175]`，signed13 足够。

## 语义

M451 只输出 `update_delta=PWP±W`，模块里没有 old_psum，因此没有复活 M426 的 overwrite 语义。但这只是“不会在本模块里丢 old_psum”；最终仍需对 integrated accumulator 做 Formality，证明每个 accepted delta 都执行 `new_psum=old_psum+delta`。

## 资源公平性

“0 个新增 memory port”目前不是 RTL 证据。DUT 内没有 memory macro 和 address generator；它只暴露了 160 B PWP 与 96 B correction 两组输入。必须在集成设计证明：

- 两个既有 memory read port 真能同拍工作；
- 没有 bank/address-generator/interconnect conflict；
- correction source address、polarity、row tag、tile、block 与 PWP center 保持一致；
- 256 B/cycle 瞬时切换、96-lane pre-adder、可能的 pipeline/Fmax 变化都被计价。

因此 517,041,352 / 430,154,216 = `1.2019906646689706x` 仍不是 RTL-measured 或 system speedup。M451 冻结 contract 中的 `1.2019932175709652x` 是一个小的抄写误差，后续必须使用正确商值。

## 决策

- standalone matched M451-vs-M433 DC：GO；
- RTL→mapped netlist Formality：GO；
- slow setup / fast hold PT：GO；
- memory concurrency、integrated old_psum、full-pop cycle、PTPX、system/DATE headline：NO-GO，等待各自证据。

`docs/359` 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
