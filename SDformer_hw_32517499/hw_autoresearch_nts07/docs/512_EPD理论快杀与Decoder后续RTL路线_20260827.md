# M512｜EPD 理论快杀与 decoder 后续 RTL 路线

日期：2026-08-27

## 裁决

原 M510 `phase-balanced multi-source EPD scheduler` 在写 RTL 前正式
`NO_GO__KILL`。原因不是 decoder 份额小，而是强 A1 已达到 product-issue
理论下界：

- 四层 `Cout={384,192,96,96}` 全部是 96 的整数倍；
- 对于每个 active source 与每个合法 tap，A1 按 96-output-channel
  slice 发射，每拍恰好 `96/96` useful lanes；
- K3/S2/P1/output-padding1 的 `4/6/9` 边界只改变完整向量数，不产生
  lane tail；
- 记 active source `e` 的合法 tap 数为 `b_e∈{4,6,9}`，
  `Cout=96q`，则 `P=96·Σ_e(b_e q)`，A1 恰用 `P/96` 拍完成。

因此仅在 product-issue 轴上，`EPD/A1 ≤ 1.0x`。若公平 A1 同样拥有
deterministic parity banking、forwarding 和同等 FIFO，原候选剩下的只是通用
memory-stall/OOO 调度；这与 FireFly-T 的 multi-spike decode、weight dispatch
和 bank-aware out-of-order execution 直接重叠，不再单独写 RTL。

M512 独立打铁为 `98/100`，`P0=0, P1=0`，封存于
`reviews/m512_epd_theory_fastkill_hammer_r1_20260827/`；其 `SHA256SUMS`
hash 为 `f591e8aa8305bd5b0d9feced37d8a0a0bae6233b9b2fd4c993b76ce245da7409`。

## 仍然成立的事实

M510 已经封存确认：

- 旧 `620,302,905 cycle/frame` 只是 included-scope envelope；
- 漏记 decoder 在分析界中占修正 envelope 约 `21.57%--22.83%`；
- dense polyphase 相对 exact activation-bit-sparse 的机会为 `4.48--4.81x`。

最后一项是 A0/A1 的 activation-sparsity opportunity，可用于说明统一 C2
source protocol 为什么应支持 decoder，但不是新 scheduler 的创新倍率。

## exact S10 后只保留两个条件候选

### PGPR：Parity-Gather Psum Residency

不再试图降低 product 数，而是减少 decoder 目的端 psum 的 SRAM
read-modify-write：按 output parity/destination tile 聚集 source descriptor，在小型
output-resident accumulator/RF 中完成一个 tile 后一次 commit。

必须收费：

- input bitmap scan/descriptor 生成；
- weight cache 和 RF 容量；
- destination tile 切换和 commit；
- 与“同 parity banking + output-stationary + 同 SRAM ports”强 A1 对比。

只有同资源总周期 `PGPR/A1 ≥1.30x`，且收益不是来自免费的额外端口/
RF，才准写 RTL。

### TDR：Exact T10 Temporal-Delta Replay

对连续 timestep 的 binary deconv input 执行 exact XOR/signed delta，仅对
`0→1` 和 `1→0` 变化发射 `+/-weight`。这不是免费跳过，必须完整收取
上一 timestep 状态：

| 状态 | 容量 |
|---|---:|
| 四层 previous input bitmap | 870,300 B = 0.830 MiB |
| 四层 previous output，INT16 | 20.215 MiB |
| 四层 previous output，Acc24 | 30.322 MiB |

除非下游 ATLIF/BN 状态能以严格整数桥接吸收 previous-output 存储，否则 TDR
大概率被 SRAM 主导。exact S10 先测 `P_delta/P_A1`；若该比值
`≥0.7692`（即理想计算加速 `<1.30x`），在还未收存储税时就直接快杀。

## 执行顺序

1. M511 捕获四层 exact S10 bitmaps，并独立验证 payload；
2. 用 exact coordinates 重建完整 decoder cycle/envelope；
3. 先算 TDR XOR/signed-delta product ratio，过门才建带状态 SRAM 的 cycle model；
4. PGPR 必须和强 output-stationary A1 做等端口/容量模型；
5. 两者都不过门时，decoder 只作 C2 统一 bit-sparse 执行模式和完整度，
   不再增加新 RTL 贡献。

相关一手前作：

- [SNE：DATE 2022 sparse event convolution](https://arxiv.org/abs/2203.12437)
- [FireFly-T：multi-spike decode 与 bank-aware OOO](https://arxiv.org/abs/2505.12771)
- [DeltaCNN：exact sparse frame-difference propagation](https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html)

禁止表述：EPD 已实现、EPD/A1 >1x、4.48--4.81x 是新 RTL 加速、或任何
decoder system speedup/energy/PPA 已准入。
