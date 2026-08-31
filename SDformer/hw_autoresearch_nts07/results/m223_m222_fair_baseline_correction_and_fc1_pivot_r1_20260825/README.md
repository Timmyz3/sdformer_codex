# M223 M222 公平基线纠正与 FC1 转向

M222 的 exact screen 保留，但其“下一步合成 96 MAC vs add-only”的建议被撤销。六个输入都是 exact binary，公平主基线也应是 96-lane add-only；96 MAC 只能作为传统 dense accelerator 的次级对照。

独立评审已从 60/60 份原始 bitpack 复算全部计数与 ratio：exact screen 为 GO（P0=0），原 next gate 为 NO-GO（P0=2），总分 88/100；封存 `SHA256SUMS` 为 `a29c86f5f38f2c4a877b5007ab5d7423db8b54ece0dd6f9a99a5b7492744aeea`。

在当前 `8 banks × 128 bit` 合同内：

- K4×D32 是最好合法点，但 serial ratio 只有 1.1049×；
- M218-like K8×D16 为 0.9463×，比强 K1 慢；
- K8×D32 的 1.7939×需要 16 个 128-bit bank 等价宽度和 256 add lanes，不是同资源点。

因此按 M222 预先冻结的 1.5× 门槛，patch 性能 RTL 为 NO-GO。最多可以做一个很小的 `96-add K1/D96 vs 128-add K4/D32` logic-only 定价作为负结果，不再占主线。

主线转 M224 FC1：已恢复 10 个 binary-eligible FC1 的 100 份 M51 bitpack，覆盖 stage 0/1/2；stage 3 两个非 binary FC1 保持 conventional path。M224 使用 Acc19、expanded destination 和强 K1/D96 基线，并把 M63 parent-residual 的 source-work reduction 与 K-bank 并行分开记账。

这也收窄论文贡献：M216/M218 暂时只能作为 context/request transaction amortization 机制，不能作为强基线性能贡献；第三项贡献是否成立改由 FC1 筛选决定。
