# MLP 同 GPU 计时：dense vs 跳零行 vs 再合并行

24 个 `fc1/fc2`，10 帧，CUDA event 只包 Linear 本体。AEE 三臂都是 **0.715769949**（与未改网一致）→ 实现是无损的。

| 实现 | Linear 合计 ms | vs dense | 行账（10 帧） |
|---|---:|---:|---|
| dense `F.linear` | **118.9** | 1.00× | — |
| AAC：取出非零行再 GEMM 写回 | 551.6 | **慢 4.64×** | 11.16M 行里 9.09M 非零（81.4%） |
| unique：`torch.unique` 后再 GEMM | 5844 | **慢 49.2×** | 非零里 8.21M 个不同行 |

名义上零行能省 MLP 的 21%、全网 5.6%；**在这块 GPU 上 gather 税把那点算术优势吃光还倒挂。** unique 更是控制开销。这就是 Sparsity Tax：跳零在 FireFly 那种 bitmap 数据通路上才便宜，在通用 gather-GEMM 上不便宜。

行加权空行只有 19%，因为 Stage0 FC1 行数极大且几乎全非零；MAC 加权空行更高（L1/FC2 空得多但行少）。

## 去留

- 无损 AAC/unique **功能正确**，GPU 这么实现 **不能当加速**。
- ASIC/专用跳零（FireFly bitmap、Gustav NRV）仍是 MLP/注意力该抄的 A，不要用这次 GPU ms 否定它们。
- 不要再在 GPU 上堆 `torch.unique` 当 Prosperity。
- 下一步若要周期：隔离 RTL 上对 **L1.b0.fc1**（空行 46%）做 bitmap 跳零对照，或继续 Codex 那条 FC1→PSN 证书链；不要再数 dense G、不要再跑 GPU gather。
