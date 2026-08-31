# M222 H67 patch K-bank + line-buffer 公平筛选

结论：**单靠 M218-like 的 source-owned K8 bank coissue 不能击败强 K1 基线。**

本轮从已校验 M51 归档恢复 60 个真实 bitpack（10 samples × 6 个二值合格 patch Conv3x3），逐 output pixel 精确展开有效 3×3 receptive field，共守恒 1,774,268,587 个 source contribution、170,329,784,352 个 source×destination product update。

公平 K1 不把一个 96-byte weight row 留在单个 128-bit bank 上分六拍读取，而是使用同样存在的 8 个 bank 中的 6 个做 output striping，一拍完成一个 source contribution。对照结果如下：

| 点 | add/product lanes | 128-bit banks | serial cycle ratio vs K1 | 解释 |
|---|---:|---:|---:|---|
| K1×D96 striped | 96 | 6/8 | 1.000× | 强基线 |
| K4×D24 | 96 | 8/8 | 0.842× | 同 96-op capacity，bank imbalance 使其更慢 |
| K8×D16 | 128 | 8/8 | 0.946× | M218-like，仍慢 5.4% |
| K4×D32 | 128 | 8/8 | 1.105× | 当前 8-bank 下的最好点，但不够 1.5× |
| K8×D32 | 256 | 16 | 1.794× | 2× bank width/capacity，不再是当前同端口点 |
| K8×D48 | 384 | 24 | 2.558× | 3× bank width/capacity，需要面积/能耗定价 |
| K8×D96 | 768 | 48 | 4.454× | 6× bank width/capacity，只是宽阵列上界 |

`serial cycle` 包含一 input channel-vector/cycle 的理想 line-buffer scan、source service 和一 output token/cycle 的 commit；dynamic BN barrier 仍未计入。profile100 patch/compute-envelope 字段只是用本轮 s10 ratio 缩放冻结 ledger 的敏感性，不是 admitted speedup。

## 对论文贡献的影响

- M216/M218 的创新仍可表述为减少 context/request update；相同 active weight work 没有被跳过。
- `4.952×` 是相对 source-owned、六 slice K1 的有界 service ratio。面对能把一行权重条带化读取的强 K1，它不能作为通用吞吐头条。
- patch 若要 2×以上，需要证明“binary event 去乘法器后，在同面积内可以部署更宽的 add-only 阵列”，同时价格化 SRAM word width、accumulator ports、combine tree 和 line buffer。

下一门建议合成两组 matched standalone datapath：`96 INT8 MAC` 与 `K8×D32/K8×D48 add-only`。若等面积归一后 K8×D32 仍无法越过 1.5×，patch 性能贡献应停止，转 FC1；不能用增加 bank 数直接制造 headline。

机器可读结果见 `m222_h67_patch_kbank_linebuffer_premodel_r1.json`，完整点表见 `m222_model_points.csv`。本轮未修改论文正文或 `docs/359`。
