# FireFly bitmap 跳零周期（L1.b0.fc1 与全部 MLP Linear）

相对 GPU gather（慢 4.6×）：这里按 **bitmap 看 G 个 valid / 周期，非零行各 1 个 MAC 周期**，dense 每行 1 个 MAC 周期。不是 ASIC PPA。

## 单层 L1.b0.fc1（空行 45.9%）

| 调度 | vs dense |
|---|---:|
| 每行先试探再 MAC | **1.54×（更慢）** |
| bitmap G=8 | 0.666 |
| **G=32** | **0.572** |
| G=128 | 0.549 |

这一层约 **−43% 周期**。它只占全网 1.19% MAC，系统只省 **0.51%**。

## 24 个 Linear，G=32，按 7.08G 加权

| | vs dense |
|---|---:|
| 行数加权 | 0.846 |
| **MAC 加权** | **0.771（MLP −22.9%）** |
| 折合全网 | **−4.90%** |

最热的 Stage0 FC1：vs ≈ **1.00–1.02**（几乎全非零，inspect 是纯税）。  
省在 L1.b0.fc1（0.57）和若干 FC2（0.52–0.73）。

## 结论

- Bitmap 分组跳零在 **空行真多的层** 成立；在 Stage0 FC1 上不成立。
- 整段 MLP 专用跳零大约 **1.3× 这一段**，全网 **约 5%**，和无损 AAC 账一致，并扣掉 inspect。
- **成不了倍。** 和 Jung 关 SSA、r0 元素跳零是同一类 A。
- 不要用逐 token 试探，也不要 GPU gather。

`results/bitmap_cycles.json`、`bitmap_all.json`
