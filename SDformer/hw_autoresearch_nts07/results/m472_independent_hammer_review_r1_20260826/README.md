# M472 official Prosperity H67 独立打铁评审

## 裁定

**90/100，PASS with strict tile-workload boundary。** 准入的只是：冻结 H67 ep35 S10 四层 Conv `original16` 的 K=16 support-tile 同负载上，Prosperity 官方 product-sparsity 相对官方 bit-sparsity 的 **2.459487119674×** 内部对比。

禁止将它与 M430/M467 直接相除；禁止当作单一完整 Conv 映射、全网、系统、FPS、能量或 PPA 数字。

## 独立复算

| 检查 | 结果 |
|---|---:|
| M410 original16 全量扫描 | 51,840,000 rows |
| phase/order/population mismatch | 0 |
| N=128→N=768 全 phase 方程 mismatch | 0 |
| 聚合 bucket mismatch | 0 / 55 |
| 官方 N=768 独立 mode-run | 84 |
| 官方 N=768 counter mismatch | 0 |
| 官方 N=128 独立 mode-run | 10 |
| bit cycles | 556,188,432 |
| product cycles | 226,140,006 |
| product / bit | 2.459487119674× |

## 主要攻击结果

1. **中等级粒度限制**：M472 把 K=6912 因子分解成 432 个独立 K=16 `run_fc` 调用。计算/product 计数可加，因此 2.459487× 的同 support-tile 内部对比成立；但每个调用都重启 buffer/initial-DRAM 状态，所以绝对 DRAM 与 memory-stall 不能冒充官方单一完整 Conv 映射。
2. **低级完整性限制**：producer 有 result SHA 和 canonical payload SHA，但没有 runner SHA 与目录外层 seal。
3. producer 的直接 N=768 检查只有 3 phase；本评审已扩展到 10 sample×4 operator 身份网格及边界，未发现错误。

## 准入标签

> Official Prosperity product-vs-bit on frozen H67 S10 four-Conv original16 K=16 support-tile workload; 432 partitions aggregated; not same-resource or monolithic Conv latency.

详细证据见 `m472_independent_hammer_review_r1.json`。
