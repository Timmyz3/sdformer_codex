# M472 admission

状态：`ADMIT_STRICT_SUPPORT_TILE_ISO_WORKLOAD_ONLY`，独立打铁 90/100。

准入数字：在冻结 H67 ep35 S10 四层 bottleneck Conv3x3 的 K=16 `original16` support-tile 负载上，官方 Prosperity product-sparsity 相对其官方 bit-sparsity 模式为 **2.459487119674×**（556,188,432 / 226,140,006 cycles），product `num_ops` 降低 63.9608%。

强制边界：432 个 K=16 调用会分别重启 buffer/initial-DRAM 状态。因此绝对周期与流量不是 monolithic 四层 Conv 延迟，不能和 M430/M467 相除，也不能写作 H67 全网、FPS、能量、PPA 或 DATE headline。

本回执补齐了 producer runner SHA，并绑定 producer result、原 receipt、独立评审、官方 Prosperity commit 与 `docs/359` SHA。目录由 `SHA256SUMS` 与二次 seal 封存。
