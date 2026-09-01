# M1753 作者侧 source-only 收据

结论：M1753 源码与合同的 CPU-only 自检通过；当前不具备任何 EDA 或论文数字准入权。

- 三轴：K1、K8、等带宽 K1x8；每轴同一五个 directed case、261 个 accepted sources、3 ns。
- SAIF：mapped top 的 public-port 激励与 DUT-only scope；15 个坐标全部通过后才允许 PTPX。
- PTPX：整块 mapped component 的 internal/switching/leakage/total；禁止 cell collection 相减或加回。
- 口径：只能称 `directed component`，不能称 production、trace、frame 或 system energy。
- 强制同列：K8 对 K1x8 周期 `1.016728x` 与 throughput/mm2 `4.562720x`。
- 边界：logic-only pre-macro；不含 weight SRAM、testbench memory、IO PHY、clock tree、寄生参数。

M1730 的 runner SHA 已与它自己的合同不一致，且无 M1731/M1732，因此 M1753 不继承 M1730 authority。下一步必须由不同作者完成 M1760 hammer，再由 M1761 精确 SHA release 授权唯一一次运行。
