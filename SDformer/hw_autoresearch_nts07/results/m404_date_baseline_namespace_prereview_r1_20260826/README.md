# M404 DATE 基线命名空间独立预评审

结论：`PASS_PREREVIEW__BLOCK_DENSE_HEADLINE_UNTIL_FORMAL_THREE_MODE_REPLAY`，评分 `82/100`，`P0/P1/P2 = 2/4/3`。本目录只封存基线口径、同资源预计算和正式 simulator 的准入清单；不修改论文正文、不修改 `docs/359`、不产生正式 headline。

## 三列公平基线

统一范围是 H67 ep35/no-running、M40 S10 的 **四个 bottleneck Conv3x3**，统一资源是 SHARED96、cmd32、L8/D8 descriptor SRAM、双 32 KiB slot、同一 weight-DMA/overlap/tail/commit recurrence。

| variant | cycles | vs dense16 weak | vs zero-elided strong | status |
|---|---:|---:|---:|---|
| Dense16 same-resource | 6,636,544,610 | 1.000x | - | prereview projection；未准入 |
| exact zero-elided bit-sparse | 742,148,386 | 8.942x | 1.000x | M397/M401 双封冻结；dense-relative 未准入 |
| M401 combined exact reuse | 641,790,704 | 10.341x | 1.156x | four-Conv trace-cycle replay；不是 RTL/system/PPA |

Dense16 的原始工作是 `51,840,000 × 16 × 8 = 6,635,520,000` term-block issue。预评审没有把这个 work 比直接当 cycle；按 M397/M401 已冻结 recurrence 另加 10 个样本首 phase preprocess `30,050`、17,280 phase tail `34,560` 和 commit `960,000`，得到未准入预测 `6,636,544,610` cycles。每 phase dense compute 为 384,000 cycles，足以完全隐藏 3,005-cycle next-preprocess，因此预计没有额外暴露 weight-DMA；这个结论仍必须由正式三模式 replay 产物确认。

## 一手文献口径

| work | 大倍率的真实分母 | 范围与方法 | 不可混淆项 |
|---|---|---|---|
| Prosperity | 平均 7.4x vs PTB；14.2x vs dense Eyeriss | cycle-accurate simulator；Fig. 8 覆盖列出的 CNN/transformer workloads | transformer 的 PTB/SATO/MINT 比较只跑其支持的 linear layers；A100 才是 PyTorch/SpikingJelly whole-model end-to-end |
| Phi | 3.45x vs strong Stellar；Table 2 单点 26.70x vs dense Eyeriss | activation/pattern-driven runtime simulator，DC/CACTI/DRAMsim3；Fig. 8 跨列出模型 | 26.70x 是 VGG16/CIFAR100 的弱 dense 单点，不能冒充强基线 headline |
| Bishop | 5.91x vs same-resource PTB | analytic cycle-accurate heterogeneous simulator；Figs. 12-13 作者明确称 end-to-end | 5.91x 包含 Bishop+BSA+ECP；Fig. 14 的 170.66x 是 self-attention-layer-only |

精确页码、表/图号和 URL 已写入主 JSON。Prosperity 官方仓库明确提供 Figure 8 的 cycle-accurate simulator 与 end-to-end time/energy 输出；本预评审没有找到 Phi 或 Bishop 的官方公开实现，因此对后二者只采用论文一手证据。

## DATE 表格建议

表题必须写：`H67 ep35/no-running four-bottleneck-Conv standalone trace-cycle comparison under one SHARED96/cmd32/L8 resource contract`。同时列 `speedup vs dense16 (weak)` 和 `speedup vs zero-elided bit-sparse (strong)`；正文主叙事优先使用强基线 `1.156x`，dense-relative `10.34x` 只作为明确标注的弱基线列，并且要等正式 replay 后才能准入。

正式 M404 simulator 至少需要 3 个 variant × 17,280 phases = 51,840 条 phase timestamp/component record，复现 `6,636,544,610 / 742,148,386 / 641,790,704` 三个周期值，冻结 scan、weight DMA、overlap、tail、commit 与 candidate-only metadata 的处理，再交独立 hammer 和双封。完成前禁止写 `10.34x cycle speedup`，也禁止把四 Conv 称作 full network 或 system speedup。
