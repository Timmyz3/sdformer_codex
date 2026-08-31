# M894｜motion c12 训练与 checkpoint 只读监测

观察截点：2026-08-29 06:33:21 +08:00。所有远端操作均为只读查询；未中断训练、未抢 GPU、未写远端文件、未启动新任务。`docs/359_DATE终局冻结_20260813.md` 未修改。

## 结论

`c12_binary_motion_ttx` 正常运行到 epoch 5，观察时约完成该 epoch 的 16%。epoch 0–4 validation loss 分别为 1.4773424580、1.4303007921、1.2563284863、1.3764584320、1.1996598925；epoch 4 是当前 validation 最优点。

当前目录只有持续增长的 `train.log`，没有新 `.pth`、checkpoint 或 identity manifest。这不是保存失败：配置明确采用 30 个 epoch、零基 epoch `{9,14,19,24,29}` 强制保存，且 `save_only_force_epochs: true`，因此最早预计在 epoch 9 训练结束后生成 `checkpoint_epoch9.pth`。以 epoch 0–4 的平均训练时间约 4,080 秒/epoch 和观察时进度估算，若吞吐稳定，首次 checkpoint 约在 2026-08-29 10:55 +08:00 出现。

需要特别纠正命名：该配置没有 `pattern_paft` 段，启动日志也没有 `[M71] installed hardware-weighted PAFT hooks`。严格身份是 `--finetune 1` 的 30-epoch full finetune，包含 H60 `binary_motion_xor_alpha=0.25`，并把 Q/K 及 all-non-QK 共 105 个 ATLIF 设为 binary output；不应把它写成已启用 M71 pattern-PAFT。

## Epoch 账本

| epoch | train loss | validation loss | train time (s) | max GPU memory (GiB) |
|---:|---:|---:|---:|---:|
| 0 | 2.2790298806 | 1.4773424580 | 4091.48 | 40.465 |
| 1 | 1.5833123297 | 1.4303007921 | 4082.60 | 40.334 |
| 2 | 1.4985018092 | 1.2563284863 | 4073.31 | 40.334 |
| 3 | 1.3268698593 | 1.3764584320 | 4076.51 | 40.334 |
| 4 | 1.3380258797 | 1.1996598925 | 4076.77 | 40.334 |

epoch 4 相对 epoch 0 的 validation loss 降低 18.79%，相对此前最优 epoch 2 降低 4.51%。这只是 loss 账本，不等于 AEE admission。

## 保存合同与风险

- `loader.n_epochs: 30`。
- `runtime.force_save_epochs: [9,14,19,24,29]`，编号为零基。
- `runtime.save_only_force_epochs: true`：保存判定不再依据 best loss；只在强制 epoch 或最终 epoch 保存模型。
- 模型路径：`.../c12_binary_motion_ttx/checkpoint_epoch{}.pth`。模型文件是 `{"model_state_dict": model.state_dict()}` 容器。
- `runtime.state_save_epochs: [29]`：optimizer/scheduler/scaler 训练状态只在 epoch 29/最终点保存为 `checkpoint_epoch29_state_dict.pth`；epoch 9/14/19/24 只有模型状态。
- 当前 validation-best epoch 4 **未保留**。原基线的“best”判定位于 validation 之前且使用 train loss；当前 patch 又被 `save_only_force_epochs` 覆盖。因此不能声称存在 best checkpoint，也不能从当前目录恢复 epoch 4 权重。
- 第一个可供硬件绑定的真实新身份是 epoch 9；若 epoch 9 精度回退，应先按固定 valid 协议验收，不能用 epoch 4 的 loss 替代 checkpoint 身份。

## 进程与 GPU 身份

- launcher PID 3653568，启动于 2026-08-26 22:01:41 +08:00。
- 主训练 PID 3716692，启动于 2026-08-29 00:41:03 +08:00，102 threads；16 个 DataLoader worker 子进程。
- 主进程命令精确绑定上述 config、父 checkpoint、输出模板与 `--finetune 1`。
- 主进程打开 `/dev/nvidia5`；远端可见 GPU 是 NVIDIA A800 80GB PCIe，UUID `GPU-499236d3-b46c-5d25-4a22-530d47ed5112`。观察时 47,483 MiB 已用、GPU utilization 95%、P0。
- `nvidia-smi` 的 compute-app PID 位于不同 PID 可见域，不能与容器内 3716692 直接数值相等；命令行、父子树和 GPU device FD 共同锁定当前作业。

## 身份锚点

- 远端 Git HEAD：`494593afa0ea81332ca21fcd68fdc9d6b72bbf1a`，但工作树有 51 条 porcelain 记录；仅 HEAD 不足以复现，必须随 checkpoint 封有效源码/patch manifest。
- config SHA256：`c7b5b994cb9f9a43478f3cb7c09e52a7aecf529fcd6a590f982a291e9eeed955`。
- H9 train entrypoint SHA256：`49c77538f2de2c54b709b05ae246da4cf7f36a147da990a03acb9e94a917446b`。
- full30 wrapper SHA256：`ce3516e48efb9596917dca44d15c40275edc182158853fa60e643f79b894c194`。
- 父 checkpoint SHA256：`7e8d524e0784977518f36b21d1c72190ebcb6fc74ed736b30bb1d93875766cbb`，大小 411,943,850 B。
- 启动 load audit：`checkpoint_overlay_keys=0, missing=210, unexpected=0`；这是从旧父 checkpoint 初始化新 overlay 的审计，不是新 c12 checkpoint 的完整加载验收。

## 硬件侧最小交接清单

epoch 9 checkpoint 出现后，先只做最小冻结包，不立即改 RTL：

1. `checkpoint_epoch9.pth`、SHA256、文件大小、config、有效 overlay/source patch、父 checkpoint 身份与一份机器可读 identity manifest。
2. 同配置回载审计：`missing=0`、`unexpected=0`；固定 valid 协议的 AEE/AAE 与样本清单。若不满足，checkpoint 不进入硬件 headline。
3. 同一冻结样本上的 full-network ordered trace：ATLIF、PatchEmbed、Conv、FC1/FC2、BN、attention、四层 ConvTranspose 的严格执行顺序和 shape/quantization 元数据。
4. C1/C2 最小 payload：每层 typed source descriptor、parent/destination ownership、zero/valid/terminal、权重与 accumulator 位宽；由于本线 105 个 ATLIF 为 binary output，必须重新测 activity/source-row，不能沿用 ep35 analog ledger。
5. decoder D0–D3 全层 payload（尤其补齐历史缺口 D1），以及 attention Q/K/V 或 score NPZ；所有 trace 文件进入同一 SHA manifest。
6. 至少三条 DSEC sequence × 10 sample 的分层统计，和当前硬件账本保持相同 sample identity；单一序列只能作诊断。

## Claim boundary

本监测只证明训练活跃、保存合同和身份边界。它不证明 epoch 4 checkpoint 存在，不证明 AEE、硬件 cycle、PPA、系统倍速或 paper-ready checkpoint。下一触发点是 `checkpoint_epoch9.pth` 的原子出现及哈希稳定。
