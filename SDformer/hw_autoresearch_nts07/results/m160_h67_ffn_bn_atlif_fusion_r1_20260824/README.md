# M160 H67 FFN BN/ATLIF 融合审计 r1

## 先回答为什么不只是 fc1/fc2

`fc1+fc2` 只是 Linear/MAC 引擎的账本简称。冻结 H67 的完整 FFN 是：

`sn1/ATLIF -> dropout1(p=0) -> fc1 -> BN1 -> sn2/ATLIF -> dropout2(p=0) -> fc2 -> BN2 -> residual add`

因此剪枝、跳过和融合都必须覆盖激活状态、BN 偏置和残差提交，不能只处理两块权重。

## ep35 checkpoint 数值结果

- 真实实例为 12 个 `MS_Spiking_Mlp`，共 17,664 个 expanded channels、4,416 个输出通道。
- 两个 Linear 都无 bias；两处 dropout 均为 0；BN 使用 eval running statistics。
- `Linear -> BN` 折叠的 24 组 PyTorch miter 最大绝对误差为 `8.29645e-6`。
- 17,664 个 BN1 offset 和 4,416 个 BN2 offset 中，精确零均为 0。
- 全零 MLP 输入下，sn1 输出为零，但 BN1 offset 在 6/12 个块触发了共 927 个 sn2 channel-time 输出。
- 完整零输入分支的 44,160 个单点 channel-time 输出全部非零；这主要说明 BN2/折叠提交常数不可丢弃，并不是活动率统计。

所以，`input==0` 或 `fc1 accumulator==0` 都不能直接整支跳过。允许的硬件优化是省掉 MAC 后走一个可证明的常数/触发旁路。

## 候选硬件结构

BN1 与 sn2 的推理代数可重写为：

`h[t,j] = gain[j] * sum_tau(Wt[t,tau] * acc_fc1[tau,j]) + offset1[j] * row_sum_Wt[t] + bias_sn2[t]`

其中 `gain[j]=alpha1[j]*weight_scale1[j]`。这使 BN1 的偏置项按“channel × time”可分离：

- 直接物化：176,640 个 channel-time bias values；
- 因式保存：17,664 个 channel offset + 每块 10 个 row sums 和 10 个 temporal biases，共 17,904 values；
- value-count 缩减 `9.866x`。这不是 cycle speedup。

BN2 可以直接并入 residual commit：

`output[j] = gain2[j] * acc_fc2[j] + offset2[j] + residual[j]`

这样 350,208,000 个 BN1 elements/frame 与 87,552,000 个 BN2 elements/frame，共 437,760,000 elements/frame，成为“不单独物化”的候选；residual 的 87,552,000 elements/frame 仍须提交。只有 RTL、VCS 和端口/重叠 recurrence 完成后才能把它们换算成周期。

## 对算法侧的硬约束

expanded-channel mask 必须原子覆盖：

1. fc1 output row；
2. BN1 channel；
3. sn2 对应 channel 的状态与参数；
4. fc2 matching input column。

PAFT 或后续 structured training 不能只 mask `fc1/fc2` 权重；还必须保持折叠后的 BN1-to-sn2 响应。M159 显示 stage 2 占 FFN 子图约 50.71%，仍应优先训练和实现。

当前等级：`PASS_CHECKPOINT_BOUND_BN_FOLD_AND_ZERO_PATH_AUDIT`。尚未接纳 BN-elision RTL、VCS、cycle/system speedup 或 PPA。
