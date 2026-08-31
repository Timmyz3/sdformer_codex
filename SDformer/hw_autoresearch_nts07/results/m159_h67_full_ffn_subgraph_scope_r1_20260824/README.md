# M159 H67 完整 FFN 子图口径 r1

## 结论

此前“FFN=`fc1+fc2`”只是矩阵计算简称。冻结 H67/Motion 实际使用 MS-spiking FFN，完整源码拓扑为：

`sn1/ATLIF -> dropout1(p=0) -> fc1 -> BN1 -> sn2/ATLIF -> dropout2(p=0) -> fc2 -> BN2 -> residual add`

推理时 `drop_path` 关闭。ordered trace 只 hook ATLIF 和 Linear，因此可见顺序是 `sn1 -> fc1 -> sn2 -> fc2`；BN、dropout、residual 由 exact-SHA 源码而非 trace hook 证明。

## 重新归类后的热点

| 部分 | cycles/frame | 当前 620,302,905 envelope 占比 |
|---|---:|---:|
| fc1 + fc2 | 159,784,111 | 25.7590% |
| FFN-local sn1 + sn2 ATLIF | 45,600,000 | 7.3513% |
| 已建模完整 FFN 子图（不含 BN/residual） | 205,384,111 | 33.1103% |

45.60M ATLIF cycles 原已包含在 128.02M 全局 ATLIF 桶内；这里是归属重分类，不能再次加到总周期。假设完整 FFN 免费时 1.4950× 只是 Amdahl 热点上限，不是设计倍率。

stage 2 的完整子图为 104,145,788 cycles，占全部已建模 FFN 子图 50.71%，仍是第一训练和硬件目标。

## 尚未进入周期分母的逐元素部分

- BN1：350,208,000 elements/frame。
- BN2：87,552,000 elements/frame。
- residual add：87,552,000 elements/frame。
- 合计相当于 5,472,000 个 96-lane element rows，但没有 affine、状态、端口、访存和 overlap recurrence，因此不能称为 cycles。

BN 默认带 running stats，是推理 fold 的候选；必须先绑定 checkpoint、eval mode、scale/bias 和 requant/threshold 数值，不能直接当免费。

## 对剪枝合同的修正

结构化 expanded-channel 删除单元必须同时覆盖：

1. `fc1` output rows；
2. BN1 对应 expanded channels；
3. sn2 对应 temporal 参数和状态 channel；
4. `fc2` matching input columns。

BN2 位于未删减的输出通道上，不随 expanded-channel mask 删除。单看 sn1 group 为零也不能跳过整条 MLP branch：必须证明 BN1 零保持且 sn2 的状态/输出仍为零。

当前 checkpoint 的 16/32-channel exact-zero group 仍为 0，所以上述合同只能用于训练，不接纳现成 skip speedup。

## 硬件映射

- sn1/sn2 时间复用已有 ATLIF engine；
- fc1/fc2 时间复用一个 Linear engine；
- BN fold/threshold bridge 与 residual commit 做外围小模块；
- 不复制第二套 MAC，也不需要总体 scheduler。

当前等级：`PASS_FULL_FFN_SUBGRAPH_SCOPE_AND_ACCOUNTED_COMPUTE_PARTITION`。BN/residual cycles、pruning speedup、完整 FFN RTL、system speedup 和 PPA 均未接纳。
