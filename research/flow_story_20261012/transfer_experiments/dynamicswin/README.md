# DynamicSwin → 完整 T10 MLP fast/slow 独立适配模块

2026-10-12。状态：**官方源码已固定并逐文件验证；独立 CPU 模块与 24 项检查通过。未训练、未跑 teacher、未接入完整光流网络，未测 AEE、GPU、周期或面积。** 本目录以外没有写文件或修改旧源码。

## 来源闭合

[官方 DynamicViT/DynamicSwin](https://github.com/raoyongming/DynamicViT) 固定为 `1322e626d1e9eca18cc150569908fb4af29e15f7`，原文为 [arXiv:2207.01580v2](https://arxiv.org/pdf/2207.01580v2)。`upstream/` 是独立 sparse checkout，保留官方 MIT LICENSE；未执行其训练入口或加载公开 checkpoint。

[source_provenance.json](source_provenance.json) 记录 README、LICENSE、`models/dyswin.py`、`losses.py`、`main.py`、`utils.py` 的 SHA256，并逐项比较本地文件与该 Git commit object 的 bytes。`dyswin.py` SHA256 为 `e3839ecca226966b22ccec3112c9b96918ee998ac0d1fd2074155bfac1f2f793`。固定源码的 `fastMlp` 位于 L34–48，`PredictorLG` 位于 L231–276；网页分支渲染的行号不作为本地定位依据。

本模块借用官方 local/global predictor 的拓扑与 fastMlp 的 LN＋Linear(C,C)，CPU 检查从原文件 AST 仅加载这两个类作为 reference，权重逐项复制后核对数值。没有导入官方完整 Swin；原生 slow MLP 则由已有光流 [CPU probe](../../../flow_story_20261011/preflight/cpu_numerics/native_mlp_probe.py) 工厂实例化，使用合成权重。

**本线改动：** 完整 T10 共享选择、T 均值形成评分视图、有效点均值形成 9×9 tile 分数、在原生 frozen-BN MLP 与 ANN fast 分支之间选择。它是机制适配，不是官方 DynamicSwin 复现。原有 ANN fast 分支没有改成 identity，也没有把连续值 Linear 当成 spike AC。[来源与迁移计划](../../idea_and_protocol/ANN_TRANSFER_SOURCES.md)

## 接口与语义

[adapter.py](adapter.py) 接收 `u[T=10,B=1,H,W,C]`，其中 u 必须由调用方完整执行 `x+Attention(x)` 得到。`slow_branch` 是返回 MLP 分支的 callable；可直接传原生 `MS_Spiking_Mlp`，不含外层残差。返回 `Result.branch` 也只含分支，`Result.add_residual(u)` 仅为独立测试方便。

```python
adapter = DynamicSwinMLPAdapter(native_mlp, Config(C, 4*C, tile=9))
adapter.train()  # 参数可求导，slow 分支的 BN running statistics 保持冻结
result = adapter(u, execution="dense_train")
y = u + result.branch

adapter.eval()
with torch.inference_mode():
    result = adapter(u, execution="sparse_eval")
    y = u + result.branch
```

这**不是可直接替换旧 `.mlp` 的 drop-in wrapper**：原接口需要 Tensor，这里返回带 gate、评分和 trace 的 Result；后续完整集成需明确提取 branch、训练损失与共享 scorer 的位置。传 `nn.Module` 时 slow 参数自动注册；传普通 callable 时其参数不在此模块内注册，BN/时空独立性也无法自动遍历，调用方必须拥有并验证这个合同。原生类路径另检查 FC 维度、BN 变体与 stateless complete-T10 PSN。

| 模式 | 实际执行 | 证据含义 |
|---|---|---|
| `dense_train` | 两条完整分支都执行；tile hard Gumbel 的 straight-through 梯度连接 predictor | 可求导的恢复训练骨架；没有训练省算主张，也不是原官方跨 block 双 stream 的完整训练图。 |
| `dense_reference` | 两条完整分支执行，再按指定 mask 混合 | 检查稀疏执行应实现的数学函数；不是稀疏硬件周期证据。 |
| `sparse_eval` | eval＋no_grad；混合时按有效 tile 只调用选中分支，T 与 C 完整；全 slow/fast 用原尺寸一次调用 | 证明 branch call / FC row 取消，尚无高效 GPU kernel 或硬件 cycle claim。 |

全 slow 输出 `u+native_mlp(u)`；全 fast 输出 `u+Linear(LN(u))`，**全 fast 仍有实值矩阵计算**。attention 与 decoder 都在模块外，不会被这里删除。`tile=1` 提供 pointT 对照；`tile=9` 使用有效点裁切，10×11 示例的四块为 81、18、9、2 点，没有 padding 生产者。

评分使用全部有效空间点的 T 平均视图，随后按官方 local/global 结构计算。global context 要等完整当前图就绪，不声称免费流式判定；T 聚合只用于 scorer，不修改 slow/fast 的 T10 输入。tile logits 是两类逐点 log probability 的有效点均值，再交给 Gumbel 或 eval 排序，不声称仍是已归一化概率。

eval 按 class-0 分数稳定降序，保留 `floor(tile_count*keep_ratio)`，并以 raster 顺序处理相同分数。这是**tile 数比例，不是每帧 PE 硬预算，也不保证同有效点比例**。`ratio_loss` 显式提供 tile-count 和 valid-point 两种权重；默认 valid-point 是本线适配，使用时应封存配置，并报告与 eval 离散选择的实际比例差异。未包含质量风险新监督。

current-stat BN 被拒绝。`.train()` 会冻结 slow 中 BN 的 running statistics，但不会冻结可训练 affine/FC/PSN 参数；这属于 D_fpBN 恢复图，不证明其精度等同论文原 BN。统计校准、共同 P→D_fpBN→D_q→G_q 的 AEE 账本仍由后续实验完成。

## 已执行 CPU 检查

运行环境为 `/opt/anaconda3/envs/pytorch310_cpu/bin/python`、Torch `2.7.1+cpu`、2 个 CPU threads；`CUDA_VISIBLE_DEVICES` 置空，结果确认 CUDA 未初始化。只做前向和一次合成 loss 的 backward，**optimizer steps=0**。

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B check_cpu.py
```

[cpu_results.json](cpu_results.json) 记录 24 项通过：

- C8/C96 的 predictor 与 fast 分支拓扑/数值对官方 AST reference 一致，最大误差 0。
- 原生 MLP 的 C8 FP64 tile9、C96 FP32 tile9、C8 FP64 pointT，各测全 slow、全 fast、棋盘和仅最后尾块，共 12 项；sparse 与 dense-reference 最大误差均为 0，全 slow 与原生完整调用逐 bit 一致。
- 三组未训练 predictor eval 检查精确保留 tile 数；分数尚无任务意义。
- 一项 ST Gumbel backward：predictor、fast、native slow 三组梯度均有限且非零，输入有梯度，BN running mean/variance/count 不变；没有声称每个参数梯度均非零。
- 六项拒绝测试：启用梯度的 sparse eval、错误 T、错误 B、错误 mask 形状、native/config 维度不符、current-stat BN。

forward hooks 独立核查实际 FC1/FC2 行数与 fast Linear 行数；例如 10×11 的最后 1×2 尾块单独走 slow 时，两层 FC 各执行 20 行，其余 108 个空间点走 fast。slow/fast 合计覆盖全部真实位置，不存在只写 mask 却仍执行全部 slow 的假取消。已执行源文件、原生类及 factory 哈希均进入结果；未读训练 checkpoint。

## 成本输出与尚未完成的实验

[cost_and_execution_examples.json](cost_and_execution_examples.json) 给出每个测试的 tile 范围、选择、实际 branch calls/输入行数及成本接口。成本分开列 native PSN real scalar products、FC binary weighted accumulates、BN/阈值元素、fast real scalar products/LN、predictor 乘积、时间/全局/tile 归约、Gumbel/排序对象以及 dense 双分支混合。它们是形状推导的算术服务与元素数量，**不是完整 FLOP 总数，也不是周期、SRAM 事务或能耗**；未知访问和周期字段保留 null。forced mask 的 selector 成本为零只表示诊断跳过了 scorer，不能代表完整动态策略。

[config.json](config.json) 给第三 stage C384/hidden1536 的待接入配置，不表示该 stage 已完成整网集成。teacher checkpoint、BN SHA、训练步数、loss 权重和 block 名均未填写结果；实际 CPU 测试的尺寸与 dtype 单独写在结果中。

后续必须完成同任务 teacher、流/特征蒸馏、ratio loss、fast/predictor 联合优化、backbone 恢复与 EMA 配方，并给 dense 同等恢复预算。此模块没有照搬分类 KL，也没有把 ImageNet 的 120 epochs 冒充 DSEC 已验证配方。全网 wrapper、训练器、真实 mask、评分时机、有限 SRAM/队列、完整量化和周期/RTL/DC 仍未实现；本交付仅提供可审查的迁移算子与 CPU 接口证据。
