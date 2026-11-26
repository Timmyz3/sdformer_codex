# 光流损伤与普通评分的有限干预探针

本轮先问一个可证伪的问题：在完全相同的可执行候选块和细计算预算下，普通输入评分与真实光流损伤排序是否有差距？本 runner 不训练预测器、不声称光流新算法成立，也不把离线 GT 排序当作推理机制。它是 ROUND4 阅读后确定的探索，继续使用同一 backbone-seen 的开发 dev8；不能算独立确认。

**执行前冻结合同。** 沿用135930步 checkpoint、MLP_CMA176、FP32、完整T10、stage0和stage2八个MLP的stride3规则代表分支。唯一变化点是 `stage2/block0`：输入30×40、通道384，在3×4共12个完整9×9块中恢复fine。四行五列的其余tail始终coarse；评价仍包括全部有效GT及既有全部分区，不能把候选范围误写成评价范围。

每帧执行全coarse基准和12次单块恢复；每个恢复增加目标FC1和FC2各720条实际输入行。首帧的baseline和`fine_r0_c0`另做dual参考，先固定106次完整前向。读取本轮结果之前又固定下列五个k=2组合；组合全部真实前向，同帧相同mask可去重复并记录别名。因此最多146次完整前向，而不是相加单块损伤。

| 组合 | 选择依据 | 可部署性边界 |
|---|---|---|
| `k2_posthoc_singleton_task_gain` | 全GT的基准AEE减单块恢复AEE，取前2 | 使用GT及12次反事实，只是事后单块排序参考，不是联合最优oracle |
| `k2_amplitude` | 目标MLP之前连续输入的绝对值均值前2 | 不用GT；评分、读特征与排序成本仍须计入硬件 |
| `k2_pooling_residual` | 同输入相对逐3×3 cell均值的残差平方均值前2 | 不用GT；相同成本边界 |
| `k2_fixed_uniform` | 完整块展平索引3和8，即(0,3)、(2,0) | 固定对照 |
| `k2_offline_branch_mse` | 同u下真实fine/coarse目标MLP输出branch的MSE前2 | 同时消费两种生产者；离线强对照，不能称免费运行时分数 |

所有排序预先固定为分数降序、坐标字典序打破同分；不按结果挑帧、挑方向或改k。branch MSE通过已有单块恢复捕获，不额外增加完整前向。恢复fine保留原逐点残差，所有其他七个目标MLP仍全coarse。不能把当前层之后的流或GT反向用于stage0选择。

每帧另封存不含GT的七维输入特征向量：幅值绝对值均值、RMS、3×3代表池化残差MSE、空间邻居绝对差、时间邻居绝对差、真实PSN1活动率和空间相邻PSN1 Hamming差。前五项来自连续signed输入，不能统称脉冲。最后两项额外执行12次完整块PSN1，每帧名义37,324,800个时间MAC，3,732,480个PSN输出元素；这不是免费预决策信息。具体评分reduction、特征读取和排序周期尚未测量。它们是机制对照，不是官方ADAPT复现。

质量与执行证据保留以下内容：

- 新baseline预测及各分区指标与ROUND4的`coarse_stage02`逐值相等；每个干预的目标层输入SHA与baseline完全相同。
- 每次真实执行全12层FC hooks，固定八层的fine/coarse producer、PSN/BN输入元素和tail行数；首帧两组actual/dual全部flow和branch逐值相同。
- 首帧baseline所有78个BatchNorm模块每次输入的shape、stride、dtype、device、训练态、running-stat设置和running_mean存在性。这是采样图的逻辑服务审计，不是物理周期或访存计数。
- 每个arm全分区、逐帧/序列/全局指标；所有预测NPZ；基准predecision输入及七维无GT向量NPZ；12块fine/coarse输出branch NPZ；各文件SHA；最终原参数/BN和模块恢复检查。

现有原生图到全fine分块图有数值差异，本探针不宣称消除了它。它只固定同一个stage02分块图，验证单块动作、同u和actual/dual的一致性。所有质量仍为小规模开发数据；周期级模型、量化同图质量和RTL/DC未完成，不能由省FC行数推导净收益。

[CPU测试](test_tile_interventions_cpu.py)直接调用新生产`forward_probe`，用12个小型原生MLP验证whole-tile动作、singleton与k2的实际行数、same-u、actual/dual、BN输入hooks、评分公式与PSN额外成本、固定同分规则、异常恢复及安全导入。无完整光流模型、远端或GPU调用。receipt见[runner_cpu_results.json](runner_cpu_results.json)，来源绑定见[freeze_v1.json](freeze_v1.json)。

执行接口沿用scope_extension的父合同、identity bank与ROUND3 receipt，再新增ROUND4 receipt。目录部署时与既有`structured_merge`平级。默认只做preflight，执行需要`--execute`并独占旧共享GPU锁；输出必须为新目录，墙钟上限30分钟。

```bash
/data/flow_a/envs/v2/bin/python -B run_tile_interventions.py \
  --parent-contract <原runtime_contract.json> \
  --parent-contract-sha256 564d70a02b65d00c1aec40eacf069babb38f600c5a553a5ceb2a263630f3c389 \
  --identity-bank <prepare_v3/mask_bank.json> \
  --identity-bank-sha256 2d831c7a3a77ffd4bc22efeb256a9d1b672f7ef8edf2300cac413c603d7fdfdf \
  --prior-screen-results <screen_v1/results.json> \
  --prior-screen-results-sha256 70208e3c0378bed2127d41e47745a373f1c02aad49b07da2a79af0ad9be1e9fb \
  --prior-scope-results <scope_extension_run_v1/results.json> \
  --prior-scope-results-sha256 e43b7e06b12ac0360a4b8799a89ee94f020e6c71b791534d98949b4280f133fb \
  --output-dir <新run目录> --preflight-output <新preflight.json>
```

实际部署由root启动。本实现者只完成本地CPU验证，不担任本轮独立期刊评审。
