# SVS-only：v2 MLP隐藏通道的静态窄化

2026-10-12。已实现独立评分和真实窄化；[11项CPU检查](cpu_checks.json)通过。**这是按QP论文重建的SVS组件适配，不是作者官方移植，也不是完整QP-SNN复现。** 官方QP代码仍未确认；本模块没有ReScaW、量化训练、AEE、恢复训练或硬件加速结果。旧模型源码与权重均不修改。

## 公式与原生依赖

固定来源是[QP-SNN arXiv v2 §4.2，式11–13及附录A](https://arxiv.org/html/2502.05905v2#S4.SS2)，依据本地已核[来源卡](../../idea_and_protocol/SNN_TRANSFER_SOURCES.md)：

```
Sbar[b,c,h,w] = mean_t S[t,b,c,h,w]
SVS[c] = mean_b sum_i 1(sigma_i(Sbar[b,c,:,:]) > 1e-6)
```

先对T求均值，再逐样本、逐通道对H×W矩阵做SVD，以**绝对阈值**严格大于1e−6计有效秩，最后mini-batch均值；不是奇异值求和、相对最大奇异值阈值、发放率或先逐时刻秩再平均。跨mini-batch按样本数加权，避免最后不足一批的成员被额外加权。实现采用CPU float64平均/SVD，这是本次声明的数值实现选择，论文dtype未在本地来源卡中固定。若大量同分，报告分布及截断边界并按原通道ID升序破同分，不事后改公式。

评分张量取原生[`MS_Spiking_Mlp.forward`](../../../../SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py:164)的`sn2`输出，布局`[T,B,H,W,C_hidden]`，在`drop2/fc2`之前，完整T10；接口也支持明确声明`TBCHW`。要求二值脉冲，不接受连续u、膜电位或FC2输出。保留提供张量的全部空间点，不使用GT、事件或keep-mask筛矩阵。这里把论文kernel评分迁到MLP hidden channel，属于组件适配。

真实闭包为`FC1输出行/可选bias → BN1 affine/running_mean/running_var → PSN2输入位置 → FC2输入列`。BN计数是标量，保留；FC2输出/bias、BN2、外部C和残差接口保持。原生[`PSN`](../../../../SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_submodules.py:183)对展平后的每个位置用共享`[T,T]`权重与`[T,1]`偏置，所以PSN1/2参数与T不裁，PSN2执行的hidden位置数会减少。若换成有通道参数神经元、GN/IN/LN/BNTT或其他MLP结构，当前工具明确拒绝，不擅自猜闭包。

**静态通道剪枝与空间旁路不同：** BN逐通道归约，删除其他通道不改变保留通道的统计域；因此工具能处理tracked或current-stat BN，并有CPU闭包案例。这个事实不消除窄网下游分布／AEE变化，也不能推翻空间tile执行必须处理跨空间BN依赖的结论。最终共同部署仍单列P→D与重新校准代价。[BN语义](../../../../SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py:101)

## 使用入口

[svs_mlp.py](svs_mlp.py)不自动导入完整v2或加载checkpoint；由已核source/checkpoint/data的调用方提供原生module。

```python
from svs_mlp import (SVSAccumulator, magnitude_score, select_hidden_channels,
                     narrow_mlp_deepcopy, narrow_model_deepcopy)

# 在固定D图完整前向时，将各MLP自己的sn2 hook输出与该批target IDs绑定。
acc = SVSAccumulator(allowed_ids=fixed_train_ids, max_samples=40)
acc.update(sn2_output.detach().cpu(), sample_ids, layout="TBHWC")
score_artifact = acc.result()  # 未到40也显式报实际样本数；不允许重复/越界ID。
selected = select_hidden_channels(score_artifact["scores"], keep_width=192,
                                  pe_group_width=16)

# 必须先移除采集hook；旧module保留，新module是真窄化的deepcopy。
pruned_mlp, report = narrow_mlp_deepcopy(native_mlp,
    selected["keep_indices"], pe_group_width=16)

# 同hidden宽度的简单控制：FC1每输出行mean(abs(weight))，不含BN folding。
mag_selection = select_hidden_channels(magnitude_score(native_mlp), 192, 16)
mag_mlp, mag_report = narrow_mlp_deepcopy(native_mlp,
    mag_selection["keep_indices"], pe_group_width=16)

# 若应用到整网，显式列实际named_modules()路径；拒绝共享MLP/外部参数别名。
new_model, network_report = narrow_model_deepcopy(
    original_model, {actual_mlp_path: selected["keep_indices"]}, pe_group_width=16)
```

每层独立采样和评分；不要将不同hidden维度或geometry的输出混入一个accumulator。ID由调用方与真实batch绑定，工具无法从张量反推出其身份；结果还须附checkpoint、source、data、BN、几何和target-list SHA。评分采集完整执行所有分支，记录CPU SVD、feature拷贝和存储成本；它是离线静态选择成本，不在每次推理重做。

PE组宽可取任意正整数（例如1/8/16/32）；**keep_width必须精确整除组宽**，工具不默默四舍五入。选择高分通道后按原ID排序，并把行/列实际compact到新连续权重张量；原通道索引可以不相邻，因为这是离线dense重排，推理无需scatter回hidden原地址。SVS与magnitude必须逐层同保留宽度／组宽、同精度、同恢复与校准机会。参数减少不自动等于周期同比减少；FC/PSN2、尾组、共享权重和内存仍需在同一引擎上重算。

可对已有**二值sn2 NPY**独立评分（仅离线诊断，不伪造数据来源）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /opt/anaconda3/envs/pytorch310_cpu/bin/python svs_mlp.py \
  --spikes /path/to/sn2.npy --layout TBHWC \
  --keep-width 192 --pe-group-width 16 --channel-chunk 32 \
  --output /path/to/new_svs_score.json
```

工具不会加载旧optimizer。新参数对象和形状要求重建optimizer；保存新权重时必须附各MLP的原始保留indices／hidden宽度和完整方法合同。加载时先按该计划构建窄图，再`strict=True`加载；不能把窄权重`strict=False`塞进旧宽模型，也不能把旧checkpoint静默变成新baseline续训。

## 已通过的CPU证据与边界

运行`/opt/anaconda3/envs/pytorch310_cpu/bin/python test_svs_mlp.py`；torch2.7.1+cpu、1线程，CUDA未初始化。测试使用真实本地`MS_Spiking_Mlp`、SpikingJelly BN/Linear及PSN；三个源码SHA、脚本SHA和实际测量在[cpu_checks.json](cpu_checks.json)。

- 合成二值图验证时间先均值、有效秩、mini-batch均值、不同chunk/layout一致；非整批样本合并不偏权，重复/越界ID、非二值/NaN/缺T拒绝。
- 原生C8/hidden16/T10全保留，输出最大误差0，state一致但storage独立，原图不变。
- 保留8个非连续hidden通道并真实compact：FC1输出/BN1/FC2输入都变8，参数**524→380，减少144**；两PSN参数/T、输出C8和外残差保持。实际FC hook观察到宽度8，非乘零假剪枝。
- 窄图与“原宽图仅将被删hidden spikes置零”的独立参考最大误差0；**与未剪原图输出不同，最大差0.943768**，这是预期的剪枝效果，不声称删后精度等价。current-stat静态channel闭包也通过。
- magnitude同宽控制、backward有限、窄图严格重载、错误索引/通道参数神经元/归一化/共享路径及未移除hook拒绝均通过。

以上只证明评分定义与结构改造可运行，未使用训练checkpoint、DSEC数据、GPU或精度结果；小模型参数数目不是v2整网节约。全网贡献、精度和周期必须后测。

## 恢复与BN重校准待执行合同

[RECOVERY_CONTRACT.json](RECOVERY_CONTRACT.json)是提案，关键checkpoint/data/D统计/逐层宽度字段仍null，不可当已就绪运行单。先在同一个来源固定的D_fpBN上采样，复用既有40个gate-fit名单；先2样本测CPU评分成本，再决定是否扩40，不无界扫6000。完整T10、full9/B1、不增广，记录同分程度与排名稳定性。

同宽SVS、magnitude和未剪dense分别给匹配的初始500步恢复（AdamW1e−5/WD0.01，固定统计BN、同crop9/FP32/B1、4573 gate-fit同顺序与增广；这是本线屏测配方，非QP原论文全流程）。500步未恢复不判算法失败；必要延长、搜索和额外成本在已有120GPUh内重新分配。每次`model.train()`后明确把BN恢复eval，不能意外重新采统计。

恢复后在各自完整窄图／dense图上按固定176→703池重采full9 BN，规则与额度相同；cal1仅按已声明条件。完整报剪枝前后、恢复前后和重校准前后的差异，不能用valid选择统计。最终固定策略才评全部2152和全部有效GT，另报B/L/U。若将来做完整QP，需另实现ReScaW并在共同量化域比较；此SVS-only分支不冒领量化收益，也不构成“按区域剪共享权重”的A机制。
