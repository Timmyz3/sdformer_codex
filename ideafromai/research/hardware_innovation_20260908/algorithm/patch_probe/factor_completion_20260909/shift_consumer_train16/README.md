# DeepShift-Q 普通 V 消费者控制

已完成两名 `demand_train16` 学生各 FP／V 移位 QAT 的四轴控制。本地 CPU 共约16.78秒；真实 train16／valid4，各帧64个原生P4、C96/T10，没有 GPU、网络 AEE 或硬件周期结果。

迁移来源是 **CVPR Workshops 2021（MAI）** 的 [DeepShift](https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Elhoushi_DeepShift_Towards_Multiplication-Less_Neural_Networks_CVPRW_2021_paper.html)。已逐项核作者 [modules_q.py](https://raw.githubusercontent.com/mostafaelhoushi/DeepShift/master/pytorch/deepshift/modules_q.py)、[ste.py](https://raw.githubusercontent.com/mostafaelhoushi/DeepShift/master/pytorch/deepshift/ste.py)、[utils.py](https://raw.githubusercontent.com/mostafaelhoushi/DeepShift/master/pytorch/deepshift/utils.py) 的量化及梯度定义。仓根未列 LICENSE、license API 返回404，因此脚本按数学独立实现小量化器，没有导入或复制作者模块。

作者默认 `weight_bits=5` 对应符号±与指数 **−15…0**。非零影子权重先投影到幅值 `[2^-15,1]`，确定性 `round(log(abs(V))/log(2))`，量化反向为恒等 STE；U 的梯度使用量化 V。结构零保持0。32种非零值可以编码为5 bit；结构零由固定连接布局排除，不能把额外零值也免费塞进5 bit。没有借此声称完整 SmartExchange 实现。

这轮只迁 V **权重**量化器。Z 保留现有连续 FP32 参考，没有启用作者层中的激活 Q16.16。固定点 Z、移位截断、Acc 位宽、BN/A 和实际复用加法器需另行闭合，不能由幂次权重直接声称完整链无乘法。

四轴均从相应同一 `demand_train16` 检查点出发，256步、batch8、Adam lr0.002、同seed910批次；这是等预算本地恢复，未声称复现作者 ImageNet 训练日程。仅训练 U/V，A/b/θ、精确空间 mask、连接、prefix `[2,3,7]`、gamma3 均冻结。损失为归一Y MSE＋0.125完整门平衡BCE＋0.125许可／回退门平衡BCE，请求λ=0。各轴在步骤0/64/128/192及结束仅用train16重校自己的Y统计。

| valid4，每轴983,040门 | 归一Y MSE | 完整学生门差 | 漏教师非零门 | 提前相对自身改门 | U请求／完整一次 |
|---|---:|---:|---:|---:|---:|
| shared_compact FP |0.07725|1.0620%|12.8264%|0.01953%|1.45043|
| shared_compact Q5 |0.13979|1.5205%|16.5022%|0.02055%|1.46271|
| hybrid FP |0.09020|1.1400%|14.8003%|0.01923%|1.43701|
| hybrid Q5 |0.14681|1.4761%|20.6831%|0.02045%|1.43630|

Q5 的 prefix＋tail V 项分别为 **16,459,056／14,691,096**，每项是幂次缩放加累加；对应 FP 控制为 **16,324,272／14,748,896** 个一般连续乘加。不能将不同算术和额外U、A、BN、状态服务直接折算成同一种操作或速度。两阶段 U 请求仍高于完整一次扫描。

固定结构下 V 系数 payload 为 shared **18,432→2,880 B**、hybrid **19,968→3,120 B**。共同裁掉全局不活跃潜变量后，hybrid 为 **19,072→2,980 B**。U 仍是 FP32：shared165,888 B；hybrid原331,776 B、共同裁全局死项后235,008 B。掩码、通用零位图及完整预留容量在 `result.json` 分列，没有把V压缩率当整个模型压缩率。

`*_deepshift_q5.npz` 的 `v` 已是最终幂次系数，保留旧 adapter 所需全部字段；另存 `v_shadow/v_shift/v_sign/v_nonzero`，可直接交原网络评价入口。四轴导出后的密集参考与本地模型均0差；Q5对首个真实P4的全部单项乘积，用独立 `ldexp` 计算也0差。未改变冻结空间mask，禁用连接没有非零泄漏。

当前结论：这次256步移位恢复**没有达到同预算FP的局部精度**。shared量化初始MSE0.13373，恢复后0.13979；hybrid初始0.15297，恢复后0.14681，但其漏激活率上升，不能只挑总门差改善。保留可复用的量化底座与负结果，不扫位宽、rank、学习率或新训练强度，也不外推整类移位网络失败。

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 ../../joint_completion_20260909/.venv_train312/bin/python train_shift_consumer.py
```
