# 单遍动态BN：完整强对照已经补入

2026-09-12。借用[BNFF，MLSys2019](https://proceedings.mlsys.org/paper_files/paper/2019/file/9db64c20dee0011899dfdf200e61ef35-Paper.pdf)和[FlexAcc，ISCAS2025 Fig.4](https://jiemingyin.github.io/docs/ISCAS2025_FlexAcc.pdf)的同遍sum/sumsq，执行E[x²]−μ²。每域192000位置×96通道，所有code0仍计入分母。两条sum支路和两条sumsq支路共48RF；paired256叶树占7680B，同96RF/128KiB状态/128KiB系数/SR64/SW64/CR256。借入代数和普通默认传播是公共底座，**X=0**。

| 条件 | 同格式原中心化两遍 | 新单遍、带code0/default | 服务减少 |
|---|---:|---:|---:|
| ordinary/ready | 107,236,936 | 81,977,173 | 23.555% |
| lifting/ready | 106,714,840 | 81,619,573 | 23.516% |
| ordinary/固定背压 | 123,442,202 | 92,515,226 | 25.054% |

单位是实际载荷CPU端口模型服务槽。新单遍dense对照也执行了：ready147,783,600、压力166,266,954槽。三个配置中，tagged与同公式dense的18,432,000个输出和288统计值全部逐位一致。带tag外部读取约62.29–62.74MB，输出仍73.728MB。

**浮点函数改变，不能直接继承AEE。** 相对旧中心化树有约831–852万输出位差；对原CUDA捕获最大绝对差分别5.72e−6/3.81e−6，RMS约1e−7。方差最小0.01934/0.01881，本捕获未出现负方差；并非任意输入数值稳定性证明。

[实际diverse10三臂](aee_check/README.md)均用相同原学生/数据/指标，旧对照和候选逐帧复测一致；两种部署函数的CPU与GPU包装分别对完整域Engine输出逐位核过。

| 学生 | 原CUDA BN | 中心化部署函数 | 单遍部署函数 | 单遍对原CUDA |
|---|---:|---:|---:|---:|
| ordinary | 1.16118708 | 1.17212272 | 1.16684178 | +0.00565470 |
| lifting | 1.18658552 | 1.18804899 | 1.18783485 | +0.00124934 |

按用户[新精度规则](../../ACCURACY_POLICY.md)，六臂均优于同diverse10的原SDformerFlow本地复现NB0（1.454602861），都可继续考虑；+0.005不再限制ordinary单遍部署。中心化和单遍共享seed+3Newton与显式MUL/ADD，但均不同于原CUDA函数；单遍对中心化AEE反而更好，不能将所有变化归给E[x²]−μ²。组合其他改动后重新测AEE，不自动继承单项结果。无新增训练或valid825。

[代码与完整计数](run.py)、[ready](results_ready.json)、[固定背压](results_stress.json)、[独立源审阅](../../review/04_onepass_review.md)。参数都由真实输入算出，没有读取捕获的mean/var来算服务。

已进一步在[同一Engine接真实PED](integrated_consumer/README.md)，从统计到最后ADD与全部最终输出重新执行。物化→普通融合：ordinary215,914,706→139,882,706，lifting215,585,714→139,553,714，ordinary压力258,347,050→156,642,706；每臂全输出和统计0差。该35.2–39.4%改善删掉中间BN写/读，依然是普通融合，不是另一个X。两个阶段的倍率未相乘。

范围起点仍是完整K864原生投影捕获；原生生产者、压缩形成和更后续网络不在服务范围。无RTL加速比、PPA或整网结论。
