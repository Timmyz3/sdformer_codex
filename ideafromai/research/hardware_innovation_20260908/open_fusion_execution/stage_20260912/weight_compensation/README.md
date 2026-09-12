# U权重补偿真实结果：免训符号布局收口，W4/W8已接真实AEE

2026-09-12。**本次符号主体＋低秩残差没有优先推进的免训交易；普通W8在更小系数存储下远低于rank16补偿的局部误差。** 这只否定本次冻结的免训参数布局，不否定ReverB/MiLo完整训练或后续权重学习。符号布局只有局部误差；随后普通W4/W8已完成真实网络diverse10，见下方。没有压缩执行周期、面积或功耗结果。

一次权重规则拟合，限定U32×96一个接口。原V96×32、偏置及原U/V两处signed24 RNE保留；两个学生各corner/interior 4×4×T10，共640个输入向量、61,440个参考输出。原函数对真实capture逐位相等；全部40个模型/窗口重放无signed48溢出、无sat24发生。CPU主过程0.547s，单线程，不当硬件性能。

| 固定配置 | 系数总B | 四窗PED NRMSE范围 | 每96输入向量的必要乘/符号项 | 累加/加减项 | dot后RNE24写值 |
|---|---:|---:|---|---:|---:|
| original32 | 12592 | 0.00000–0.00000 | 6144 个16×24乘 | 6112 | 128 |
| sign_residual_r0 | 6896 | 0.39587–0.53210 | 3072 个16×24乘 + 3072符号项 + 32尺度乘 | 6112 | 128 |
| sign_residual_r4 | 7924 | 0.34583–0.44785 | 3584 个16×24乘 + 3072符号项 + 32尺度乘 | 6620 | 132 |
| sign_residual_r8 | 8948 | 0.27585–0.38591 | 4096 个16×24乘 + 3072符号项 + 32尺度乘 | 7128 | 136 |
| sign_residual_r16 | 10996 | 0.19357–0.23050 | 5120 个16×24乘 + 3072符号项 + 32尺度乘 | 8144 | 144 |
| W4 | 8048 | 0.06025–0.08199 | 3072 个16×24乘 + 3072 个4×24乘 + 32尺度乘 | 6112 | 128 |
| W8 | 9584 | 0.00414–0.00497 | 3072 个16×24乘 + 3072 个8×24乘 + 32尺度乘 | 6112 | 128 |
| original_ordered24 | 9520 | 0.15904–0.22412 | 4608 个16×24乘 | 4584 | 120 |
| weight_svd24 | 9520 | 0.15504–0.22877 | 4608 个16×24乘 | 4584 | 120 |
| activation_whitened24 | 9520 | 0.08839–0.15275 | 4608 个16×24乘 | 4584 | 120 |

每行额外都有96次原bias加后sat24。总B包含U/V权重、32行尺度、补偿因子/指数、packed24偏置288B及16B模式/秩/原指数元数据；不是npz文件大小。sign-r>0还需额外r个24位潜变量（3r B），保持原U32潜变量96B。W4/W8的小位宽乘与16×24乘、scale16×宽累加器乘不能等价计价；没有把“去掉多位乘”写成已知周期节省。

**相对强对照。** rank4只比W4少124B（1.54%），NRMSE却约0.346–0.448 vs 0.060–0.082；rank8/16在存储与四窗误差上均被W4压过。W8总B比原PED系数总B少23.89%，NRMSE仅0.00414–0.00497；这不是AEE预测。三个保存的R24（原序、weight-SVD、activation-whitened）也全部实际重放，未重新校准。它们都使用原16/15指数，避免用弱纯截断对照。

**数值边界。** 符号s=sign(U)，αq16=RNE(mean|Uq16|)；r>0将权重残差SVD后做确定性潜变量幅度平衡，A为signed16/e16，所有实测B也是signed16/e16，零系数裁剪。前向为：`sign_dot -> α精确整数乘`；`Bq*x -> 新RNE/sat24`；`Aq*z + α*sign_dot -> 原U RNE/sat24(e16)`；`Vq*u -> 原V RNE/sat24(e15)`；原bias/sat24。新B边界明确是有损新函数。残差rank16仍剩原sign残差约27.4%/27.6%的平方谱能量，部署U权重相对误差约31%，局部PED误差高并非本次饱和造成。更改此参数化/训练需另立试验，不能回写当前结果。

**GPU小包与后续执行。** 普通W8与W4均已进入阶段算法任务；符号族仅保留probe-only参数。`lowbit_ped_adapter.py`支持original32/W8/W4，用`code*row_scale`离线精确展开Uq16，所以现有helper能原样执行两处RNE，V和bias不变。本目录8个展开fixture共122,880输出逐位一致；算法任务随后完成actual-helper的12个fixture、184,320个输出0差。

在各自原R32/CUDA父网络上，只换U：ordinary W8/W4的diverse10 AEE为 **1.177205/1.172908**，lifting为 **1.188024/1.190508**；四臂均优于同集NB0 **1.454603**，排除旧校准帧后的9帧也均过门。[实际逐帧与参数](../algorithm/weight_controls/aee/run.json)。这些函数未叠R24/onepass，没有新valid825；GPU展开是数值评价，不是压缩硬件执行。局部NRMSE次序不能代替真实AEE。

包入口：[gpu_package_manifest.json](gpu_package_manifest.json)、[ordinary参数](ordinary_lowbit_gpu_parameters.npz)、[lifting参数](lifting_raw_lowbit_gpu_parameters.npz)、[adapter](lowbit_ped_adapter.py)、[fixture](replay_fixture_probe_only.npz)。所有配置/误差/最大累加值/饱和计数见[results.json](results.json)和[results.csv](results.csv)；参数小包文件包含测试材料和展开矩阵，不能用它的zip大小当硬件存储。

复现：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.12 probe.py`，然后同样运行`make_package.py`。计划先于结果保存在[PLAN.md](PLAN.md)。ReverB/低秩补偿是先验A，本次不是完整HQQ、作者训练或原工件复现；目前没有证实超越普通W4/W8控制的新X。
