# BBS → 实际 W_res → 完整 CMVM 的固定三轴迁移

本轮完成了两种官方 BBS 权重变换、一个普通 5-bit 组量化控制，以及每轴 whole96／12 个 H8 的完整 da4ml 图。**ZPS和普通group5的完整825也已完成，AEE为1.216633／1.215144。普通group5在本次AEE、整图加减和活状态上更小；ZPS整图深度较短。这仍是成熟底座间的取舍，没有新增机制或硬件加速比结论。**

输入仍是实际 PED 连续残差张量的 X12，范围 `[-2048,2047]`；不是发放标志。参数继承 [projection_control_parameters.npz](../../projection_control_parameters.npz) 的相同 `input_scale=2^-6`、逐输出 `row_scale`、原 bias、stride2 几何与真实训练捕获。所有图实现各自新学生的 `Wq @ Xq`；不冒称与原 FP 网络等价。

## 完整继承及固定范围

[BBS MICRO 2024 原文](https://arxiv.org/html/2409.05227v1)的 §III-B、Algorithm 1、§III-C 提供两种变换与敏感通道原则。作者仓库四个文件原样保存在 [official/](official/)；[build_parameters.py](build_parameters.py) 实际调用 `roundAvg_fc`、`zeroPointShifting_fc`，未改作者函数。

固定 `C16`、剪 3 列、ZPS 常数 6 bit。按 `row_scale` 降序、原行号破同分，保留 32 行原 W8，其余 64 行变换。ZPS 完整搜索作者规定的 `[-32,31]` 常数域；这是算法内部步骤，不是本地参数扫描。没有全网 global pruning、重训或 BitVert 电路复现，也没有重排输出行。

普通轴给同一组划分和同一敏感行：先识别各组原整数的最小 signed 位宽，再 RNE 到 signed5 并还原 dyadic 组移位。它也利用冗余符号高列；不是仅清固定低三位。该控制未覆盖所有校准或训练型 PTQ。

三份可直接供原网络评价器读取的参数为 [round_average.npz](round_average.npz)、[zero_point_shift.npz](zero_point_shift.npz)、[uniform_group5.npz](uniform_group5.npz)。Wq 均存 INT16，不增加最终 W8 截断；实际范围分别为 `[-127,125]`、`[-127,127]`、`[-128,122]`。最终点积均需最大 signed25；输出仍按原 dyadic scale 和 bias 解码。

## 同一真实输入上的结果

局部误差用四个训练帧、每帧 64 个真实 anchor、完整 T10，共 2,560 个连续输入向量。下表绝对 RMSE 和无量纲 relative RMSE 均以捕获 FP 输入经原 C 的 NumPy FP64 点积为参考；**不是 AEE，也不是 CUDA FP32 的逐位误差**。

| 轴 | 绝对 RMSE | relative RMSE | whole 加减节点 | 12 个 H8 节点合计 | whole / 最大 H8 深度 |
|---|---:|---:|---:|---:|---:|
| 原 W8 | 0.0143310 | 0.00470087 | 9,527 | 11,320 | 12 / 14 |
| 官方 rounded averaging | 0.0631888 | 0.0207272 | 7,785 | 10,098 | 23 / 18 |
| 官方 zero-point shifting | 0.0574714 | 0.0188518 | 7,855 | 10,141 | 11 / 14 |
| 普通 group5 | 0.0729245 | 0.0239207 | 7,291 | 8,698 | 16 / 13 |

采用原 W8 独审已给足的普通输出依赖 DFS、两旁路寄存器、按最后使用释放、脏值驱逐才写 RF。一个图实例是单个 `(p,t)` 的 96 输入通道；不能再把状态乘 96。输入缓存共同为 1,152 bit，结果可背压暂停后流出，均不强制保存全部 96 输出。

| 轴 | whole 临时 RF 峰值 bit | H8 顺序执行最大 RF bit | whole 跨 H8 活值峰值 bit | whole 输入读／临时读／临时写 |
|---|---:|---:|---:|---:|
| 原 W8 | 27,058 | 2,997 | 27,016 | 2,739 / 10,329 / 5,309 |
| rounded averaging | 21,003 | 2,791 | 20,962 | 2,415 / 8,232 / 4,316 |
| zero-point shifting | 21,447 | 2,791 | 21,314 | 2,246 / 8,529 / 4,384 |
| 普通 group5 | 19,372 | 2,791 | 19,351 | 2,345 / 7,674 / 4,126 |

RA 的跨 H8 算术节点数反而由 4,600 增到 4,962，但同时活值下降；ZPS 为 2,273。不能把跨组节点总数等同同时端口压力。跨组节点的「位宽×已发射节点距离」下降也只是该合法遍历的寿命诊断，未转换成周期、功耗或布线。whole 与 H8 各自重新分解，节点差额并非纯跨 H8 CSE 消融。

## 已核验与未完成的强控制

[compile_bbs.py](compile_bbs.py) 使用与原 W8 相同的官方 `wmc/auto`、无延迟限制、官方内部分解搜索；固定每个矩阵先完整分解，再跨输出有符号移位 CSE，不是独立 CSD 乘法。39 个图均导出展开整数 DAG、官方 pipeline、DAIS、真实节点值域。每轴每组织 245,760 个真实输出全部与整数点积一致；另有逐节点系数证明、严格非对称 X12 位宽，以及官方 DAIS 的真实向量和边界核验。所有数值检查 0 差。结果见 [parameters_result.json](parameters_result.json) 和 [compilation_result.json](compilation_result.json)。

**BBS §IV-A/D 的显式组和强控制已补齐。** 六个 `ΣX_C16` 共用一次，再计算低位修正与常数项；RA/ZPS各完成十二个官方H8图和245,760个真实输出核验。RA NPZ保存 `group_low_bit_width` 与 `group_constant`，ZPS保存被选offset与shifted-pruned值，均保留精确还原。这些全是继承机制，不把它们算成X。

| 含六组和生成的完整标量费用 | RA显式 | ZPS显式 |
|---|---:|---:|
| 加减操作 | 9,038 | 9,393 |
| 总逻辑读写 | 16,817 | 17,472 |
| 相对直接十二H8的操作减少 | 10.50% | 7.38% |
| 相对直接十二H8的逻辑访问减少 | 10.70% | 7.63% |
| 临时RF峰值，另列S6的96bit | 2,791bit | 2,791bit |

两者仍比各自整图多操作和访问，以换取较小临时状态。组和生成、最终写入和后继读取已计；控制表、端口、背压与指令未转换为周期。见[RA显式对照](explicit_round_average/README.md)、[ZPS显式对照](explicit_zero_point_shift/README.md)。

理论编码 payload（32 行×8 bit＋64 行×5 bit）为 6,912 B；BBS 384 个组的 8-bit 元数据另为 384 B，合计 7,296 B，未计行选择和 scale。普通移位元数据更少，可选择相同 8-bit 槽作明确对齐控制。**静态 CMVM 图本身未在运行时读取该压缩权重包，不能将这一包大小直接当作图的供数流量。** BitVert 的 ΣX、常数乘、选择、通道重排／输出还原、bank 与实际端口仍需收费。

## 完整网络后继的实际精度

同一R32 U8/VQ5预览父模型、同一非anchor整BN分支删除、同一coarse-head评价函数。只改变PED连续投影；每轴完整825帧、48,152,523有效像素，没有额外恢复训练。

| 投影学生 | diverse10 AEE | valid825帧均AEE | valid825像素均AEE |
|---|---:|---:|---:|
| X12/W8 | 1.180678 | 1.215924 | 1.168447 |
| BBS RA | 1.199519 | 未运行 | 未运行 |
| BBS ZPS | 1.204119 | 1.216633 | 1.168105 |
| 普通group5 | 1.221151 | 1.215144 | 1.166673 |

10帧中的BBS精度优势未保持到完整825，不能据局部误差或10帧排序选择赢家。三份全量评价均有15,206,400,000个投影输入值，其中910个发生X12裁剪（约5.9843e−8）；共同父输入相同，裁剪数相同。输入是真实连续张量，θg约定未变。每帧与序列配对结果见[comparison_valid825.json](comparison_valid825.json)。这些都是新学生，不替换冻结ep34；软件墙钟不参与性能比较。

目前结论只到：位结构变化确实改变了完整CSE图，显式BBS分解也确实减少了H8开销，包含状态与深度的实质取舍；还没有排他X。有限端口、MAC同面积和完整PED消费者服务未闭合。

运行：生成参数用现有 `joint_completion_20260909/.venv_train312/bin/python build_parameters.py`；编译用 `psn/cmvm_20260909/.venv/bin/python compile_bbs.py`。脚本不修改官方源码、旧图或生产树，不训练、生成 RTL 或运行 EDA。
