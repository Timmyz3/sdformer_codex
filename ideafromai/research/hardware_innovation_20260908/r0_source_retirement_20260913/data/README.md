# 独立训练校准：完整 Cin4 删除与配对质量

本分支已完成**独立训练帧336规则窗口捕获、三个固定25%控制、根代理第四布局的六臂diverse10评估**。全部掩码低于历史更严NB0门1.45460286107；完整费用控制优于完整Cin幅值，但仍比普通O8×Cin4块幅值误差大。第四融合规则实际没有删除任何完整Cin4组，AEE也未胜普通块幅值，未形成新的获胜机制。

## 固定对象与数据身份

沿用 matched-dense stage320 的真实 `...resblocks.0.conv2.0`：C96/N96/T10、3×3 stride1 pad1、r0原生入口 `[10,1,96,240,320]`。sn2实际是 `{0,θ}`，本次θ=1，在完整73,728,000元素上验证幅值误差为0；卷积无bias。Wq=`round_even(θ Wfp32 65536)`，signed16范围[-13925,16243]，与前轮W及旧profile逐数组相同。没有改变权重量化格式。

原 `algorithm/samples.json` 实际没有train键。因此复制真实 [train_split_seq.csv](train_split_seq.csv)，按原CSV顺序选择首个不属于diverse10任一序列的训练帧 **thun_00_a_0002.npy**，固定规则在观察活动前写入 [calibration_selection.json](calibration_selection.json)。它与diverse10的frame和sequence均不重叠。检查点、模型代码、之前实验目录只读。

最初八个预定窗口只有21个spike；三控制初稿已生成但尚未AEE或交新RTL。根代理在评价前明确修改一次覆盖方案：同帧增加规则栅格 `y=0,16,...,224,238`、`x=0,16,...,304,318`，共336个连续4×4源块，全部纳入，不按活动筛位置。正式 [calibration_grid_t10.npz](calibration_grid_t10.npz) 有183324个spike，24个Cin4贡献均非零，避免了原8窗口大量平分退化。旧8窗口和初稿保存在 [calibration_train_t10.npz](calibration_train_t10.npz)、`superseded_8tile/`，不作当前机制结论。修正规则见 [PLAN.md](PLAN.md)。

校准栅格跨整个视场，但只包含选定336个2×2输出块，**不是整层回放/周期**。完整源层有2772302个spike；校准不替代全帧统计。未换帧、以AEE选mask、扫描剪枝率或训练。

## 可执行数据合同

| 文件 | 用途与主要字段 |
|---|---|
| [calibration_grid_t10.npz](calibration_grid_t10.npz) | 训练校准，source_bits `[336,10,96,4,4]`、output/golden `[336,10,96,2,2]`、Wq16、真实FP32 W/output、θ、origin、valid边界 |
| [dense_q16.npz](dense_q16.npz) | 前轮8个真实评价tile、未剪Wq16、完整integer golden |
| [block_magnitude25.npz](block_magnitude25.npz) | 72个普通O8×Cin4×9tap块删除；live `[12,24]` |
| [cin_magnitude25.npz](cin_magnitude25.npz) | 完整Cin4幅值删除6/24组，等于72块 |
| [cin_fullcost25.npz](cin_fullcost25.npz) | 完整Cin4输出贡献SSE/完整执行收益删除6/24组 |
| [第四融合NPZ](../selector/mixed_retirement25_q16.npz) | 根代理预定混合动作的最终72块mask，供同一次AEE与RTL |
| [calibration_scores.npz](calibration_scores.npz) | 24组完整整数贡献、SSE、完整周期收益、评分和所有控制mask |

各控制NPZ主 `source_bits` 仍为旧真实评价8tile `[8,10,96,4,4]`，frame=`zurich_city_09_a_0001.npy`；`weight_q16` 为实际masked系数、`golden_accum` 为对应完整K864/N96/T10线性整数输出 `[8,10,96,2,2]`。原点和图内有效位保留，边界在fixture前补0且硬件可独立检验。`weight_fp32` 是未剪原参考，不应拿它代替masked Wq。训练源单独在 `calibration_source_bits`；不得把336训练窗口当作硬件评价8块。

[prepare_controls.py](prepare_controls.py) 验证24组整数输出贡献相加等于完整calibration golden；每臂在旧8tile重新算完整整数gold。六臂评价入口还验证 `maskedWq=baseWq×展开live`，防止质量和RTL用不同函数。

## 三个强控制及完整费用

全部固定删除72/288个物理块，每块O8×Cin4×完整3×3。实际非零系数可能因原Wq既有零而相差几个，因此是**同物理块预算**，不是逐系数精确同nnz。

| 规则 | 完整删除的Cin4索引 | 剩余Wq非零 | 校准输出SSE |
|---|---|---:|---:|
| 普通块幅值 | 无 | 62202 | 2.21880025553634e14 |
| 完整Cin幅值 | 1,3,4,10,13,20 | 62204 | 3.03142109199977e14 |
| 完整Cin费用 | 3,8,18,20,22,23 | 62202 | 2.15392969681867e14 |

幅值都取平方范数，普通块选72个最小块，完整Cin选6列最小总范数。完整费用控制的分子是该Cin4在336训练窗口、全T10/P4/O96输出上的**完整卷积贡献**平方和；不是单权重平方或只计spike数。不同Cin4联合删除的相关项不在这个独立排序控制中优化，根代理第四臂另用边际误差。

费用固定采用前轮最快mode3消费者枚举器，不在看过新C4硬件后重选mask。令F=336、G为一个Cin4剩余live O8组数。对每tile、每通道pair、每图内原生位置，a/b是两个10bit时间字，P是几何合法空间消费者数：

`E_cg = Σ P·[3·I(a|b≠0)+I(a≠0)+I(b≠0)+I(a&b≠0)+3·popcount(a|b)]`。

每live Cin4固定source/循环部分160F拍，全死时只需2F跳过；该列删除的完整收益为 **158F+12E_cg**。它包含源读、图外置零状态、W读、两源和、NEXT_TIME、psum读写、目的枚举/推进。固定清零、drain和不变装入部分不放进可节省分母。每列真实source读取消数为 `4·Σ validXY`，不把全部160F误称实际source bank读取。

[controls_manifest.json](controls_manifest.json) 保存24列source/W/sum/psum/control分解，代码断言这些分项之和等于完整周期收益。该模型是选择费用，实际周期仍由RTL测量；根代理另以116条旧mode3收据核对模型。完整Cin控制预期能关掉25%源读取，普通分散块掩码没有完整死列，不能关掉整Cin4源词。

第四臂允许删除单O8块或一次关闭某Cin4全部剩余块，在总72块预算内按新增校准误差/精确边际周期收益贪心。最终选择72个单块、完整死列0；保留 [selection.json](../selector/selection.json) 和原规则，不修改比例强迫出现源删除。普通块剪枝、完整输入通道剪枝及bitmap/priority都属先验A，此结果不支持“已实现新退休机制”的说法。

## 六臂实际diverse10

[evaluate_controls.py](evaluate_controls.py) 在同一个加载模型上依次跑原FP32、dense Q16和四份固定mask，源自原十帧/GT/mask，使用真实preds.2时间求和、bilinear480×640、align_corners=False、帧等权AEE。[diverse10.json](diverse10.json) complete=true，每臂10帧、516735个有效像素；[逐帧记录](diverse10_aee/)。

| 配对函数 | diverse10 AEE | 低于历史NB0 1.45460286107 |
|---|---:|---|
| 原FP32 parent | 1.157519653249069 | 是 |
| 无剪dense Q16 | 1.163296947981814 | 是 |
| 普通块幅值25% | 1.18599666968428 | 是 |
| 完整Cin幅值25% | 1.3115772136838026 | 是 |
| 完整Cin费用25% | 1.2603504201296971 | 是 |
| 混合动作最终72块 | 1.2010505064282313 | 是 |

完整费用比完整Cin幅值低0.051227 AEE，但高于普通块幅值0.074354；要结合实际源/控制/psum节省看质量与周期取舍，不能从此单称赢家。第四布局比普通块幅值高0.015054且没有完整死Cin4，原定混合目标在此点没有兑现源删除。所有布局仍过用户的NB0质量筛查，不用旧parent+0.005淘汰。

复用前轮同环境NB0 **1.462941239319642**，原PSN/SDSA最终头、78BN no-running、10帧516735有效像素；没有再次运行NB0。Python3.12.7、Torch2.7.1+cu128、CuPy13.6、NumPy1.26.4、3090、TF32设置均未变；本次parent/dense/普通块三项AEE也与前轮完全相同。历史更严1.45460286107仍保留，不借新NB0浮动放宽门。

这是相同Q16系数解码后的浮点/TF32消费者质量，不是全网整数bittrue；无valid825（门仍1.44535253468097）、训练或ASIC准入。当前“退休”只覆盖所测线性叶里的source读取及消费者服务，未证明上游非因果PSN生产或整层halo供数能一起省去。没有因输入静默便认为后继BN/PSN必定零输出。

## 复现与后续审阅

[run.sh](run.sh) 只写本目录，使用前轮已经安装的明确Python3.12解释器和只读数据镜像；第四mask由根代理的selector先生成，本脚本只读取。最初8窗口与语法修正失败记录保留，不进入正式结果。没有写前轮/生产目录、hash、训练或参数扫描。

新C4执行器独立审阅已经完成：[REVIEW_C4.md](REVIEW_C4.md)。54个fixture、648条命令的5832项独立计数/完整周期检查全相等，未发现具体功能bug。mode4相比拥有共同C4供数的mode5更慢，必须保留这个负结果；普通循环合并的收益不能全算作跨四源复用。最后授权的双父和合并接口另作兄弟版本验证，不改变本表掩码、质量或mode4结果。

最后接口的独立审阅也已完成：[REVIEW_PAIR_PARENT_MERGE.md](REVIEW_PAIR_PARENT_MERGE.md)。mode5/6的368条命令、1413120个输出通过；3680项独立计数核对全部相等，184条mode5的2392项对应计数与旧版相同。完整计入父和与合并费用后，dense核心391120→371056拍，完整Cin费用294316→277996拍；五臂相对强mode5少约4.9%–5.5%。这是固定整数线性叶的实测收益，未证明Fmax/PPA或整网加速，也不单凭普通父和共享称新X。全部授权数据、质量和两次硬件审阅已完成，本阶段不再开新实验。
