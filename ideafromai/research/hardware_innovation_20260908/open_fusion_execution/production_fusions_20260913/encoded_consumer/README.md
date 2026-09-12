# 真实编码到连续消费者：35条已完成，相位接口另列

**此前免费Q8入口的收益没有兑现为快于原始路径的完整局部链。** 真实编码、U/V、输出全部付费后，fixed/affine码消费比同函数展开少0.73%–1.34%，但仍比raw慢0.70%–2.06%；完整D门预测的当前码内放置比自身展开慢2.41%–3.18%。这次已实际完成完整常量图、控制读取与signed48 spill，不再把它列作“未适配”。保留普通Q8底座，停止当前full-D放置的加速贡献句，未否定整个条件表示家族。

相位表示是后续**不同接口的一次实际尝试**：同一个ordinary/corner前缀，完整表＋RF内重建为2,057,099槽，相位z9按16bit真实写读为2,095,499槽，多38,400（+1.8667%）。SR/SW各多30,720B，CR/CW相同；其余增量来自floor/add与phase提取。旧full-D展开为2,074,612，新表/RF融合少17,513，但包含取消中间物化的共同优化，不能归给相位。数学普查见[PHASE_PROBE.md](PHASE_PROBE.md)，执行与独立审阅见[PHASE_EXECUTION.md](PHASE_EXECUTION.md)、[PHASE_REVIEW.md](PHASE_REVIEW.md)。

## A、B与待证的X

- **A：**普通仿射整数推理、P1保留潜变量、H8供数、完整da4ml CSE及有限寄存器spill。原始来源、借入与未复现边界沿用[来源表](../literature_owned/PRIMARY_SOURCES.md)；没有把这些通用手段改名申领创新。
- **B：**门已经产生，但连续PED仍需真实I24；旧试验免费给码，未付Dg编码、独立Ug及宽D还原。完整编码可能吃掉压缩收益，完整D图也可能增加控制与状态搬运。
- **候选X：**同生产者门能否提供一种比普通量化更好的连续消费接口。当前这一布局没有留下性能增量。更窄相位、改变真实编码边界和训练表示仍是不同接口，不能从本次负结果一并淘汰。
- **最强对照：**每种函数各有自己的展开路径，允许RF内直接q→I24，不强迫中间Q8写回；所有臂共同采用H8源驻留、相同P1/H48 V。raw是不同函数的成本参照，fixed/affine/full-D之间也不能宣称无损替代。

## 实测表

以下为**CPU实际载荷有限资源模型的总服务槽**，包含真实源、preview/sn2和当前局部消费者；不是RTL周期、整层或整网。两个父各两个已存halo，同一组16消费位置，另有ordinary/interior压力。35条不是35个独立网络样本。

| 父/窗口 | raw | fixed展开 / 码 | affine展开 / 码 | full-D展开 / 码 |
|---|---:|---:|---:|---:|
| ordinary/corner | 2,007,870 | 2,053,950 / 2,030,910 | 2,068,632 / 2,045,862 | 2,074,612 / 2,140,493 |
| ordinary/interior | 2,700,748 | 2,746,827 / 2,723,787 | 2,761,510 / 2,738,740 | 2,768,652 / 2,839,011 |
| lifting_raw/corner | 1,843,800 | 1,889,880 / 1,866,840 | 1,904,562 / 1,881,792 | 1,910,000 / 1,965,835 |
| lifting_raw/interior | 2,476,800 | 2,522,879 / 2,499,839 | 2,537,562 / 2,514,792 | 2,544,476 / 2,605,712 |
| ordinary/interior压力 | 2,974,170 | 3,035,450 / 2,994,906 | 3,035,802 / 3,013,690 | 3,045,050 / 3,128,282 |

ready下fixed码每窗省23,040槽，affine省22,770槽；full-D反增55,835–70,359槽。压力下分别省40,544、22,112，full-D反增83,232。完整分项见[summary.csv](summary.csv)、[同函数比较](comparisons.csv)、[阶段表](stages.csv)，原始五份`*_final*.json`包含实际回调时间线、端口、数值与原指令计数。

## 算法质量仍有资格，不能与硬件结论混写

这些同参数既有[diverse10结果](../../breadth_20260912/representation/README.md)均优于NB0的1.454603，本轮没有因为旧+0.005门停止任何候选，也没有重跑或借用父825。

| 既有函数 | ordinary AEE | lifting_raw AEE |
|---|---:|---:|
| fixed Q8 | 1.181770203 | 1.213944241 |
| affine Q8 | 1.186739905 | 1.180753363 |
| full-D Q8 | 1.168702647 | 1.203848848 |

ordinary门预测在这组小集较普通Q8准，但码内放置更慢；lifting则普通affine更准。因此当前没有门预测独占的质量/服务优势。相位两臂精确执行已有full-D函数，保留其精度归属，不能因此声称新增valid825。

## 代价为何转向

以ordinary/corner为例，raw后缀535,680槽；fixed码558,720、affine码573,672、full-D码668,303。编码起点是真实updated，不再是免费码，因此输入字节少并不自动快于raw。

| ordinary/corner完整链端口字节 | raw | fixed码 | affine码 | full-D展开 | full-D码 |
|---|---:|---:|---:|---:|---:|
| SR64 | 2,724,288 | 2,739,648 | 2,739,648 | 2,773,440 | 2,833,344 |
| SW64 | 1,043,360 | 1,058,720 | 1,058,720 | 1,089,440 | 1,141,664 |
| CR256 | 3,004,704 | 3,004,704 | 3,040,032 | 3,029,120 | 3,339,392 |
| CW256 | 165,568 | 165,568 | 167,488 | 166,048 | 173,472 |

full-D码的Dg编码预测和独立Ug均实际执行；D作用时间轴，U作用通道轴，不能用已有projection门的其他权重代替Ug。完整D图在原两读口上执行shift/add，ordinary/lifting分别有289/283个算术节点、10个输出累加；每halo48次H8图调用。固定无spill顺序曾需127/120总RF，这只是该顺序，实际用现有96RF并付spill后合法执行。

ordinary/lifting每halo的D图阶段为73,872/65,664槽，控制冷fill另1,122/1,068；ordinary压力为85,866槽。图控制在原系数SRAM中，每条取真实CR响应；每个signed48溢出向量通过64B共同暂存支付6次SW64，恢复支付6次SR64及collector/load/等待。不是免费host中间值，也不是添加第二ROM。

## 数值和资源范围

代码：[binding.py](binding.py)绑定旧R24+onepass父；[kernel.py](kernel.py)执行真实编码与U/V；[predictor.py](predictor.py)收费计算Dg；[latent_cse.py](latent_cse.py)执行完整D图与spill。新matched320或两项源学生的825结果不属于这些参数。

35条updated、projection门、U及PED均对各自函数0差，实际PED SRAM/DMA外送全部核验；30条量化臂各检查15,360个真实码或重建值。每个码臂的1,920次SW64与1,920次H8源RF load来自实际计数。原U/V的RNE、sat、bias顺序保留，常数外提仅在当前参数全1024门码及signed8范围无重建sat24时合法。实际halo未触发clip8，边界覆盖不能说成已全网测试。

同96×8×48 RF、128KiB状态/系数池、SR64/SW64/CR256、单issue及64B暂存。full-U最多73活向量，CSE+spill最多96，V最多90。新增clip8、门条件加等沿两槽写回计费，尚无对应完整RTL频率证明。中间日志的逐k收集不是最终分母；最终全部臂共同使用H8驻留，普通控制同得该供数优化。

独立审阅见[REVIEW.md](REVIEW.md)。当前不含native projection、全域动态BN/join或上游整层，未新增AEE、训练、RTL全链、EDA/PPA和生产修改。算法资格仍按同口径NB0，未恢复旧+0.005门；这次不能给其他父版本补精度。

## 后继只选一个不同接口

[NEXT_FAMILY.md](NEXT_FAMILY.md)选择跨独立推理帧、同坐标的源门安全半径缓存：输入发生小变化时，尝试先认证整个H8×T10门不变，再取消完整源图。它不复用旧T10位移或水平P2证书，也不删除连续残差/PED。普通exact memo、变化检测及同缓存对照同权；安全半径本身属借入，补核最近邻后独立概念评分仅3/10，当前不作为标题候选。需要新采matched dense连续帧I24，旧四帧只有另一身份的门/flow，不能代填。该后继尚未执行，不报命中或性能收益；若实做仅得到通用缓存增量，同样只保留为底座。

相位因子化的`U*z + U*delta_phi`、更窄PE、训练表示与物理组剪枝仍保留未测范围；当前跨SRAM反例不能替它们作裁决。本阶段不追加同布局扫参。

## 复现

使用Python3.12和单线程BLAS，例如：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.12 run.py ordinary corner
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.12 run.py ordinary interior --stress
/usr/bin/python3.12 summarize.py
```

其余ready轴为ordinary/interior、lifting_raw/corner和lifting_raw/interior。汇总只读最终五份，不执行新实验；相位双臂用独立入口，不改变本表。
