# HIT-Flow统计证据数据质量与可引用等级审计

**日期**：2026-07-13  
**审计对象**：H67/H68 workload、ATLIF执行图、encoder存储合同、DP-TME整数映射  
**总体结论**：带限制内部使用；可继续架构设计，但尚未达到论文最终数据签核条件

## 1. 先给结论

当前证据足以支持以下架构决策：

1. H67作为功能超集、H68编译期特化，共用一套物理数据流；
2. attention采用128-bit时间对驻留、class-stationary后端和参数化context；
3. 全encoder采用`T=10/T=2`双模式时间矩阵阵列，而不是实例化105套ATLIF；
4. 二值event bank与多位residual/skip bank物理分离；
5. 首版DSE至少比较`2×320-MAC`与`4×320-MAC`，单阵列不作为30 FPS主候选。

当前证据**不能**支持以下论文表述：

- “全encoder激活均为1 bit”；
- “HIT-Flow已经达到30 FPS、0.5 mm2或100 mW”；
- “多context、PCCC或蝶形压紧已获得某个真实芯片加速比”；
- “4/6/8-bit ATLIF参数量化不损失AEE”；
- “删除12个`attn_sn`对所有软件模式均无影响”；
- “100个样本代表完整valid825和跨数据集分布”。

因此，当前状态不是“缺数据所以不能设计”，而是：**结构和映射已经可以冻结到参数化RTL规格，配置数量、位宽和论文PPA数字仍需ordered/full825与DC补证。**

## 2. 证据等级定义

| 等级 | 含义 | 论文使用方式 |
|---|---|---|
| A | 代码、checkpoint、精确计数或bit-exact参考模型可复现 | 可直接引用，并注明部署图和配置 |
| B | 真实模型trace统计，但样本或运行范围有限 | 可引用分布和趋势，必须注明`100/825`样本范围 |
| C | 周期、容量或理想端口模型，用于DSE | 只能写“模型预测/上界/下界”，不能写实测PPA |
| D | 待运行、待综合或仅有假设 | 不进入结果表，只进入未来工作或签核清单 |

## 3. 数据源与粒度

| 数据源 | 粒度 | 覆盖 | 等级 | 主要用途 |
|---|---|---:|---|---|
| H67/H68旧profile100 | 每样本、每block聚合计数 | 各100样本、12 blocks | B | pair-empty、K-zero、active-entry、block异质性 |
| checkpoint与module审计 | 每wrapper参数和forward调用 | 105安装、93调用 | A | ATLIF数量、T模式、参数容量 |
| 固定部署图静态活性审计 | 正常forward的数据依赖 | 12个dead `attn_sn` | B | 81点候选执行图 |
| encoder shape/容量分析 | stage张量shape | S0-S3 | A/C | 元素数为A，假定位宽容量为C |
| DP-TME整数参考 | 合成整数向量 | 100组随机输入 | A/C | 整数映射等价为A，PPA与真实量化为C/D |
| 新ordered profile100 | 有序pair、数值域、margin、bank冲突 | 尚未完成 | D | context、PCCC、bank mapping、RPI位宽 |
| valid825量化部署 | 完整验证集 | 尚未完成 | D | ATLIF与residual最终位宽 |
| DC/工艺库/SAIF | 门级与真实活动 | 本机环境尚不具备 | D | 面积、频率、功耗、能效 |

旧profile中的attention记录是聚合计数，不能恢复逐row到达间隔和burst；已有context结果来自按已知block顺序重放的代理模型，不是硬件FIFO实测。

## 4. 已通过的交叉核对

### 4.1 ATLIF执行图守恒

```text
安装：60个T2 + 45个T10 = 105
调用：48个T2 + 45个T10 = 93
固定部署功能活跃：36个T2 + 45个T10 = 81
死亡调用：93 - 81 = 12
```

12个未调用wrapper均为原carrier `sn2_q`；12个被调用但结果死亡的点均为attention内部`attn_sn`辅助返回值。该结论对当前正常推理路径成立；`return_attention`、调试导出或未来改图不属于删除合同。

### 4.2 长skip元素数守恒

```text
S0：6,635,520
S1：3,317,760
S2：1,658,880
合计：11,612,160元素
```

对应容量换算：

| 位宽 | 容量 |
|---:|---:|
| 1 bit | 1.384 MiB |
| 2 bit | 2.769 MiB |
| 4 bit | 5.537 MiB |
| 8 bit | 11.074 MiB |
| 16 bit | 22.148 MiB |

元素数和容量换算可信；位宽尚未冻结。旧profile只证明这些张量“非零率接近100%”，没有证明其值域属于`{0,1}`或`{-1,0,1}`。

### 4.3 时间矩阵操作量守恒

81个固定部署活跃ATLIF点每帧产生526,046,400个输出元素，对应4,424,388,480个标量时间MAC。该数字由真实shape、调用点与`T×T`计算规则得到，是**算法操作量**，不是RTL执行周期，也不包含空间projection、SRAM、地址生成和decoder。

在500 MHz、理想持续满载且不计其他模块时：

| DP-TME阵列数 | 理论ATLIF周期 | 理论ATLIF时间 | 仅ATLIF理论FPS |
|---:|---:|---:|---:|
| 1 | 13,826,214 | 27.652 ms | 36.16 |
| 2 | 6,913,107 | 13.826 ms | 72.33 |
| 4 | 3,456,554 | 6.913 ms | 144.65 |

单阵列虽有36.16 FPS的ATLIF-only上界，但加入attention、projection、residual和访存后没有30 FPS余量，因此淘汰单阵列作为完整encoder主候选是合理的；这不是完整系统吞吐实测。

### 4.4 时间对类别守恒

H67 profile100：

```text
双K-zero 83.11% + 单K-zero 11.09% + 双active 5.80% = 100.00%
```

H68对应为83.29%、10.70%、6.00%。这证明PCCC处理双K-zero事务具有较大理论覆盖面，但“两个score属于同一class”的真实比例仍需新profile，不能把83%直接当作commit合并率。

### 4.5 稀疏指标关系

H67的K lane密度约1.1657%，active token/head约11.347%。每个非零token/head的平均active lane可由下式恢复：

```text
32 × 1.1657% / 11.347% ≈ 3.287 lane
```

H68同样约3.452 lane。该关系通过，支持factorized gated projection优先按active K lane执行；最终收益还取决于权重读取、输出累加和tag开销。

### 4.6 DP-TME整数映射

100组随机整数输入下：

| 模式 | hidden比较数 | event比较数 | 不一致 |
|---|---:|---:|---:|
| T10 | 2,592,000 | 2,592,000 | 0 |
| 五路打包T2 | 518,400 | 518,400 | 0 |

T2处理81个位置需`ceil(81/5)×2=34`周期，尾组slot利用率95.29%；相对未打包162周期下降79.01%。这证明调度映射没有改变整数计算，不证明权重/输入位宽、饱和规则、门级时序或面积。

## 5. 主要质量风险

### 高风险：必须在论文结果前关闭

1. **RPI位宽未知。** 长skip和ADD residual没有新数值域统计；1-bit容量只能作为下界。
2. **ordered行为未知。** 旧数据没有逐row顺序，不能真实评估FIFO深度、p99拥塞、bank conflict和多context收益。
3. **ATLIF量化未闭环。** checkpoint参数Q4/Q6/Q8误差不是event等价，更不是valid825 AEE等价。
4. **无工艺PPA。** Yosys通用单元、周期代理和理想MAC吞吐均不能替代DC、SRAM宏和真实活动功耗。
5. **完整encoder周期未闭环。** 当前DP-TME、TESSA和FGP尚未放入同一资源/带宽约束模型。

### 中风险：可以带限制使用

1. profile100取100个验证样本，尚未证明与完整825样本的均值、尾部和stage分布一致；所有百分比必须标注`n=100`。
2. 81活跃点是当前固定部署图的数据依赖结论；应补一次删除/旁路前后的输出hash或valid825无差异检查。
3. H67/H68 workload相近已在多个指标出现，但目前只覆盖各自一个checkpoint，不代表所有后续H7x候选。
4. 旧SOPS约42.27G/2.23G来自更早配置和代理口径，不能替代H67 epoch19的完整encoder操作分账。

### 低风险：已适合冻结规格

1. 四stage的`head_dim=32`、window 9×9和`T=10/T=2`结构；
2. 105/93 wrapper调用计数及T模式构成；
3. 128-bit时间对布局与81位置窗口；
4. S0-S2三条长skip，不包含S3长期skip；
5. DP-TME的T10和五路T2整数调度关系。

## 6. 新profile必须通过的自动检查

新ordered profile完成后，统计脚本应先执行以下完整性检查，再生成架构结论：

1. H67和H68均为100个样本，每个样本恰有12个attention block记录；
2. stage边界必须包含`finite_ratio/min/max/absmax/mean_abs/near_integer/binary01/ternary`字段；
3. 93个动态ATLIF点均有首次输入格式记录，且12个dead点带显式标签；
4. 每个ATLIF采样记录的T与module参数维度一致；
5. reference recompute mismatch必须为0，否则量化margin结果作废；
6. pair类别满足`双K-zero + 单K-zero + 双active = 总pair`；
7. PCCC同类率不超过双K-zero率，active-entry不超过162；
8. 每种bank映射报告访问数、冲突数和分母，不能只给百分比；
9. ordered FIFO模型报告均值、p50/p90/p99/max、stall原因和上下文占用；
10. H67/H68分别输出全局、stage、block三级结果，禁止只报全局均值。
11. Linear/Conv2d/Conv3d逐算子记录必须按encoder、bottleneck、decoder和prediction分账；dense标量MAC与活动率加权MAC代理不得混为同一指标。

## 7. 架构冻结门槛

| 决策 | 当前状态 | 冻结条件 |
|---|---|---|
| 2或4个DP-TME | 候选B/C | 完整encoder 30 FPS、带宽和PPA DSE |
| 2或4个row context | 首版逻辑支持1/2/4 | ordered trace下4相对2的EDP改善至少8% |
| PCCC默认开启 | 可旁路候选 | 真实同类合并率与额外比较/仲裁PPA后净EDP改善至少5% |
| BMRF蝶形压紧 | 条件候选 | 同约束DC下总子系统EDP改善至少8%，否则删除 |
| 4/8/16-bit RPI | 未冻结 | full825部署AEE、溢出和存储能量联合Pareto |
| 4/6/8-bit ATLIF参数 | 未冻结 | event mismatch、valid825 AEE和DC PPA联合结果 |
| 删除12个dead `attn_sn` | 固定部署候选 | 输出hash/valid825无差异并锁定部署模式 |

## 8. 当前可用于DATE的克制表述

在ordered profile和DC完成前，最严谨的表述是：

> 对H67/H68各100个真实验证样本的profile显示，时间对全空约74%、双K-zero约83%，且不同block的active-entry相差近20倍。代码和checkpoint审计进一步显示，固定部署图包含81个功能活跃PSN时间矩阵点，时间尺度仅为可整除的T10/T2，四个stage均保持32维head和9×9窗口。基于这些结构和workload事实，我们提出HIT-Flow参数化架构，并将多context、PCCC、蝶形压紧和精度配置保留为需经ordered trace及同约束PPA淘汰的候选。

不能把“提出”写成“实现并达到某PPA结果”，也不能把理想上界写成实测加速比。

## 9. 下一步最小闭环

1. 等待已排队的新H67/H68 ordered profile100，执行本文件第6节检查；
2. 用真实PCCC同类率、burst和bank conflict冻结TESSA context与供数结构；
3. 对12个dead `attn_sn`做软件旁路输出等价验证；
4. 运行ATLIF Q4/Q6/Q8与RPI 4/8/16-bit的最小valid825部署矩阵；
5. 建立DP-TME、TESSA、FGP、RPI共享带宽的全encoder周期/事务模型；
6. 仅对通过门槛的配置进入RTL、Verilator、综合和DC交付流程。

## 10. 自动回填流程

为避免长训练队列完成后仍沿用旧代理，已启动独立CPU watcher：

```text
scripts/run_hit_flow_postprofile_after_ttb_v2.py
```

它等待`ALL COMPLETE TTB/DELTA CYCLE V2`后执行：

1. 审计H67/H68均为100样本、1200条attention记录和800条stage边界记录；
2. 生成`results/hit_flow_ordered_profile_analysis.md/json`；
3. 用H67逐算子encoder活动率加权MAC替换旧全网SOPS代理；
4. 生成`results/hit_flow_full_encoder_budget_ordered.md/json`；
5. 任一字段不完整或守恒失败时停止，不输出冻结结论。
