# H67主线DATE贡献冻结与投稿前签核清单

## 1. 主线选择

硬件主线冻结为H67 epoch19，H68不作为第二套硬件主线。理由不是只看AEE，而是：

- H67逐位RTL模型AEE `1.462688`，优于H68的`1.472654`；
- H67和H68 spikes、firing非常接近，H68没有显著事件能效补偿；
- H68的可部署图就是TTX，单独实现matrix辅助会违反软件eval图；
- H68适合作为“训练期富分支、部署零矩阵增量”的软件消融，不足以单独构成硬件创新。

当前论文硬件基线应至少包含TTX、H67 dense replay、H67固定35类扫描和H67占用类SCS四项。

## 2. 当前可以主张的贡献

### 贡献一：运动感知的全二值部署协同

H67把`K0 XOR K1`运动证据加入统一all12 H60 token score，在冻结dyadic网格上只需要XOR、
popcount和整数加法。完整RTL Shiftmax valid825相对原部署AEE变化仅`+0.000055`。

可写成“针对T=2二值事件光流的运动感知定点score/硬件协同设计”。不能写“首次时空spiking
attention”，STAtten、LoAS、SpikeTA等已有明确时间处理先例。

### 贡献二：SCS-Shiftmax精确分数类流处理

利用`gate*K`中零K输出为零、但分母贡献非零的不对称，把零K token按最终Q7 score code聚合：

```text
sum_token exp(score) -> sum_class count[class] * exp(class)
```

活动K仍逐token回放，数值不变。H67使用35类占用位图和两拍类流水，profile100周期代理下降
`12.86%`。这部分是当前最接近独立硬件创新的内容。

不能写“提出Shiftmax”“首次重复消除”或“跳过零K score”。I-ViT、Bipolar Self-attention、
Softermax和BLADE分别覆盖了Shiftmax/base-2 normalization/重复计算等邻近概念。论文必须强调
最终score multiplicity、精确分母和gated-zero输出三者的限定组合。

### 贡献三：部署执行图消除carrier状态访问

软件安装覆盖为ATLIF105，但真实H67/H68部署执行覆盖为93；未执行的12个全部是原
`sn2_q` carrier。固定H60部署核据此删除12次逻辑神经元调用及其状态访问，并统一为`gate*K`。

该项属于软硬件图冻结和系统优化，不应单独声称算法新颖性。它的价值是防止硬件按PyTorch
module数错误实例化，也给全encoder状态SRAM容量提供准确口径。

### 贡献四：描述符复用的统一实现

12个attention block由一个`stage/block/head/window`调度器时间复用，93个实际ATLIF调用点也
应由一个或少量lane cluster按描述符执行，不实例化12套Shiftmax或93套算术单元。

descriptor/time-multiplexing属于常规架构组织，只作为前三项的落地方式，不列为首要新颖性。

## 3. 当前不能主张的贡献

- ETCR三级`unchanged/TPSF/active-K`路由尚无RTL；只能列为下一轮架构假设。
- Motion-Delta Fusion尚未完成共享前端综合对照；不能声称面积或功耗下降。
- H68现有训练没有class/toggle硬件正则，不能把AGDS写成已经实现。
- 64-bit temporal pair不降低信息容量；只有地址trace证明baseline未合并时才能报告事务下降。
- 500MHz是探索约束，不是已达到频率。
- spike activity energy不是ASIC能耗。

## 4. 推荐命名

当前已实现版本建议使用：

- 中文工作名：**面向全二值脉冲光流的运动感知分数类流式加速器**；
- 英文工作名：**SCS-H67: Score-Class Streaming for Motion-Aware Binary Spiking Flow**；
- 核心单元：**SCS-Shiftmax: Score-Class Streaming Shiftmax**；
- 类扫描子机制：**Occupied-Class Scan，OCS**。

只有在Exact Delta三级路由RTL、PPA和逐位验证完成后，才升级为：

- **ETCR: Exact Temporal-Class Routing for All-Binary Spiking Flow Transformers**。

命名不使用`first`、`novel Shiftmax`或`zero-token pruning`。

## 5. 论文插图

### 图1：软件图到部署图

左侧画训练图：H67主支；旁边以虚线画H68 training-only matrix auxiliary。右侧画统一部署图：
binary ATLIF -> H60/H67 score -> SCS-Shiftmax -> gated-K。明确aux在eval为零、12个carrier
`sn2_q`不进入部署执行图。

### 图2：全encoder描述符数据流

画四个stage和12个block descriptor进入单个共享row engine。S0/S1/S2画三条encoder skip；S3
画成bottleneck输出到第一个decoder，不画成第4条skip。block内部残差保留在主数据流旁路。

### 图3：H67 row engine

从`{K1,K0}` temporal pair开始，分出overlap/same-zero和Motion-XOR；之后分成零K class bank与
active-entry replay bank，在共享exp2/denominator/gate后只发活动K。

### 图4：SCS两拍时序

横轴画`LOAD -> SUM_ACTIVE -> FIND_CLASS -> CLASS_MAC -> EMIT`。同时画H68编译期单拍路径。
标出位图pop、histogram clear、exp2 transaction和backpressure稳定点。

### 图5：结果图

包含四组数据：AEE/spikes；固定类与占用类周期；精确与256填充存储；DC后的area/power/Fmax。
前三组已有数据，第四组必须等待工艺库。

### 图6：创新边界图

用表格区分已有工作：Shiftmax、training-rich/deploy-simple、token-time packing；再突出本项目的
限定组合：final-score multiplicity + exact denominator + gated-zero replay elimination。

## 6. 投稿前硬门槛

### P0：必须完成

1. 目标工艺库和PVT下完成TTX/H67/H67+SCS同约束DC；WNS/TNS均满足目标。
2. active-entry buffer选择触发器或同步SRAM宏，并把宏面积、读写能耗和读延迟计入周期。
3. 用真实H67 trace生成SAIF/VCD，报告clock、组合、时序、SRAM、漏电分项。
4. Formality RTL到DC网表全部compare point通过。
5. 将projection、93个ATLIF执行点、状态SRAM、三条skip和必要数据搬运纳入系统PPA模型。
6. 逐row软件参考与RTL trace比较row max、denominator、gate、token index和gated-K。
7. 至少补H67三个训练随机种子；若论文强调泛化，再补第二事件光流数据集。

### P1：强烈建议

1. 固定35类、占用类和dense三种分母实现做break-even曲线。
2. H67 Motion-XOR前端与H68/TTX前端做面积、功耗和关键路径差分。
3. 完成Exact Delta/ETCR RTL前，至少给出previous-state SRAM成本后的净能耗模型。
4. 沿I-ViT、Softermax、BLADE、Bishop的引用文献继续追踪histogram/equivalence-class softmax先例。
5. 报告平均值之外的p50/p90/p99 active class、active entry和row latency。

## 7. 当前投稿判断

当前创新组合已经比“统一H60+Shiftmax”更具体，也有真实RTL和数值证据，但**还不能直接投稿
DATE硬件论文**。最主要的缺口不是继续包装名称，而是目标工艺DC、SRAM、真实活动功耗、正式
LEC和同约束基线。

完成P0后，论文故事可以收敛为：

```text
H67运动感知全二值算法
  + 无carrier部署图
  + 精确最终score类SCS-Shiftmax
  + 描述符时间复用
  + 工艺库下可复现PPA与逐位验证
```

H68保留为“训练辅助不增加部署矩阵硬件”的消融，不与H67争夺主硬件标题。
