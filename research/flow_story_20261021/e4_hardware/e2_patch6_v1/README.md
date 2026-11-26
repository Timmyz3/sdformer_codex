# E2 PATCH6：真实动作布局的有限供数预算

本文件只回答：冻结E2的15个pair，在原E4资源与调度中是否仍能取消服务，控制需要多少费用，同pair静态方法有多便宜。**没有评分器实现、GPU、GT读取、新RTL或新DC；没有网络精度结论。** 原E4冻结文件未改。

## 绑定实际动作

来源：[PATCH6_CONTRACT_V1.json](../../e2_group_damage/PATCH6_CONTRACT_V1.json)，SHA `a6926959cd24167e73fff38c1f7bb0024955cf9c035a50fd10ed081703bb862a`。四个patch conv的几何和tile分组逐项相同，都是T10、240×320、K864、Cout96；每组151个完整9×9 tile，选择两组降为Q4，其他完整tile Q6，真尾Q8。四个未分组完整tile保持Q6。

每conv实际行数：Q4 **244,620**，Q6 **492,480**，真尾Q8 **30,900**。所有通道及T完整保留。pair合计302个tile，取消Q6的bit2/3；每conv取消20,548,080个算术beat，四conv合计**82,192,320**，与算法合同一致。不是token局部屏蔽，也没有先执行Q6再抹掉输出。

[prepare.py](prepare.py)验证合同SHA、四层布局相等及6组互不重叠，然后由冻结E4源码另建[patch6_supply.cpp](patch6_supply.cpp)及[tile_groups.h](tile_groups.h)。每个pair均真实重放完整单conv域；四conv的同形状映射允许复用单域结果。总计46臂：Q6背景1臂＋15pairs×3控制臂。背景结果与原E4逐字段完全相同。[准备收据](PREPARE_RECEIPT_V1.json)

## 三种控制口径

完整算子仍保留E4的有限I/O槽、RF容量/端口、halo、权重全部8plane装载、共享DMA及三遍current-BN屏障。

|control|动作来源及存储|额外控制服务|
|---|---|---|
|0：静态编译乐观下界|同一pair固定于部署程序，假设可由原几何地址控制直接给出mode|不增加table或额外tick；是对普通静态方法有利的控制下界，未综合该静态decode逻辑|
|1：静态precision表|预先生成每tile的start-plane表，910byte，DMA按960B收费|每次tile-span读一个32-bit word，1tick，mode保持至下一span|
|2：动态group表|固定910byte group-ID表＋6-bit selected-group寄存器；pair已给出，尚不包含其预测|每次tile-span读一个32-bit word再做membership decode，共2tick；pair寄存器初始化另3tick|

无需每帧重新生成910byte precision map。group-ID0…5为六组，6是四个leftover；动态只改变6-bit选择寄存器。表及控制状态放在既有stats RF明确保留的2KiB内，另给BN的12×Cout×4B carry/参数区，容量并存。32-bit读最后一个word会触及表尾padding，整个分配按960B计；不读取不存在的tile。

“每tile-span”不是每tile只读一次：同一tile跨9个输出y、T10、六个output16块；batch128行边界若截断9宽span也会重读。每conv实际**519,480次32-bit descriptor读取**，局部读2,077,920B。动态控制串行付**1,038,960tick**，静态表付519,480tick。普通几何/真尾处理沿旧E4共同费用；true tail不查表。读取/解码均放在受控PE行的第一个beat前，没有免费后见信息或未收费的并行。

这是未综合控制器的有限服务模型；1/2tick是明确的串行设计合同，不是新RTL实测。现有Q8/Q6/Q4起始plane RTL协议可消费这些mode，不等于整个table/controller已完成RTL/DC。未来若优化为连续区间译码或descriptor预取，须另证地址、端口与控制面积，不能现在取免费最优值。

## 15个pair的结果

[全部46臂](results/SUPPLY_V1.json)、[全部15个pair费用与预算](results/SUMMARY_V1.json)。单位沿用E4抽象机器tick，含仿射和三个完整BN服务pass。

|单conv口径|15pairs范围|
|---|---:|
|Q6背景|259,342,002|
|同pair静态编译乐观下界|238,793,922–238,794,288|
|同pair静态precision表|239,313,429–239,313,785|
|动态group表，尚无评分器|239,832,892–239,833,238|
|动态比Q6背景少|19,508,764–19,509,110（约7.5224%）|
|动态比同pair静态表多|519,453–519,463|
|动态比静态编译下界多|1,038,950–1,038,970|

所有pair算术beat完全相同；有限DMA相位和输出等待使服务相差至多346tick。descriptor总费用相同，也不能直接把其总tick代替净服务差，因为新增控制改变了少量原等待。表中报告的是实际完整调度完成时间。

四个**孤立算子服务的和**：Q6为1,037,368,008tick；动态pair为959,331,568–959,332,952tick。若附加一个完全串行的运行时预测、输入读取/保持、选择及等待，总新增费用必须**严格小于78,035,056–78,036,440tick**才仍比这个Q6背景省服务。等于此值时净收益为零。这只是目前可支付的上限，不是预测器的实测费用或已经保住的加速。

这个上限只相对Q6背景：**同一个pair的静态控制始终更便宜。** 要证明动态分配有价值，仍须让实际任务质量优势补偿相对强静态方法的额外费用；静态省下的费用也可换成更多Q6计算。不能用“相对Q6还有预算”替代这项公平比较。

四层相同group表可被保留复用，但本轮按每个孤立算子各收费一次，未免费给跨层缓存。完整网络的PSN全T混合、BN数值顺序、残差、跨层存活和评分器依赖仍未闭合，因此**四项服务之和不是整网延迟，不能与旧R27全网百分数相加。** 全部硬件成本对应nested截断，不适用于独立scale/RNE4的双表示。

## 核验与复现

[analyze.py](analyze.py)不读质量，用实际tile集合独立复算每pair的Q4/Q6/Q8行及取消量、519,480次descriptor span、DMA960B、元数据容量、保存样例的mode/plane/地址，并逐字段核对原E4 Q6背景。[检查收据](results/CHECKS_V1.json)通过。它是作者另一路径计数，不冒充独立代理评审。

在新输出目录编译运行：

```bash
g++ -O3 -std=c++17 -Wall -Wextra patch6_supply.cpp -o /new/output/patch6_supply
/new/output/patch6_supply > /new/output/SUPPLY_V1.json
```

当前冻结结果及其源通过[FREEZE_V1.json](FREEZE_V1.json)绑定。没有新完整网络前向，没有任何GT支持或误差筛选硬件布局；15pair全部保留。
