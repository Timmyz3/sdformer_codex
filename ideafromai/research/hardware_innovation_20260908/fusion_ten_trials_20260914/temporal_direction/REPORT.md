# 精确T10共同方向：新增缩放没有胜过强Δ控制

**真实八块固定点为负，停止扩展。** 三臂完整原生C96/K864/R8/N96/T10→rawp→真实FP32 identity→J20→I24。新增α∈{−1,+2,−2}在真实数据没有被选中；候选比full/prevΔ/anchorΔ强控制多 **1,058拍**，完全等于额外529次在线候选×2拍。没有删除非零残差，没有改变整数函数或重新训练。

|八个真实tile，无外部stall|full mode0|强Δ mode1|扩α mode2|
|---|---:|---:|---:|
|冷服务总拍|134140|135102|136160|
|暖服务总拍|119356|120318|121376|
|encoder拍|0|1226|2284|
|实际候选次数|0|293|822|
|Q2向量MAC|17604|17292|17292|
|anchor读取/保存词|0 / 0|24 / 24|24 / 24|
|prev / anchor选择次数|0 / 0|18 / 2|18 / 2|
|新增−1 / +2 / −2选择|0 / 0 / 0|0 / 0 / 0|0 / 0 / 0|

主差分冷拍+0.783%，暖拍+0.879%。冷表每fixture付1848静态词，八块14784拍；warm同参数无reset，但每块1536新源词、origin及480个真实identity照付。不是只比较MAC计数或省略加载。producer和consumer重叠，不能相加其active周期。

每P固定第一个非零真z为anchor；所有残差由真实SV八ALU算出，选后原位写13bit z。评分用Q2静态非零行数求12个N8组的残差MAC费用，加真实base读取/负号费用，lower-bound无法获益的候选不枚举。prev直接保留原acc，绝不清零再读回；anchor只在本P的真实已编码metadata显示将被使用时保存。正2倍在base读取中wire shift，负1/2才另付一拍共享ALU取负。首次非零、每P重启及全部T输出的I24仍完整执行。全部端口/寄存器/控制和配置顺序见[RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md)。

最终19fixture×3臂×2背压×2命令=**228条**，rawp、实际J20、I24各 **875,520值全绿**。原16含8真、零/一、padding poison、正负权极值、FP转换/负tie/饱和；另3个固定功能向量覆盖所有新增α、三种缩放各带rank0非零残差、signed13超界拒绝。range_guard中真anchor=2592、当前=27，+2残差−5157、−2残差5211均被拒绝；每mode2命令8次，四种stall/restart记录合32次。任意超界候选回退合法表示，没有截断冒充精确。

定向direction_alphabet暖拍强控30506→候选28794、direction_residual24479→22919，说明实现能利用真实共同方向；它们是功能反例，不替代真实分布性能。固定真Q1范数使所有新增残差≤2001，range_guard则覆盖可配置权重下必须保留的安全分支。

[verify.py](verify.py)从原生source与Q1/Q2独立重建z、完整p/I24、候选选择、reference生命周期及所有事务/总拍，10,648项通过；没有读取RTL导出的latent/goldmask作机制输入。原full mode0与冻结完整消费者控制的共同16fixture全部整数计数复现。独立意见见[INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md)。

“乘积复用+精确残差”已有强A背景，接口来源与候选边界见[原提案](../decompositions/NEXT_INTERFACE.md)。本点没有建立缩放X。首轮带不必要prev缓存读写的旧控制仅保留first_control_receipt.json追踪，**不叠加最终228条或性能统计**。本点独立于原十项及D3，不拼接收益。无整帧扩展、quality重跑、EDA/Fmax/能耗结论；负结论限当前固定anchor/字母表/布局。
