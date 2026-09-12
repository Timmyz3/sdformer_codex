# 结构T10源：同机器完整局部集成

已把两套真实源程序接入一台共享端口/状态的机器，经过真实门缓存后完成完整K864整数双消费者。共同R32下，lifting在corner/interior分别减少7.784%/7.996%总服务；双方都享R24后仍减少约7.95%/8.13%。这是局部模型服务，不是整网或RTL加速比。

|共同执行底座|窗口|ordinary服务槽|lifting服务槽|lifting净减少|
|---|---|---:|---:|---:|
|R32|corner|2,107,838|1,943,768|7.784%|
|original_ordered24|corner|2,062,654|1,898,584|7.954%|
|activation_whitened24|corner|2,062,654|1,898,584|7.954%|
|global_group2_R32|corner|2,015,403|1,853,929|8.012%|
|R32|interior|2,800,716|2,576,768|7.996%|
|original_ordered24|interior|2,755,532|2,531,584|8.127%|
|activation_whitened24|interior|2,755,532|2,531,584|8.127%|
|global_group2_R32|interior|2,678,362|2,457,462|8.248%|

计量起点是两学生完全相同的真实I24 halo：corner 77,760值，interior116,160值，逐值0差。源按原two_stage_writeback程序执行，再经过完整K864 preview U32/V、固定BN1、非因果T10 sn2、驻留门缓存、完整K864 Conv2 U16/F/BN2+raw I24，最后生成projection gate和真实PED U32/V96（或已导出U24/V96参数）。

边界是真实同Machine状态交接：不外写/重装sn2门。前级124,544B系数及后级44,096B R32系数在同128KiB池里实际依次替换，raw I24后继重读、输出DMA、地址/目录和所有RNE/sat仍收费。R24后级少3,072B系数，并在实际24维循环中减少计算/状态访问。R24分支从已实际执行后的完整Machine状态复制；这是替代方案仿真复用，未导入独立时间表或以gold喂入门缓存。

普通源282条程序含260次加减；lifting216条含159次加减和35次norm24（原位置的RNE+signed24饱和）。两者同8192B ROM、96×8×48 RF、128KiB state/coef、SR64/SW64/CR256。源本身减少16.369%服务，进入真实消费者后成为上表约8%的净值，不能把260→159写成硬件收益。

**归因需拆开。** 上表比较两套不同训练后权重/门活动的学生，不是固定后级工作量下只换源ALU。R32 corner总少164,070槽，其中源少84,564、后级少79,506；interior总少223,948，其中源少126,324、后级少97,624。后级占总节省约43.6–48.5%，不能全部归给源结构的纯电路效应；源服务差分别相当于ordinary局部总服务的4.012%/4.510%。这是当前软硬协同学生的费用分解，尚无仅源结构的独立因果消融。

双方的原排列R24与白化R24在这四窗费用相同，每窗均减少45,184槽；其精度不同，不能以相同服务推断相同输出。global-H8只作已固定R32掩码消融，metadata在同state池冷填一次并供前后级复用，未混入未经评价的R24+mask网络。该地址/剪枝底座不单独构成X。

独立审阅发现mask相对unmasked同时改变了目录调度，因此补跑了四窗 `all_keep`，通过同一mask管线但不删门，所有gold与unmasked一致。all_keep相对最强unmasked每轴corner多825槽、interior多1,588槽；同管线剪枝分别少ordinary93,260/123,942、lifting90,664/120,894槽。保留更强原unmasked作主分母后，净省ordinary92,435/122,354、lifting89,839/119,306槽。没有换用更慢all_keep夸大收益；具体对照在 `mask_fusion_all_keep.json` 和 `summary.mask_same_pipeline_control`。

检查：R32整数消费者618,240值0差；R32+两R24分支共1,854,720值对各自独立gold 0差。源sn1和sn2门均符合原捕获。preview的raw/BN1 FP32与CUDA存在原来已有的微差，本轮逐项复现旧模型差异；没有把全链所有浮点值误称逐位相等。ordinary/interior固定压力的R32、两R24均通过，服务见summary.pressure。

质量归属：原fixed R32完整825帧的ordinary AEE1.219801、lifting1.232979均优于同人口本地SDformerFlow NB0的1.445353；该结论来自accuracy_baseline中已逐帧配对的算法验证。R24与mask不继承原R32的825结论，后续新验证由独立算法任务记录。本模型没有回放完整帧native projection/global BN。

创新归属：CSE、Gustav式稀疏供数、普通word coalescing、resident MAC、R24本身都归借入或共同底座；当前保留的候选是可学习T10源结构经过真实双消费者后仍剩的净服务。约8%是实测线索，尚不足以宣称TCAS-II强接收。本阶段没有轻率否掉源结构家族，也没有把条件消融改成新标题。

文件：PLAN.md记录先写的B/A/X与强对照；integrated.py执行共同R32；rank_fusion.py与consumer_ranked.py执行真实R24；mask_fusion.py为固定global R32消融；source_rtl_inputs提供原程序、真实输入/gold和实际sink事件；RTL_REVIEW.md独立检查共同源RTL。rtl_source结果与CPU表分开，不相乘或互相替换。
