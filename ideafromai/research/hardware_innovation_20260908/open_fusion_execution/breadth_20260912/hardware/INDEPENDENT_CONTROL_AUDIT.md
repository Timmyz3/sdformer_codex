# 普通affine、W4父身份及端口费用复核

结论：本轮没有发现把expanded U冒充packed存储或漏收PED写出的错误；但两个父模型不同，必须保留分表。affine是应保留的普通强对照，不能将它说成与fixed/full-D同step的消融。

**普通affine。** 独立读取两轴训练捕获，各3,932,160个实际I24值。用整数不等式`255×step≥max−min`求最小dyadic步长，再核导出c的signed8非对称中心，两轴所有10个参数都一致；D确实全零。ordinary的t4/t7步长4096，其余2048；lifting_raw仅t4为4096，其余2048。训练范围完整落入`[c−128step,c+127step]`，最小端点余量为7,531／4,530整数单位。独立float64精确dyadic RNE重建的code均未越signed8范围，误差均不超过半step。这里校验的是训练范围；不能保证验证帧无clipping。

`representation/run_aee.py`在真实父helper的`U_ped`入口改写I24，而后使用实际R24 U/V/bias；门由同一个真实compare与原permutation生成，并与后续实际projection gate核对。没有用验证帧拟合或选择参数。该实现是实际函数验证，门在硬件端是否已经可用仍要付出跨边界转发/重排费用，decoder叶没有替它免费产生g。affine只有c、无g依赖；本轮c-only decoder采用相同常量缓存权限。其step缩放与残差编码不在decoder叶费用中，不能用7,694槽代替完整量化/PED链。

**W4/W8父身份。** `stage_20260912/algorithm/run_weight_controls.py`明确使用原R32＋CUDA BN，实际U形状32×96，V为96×32，U指数16、V指数15。与本轮representation的ordinary原排列R24／lifting白化R24＋onepass BN不同（U为24×96，V为96×24）。独立读取四份实际GPU部署constants，`code×row_scale`与导出U逐值相等，V与低位包中的原V也相等。W4/W8的新十帧AEE可以对应本轮packed R32局部执行，不能拼给R24＋onepass组合或继承其825。

**端口和算术收费。** 五个固定case中，W8相对同函数expanded实际CW少2,976B、CR少23,808B；W4相应少4,512B／36,096B。所有case的SR和SW差均为0，包括压力例。代码检查确认code/scale/头部是实际coefficient字节，packed体没有expanded U。RF解包→共同64B staging的16B权重区域另付一槽；MAC的两RF读为acc＋source，权重由staging读取，不是额外RF第三口。尺度用signed24高低拆分、两个已有16×24乘法、shift/add，之后执行原U RNE/sat；V RNE和bias及所有输出写回保留。这里确认的是当前解释器明确收费的接口，未验证新低位datapath时序或ASIC面积。

原R32完整Machine及count逐项复现旧结果；各候选复用的是实际执行后的前级Machine对象，包括状态、时刻与仲裁，不是把两张服务表相加。packed相对同函数expanded仍多15,650／16,706 ready槽，故当前布局只证明带宽减少，没有净服务收益。native投影、全域BN和join均不在这个局部范围，保持原缺口说明。

审计脚本：[independent_control_audit.py](independent_control_audit.py)，数据：[independent_control_audit.json](independent_control_audit.json)。本次没有新增AEE、训练或生产修改。
