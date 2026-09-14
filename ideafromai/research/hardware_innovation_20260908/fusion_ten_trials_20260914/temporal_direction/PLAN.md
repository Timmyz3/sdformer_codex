# 精确T10共同方向：固定试验与最终公平控制

旧dataflow及生产只读。完整packed15原生C96/K864/R8/N96/T10、4×4→2×2和真实FP32 identity→J20→I24；非零残差全部精确执行，不训练、剪枝或改变quality。

三臂：0完整R8；1允许full、前一真z的Δ、首个非零真anchor的Δ；2只多anchor系数−1、+2、−2。每P从t0重启，anchor固定为首个非零真z。current/previous/anchor/best四个8×13寄存器和八条32bit数据ALU共用。每实际候选一拍宽减法、一拍范围/费用比较，strong控制只求自己有可能改进的候选；当前best已达到候选base必要费用时直接剪掉该候选。prev==anchor时跳重复anchor。零输入直接full0。完整残差原位写signed13 z，非法缩放候选拒绝、full保底。

费用是所有12个N8组实际残差MAC数：由有序cfg5的96个Q2行在RTL生成8个rank_cost（非零行计数），按残差非零位求和。full/prev无base费；anchor有12次base读，负号再12次共享ALU。正2倍为BASE_READ固定wire shift，负2倍同shift再一次取负。此评分比较已算出的表示后续服务，不把已经付出的候选编码费隐藏掉；完整报表仍加入所有encoder拍与后续cache/control费用。

prev p直接使用旧acc；选择prev时POSLOAD不清acc、不读写prev缓存，与旧强Δ控制同权。只保留一个256bit anchor寄存器bank；每P的anchor_needed由本轮SV已产生的40项choice在EWRITE中按P清零/OR，Q2开始前已就绪，只有确实会引用的first-anchor才用独立ANCHOR_SAVE一拍保存。元数据来自真实encoder，没有TB未来信息。所有模式拥有相同状态、端口和共享算术；完整R8可以不执行无用途encoder/cache操作。

固定真Q1逐rankΣ|Q1|最大667，缩放残差绝对值≤2001，signed13足够；任意可配置Q1仍做32bit临时计算/范围判断。32bit base的wire倍增/取负安全；逐rank残差修正保持每项为真z或缩放参考值，累计也在signed32范围内，不依赖溢出抵消，后继仅在原I24处RNE/sat。所有T都消费真实identity。

实际19fixture：原16（8真实+零/一/padding/正负极值/FP tie/saturation）加3个固定机制验证（各α、非零残差、超界回退），每臂双背压/无reset重启。真实固定点为负，停止扩帧/α字母表/anchor数/比例扫描。初始带prev读写的控制收据只留first_control_receipt.json追踪，不属于最终主结果或新增试验。
