# 共同资源与事务合同

|资源|组织与共同权限|
|---|---|
|原生source|1536×10bit、一词读口；C1 16×10bit局部窗/四10bit读取mux。全C96/K864服务。|
|Q1/Q2|Q1 8×864×signed3，一24bit共同K行；Q2 8×96×signed16，一128bit行；R8×N8局部Q2 cache128B。静态一次常驻。|
|z|8×20×26bit=520B，一208bit向量读写与26bit标量选择；两13bit空间半字。encoder每tile额外40读/40写，保留非目标半字。|
|psum|8×480×32bit=15360B；8×32acc、256bit结果holding。每tile480写/480读；prev引用直接保持acc。|
|producer算术|8个signed19×13乘法器、8条32bit加减链。ZADD在bit13断进位，MAC和encoder/负base用整32bit链。没有另加encoder数据减法器或乘法器。|
|encoder数据状态|current/previous/anchor/best四组8×13=416bit；difference8×32=256bit。只保留一个原位完整z，不增加40×8的残差副本。|
|参考结果|一个8×32=256bit anchor bank，单256bit读取或写入/拍；BASE_READ和ANCHOR_SAVE各真实一拍。prev结果是共同acc，不复制prev p。|
|metadata|40×3bit choice、40bit first-anchor、4bit anchor_needed。当前choice/first标志同索引组合读取，P级4项summary selector；summary由EWRITE按P清零/OR，完整Q1/encoder之后Q2才能读取。|
|费用控制|8×4bit rank_cost寄存器；小位宽费用求和/比较、优先选择、范围检查与candidate pending。费用归约属明确新增组合控制，不计成不存在的“免费八数据ALU”；没有数据位宽乘法。|
|后继consumer|与旧冻结i24_consumer.sv文本完全相同：另有8×32×32乘法、8×64加法、8lane IEEE32→J20/RNE/sat，系数768B、一256bit共同A/B行口和224B向量buffer。不能将它免费并入producer八ALU。|

cfg顺序合同：冷命令完整Q1(kind4,0..863)、Q2(kind5,0..95按og/r升序)、k_live(kind6,0..863)、consumer(kind7,0..23)，共1848beats；rank_cost首og写初值，后11og累加。warm同参数驻留，无任意乱序/局部权重patch合同。source1536词+origin1每命令重新由SV绝对地址装入，图外padding生成也付拍；每tile真实480个FP32 identity和480个I24输出向量。外部W/source/identity/result背压实测；本次只有单tile命令，无整帧吞吐/DDR延迟模型。

每候选EDIFF一拍共享ALU，EVAL一拍范围/费用/选择；EREAD/EWRITE每向量两拍。`encoder=80+2*candidate_trials`（mode0为0）。

完整producer拍：`5437+liveK+2*activeK+3*packed_Q1issues+Q2MAC+encoder+base_reads+base_negations+reference_writes+source_stalls+weight_stalls+output_stalls`。BASE_READ里的正2倍是wire shift；负1/2再BASE_NEG一拍。仅被需要的anchor在其真实p完成后另写一拍。实际cost中参考读取为12N8/选择，negative再12，prev0；cachefill与所有FSM都在总拍，不以候选费用表冒充总性能。

consumer拍=`3385+join_wait+output_stalls`。top总拍=`consumer_cycles+static_words+parameter_stalls+source_load_words+origin_words+source_load_stalls+3`。producer与consumer重叠，不能相加。warm命令只省静态1848，source/origin/identity都不是免费。

数值：真实Q1范数[667,480,501,447,589,597,472,662]给出所有α残差≤2001。可配置signed3 Q1的范数≤3456；原z仍在signed13，但缩放残差可到10368，所以必须判断[-4096,4095]再提交13bit。全Q2 signed16使|p|≤8×3456×32768=905969664，2倍≤1811939328，signed32 base shift/neg安全。逐rank残差修正后的acc中，每个rank分别已经变为真z或仍为αb，所以所有中间累计同样≤1811939328，不依赖溢出后的抵消。完整执行后得到范围内真p，之后原signed64消费者与唯一RNE边界保持。

没有PPA/Fmax/能耗数据；union资源共同并不证明分别裁剪后等面积。未训练/未重跑质量，因为所有出口与已评价固定整数函数逐位相同。不将本点拼入D3或计作原十项之一。
