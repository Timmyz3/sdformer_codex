# temporal_direction 独立实现审阅

2026-09-14。审阅者提出过前置接口假说，但未参与本目录RTL/TB实现；这是独立实现审阅，方法意见仍披露该假说参与关系。只读[core](temporal_core.sv)、[wrapper](consumer_stream.sv)、[consumer](i24_consumer.sv)、[prepare](prepare.py)、[TB](tb.cpp)与运行收据，没有重跑RTL或新增数学实验。**现有三模式主路径未发现功能阻断；必要prev强控制已补齐。真实新增缩放零命中，当前固定点为负。** 最终228条运行收据已核对，详细覆盖见页末。

**三模式的同函数身份。** mode0完整执行Q2 z。mode1允许full、前一真z的Δ及本P首个非零真anchor的Δ；mode2只新增anchor的−1/+2/−2倍。EDIFF真实读取完整Q1产生的z，用共同八条32位链求残差；EVAL判范围与费用；EWRITE才存残差及选择码。TB只给原生二值源、常量、FP32 identity与完整gold，没有α、latent或动态PWP输入。“在线”指运行时在完整Q1结束后编码，不是免费预测或边源生产边分解。所有非零残差都执行，最终rawp、J、I24的gold对三模式相同。

每P的t0禁用旧prev/anchor，首个非零强制full；previous_z始终写真z，不写残差。first_anchor和anchor_needed均按P重新形成；只有本P后续确有anchor引用才保存对应Q2结果。Q2按og、P、T顺序执行：full清acc，prev直接保留上一T精确acc，其他引用读取该P/og的anchor并按α缩放，再加入全部Q2残差。该归纳保持精确p；不会把前一P/og或命令的旧缓存当当前参考。首次anchor以full形式保留必要rank，最终对编码残差的rank_live扫描不会漏掉生成anchor所需的Q2权重。I24消费者文件与此前consumer_packed逐字一致，每个输出都消费自己的identity，未复用舍入后的I24。

**强控制与费用。** 首版先清acc再重读prev，并每输出REF_SAVE，审阅指出它遗漏旧rank Δ已有的acc直通权限。当前代码已删除prev bank/REF_SAVE，choice_prev不清acc、不BASE_READ；其费用只有残差MAC，anchor仍付读/必要保存，负号仍付共享ALU一拍。该缺口已经关闭，旧首版不得作为主分母。rank_cost由完整有序cfg5累计每rank实际非零N8权重组数（最多12），full与残差评分都用它；prev/anchor相同跳重复候选，低于最小base费用时停止检查，两臂同权。该评分用于比较消费费用，额外候选每次EDIFF/EVAL的两拍仍真实收费，不能把“选中后MAC更少”当总净收益。

共同资源包括原生16×10源窗、208bit z向量口、signed19×13的八个Q2乘法表达式、八条32bit主加减链、完整cachedOS及psum；新增current/prev/anchor/best、32位difference、40项选择/first-anchor、4项anchor_needed和单256bit anchor结果bank。Q2输入实际signed16先扩到19。rank_cost/nnz归约、比较与优先选择还有小组合逻辑，不能称整模块仅八个加法逻辑。正2倍在BASE_READ线路移位，负号使用BASE_NEG，不引入隐形乘法器；该读→移位/选择路径尚无Fmax证据。全模式具备共同资源且各自支付实际访问，不等于裁剪后等面积。source/权重/identity/output握手、每tile1536个源装入与origin、冷参数1848拍、暖命令参数驻留，均由同一wrapper执行。

**位宽。** 实际固定Q1每rank绝对和最多667，缩放残差保守界`(1+|α|)×667≤2001`，signed13足够。可配置极值不借此界免检：32位EDIFF先判[-4096,4095]再截存，不满足便保持合法full候选。对Q1∈[-3,3]，每真z绝对值≤2592；部分残差修正后的acc等价于每rank分别取真z或αb，因此绝对值≤`2×8×32768×2592=1358954496`；完整signed3含−4也≤1811939328，均小于signed32上限。base加倍及逐rank修正不需要依靠溢出后抵消。残差×系数亦在32位内；下游signed64仿射与唯一RNE26/sat24保持原定义。

| 真实八块完整冷命令 | full 0 | 强Δ/anchor 1 | 新增缩放 2 |
|---|---:|---:|---:|
| 无背压总周期 | 134140 | 135102 | 136160 |
| 有背压总周期 | 144652 | 145612 | 146667 |
| encoder拍 | 0 | 1226 | 2284 |
| 在线候选次数 | 0 | 293 | 822 |
| Q2向量MAC | 17604 | 17292 | 17292 |
| anchor读/写 | 0/0 | 24/24 | 24/24 |
| prev/anchor选择 | 0/0 | 18/2 | 18/2 |
| 新增−1/+2/−2选择 | 0/0/0 | 0/0/0 | 0/0/0 |

无背压mode1比full慢962拍，准确闭合为`1226−312+24+24`。mode2比mode1慢1058拍，准确等于`(822−293)×2`，其他实际工作相同；暖无背压119356/120318/121376也保留同差分。背压由命令相对波形驱动，周期相位改变后差额不必等于1058，不能把两种条件混比。自然数据没有一次新增缩放被选中；direction_alphabet/direction_residual两个合法原生源fixture各使−1/+2/−2分别选8次，后者还保留非零rank残差。这证明分支实际执行，不证明真实网络存在该分布。

**新颖性差分。** 三模式把“改变分解单位到T10”做成了完整可审实现，优于只报告共线统计；但既有乘积复用、有限参考残差、符号与二倍移位仍属于A。当前新增α在真实八块零命中，连相对普通Δ的局部执行增量也未形成。该固定anchor/字母表停止为性能主线，不能因定向向量获益而抬成论文X，也不据此否定其他连续分解单位。这里没有新质量函数、训练、825、全帧RTL或PPA；已有相同整数函数的质量只能标继承，不称本项重测。

最终覆盖与收口：作者[results.json](results.json)为19fixture×3模式×2背压×2次命令=228条，rawp/J/I24各875520值全部通过；本审阅只交叉核对收据与生成公式，不称第二次独立RTL运行。19项包括原16基础fixture、两个方向向量及range_guard。direction_residual在t2/3/4各加一个rank0真源贡献，对应+2/−1/−2三种新增α下的非零残差，均实际选择并保持gold；其余α零残差情况同样覆盖。range_guard用合法全3 Q1，anchor全源得到2592，下一T仅1个Cin得到27，+2/−2候选差分别为−5157/+5211，确实超signed13；mode2在四个P分别拒绝这两项，每个背压/重启条件8次，四条命令共32次，随后full仍输出正确。固定实值Q1的≤2001域内不会触发该拒绝，但可配置域的防护已实测，不再列为未覆盖。

首命令实际装1848静态词，第二命令无reset只复用常量，真实source/origin/identity重新输入；TB检查参数、源、identity请求及受压结果的稳定性，并在全部480个raw/J/I24字及最后tile退休后结束。这里每次go仅一个tile，没有64tile/full或无reset换mode覆盖。最终算法在必要prev acc直通与lazy anchor补强后收口，无未关闭的阻断发现；保留上述物理映射与范围限制。
