# r0 大卷积的表示与分解：从强 A 到两个可检验 X

2026-09-13；范围：当前 matched-dense stage320 的 `patch_embed.residual_encoding.resblocks.0.conv1.0 / conv2.0`。本轮只有 primary 阅读、代数推导与设计定位，没有新增目标数据探针、训练、RTL、EDA 或硬件性能结果。已精读 **10 篇**直接相关 primary 的方法/实现段，含 2024、2025、2026 和老先验；阅读边界、发表身份及代码状态逐项保存在 [source_table.csv](source_table.csv) 与 [source_table.json](source_table.json)。精读不表示每篇全文通读或完整复现。

第一候选是 **D2「WINS 消费者掩码与二值 Winograd 幅值类的联合请求流」**，D1「源索引到有限整数桶」和 D3「Kronecker 支撑分层收缩」作为两个挑战者，D4 精确支撑表作为覆盖附录。建议先审查 D2 的同函数边界与完整强 A，再决定是否进入其小范围 RTL。它们的共同目的，是在已经可用的强分解上消掉连续中间量及其必要访问。这些目前都是待反驳的 X 假说；不能据本报告称为已证新颖或硬件胜出。D3 Kronecker 和 D4 精确支撑表仍保留，并指出怎样试，未因旧负结果或未得全文关闭家族。

## 1. 冻结的当前事实与分母

先读指定的 [profile.json](../../open_fusion_execution/major_operator_fusions_20260913/root_owned/profile.json)、[current_bottlenecks.md](../../open_fusion_execution/major_operator_fusions_20260913/root_owned/current_bottlenecks.md)、[既有 decomposition README](../../open_fusion_execution/major_operator_fusions_20260913/decomposition_owned/README.md)，再读 catalog/WORKS 与 COVERAGE 相关项。profile 的首两行实际观测分别为 3,214,695 / 73,728,000、2,663,570 / 73,728,000 非零，约 **4.3602%、3.6127%**。两层形状均为 T10×96×240×320，核为 96×96×3×3；每层名义 63.701 G MAC，占该次 596.546 G ATen 稠密算术 extent 的 10.678%。padding 零项、访存、归约、BN、布局均使这个分母不能充当周期或能效。

写成一个输出位置的真实算子：

\[
y_{o,t,p}=b_o+\sum_{k=1}^{864}\bar W_{o,k}s_{k,t,p},\qquad s\in\{0,1\},\quad\bar W_{o,k}=W_{o,k}\theta_k .
\]

这里 k 是输入通道与 3×3 tap 的联合索引，静态 θ 可吸入 W。原算子非零项是常量加权加法（AAC），没有逐事件 θ 乘法机会。按均匀发放率仅作规模估计，原 r0.conv2 每位置约有 864p≈31.2 个源事件，每个扇出 96 个输出；这不是边界修正后的实测 AAC。

已有同协议 diverse10：parent=1.1597366283，NB0=1.4546028611；纯 SVD R8=1.3479650409，activation-SVD R8=1.4030194175，spatial R16=1.2691297866，Tucker R8=1.4170152223，均优于 NB0。后续不能为使误差更小而默认再加 50% 稀疏残差，也不能用局部 L2 杀掉这些纯分解。当前 S2 FFN FC1 已改造，本报告不使用旧 FFN 份额。

既有 K4 selected-pair 与 H8 共同 pair-ID 已做数值/十帧/有限服务模型：H8 pair 仍比同次 dense 多 24.75% 服务。它约束的是该特定计数/metadata/遍历布局；D1 改变整 K 的共享映射与连续因子接口，不能直接继承该负结论，也不能忽略其中已经付出的共同 walker 税。

## 2. 十篇 primary 的实际所得

以下只概括直接支撑本轮边界的段落，代码未运行。详细链接与范围见来源表。

| ID | primary 与实际阅读 | 本轮不能重复申领的 A / 对 X 的约束 |
|---|---|---|
| S01 | [SmartExchange，ISCA 2020](https://arxiv.org/pdf/2005.03403)，PDF pp.3–7，§III–IV | W≈CeB，稀疏 PoT 系数、投影/拟合交替、向量稀疏、PE 邻近重建器、基底驻留、索引跳过、Booth bit-skip 均已有。原硬件主要先重建权重再算；不能将任何有限系数因子化称为新方法。 |
| S02 | [StrassenNets，ICML 2018](https://proceedings.mlr.press/v80/tschannen18a/tschannen18a.pdf)，PDF pp.3–5，§2.1–2.4 | y=Q[α⊙(Ps)]，P、Q 可三值；卷积版包括分组与 p×p 空间共享，尺度可预折。已经覆盖三值前变换、小连续尺度、三值后变换。作者代码 [mitscha/strassennets](https://github.com/mitscha/strassennets) 可访问，旧 MXNet 工件未迁。 |
| S03 | [GKPD，AAAI 2022](https://cdn.aaai.org/ojs/19958/19958-13-23971-1-2-20220628.pdf)，PDF pp.2–4 / 印刷 pp.772–774，Method、Alg.1 | 多项 Kronecker 和、重排后最优 Frobenius 截断、Conv3d→batched Conv2d 免核重建已有。压参数不保证 latent 稀疏。指定作者实现未定位。 |
| S04 | [Fast Algorithms for CNNs，CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)，PDF pp.2–3 / 印刷 pp.4014–4015，§4.1、Eq.5–13、Alg.1 | F(2,3) 的 B/G/A、16 点计算、先在通道上归约再逆变换、变换坐标矩阵化均是底座；32 次输入加法与 24 次输出加法不能隐去。 |
| S05 | [WINS，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_WINS_Winograd_Structured_Pruning_for_Fast_Winograd_Convolution_ICCV_2025_paper.pdf)，PDF pp.2–6 / 印刷 pp.22478–22482，§2、4、5 | WINS-R/C 是各变换坐标矩阵的整行/整列剪枝；WINS-B 使 16 个矩阵的剪枝率相等；AB 逐层选择并用短适应 BN 评估。不能把这些当本网新机制。catalog W0094 已存在但未迁；作者代码未定位。 |
| S06 | [FastKron，PPoPP 2024](https://arxiv.org/pdf/2401.10187)，PDF pp.3–6，§3、4.1–4.2 | sliced multiplication 直接写正确次序、shift caching 降 bank 冲突、多因子留 shared memory 融合已有。必须给 Kronecker 强控制这些权限；作者 [FastKron](https://github.com/abhijangda/fastkron) 可访问。 |
| S07 | [LUT-GEMM，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/a4f98ce85f440ee269b0df57b4368719-Paper-Conference.pdf)，PDF pp.4–6，§2.3、3.1–3.4 | 多二值权重基＋尺度/偏置，运行期输入组合表、组尺度、GPU shared-memory 分块已有。其大 M 摊销不能直接迁给 O=96。作者 [lut-gemm](https://github.com/naver-aics/lut-gemm) 可访问。 |
| S08 | [LUT-DLA，HPCA 2025](https://arxiv.org/pdf/2501.10658)，PDF pp.2–7，§II–V | 输入 VQ、CCM 质心查找与 IMM 查询解耦、LUT-stationary、索引复用、分阶段 LUTBoost、L1/Chebyshev 距离均已有。二值源直接地址可免质心距离，但这种特化也不足单独申领 X。论文说 Chisel generator，公开作者仓库未定位。 |
| S09 | [Platinum，ASP-DAC 2026](https://arxiv.org/html/2511.21910v1)，§II–IV、V-A；发表身份由[作者机构](https://scholars.duke.edu/publication/1908342)核对 | 离线 MST 构造路径、四段流水、正负镜像、五个三值权重压到一字节、bit-serial/ternary 路径切换、8列并行查询与额外归约加法器均已有。原始 RTL 说已综合，不等于已公开；代码未定位。 |
| S10 | [UCNN，ISCA 2018](https://www.kartikhegde.net/media/UCNN_ISCA.pdf)，PDF pp.3–8，§III–IV | 先按共同权重求 activation group sum、跨滤波器层级交集、预排序 iiT/wiT、空组 skip、空间 bank 映射已有；组长16控制位宽，跨输出共用目录会削弱独有零权重跳过。不能将分桶或先加后乘称为新 X。 |

另找到 [FINEA，ICCD 2025，pp.540–548](https://pure.korea.ac.kr/en/publications/finea-an-efficient-neural-network-accelerator-exploiting-factoriz/)：**仅阅读作者机构摘要及元数据**。摘要已明确“重复权重下的 factorized/unfactorized 双执行”和预处理权重表指引特征 gather，因此是 D1 的近期碰撞；未定位全文，不能据摘要认定它已经实现 D1 的特定全输出共享码流，也不能据此关闭 D1。下载但未精读的 HybridNet、OpenReview PDF 抓取失败的 Kronecker-sparse 不计入上面十篇。

## 3. D2：WINS 消费者掩码与二值 Winograd 幅值类的联合请求流

**A：**完整 WINS-R / WINS-C / WINS-B，加标准数字 Winograd 的常量变换、预折核、同通道归约后逆变换、零跳过、融合和存储复用。不能用无优化 dense Winograd 作分母。

**B：**原 r0 的 W 稠密但 s 二值且很稀疏；普通 Winograd 以 dense multiplication reduction 解释的2.25×并不适用。直接采用高位 V 与16个坐标GEMM还会破坏 event AAC 优势，而 WINS 剪枝后的真实输入/输出重排与变换量存储尚未在本网支付。

对 F(2×2,3×3)，取标准

\[
B^T=\begin{bmatrix}1&0&-1&0\\0&1&1&0\\0&-1&1&0\\0&1&0&-1\end{bmatrix},\quad
A^T=\begin{bmatrix}1&1&1&0\\0&1&-1&-1\end{bmatrix}.
\]

对每输入通道的4×4二值块 S：\(V=B^TSB\)，\(U=G\bar WG^T\)，

\[
M_{o,\xi}=\sum_c m_{o,c,\xi}U_{o,c,\xi}V_{c,\xi},\quad Y_o=A^TM_oA.
\]

WINS-R 的 m 只随 (o,ξ)；WINS-C 只随 (c,ξ)。这是变换坐标下的行列 mask，不是普通空间卷积 filter/channel 删除。

**本轮代数推导：**每个 V 坐标恰为四个 S 位的±和；**15 个坐标只取 {-2,-1,0,1,2}，唯一双和坐标 ξ=(1,1) 取 {0,1,2,3,4}**。因此对预折 U，所有非零坐标仅需一次带符号/移位的常量累加，唯有双和坐标值3需要 U+(U≪1)。这比“变换后可量化”为更严格的有限集合合同，且对原二值 S 无需量化或校准误差。根据root授权，本轮另用标准库脚本对 **65,536个二值4×4输入全部穷举**，16个坐标均匹配上述集合；正3仅出现于零基坐标(1,1)，没有负3。见 [可复跑数学核验](verify_winograd_alphabet.py) 与 [全部坐标集合/频数](winograd_binary_exhaustive.json)。它是公式核验，不是目标网络实验、RTL或性能结果。

**待检验 X：**从原16位 S 直接产生有效/符号/×2/×4/双和=3码，并在发起 U 的物理读取前，与当前输出块的 WINS mask 相交。执行流的载荷是有限幅值类和消费者mask，不生成连续 V 张量，也不发出已被 m 或精确整数抵消消掉的 U 请求。对15个差分坐标用共同幅值语义，唯一双和坐标采用单独小路径，避免为不可能出现的幅值配置通用乘法/解码。**仅靠 generic range analysis 就能得到的位宽缩窄、普通输入/逆变换融合、双模式切换均不独立算 X**；拟申领增量必须是联合原生码流比完整有相同范围证明的 WINS执行确实少了物理中间访问/服务。

**变换税和稀疏损失：**核由每对通道9项变成16项，即未剪枝状态1.778倍；精确有理核还需额外 guard/fraction bits。V 中一个原事件会触发多个坐标，原发放率不能继续乘在V后；输入变换、邻tile重叠 gather、m读取、packed码生产/回放和输出逆变换均收费。原T10非因果 PSN不允许提前按因果神经元完成，必须输出全部T10的原接口值。

在仅用于直觉的 iid Bernoulli(p) 模型下，差分坐标非零概率为

\[
q_d=1-[(1-p)^4+4p^2(1-p)^2+p^4],\quad q_s=1-(1-p)^4,
\]
\[
\mathbb E[\text{Winograd weighted-add terms/input-output-channel tile}]
=15q_d+q_s+4p^3(1-p).
\]

末项是双和=3多付的一次加法；原直接卷积是36p。p≈0.036时前者约2.1、后者约1.3：**无剪枝 Winograd 在加权项上反而吃亏**。因此 D2 需要足够的 WINS 删除、真实局部聚集/抵消或端口优势；这不是本网目标数据测量，更不是性能预测。逆变换和所有来源税会进一步提高 break-even 所需剪枝率。

**一个不能省略的同函数问题：**任意掩掉16维 U 的分量后，Û 通常不再位于 \(GgG^T\) 的9维空间核像空间。因此不能把剪枝Û“反变换成普通3×3核”当严格同函数对照。正确控制是展开

\[
\widehat K_{o,c}=(A^T\otimes A^T)\operatorname{diag}(\widehat U_{o,c})(B^T\otimes B^T),
\]

按一致 vec 顺序得到4×16 tile线性算子，保持输出tile相位、重叠、padding。其数学上完全等价的事件 scatter 是同函数 direct 控制；原3×3 event-conv 是质量对照。若坚持原3×3卷积函数族，需要约束Û始终可由GgGᵀ生成，和一般 WINS 不是同一个模型。

**首 RTL 范围：**C8、O8、T10的一个4×4输入tile及相邻tile重叠输入，固定一份真实 WINS-R/C mask；原始bit gather→幅值类/消费者码→实际U ROM请求→有界移位AAC→通道归约→A逆变换，交付2×2×O8×T10完整值。可先用逐坐标归约+4输出累加器减少M状态，但同样给普通WINS；不能把该普通fusion计为X。必须测全零、单事件、四位相等导致抵消、双和=3、边界padding、被剪坐标、回压与码流栈深。

**无损/有损界：**二值编码对固定Û可数学无损；WINS剪枝相对原卷积有损。每tile误差为
\(\Delta Y_o=A^T[\sum_c E_{o,c}\odot V_c]A\)，从而
\(\|\Delta Y_o\|_F\le\|A\|_2^2\sum_c\|E_{o,c}\odot V_c\|_F\)。重新舍入Û、动态BN/PSN门翻转的影响另评；不能继承 WINS 的 ImageNet结果。

**最强控制与最近碰撞：**标准位串行WINS＋完整零skip＋range-specialized变换＋相同码缓存是必须的强 A；再比相同Û的4×16 direct和同质量原卷积/纯分解。WINS、CVPR2016的通道归约和通用稀疏Winograd是明确碰撞；搜索还发现二值Winograd相关学位工作和量化Winograd硬件，但未精读正文，故不作“没人做过”的证据。若只省掉全宽V，给A同样有限域编码后优势消失，则该X退化为普通特化；若联合请求仍少物理字/较短完成链，才值得后续RTL继续。

**独立审阅后的补强：**[sparse审阅](../sparse/review_decomposition_d2.md)指出的32B字反例成立：若一字含同ξ的8个Q32系数，任一消费者仍活跃就必须读整字；若Q16把两ξ共装一字，一个ξ抵消也未必省字。紧密打包必须支付descriptor与散写，且给强A相同布局。值3的控制也可以预存3U或按需生成后缓存，付出容量/填入税；本报告“两加”仅指不额外存常量的直接展开，不是不可突破下界。C8/O8首RTL只验证机制；任何性能晋级必须至少扩到全C96、T10与完整空间tile的全部N96归约，再支付r0 norm2＋identity连续后继的真实桥接。当前X仍可能退化为普通范围特化后的稀疏地址使能，独立审阅没有给出PASS。

## 4. D1：源索引到有限整数桶的分解请求流

**A：**采用完整离散系数矩阵分解、StrassenNets 训练/尺度权限与 UCNN/SmartExchange 的索引、重建、驻留、bit-skip；另给纯 SVD8 / spatial16 / Tucker8 同宽度、同资源的完整实现。

**B：**当前纯分解通过质量门，但 z=Vs 从稀疏脉冲变成连续值，后因子 U 的 96×R 连续 MAC 及 z 存取仍存在。旧 pair 只在 K4 的局部两个等权坐标节省几个 AAC，并支付输出私有或 H8 ID 查询。

**真实表示：**先研究无残差的列稀疏离散因子，避免再退回 50% direct 残差：

\[
\widehat W=C D,\quad D_{j,k}\in\{-1,0,1\},\quad\|D_{:,k}\|_0\le d,\quad\|D_{j,:}\|_0\le L,
\]
\[
n_{j,t,p}=\sum_{k\in\mathrm{supp}(s_{:,t,p})}D_{j,k},\quad |n_j|\le L,\qquad\widehat y_o=b_o+\sum_j C_{o,j}n_j.
\]

一种更强 A 是 \(C=Q\operatorname{diag}(\alpha)\)，Q 三值，直接成为 StrassenNets 的形式。第一版不要强加该额外精度限制；连续 C 与三值 Q 两条链都要保留。d=1 为有符号列原型聚类，d>1 为少数原型叠加。与普通 U/V 的差异是前因子把源变成**有界整数**；不声称整数×全宽 C 完全免费，只把连续×连续 MAC 换成受位宽约束的移位/加法或已收费的小表。

**待检验 X：**离散表示和执行合同共同限制“每个源只给 d 个共享桶发令牌”，D 对全部 96 个输出共享；源事件先驱动桶，而不是逐输出扫描 group 目录。桶完成时直接交付全 T10 的非零/符号/幅值码，最终 n=0 的桶不产生 C 请求。拟合目标除输出误差外，必须显式约束实际源→桶 fanout、同物理 C 字的请求并集、计数位宽与完成边界。**新贡献不包括读 C 一次广播、普通缓存、整数计数和零跳过**；若完整 Strassen/UCNN 的同码布局已做到同样服务，D1 的 X 就没有增量。

可被减掉的是：全宽 z RF、无效桶的 C 字访问、后因子一般乘法器需求；增加的是 D 源索引、d 倍事件分派、计数清零/读改写、桶完成标志、计数幅值解码及 C×小整数展开。共享 D 不能随输出块另拟合，否则一次源扫描会变为 12 个 H8 私有扫描，需重新记账。

**首 RTL 范围（未来，仅规范）：**一个 K64 源块，全部输出共享的 R16、d≤2、每桶 L≤16；T10；消费端先做 H8，其余 H8 用同一计数块回放。输入口为真实位字+有效位，D ROM 保存两个 `(bucket,sign,valid)`；n 用有符号6位，16×10×6=960 bit，对照同 R 的 I24 latent 是3840 bit。两者都必须把最终完整输出交给原动态 BN/PSN，后继输出存储不能扣除。C ROM 128个系数/当前H8；负载/回压、计数bank冲突、全零K块、桶到达完成均纳入。对比移位加法与小表时，小表绝非零面积：若存每系数 |n|=1…16 的24位值，该 H8×R16 表为6144 B，原 C 仅384 B。

**无损/有损界：**D/C 拟合原 W 通常有损，不能继承 SVD8 的 AEE。若 E=\(\bar W-CD\)，单样本 \(|e_o|\le\sum_{k:s_k=1}|E_{o,k}|\)；总体二阶目标可写 \(\operatorname{tr}(E\Sigma_s E^T)\)，Σ 必须来自授权校准数据。执行对固定 CD 的数学映射可无损；定点位宽、舍入及累加顺序需独立合同。

**最强控制与区分预测：**C/D 与同函数普通事件驱动稀疏因子、UCNN层级目录、Strassen直接三值卷积、按源排序离散因子全部同权；同时用已过 NB0 的纯低秩质量/服务曲线。D1 若成立，收益应随“源→桶 fanout 下降、完整桶抵消且未请求 C”的比例变化，而不是仅随 C 缓存命中变化。把桶码生成改回逐输出私有 D 的消融，应明显放大 metadata/计数费用；若没有，所谓共享表示不是机制。普通同布局给足后无服务净减，即可否定**这个 X**，仍保留离散分解作为 A。

## 5. D3：Kronecker 的支撑分层收缩——值得补试的 A，不急于另立新标题

把 o=(a,b)、c=(u,v)，其中96=12×8。一个明确可执行展开为

\[
\widehat W_{(a,b),(u,v),\delta}=\sum_{r=1}^{R_K}A_{r,a,u}B_{r,b,v,\delta},
\quad z_{r,u,b}=\sum_{v,\delta}B_{r,b,v,\delta}s_{u,v,\delta},
\quad y_{a,b}=\sum_{r,u}A_{r,a,u}z_{r,u,b}.
\]

**A/B/X：**GKPD已经有核分解和先内层再外层收缩，FastKron已经有次序/融合/银行优化；B是只含事件的u切片可能较少，但普通实现仍分配全部连续z。候选X是按零/单事件/多事件三种源支持交付 tagged latent：zero不分配，singleton交付B列引用，multi才物化z。这个tagged接口不是现成硬件新颖性结论；它与按需权重重建/压缩指针以及稀疏tensor compiler高度接近。

若一次跨v×9收缩，则每u有72个源位，独立p模型下非空概率1−(1−p)^72约0.93，说明“原p很小所以u很稀疏”是不成立的。可改为逐tap的8通道slice，非空概率1−(1−p)^8约0.25；但这会增加9倍latent slice与外层归约，必须实算。单事件引用若在消费时仍需A×B一般乘法，或需要缓存整重构W，就不自动省算；同值dense W缓存也有同等权限。

**删除/税：**只可能删空slice状态和部分中间写回；增加source支持分类、tag、B列地址、A/B重建cache、多term归约与continuous multi-slice MAC。没有从稀疏s到稀疏z的普遍保证。

**首RTL范围：**先固定一个12×8通道拆分、R_K=2、C96/O96的一位置T10；H8输出逐块消费，同一小RF中比较两收缩次序、逐tap和跨tap，支付全部tag/cache端口。先不将TT/TR一并扫入。**误差：**对固定Kronecker核，收缩可数学无损；截断核仍有EΣEᵀ型有损界。**强控制：**相同核的FastKron式融合顺序、稀疏序列收缩、按源事件重构W、已过NB0的纯分解。若zero/singleton潜在好处不足支付tag，即否定此布局，不能否定整个Kronecker族。

## 6. D4：精确局部支撑表——保存覆盖，当前普通先验碰撞最直接

对K分成大小g的组，离线构造
\[
L_{o,h}[b]=\sum_{i=0}^{g-1}\bar W_{o,hg+i}b_i,\quad
y_o=b_o+\sum_h L_{o,h}[s_h],\quad b\in\{0,1\}^g.
\]

**A/B/X：**直接二值地址的离线表、差分构造、共享decision DAG、镜像与表驻留均属于LUT-GEMM/Platinum/UCNN等强先验。B是g小使稀疏事件碰撞少，g大使表指数膨胀且多个输出/空间tap要不同表。当前可检验X仅保留“源支撑码与多个真实消费者绑定成原子向量请求，再按有限存储选择部分表项/例外”，必须在普通完整表/partial table/共享cache给予同权后还显示净减；否则没有新X。

以g=4、K864、O96为例，原完整核每系数b_w位是864×96×b_w；全表含零项变成216×16×96×b_w，即4倍，删零表项仍3.75倍。iid低p下非零组数为216[1−(1−p)^4]，每组原事件数期望4p；组内合并能省的加权累加只占低阶p²项。不可将“16种可查”当作必然收益。

**首RTL范围：**g=4、H8、单tap四通道源位、T10，1R小表与同容量常量W表，实际组码生成、全零skip、查表/事件分派、psum同宽端口；然后才扩g8。**删除/税：**合并同组多个事件的宽AAC，换取表容量、字读取与寻址；没有生成连续低秩z，但表输出仍是连续psum。**无损/有损：**完整表对既定定点W可无损；仅保留部分表并走exact event fallback也可无损；用近似码字则需独立量化/模型误差。**强控制：**完整Platinum路径/MST的可适配范围、exact sparse AAC、普通多输出共享表、LUT-DLA的有损VQ版本。旧K4 pair的负结果只约束一种近似重建和服务布局，不替代这些范围。

## 7. 进入实现前的具体交接

1. **先定函数。**D1导出C/D与整数界；D2导出Û/m/B/A与4×16同函数直接算子；D3导出A/B和具体收缩次序。只拟合/训练需要的候选，不因rank8已通过就假定其它表示通过。
2. **先给完整强A。**纯SVD8、spatial16、Tucker8、原bit-skip AAC、普通结构稀疏均按相同precision/storage/ports；D2还给完整WINS-R/B/C而非故意弱化Winograd。所有公共fusion、bank优化、cache、编译范围证明同权。
3. **再逐个RTL。**建议D1和D2各自先封闭一个上述小tile的源到输出合同；每个候选只做一组已选函数和一套有限资源，不把十几个rank与端口扫参伪装成新idea。首测记录：源字、metadata字、系数字、wide/narrow add、连续MAC、状态峰值、bank冲突、stall和最终完整T10输出时间。
4. **质量与硬件分开。**全网同协议优于NB0即可作为质量门；diverse10只作探索，不写valid825。硬件先看相同函数/资源的物理事务，再看同质量的表示比较；当多项假说均未通过，不把最小坏值称为PASS。

## 8. 检索边界与过程说明

检索日2026-09-13。使用公开网页检索、arXiv、CVF、PMLR、AAAI、HPCA官方节目及作者机构/仓库；没有外发本网权重、输入或结果文件。检索串覆盖 SmartExchange/StrassenNets、Kronecker CNN/GKPD/FastKron、Winograd binary/sparse/WINS、LUT-DLA/LUT-GEMM/Platinum、UCNN/FINEA，以及论文名+github/code。它是有界的直接相关 primary 研究，不是对全部775条catalog重新全文阅读，不是系统综述，也不能证明没有更近先验。发表年份以正式venue核查，Platinum从catalog的2025预印本身份更新为作者机构所列ASP-DAC2026；原catalog未改。

采用 [hypothesis-generation SKILL.md](/home/zhumd/.agents/skills/hypothesis-generation/SKILL.md) 区分观察、假说、竞争解释和区分预测。过程资源引用：Timothy Kassis、Vinayak Agarwal、Yuhuan He、Darshil Patel、Aubrey M. Brueckner，2026，[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065)；本轮核到最新v2（2026-09-02），仅作过程工具说明，不计十篇技术primary。

### 工作文末保留的检索前四假说

这些先在 [independent_hypotheses.md](independent_hypotheses.md) 写出，已经受项目历史资料启发，**不伪称盲法**：①列原型→整数事件直方图；②事件条件Kronecker切片收缩；③二值支撑驱动精确Winograd双路径；④局部支撑decision-DAG/差分表多输出分解。检索后①收窄为D1的共享源映射与原生请求合同；③去掉“双路径”作为新意，转成D2的WINSmask×有限幅值类；②、④保留为待补强A/待测接口，未包装成已成熟X。
