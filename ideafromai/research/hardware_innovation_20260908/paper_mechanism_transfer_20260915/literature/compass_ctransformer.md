# COMPASS、C-Transformer：已核机制与本图可迁接口

2026-09-15。仅文献、公开代码和现有本地合同审阅；未训练、运行架构评估、EDA或修改生产代码。本文的三个候选是待验证接口，不是三个新标题，也不是已获得的性能结果。

最重要的区分：**C-Transformer 的 OSS 用采样概率重新生成后续脉冲，不能保证逐位等于原脉冲；BiLD 的 rollback 是另一层的算法，并不能据此声称 OSS 会验证并恢复。NeRN 是需要训练的参数表示，公开评估先还原全模型，不能直接当作免费、即时的权重供数器。** COMPASS 的身份已核正确，但本轮仍未获得其正文，不能给它编造逐节机制。

## 1. 身份与材料边界

| 对象 | 已确认身份 | 本轮实际读到 | 尚缺 |
|---|---|---|---|
| COMPASS | Zongwu Wang、Fangxin Liu、Ning Yang、Shiyuan Huang、Haomin Li、Li Jiang；**MICRO 2024，pp.1090–1106，DOI 10.1109/MICRO61859.2024.00083**；题名确为 *SRAM-Based Computing-in-Memory SNN Accelerator with Adaptive Spike Speculation* | [会议官网 Session 7C](https://microarch.org/micro57/program/)、[作者主页](https://mxhx7199.github.io/publications/)、[作者 AE](https://github.com/ZongwuWang/COMPASS_AE) 的 README、配置、批处理、hook、历史 update 和一份发布日志；出版社注册参考文献 | IEEE正文/PDF未获取；作者主页 Paper 链接实际为 `#`。本轮不能确认各节标题、最终验证/恢复算法或全部硬件状态 |
| C-Transformer 短文 | Sangyeob Kim 等六人；**ISSCC 2024，20.5，pp.368–370，DOI 10.1109/ISSCC49657.2024.10454330** | 三页论文正文、完整 Fig.20.5.1–7及续页参考文献；[本地三页](ctransformer_isscc2024_full.pdf)，来自[公开会议文集副本](https://iccircle.com/static/upload/img20240529102116.pdf)，由[作者项目页](https://ssl.kaist.ac.kr/bbs/board.php?bo_table=Neuromorphic&wr_id=4)核身份；[机制图](ctransformer_figures.png)已目视阅读 | 未找到这颗芯片的公开 RTL/软件仓库；不能把引用的 BiLD/NeRN 仓库当芯片源码 |
| C-Transformer 长文 | *An Energy-Efficient Homogeneous DNN-Transformer/SNN-Transformer Processor for Large Language Models*；**JSSC 2025，60(10):3802–3815，DOI 10.1109/JSSC.2025.3554699** | [作者单位摘要](https://pure.kaist.ac.kr/en/publications/c-transformer-an-energy-efficient-homogeneous-dnn-transformersnn-/)、[实验室论文条目](https://ssl.kaist.ac.kr/bbs/board.php?bo_table=Journal&wr_id=251)、出版社注册的22条参考文献 | 长文全文未获取；实验室条目的 File 栏为空。以下电路细节只归 ISSCC 短文，不能冒称读完 JSSC |

因此旧 catalog 的 COMPASS 身份不是错误；真正需要纠正的是“身份待核”到“官网已核、正文仍缺”。不要混入 DATE 2025 同名的 crossbar compiler、OSDI 2025 encrypted-search COMPASS。IEEE常规 document/PDF入口本轮返回拦截页，未把它保存为论文。注册数据保存于 [COMPASS](compass_crossref.json)、[ISSCC](ctransformer_isscc_crossref.json)、[JSSC](ctransformer_jssc_crossref.json)。

版本数字也必须分开：ISSCC正文/实验室项目页报 EMA 能量为 baseline 的0.24–0.29、GPT-2延迟477 ms；JSSC摘要分别为0.37–0.41、656 ms。它们不是同一版结果，不能拼取更好值。均不是本28 nm数字实现的预测。

## 2. C-Transformer：按短文段落与图还原“发现问题→改接口”

**问题发现，p.368 引言、Fig.1。** 作者先定位大模型的外存参数访问，而不是先认定“有SNN所以乘法可删”：所用LLM的EMA占总功耗68%。稀疏化在不同语言任务的质量约束下不能同样激进；这是一组工作负载观察，不是“ANN无法高稀疏”的普遍定理。进一步加入 little model 后，小/大数值分布更动态，原 C-DNN 独立 CNN/SNN 核会出现利用率波动。因此本文同时处理**模型供数**和**两数域资源利用**，单纯套已有 SNN AC 核没有解决两者。

**模型选择和参数表示，Fig.1、5。** 原GPT-2加约1/10规模little model；先执行little，用其token预测概率阈值决定是否调用big。正文将39–59%的EMA减少归于big-little路径。另训练一个 `MLP_IWG`，输入kernel位置的嵌入、输出原Transformer参数。它自己仍可能有284M参数，因此又压缩**生成器的权重**。这里有两层权重，不能把生成器参数和被生成Transformer参数混为一份，更不能把这些方法都说成无训练编码。

**真正共同算术，Fig.2、3。** 芯片含48个HDSC，每个8×16个HMAU，另有2个Weight Generator和INT16 outlier SIMD。一个HMAU把full-adder阵列的中间连接重配：DT模式传播两半carry，构成一次8b×8b乘法；ST模式切开carry并重排权重位，形成8路累加。非零输入通道被装载器重新分配到可用行。作者报告9.1%的面积开销；它本身就否定“复用现有乘法器因此没有额外电路”。这不是复制完整两核并行，也不是我们的19×13乘法器自动具有同样模式。

**供数限制不是算术图之外的小事，Fig.3 WFL。** DT一次载8个权重、接着8拍逐个广播；ST同拍给8行不同权重。两模式瞬时需求相差8倍，WFL通过对齐和调度平衡周期，Aligner由同列HMAU共享。本数字接口若仍只有256bit单W口，必须给重排、预取、缓存填充和冲突实付，不能按“8个AC替1个MAC”直接写8倍吞吐。

**OSS，Fig.4。** 输入标量先以随机rate coding生成脉冲。PSU在一段采样TS内真正积分、阈值发放并计数；SU取得输出发放概率，再把RNG值与概率比较，合成剩余TS的输出。示例8/32步采样后省去24步IF。作者用WikiText评估采样比例和perplexity变化，并选平均40.7%采样比。**短文没有展示对每个合成脉冲进行精确验证、失败撤销及重放。** 保持发放概率不等于保持每个时间位；更不等于保持我们T10各位置的连续量、残差或动态BN统计。

**ESC与生成流水，Fig.5。** 常见8b小权重的高4b都是符号扩展；它们用低4b+符号存储，例外上半字另存并由索引定位。ESDU有LSB、MSB、Sign、IDX四类存储读入，随后IWGU执行MLP，生成的下一块权重进入HDSC。Fig.5明确有两个WG及权重/输出缓存，靠生成下一W与当前W计算重叠隐藏时间。ESC对固定8b字本身可无损；IWG拟合和OSS则需质量评估。不能把“ESC无损”扩展到整条链。

**评估范围，Fig.6、7。** 为28 nm实测芯片、GPT-2/mT5/T5/FSMT四任务，比较包含作者按DDR3估算的外存能量；不是全部对照都重新流片、同模型和同工艺。正文和表对“perplexity drop/loss”的正负用词也不够统一，本文不据该措辞推导质量改善。这里借接口，不借TOPS/W、面积比例、频率或任务延迟。

## 3. COMPASS：能核到的实际接口，以及不能越过的缺口

没有全文就不能回答“作者第几节如何证明恢复正确”。本轮以可读AE给出以下较窄事实；它比仅重复摘要前进了一步，但不是完整COMPASS复现。

| 已核材料 | 确定事实 | 不能据此推出 |
|---|---|---|
| [demo.cfg](compass_demo.cfg) | `column`粒度；8b权重；合并T100；128列；`sub_tw=16`；动态子窗口、推测、CSR是独立开关；hit/miss阈值2/1 | 默认配置不等于论文全部实验；阈值名字不等于已知命中判定方程 |
| [run_ablation.sh](compass_run_ablation.sh) | 比较PTB、Strawman、ComPASS，并分离动态窗口、speculative、CSR组合 | 不是“只有一个不公平的dense分母”；但也不能仅凭脚本证明同PPA、恢复计费正确 |
| [hooks.py:219–274](compass_hooks.py) | trace含真实W、展开的输入、输入稀疏/压缩比，**还含神经元输出CSV及输出稀疏/压缩比**；卷积展开为T×K×位置 | 输出trace可能用于性能验证/统计；未读核心之前，不能断言它已是无需oracle的输入→预测→验证电路，也不能反过来指控作者用了免费oracle |
| [模拟入口](compass_simulator_link.txt)及公开目录 | `PyNeuroSim/NeuroSim`指向`dist/NeuroSim/NeuroSim`；该目录为PyInstaller打包分发，含`cformula`二进制。所检查树未出现可读的最终SubArray/预测/恢复Python或RTL源 | README的“公开代码”不能被扩大成“最终核心源码已逐行审阅” |
| [update.md:278–364](compass_update.md) | 历史说明popcount、按列处理、recover费用、动态时间窗口、输入/输出trace处理的开发过程；提到早期以denseRatio估计恢复 | 早期TODO不能当作最终论文仍有的bug；历史实现也不能替代最终算法 |
| [发布日志样例](compass_ae_published_log.txt) | 2024-04-17导出的例子同时有`speculative=True`与`speculationRecover=False`，并单列popcount开销 | 名字不足以确定两个开关的最终语义，尤其不能借它宣布“有推测所以有精确恢复” |

AE要求神经网络trace和打包仿真环境；本轮没有启动它，也没有把论文PPA移植到我们的数字逻辑。当前准确缺项是：**预测输入实际可用时点、命中/失误的判定、验证依赖是否使用输出trace、恢复到哪一时间/膜状态、队列/端口/错误传播范围、最终费用模型。** 公开材料不足时，这些应空缺，而不是用C-Transformer的概率采样或BiLD的后缀回滚填进去。

## 4. 追到的非SNN主来源，以及一条强SNN控制

### BiLD：C-Transformer ISSCC [7] / JSSC [12]

[NeurIPS 2023全文](https://papers.nips.cc/paper_files/paper/2023/file/7b97adeafa1c51cf65263459ca9d0d7c-Paper-Conference.pdf)、[本地](ctransformer_bild.pdf)、[作者代码](https://github.com/kssteven418/BigLittleDecoder)。读了§1–5、算法1及相关实验；未逐项复现附录全部阈值实验。

- §3.1先做**机会探针**：每步都跑大小模型，用大模型输出替换少数预测，观察约20%干预是否保住质量。这一阶段明确使用理想信息；实际方法不能用这份oracle成本作结果。
- §1–2定位自回归每token加载W/KV、算术强度低；直接非自回归会损失条件依赖。§3.2不是主张FLOPs更少，而是小模型串行拟稿后，大模型一次处理多token以增加复用。
- §3.3以小模型**已经算出的**最大token概率触发fallback；§3.4用大模型整批已有logits检查已生成token，取最早超过差异阈值者并丢弃其依赖后缀。验证logits随大模型运行得到；重复计算仍需付费。
- §3.5.1可做prediction alignment训练，减少同义不同词导致的无效回滚；“plug-and-play”指不强制这一步，不是全实验从未训练。§4消融去掉fallback或rollback均变差。
- [T5BiLDModel](ctransformer_bild_modeling_t5.py:2026)实际截断大小模型的self-attention KV，保留不变的cross-attention KV；低置信小模型本拍结果被丢弃；[2128–2149行](ctransformer_bild_modeling_t5.py:2128)定位首个差异并在新token确实不同才回滚。**可回收状态与不可回收状态的区别，是比“大小网络”这个名字更值得借的接口。**

BiLD是质量—延迟折中，不是与大模型逐token或分布精确等价的证明。正文主表是T4；结论段仍出现Titan Xp及另一组旧数字，本文不混用。我们已完成非因果T10输入后不能再获得“用小模型消除自回归依赖”的同一收益；若借，应借**可延迟提交的任务、真实验证机会、失败重放范围**。

### NeRN：C-Transformer ISSCC [8] / JSSC [22]

[ICLR 2023作者全文](https://arxiv.org/pdf/2212.13554)、[本地](ctransformer_nern.pdf)、[作者代码](https://github.com/maorash/NeRN)。读了§1–6的方法、适用范围及主要消融。

- §3.1把`(layer,filter,input-channel)`坐标的正余弦嵌入映射成一个k×k卷积核，预测器是5层MLP。普通隐式图像表示依靠空间平滑，但网络中相邻kernel没有天然连续性，这是作者发现的表示障碍。
- §3.2联合权重重建、logit蒸馏、feature-map蒸馏损失；不需要任务标签并不等于不训练。§3.3以cosine距离的贪心路径排列kernel坐标，网络的**实际权重顺序保持不变**。cross-filter/in-filter的排列元数据分别有约4–6%/2–3%原权重存储开销，作者已经记账。
- §4.3 ImageNet只预测ResNet18的3×3卷积，跳过首7×7与downsample层；§4.4的data-free仍训练预测器并使用噪声输入；§4.6显示仅重建损失或仅蒸馏不等价。§5将再压缩预测器列为额外应用，不是隐含免费。
- [evaluate.py:23–37](ctransformer_nern_evaluate.py:23)是`predict_all→sample_weights_by_shapes→update_weights→eval`；[predictor.py](ctransformer_nern_predictor.py:139)按批生成后拼接。故原软件证明了参数表示/质量，**没有证明按消费者请求、固定端口、窄整数生成器的服务时间**。

这不是直接分解运算 `Wx=A(Bx)`：NeRN先生成W，再执行Wx。若要只保一小块W，生成器MAC、positional embedding、逆排列、缓存与重生成费都会进入关键路径。现r0/R8静态W已驻留，NeRN不能省一份原本不存在的逐tile外存加载。公开预训练文件是ResNet等任务，不是本SDformer因子，不能说现成可跑。

### PTB：COMPASS [24] 的参考文献路径

出版社注册确认为 *Parallel Time Batching: Systolic-Array Acceleration of Sparse Spiking Neural Computation*，HPCA 2022，DOI10.1109/HPCA53966.2022.00031。[作者公开稿](https://web.ece.ucsb.edu/~lip/publications/SparseSNNAccelerationIEEE-MICRO-Submitted2021.pdf)带MICRO 2021投稿水印；此处只读其问题与PTB/StSAP数据流，不把它的数值当最终HPCA版。它已明确以多时间窗口重用W、合并不重叠活动，改善稀疏阵列空转。因此数字T10整词、共享W、时间并行本身已有强A；本轮借COMPASS不得再用逐tick反复取W的弱分母证明新意。

## 5. 到本28 nm数字接口的限制

| 当前合同 | 对借入机制的具体影响 |
|---|---|
| AT-LIF输出为`θ·g`，θ静态可折入线性W | 二值门支路已经能走AC；不要把一般动态幅值MAC当强分母。折权后的量化尺度、RNE、饱和需定义，不能凭实数结合律跨已有整数边界 |
| T10为可非因果混合的完整窗口 | 不能把BiLD因果后缀恢复直接改名为T10 rollback；时间错误可能影响全部输出T。可取消的最小单元需按真实依赖闭包求得 |
| 同一生产值可能既生成门，又被残差/PED完整值消费 | 概率正确/门确定只满足一个消费者；不得丢失连续量，也不得在不可撤销I24/RNE/saturation之后声称补差恢复。回收必须等待全部义务完成 |
| 相关BN/τ可依赖当前全域Y | 阈值在统计完成前并非已知常量。需要先付统计/屏障，或另做网络质量验证的冻结BN；静态θ折权不意味着τ也静态 |
| 共同8×32 ALU、8×19×13乘法、W256、Z/psum单服务仲裁 | 多放一个WG/验证器就是额外算力，不能引用C-Transformer“双WG隐藏全部时间”。借现有闲置执行槽也需两臂同权且记录仲裁 |
| 已有粗头及删末级分支的质量/收益 | 新detail控制的分母是**同一个既有coarse+完整detail**；只能计本次取消的detail工作，不能再累计旧删支路收益 |

本地边界来自[最新共同执行合同](../../shared_execution_20260915/README.md)、[阈值/双消费者待闭项](../../shared_execution_20260915/NEXT.md)和[历史总账#43](../../FUSION_STATUS_20260915.md)。数字28 nm可迁的是复用控制、位连接和队列协议；工艺节点相同不支持直接搬面积/Fmax/能量。

## 6. 三个候选接口：一项优先测，两项保留条件

### 候选A：既有coarse之后，按真实detail依赖准入与有界补算

**借入：** BiLD把大模型调用推迟到有信息时，并保留失败重放状态。**本图障碍：** 光流没有现成词表softmax置信度；coarse输出晚到时，detail供数可能早已发生；decoder的卷积、上采样、skip和halo使一个输出mask不等于一块可直接删除的输入计算。

**最小可检验接口：** 先固定一处现存coarse→detail图边，导出`task_id、原始空间位置、所需源/skip区域、decision_ready`。以实际网络图反推mask的依赖闭包 `D(M)`；为每个任务记录 `t_decision` 和首次可取消的W/源读取时间 `t_fetch`。只有决策提前到达、且依赖未被别的输出/消费者请求的部分才准取消。暂存输出到最终决策后提交；失败保留原source/identity并补算依赖闭包，不能把更大的已提交后继结果免费撤销。

**账与强控：** `coarse + 决策 + mask形成/传播 + detail(D(M)) + 验证/失败补算 + 排序/holding`，对`同coarse + 全detail`及普通静态/空间块选择。先量提前量分布、闭包膨胀、重复供数与保留字节，再判是否值得RTL。若用误差预测而非确定性证书，它定义新的有损函数，必须同环境AEE；不能以少数oracle优胜tile当真实决策器。**这是本轮最值得先做的依赖/时序测量，尚无可直接调用的本图BiLD模型或净收益。**

### 候选B：生成器只交付实际消费的θW块，候选端口与原生读W同权

**借入：** NeRN坐标→权重及C-Transformer生成下一W的流水。**本图障碍：** 已有W驻留、窄整数因子小；另训MLP可能比直接读W更贵，生成器的训练和量化没有现货。

**接口：** `coefficient_request(layer,block,quantization_generation)`→有背压的固定256bit `coefficient_word`；小W缓存带块号/代次，theta和量化参数更新即失效。只生成系数，不让TB赠激活或连续中间结果；不同消费者各自的W地址和最终函数不能混淆。若采用现成线性/查表解释器来建立无损强控，它是NeRN的接口反事实，不冒充已复现NeRN训练。

**实付：** 坐标嵌入/逆排列metadata、生成器自己的W、生成MAC与结果buffer、signed-width/escape字段、缓存首装/冲突和重生成。对同预算原生驻留W、无损ESC/普通窄位宽以及现有低秩因子。仅当明确找到被实际外传W支配且复用有限的算子才继续；当前r0优先级低。不可同时声称压缩W、又把全部展开W和额外WG留在片上而不列账。

### 候选C：门支路先完成，完整值支路继续；推测状态只用于调度

**借入：** 输出稀疏可减少后继工作的观察，以及BiLD对可撤销状态的精确界定。**本图障碍：** C-Transformer OSS的随机补脉冲不适合当前逐位函数；COMPASS最终恢复协议也尚未核到。这里提出的是本方待证接口，不能冠为它们的原算法。

**接口：** 一个生产task有`gate_pending`和`value_pending`两项义务。只有在真实有费的保守界确定门后，才允许门消费者执行；完整值计算继续至原RNE/I24端点。若只做预测，预测门仅可用于预取/排队或未提交的暂存，不得提前产生不可撤销输出。两义务均退休才回收context；统计/τ未就绪不能发“确定门”。

**强控与止损：** 给完整计算同样的门提前通知、holding和仲裁，区别收益究竟来自证书省算还是普通解耦。若完整值消费者本来就要求所有剩余项，门早定不能记成生产端少算，只能测是否藏住了后继等待；再扣状态、验证与排队费。当前只适合先导出真实消费者集合和τ可用时点，不先造大推测/回滚RTL。

## 7. 本轮结论与未完成项

这两篇并未提供一个可直接贴到当前图上的“推测通用模板”。C-Transformer已落实到full-adder重配、8倍瞬时W需求、双WG供数和有损rate-code采样；BiLD最可迁的是实际决策提前量与可回滚依赖；NeRN最可迁的是系数生成和消费之间的接口，而非小模型名字。**目前没有证据把这三项评成强新颖机制，也没有本方28 nm PPA/速度结果。**

未完成：COMPASS正文/最终预测恢复核心、C-Transformer JSSC全文/芯片代码、本图detail依赖与提前量、本图NeRN权重/整数链，以及双消费者生产端的阈值统计闭环。已发现借阅缺口后继续完成了两条非SNN primary及其实际代码检查；没有请求用户操作，也没有用摘要或早期开发日志替代最终全文。
