# 从成熟加速器的失配处寻找 SDformer 硬件增量

2026-09-15。当前最值得推进的组合是：**保留 Prosperity 父图、用完整 Phi 改 root/delta 求值；有限模式表的生成与消费；粗流后的真实细节依赖调度。**第一条延续 C1，但不会再复刻先改父图的旧 APEC 布局；第二条面向已有支持码/FFN 候选的读表税；第三条利用光流任务结构。它们是待验证的研究方向，尚没有“强接收已成立”的证据。

本轮同时完成一项硬件推进：实际 packed 源字写入→单口源存储→NRV∩W→完整 T10 门的四臂 Verilator 比较，共1728次任务、552,960门判决零差。新增 seen-code 摘要与普通非空标志效果相同，因此完成的是可复用的执行底座，摘要本身不保留为新贡献。[完整结果](PRODUCER_RESULTS.md)

**一、这次读到了什么，哪些材料仍缺。**

| 工作 | 会议/期刊与版本 | 本轮阅读范围 | 可复现材料的真实范围 |
|---|---|---|---|
| Prosperity | HPCA 2025；作者 v2 | 正文、架构、评估、消融、引用；作者关系生成 CUDA 与周期模拟关键路径 | 官方 simulator/CUDA；不等于公开完整 ASIC RTL |
| Phi | ISCA 2025；作者 v1 | 正文、L1/L2、matcher/packer、预取、PAFT、消融 | 未定位对应作者完整训练/RTL包 |
| Bishop | ISCA 2025；作者 v1 | 正文、TTB/BSA/ECP、异构核、消融；公式页目视复核 | 未定位 BSA 的作者 surrogate 实现 |
| GustavSNN | HPCA 2026 | 主文 §§II–VII、评估、引用 | 本方已有部分数字适配；不是照搬作者全芯片 |
| FireFly | TVLSI 2023 | 作者全文与 v1 代码关键路径 | Scala、仿真与部分板端材料 |
| FireFly v2 | TCAD 2024 | 作者接受稿全文与 v2 代码关键路径 | Scala/仿真；不等于全部论文板端包 |
| FireFly-S | TCSI 2025 | 接受稿及 arXiv v3 全文 | 对应完整作者工程未定位 |
| FireFly-T | TC 2026 | arXiv v1 全文；正式卷期已核 | 正式 VoR 全文、对应完整工程未取得 |
| C-Transformer | ISSCC 2024 | 三页全文及所有机制图、引用 | 没有取得这颗芯片的公开 RTL |
| C-Transformer 长文 | JSSC 2025 | 作者摘要、身份和参考文献 | 全文未取得，不能与 ISSCC 数字拼接 |
| COMPASS | MICRO 2024 | 官方身份、作者 AE 的 trace/config/hook/运行及历史说明 | **主文未取得**；分发模拟核心部分为打包二进制，不能声称读完恢复协议 |

对应逐篇分析：[Prosperity/Phi/Bishop](literature/bishop_phi_prosperity.md)、[Gustav 与 SpMM](literature/gustav_and_spmm.md)、[FireFly 四篇](literature/firefly_family.md)、[COMPASS/C-Transformer](literature/compass_ctransformer.md)。各报告列出具体章节、代码入口、原文链接及未读范围。这里“全文读过”和“完整复现”是分别记录的状态。

**二、这些论文怎样把一个大想法变成实际机制。**

**Prosperity：发现重复乘积后，主动限制复用图。**从“二值稀疏”推进到“包含关系可共享结果”，随后用单父森林限制依赖、检测、缓存与调度。TCAM 查子集还不够，必须有合法相同父顺序、最大子集选择、排序、父值读取及最终规约。论文试过两前缀后其架构更慢，所以保留一前缀；这是具体实现的取舍，不能扩大成公共子表达式都无效。值得学的是先找能兑现的依赖结构，再为它设计 detector/processor 流水。[原文](https://arxiv.org/html/2503.03379v2)

**Phi：把不满足子集关系的模式也纳入，再补齐表示的费用。**它将二值行表示为中心加 signed 残差，中心乘权重预计算，残差精确修正；zero/onehot 和原始稀疏路径本来就有。随后解决残差太短造成的低利用率、bank 冲突、旧 psum 汇入和 PWP 预取。其选择预取已减少了表流量，但未消除它。因此本项目应该借完整的查询—修正—汇合，而不能只拿一个码本。[原文](https://arxiv.org/html/2505.10909v1)

**Bishop：短 T 不够摊销，加入 token 维后又遇到不同密度。**TTB 同时组织 token/time，按 feature 的活跃 bundle 分流到稀疏/稠密核；训练改变 bundle 活动，注意力再另用二值支持界剪枝。其逻辑链是“复用粒度→密度不均→不同执行核→使工作量适配该粒度”。单共享核可以借其数据流，却不能免费借两核并发收益。BSA 公开式(9)(10)按通常 L0 定义相加似乎只是总 spike 数，和文字所说 active-bundle 目标不一致；真实 surrogate 待核，不能自行改式后称忠实复现。[原文](https://arxiv.org/html/2505.12281v1)

**GustavSNN：二值运算便宜以后，宽状态搬运变成主要矛盾。**它先比较内积、外积与 GP，选择让输出状态驻留；整行状态过大，就沿输出列分块；时间大位图吃掉动态稀疏，就用小位置组 NRV；W 再变稀时，补 NRV∩W 的双指针。每一步都由前一步仍贵的部分引出。GAMMA 的归并与供数是其明确来源，Prosperity 是并列对照。[论文](https://doi.org/10.1109/HPCA68181.2026.11408587)

**FireFly 系列：相同器件也会被不同问题推到不同架构。**初代把 DSP 的乘法路径改用于选择加法并复用存储；v2 面对首层/残差的多位输入，用位分解与时间组织复用阵列；S 通过剪枝量化使各层常驻，采用空间流水；T 则回到 overlay，以多路解码、宽 W 广播、负载均衡和独立 binary-attention 处理 transformer。不能把四篇的最佳局部机制和四套资源合并成一个免费的 A。[v1](https://arxiv.org/abs/2301.01905)、[v2](https://floyedshen.github.io/pdf/li2024fireflyv2.pdf)、[S](https://floyedshen.github.io/pdf/li2024fireflys.pdf)、[T](https://arxiv.org/abs/2505.12771)

**C-Transformer：算法压外存，电路处理新的供数形态。**大小模型协作减少大模型调用，隐式网络生成权重；为了避免分设 DNN/SNN 核闲置，重配同一加法阵列。ST 模式的瞬时 W 需求与 DT 不同，还需要装载器、对齐和缓存。OSS 用采样概率合成剩余 spike，属于有损近似；它没有因引用 BiLD 就变成逐 spike 精确回滚。[作者项目](https://ssl.kaist.ac.kr/bbs/board.php?bo_table=Neuromorphic&wr_id=4)、[ISSCC DOI](https://doi.org/10.1109/ISSCC49657.2024.10454330)

**COMPASS 暂只能沿已核部分学习。**它研究 SRAM-CIM 上的自适应 spike speculation，作者 AE 包含时间子窗、稀疏表示与推测的分离消融。但公开可读材料不足以还原最终命中判定和恢复状态，不能拿别篇的算法替它补全；也不把其 CIM PPA 当本方数字28nm路线。待取得正文后再判断应借哪一种推测接口。[作者 AE](https://github.com/ZongwuWang/COMPASS_AE)、[MICRO 官方议程](https://microarch.org/micro57/program/)

**三、从非 SNN 原作迁入，具体改变了什么。**

| 原作与来源关系 | 本轮追读的核心 | 对本图可借的内容 | 迁移障碍 |
|---|---|---|---|
| GAMMA，ASPLOS 2021 → Gustav | 归并、FiberCache、fetch/read 分离、临时结果消费后失效 | 有限结果工作集和显式消费义务 | T10 PSN 与连续 PED 不是普通一读即释放 fiber |
| MatRaptor，MICRO 2020 → Gustav 对照 | C²SR、独立输出所有权、有序队列归并 | 格式与 bank/输出所有权共同设计 | 本方捕获布局固定，不能擅自用 channel%bank 代替地址 |
| GROW，HPCA 2023 → Gustav | 高连接度缓存、cluster、多行 runahead | 固定消费者图的复用及等待表 | GROW 图静态；动态 spike 图的每帧建图费用不能隐去 |
| SIGMA，HPCA 2020 → Bishop/Phi | 分发网络、变长规约、bitmap 与实际供数 | 完整 sparse pack/merge 的强 A | 两个大网络或多写口不等于本方单口N8 |
| HAG，KDD 2020 → Prosperity 讨论 | 有限容量下建立新 aggregation 节点 | 原森林外的少量公共结果 | 动态检测、建值、保存、scatter 未必比 AAC 便宜 |
| Transitive Array，2025 作者稿 → Phi 引用 | 静态权重 bitplane/Hasse 图、前缀状态 | 连续输入乘固定权重的另一求值图 | 本方已有 CMVM/DA；不能忽略强8乘法器和符号恢复 |
| Bit Fusion，ISCA 2018 → FireFly v2 | 位宽相关空间融合、供数与重建 | 可重组算术与低位宽利用 | 不是普通 bit-serial；宽恢复与端口要一同继承 |
| Trapezoid，ISCA 2024 → FireFly-T | 内积/GP 模式、双分配网络、缓存归约 | 稀疏分布不同应改变完整执行组织 | “按密度切换”已有人做；本方新增内容需落在真实失配 |
| LSQ，ICLR 2020 → FireFly-S | 学习步长和梯度规模 | θ折权后联合整数尺度与剪枝 | 连续残差尺度/RNE不能按二值神经元直接消掉 |
| JPEG-ACT，ISCA 2020 → Phi 动机 | 激活表示压缩及编码器 | 先验证数据统计再设计表示 | 原作训练offload，有损变换；稀疏激活不必频域更稀 |
| BiLD，NeurIPS 2023 → C-Transformer | 实际fallback时点、批验证、后缀回滚 | 决策提前量、暂存与失败依赖闭包 | 光流无现成词表置信度；非因果PSN错误不局限后缀 |
| NeRN，ICLR 2023 → C-Transformer | 坐标到权重、排序、重建与蒸馏 | 系数生成器与消费者接口 | 原软件先生成全W；本方W已驻留时未必有可省外存 |

补读的强对照包括 SNN 的 PTB、TCAS-II 2023 量化/剪枝研究，以及 ANN attention 的 SpAtten。PTB 读到的是作者2021投稿稿的方法，不能将稿中PPA写成最终HPCA22结果。SparTen、PQ、Batcher原作以及部分正式长文仍有全文缺口，没有计入全文完成数。

源论文与具体章节在四篇专题报告中逐项列出。这里是跨论文推导，不声称这些作者已经解决了本项目的三条候选。

**四、先按当前负结果写 B，再重新发散。**

当前已测共同执行中，空间分解虽然 AEE 更好，连续 Q2 发射约为 R8 的3.1倍，同资源服务更慢；只把一乘变成几加不够。R8 已有 bitmap、原生onehot、时间归约、Q1驻留、Q2 cache、RR，不能再把这些当新X。旧 patch≈35%、FFN≈26%是历史工作量代理，不能直接当当前学生的周期份额；需要针对当前图重列真实算子账本。[现有证据与完整分母](../FUSION_STATUS_20260915.md)

与 Prosperity/Phi 有关的旧实验尤其应分开：

| 旧布局 | 实际结果 | 能否否定这次接口 |
|---|---|---|
| APEC先扣公共项，再重建残差父图 | 完整层CPU服务比原Prosperity慢16.44%；扩大缓存仍慢 | 说明该图变换/共享缓存放置有问题；没有执行固定原森林的Phi求值 |
| 原行先Phi分解，基图与残差共同16槽 | 加法比原Prosperity多34.07% | 是加法数，不是周期；原森林也未保留 |
| 固定原父，最多两个残差子集字典项 | 6557→6154加法，少6.15% | 这一小接口已经试过；不能笼统称“固定森林+字典”未试 |
| 有限槽、按需基图生成与直接PWP对比 | q8/16的子集构造比预计算PWP多8.03%加法，另增W读/写 | 按需构表思想已有数值探针；②未完成的是实际同口RTL，不是首次想到本地生成 |
| 固定原父，root/delta各走signed Phi两路与实际预取汇合 | 未找到已闭合执行收据 | 这是仍可有界尝试的接口；不是宣布它会赢 |

本轮将以下18项作为**问题假设清单**，不是18个已写RTL的新机制，也不另开18个Card目录。

| # | A与可尝试改变 | 本图B与第一项辨别 | 处理 |
|---:|---|---|---|
| 1 | Prosperity固定森林＋Phi root/delta求值 | 旧图重构破坏原复用；比较保留图后的实际PWP/修正/父退休 | 短名单① |
| 2 | Phi按需预取＋有限局部PWP生成 | 支持码CPU省槽但表字节增加；把实际生成、查表和回退落RTL | 短名单② |
| 3 | HAG两个公共节点＋Prosperity | 非子集公共支持可能有用；先与旧−6.15%残差字典同权比较 | 并入①消融，非另一个标题 |
| 4 | GAMMA式部分结果生命周期 | 双消费者使状态不能早释放；普通引用计数也给A | ①/②共享实现条件 |
| 5 | GROW固定复用簇＋有限等待表 | spike图动态但W静态；先验证实际地址复用而非假设power-law | ②可选后续，未实施 |
| 6 | Bishop正确active-bundle训练＋实际事务目标 | NR4/activity已出现粒度吞收益；必须比普通组稀疏损失强 | 保留，等一个执行器固定后再训 |
| 7 | FireFly-S/LSQ＋二值前级/连续后级联合剪枝 | Q1省算可能转移成Q2多位工作；按两侧实际服务比较 | 短名单④，算法恢复候选 |
| 8 | FireFly-T写时转置＋PSN最终提交 | BN和全T10屏障之后还有重排；同容量写时转置是强控制 | FFN闭合底座，非独立标题 |
| 9 | FireFly-T跨context W共同驻留 | 只共享Q2读的乐观上限约0.409%；不能靠它撑大收益 | 只留普通优化，不开标题 |
| 10 | Trapezoid式完整模式切换 | 交集/归约/供数都随模式变化；现dense/bitmap已涵盖一部分 | 作为②强控制，不把阈值切换当X |
| 11 | Transitive Array连续Q2权重图 | 可能少乘却多前缀/符号重建；本方DA已收费失败 | 保留不同图接口，不重复旧DA |
| 12 | Bit Fusion/v2宽算术重组 | 本方已有多plane/进位修复；需新资源利用矛盾 | 现有执行底座，不重新命名 |
| 13 | BiLD式决策提前量＋粗流细节依赖 | 粗流先有了才能少算；输出mask会经halo膨胀 | 短名单③ |
| 14 | NeRN只生成实际请求W块 | 若原W已驻留则生成更贵；先找真外存/缓存miss挂点 | 当前R8低优先，未否定FFN大W |
| 15 | C-Transformer OSS概率时间近似 | T10 PSN利用时间位置，频率相同不保证函数相同 | 需独立训练身份；不套当前精确链 |
| 16 | COMPASS推测→有界数字补算 | 最终恢复机制原文仍缺 | 等正文；不据缺稿判性能失败 |
| 17 | 证书门早定＋完整值继续 | τ何时可得、连续消费者是否还需所有位 | 延续已有门核，但不能计生产少算 |
| 18 | 真实生产seen-code→稀疏准入 | 与普通写时非空flag是否不同 | **本轮RTL已试：无增量，归入A** |

**五、收敛为四条短名单，前三条先执行。**

下列评分是对“是否值得拿资源验证”的主观判断，采用新颖性潜力/本图适配性两个0–10维度，**不是录用概率，也不是已证明创新性**。实测证据栏单列，避免把想法评分冒充成果。

| 候选 | 潜力/适配自评 | 已有支持 | 目前最强反对 |
|---|---|---|---|
| ① 固定森林的Phi精确求值 | 5–6 / 7 | 确认不同于两种旧失败图；旧固定父子集字典有薄加法正点 | HAG/ExSpike/Phi已覆盖大部分思想；N8与同口税可能吞掉全部增量 |
| ② 有限PWP生成—消费组织 | 5–6 / 8 | 支持码CPU14.033M→8.708M服务槽；表字节13.681→34.848MB | Phi本来已按需预取，DA也会lazy构表；必须超过这两个完整A |
| ③ 粗流后的依赖闭包调度 | 6 / 6 | 已有coarse模型与可用质量；新mask/调度尚未完成 | 动态推理/稀疏细节已有先验，决策晚到或halo会吞收益 |
| ④ 两侧执行费用约束剪枝 | 4–5 / 7 | 已有R8/空间质量—服务差距，说明仅省前级无效 | 普通硬件感知组剪枝可能解释全部改善 |

②的旧CPU数字属于其自身direct-code学生与工作负载，不是①的预测，不是R8的周期，也不是RTL加速比。④允许有损恢复，质量以同协议SDformerFlow NB0为门，**不恢复+0.005限制**。以上每条都保留性能与质量各自的分母。

[独立复核](REVIEW.md)对**目前已经能具体指出的新颖增量**只给①3/10、②2–3/10：K-local到齐才发布、引用计数和普通按需预取本来就是成熟做法。主观未来潜力不抵消这个低分。继续做它们的理由是完整迁移与失配分析仍有明确缺口，并非已经选出两个足够新颖的标题；如果后续只有普通A的收益，就保留实现、继续寻找由实测瓶颈导出的X。

**六、首个机制的完整推进页：固定父图，改变求值方式。**

**问题B与对照先行。**当前没有证据说明换父图一定优于原Prosperity；旧APEC还会把原先便宜的父边拆掉。另一方面，原Prosperity只复用现有子集行，root和大delta仍需执行。强对照至少有：原生bitmap/onehot；同资源原Prosperity；固定父＋已有两项残差子集字典；Phi单独完整两路；候选固定父＋Phi。所有臂享有相同W缓存、输出宽度、预扫描/编译权限和psum口。

Phi-alone消费原始S，必须以相同校准样本和代码本预算独立选中心，不能被迫使用为root/delta选的中心。②另加普通压缩PWP＋同cache/同预取控制，避免只打未压缩的34.848MB表；给各臂同样的原生packed issue和W驻留。第一片若只是PWP加串行修正，名称应为“Phi精确分解的共享执行适配”，直到matcher/packer/冲突处理实际补全前，不称完整Phi硬件复现。

**借入A。**保留原父选择和稳定顺序；针对root或delta `d` 选择中心 `c`，执行 `dW=cW+(d−c)W`。取零中心、onehot和长残差回退必须保留；signed修正每项精确计入；只对实际请求的PWP取数。权重已吸收θ，符号和累加宽度依原合同，不能跨RNE/饱和边界。

**候选X。**只改变K-local求值和结果的可消费时点：原父值、PWP贡献、正负修正在同一个受限结果槽内按明确义务汇合；全部到齐才发布该K-tile的`parent_ready`。固定森林不跨K拉依赖，不能把某K的partial父冒充全K父。若节省只来自一次普通缓存命中，必须归给共同A；X要表现为减少了原方案无法同时避免的父重算、表搬运或存活期冲突。

**两句话研究描述。**已有模式复用要么受限于现存子集父，要么用较大的模式乘积表与额外汇合换取自由。我们检验在保持原有父依赖不变的前提下，有限的模式求值和局部结果完成协议能否同时保住原复用、减少剩余根/差分服务，并在窄端口下产生净收益。

**具体RTL。**第一片用K16、N8、真实M行，固定一个有限PWP/结果容量；模块划为原始位图输入与父描述符、pattern选择/原生旁路、实际W/PWP读取、signed残差发射、单口psum更新、K-local完成发布。RTL第一片可以接受已计算父图作为明确边界，但检测/排序必须另计并在完整层试验接回，不能冒称完整TCAM链。禁止TB输入乘积答案、理想缓存命中或免费scatter。

**三个辨别实验。**

1. 同一组真实root/delta，依次跑五个强对照；记录最末结果、W/PWP字节、构表、父读、psum写、等待。先看完备求值是否确实不同于旧子集字典。
2. 相同周期端口及相同容量，分别从冷表开始和合法跨tile驻留；引入实际背压、全空/onehot/长残差、两路乱序到达与最后消费者延迟。任何“只在暖表赢”的点必须报告首装摊销。
3. 在当前R8 K864/N8本图闭合后，另回完整K6912的旧瓶颈算子，分开记录K-local父与跨K规约；各自与原Prosperity同工作负载表对比，并保留held/disjoint窗口。两个算子不串成一个倍率。最终VCS/综合/时序/等价只在选定组件之后进行。

**停止与保留。**若①只达到旧−6.15%加法、冷服务无改善，停止这个槽/调度布局；不据此否定Phi、Prosperity或HAG家族。10–15%服务改善是值得进一步做物理验证的候选区间，低于它也可保留明确的面积/能量问题，但不得在未测前许诺。若X被普通固定父字典或Phi原生预取解释，保留有效A、撤回X标题。

**有界试点顺序。**先冻结上述五臂的原始输入和两路接口，完成一个K16完整求值RTL；接着只做同容量冷/暖与背压；最后才扩完整层。②利用同一W/PWP/结果接口研究构表位置，避免另起一座无关小核。③在算法侧同步导出真实decoder依赖与决策时点，先写可取消任务队列/原生全算对照；有损门通过匹配NB0后再扩大验证。训练已获授权，何时启用取决于可检验接口是否就绪，无需再申请一次权限。

**七、对当前硬件工作的实际影响。**

本轮不再把旧C1/C2静态共享、普通广播或lifting节点数当作投稿贡献。Gustav生产接口继续作为可用底座；seen-code独立X已经实测淘汰。C1按①延续原Prosperity的强分母并完成Phi求值；C2/FFN按②解决实际表与完整消费者费用。③保留光流特有的有损/无损调度空间，④在执行器固定后做成成对训练对照。

还没有完成：完整Phi matcher/packer的本图实现、①的五臂RTL、②的有限构表RTL、真实FC1/BN/PSN/FC2整链、③的因果mask与依赖控制、对应ASIC PPA。生产nts07与TCAS-II稿未修改。当前可交付的是主论文与关键原作的机制分析、明确去重后的执行计划，以及一项真正写过并测过的生产接口结果。

**重点原作链接。**

- [GAMMA，ASPLOS 2021](https://people.csail.mit.edu/sanchez/papers/2021.gamma.asplos.pdf)；[MatRaptor，MICRO 2020](https://www.csl.cornell.edu/~albonesi/research/papers/micro20-2.pdf)；[GROW，HPCA 2023](https://arxiv.org/abs/2203.00158)。
- [SIGMA，HPCA 2020](https://anands09.github.io/papers/sigma_hpca2020.pdf)；[HAG，KDD 2020](https://www-cs-faculty.stanford.edu/people/jure/pubs/hags-kdd20.pdf)；[Transitive Array 作者稿](https://arxiv.org/html/2504.16339v1)。
- [Bit Fusion，ISCA 2018](https://jongse-park.github.io/files/paper/2018-isca-bitfusion.pdf)；[Trapezoid，ISCA 2024](https://yang-yifan.github.io/papers/isca24_trapezoid.pdf)；[LSQ，ICLR 2020](https://arxiv.org/abs/1902.08153)；[JPEG-ACT，ISCA 2020](https://people.ece.ubc.ca/aamodt/papers/evans.isca2020.pdf)。
- [BiLD，NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/file/7b97adeafa1c51cf65263459ca9d0d7c-Paper-Conference.pdf)与[作者代码](https://github.com/kssteven418/BigLittleDecoder)；[NeRN，ICLR 2023](https://arxiv.org/abs/2212.13554)与[作者代码](https://github.com/maorash/NeRN)。
- [量化与剪枝的硬件影响，TCAS-II 2023](https://doi.org/10.1109/TCSII.2023.3260701)；[PTB 作者公开投稿稿](https://web.ece.ucsb.edu/~lip/publications/SparseSNNAccelerationIEEE-MICRO-Submitted2021.pdf)；[SpAtten，HPCA 2021](https://arxiv.org/abs/2012.09852)。
