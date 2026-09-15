# Claude 三报告定向审阅（2026-09-15）

**结论：原局部门核零差和 T11 分桶计数应保留；“4.2 拍已达同端口下界”“库内机制穷尽”“T6 oracle 边界闭合”不由当前证据支持。前两项有具体代码/数学反例，第三项与交接稿接受的纠错直接矛盾。**

本次读完整 [HANDOFF](../claude_fusion_trials_20260914/results/CLAUDE_FULL_HANDOFF_20260915.md)、[TOTAL](../claude_fusion_trials_20260914/results/TOTAL_REPORT_20260915.md)、[T12](../claude_fusion_trials_20260914/results/T12_BENCHMARK_READING_20260915.md)，追查 T10 SV/TB、T5/T6 模型、七个 classify 字典、输入与合并表。只做小整数反例和现存 JSON 统计；未重跑 320 万判决、未编 RTL、未改 Claude/生产目录。附 [check_review_math.py](check_review_math.py)，运行 `/opt/anaconda3/bin/python3.12 check_review_math.py` 即可复核四组数学结论。

**1. 4.2 拍是当前协议和样本的平均值，不是同端口最优性证明。**

问题位置：T12 第 114–128 行；HANDOFF 第 151 行。实际 [T10 SV](../claude_fusion_trials_20260914/t10_rtl/cert_gate_bitl.sv) 第 109–121 行在 SOP 只初始化状态，并强制 `locked<=0`；检查仅在后续 `in_plane_valid` 分支发生。[TB](../claude_fusion_trials_20260914/t10_rtl/tb_bitl.cpp) 第 80–81、95–96 行又强制至少送一个数据平面。[T5 模型](../claude_fusion_trials_20260914/t5_c1_cert_core_model.py) 第 119、126 行明确使用 `1+max(planes_bf,1)`，而不是头拍信息足够就停止。

具体反例：十路 `A=4096I`、`Y=(1,…,1)`、`thr=(0,…,0)`，真实指数 `e=1`，符号全零。头拍已经知道每个 Y 在 `[0,1]`，所有判决必为 true；需要的信息只有头拍。当前实现仍收一数据拍，计 **2 拍**；允许头拍检查则信息需求为 **1 拍**，h/sign/e 字段及端口均不增加。原合同没有明确禁止这种头拍检查；若把“禁止头拍完成”本身写入合同，则只能证明这个人为限制下的计数，不能称同端口信息下界。

这不只是假想数据。[原 t5_c1_cert_model.json](../claude_fusion_trials_20260914/results/t5_c1_cert_model.json) 已记录 `bf_zero_data_plane_groups`，即当前证书在头拍所含前缀已经足够的组；每 trace 20,000 组：

| trace | 零数据平面组 | 当前拍/组（原比例×24） | 若头拍即可完成的算术机会值 |
|---|---:|---:|---:|
| s0_stage0 | 460 | 4.21095 | 4.18795 |
| s0_stage3 | 1,373 | 4.17625 | 4.10760 |
| s10_stage0 | 1,027 | 4.14925 | 4.09790 |
| s10_stage3 | 1,414 | 4.23910 | 4.16840 |

最后一列仅为 `当前均值−零平面组数/20000`，**不是新 RTL 或完整服务测量**。实际头拍须令阈值寻址使用新 h，并加入初始化/比较选择；组合路径、Fmax、阈值就绪与生产背压仍需收费验证。这个物理实现缺口限制收益声明，却不能补成“现实现已最优”的证明。

T12 对指数的解释也与代码不符：`e=bit_length(max(abs(Yq)))` 是无损整数位宽，低位一直传到 bit0，没有固定宽尾数或随 e 移动的舍入边界。因此 **e′>e 不导致低位截断**，只是冗余符号扩展；e′=e 更不可能截断。**e′<e 也不必溢出**：`Y=(-4,…,-4)` 原 maxabs 指数为 3，取合法 signed 指数 2 仍精确表示；同 `A=4096I, thr=(-3×4096,… )` 时当前证书 4 拍，指数 2 为 3 拍。脚本验证 e=2/3/4 都精确重构 −4。此例不建议未经修改就替换原协议：最短 signed 指数的 e=0 可含 −1，必须同时修正旧“补零哑平面”语义，旧审计已有提示。

可保留的窄声明是：固定当前整数编码、固定 MSB 顺序、每拍一平面、规定至少一次数据拍时，现计数与该证书停止规则吻合。SV 第 15–20 行还是独立 h/sign/e/plane 信号，没有单一物理宽度、序列化/ready 握手的最优性合同；也没有给允许的编码、状态、组合深度逐一限定。建议把原句改为“当前协议消除了 FX 重复符号位，四 trace 平均 4.15–4.24 拍/组；最优性未证”。

**2. “未知 bit 真正独立、P/N 已捕获唯一结构”遗漏了现有精确指数所携带的约束。**

问题位置：T12 第 97 行、HANDOFF 第 164–165 行。若把每个未知整数尾部独立放宽为 `[0,2^m−1]`，P/N 区间的两端对单个输出确实可达；这个数学结论成立。但实际 BF 前端报告的是**精确** `maxabs.bit_length()`，不只是任意安全上界，会排除一部分独立尾部组合。

具体反例：头拍 `e=1`、十个符号均为零。所有 Y∈{0,1}，且至少一个 Y=1，否则 e 应为 0。取每个输出行 `A[t,s]=1`、`thr_t=1`，则实际点积一定在 **[1,10]**，所有门已确定 true；原独立盒区间是 **[0,10]**，不能锁定。小脚本穷举这 **1,023** 个合法十位向量验证下界为 1。无需给 TB 新信息，约束已经在 e 中。

这只否定“无剩余相关结构/可解析杀死所有改进”的全称断言；**不证明仿射算术本身适合编码这条非线性最大值约束，也不证明本任务的净周期会改善**。任何增加界计算的方案仍须与同资源普通控制比较。T9 的词序平均代价也不能反过来证明另一个编码/信息集合的界最优。

**3. T11 证明的是 698 条记录完成分类；其证据粒度不足以推出机制穷尽，而且 226/472 的解释有可复核的统计错误。**

七个 `classify_b*.py` 均为预写 `id→(disposition,family,reason)` 字典；逐条与 [t11a_merged.json](../claude_fusion_trials_20260914/t11_triage/t11a_merged.json) 比较完全一致。698 条最终记录全部只有 `id/name/disposition/family/reason` 五字段。[merge_t11a.py](../claude_fusion_trials_20260914/t11_triage/merge_t11a.py) 第 34–46 行只校验 ID、重复、合法枚举，以及 NEW 才要求的 how/trial_type；没有已跑实现/结果路径、机制覆盖范围或正文页码的校验。因此 **50/154/78/416/0 是真实分类输出**；不是 50 个完整 A 均复现、154 个家族所有接口已实验否定的验证结果。

筛选规则本身也限制了结论：[AGENT_PROMPT.md](../claude_fusion_trials_20260914/t11_triage/AGENT_PROMPT.md) 明示“已否决方向命中即 FAMILY_COVERED，不必立项”和“只依据输入记录内的信息判断，不做网络检索”；[make_tsv.py](../claude_fusion_trials_20260914/t11_triage/make_tsv.py) 第 21–25 行把名称裁到 45 字符、`A or untried` 裁到 150 字符。没有证据说每篇**只**读了 TSV，也不否认已有 P0 精读；但最终表缺乏逐项阅读层级与证据链接，不能从“有一行”升级成“该机制已经完整迁移或排除”。T11e 声称的 164 篇复查也没有在最终记录中保留对应原文范围和复判痕迹，不能用整体一句话补齐这条链。

可复核的具体例子：

| ID | 输入保留的缺项 | classify 输出及其限制 |
|---|---|---|
| W0005 Phi | `untried` 明写原八输入归约、PWP 存储/预取、冲突处理、完整 Phi→TA，及 PAFT/布局执行未做 | [classify_b4.py 第 8 行](../claude_fusion_trials_20260914/t11_triage/classify_b4.py) 标 INCORPORATED，仅理由“已作…对照”。只能解释成已借入部分机制，不能覆盖上述未做接口。 |
| W0002 GustavSNN | `untried` 明写完整 8×8 调度/带宽及 producer→FC1→PSN→FC2 未全部完成，局部四 ID 重读负结果不能当原作分母 | [classify_b1.py 第 8 行](../claude_fusion_trials_20260914/t11_triage/classify_b1.py) 标 INCORPORATED“分母已继承”。部分借入可成立，但不能推出“所有适配项都已跑过”。 |
| W0283 RAFT | `untried` 明写完整另网络、相关体积、在线初始 flow 未移植 | [classify_b0.py 第 46 行](../claude_fusion_trials_20260914/t11_triage/classify_b0.py) 标 INCORPORATED“算法底座”。算法背景/对照引用被计入 50，不能把 50 直接当硬件实试数。 |
| W0300 C-Transformer | 输入及 [classify_b2.py 第 48 行](../claude_fusion_trials_20260914/t11_triage/classify_b2.py) 为无全文 APPLICABLE_BLOCKED | T12 §1.7 却写“域失配维持”，状态理由不一致。当前已有 [ISSCC 全文及 BiLD/NeRN 机制审读](../paper_mechanism_transfer_20260915/literature/compass_ctransformer.md)，可更新证据；这不自动把它变成适用或已验证候选。 |

TOTAL 第 107–111 行又将 **226 写成 curated 层、472 写成 arXiv 噪声**。实际恰好对应 `bool(A or X)` 是否非空，而不是来源标签：

| 输入 A/X 字段 | 数量 | INCORPORATED | FAMILY_COVERED | BLOCKED | NOT_APPLICABLE |
|---|---:|---:|---:|---:|---:|
| 有内容 | 226 | 22 | 58 | 42 | 104 |
| 均为空 | 472 | 28 | 96 | 36 | 312 |

第二组包含 RSR/RSR++、Swift、GoSPA、Kaleido 等既有目录记录及本地 `row2of4_N_M`，不能整体称 arXiv 噪声；自身分桶也有 **160 条不是 NOT_APPLICABLE**。按输入 `cat` 含“arXiv增量”计为 227 条，亦不等于 472。这是数据口径错误，不是对人工判断风格的泛泛意见。

建议改为：“在当前限定为 C1 门核供数、按既定家族归并规则的 698 条记录筛选中，未标出 NEW；部分 A 只借入局部机制，未测接口与缺正文项仍保留。”删除 TOTAL 第 113–115、146–149 行的机制穷尽及无对象可跑推论。反过来，本审阅也**没有证明**一定存在能过质量/净服务门的新机制。

**4. T6 已接受的纠错没有一致传播；已知阈值局部精确不等于部署统计链闭合。**

HANDOFF 第 138–141 行明确接受“五点扰动只证明样点敏感性；不证明部署阈值/动态 BN 为训练期冻结常量”。但 TOTAL 第 **32、55** 行仍写“oracle 边界闭合”；[T6_REPORT](../claude_fusion_trials_20260914/results/T6_REPORT.md) 第 **30、34** 行仍称包络内任何部署阈值都成立、可作部署态声明；T12 第 **104** 行仍将 tau0–9 ROM 称作阈值侧已静态化。它们与已接受边界直接冲突。

[t6_ptau_supply.py](../claude_fusion_trials_20260914/t6_ptau_supply.py) 第 47–60 行仍从完整当前 Y 计算均值、方差及 B32 前缀矩，第 93–94 行只取五个 k。readmemh 装入已经算好的 tau 表，是仿真接口静态配置，不能证明模型中的 tau 与当前输入无关。T6 的负 gamma 转换在第 63–67 行已有修正；本审阅不把旧数值 bug 重报为未修。

原 [T6 包络反例](../shared_execution_20260915/claude_review/check_t6_envelope.py) 仍适用，附脚本独立重写了这几行整数计算：`A=4096I, Y=(5,…,5), e=3`，`thr_final=−10×4096`、`thr_ptau=30×4096`。k=0、1/2、1、2、4 均为 **2 拍**，包络内部 k=3/8 为 **4 拍**。这反驳五点外推的逻辑，不声称四份真实 trace 已出现该最坏例。完整净服务也须支付原始 Y 生产、矩统计/等待、阈值生成/装载与消费者退休；post-Y/known-threshold 的付费前端只能闭合其中一段。

建议统一保留：“四 trace、规定五个插值点上，局部 BF 证书拍比 16.5–17.7%；动态统计与部署阈值的供数尚未闭合，不能继承整网 AEE。”已有局部整数零差和修 bug 回归继续有效，无需为撤回不成立的最优性/穷尽性推论重跑大型激励。

**5. T12 的标杆机制描述有几处会直接误导后续迁移。**

此段由主代理对照此前实际原文阅读补充，不把该代理的字典核查冒称原文复现。

- §1.5 的“权矩阵行重复→二值模式码本”把对象写反了。Phi 量化/匹配的是二值**脉冲支持行**，固定W允许预计算中心乘权重；不是INT8权重本身成为16bit二值行。其matcher、两个packer、signed残差与PWP汇合都不能从“静态离线”一句话删掉。本轮RTL已实付其中的求值/取表子集，但仍缺原两路并行。[Phi原文与代码范围](../paper_mechanism_transfer_20260915/literature/bishop_phi_prosperity.md)
- §1.2 的ECP“分数低于界所以精确剪除、ANN下此界不存在”过强。支持量能给二值QK分数确定上界，**确定低分不等于该项为零，也不等于删除后的网络逐位相同**；还要看后续门/误差合同。连续Q/K有幅值/范数约束时也存在点积界，区别在于获得/计算界的成本与紧度，不能以“ANN完全没有界”支撑新颖性。[Bishop界、缩放及后继边界](../paper_mechanism_transfer_20260915/literature/bishop_phi_prosperity.md)
- §1.1 将前人概括成“用了GP但全是OP”不能作为方法继承关系。Gustav比较IP/OP/GP并为SNN改GP的状态驻留、分块和NRV；GAMMA是其明确归并来源。PTB/LoAS的时间批处理与GP不能混成一个原算法。[Gustav §II–VII与GAMMA原文链](../paper_mechanism_transfer_20260915/literature/gustav_and_spmm.md)
- §1.4 将FireFly-T的bank冲突归结为乱序执行，省掉了宽W向量取回与跨稀疏源复用的存储组织。只搬调度器、不给对应W布局，不能继承原来的冲突消除收益。[FireFly系列原文和实际代码范围](../paper_mechanism_transfer_20260915/literature/firefly_family.md)
- §1.7 对COMPASS仍应保持“未读到最终恢复协议”，不能仅凭题名填成精确补算。C-Transformer的ISSCC全文现在已取得，其OSS是有损概率合成剩余spike，不是逐spike精确回滚；BiLD和NeRN的原文/代码也已经补读。因此“无全文/只剩索取”需要更新，但不能反向把这些机制说成已经适配当前非因果T10。[ISSCC全文、AE及两条非SNN引用](../paper_mechanism_transfer_20260915/literature/compass_ctransformer.md)

这些纠错保留了用户要求的研究路径：借足已有A，寻找本图的真实失配，再试具体接口。它们不推出任何一个未试候选必然有效，也不要求先穷尽整库才准写RTL。
