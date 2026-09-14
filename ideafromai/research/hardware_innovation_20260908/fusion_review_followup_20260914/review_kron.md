Kronecker 输出组逃逸：独立复审，2026-09-14。

**结论：修复后的固定整数函数与周期统计获得支持；这次候选没有通过质量门槛，应停在当前布局。** 原版存在 cold mode0 → warm mode1 未加载因子的公共接口错误，本审阅独立复现，作者补充 `factor_resident` 后已独立复跑通过。同函数展开 Wh 与候选相比，真实八窗口无外部停顿的冷启动总周期 134140 → 131147，减少 2.231%；十帧新函数 AEE=1.898806137，高于同集 NB0=1.469968114。后者是作者运行的质量结果，本审阅读取代码和结果，没有重新运行 GPU，也没有把它写成全网 bit-true。[RTL 汇总][sum]、[质量结果][quality]。

| 审查项 | 判定 | 依据与边界 |
|---|---|---|
| A[12,4]、B[8,2]、两个 exact 输出组和展开 Wh 的同函数性 | 支持 | 336 个训练原始 source 重建 z；16 个 fixture 独立重算 S、Wh、raw、J、I24；escape 为 0-based 组 6、7 |
| 19 位 Wh/S、符号与最终舍入 | 支持有界契约 | Wh 最大绝对值36855，不能按 signed16 存；新增 S=221184/−214272 角落验证19位路径；中间没有 RNE |
| 8 lane 乘加共享、S 构建和读写费 | 支持 RTL 周期模型 | 生产者共用8个19×13乘法和8个32位加法；S读写均发生于实际状态；尚无 SRAM 映射/Fmax/PPA证据 |
| 冷配置与 warm 模式切换 | 已修正 | 修复前实际输出错误；修复后0→1只补15拍，1→0补0拍；原同mode测试没有覆盖这一点 |
| 端到端 I24 和完整 K864/C96/R8/N96/T10 | 支持本地范围 | 原始source、FP32 identity、参数进入RTL；raw/J/I24逐字检查；无TB注入z、S或中间p |
| 超越相同函数强控制的算法优势 | 未证实 | 同质量小模型、普通精确行保护等没有比较，且当前十帧质量已失败 |
| 将“Kronecker+稠密保留”作为新分解 | 修正 | HKPRNN 已有直接结构前例；本地差分限于量化格式、8lane执行与状态调度 |

**原始输入与数学。** 独立脚本按原始3×3 tap、96通道、四空间位置和T10逐项卷积，重建训练336窗口的z，逐项吻合原捕获；重新计算 `mean((z·(Wk−W)^T)^2)·(a_q40/mean(a_q40))^2`，训练误差最大的两组确为6、7。真实八fixture的打包source、padding origin、FP32 identity、Q1、A/B及Wh也与原始捕获/拟合文件一致，不能用另一份同名real数据替代。[独立重算][mathcode]、[数学结果][mathresult]。拟合只读训练latent，以重排矩阵的rank1 SVD初始化并进行8轮整数ALS；这里加权误差是舍入前误差的代理目标，没有证明它最小化完整网络AEE。[拟合][fit]

令 `z[k,j]=z[2k+j]`，`S[k,l]=Σ_j B[l,j]z[k,j]`，非逃逸组输出 `p[o,l]=Σ_k A[o,k]S[k,l]`，逃逸组输出 `Σ_r Q2[8o+l,r]z[r]`。展开控制逐项使用 `Wh[8o+l,2k+j]=A[o,k]B[l,j]`，并在两个逃逸组替换回原始Q2。用int64直接展开与收缩，两条路径全等；A、B、Wk实际矩阵秩分别为4、2、8，因此重排矩阵的rank1不等于原矩阵降为rank1。[输入/金值生成][prepare]。该Wh在其余80个输出通道上改变原Q2函数，不能把同Wh的RTL等价称为原模型无损。

固定拟合Q1对全部二值输入的逐rank下界为 `[-419,-361,-211,-352,-367,-259,-236,-399]`，上界为 `[248,119,290,95,222,338,236,263]`。独立三角界得到 `|S|≤16451`、`|p|≤67955134`，与作者报告一致；这不是只看八窗口最大值。`v_mem/qblock`均为signed19，配置也取19位，实际36855未被16位截断。[位宽/存储声明][corewidth]、[配置写入][corecfg]。SSTORE取低19位在上述界内精确；S没有缩放或RNE。全signed3 Q1范围允许−4，K864使z范围落在signed13内；任意两个signed6 B项的S最坏绝对值221184仍落在signed19内。**32位p仍要求有界参数，不能推论所有合法端口比特组合都无溢出**：例如Q1全−4、A全−4096、B全−32可使p绝对值3623878656。固定拟合与本次所有角落满足p界，这个反例不推翻已声明的有界格式。[拟合界检查][fitbounds]

本审阅新增 `large_signed_S`：原生全K864、Q1全−4、每通道T10全1；B交替为(−32,−32)/(31,31)，使S取221184/−214272；A每组仅一项±1，同时设置escape组1、10，稠密系数131071/−131072；完整raw再进入带饱和的I24。两个方向的符号、超过signed18的S、稠密与因子组转换、零A掩码均获覆盖。另复跑原有负最大因子、组内抵消、all-factor/all-exact、padding poison、identity ties/saturation、real0/real2及停顿。[新增角落与运行脚本][rtlcode]

**运算与存储预算。** `product[l]`只有一处19×13乘法表达式，KMAC选择S与A，SMAC选择B与z，BASE_MAC选择缓存Wh与z；8个32位位级加法器也在Q1 packed双P累加和三种MAC间复用。这里“8个乘法器”专指生产者；公共I24消费者另外保留8个32×32乘法及8个64位加法，两个mode均获得这份公共预算，整顶层不能宣传为总共仅8个乘法器。[共享运算选择][corealu]、[消费者运算][consumeralu]

| 项目 | 实际组织和服务 |
|---|---|
| S数据 | 8银行×160地址×signed19，共24320bit=3040B；地址为`fp*4+pair` |
| S live | 40×4bit=20B；SSTORE用8lane实际累加值归约生成，逐项覆写，没有TB预告 |
| S构建 | 每个160地址一拍SINIT、一拍SSTORE，非零z每项一拍SMAC；零输入仍付320拍 |
| S读取 | KMAC从8银行同一地址各读19位，共152bit，每拍一次8lane读取并完成MAC；`s_reads`逐次加1 |
| S端口约束 | SSTORE写与KMAC读分时，不需要同周期多地址读写；当前是组合读数组，经选择、乘法、加法到acc，未证实同步SRAM可保持同周期 |
| A/B | A为48×signed13=78B，B为16×signed6=12B；ALOAD每组读4个A共52bit，SMAC使用8个6bit B寄存器值 |
| Wh及缓存 | 两mode共用完整8×96×19bit=1824B Wh；qblock为64×19bit=152B；候选没有移除完整Wh存储 |

相关源位置为[声明][corewidth]、[组合S读/乘法][corealu]、[SINIT/SMAC/SSTORE][sbuild]、[ALOAD/KMAC][sconsume]。`s_reads`和`s_writes`是8lane向量拍数，不是单标量次数；真实八窗口为8090读拍、1280写拍。B寄存器读取随1467次SMAC发生，虽然没有单独计入`weight_words`，并非未装载的外部免费查表。该RTL时序预算允许一拍组合S读+MAC，控制的z标量读也采用组合读模型；相同抽象服务许可不保证相同物理关键路径，实际面积/频率/能耗未证实。

**运行时费用独立核算。** 令K为非零Q1列数，A为有源输入的非零Q1列数，U为双P时间掩码并集的popcount之和，M0为展开控制的有效MAC数，MS/MK/ME分别为S构建/因子组/逃逸组的MAC数，F为因子组数。每个fixture去除显式停顿后：

```
control = 5437 + K + 2A + 3U + M0
factor  = 5437 + K + 2A + 3U + 320 + MS + MK + ME - 7F
```

常数5437来自原生Q1阶段3420拍和输出阶段2017拍：后者包含96个VLOAD槽、480个POSLOAD、480个STORE、480个DRAIN_READ、480个DRAIN_SEND及FINISH。因子组将8个VLOAD槽变为1个ALOAD，故每组省7拍；它仍付每个P/T的KPOS与STORE。SSTORE真实发生160次，1280次是八窗口累计值，没有遗漏清零或写回费用。128个作者命令以及72个独立复跑命令的计数和周期均被上述原始数据派生的公式逐项核对。[独立公式与核验][mathcost]、[复跑结果][rtlresult]

| 真实八窗口，无外部停顿 | 展开Wh mode0 | 因子/逃逸 mode1 |
|---|---:|---:|
| 有效生产者MAC | 17604 | 12491 |
| S构建MAC | 0 | 1467 |
| 因子组MAC/S读取 | 0 | 8090 |
| 逃逸组MAC | 0 | 2934 |
| S构建总拍数 | 0 | 4027 |
| 第二级权重服务拍 | 756 | 206 |
| 核心自身工作，扣除消费者回压 | 87636 | 84523 |
| 含消费者回压的core_cycles | 106988 | 103875 |
| 冷配置拍 | 14784 | 14904 |
| 完整冷启动total_cycles | 134140 | 131147 |
| 完整warm total_cycles | 119356 | 116243 |

MAC少5113拍，输出组装载省560拍，S的初始化/写回多2560拍，净省3113拍；冷配置另多120拍，最终省2993拍。不能把29.045%的MAC下降当成29%的总周期收益。收益也并非各输入皆正：real2冷启动12223→12466，zero为12112→12377；all-exact仍无条件构建S，冷启动68916→69531。这些是当前调度已实测的代价，并非数据错误。[作者逐命令结果][result]

**冷配置与模式切换修复。** 两mode公共装载Q1=864拍、Wh=96拍、k_live=864拍、I24系数=24拍，共1848个256bit传输拍；候选额外装载A=12、B=2、escape=1，共15拍。每个A拍只用4×13bit，但总线仍服务完整一拍。mode0没有被强制加载这15拍。mode1仍装载全部Wh，因此这里证明的是执行重排的小幅收益，尚未实现因子表示的静态存储/传输压缩。[顶层配置状态][topcfg]

旧顶层只有一个resident位：mode0冷启动完成后resident置1，下一命令mode1直接进SOURCE，A/B/escape均未配置。独立旧快照测试real0、tile0、0→1时，第一命令正确，第二命令raw第0行第0lane得到0而期望844495，退出码20。[修复前失败日志][prefixlog]。这是公共mode接口上的功能bug，不是只补计数即可解决。原TB虽然检查两次命令，却将两次都设为同一个mode，故128命令全PASS未能排除它。[原TB][tb]

作者新增factor_resident，reset清0，首次factor配置完成置1，已有公共参数但无factor时仅进入kind8..10。本审阅复制修复版重新编译，加入实际0→1与1→0序列：32个跨mode命令包含停顿，0→1第二命令精确服务15拍，1→0第二命令服务0拍；另外正常mode/角落命令一起共72命令，raw/J/I24各276480值通过，所有source/parameter/result背压检查保留。[修复][topcfg]、[同步的生成器][generator]、[独立运行汇总][rtlsummary]。`audit_kron/pre_fix`保留旧错误快照供反例复现，`post_fix`为已通过快照；它们不是新的候选实现。

**完整消费者与测试性质。** TB只把source.bin、静态参数和FP32 identity送入请求接口，raw.bin、identity.bin和gold.bin仅作观察比较。全480个N8输出拍即3840个标量逐一验证；Q1扫描全部864列，四P/T10都输出。[TB数据接口][tb]、[原生K终点][kfull]、[输出存储/退休][output]。公共消费者先把每个FP32 identity转换到signed32 Q20，合流raw和identity，计算signed64 `p*a + (b+j)*2^20`，最后一次RNE26并夹到I24；有独立identity转换RNE，但在Q1→S→p之间不存在中间RNE。[消费者转换/舍入][round]。新函数质量脚本在每帧的完整张量上重新执行上述整数p和I24公式，再注入父网络；其余网络仍有FP/TF32路径，文件明确`fullnet_bittrue=false`，与局部RTL逐值证据的范围不同。[质量代码][qualitycode]

**最近邻与新颖性评估。** HKPRNN（Thakker等，2019）Algorithm1直接以reshape后的`B X Aᵀ`计算Kronecker矩阵向量积；§3.3/Algorithm2把不受约束的稠密上部与Kronecker下部合并，以保留部分输出的表达能力。[HKPRNN 原文](https://arxiv.org/html/1906.02876v1#S3.SS3)。把本地两组逃逸行排列到前面，余下10组A行仍构成一个Kronecker块，数学结构即落在这个已有混合形式内；这一映射是本审阅的推导。GKPD还提供卷积场景的广义Kronecker分解来源，但“稠密保留+Kronecker”最近邻应明确到HKPRNN，不能只列GKPD泛称灵感。[GKPD 原文](https://arxiv.org/html/2109.14710v1)

本地借入的是上述分解/收缩机制；没有复现HKPRNN的RNN单元、完整训练与Arm CPU系统，也没有复现GKPD整套卷积压缩/训练评估。可明确描述的本地差分为：训练I24增益加权的固定整数A/B、N8整组选择、19位S的分时银行访问、稀疏z/S跳过、生产者8lane共享后端和原生完整消费者计费。这些差分产生了可检查RTL，但目前只证明特定抽象时序下2.23%冷周期下降。

按“0=已有机制直接实现，5=独立新接口并证明可用优势，10=广泛验证的显著新原理”标尺，**当前新颖性给2/10**：分解数学0/10；本地接口组织2/10。依据是已知混合矩阵形式可直接映射到当前结构，N8银行化/复用尚未显示超出常规实现选择的不可替代性；分数不是替代上面的位宽、费用和强控制分析。应保留这个负质量/小周期正点及修复过的模块证据；当前结果不支持论文主贡献、同质量加速、PPA优势或扩大64/825帧运行。

复现入口：在 `audit_kron` 执行 `/opt/anaconda3/bin/python3.12 audit.py` 和 `/opt/anaconda3/bin/python3.12 rtl_audit.py`。后者只复制当前候选到自身post_fix并重新用Verilator4.028 `--cc --exe`构建；旧快照原失败日志、当前结果、独立角落都位于审阅独占目录。旧实验树、作者候选源码与生产文件未被本审阅修改。

[sum]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/SUMMARY.json
[quality]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/quality_diverse.json
[mathcode]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/audit_kron/audit.py:97
[mathresult]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/audit_kron/math_results.json
[fit]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/fit.py:7
[prepare]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/prepare.py:20
[corewidth]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/kron_core.sv:19
[corecfg]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/kron_core.sv:108
[fitbounds]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/fit.py:37
[rtlcode]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/audit_kron/rtl_audit.py:10
[corealu]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/kron_core.sv:63
[consumeralu]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/i24_consumer.sv:93
[sbuild]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/kron_core.sv:180
[sconsume]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/kron_core.sv:218
[mathcost]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/audit_kron/audit.py:47
[rtlresult]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/audit_kron/rtl_results.json
[result]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/results.json
[topcfg]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/consumer_stream.sv:289
[prefixlog]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/audit_kron/pre_fix/real_0_mode2_stall0.log
[tb]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/tb.cpp:17
[generator]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/generate_rtl.py:23
[rtlsummary]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/audit_kron/rtl_summary.json
[kfull]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/kron_core.sv:174
[output]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/kron_core.sv:234
[round]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/i24_consumer.sv:46
[qualitycode]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/kron_escape/evaluate_quality.py:24
