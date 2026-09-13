# PED Kronecker 有界 RTL 独立复审

**修正后结论：前次提出的整零权重字控制缺口已关闭。** 最新 [ped_kron.sv](../kron/ped_kron.sv) 为全部模式加入相同的配置期 live mask 和 24 位 next-live 选择器；直接模式现在实际跳过 expanded_k1 的 36 个全零 128-bit 字。Kron1 与自己整数展开矩阵的同函数比较，更新为无背压 **93,824 → 33,394 cycles，节省 64.4078%，2.8096×**；同背压规则 **96,103 → 35,321 cycles，节省 63.2467%，2.7208×**。此前 3.1546× 是未跳整零字的旧直接控制数字，不能继续当作当前最强已测同函数对照。Kron2 同函数对照不变。

本轮只读新旧源码、[static_zero_words.json](../kron/static_zero_words.json)、[resource_contract.json](../kron/resource_contract.json)、[rtl_results.json](../kron/rtl_results.json)，并重算已有计数关系，没有执行新 RTL、拟合、训练或 EDA。原 10 个 case 已由实现代理重跑，最新结果记录 307,200 个连续整数值零差异；下述结论是对源码和这些实际结果的独立复审。旧实现/结果留在 [old_control](../kron/old_control/REPORT.md)，本文件后半保留首次审查的历史记录。

**共同控制权限成立。** `cfg_valid` 接受一个 128-bit 系数字时，RTL 同时执行 `coeff_live[cfg_addr] <= |cfg_data`；没有从 TB 接收免费 precomputed mask。288-bit mask 由同一 coefficient 配置拍产生，所有模式仍支付原本的系数配置输入；expanded_k1 仍配置 288 words，没有将跳过运行读误计成删除配置读。总本地状态由 45,440 增至 **45,728 bit**，增加的 288 bit 已列账；128-bit OR 逻辑和控制选择器不是免费硬件，但本轮没有面积测量。

运行时直接模式填充 24 个 loop-live 位，A 收缩填充 6 位，B 收缩填充 4 位，全部进入同一 24 位选择器。不同有效长度来自实际 contraction 的 K 长度，不是直接模式被降低读口或候选独占跳零权限。factor A 的物理字含 4 个有效 A 系数和 padding，mask 对完整 128-bit 字取 OR；即使当前 group 的部分 A 系数为零，只要同字其他系数非零也不跳。这一实现遵守共同的整物理字粒度，没有暗中给候选逐 lane 或更细粒度跳零。

**地址与跳过后终止逻辑成立。** `eligible = loop_live & (0xffffff << k)` 保留当前起点及以后的位置；priority encoder 选最小 live `next_k`，输入源索引、权重地址都同步改用 `next_k`。执行后存在其他 live 位才令 `k=next_k+1`，否则直接 COMMIT；因此不会重复读取某个 live word，也不会从被跳过 k 取错 src。每组 CLEAR 把 k 归零，其是否进入 MAC 只由完整 loop_live 判断，避免沿用上一组结束位置。若整个 loop 为空，CLEAR→COMMIT：初项保持零，Kron2 第二项 B 收缩保持已取回的第一项结果，源码路径合理。实际五套系数没有整 loop 为空，故该分支属于代码审阅覆盖，不能声称此次真实 fixture 已测试该边界。

`coeff_live` 未在 reset 清零是可接受的配置合同：TB 在 start 前配置每种模式将引用的全部 words；direct 为 0..287，Kron1 为 0..17，Kron2 为 0..35，未配置地址不会被相关循环引用。通用接口仍要求模式使用前完成所需系数配置，不能只依赖未初始化 mask。没有证据表明当前测试违反该合同。

**算术函数没有变化。** 新旧源码 diff 只改变 live-mask/地址和 MAC 次数；8 个 signed32×signed16→signed48 MAC、latent32 写回、Kron2 分项累加和最终 RNE15/sat24→bias→sat24 均保持。只跳过整个零权重字对应的八个零乘积，对已证明无溢出的整数和严格无损；前次的展开等价、latent32 和 acc48 上界证明继续有效。TB 新增了运行 CR 读计数，并断言 MAC 状态必须有真实 live word；输入/输出和输出背压稳定性检查保留，没有把 contraction 移至 TB。

**新计数关系通过。** 独立对全部 10 行核对 `total=IDLE+LOAD+CLEAR+MAC+COMMIT+OUTPUT`，以及 `coefficient_read_words=MAC`。expanded_k1 的运行 MAC/CR 读从 92,160 降至 **80,640 = 252×320**；无背压总周期准确减少 **11,520 = 36×320**，其它阶段不变。带背压时输入/输出命中相位改变，所以总差不是单纯 11,520；对应 stall 列已保留，96,103 的周期账也相符。original、expanded_k2、kron1、kron2 没有全零已配置物理字，运行 cycles 与前次相同，这与零字清单一致。

**保留的边界。** next-live 选择器位于 mask 选择→priority/地址→异步 CR→乘加的组合路径上，mask OR 也增加配置路径逻辑。本次 cycles 假设它能在一个既定周期内完成，未有时序收敛、频率或面积证据；不能把周期减少直接等同于相同 Fmax 的真实时间/能量减少。三种模式使用同一选择器和资源合同，故本地周期比较没有新的候选独占特权，但这仍不是所有可能控制/调度的全局最优证明。

当前试验仍是 matched-dense stage320 独立快照的完整 96×24 后因子，不包含 U 前因子、整个时间混合、全层或网络。Kron 拟合对原 V 的函数改变与较大本地误差不因控制修正而消失，Kron2 本地失真仍差于同参数 low-rank2。可以更新为“相对同函数、含共同静态整零字跳过的直接执行，局部 RTL 周期显著降低”；不能更新为原网络无损、光流精度达标、新算法原理或端到端部署成功。

**以下是首次审查的历史记录；其直接控制缺口、旧资源数及旧周期比已由上文取代，保留用于追溯。**


结论：所审版本的整数功能、共同 8-MAC 数据路径、端口和主要状态费用成立；未发现通过减少直接模式的 MAC 数量或可用读口来制造收益。**Kron1 的展开矩阵直接对照仍有一个更强的简单控制缺口：36 个全零 128-bit 权重字没有跳过。** 当前 3.1546× 可表述为相对这份完整密集展开执行的 RTL cycles 比，不能表述为已经战胜最强直接稀疏执行。另一个根本边界已由对方 `SUMMARY.json` 正确披露：拟合后的 Kron 矩阵改变原函数，且本地失真很大；没有 AEE 或 PPA 证据。

本复审只读了 [ped_kron.sv](../kron/old_control/ped_kron.sv)、[tb.cpp](../kron/old_control/tb.cpp)、[fit_and_fixture.py](../kron/fit_and_fixture.py)、资源合同和结果文件，并独立用 NumPy 对文本 fixture 重构系数、原始整数累加、最终量化及安全位宽。没有改动 kron 文件，没有新增或重跑其 RTL、训练、EDA。该试验源是 `breadth_20260912/.../hardware_exports/dense` 的 matched-dense stage320 独立快照；本目录 interval 主测是 ordinary R24+onepass 另一快照。二者不能仅按目录名推断父学生关系或训练先后。

**整数展开和因子化一致，复审通过。** 直接执行的系数装载次序是输出 8-channel group、输入 k、lane。独立恢复后，`expanded_k1/k2` 的 96×24 矩阵与 `Σr kron(Aq[r],Bq[r])` 分别逐系数零差异。A 的量化指数 7、B 的指数 8，乘积指数 15；RTL 先完成 `Aq × input24`，保留完整整数 latent，再乘 Bq，最后只做一次 V 边界 RNE15/sat24，加入原 bias 后再 sat24。Kron2 第一项先写 result，第二项在 CLEAR 从该 result 取回作累加初值，直到两项都完成才输出，没有逐项 RNE。

独立重算的 320 向量覆盖 original、expanded_k1、expanded_k2、kron1、kron2：因子化 raw accumulator 与各自展开的 `X×Wqᵀ` 零差异；最终 output 与对应 golden 零差异。原 V 与 Kron 拟合矩阵不是同函数；严格等价只发生在每个拟合矩阵的展开和因子执行之间。已经记录的 10 行 RTL 测试（5 模式×2 背压条件）共 307,200 值零错，与这一代数边界一致。

**latent32 的截位在本系数下不丢精度，复审通过。** [SV 第 100 行](../kron/old_control/ped_kron.sv:100) 用 `acc[31:0]` 写 latent32，形式上需要证明安全，不能仅凭实际样本没溢出。本次独立核得对任意 signed24 输入：

| 项目 | Kron1 | Kron2 | 有效上界 |
|---|---:|---:|---|
| A 中间量绝对值上界 | 1,509,949,440 | 1,694,498,816 | 均小于 2³¹ |
| 实际 320 向量 latent 最大绝对值 | 47,660,205 | 56,787,555 | 远小于通用界 |
| 展开系数最大加权绝对和界 | 348,798,320,640 | 560,946,216,960 | 均小于 2⁴⁷ |

Kron2 分项累加的合同使用更保守的 703,602,884,608，仍在 signed48 内；原 V 的对应界为 1,223,679,803,392，也安全。因此这里没有隐含的中间 RNE，也不是以截位偷偷改变展开等价性。安全性依赖已固定的 Aq/Bq；若未来更换或训练这些系数，须重算界，不能把本证明当作任意 Kronecker 因子的通用保证。

**同 MAC/端口的比较基本公平。** [SV 第 32–34 行](../kron/old_control/ped_kron.sv:32) 只有一组 8 个 signed32×signed16→signed48 乘法器，所有模式复用它们及 8 个 acc48。CR 每个 MAC cycle 只读取一条 128-bit word。直接模式用该 word 的 8 个权重对应 8 个输出，广播一个 src[k]；A 收缩每 cycle 最多读 4 个不同 src 并广播给 8 lane；B 收缩广播一个 latent。资源合同为各模式共同预留至多 4 个源读的 flop/mux 权限，直接模式的映射没有使用额外读能力，但没有被减为低吞吐 MAC。

原 V 真实输入非零比例为 100%，系数 2,304 个也全部非零。原 V 直接模式的 `2,304/8=288` MAC cycles/向量已经达到“逐系数直接点积、8 个 MAC”下界；额外给它 4 个源读不会让这 2,304 次乘加低于 288 拍。它有一拍 CLEAR 和一拍 COMMIT/8 输出组，因子模式也支付相同阶段；因子模式还有 latent 写回和第二项 result 取回。没有发现只给直接模式加空泡、限制权重带宽或不给结果状态的情况。

共同状态的 45,440 bit 不含控制，包含 coeff、bias、src、latent、result 和 acc；直接模式虽不用 latent，仍预留相同容量。RNE 增量和 bias 加法位于共同 OUTPUT 组合函数，不属于“8 个 MAC 就是全部算术电路”的说法；输出阶段已计时，但面积、组合路径和 Fmax 未测。运行期间 CR 是片内驻留、异步读取，无动态权重背压模型；TB 对配置输入的阻塞和输入/输出 valid/ready 分别计费。该合同支持本地 cycles 对比，不支持据此推断 SRAM/DRAM 或 PPA。

**Kron1 的最强简单直接控制尚不完整，建议标为中等强度限制。** 独立计数发现：

| 直接执行矩阵 | 系数零数 | 全零 128-bit word 数 | 8-MAC 非零乘加计数下界/向量 |
|---|---:|---:|---:|
| original | 0/2,304 | 0/288 | 288 |
| expanded_k1 | 288/2,304 | **36/288** | **252** |
| expanded_k2 | 5/2,304 | 0/288 | 288 |

Kron1 的全零 word 呈固定范围：输出 groups 0–2 的 k=20..23，以及 groups 6–8 的 k=4..11。它们可以在静态已知系数的直接执行计划中跳过，保持同一输入/权重端口与相同数据位宽；合法计划/索引本身仍要计费。当前 [MAC 控制第 93–96 行](../kron/old_control/ped_kron.sv:93) 对全部 k=0..23 无条件执行，未实现该对照。

若仅作分析，把这 36 个 MAC cycles/向量理想去掉而保持其余费用不变，320 向量的 dense-expanded `105,344` 会成为 `93,824` cycles，对 Kron1 `33,394` 的比值约 2.81×；这只是说明缺口幅度的乐观计数，**不是新增 RTL 测量，也不能替换原始结果**。是否还可以隐藏 CLEAR/COMMIT、在内部压缩零系数，双方都可以继续优化；本轮授权的有界实验不需要为此继续扫参或续跑。报告应直接保留 dense-expanded 标签和静态跳零缺口。

**周期计数自洽，结果范围需要保持。** 无背压时原 V 每向量 `1 start+3 load+12 clear+288 MAC+12 commit+12 output=328`，乘 320 加 `288 coefficient+96 bias` 配置为 105,344。Kron1 为每向量 104 拍，加 18+96 配置，得到 33,394；Kron2 为 192 拍，加 36+96 配置，得到 61,572。均与 RTL 表匹配。18/36 是配置 word 数，**不是每向量运行的 CR 读取数**；运行 CR 访问随每项 60 MAC cycles 发生，不应将配置复用误写成每向量只读 18 次。

TB 在每个配置的 320 个连续输入之间不复位，状态由 RTL 的 LOAD/CLEAR/COMMIT 覆写；没有用 TB 提前算 A 或 B 的收缩。TB 使用独立 golden 检查并核背压时输出保持，输入源是已有 U24 边界的真实连续量。U 的执行、完整 T10 时间混合、全层和下游 BN/flow 不在这个 96×24 后因子的 RTL 计时内。

本地 NRMSE 在 bias 后为 Kron1 0.8993、Kron2 0.7746；相同参数数量普通 low-rank1/2 是 0.9173、0.7363。故 Kron2 在当前输入上甚至不优于同参数 rank2 控制。已有 `candidate_status` 对大失真、X=0、无 AEE 的限制合理。可以确认“等整数拟合矩阵的结构执行显著减少该密集矩阵控制的 cycles”，不能确认原网络可直接部署、无损替换原 V、存在新的算法原理，或具有端到端光流收益。
