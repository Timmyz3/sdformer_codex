# T10 端点加前缀执行：真实接口与有界 RTL 对照

本次完成了一个完整 `P2 × T10 × K864 × N16` tile 的 SystemVerilog / Verilator 对照。当前 ordinary R24+onepass 导出的真实 sn2 脉冲、配套真实 U16 整数权重，在直接跳零和端点加前缀下输出完全一致；端点模式在该 tile 上更慢。无背压为 **2,404 → 3,502 cycles，增加 45.67%**；相同背压规则为 **2,652 → 3,710 cycles，增加 39.89%**。这已经是含清零、完整 K 累加、RTL 前缀、最终舍入、连续输出和背压的结果，并未因统计不利而省略 RTL 验证。

该结论只限于下面明确给出的原始 T 顺序、固定空间锚点、U16 算子和资源合同。它不证明其他层、跨真实输入窗口、运动对齐后的帧间特征复用、经训练形成长区间的布局都无效。这里也没有整层或整网加速、面积、频率、能量或新 AEE 结果。

**真实输入、版本和时间轴。** 主结果读取 [current ordinary capture](../../open_fusion_execution/stage_20260912/algorithm/hardware_exports/ordinary/000_zurich_city_09_a_0001.npz) 和同目录 [deployed_constants.npz](../../open_fusion_execution/stage_20260912/algorithm/hardware_exports/ordinary/deployed_constants.npz)、[live_parameters.npz](../../open_fusion_execution/stage_20260912/algorithm/hardware_exports/ordinary/live_parameters.npz)。最终 GPU R24+onepass 导出与已有局部接口的绑定见 [final_combo_alignment](../../open_fusion_execution/stage_20260912/hardware/final_combo_alignment/README.md)。matched dense stage320 是独立快照，另列统计，绝不与 current 权重交叉混用。

当前数组 `interior_sn2_gate.shape = [10,96,9,9]`，metadata 明确 `tensor_order=T,C,y,x`，gate 的全局原点为 `(119,159)`；对应 4×4 投影输出的原点为 `(60,80)`。捕获实现保留原始 T，原代码 [capture_reference.py](../../algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain/capture_reference.py) 的 `crop` 只去掉 batch=1 并裁空间；`sn2_values` 交给 `save_gate`，逐元素断言非零值恰等于输出 θ（该文件第 30、99、126 行）。本次固定取已有 interior anchor 遍历开头的两个原坐标 `(120,160),(120,162)`，各自完整 3×3×96 邻域；`k=c*9+ky*3+kx`。保持每个真实 k 的原 T10 顺序，随后只把消费者位打成 `bit=p*10+t`。没有将 640 个随机 patch 行重排成时间，也没有按数据好坏重选锚点。

`U_conv2_theta_q16` 是配套的 `[16,864]` int16 系数，exponent=17，当前范围 `[-27128,29504]`。`sn2_theta=1.0`、`preview_theta_output=1.0` 均为标量，参数没有 T 轴。阈值 `tau_t` 随 T 不同与发射幅值 θ 随 T 不同是两件事；前者可以改变 S，后者才破坏这里采用的同一 θW。当前源是完整 T10 PSN 之后的脉冲缓存，本算子的执行起点在源已可读之后；没有声称获得因果的逐事件或逐帧输出。

**同一整数函数。** 对每个空间位置 p、时间 t、通道 n，直接模式计算

`A[p,t,n] = Σ(k=0..863) S[p,t,k] · Uq[n,k]`

`Z24[p,t,n] = sat24(RNE(A[p,t,n] / 8))`。

这与既有 [fixed_temporal_coordinates.py](../../algorithm/patch_probe/residual_consumer_probe/projection_chain/fixed_temporal_coordinates.py) 的 `conv_forward`（第 287 行）相同：二值输入乘 `U_conv2_theta` 后，先完成 dot product，再按 `exponent−STATE_FRAC = 17−14 = 3` 位写 signed24。它是既有部署的整数子函数，不能据此宣称与原 FP32 卷积逐位相同；U16 本身也只是该恢复结构中的 rank16 子算子。

端点模式令 `S[p,-1,k]=0`，`D[p,0,k]=S[p,0,k]`，`D[p,t,k]=S[p,t,k]−S[p,t−1,k] ∈ {-1,0,+1}`。先算完整 `ΔA[p,t,n]=Σk D·Uq`，随后在 RTL 内依时间做 `A_t=ΔA_t+A_(t−1)`。**所有前缀在最终 RNE/sat24 之前完成**，没有对 ΔA 提前取整。只请求 t=0..9，因此无需计算窗外 `D_10=−S_9`；区间统计仍分别记录窗内下降端和末尾未关闭区间，避免漏计被误认为算术收益。

静态 θ 可以直接吸收进 W；该变换要求有效 θW 在 T10 内一致。若 θ 随 t 改变，`S=[1,1]` 对应的幅值也可能改变，单靠 `D=[1,0]` 无法恢复目标输出；必须差分真实幅值或计入参数变化项。本次不做这种有损或变权扩展。

累加器使用双方相同的 signed32。对任意本格式输入，完整 K dot product、任意部分端点累加的绝对值都不超过 `864×32768=28,311,552`；前缀完整结果通过望远镜求和等于二值 dot product。signed32 安全。舍入采用有符号算术右移的商、低三位余数、ties-to-even 增量；最终 signed24 饱和也在 RTL。此处范围内饱和不会被触发，饱和边界并非本次数据覆盖的功能分支。

**真实脉冲和区间。** 完整可审计统计在 [statistics.json](statistics.json)，准备代码在 [prepare.py](prepare.py)。表内单位是每个 `(p,t,k)` 二值事件；一个权重请求组含固定 k 下共享同一 16 元权重的 P2×T10 消费者。

| 数据/相同固定 tile | 脉冲 A | 窗内非零端点 E | 区间数 | 长度 1 / 2 / ≥3 | 平均长度 | union-K |
|---|---:|---:|---:|---|---:|---:|
| 当前 ordinary R24+onepass | 557 | 1,088 | 544 | 531 / 13 / 0 | 1.0239 | 305 |
| matched dense stage320 独立快照，仅统计 | 550 | 1,072 | 536 | 522 / 14 / 0 | 1.0261 | 300 |

当前脉冲密度 3.2234%，端点密度 6.2963%；每个 t 的脉冲计数为 `[34,15,14,10,333,48,0,99,4,0]`。当前最后时刻没有未关闭区间，因此 `E=2×544`。两种执行法的 union-K **逐位相同**：某个二值序列只要出现过 1，从初值零出发就至少有一个上升端。因此，在本次强直接基线已经跨 T10/P2 复用权重后，端点并不节省任何权重向量读取。

**被实际执行的完整数据路。** [interval_tile.sv](interval_tile.sv) 包含清零、源读取、可选统计扫描、选择、权重读取、稀疏加/减、原位前缀、RNE/sat24、valid/ready 连续输出。 [tb.cpp](tb.cpp) 只提供源和权重 ROM、端口阻塞、golden 检查；它不生成 D，不累计中间和，不清零内部状态，不做前缀或舍入。输入 mask 的空间 gather/打包来自准备脚本，这一生产者格式转换在算子入口之外，对两模式相同，未计作硬件服务。连续 signed24 输出的 40 个 8-lane beats 全部在 RTL 产生。

| 固定资源/接口 | 直接与端点共同权限 |
|---|---|
| 数据加法器 | 8 条显式 32-bit carry chain；加减、前缀、舍入增量在同 8 条链上分时执行 |
| 累加状态 | 320×32 bit = 1,280 B；40 组×8 words，双方预留两组读、一组写权限 |
| 源请求 | 每 k 一拍 20-bit P2×T10 mask，可阻塞；完整 K 扫描 864 次 |
| 权重供数 | 单次 256-bit 向量，32 B 寄存器；每个 union-K 一次，跨全部 T/P 复用 |
| 端点和索引 | RTL 组合 XOR/方向判定，20-bit active、20-bit negative、20-way priority encoder；不依赖免费区间表 |
| 连续输出 | 每拍 8×signed24 =192 bit，24 B 输出寄存器，停顿时数据和索引保持 |
| 状态生命周期 | 每命令 40 拍清零；不同命令间不复位；完整 tile 后才输出 |

没有 SRAM 宏实现、布局布线或频率测量；组合 endpoint decoder 和 priority encoder 的时序不是凭 cycles 证明的。8 条数据路径加法链双方相同，控制用 popcount/计数逻辑另列为选择器开销；不声称选择逻辑没有面积成本。当前阵列语义允许前缀两组状态读，直接模式也预留同样权限，没有限制直接模式的端口。

固定模式 0/1 由调用者给定。模式 2 是实际 RTL 的**整 tile 选择**：先付费完整扫描 K，统计 A/E，再用 `2E+36 < 2A` 选择端点；相等时选直接，然后再次读源执行。额外无背压服务为 864 次 mask 读取 +1 拍选择。不存在按结果预先给出的免费模式。选择器没有预测未来背压，采用算术事件数作为合法但非最优的控制。在未增加独立状态前，不能随意对每个 k 混合直接和 delta 累加后再统一前缀；直接贡献也会被再次累加，所以本版刻意只支持整 tile 选择。

**实测 cycles 与功能。** [rtl_results.csv](rtl_results.csv)、[simulation.log](simulation.log)、[result_summary.json](result_summary.json) 保留原始结果。下表均是同一当前真实 tile，done 包含最后一拍结束控制。

| 模式 | 清零 | 源读 | 权重读 | 稀疏向量加/减 | 前缀 | 舍入 | 输出 | 总 cycles，无背压 | 总 cycles，同背压 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 固定直接 | 40 | 864 | 305 | 1,114 | 0 | 40 | 40 | **2,404** | **2,652** |
| 固定端点 | 40 | 864 | 305 | 2,176 | 36 | 40 | 40 | **3,502** | **3,710** |
| 付费选择→直接 | 40 | 1,728 | 305 | 1,114 | 0 | 40 | 40 | **3,269** | **3,587** |

无背压时 `Cdirect=40+864+305+2A+40+40+1`，`Cendpoint=Cdirect+2(E−A)+36`，差值 `2×531+36=1,098 cycles`，与 RTL 计数一致。权重读取都是 305×32=9,760 B。选择模式多 865 cycles；背压下相对固定直接多 935 cycles，取决于重新调度后端口可用相位。

阻塞规则按每命令的同一全局 cycle：源在 `cycle mod11=3` 不可用；权重在 `mod7∈{2,3}` 不可用；输出在 `mod5∈{1,2}` 不 ready。双方使用相同规则，工作调度不同使实际命中的 stall 数不同，因此没有强迫两模式拥有相同 stall 计数。源/权重/输出 stall 原始列全部保留，没有把供数“消失”在 testbench。30 个命令的逐项 cycles 和功能计数由 [analyze_results.py](analyze_results.py) 再核对。

真实输入之外，只使用四个固定全尺寸功能控制：全零、全一、交替、含起始/末尾/中间区间的边界图案；所有控制仍用相同真实 W，作用是覆盖控制路径，不作为真实数据性能样本。共 5 输入×3 模式×2 阻塞条件=30 个连续命令，**9,600 个 signed24 值/1,200 beats 零错**；没有命令间 reset，背压时输出稳定性也核对。全一控制无背压直接 36,409、端点 5,341、选择 6,206 cycles，实际验证了较长区间下端点执行和合法选择能够获益；交替控制端点更慢。没有扫区间长度、tile 尺寸、阈值或训练参数续命。

**与已有工作的差异边界。** 这条公式不是新的差分恒等式。[Sigma-Delta Quantized Networks](https://arxiv.org/abs/1611.02024) §3.1、Alg.1–2、Eq.1 已明确差分/积分与固定线性层交换，§3.3 将稀疏整数差分化为权重加减。本次利用二值 S 让 D 天然三值，完整恢复所选部署整数子函数，属于该已知原理在真实 T10/U16 切口上的有界执行试验，不能把“区间端点”命名当作方法创新。

[LoAS](https://arxiv.org/abs/2407.14073) §III–IV 通过时间内层布局、全 T 位包与 silent-neuron fiber 压缩改善多时刻复用；§IV-B/C/D 还使用 pseudo accumulator 与逐时刻 correction。其 fast/laggy prefix-sum 是压缩 fiber **索引偏移**生成，不是本次 `ΔA_t` 的时间积分，但已覆盖全 T 权重复用与 base/correction 动机。本次直接基线因此保留 union-K 一次供数，不能和重复 T 次取权重的弱直接实现比较。

[Phi](https://arxiv.org/abs/2505.10909) §3.1、Fig.2 使用离线 pattern-weight products 加在线 `{+1,−1}` correction，差分比原始更稠时保留原始稀疏表示；§3.2–3.3 还包括 k-means pattern 选择及微调。本次只有固定时间差分基，无 pattern dictionary、PWP 表或匹配器；符号修正和回退本身已是明确近邻。

[Comperity](https://doi.org/10.1145/3828526) 的 ACM 出版者存入 Crossref 的摘要明确相邻空间 spike rows→共同 Base Vector + Differential Vectors、AND/XOR 和 stateless index-driven 执行。出版日期核为 2026-07-24、TACO。ACM PDF 403，**未取得全文**，不能推断它的全部时序、状态或严格公平配置，更不能宣布本方案避开了其所有权利/方法范围。这里只把“共享基+差分”和空间相邻关系记为已有；本次明确沿真实 T 并支付前缀状态。摘要原始记录留在 [comperity_crossref.json](papers/comperity_crossref.json)。

[DeltaCNN](https://arxiv.org/abs/2203.03996) §3.1、Eq.1/3 强调固定线性卷积可以传递 delta，非线性必须恢复当前状态再计算差。它是跨视频帧的 GPU 系统，不代表当前内部 T10 已有跨帧相关性；也说明不能把端点形式直接越过 PSN、BN 或每步舍入。[Lightweight LIF-only SNN accelerator using differential time encoding](https://arxiv.org/abs/2505.11252) §III–V 编码的是相邻 spike **时间戳差/等待时间**并合并排序，和对二值幅值做 `S_t−S_(t−1)` 的上升/下降端不同，不能因名称相似而混为本次方法。

本次完成的证据是：真实 layout 和 θW 合法性已追到来源；固定 tile 的脉冲/端点/区间和 union 已测；端点生成、前缀、清零、连续整数输出已在等资源 RTL 执行；当前接口没有收益。尚未测到的是更长真实时间区间、跨输入窗口对应关系、运动补偿后冗余、整层负载与生产者格式转换的端到端收益。因此现在可以停止这个固定布局的参数探索，保留复现文件和精确失败边界，而不封死真正跨帧的新接口。

复现命令：在本目录执行 `bash run.sh`。环境为 `/opt/anaconda3/bin/python3.12`（NumPy）和 Verilator 4.028；老版本采用 `--cc --exe` 后独立 `make`，全部输出只落在本隔离目录。未运行 GPU、训练、EDA 或修改生产目录。引用核查细节见 [source_table.md](source_table.md)。
