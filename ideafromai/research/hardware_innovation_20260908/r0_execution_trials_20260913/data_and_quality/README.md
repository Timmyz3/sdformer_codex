# r0 连续源、固定剪枝与质量诊断

正式的真实连续源捕获、三份固定有损掩码、整数函数核对和 **五臂 diverse10 AEE 全部完成**。physical25、magnitude25、coordinate25 的 AEE 分别为 **1.190684、1.185997、1.172116**，均低于当前历史 NB0 门1.45460286107；physical费用布局在本次质量和读取词上都没有胜过同组幅值控制。没有重新训练或扫描剪枝率。

## 真实捕获合同

正式入口为 Python **3.12.7** 的 owned [py312/bin/python](py312/bin/python)，Torch 2.7.1+cu128、torchvision 0.22.1+cu128、CuPy 13.6.0、NumPy 1.26.4，GPU 为 RTX3090。cp312 PyTorch/CuPy 轮子安装在 owned 目录；CUDA 共享库从已有环境只读复用。早期 Python3.10 捕获移至 `rejected_py310/` 并标为不接受，不进入任何正式 RTL fixture。

[model_access.py](model_access.py) 复用现有 `ParentNetwork` 和 `profile_current.py` 的 matched-dense stage320 安装步骤，只重定向代码、检查点、配置和数据路径。没有改 YAML 标量；检查点 strict=True 加载成功，window=[2,15,15]、T10、Motion-XOR α=0.125。实际 r0 模块是：

`sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0`

原生模块入口为 `[T10,B1,C96,H240,W320]`，后接3×3 stride1 pad1 卷积。捕获首个 diverse10 frame `zurich_city_09_a_0001.npy` 的 8 个连续源4×4块，四个角点加四个内部位置。完整 T10/C96/N96，输出每块2×2；不是完整层周期范围，也不是把旧64个离散3×3patch拼成连续块。

输出原点固定为 `(0,0),(0,318),(238,0),(238,318),(32,48),(60,80),(120,160),(180,240)`；输入原点各减1。边界越界源在捕获接口前填0，并保存 `source_valid_yx` 与原点，RTL 作者可独立生成地址/边界门。

[r0_contiguous_t10.npz](r0_contiguous_t10.npz) 与 [capture_manifest.json](capture_manifest.json) 的主要字段：

| 字段 | 形状/意义 |
|---|---|
| source_bits | bool [8,10,96,4,4]，轴 tile,T,C,y,x |
| source_fp32 | 同形状，真实神经元输出 |
| source_valid_yx | bool [8,4,4]，外图边界 |
| output_fp32 | [8,10,96,2,2]，原卷积输出 |
| weight_fp32 / weight_q16 | [96,96,3,3]，真实权重 / signed16 Q16 |
| golden_accum | int64 [8,10,96,2,2]，完整 K864 直接整数卷积 |
| input_origin_yx / output_origin_yx | 原生4×4源与2×2输出块位置 |

实际 sn2 输出只有 `{0,1}`，静态 θ=1 在全 T10/完整73,728,000个元素上验证最大误差0。全层非零数2,663,435（3.6127%），8块非零2813。卷积无 bias。定义 `Wq = round_even(θ·Wfp32·65536)`，范围[-13925,16243]，可装signed16。所有整数 golden 使用该同一 Wq；Q16 是本轮诊断函数，不继承旧 FP32/整数全网准入。

[leaf_identity.json](leaf_identity.json) 验证：W 与旧 r0 profile NPZ 直接 `np.array_equal=True`，两个捕获域重合的首/末3×3源和首/末卷积输出也均逐值相等。旧 profile 首帧 AEE 为1.381253130152104，本次原模型首帧为1.3208328825980986；文件名和 valid_pixels=33225 相同。这个**整网首帧差异仍未解释**，不能将本次指标写成旧环境的重复或校准到旧数值。另将远端原始10帧共30个event/GT/mask复制到owned数据镜像；本地原有的3帧9个文件全部与远端逐数组相等（[data_transfer.json](data_transfer.json)），因此差异不来自这些数据。源/叶的直接核对支持本轮局部 RTL 数据使用。

## 固定三布局

[prepare_masks.py](prepare_masks.py) 一次生成所有掩码，记录完整分数到 [selection_scores.npz](selection_scores.npz)。[mask_manifest.json](mask_manifest.json) 保存选择规则、范围和统计。

原生组为 O8×C4×完整3×3，共12×24=288个 live 位，固定删72组。物理费用按 strongest weight-major 调度定义：

`C[g] = Σ(tile,c∈C4,tap∈9) any_{T10,输出P4} source(t,c,p+tap)`。

每项是一个真正的128bit权重词请求（8输出bank并行），没有乘8，也没有用source popcount或两个C的OR代替每C需求。分子是该组完整 T10/P4/O8 卷积贡献的平方和；以分子/C[g]最小的72组删除。零费用组设无限分数，平分按固定 O组/C组顺序。此规则是一次校准探针，未实现全局误差最小化或训练恢复。

幅值控制删除同72个物理组，按 Wq 平方范数排序。由于 Q16 原已有7个零系数，候选与控制剩余实际 nnz 为62203与62202；因此准确身份是**同物理组数控制**，不能说完全同 nnz。

Winograd 坐标掩码为 `[Ogroup12,ξ16]`，每个 O8 对全 C96 一起删除4/16个坐标。它不反投影回3×3。坐标分数使用该坐标真实 U4·V 经逆变换后的单独输出贡献能量，只选一次。其代码没有复用原生组 mask。

| fixture | live | Wq 非零 | weight-major 剩余词 | native-source 剩余词 |
|---|---:|---:|---:|---:|
| [dense_q16.npz](dense_q16.npz) | 288/288 原生组 | 82937 | 23580 | 41088 |
| [physical25_q16.npz](physical25_q16.npz) | 216/288 原生组 | 62203 | 17863 | 31843 |
| [magnitude25_q16.npz](magnitude25_q16.npz) | 216/288 原生组 | 62202 | 17825 | 30977 |
| [coordinate25_q16.npz](coordinate25_q16.npz) | 每O8为12/16坐标 | 另一个函数 | 由Winograd RTL实测 | 不适用 |

上述词数是同8个**校准窗口**的真实请求账本，不是 RTL 周期。physical mask 比幅值控制少取消38个权重词，不能声称物理收益更强；它只具有较低的校准输出扰动。校准空间、RTL窗口相同，且校准帧与 diverse10 第1帧重叠，没有空间留出或独立泛化。

## Winograd 函数与舍入边界

`BT=[[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]`，`AT=[[1,1,1,0],[0,1,-1,-1]]`，`G2=[[2,0,0],[1,1,1],[1,-1,1],[0,0,2]]`。

`U4=G2·Wq·G2ᵀ`，`V=BT·S·BTᵀ`，`Z4=AT·ΣC(mask·U4·V)·ATᵀ`。ξ=i·4+j，输出phase=a·2+b，源位置p·4+q。无mask Z4严格等于直接整数卷积×4；有mask后双方同函数输出为 `RNE_even(Z4/4)`。8块的变换与4×4展开整数完全一致，且无mask版本与原生 golden×4一致。

AEE helper 使用 phase kernel：

`E4[n,a,b,c,p,q] = Σij AT[a,i]·AT[b,j]·mask[O8,ij]·U4[n,c,i,j]·BT[i,p]·BT[j,q]`。

解码 `E4/(4·65536)`，对偶数 H240/W320 运行4×4卷积、pad1、stride2，再 pixel_shuffle(2) 恢复2×2相位。没有丢失phase或使用3×3近似反变换。原 U4 范围[-76576,95214]，masked E4 范围[-76576,90645]。

浮点 AEE保留父模型TF32设置（matmul/cudnn均true）和浮点消费者，并未施加 RTL 的最终RNE2。单校准叶 FP32 CUDA 与理想未舍入整数输出最大误差5.4931640625e-4，与RTL RNE结果最大误差5.53131103515625e-4。故这里是**相同量化系数的整数线性 RTL + 浮点消费者质量诊断**，不声称全网bittrue，也不能把少于一个Q16 LSB的理想RNE差当作全部实际差异。

## 质量实际完成范围

[evaluate_quality.py](evaluate_quality.py) 已在同一Python3.12/3090环境完整运行原diverse10五臂，详见 [diverse10.json](diverse10.json)、[逐帧文件](diverse10_aee/)。每臂10帧、516735个valid pixels；真实模型、原GT/有效mask、preds.2时间求和与480×640双线性恢复、帧等权AEE口径未更改。

| 函数 | diverse10 AEE | 除校准首帧的9帧AEE | 低于历史NB0门 |
|---|---:|---:|---|
| 原FP32 parent | 1.157519653249069 | 1.1393737388769545 | 是 |
| dense Q16 | 1.163296947981814 | 1.1373011284377197 | 是 |
| physical25 Q16 | 1.1906838362998358 | 1.1878059216404162 | 是 |
| magnitude25 Q16 | 1.18599666968428 | 1.1754742772839517 | 是 |
| coordinate25 Q16 | 1.1721160357095002 | 1.1541872264812645 | 是 |

本轮历史NB0 diverse10门仍是 **1.45460286107**，valid825门仍是 **1.44535253468097**。这只是十帧质量条件，不是硬件/valid825/全网bittrue准入。校准与第1帧重叠，9帧另列，但只运行固定布局，未以验证结果选择阈值或布局。physical25比幅值控制多0.004687 AEE，也多38个校准weight-major请求，因此本固定费用目标没有形成更好的质量/周期权衡。

无剪枝Q16已使任务均值从1.157520变为1.163297，不能把所有后续误差归因剪枝。单校准帧五臂原始诊断保留于 [diagnostic_frame0.json](diagnostic_frame0.json)，它的掩码排序不能代表十帧排序。

最初本地缺7帧21文件，首次完整任务在第2帧失败；[quality.json](quality.json)、[quality.log](quality.log) 和 [protocol_audit.json](protocol_audit.json) 保存这次未完成记录。之后通过已有合法SSH连接读回原固定10帧30文件到 `data_mirror/`，没有替换样本。当前正式结果以 `diverse10.json` 为准。另复制原NB0 ep29检查点，使用原PSN/SDSA、原最终头、无运行统计BN，在相同本地环境复测；实际十帧均值 **1.462941239319642**，同516735有效像素、78处BN无运行统计、ATLIF/Shiftmax均0，加载无缺失/多余键；结果见 [nb0_diverse10.json](nb0_diverse10.json)。五臂也全部低于这个fresh NB0。历史门1.45460286107仍保留为更严格的比较线，未用新值放宽门；NB0新旧差异0.008338，差异原因仍未定位，相同数据排除这批输入文件差异，但没有证明全网代码和CUDA执行完全相同。不能称历史结果的重复复现。

[run.sh](run.sh) 用明确的owned Python3.12解释器依次捕获、固定mask和五臂diverse10；[fetch_data.py](fetch_data.py) 记录本次只读原数据来源。未恢复+0.005，未跑valid825，未重训，未跑EDA/hash。实际RTL周期与完整供数/状态开销由相邻 [native_sparse](../native_sparse/) 和 [winograd](../winograd/) 作者的结果给出，本目录不把请求账本冒充周期。
