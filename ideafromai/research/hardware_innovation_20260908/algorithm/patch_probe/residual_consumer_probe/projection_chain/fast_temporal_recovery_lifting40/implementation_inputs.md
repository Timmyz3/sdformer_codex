# Lifting Stage B：同端口服务比较的既有输入

2026-09-10，只读源码/产物定位；没有新训练、捕获、编译、排程或 RTL。Stage B 比较 **ordinary dense-source/raw** 与 **lifting40 `fast_raw_diagonal`** 的同资源净服务。节点数是输入，不是结果。以下路径以 `CHAIN=algorithm/patch_probe/residual_consumer_probe/projection_chain` 为基准；`LIFT=CHAIN/fast_temporal_recovery_lifting40`。

## 1. 锁定数值身份与完整源图

| 项 | ordinary dense-source/raw | lifting40 raw |
|---|---|---|
| 最终训练参数 | `CHAIN/temporal_structured_recovery/stage128x256/identity_permuted_base.npz` | `LIFT/stage320/fast_raw_diagonal.npz` |
| 实际部署常量 | `CHAIN/temporal_structured_recovery/fixed_valid825/identity_permuted_base_coordinate_constants.npz` | `LIFT/fixed_lifting_valid825/fast_raw_diagonal_fixed_constants.npz` |
| 对应完整 825 收据 | 同目录 `identity_permuted_base_summary.json`，AEE **1.219801338299** | 同目录 `fast_raw_diagonal_summary.json`，AEE **1.232979367919** |
| 完整源 DAG | `LIFT/ordinary_source_cmvm.integer_dag.json` | `LIFT/constant_compilation_graphs.json` |
| 编译信息、cutoff、检查 | `LIFT/ordinary_source_cmvm.json` | `LIFT/constant_compilation_result.json`、`constant_compilation.md` |
| 正式数值实现 | `CHAIN/fixed_temporal_coordinates.py:FixedTemporalForward` | `CHAIN/fixed_lifting_coordinates.py:FixedLiftingForward` |

两者均为新 fixed-coordinate 学生，825 均为 48,152,523 有效像素，不能标同函数。差值 +0.013178030 也不能被 source 节点减少抵消；当前任务只补服务判别。普通出口 As 为 `As_q16[10,10]`、指数 15；lifting 用 `lifting_q12[4,5,2]`、指数 12、`lifting_matchings[4,5,2]`。后者 `source_A/basis_B` 是无中间 RNE 的实矩阵说明，**不可替代执行图**。`gate_collapse_probe.*` 是另一个已停止的近似全合并布局，不是 Stage B 输入。

普通整图 260 加减，66 个直接共享节点，算术结果位宽总和 8606，最大 41 位。JSON 的 `static_orders.official` 和 `pressure_heuristic` 可作同权限固定调度输入；后者暂存峰值 1981 bit/单个 T10 通道上下文，但假定出口随时可接收，尚无端口/背压闭合。不能把它直接当物理 RF 容量或周期。

lifting 取 bundle 的 `whole_halfstage_graphs['fast_raw_diagonal/forward/0'..'7']`，每项有 `layer/coefficient_half/write_time_indices/whole_five_pairs_graph`。图输入 `source=0..9` 指当前时间坐标，输出 `t=0..4` 必须映射到该项 `write_time_indices`；经 RNE/sat 后才成为后续图输入。总 159 加减，结果位宽总和 5143，完整 carry 位宽总和 5234，最大节点/分子 38/39 位。159 已包含恒等项，不能再加 40 次。各节点 `lhs/rhs`、精确 shift、subtract、output sign/shift 和 fanout 都已保存；同一父被左右边重复引用时，引用次数与物理独立读取次数须按双方相同的转发规则处理。

**必须保留的 35 个 RNE/sat：** 每半步 `N=(old<<12)+q12*other`，完整分子舍入后夹到 signed24。原程序 40 次数值写回，最后 layer3-b 的 5 个值只供源门，可合并成分子 cutoff。对应时间坐标 `[6,7,8,9,5]`、源门 `[0,7,8,1,9]`。其它 35 次仍含 guard/sticky/parity、条件加一、饱和选择以及依赖延迟；可转发而不落 RF，但不能删除数值语义。普通的全部源 Q 出口也只供门，10 个 RNE/sat 均可折入 `ordinary_source_cmvm.json.postprocess.rows` 的 exact cutoff。负 gain、`<=` 和等号完整保留；不要给 lifting 免费的专用舍入端口。

## 2. raw 的完整 r1 服务边界

实际模块根是 `sttmultires_unet.encoders.swin3d.patch_embed`，`r1=residual_encoding.resblocks.1`。两轴共同执行：

`I24 → source temporal graph → sn1 θg → preview Conv1 U32/V → fixed BN1 → full sn2 → θg → Conv2 U16 → F → anchor merge(I+c) → {proj.sn gate, PED U32/V32 continuous} → native proj spike Conv/add`。

* 源输入全 T10/C96/240×320；每空间点保留的 raw I 是 2880 B，P2 为 5760 B，P4 为 11520 B。两者都保留 I，lifting 的可覆盖工作坐标不能破坏后继仍需的原 I；DAG 临时量、输出 gate 队列另计。raw **不执行 inverse**。
* `r1.sn1.spiking_neuron → r1.conv1.0`：3×3、padding1、stride1，preview 有 32 个有效 latent。虽然 Conv2 只保 even/even anchor，但其 3×3 源邻域并集覆盖全空间，因此 sn2 和 preview Conv1 仍需完整 240×320。
* `r1.sn2.spiking_neuron → r1.conv2.0`：实际只计算 even/even 所需 U16 输出，即 120×160；源仍为完整 240×320。取 `U_conv2_theta_q16[16,864]`，theta 已折系数，不能重复乘。k 顺序 `(c*3+kh)*3+kw`。
* `F_q16[96,16]` 已含固定 BN2 gain；`BN2_constant_q24[96]` 在 anchor 合并一次。非anchor 整个 BN2 branch（包括常量）已共同删除，只对 I 作 consumer permutation/阈值判门。
* `proj.conv_res` 的 1×1/stride2 连续消费者由 `U_ped_q16[32,96] → V_ped_q16[96,32]` 实现；anchor merge、U、V、bias merge 各自的 signed24/f14 RNE/sat 不能任意跨越线性融合。`proj.sn` 的门消费者同时保留；输出 θg 喂实际 96×96×3×3/stride2 的 `proj.conv`，最后还有 PED 两支相加。
* 两轴后继指数一致：U16=17、F=14、U_ped=16、V_ped=15。真实非零数 ordinary 为 13824/1536/3072/3072，lifting raw 为 13824/1535/3071/3072；静态零省略双方同给。当前 source/consumer/sn2 θ 都实测 1，但接口仍以导出 θ 和独立阈值为准。

`FixedLiftingForward.conv_forward/finish` 和普通 helper 中原 Conv2/BN/residual forward 是为软件模块调用保留的 **FP 影子**；实际 PED 两个消费者读取 fixed helper 的值。硬件账不能同时收费完整影子 Conv2 和 U16/F，也不能用影子提前供应真实 fixed 结果。

冻结 preview 参数为 `algorithm/patch_probe/factor_completion_20260909/latent_stage_train16/flow_recovery64/preview_only/shared48_u8_vq5.npz`。`u[864,48]`/`v[48,96]` 中 32 行有效，保存 a、temporal_bias、bn_scale/bias、θ。当前网络由 `flow_backward_probe.py:TrainableLatentPair` 执行 FP32 两因子及 sn2。`CHAIN/preview_gate_cmvm/whole_integer_dag.json` 已完整编译相同 V32→96，但它声明 signed24 组件接口；**当前 frozen FP32 preview 并未自动变成该整数接口**。现有 A 前移/V 编译可以复用结构和账本，不能未经等价或新数值验证直接承接本轮 AEE。

## 3. 可复用的有限服务部件及其边界

| 既有代码 | 可直接借的部件 | 必须重接或不能继承的部分 |
|---|---|---|
| `algorithm/patch_probe/joint_completion_20260909/hardware/finite_frame_service.py` | `run_jobs` 有限上下文、事件依赖、单资源互斥、严格 PACK 出口顺序；`resource_contract`、`layer_schedule` 给出来源/halo/共享 DMA 的收费实例 | 原 32 KiB 状态、128 KiB 权池、P4/H32、FP32 32 add/8 FMA；原完整 W1/W2、row34/common3 学生。没有源时间 DAG、当前 R32/R16 或 PED 双消费者，不能直接沿用步数或资源分配 |
| 同目录 `finite_frame_core.cpp`、`full_frame_requests.py` | 真源时间字、padding、P4/H8 与共享 W 向量需求计数的索引逻辑 | 当前计数是 P2、实际 U8/U16、不同静态零与 stride2；须重映射，不能将汇总数字填成局部事件 |
| `algorithm/patch_probe/partial_completion/finite_service.py:Engine` | 一个资源的 issue 占用与结果 latency 分开，64-bit state/W 端口争用；有明确依赖 ready 时间 | 旧 INT24/PSN48 检查 slice，只覆盖旧 PSN/Conv1；其 2 KiB、预测表和私有尾控制不是当前 raw 源图 |
| `psn/gustavsnn_resident_core.cpp` / `gustavsnn_resident.py` | NR4 双缓冲、共享源 bank 读口、有限广播收件人、W 仲裁和背压；F_cache 与 F_live 分离 | 旧 S2 time/class、S15 和 64PE。不能把非因果完整 T10 源图当 LIF 逐 tick 就绪，也不能移用旧百分比 |
| `CHAIN/preview_gate_cmvm/service_roofline.py` | DAG 位宽、bit-work、读写及 issue 的分类方法 | 明确只是 roofline 下界，没有有限调度/背压；不能称“已闭周期” |

新 `schedule_compare_same_port` 可先直接排这两个完整 **静态 source 图**：同总算术资源、同 RF 容量/位宽/端口/延迟、同输入到达与出口 backpressure，RNE/比较/指令 issue 明确占用资源。输入域/图功能已独立验证，这一步不依赖新增真实捕获。48-bit 固定 ALU 或按真实 bit 宽串行都可声明，但不能一轴固定宽、一轴按 bit-work 免费缩短；也不能只给 ordinary 原节点顺序而不给相同合法排程/转发权限。输出阻塞时 terminal 及共享父节点必须保活至真实最后消费者，不沿用“出口永远可接受”的临时峰值。

## 4. 现成数据与最小缺口

同一 diverse10 的真实源账位于 `LIFT/fixed_lifting_diverse10/fast_raw_diagonal_sources.json` 与 `CHAIN/temporal_structured_recovery/fixed_sources_diverse10/identity_permuted_base_sources.json`。每帧三处源各有 `active_columns[864]`、`nrv_rows_by_k[864]`、源幅值/shape、padding/stride；`shared_temporal_source_counts.py` 给出实际模块身份和 `count_projection_sources.py:count_source` 的 T10/P2 OR 定义。`LIFT/count_fixed_sources.py` 已按各轴真实零系数计数。这些是位置汇总，无法还原 source bank 冲突、NR4 尾部或消费者到达顺序。

当前两组 fixed 目录没有逐位置支持/源值 trace。若 Stage B 扩为**完整 r1→PED 的真实有限服务**，最小补口是同帧三个 `gate[T10,96,240,320]` 位图（sn1、sn2、proj.sn），各自 θ 和 frame identity；一帧裸 packed 共 27,648,000 B，直接从真实 producer hook 提取，不保存三套 float gate。要同时做 source 数值/边界重放，再保存共同上游 `I24` 的一个有 halo 的固定连续空间块及 source 期待门；静态 source 调度本身无需整幅 221,184,000 B packed I。完整幅面 backpressure 重放则需要完整位图或同等有序在线 trace，不能用 64 个分散位置或 per-k 总数伪造。

旧 `temporal_consumer_capture.py` 的 64 P2 Q/I/Z/AZ 捕获来自另一轮 FP32 学生；preview_gate 的 `az_q` trace 是另一个 V 接口。两者均不能替代本轮 fixed lifting/ordinary 三源。当前缺口是 Stage B 尚需连接的输入/模型边界，不是对 lifting 家族的否决。
