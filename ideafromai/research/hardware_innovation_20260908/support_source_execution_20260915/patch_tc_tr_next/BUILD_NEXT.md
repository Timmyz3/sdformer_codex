# Patch R32 / TC→TR：下一次实际构建的最小合同

**下一项应补完整两因子与真实 PSN；目前没有现成的 patch TC/TR 全链 RTL 可直接运行。** [FUSION_REMAINING ②](../FUSION_REMAINING.md) 的普通 R32、TC/TR NumPy 地址参考、整数 shared48、下游 CPU 有限机器各自存在，不能合称一条已闭 RTL。以下只读检查文件与 NPZ 字段；未训练、导出新学生或运行实验。

设 `F=algorithm/patch_probe/factor_completion_20260909`。两份明确现货及关键缺口如下。

| 固定臂 | 真实现货 | 下一步必须补的数值合同 |
|---|---|---|
| 普通 R32 无 mask 强控制 | [preview_only/shared48_u8_vq5.npz](../../algorithm/patch_probe/factor_completion_20260909/latent_stage_train16/flow_recovery64/preview_only/shared48_u8_vq5.npz)：U8 `[864,48]`、逐 rank dyadic scale、V sign/shift/nonzero `[48,96]`；`shared_rank=effective_compiled_rank=32`，尾 V 为零，可普通删除。旧 QDQ valid825 AEE **1.203043**。 | **未发现该 preview-only 的 Aq14 整数部署包/整数825**。可单独调用 [compile_model](../../algorithm/patch_probe/factor_completion_20260909/latent_stage_train16/compile_integer_factors.py) 编译这一个已保存文件，再删除断开尾 U/V；保留它算出的 per-H 对齐尺度与 threshold。U/V 不需重新训练，但 A 的 RNE 和移除 FP 舍入产生新整数函数，不能沿用 QDQ AEE。 |
| 一种固定、非退化 TC/TR 学生 | [demand_train16/shared.npz](../../algorithm/patch_probe/factor_completion_20260909/demand_train16/shared.npz)：U `[864,96]`、V `[96,96]` **FP32**，8 份互不相同的24bit mask，每区12个活 J4，活 R48，θ源/输出均1。区域 `min((group_id//80)*8//240,7)`，横向与T10共用。 | 此包**没有 U8、VQ5 或整数阈值**，现有整数编译器不能无条件喂入。必须先一次冻结量化/尺度对齐、mask 后任意归约界和完整阈值，再产该新函数 gold；不能用 `integer_factors/shared48_u8_vq5.npz` 冒充它。后者确有 Zi16/Yi32/Ui48 与整数 diverse10 **1.091660**，但 mask 全1，是不同学生。 |

**输入和 gold 生成入口已经齐，旧输出不能直接移用。** [partial_completion/integer_valid10/capture_00.npz](../../algorithm/patch_probe/partial_completion/integer_valid10/capture_00.npz) 至03确实有 `source_gate_words[64,864,4]` 及 `group_ids`；低10bit为真实原生 T10，`k=((c*3)+kh)*3+kw`，P4是**水平四位置**，不是空间R16的2×2。按 [compile_integer_factors.functional](../../algorithm/patch_probe/factor_completion_20260909/latent_stage_train16/compile_integer_factors.py) 的展开 `(words[...,None]>>arange(10))&1 → [G,T,P,K]` 取源，再独立 int64 算 `Z=(g·U8)⊙mask; Y=Σ Z·V_aligned; Ui=Σ Aq14·Y; gate=Ui≥threshold`，不能读取旧 Yi/gate 作新答案。其 [check_integer_adapter.py](../../algorithm/patch_probe/factor_completion_20260909/latent_stage_train16/check_integer_adapter.py) 已有独立 c/kh/kw gather、真实边界/内部源、整数阈值有理数边界核验；三个旧整数模型 **2073600 门**已核过，不必重写数学参考。需要新采完整源时，现成入口是 [evaluate_network.py 的 `--capture-sampled-source`](../../algorithm/patch_probe/joint_completion_20260909/evaluate_network.py)，本次不启动采集。

**最小 P4×T10、K864、H96 RTL**：先到完整 sn2 门为终点；普通 R32 与同一 TC/TR 学生的 dense-masked 执行是必须的同函数控制，R32跨学生比较另列。

1. 真实门字经受控口供数；一次 K 的四个 T10 字是40bit，可定为一个64bit物理字并实付864次上限读取（6912B配置布局），或用原生 `3×6×C96` halo 的2160B载荷＋真实 gather。双方同一种选择，不能把 unfolded 重复字和原生窗口零成本互换。
2. `TC` 在 U 生产前从实际24bit mask 解码；按 J4 产生 `P4×T10×J4` Zi，再写紧凑 Z 和 TC ID。**同一 ID** 经 `TR` 选 V 原行、按H8恢复全部 Y32；不是 TB 给活动列或 Z。完整活 R48 的 Zi16 为3840B；流式两个 J4包是640B但必须记 U/V 重读、单口冲突与 lifetime，不能既称全驻又称小缓冲。所有R槽/活槽、metadata和W/cache容量先给普通控制同权限。
3. 按真实 Yi32/T10 地址读，做 signed16 Aq14×signed32 Yi 的48bit乘积及有界 Ui48归约，最后门比较；Y为15360B，若全存Ui为23040B，逐H8上下文可小存但要收费。BN1已折阈值，不能再加一次BN；完整执行不加载统计完成器的大 pos/neg 表。TC/TR 的一次排布适配留到两完整臂通过后。

**已跑 RTL 可复用什么，不能重造或误接什么。** [spatial_r16_rtl/spatial_core.sv](../../representation_transfer_20260914/spatial_r16_rtl/spatial_core.sv) 的 U8 AAC、Z/psum银行、稀疏源与排空握手是真实已跑底座，但其Q1为竖3tap、Q2为横3tap；须改成K864→latent→H96，不能只换系数。 [wide_phase_alu.sv](../../representation_transfer_20260914/spatial_r16_rtl/wide_phase_alu.sv) / [i24_consumer.sv](../../representation_transfer_20260914/spatial_r16_rtl/i24_consumer.sv) 可借宽加/乘及BP，但 **I24的BN+identity函数不是此sn2 PSN**。近期 [support_fc1.sv](../../support_lut_execution_20260915/support_fc1.sv) 的T10 PSN PREAD/PMAC/PDRAIN/POUT及消费退休已实跑；其输入是Y24、96路，应改成声明的Y32/算术预算，不能复制96MAC再称8路。更早 [temporal_source.sv](../../open_fusion_execution/stage_20260912/hardware/rtl_source/temporal_source.sv) 已跑真实I24→sn1与RNE/sat/BP，**只覆盖源PSN，没覆盖preview U/V**。[factor_reference.py](../../algorithm/patch_probe/factor_completion_20260909/factor_reference.py) 的 `decode_tc/store_v_column_banks/read_tr/sparse_forward` 已逐地址核过TC/TR NumPy功能，尚无物理端口时序；复用公式，新增真正grant/请求/返回/紧凑state即可。

**后继缺口必须保留。** [stage hardware/integrated.py](../../open_fusion_execution/stage_20260912/hardware/integrated.py) 已实跑同一CPU有限机器的 source→普通preview→sn2→Conv2/BN2/shortcut→门/PED双消费者，现成权重/整数边界与实际handoff不必重新发明；报告明确它不是全链RTL。新TC/TR改变sn2，原捕获的sn2/Conv2输出只能检查旧臂，不能喂新臂。若要最终Conv2的水平P4，至少需其 **3×6=18个 sn2 位置**及前级halo，原P4门不足；还需真实identity供数与当前Conv2整数部署/BN2加残差合同。可从 [projection_chain/capture_train4](../../algorithm/patch_probe/residual_consumer_probe/projection_chain/capture_train4/summary.json) 的参数、位置及原生权重复用来源，但其中 `r1_source_gate_words` 是**Conv2的旧sn2源**，不得当新Conv1源或答案。先闭P4全因子→PSN，再扩大halo接真实后继，分别报告终点。

已完成的是训练/质量点、整数部分学生、CPU TC/TR地址和部分RTL；缺的是这一固定 mask 学生的整数准入，以及同资源 U→紧凑Z→TR→V→PSN/后继时序。CFMP作者可见功能链可作为 A；当前本地秩、J4、八区域训练和存储安排不是作者全部迁移。原private56请求0.03999%不是这条尚未闭链的失败证据，也不能复活已删尾V当普通R32分母。
