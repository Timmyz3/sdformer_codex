# 32 帧固定输入扩大：来源与接口

已用 **Python 3.12.7 / NumPy 1.26.4、无 Torch/GPU** 实际导出 1,024 个真实 T10 源包和原有 3 个诊断。只扩大原始输入范围；原 A/X/τ 定点合同、D、变量顺序、exact-code 图、各自 class 图和参数容量均保持。没有训练、重新选择 pair、FP32 门差统计或新 AEE。[生成记录](prepare.log)、[机读来源与检查](manifest.json)。

来源是 `algorithm/support_training.tar.gz` 中的 `support_training/teacher_cache.pt`，读取每条记录的原始 `x[10,512,96]`；不读取 teacher gate 充当学生门。32 个文件及顺序逐一核对 `algorithm/support_training/run.json` 的 `train_files`。源参数来自同目录 `forced_code_parameters.pt`，重新从真实 A/bias/center/theta 导出整数 A/τ，D 与 `dictionary.npy` 逐值一致。

每帧固定取缓存 sample **0..31**，无活动率筛选，文件与 sample 身份保留在 NPZ/manifest。缓存 sample 不是连续空间位置；空间坐标只按原固定采样规则重建，明确不是缓存中保存的坐标。

- frame0：原 pair 选择使用的 32 包。
- frame1..31：没有用于这次 pair 选择的训练探针；frame1 曾用于小表报告，其他帧只作本次输入扩大。均不能称验证集，也不是对支持字典/学生训练的独立留出数据。
- `train0_p0` 至 `train31_p31` 对应 1,024 个真实包。之后依次保留 `diagnostic_zero`、`diagnostic_positive_extreme`、`diagnostic_signed_extreme`。

## 固定数值合同

`Aq = RNE(A × 2^12)`，signed16；`Xq = RNE(X × 2^16)`，signed24，bin 以 int32 容器保存；`τ = RNE((theta + center − bias) × 2^28)`，signed48。实际源门为 `Σ_s Aq[t,s] Xq[s,p,c] >= τ[t]`，全部 T10 项累加至 signed48，无中间舍入。nearest-code 使用 Hamming 距离，平局取原 D 最小 code index。没有 clip；超出界会停止导出。

本选定 1,024 包的 Xq 范围为 **[−2,187,051, 2,204,738]**，逐 T10 累加观测前缀绝对最大值 **7,271,151,639**。固定真实 A 对任意 signed24 X 的保守绝对界 **74,524,393,472 < 2^47**，包括原极值诊断，保持原 signed48 合同。这不是新定点精度评估；先前全缓存 FP32/整数差异统计未重复计算。

| NPZ 字段 | Shape | 含义 |
|---|---|---|
| `X_q16` | `[32,10,32,96]` | 原始输入的既定 Q16 定点值，DUT 输入 |
| `A_q12`, `threshold_q28`, `D` | `[10,10]`, `[10]`, `[6,16,16]` | DUT 静态参数 |
| `frame_file`, `sample_index`, `frame_role` | `[32]`, `[32]`, `[32]` | 原帧名、固定 sample 身份、选择/报告角色 |
| `raw_g_int`, `projected_g_int` | `[32,10,32,96]` | 独立 NumPy oracle，不在 DUT bin 中 |
| `code_index_int` | `[32,32,10,6]` | 原 nearest-code oracle，不是 class 标号，不在 DUT bin 中 |

NPZ 同时保留真实 A/bias/center/theta 和阈值来源。`source.bin` 只有 A/τ/D、实际需要读取的静态图及 rank、病例名字与 X。TB 必须从原始 X 做真实源 MAC、门和判码，不能读取 NPZ oracle 给 DUT。

## 两份运行输入及重叠检查

`source.bin` 使用父级原 exact-code 图与 `../adapt_tables.npz` 的新 class 图/roots/canonical。`old_class/source.bin` 使用原 nearest-response W′ 的 class 图，两者的 X、源参数、code 图、变量顺序均相同。自然/熵序两份静态描述都按原格式保留；父级扩大运行仅选既定熵序，未重新排序。

生成时完成以下实际断言，不依靠哈希：

1. 前 64 包共 **61,440 个 int32 X word / 245,760 个容器字节**与原 `source_cases.npz`、原/适配 `source.bin` 逐值相同；名字、real 标记及 integer gate/projected-code oracle 相同。
2. 三个诊断的名字、标记和 2,880 个 X word 与原 67-case bin 相同。
3. 新旧各自全部静态前缀（参数、图、roots、canonical、rank）与对应已跑 67-case bin **逐字节相同**；仅替换病例计数及追加真实输入。
4. 写出的两份 bin 重新解析，与期望 prefix 和 1,027 条病例逐值相同。NPZ 中 oracle 从既定整数合同独立重算，没有修改原主 NPZ/main CSV。

## 再生与执行边界

在本目录执行 `/opt/anaconda3/bin/python3.12 prepare_expanded.py`。依赖原 archive/学生参数、父级原 NPZ/图/bin 及已冻结的适配图/bin；这里只扩展既有固定实验，源图和 pair 编译不在本脚本中重做。输出 `source_cases.npz`、两份 `source.bin`、`manifest.json`；二进制是可再生数据，不入 Git。

父级 TB 新选择参数可在本目录运行 `../../obj_dir/Vsource_classifier 1027 --strong-only`，只做熵序 static64+PF / exact-code32+PF / 新 class32+PF，各 ready/BP；在 `old_class/` 运行 `../../../obj_dir/Vsource_classifier 1027 --class-only`，只补旧 class 两臂。预期分别 6,162 / 2,054 个任务，包含诊断；计真实输入性能时分母分别为 1,024 包，须单列 frame0 与 frame1..31。此处只完成导出及输入一致性检查，RTL 运行由父级负责。
