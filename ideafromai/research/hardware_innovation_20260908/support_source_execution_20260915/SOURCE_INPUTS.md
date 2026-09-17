# 原始 source 与固定定点 producer 输入

**已从既有 X 缓存重建 forced student 的投影前门，导出可直接供 source RTL 使用的真实定点输入。** 未训练、未改参数、未接远端、未测新 AEE。Q12/Q16/Q28 不是原 FP32 source 的无损替换；下面同时保留两份 gold。

## 1. 真实来源与 Student.source

精读 `../algorithm/train_exact_support_probe.py` 的 `Student.source`、缓存捕获钩子及验证安装路径。原函数是：

`h = torch.addmm(bias, A, X.reshape(10,-1)) - center`

`raw_g = (h >= theta)`，然后每个 C16 组在 16 个二值模式中选最小 Hamming 距离，平局选最小 code index，输出 `projected_g`。

真正使用的缓存在 `../algorithm/support_training.tar.gz` 内的 `support_training/teacher_cache.pt`，成员 377,516,110 字节。脚本只读该成员，不解包到原目录。32 个记录各有 `x[10,512,96]` FP32，以及 teacher 的 gate/yi；**只把 x 用作学生输入**，使用 `../algorithm/support_training/forced_code_parameters.pt` 的 A/bias/center/theta/D 重新计算 forced student。学生修改的是第一个 MLP source/FC1，位于该 X 捕获边界之后，因此原 teacher 前缀所产生的 X 也适用于 forced student 的同一前缀。

缓存 `gate` 不能当作 forced student 原门：本次实际重算的 forced raw_g 与它有 **374,953 个门位不同**。也未尝试从旧 projected g′ 倒推原 g。

全缓存是 run.json 中的 **32 个训练帧×每帧 512 固定抽样位置**，不是留出验证；另核本地官方 825 个验证文件名，与这 32 个文件交集为零。缓存没有保存位置数组；原捕获代码为 `torch.linspace(0, P-1,512).round()`。结合既有完整 source 形状 P=19,200，可重建位置索引；NPZ 将该索引标为 reconstructed，实际计算以缓存 sample index 为准，不把抽样位置冒充连续空间 P32。

## 2. 已冻结的定点接口

| 量 | 接口 | 本次范围/规则 |
|---|---|---|
| A | signed16，Q12 | FP32 参数在 float64 中乘 2^12 后 ties-to-even，整数范围 [-2199,1843] |
| X | signed24，Q16 | 同样 RNE；全缓存 FP32 [-41.05865478515625,42.931297302246094]，整数 [-2690820,2813546]；无 clip |
| threshold[t] | signed48，Q28 | 用存储的 FP32 常数在 float64 中算 `theta+center-bias`，一次 RNE；范围 [431371328,504250256] |
| dot / prefix | signed48，Q28 | 按源时间 s=0..9 累加 `A_q12[t,s]*X_q16[s,p,c]`，无中间 RNE；实际 prefix abs 最大 8,724,322,822 |
| raw gate | 1bit | `dot >= threshold[t]`，含等号 |
| projection | D[6,16,16] 二值 | 完整 Hamming 最近码，平局最小编号；不得把 gold code ID 喂给 DUT |

固定 Aq 对**任意 signed24 X** 的任意累加前缀都有保守界 `2^23 * max_t Σ_s abs(Aq[t,s]) = 74,524,393,472 < 2^47`，signed48 足够；这不依赖选出的 P32 活动率。实际最终 dot 范围 [-8571134764,7790710238]。A_fp32、bias_fp32[10,1]、center_fp32[10,1]、theta_fp32 标量全部原样保存；当前 center 全零、theta=1。

这个输入边界已经包含 X 的定点编码；RTL 从 X_q16 做真实时间 MAC 和判门。它不表示上游浮点 X→Q16 转换已在 RTL，或完整上游算子已量化闭合。

## 3. source_cases.npz 字段

固定取缓存首两帧 `thun_00_a_0012.npy`、`zurich_city_01_a_0025.npy`，各取 **sample index 0..31**，没有看活动率或差异选点。

| 字段 | shape / dtype | 用途 |
|---|---|---|
| X_q16 | [2,10,32,96] int32 容器、有效 signed24 | DUT 动态输入 |
| A_q12 | [10,10] int16 | DUT 静态参数 |
| threshold_q28 | [10] int64 容器、有效 signed48 | DUT 静态参数 |
| D | [6,16,16] uint8 | DUT 静态字典 |
| X_fp32 | [2,10,32,96] float32 | 原始 X，对照与追溯 |
| raw_g_fp32 / raw_g_int | [2,10,32,96] uint8 | 两种函数的原门 oracle |
| projected_g_fp32 / projected_g_int | [2,10,32,96] uint8 | 两种函数的最终投影 oracle |
| code_index_fp32 / code_index_int | [2,32,10,6] uint8 | TB-only，注意 P/T 顺序与门数组不同 |
| fp32_margin | [2,10,32,96] float32 | `h-theta`，TB-only |
| int_accum_q28 / int_margin_q28 | [2,10,32,96] int64 | 完整 MAC/判门 oracle；后者为 `dot-threshold` |
| A_fp32 / bias_fp32 / center_fp32 / theta_fp32 | 原参数 shape | 追溯；不是隐式额外 DUT 输入 |
| frame_file / sample_index / reconstructed_spatial_index | 来源元数据 | 训练帧与抽样位置边界 |

硬件如按 `p*10+t` 顺序工作，需把单 case 的 `[T,P,C]` 转成 `[P,T,C]`，不能直接错误 flatten。NPZ 中另列 `dut_input_fields` 与 `oracle_fields`。文件约 996KiB，可由脚本重建，不需要将二进制提交 Git。

首两 P32 共 61,440 个门：case0 无差；case1 原门差 1、投影门差 1、code 差 1。原门差异坐标 `[case,t,p,c]=[1,1,29,30]`，FP32 为 0、整数为 1；FP32 margin 为 **-0.0007497668266296387**，整数 margin 为 **+21578/2^28 ≈ +0.0000803843**。两份答案都保留，不以选中小集掩盖新数值函数。

## 4. 全 32×512 的实际比较

历史 GPU 脚本（现原样存为 `prepare_sources_gpu_legacy.py`）直接调用原 `Student.source(..., differentiable=False)`。它实际用了 **Python 3.10**（`/opt/anaconda3/envs/pytorch310/bin/python`），违反本任务要求使用 Python 3.12；不能把这次运行改记成 3.12。该次本机为 RTX3090、Torch 2.7.1+cu128，FP32 参考开启 TF32，与原 run 的配置一致；原模型运行是 A800/Torch 2.2.2+cu121，不能声称运行环境完全相同。历史交叉检查比较本机 CPU FP32、GPU TF32 关闭，**15,728,640 个 raw gate 及全部 projected gate 均与本机 TF32 开启零差**；这些 GPU 记录保留，但后续再生已换成下述 Python 3.12 NumPy-only 入口。

| 比较 | 实测数量 | 分母 / 比率 |
|---|---:|---|
| 定点 raw gate ≠ FP32 raw gate | 1,093 | 15,728,640 门，0.00694911% |
| 定点 projected gate ≠ FP32 projected gate | 887 | 同分母，0.00563939% |
| 定点最近码 index ≠ FP32 最近码 index | 357 | 983,040 个 C16 组，0.03631592% |

发生原门差异的位置，FP32 abs margin 范围为 **[3.5762786865234375e-7,0.002469301223754883]**。全缓存没有恰好零 margin；abs margin≤1e-6/1e-5/1e-4/1e-3/1e-2 的门分别为 2/26/344/3,720/36,897。余量小不能当作自动精度豁免。

| 机会统计，同为 983,040 组 | FP32 raw | 定点 raw | FP32 projected | 定点 projected |
|---|---:|---:|---:|---:|
| active bits | 2,973,348 | 2,973,641 | 1,667,573 | 1,667,898 |
| exact dictionary groups | 241,025 | 241,028 | 983,040 | 983,040 |
| exact 且 popcount≥2 | 95,884 | 95,923 | 479,188 | 479,276 |
| exact 支持合并可省加法项，未计执行费用 | 210,219 | 210,328 | 1,188,385 | 1,188,622 |

从 raw 到 projected 本身就有损，FP32 投影改动 1,753,239 个门位；本次定点化又改变了投影结果。**原 forced student 的 AEE 不能直接赋给新定点 producer**。本轮保留候选与两侧 gold，不新增 AEE、不训练、不扫位宽。

## 5. 再生与留出验证补采入口

当前受支持入口已实际用 **Python 3.12.7 / NumPy 1.26.4** 重跑：

```bash
/opt/anaconda3/bin/python3.12 prepare_sources.py
```

`prepare_sources.py` 调用 `prepare_sources_numpy312.py`，复用 `bn_state/support_service_model.py::read_torch` 的受限归档读取函数；该函数本来就支持 BytesIO，因此直接读取 tar 中的 teacher_cache.pt，不需要修改共享 reader、导入 Torch、安装新环境或使用 GPU。

默认直接生成主输入 **`source_cases.npz`**（与 `export_source_tb.py` / `run.sh` 一致）及 `source_statistics_numpy312.json`；仅依赖原 archive、student 参数/D/run.json 和共享 reader。旧 NPZ、旧统计和 GPU 历史脚本都不是再生依赖。若要核对历史，可显式传 `--compare-to <old.npz>`、`--compare-statistics <old.json>`；默认不执行这些比较。

已实际在空输出目录执行 `python3.12 prepare_sources.py --output regeneration_check/source_cases.npz`，没有传历史比较参数，收据为 `regeneration_check.log`。该新文件随后与 root 活动 NPZ 的 **26 个字段逐值全同**，包括原 X、A/Q12/阈值、全部整数累加/门/code、FP32 门/code 和 margin；当前活动主 NPZ 未被这次隔离检验覆盖。此前另存的 `source_cases_numpy312.npz`、`source_statistics_numpy312.json` 与 `prepare_sources_numpy312.log` 保留有历史比较收据。全 32 帧的 raw/projection/code 差异计数、teacher 对比和四组机会统计已逐帧重现原值；历史完整 raw gate 数组未保存，因此不把统计一致谎称为完整历史数组逐元素比较。全缓存整数 MAC 另用显式 s 循环与 NumPy einsum 核对，prefix 界重现为 8,724,322,822；写回重读通过，运行中 `torch` 未导入。

原 `source_statistics.json` 与 `prepare_sources.log` 仍是 Python 3.10 GPU 历史收据，没有覆盖或改写其结果。新入口明确要求 Python 3.12，后续不再执行旧 3.10 路径。

已写 **`capture_validation_source.py`** 作为下一次仅推理的补采入口：复用 `run_bn_probe.build_model`、固定 BN 与整数消费者安装路径，载入同一 forced student；在第一处 `sn1.spiking_neuron` 的 pre-hook 捕获完整 X[10,19200,96]，计算学生原门与投影门，然后立即中止该帧，既不训练也不跑整网 AEE。固定使用 run.json 首两个验证帧，并与既有 `forced_code_<frame>_source.npz` 逐门对照投影结果，环境差异会显式报告。

**此补采脚本本轮未执行**（只做了语法检查）。本地能读训练缓存及代码 overlay；第一验证帧的原事件输入在 `SDformer/data/Datasets/DSEC/saved_flow_data/event_tensors/10bins/left/zurich_city_09_a/zurich_city_09_a_0001.npy`，第二帧对应本地路径缺失，所需 epoch34 checkpoint/config 的同名本地路径也缺失。既有远端数据路径由 `../algorithm/valid825_cal32/run.json` 提供：

- code root：`/root/private_data/work/sdformer_codex/SDformer`
- checkpoint：上述根下 `neuron_experiments/H9_bipolar_self_attention/results/dsec_c12_alpha0125_ep29_resume5_20260830/checkpoint_epoch34.pth`
- config：上述根下 `neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_c12_alpha0125_ep29_resume5_20260830.yml`
- data：以已有执行入口的 `--data` 指向包含 `event_tensors/10bins/left` 的 DSEC saved_flow_data 根；本地根如上，不假定远端挂载完全同名。
- calibration：本地已有 `../algorithm/valid825_cal32/train_calibration.pt`；student-root 为 `../algorithm/support_training`。补采脚本这六个路径都要求显式 CLI 参数，避免悄悄借错模型。

本次没有连接或恢复旧远端会话，也没有因缺完整验证输入而冒用 teacher gate 或反演 g′。训练缓存已足以供应当前有界 RTL，留出补采是下一数值/泛化边界。
