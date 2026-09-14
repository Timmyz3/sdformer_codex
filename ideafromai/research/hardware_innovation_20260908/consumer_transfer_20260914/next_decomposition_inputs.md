# 下一条连续分解：现货与最小迁接边界（只读核验）

**优先现货是CFMP目录的普通shared48整数链；若要求原位接当前r0→I24消费者，则没有“窄整数质量已证、可直接插接”的现成路线。** [旧审计第13/28项](../transfer_adapt_20260914/audit/REVIEW.md)保留了这两个方向，但不能把空间浮点AEE、CFMP整数AEE和当前R8的A800/valid825结果拼成同一部署证据。本轮只读NPZ真实键/shape和执行代码，未训练、导出新权重或写RTL。

|路线与实际文件|现有质量证据|是否已有可用窄整数链|
|---|---|---|
|空间R16：[目录D](../open_fusion_execution/major_operator_fusions_20260913/decomposition_owned/README.md)的`results/spatial_r16_w32.npz`及`spatial_r16_w8.npz`均存在；first[16,96,3,1]、second[96,16,1,3]|diverse10 1.2691297866仅对应w32|w8有FP32解量化值及逐输出行scale；本次只读恢复`q=RNE(factor/scale)`为−127…127，回乘再转FP32逐值等于文件。**连续Z定点/scale对齐与该整数函数AEE不存在**|
|Tucker R8：D的`results/tucker_r8_w32.npz`及`…_w8.npz`均存在；first[8,96,1,1]、core[8,8,3,3]、second[96,8,1,1]|diverse10 1.4170152223仅对应w32|同样只是W8-QDQ；还多一个连续3×3 core。不能把“R8”当现有两因子R8核兼容|
|CFMP普通shared48：[整数目录F](../algorithm/patch_probe/factor_completion_20260909/latent_stage_train16/integer_factors/compile_result.json)的`shared48_u8_vq5.npz`确实存在|[完整整数diverse10](../algorithm/patch_probe/factor_completion_20260909/network_integer_recovery64_diverse10/02_integer_factors_shared48_u8_vq5_summary.json)：10帧/516735有效像素，frame-AEE **1.0916604779**，pixel-AEE 1.1162837370|已有U8、V code5/zero、dyadic尺度、Aq14、完整整数阈值与范围证明；建议先跑**full**普通控制，不先接统计预测/private尾|

CFMP质量记录的[run.json](../algorithm/patch_probe/factor_completion_20260909/network_integer_recovery64_diverse10/run.json)保留旧整数S2/coarse-head父、真实Conv2/BN2/shortcut，temporal TF32=false、其余conv TF32=true；目录说明为A800链路，但该run未记完整torch/CUDA版本，**不是当前R8父/825同协议复测**。空间两项的[旧AEE汇总](../open_fusion_execution/major_operator_fusions_20260913/root_owned/aee_all.md)亦明确候选粗头/FP64 EPE与NB0最终头/FP32归约不同。二者是有效历史算法点，不应改称matched当前部署。

**CFMP可直接取数的合同。** F内NPZ已有`u_int8[864,48]`、`integer_v_code5[48,96]`和`integer_v_nonzero`、`integer_v_align_shift[48,96]`（0…14）、`integer_a_q14[10,10]`（int16，−16908…8491）、`integer_full_threshold[10,96]`（int64载体）、`integer_y_exponent[96]`（−24…−15）。U逐latent指数−10…−8，θ_source=θ_output=1；`shared_rank=32`仅是原32+16分段，full需全部48列。执行入口为[integer_adapter.py](../algorithm/patch_probe/factor_completion_20260909/latent_stage_train16/integer_adapter.py)与[evaluate_factors_network.py](../algorithm/patch_probe/factor_completion_20260909/evaluate_factors_network.py)，编译及静态界在[compile_integer_factors.py](../algorithm/patch_probe/factor_completion_20260909/latent_stage_train16/compile_integer_factors.py)。

数值为`Zi=Σg·Uq8; Yi=ΣZi·(sign<<aligned_shift); Ui=Σ_s Aq14[t,s]·Yi[s]; gate=(Ui>=threshold[t,h])`。**RNE只在冻结系数Aq=RNE(A·2^14)等离线量化；Zi→Yi→Ui中间无RNE或截断。** 正BN已按保存常数的精确有理数ceil折入完整阈值，不能再套一次BN或identity。静态任意归约界为Zi15（建议存16）、Yi32、Ui47（建议存48），完整阈值可保守存40bit（当前full实值39bit足够）；这是新整数学生，不能沿用旧QDQ或当前I24的gold。

**已有真实源和检查。** `algorithm/patch_probe/partial_completion/integer_valid10/capture_00…03.npz`真实存在，`source_gate_words[64,864,4]`的低10bit可直接展开原生P4/T10；其旧Yi/gate字段属于旧函数，不能当shared48 gold。另有[完整首帧gates.npz](../algorithm/patch_probe/joint_completion_20260909/full_capture4/capture/00_optimized_prefix_train16_row34_packed_word_exact/000_zurich_city_09_a_0001/gates.npz)，含little-bitpack的T10/C96/H240/W320源。现成[adapter_checks.json](../algorithm/patch_probe/factor_completion_20260909/latent_stage_train16/integer_factors/adapter_checks.json)已核三个真实10×12区域、K864=C96×3×3的Zi/Yi/Ui/门，可作新gold重建入口。

**最小迁接工作。** 可复用[R8 consumer_rtl](../r8_consumer_fusion_20260914/consumer_rtl/REPORT.md)的源握手、N8输出编排、psum银行和八lane宽乘加，但Q1必须3bit→8bit，Zi13→16bit，48列分六个R8条带；每条P4/T10/R8的Zi16为640B，超过旧520B。逐条带消费后释放Z、将全部Yi累加到已有15360B psum是可审查的最小布局；U一个R8条带864×8B=6912B也超过旧Q1载荷，必须真实配置/装入。V是带符号幂次连续消费，不能继续按脉冲加法；A阶段需T10的Yi32→Ui48，可借现有32×32→64后端，但需新时间归约、阈值口与门输出模式。**r1.conv1→PSN与当前r0.conv2→BN+FP32 identity→I24是不同图边，禁止将CFMP门塞进原I24端口。** 尚缺这条全链的端口/状态排程、shared48新函数逐tile整数gold导出、真实后继连线与当前环境复评；完整CFMP TC/TR压缩bank仍未实现。

若根优先维持当前r0→I24挂点，则选空间R16作下一准备轴更贴近：D/adapter.py已有实际3×1→1×3前向，当前r0权重/64位置捕获在D相邻`root_owned/sttmultires_unet_encoders_swin3d_patch_embed_residual_encoding_resblocks_0_conv2_0.npz`；可复用R8目录同r0完整source/FP32 identity。首先还须冻结逐rank尺度传递、水平连续Z/halo和宽度，重新生成output_scale/a/b并独立评价整数AEE。**不能把w32的1.269130当此尚未定义整数链的质量许可。**
