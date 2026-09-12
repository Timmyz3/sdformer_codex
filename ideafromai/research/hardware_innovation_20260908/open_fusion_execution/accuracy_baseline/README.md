# 原 SDformerFlow：同帧精度基线

2026-09-12。原 PSN/SDSA 的本地 upstream 复现 NB0 ep29；**不是 ordinary 学生、Motion ep34，也不是作者发布的检查点**。本目录仅提取旧实测记录，没有新推理、训练或硬件性能测量。用户新门是同人口 AEE 优于 NB0，旧 ordinary `+0.005` 不再否决候选。

| NB0 范围 | 帧数 / 有效像素 | AEE 帧均值 | AEE 像素均值 |
|---|---:|---:|---:|
| valid825 | 825 / 48,152,523 | 1.445352534681 | 1.377819674707 |
| diverse10 | 10 / 516,735 | 1.454602861070 | 1.504901675309 |
| 去校准源帧九帧 | 9 / 483,510 | 1.446425661411 | 1.503300862451 |

**原 fixed R32 的完整825也已逐帧配对。** 两学生均825/825文件与每帧有效像素相同，共18序列、48,152,523像素；这不是仅有汇总。

| 旧完整验证身份 | AEE帧均值 | Δ对NB0 | 相对AEE下降 |
|---|---:|---:|---:|
| ordinary / identity_permuted_base | 1.219801338299 | -0.225551196 | 15.605% |
| lifting_raw / fast_raw_diagonal | 1.232979367919 | -0.212373167 | 14.694% |

旧原学生抽出的十帧AEE与当前 `aee_rebase/original32` 各轴逐帧相同（最大差0）。**原 lifting40 已过新NB0完整825精度门**；此前仅相对 ordinary 的回退不能继续否决它。证据与原参数/定点语义路径见 [valid825_summary.json](valid825_summary.json) 和 [825行配对](valid825_matched_frames.csv)。新R24、BN改法、剪枝及恢复权重仍只有各自十帧，不继承此全量结果。

825 行重新聚合相对原 profile 仅差 -1.43e-09 AEE（CSV 小数序列化及旧聚合）。十帧全部命中原记录，逐帧有效像素一致。**下表 26 臂均低于同十帧与同九帧 NB0**；重复的原学生控制按实验来源保留，不冒充独立模型。

| 来源 / 学生 / 臂 | 十帧 AEE | 九帧 AEE | 十帧 Δ对NB0 |
|---|---:|---:|---:|
| PED_R24/ordinary/original32 | 1.161187 | 1.145265 | -0.293416 |
| PED_R24/ordinary/original_ordered24 | 1.163094 | 1.138897 | -0.291509 |
| PED_R24/ordinary/weight_svd24 | 1.182310 | 1.163213 | -0.272293 |
| PED_R24/ordinary/activation_whitened24 | 1.164880 | 1.140007 | -0.289723 |
| PED_R24/lifting_raw/original32 | 1.186586 | 1.154097 | -0.268017 |
| PED_R24/lifting_raw/original_ordered24 | 1.196322 | 1.167896 | -0.258281 |
| PED_R24/lifting_raw/weight_svd24 | 1.195021 | 1.171233 | -0.259581 |
| PED_R24/lifting_raw/activation_whitened24 | 1.190853 | 1.164876 | -0.263750 |
| BN/ordinary/original_cuda_bn | 1.161187 | 1.145265 | -0.293416 |
| BN/lifting_raw/original_cuda_bn | 1.186586 | 1.154097 | -0.268017 |
| BN/ordinary/centered_engine | 1.172123 | 1.150343 | -0.282480 |
| BN/lifting_raw/centered_engine | 1.188049 | 1.155723 | -0.266554 |
| BN/ordinary/onepass | 1.166842 | 1.151548 | -0.287761 |
| BN/lifting_raw/onepass | 1.187835 | 1.155485 | -0.266768 |
| pruning_no_train/ordinary/unpruned | 1.161187 | 1.145265 | -0.293416 |
| pruning_no_train/ordinary/global_group2 | 1.214953 | 1.200332 | -0.239650 |
| pruning_no_train/ordinary/phase_joint | 1.211914 | 1.184565 | -0.242689 |
| pruning_no_train/lifting_raw/unpruned | 1.186586 | 1.154097 | -0.268017 |
| pruning_no_train/lifting_raw/global_group2 | 1.254436 | 1.216661 | -0.200167 |
| pruning_no_train/lifting_raw/phase_joint | 1.229379 | 1.197293 | -0.225224 |
| pruning_row_phase/ordinary/global_joint_pair | 1.214953 | 1.200332 | -0.239650 |
| pruning_row_phase/ordinary/row_phase_joint_pair | 1.217386 | 1.195356 | -0.237216 |
| pruning_row_phase/lifting_raw/global_joint_pair | 1.254436 | 1.216661 | -0.200167 |
| pruning_row_phase/lifting_raw/row_phase_joint_pair | 1.247094 | 1.213936 | -0.207509 |
| pruning_recovery64/ordinary/global_group2 | 1.220647 | 1.199235 | -0.233956 |
| pruning_recovery64/ordinary/phase_joint | 1.201610 | 1.178377 | -0.252992 |

**身份与口径。** NB0 使用本地60轮 crop ep59 起点，再30轮 full-resolution；equal+10 的ep34/39未超过ep29。论文写80轮 crop，不能称严格作者训练复现。分辨率480×640、窗口T2×15×15、batch1、原78处BN无运行统计；AT-LIF/Shiftmax安装数均0。checkpoint/config/A800原CSV路径保存在 [source_metadata.json](source_metadata.json)，配置原文在 [source_nb0_config.yml](source_nb0_config.yml)。NB0走完整模型最终 `flow[-1]`；本轮候选走 `preds.2` 时间求和并双线性恢复480×640的粗头。可以比较任务质量，不能把全部精度差归因U/V、BN或剪枝，也不能将其称同一个网络。

**配对边界。** 两侧同一canonical GT及 `mask_tensors/<file>`，不交事件掩码，flow scaling=1，先每帧有效GT像素平均再帧等权；程序逐项检查帧名、每帧有效像素和已完成记录均值。旧NB0使用FP32误差图归约并保存10位小数，候选以FP64累加FP32 EPE；不是逐位算术复现。旧mask全部位图未归档，本次不伪称仅像素数就证明历史mask逐位相等。九帧仅排除 `zurich_city_09_a_0001.npy` 校准源帧，不将反复使用的小集合冒称未见验证。

**交付。** [825行原CSV](source_nb0_valid825.csv)、[同十帧NB0](matched_diverse10.csv)、[逐臂逐帧配对](candidate_matched_frames.csv)、[小集完整汇总](summary.json)。未训剪枝6臂、行相位4臂与恢复64步2臂均重新按NB0比较；恢复使用新训练身份，不能继承免训硬件活动。原各实验旧门文字不在本目录改写。新变体还须完成自己的valid825；本表不提供RTL/PPA、创新性或录用证据。公开论文1.602/1.61人口不同，不用于本门。

复算：`/usr/bin/python3.12 extract.py`（离线，标准库）；需更新源副本才加 `--fetch --socket /tmp/codex_bn_onepass_a800_20260912.sock`，仅SSH读取既有CSV/config/profile，不加载网络。
