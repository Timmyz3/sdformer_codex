# 所有神经元/注意力短测结果汇总

更新时间：2026-05-22。当前已停止训练进程，本文件只整理已有 short/rapid/profile 结果。

- 汇总 CSV：`neuron_experiments/H9_bipolar_self_attention/analysis/all_short_tests_aggregate.csv`
- 纳入 summary.csv 文件数：68
- 纳入实验行数：322
- 排除：`smoke_*`、`debug_*`、`probe_parallel_*` 和未完成/非正式探针。

## 通过 gate 的 Top 30

| rank | 实验 | stage | steps | samples | attention | FFN/范围 | LR策略 | ang | AEE | AAE | SOPs(G) | firing | worstPN | score |
|---:|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `h40_p2_SCS012_F_steps80` | screen | 80 | 5 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | differential_lr | 0 | 0.977 | 6.849 | 2.977 | 0.0698 | 5.53 | 1.058 |
| 2 | `h40_p4_TXS012_ang05_warm_steps160` | screen | 160 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s1_ffn+s2_half | warm | 0.5 | 0.980 | 7.046 | 2.974 | 0.0698 | 5.09 | 1.066 |
| 3 | `h40_p2_TXS012_F_steps80` | screen | 80 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s1_ffn+s2_half | differential_lr | 0 | 0.982 | 7.041 | 2.921 | 0.0685 | 5.95 | 1.068 |
| 4 | `h40_p3_SNS02_ang05_steps80` | screen | 80 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | differential_lr | 0.5 | 0.948 | 7.288 | 3.227 | 0.0757 | 5.68 | 1.083 |
| 5 | `h40_p2_SNS02_F_steps80` | screen | 80 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | differential_lr | 0 | 0.963 | 7.061 | 3.231 | 0.0758 | 5.88 | 1.093 |
| 6 | `h40_p4_SCS012_ang05_dlr_steps160` | screen | 160 | 5 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | dlr | 0.5 | 1.021 | 6.550 | 2.948 | 0.0691 | 5.07 | 1.094 |
| 7 | `h40_p4_TXS012_ang05_slowbb_steps160` | screen | 160 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s1_ffn+s2_half | slowbb | 0.5 | 1.018 | 6.856 | 2.924 | 0.0686 | 5.64 | 1.100 |
| 8 | `h40_p4_SNS012_ang05_dlr_steps160` | screen | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | dlr | 0.5 | 1.030 | 6.690 | 2.901 | 0.0680 | 5.10 | 1.107 |
| 9 | `h40_p4_HTS02_ang02_warm_slowbb_steps160` | screen | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | warm_slowbb | 0.2 | 0.993 | 7.276 | 3.186 | 0.0747 | 4.73 | 1.115 |
| 10 | `h40_p2_SNS012_F_steps80` | screen | 80 | 5 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | differential_lr | 0 | 1.037 | 6.893 | 2.984 | 0.0700 | 5.71 | 1.119 |
| 11 | `h40_p4_HTS02_ang02_slowbb_steps160` | screen | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | slowbb | 0.2 | 1.024 | 7.453 | 3.039 | 0.0713 | 4.64 | 1.120 |
| 12 | `h40_p4_HTS02_ang05_slowbb_steps160` | screen | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | slowbb | 0.5 | 0.987 | 7.683 | 3.036 | 0.0712 | 4.48 | 1.129 |
| 13 | `h40_p4_SCS012_ang05_warm_slowbb_steps160` | screen | 160 | 5 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | warm_slowbb | 0.5 | 1.050 | 6.768 | 2.926 | 0.0686 | 5.16 | 1.130 |
| 14 | `h40_p4_SCS012_ang05_warm_steps160` | screen | 160 | 5 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | warm | 0.5 | 1.043 | 7.308 | 2.866 | 0.0672 | 5.00 | 1.135 |
| 15 | `h40_p4_HTS012_ang05_slowbb_steps160` | screen | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s1_ffn+s2_half | slowbb | 0.5 | 1.050 | 7.155 | 2.947 | 0.0691 | 4.40 | 1.139 |
| 16 | `h40_p4_HTS012_ang05_dlr_steps160` | screen | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s1_ffn+s2_half | dlr | 0.5 | 1.055 | 7.345 | 2.643 | 0.0620 | 4.55 | 1.149 |
| 17 | `h40_p4_HTS02_ang02_dlr_steps160` | screen | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | dlr | 0.2 | 1.046 | 7.525 | 3.105 | 0.0728 | 4.69 | 1.150 |
| 18 | `h40_p4_HTS012_ang05_warm_slowbb_steps160` | screen | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s1_ffn+s2_half | warm_slowbb | 0.5 | 1.058 | 7.281 | 2.853 | 0.0669 | 4.07 | 1.150 |
| 19 | `h40_p4_SNS012_ang05_slowbb_steps160` | screen | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | slowbb | 0.5 | 1.064 | 7.404 | 2.901 | 0.0681 | 5.31 | 1.159 |
| 20 | `h40_p4_SNS012_ang05_warm_steps160` | screen | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | warm | 0.5 | 1.068 | 7.268 | 2.918 | 0.0685 | 5.28 | 1.160 |
| 21 | `h40_p4_SNS012_ang05_warm_slowbb_steps160` | screen | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | warm_slowbb | 0.5 | 1.079 | 6.898 | 2.987 | 0.0701 | 5.55 | 1.162 |
| 22 | `h40_p4_SNS02_ang05_slowbb_steps160` | screen | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | slowbb | 0.5 | 1.051 | 7.317 | 3.151 | 0.0739 | 5.42 | 1.164 |
| 23 | `h40_p2_SNS02_F_steps80` | screen | 80 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | differential_lr | 0.2 | 0.977 | 7.011 | 3.274 | 0.0768 | 5.63 | 1.169 |
| 24 | `h40_p4_SNS02_ang05_dlr_steps160` | screen | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | dlr | 0.5 | 1.084 | 7.447 | 3.018 | 0.0708 | 5.47 | 1.180 |
| 25 | `h40_p4_TXS012_ang05_warm_slowbb_steps160` | screen | 160 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s1_ffn+s2_half | warm_slowbb | 0.5 | 1.092 | 7.159 | 2.948 | 0.0691 | 5.24 | 1.181 |
| 26 | `h40_p4_TXS02_ang05_warm_steps160` | screen | 160 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s2_half | warm | 0.5 | 1.041 | 7.682 | 3.078 | 0.0722 | 5.57 | 1.183 |
| 27 | `h40_p4_SCS012_ang05_slowbb_steps160` | screen | 160 | 5 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | slowbb | 0.5 | 1.085 | 6.877 | 3.153 | 0.0740 | 5.40 | 1.187 |
| 28 | `h40_p2_SCS02_F_steps80` | screen | 80 | 5 | signed_consensus_shiftmax | s0_ffn+s2_half | differential_lr | 0 | 0.997 | 7.076 | 3.285 | 0.0771 | 5.87 | 1.194 |
| 29 | `h40_p4_HTS012_ang05_warm_steps160` | screen | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s1_ffn+s2_half | warm | 0.5 | 1.096 | 7.589 | 2.672 | 0.0627 | 4.63 | 1.196 |
| 30 | `h40_p4_SNS02_ang05_warm_steps160` | screen | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | warm | 0.5 | 1.059 | 7.530 | 3.229 | 0.0757 | 4.83 | 1.200 |

## valid40 / confirm 结果

| rank | 实验 | gate | steps | attention | FFN/范围 | AEE | AAE | SOPs(G) | firing | worstPN | score |
|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `h23e_h13v_lr1e5_target035_guard120_steps120_valid40` |  | 120 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.503 | 7.371 | 3.586 | 0.0841 |  | 1.670 |
| 2 | `h23a_h18c_lr1e5_target040_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.515 | 7.470 | 3.630 | 0.0852 |  | 1.686 |
| 3 | `h34_hybrid_h9_stage23_ffn_s150k_act2p0_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage23_ffn_official | 1.520 | 7.737 | 3.738 | 0.0877 |  | 1.700 |
| 4 | `h37_strict_bsa_qkv_sqrt_signv_neuronfast_steps120_valid40` |  | 120 | strict_bsa_qkv_shiftmax | stage02_highsop_official | 1.535 | 7.580 | 3.498 | 0.0821 |  | 1.702 |
| 5 | `h25g_ffn_all_binary_no_downsample_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn | 1.523 | 7.759 | 3.720 | 0.0873 |  | 1.703 |
| 6 | `h34_hybrid_h9_stage02_highsop_s150k_act2p0_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | 1.536 | 7.496 | 3.539 | 0.0830 |  | 1.704 |
| 7 | `h13w_sparse_feedback_stronger_guard120_steps120_valid40` |  | 120 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.535 | 7.557 | 3.581 | 0.0840 |  | 1.705 |
| 8 | `h34_hybrid_h9_ffn_sn1_s150k_act2p0_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | ffn_sn1_official | 1.528 | 7.728 | 3.682 | 0.0864 |  | 1.706 |
| 9 | `h30b_strict_bsa_thresholdv_diff_lr_steps120_valid40` |  | 120 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.535 | 7.633 | 3.595 | 0.0843 |  | 1.707 |
| 10 | `h35_signed_consensus_shiftmax_s150k_act2_steps120_valid40` |  | 120 | signed_consensus_shiftmax | highsop_official | 1.541 | 7.688 | 3.614 | 0.0848 |  | 1.715 |
| 11 | `h22e_h18c_score0p5_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.540 | 7.879 | 3.604 | 0.0845 |  | 1.717 |
| 12 | `h35_strict_bsa_signv_head_s150k_act2_steps120_valid40` |  | 120 | strict_bsa_shiftmax | highsop_official | 1.548 | 7.684 | 3.553 | 0.0833 |  | 1.719 |
| 13 | `h29b_diff_lr_binary_target_strong_steps360_valid40` |  | 360 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.548 | 7.817 | 3.517 | 0.0825 |  | 1.721 |
| 14 | `h23b_h18c_lr1e5_target035_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.553 | 7.634 | 3.532 | 0.0829 |  | 1.723 |
| 15 | `h28b_diff_lr_newfast_steps360_valid40` |  | 360 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.557 | 7.548 | 3.499 | 0.0821 |  | 1.724 |
| 16 | `h27a_strict_bsa_signv_sqrt_sparse040_guard120_steps120_valid40` |  | 120 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.543 | 7.865 | 3.712 | 0.0871 |  | 1.724 |
| 17 | `h36_stage02_highsop_conservative_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | 1.551 | 7.610 | 3.623 | 0.0850 |  | 1.724 |
| 18 | `h24f_h9ascope_axnor_ang005_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | 1.551 | 7.944 | 3.567 | 0.0837 |  | 1.729 |
| 19 | `h22g_h18c_alpha001_penalty05_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.551 | 7.814 | 3.632 | 0.0852 |  | 1.729 |
| 20 | `h22i_h18c_active_norm_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.552 | 7.778 | 3.642 | 0.0854 |  | 1.730 |
| 21 | `h36_stage02_highsop_threshfast_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | 1.557 | 7.655 | 3.630 | 0.0852 |  | 1.731 |
| 22 | `h24b_h9ascope_axnor_lr1e5_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | 1.564 | 7.442 | 3.575 | 0.0839 |  | 1.732 |
| 23 | `h31e_strict_bsa_sparse030_bin055_steps360_valid40` |  | 360 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.559 | 7.858 | 3.492 | 0.0819 |  | 1.732 |
| 24 | `h34_hybrid_h9_attn_aux_s150k_act2p0_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | attn_aux_official | 1.551 | 7.717 | 3.792 | 0.0890 |  | 1.733 |
| 25 | `h36_stage02_highsop_neuronfast_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | 1.562 | 7.602 | 3.592 | 0.0843 |  | 1.734 |
| 26 | `h34_hybrid_h9_highsop_s150k_act2p0_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | highsop_official | 1.567 | 7.498 | 3.543 | 0.0831 |  | 1.734 |
| 27 | `h24d_h9ascope_axnor_sparse035_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | 1.559 | 7.771 | 3.620 | 0.0849 |  | 1.735 |
| 28 | `h34_hybrid_h9_highsop_s300k_act4p0_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | highsop_official | 1.570 | 7.620 | 3.486 | 0.0818 |  | 1.738 |
| 29 | `h22a_h18c_target045_eta04_act07_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.558 | 7.915 | 3.657 | 0.0858 |  | 1.738 |
| 30 | `h23d_h13v_lr1e5_target040_guard120_steps120_valid40` |  | 120 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.560 | 7.814 | 3.643 | 0.0855 |  | 1.738 |
| 31 | `h28c_diff_lr_balanced_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.563 | 7.835 | 3.574 | 0.0838 |  | 1.739 |
| 32 | `h36_highsop_sparse_cur_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | highsop_official | 1.568 | 7.813 | 3.534 | 0.0829 |  | 1.742 |
| 33 | `h37_strict_bsa_qkv_sqrt_signv_conservative_steps120_valid40` |  | 120 | strict_bsa_qkv_shiftmax | stage02_highsop_official | 1.567 | 7.799 | 3.584 | 0.0841 |  | 1.743 |
| 34 | `h24c_h9ascope_axnor_sparse040_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | 1.565 | 7.674 | 3.705 | 0.0869 |  | 1.743 |
| 35 | `h22j_h18c_sign_value_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.572 | 7.578 | 3.592 | 0.0843 |  | 1.744 |
| 36 | `h23c_h18c_lr1e5_target040_score075_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.565 | 7.828 | 3.677 | 0.0863 |  | 1.745 |
| 37 | `h27b_strict_bsa_thetav_sqrt_sparse040_guard120_steps120_valid40` |  | 120 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.572 | 7.869 | 3.640 | 0.0854 |  | 1.751 |
| 38 | `h22b_h18c_target040_eta05_act08_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.579 | 7.665 | 3.566 | 0.0836 |  | 1.751 |
| 39 | `h35_signed_consensus_l1_s150k_act2_steps120_valid40` |  | 120 | signed_consensus_popcount_l1 | highsop_official | 1.584 | 7.732 | 3.480 | 0.0816 |  | 1.753 |
| 40 | `h31a_newfast_sparse030_bin055_steps360_valid40` |  | 360 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.576 | 7.977 | 3.539 | 0.0830 |  | 1.754 |
| 41 | `h22e_h18c_score0p75_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.576 | 7.919 | 3.588 | 0.0842 |  | 1.754 |
| 42 | `h24e_h9ascope_axnor_ang002_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | 1.571 | 8.058 | 3.684 | 0.0864 |  | 1.755 |
| 43 | `h28b_diff_lr_newfast_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.580 | 7.848 | 3.560 | 0.0835 |  | 1.756 |
| 44 | `h31d_h29b_high_binary_eta_steps360_valid40` |  | 360 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.585 | 7.812 | 3.500 | 0.0821 |  | 1.758 |
| 45 | `h36_stage02_highsop_backbone2x_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | 1.583 | 7.891 | 3.586 | 0.0841 |  | 1.760 |
| 46 | `h22f_h18c_alpha0_penalty025_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.585 | 7.731 | 3.647 | 0.0856 |  | 1.762 |
| 47 | `h35_strict_bsa_thresholdv_sqrt_s150k_act2_steps120_valid40` |  | 120 | strict_bsa_shiftmax | highsop_official | 1.593 | 7.781 | 3.452 | 0.0810 |  | 1.762 |
| 48 | `h35_alpha_xnor_shiftmax_a005_p05_s150k_act2_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | highsop_official | 1.593 | 7.736 | 3.491 | 0.0819 |  | 1.763 |
| 49 | `h30b_strict_bsa_thresholdv_diff_lr_steps360_valid40` |  | 360 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.589 | 7.821 | 3.562 | 0.0836 |  | 1.764 |
| 50 | `h35_alpha_xnor_shiftmax_a002_p025_s150k_act2_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | highsop_official | 1.589 | 7.866 | 3.558 | 0.0835 |  | 1.765 |
| 51 | `h25c_ffn_sn1_ternary_sn2_binary_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | ffn_sn1_ternary+ffn_sn2_binary+down02_binary | 1.584 | 7.870 | 3.686 | 0.0865 |  | 1.765 |
| 52 | `h22c_h18c_target035_eta08_act10_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.594 | 7.755 | 3.515 | 0.0825 |  | 1.766 |
| 53 | `h26h_hamming_ternary_sparse035_guard120_steps120_valid40` |  | 120 | hamming_ternary_active_direct | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | 1.597 | 7.701 | 3.514 | 0.0824 |  | 1.768 |
| 54 | `h29a_diff_lr_binary_target_mild_steps360_valid40` |  | 360 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.590 | 8.051 | 3.530 | 0.0828 |  | 1.768 |
| 55 | `h22e_h18c_score1p5_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.586 | 8.044 | 3.629 | 0.0851 |  | 1.768 |
| 56 | `h28c_diff_lr_balanced_steps360_valid40` |  | 360 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.589 | 8.186 | 3.559 | 0.0835 |  | 1.771 |
| 57 | `h35_strict_bsa_thresholdv_head_s150k_act2_steps120_valid40` |  | 120 | strict_bsa_shiftmax | highsop_official | 1.595 | 7.866 | 3.592 | 0.0843 |  | 1.772 |
| 58 | `h34_hybrid_h9_attn_aux_highsop_s150k_act2p0_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | attn_aux_highsop_official | 1.596 | 7.923 | 3.543 | 0.0831 |  | 1.772 |
| 59 | `h31f_strict_bsa_sparse028_bin045_steps360_valid40` |  | 360 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.608 | 7.918 | 3.376 | 0.0792 |  | 1.777 |
| 60 | `h35_alpha_xnor_l1_a002_p025_s150k_act2_steps120_valid40` |  | 120 | alpha_xnor_matrix_l1 | highsop_official | 1.609 | 7.926 | 3.507 | 0.0823 |  | 1.783 |
| 61 | `h24g_h9ascope_axnor_flowreg0003_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | 1.608 | 7.833 | 3.611 | 0.0847 |  | 1.785 |
| 62 | `h25b_ffn_sn2_only_binary_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | ffn_sn2_only_binary+down02_binary | 1.604 | 8.003 | 3.725 | 0.0874 |  | 1.789 |
| 63 | `h25f_ffn_all_ternary_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | 1.624 | 7.475 | 3.628 | 0.0851 |  | 1.794 |
| 64 | `h27d_strict_bsa_thetav_head_sparse040_guard120_steps120_valid40` |  | 120 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.614 | 7.921 | 3.713 | 0.0871 |  | 1.797 |
| 65 | `h31b_newfast_sparse028_bin045_steps360_valid40` |  | 360 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.622 | 7.990 | 3.490 | 0.0819 |  | 1.797 |
| 66 | `h26c_hamming_ternary_sparse040_guard120_steps120_valid40` |  | 120 | hamming_ternary_active_direct | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | 1.612 | 8.332 | 3.648 | 0.0856 |  | 1.800 |
| 67 | `h34_pure_official_qkonly_s150k_act2p0_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax |  | 1.611 | 8.538 | 3.597 | 0.0844 |  | 1.802 |
| 68 | `h35_hamming_ternary_active_s150k_act2_steps120_valid40` |  | 120 | hamming_ternary_active_direct | highsop_official | 1.628 | 8.103 | 3.563 | 0.0836 |  | 1.809 |
| 69 | `h22d_h18c_target030_eta10_act12_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.633 | 8.023 | 3.504 | 0.0822 |  | 1.809 |
| 70 | `h25d_ffn_sn1_binary_sn2_ternary_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | ffn_sn1_binary+ffn_sn2_ternary+down02_binary | 1.622 | 8.156 | 3.727 | 0.0874 |  | 1.810 |
| 71 | `h26d_hamming_binary_sparse040_guard120_steps120_valid40` |  | 120 | hamming_binary_direct | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | 1.626 | 8.272 | 3.646 | 0.0855 |  | 1.813 |
| 72 | `h35_compat_qk_shiftmax_s150k_act2_steps120_valid40` |  | 120 | compat_qk_product | highsop_official | 1.630 | 8.464 | 3.441 | 0.0807 |  | 1.813 |
| 73 | `h35_signed_consensus_shiftnorm_s150k_act2_steps120_valid40` |  | 120 | signed_consensus_shiftnorm | highsop_official | 1.639 | 8.399 | 3.665 | 0.0860 |  | 1.829 |
| 74 | `h24a_h9ascope_axnor_base_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | 1.662 | 7.754 | 3.552 | 0.0833 |  | 1.835 |
| 75 | `h22k_h18c_lr3em05_guard120_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.642 | 8.651 | 3.649 | 0.0856 |  | 1.837 |
| 76 | `h27c_strict_bsa_signv_head_sparse040_guard120_steps120_valid40` |  | 120 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.643 | 8.571 | 3.693 | 0.0866 |  | 1.838 |
| 77 | `h27f_strict_bsa_signv_sqrt_sparse035_guard120_steps120_valid40` |  | 120 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.644 | 8.591 | 3.706 | 0.0869 |  | 1.840 |
| 78 | `h27e_strict_bsa_signv_active_sparse040_guard120_steps120_valid40` |  | 120 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.647 | 8.635 | 3.703 | 0.0869 |  | 1.844 |
| 79 | `h18a_alpha_xnor_shiftmax_guard120_steps120_valid40` |  | 120 | ternary_alpha_xnor_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.683 | 7.935 | 3.429 | 0.0804 |  | 1.855 |
| 80 | `h34_pure_official_stage02_highsop_s150k_act2p0_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | 1.677 | 8.849 | 3.259 | 0.0764 |  | 1.861 |
| 81 | `h21b_hamming_ternary_active_direct_guard120_steps120_valid40` |  | 120 | hamming_ternary_active_direct | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | 1.677 | 8.424 | 3.586 | 0.0841 |  | 1.865 |
| 82 | `h34_pure_official_highsop_s300k_act4p0_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | highsop_official | 1.734 | 9.091 | 3.296 | 0.0773 |  | 1.924 |
| 83 | `h34_pure_official_highsop_s150k_act2p0_steps120_valid40` |  | 120 | alpha_xnor_matrix_shiftmax | highsop_official | 1.781 | 9.161 | 3.146 | 0.0738 |  | 1.966 |
| 84 | `h40_p2_SNS02_F_steps360_valid40` | AEE>1.58 | 360 | signed_consensus_shiftnorm | s0_ffn+s2_half | 1.776 | 8.391 | 3.072 | 0.0721 | 5.70 | 2.152 |
| 85 | `h40_p2_SCS02_F_steps360_valid40` | AEE>1.58 | 360 | signed_consensus_shiftmax | s0_ffn+s2_half | 1.767 | 8.642 | 3.134 | 0.0735 | 4.88 | 2.175 |
| 86 | `h40_p2_SNS012_F_steps360_valid40` | AEE>1.58 | 360 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | 1.866 | 8.841 | 2.889 | 0.0678 | 5.01 | 2.369 |
| 87 | `h40_p2_SCS012_F_steps360_valid40` | AEE>1.58 | 360 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | 1.879 | 9.099 | 2.842 | 0.0667 | 5.30 | 2.421 |

## 按批次展开的全部短测

### h34_h35_official_redo_priority_20260521_005427

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h35_signed_consensus_shiftmax_s150k_act2_steps120` |  | 120 | 10 | signed_consensus_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.013 | 6.128 | 3.764 | 0.0883 |  |  | 1.162 |
| `h35_signed_consensus_l1_s150k_act2_steps120` |  | 120 | 10 | signed_consensus_popcount_l1 | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.027 | 6.049 | 3.671 | 0.0861 |  |  | 1.170 |
| `h35_signed_consensus_shiftnorm_s150k_act2_steps120` |  | 120 | 10 | signed_consensus_shiftnorm | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.057 | 6.220 | 3.809 | 0.0894 |  |  | 1.210 |
| `h35_strict_bsa_signv_head_s150k_act2_steps120` |  | 120 | 10 | strict_bsa_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.065 | 6.624 | 3.739 | 0.0877 |  |  | 1.223 |
| `h34_hybrid_h9_attn_aux_s150k_act2p0_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | attn_aux_official | differential_lr | 0.035/1.8 | 0 | 1.062 | 6.480 | 3.943 | 0.0925 |  |  | 1.225 |
| `h35_alpha_xnor_shiftmax_a005_p05_s150k_act2_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.075 | 6.606 | 3.698 | 0.0867 |  |  | 1.231 |
| `h34_pure_official_qkonly_s150k_act2p0_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax |  | differential_lr | None/None | 0 | 1.081 | 6.488 | 3.699 | 0.0868 |  |  | 1.235 |
| `h34_hybrid_h9_stage02_highsop_s150k_act2p0_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.082 | 6.496 | 3.762 | 0.0882 |  |  | 1.238 |
| `h35_a2os2a_direct_l1_s150k_act2_steps120` |  | 120 | 10 | a2os2a_direct | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.071 | 6.317 | 4.280 | 0.1004 |  |  | 1.245 |
| `h35_alpha_xnor_shiftmax_a002_p025_s150k_act2_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.085 | 6.692 | 3.750 | 0.0880 |  |  | 1.245 |
| `h34_hybrid_h9_highsop_s300k_act4p0_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.102 | 6.667 | 3.691 | 0.0866 |  |  | 1.259 |
| `h35_compat_qk_shiftmax_s150k_act2_steps120` |  | 120 | 10 | compat_qk_product | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.108 | 6.674 | 3.579 | 0.0840 |  |  | 1.260 |
| `h34_hybrid_h9_ffn_sn1_s150k_act2p0_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | ffn_sn1_official | differential_lr | 0.035/1.8 | 0 | 1.104 | 6.738 | 3.861 | 0.0906 |  |  | 1.269 |
| `h34_hybrid_h9_stage23_ffn_s150k_act2p0_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage23_ffn_official | differential_lr | 0.035/1.8 | 0 | 1.104 | 6.804 | 3.928 | 0.0921 |  |  | 1.274 |
| `h34_hybrid_h9_attn_aux_highsop_s150k_act2p0_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | attn_aux_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.112 | 6.848 | 3.736 | 0.0876 |  |  | 1.274 |
| `h35_strict_bsa_thresholdv_head_s150k_act2_steps120` |  | 120 | 10 | strict_bsa_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.111 | 6.799 | 3.795 | 0.0890 |  |  | 1.275 |
| `h35_strict_bsa_thresholdv_sqrt_s150k_act2_steps120` |  | 120 | 10 | strict_bsa_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.126 | 6.877 | 3.647 | 0.0856 |  |  | 1.286 |
| `h34_hybrid_h9_highsop_s150k_act2p0_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.130 | 6.911 | 3.768 | 0.0884 |  |  | 1.295 |
| `h35_alpha_xnor_l1_a002_p025_s150k_act2_steps120` |  | 120 | 10 | alpha_xnor_matrix_l1 | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.142 | 7.237 | 3.734 | 0.0876 |  |  | 1.312 |
| `h34_pure_official_stage02_highsop_s150k_act2p0_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | None/None | 0 | 1.153 | 7.319 | 3.423 | 0.0803 |  |  | 1.312 |
| `h35_hamming_ternary_active_s150k_act2_steps120` |  | 120 | 10 | hamming_ternary_active_direct | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.159 | 7.049 | 3.698 | 0.0867 |  |  | 1.324 |
| `h34_pure_official_highsop_s300k_act4p0_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | None/None | 0 | 1.289 | 7.835 | 3.444 | 0.0808 |  |  | 1.459 |
| `h34_pure_official_highsop_s150k_act2p0_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | None/None | 0 | 1.299 | 8.018 | 3.317 | 0.0778 |  |  | 1.468 |
| `h34_hybrid_h9_stage23_ffn_s150k_act2p0_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage23_ffn_official | differential_lr | 0.035/1.8 | 0 | 1.520 | 7.737 | 3.738 | 0.0877 |  |  | 1.700 |
| `h34_hybrid_h9_stage02_highsop_s150k_act2p0_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.536 | 7.496 | 3.539 | 0.0830 |  |  | 1.704 |
| `h34_hybrid_h9_ffn_sn1_s150k_act2p0_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | ffn_sn1_official | differential_lr | 0.035/1.8 | 0 | 1.528 | 7.728 | 3.682 | 0.0864 |  |  | 1.706 |
| `h35_signed_consensus_shiftmax_s150k_act2_steps120_valid40` |  | 120 | 40 | signed_consensus_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.541 | 7.688 | 3.614 | 0.0848 |  |  | 1.715 |
| `h35_strict_bsa_signv_head_s150k_act2_steps120_valid40` |  | 120 | 40 | strict_bsa_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.548 | 7.684 | 3.553 | 0.0833 |  |  | 1.719 |
| `h34_hybrid_h9_attn_aux_s150k_act2p0_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | attn_aux_official | differential_lr | 0.035/1.8 | 0 | 1.551 | 7.717 | 3.792 | 0.0890 |  |  | 1.733 |
| `h34_hybrid_h9_highsop_s150k_act2p0_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.567 | 7.498 | 3.543 | 0.0831 |  |  | 1.734 |
| `h34_hybrid_h9_highsop_s300k_act4p0_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.570 | 7.620 | 3.486 | 0.0818 |  |  | 1.738 |
| `h35_signed_consensus_l1_s150k_act2_steps120_valid40` |  | 120 | 40 | signed_consensus_popcount_l1 | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.584 | 7.732 | 3.480 | 0.0816 |  |  | 1.753 |
| `h35_strict_bsa_thresholdv_sqrt_s150k_act2_steps120_valid40` |  | 120 | 40 | strict_bsa_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.593 | 7.781 | 3.452 | 0.0810 |  |  | 1.762 |
| `h35_alpha_xnor_shiftmax_a005_p05_s150k_act2_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.593 | 7.736 | 3.491 | 0.0819 |  |  | 1.763 |
| `h35_alpha_xnor_shiftmax_a002_p025_s150k_act2_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.589 | 7.866 | 3.558 | 0.0835 |  |  | 1.765 |
| `h35_strict_bsa_thresholdv_head_s150k_act2_steps120_valid40` |  | 120 | 40 | strict_bsa_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.595 | 7.866 | 3.592 | 0.0843 |  |  | 1.772 |
| `h34_hybrid_h9_attn_aux_highsop_s150k_act2p0_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | attn_aux_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.596 | 7.923 | 3.543 | 0.0831 |  |  | 1.772 |
| `h35_alpha_xnor_l1_a002_p025_s150k_act2_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_l1 | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.609 | 7.926 | 3.507 | 0.0823 |  |  | 1.783 |
| `h34_pure_official_qkonly_s150k_act2p0_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax |  | differential_lr | None/None | 0 | 1.611 | 8.538 | 3.597 | 0.0844 |  |  | 1.802 |
| `h35_hamming_ternary_active_s150k_act2_steps120_valid40` |  | 120 | 40 | hamming_ternary_active_direct | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.628 | 8.103 | 3.563 | 0.0836 |  |  | 1.809 |
| `h35_compat_qk_shiftmax_s150k_act2_steps120_valid40` |  | 120 | 40 | compat_qk_product | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.630 | 8.464 | 3.441 | 0.0807 |  |  | 1.813 |
| `h35_signed_consensus_shiftnorm_s150k_act2_steps120_valid40` |  | 120 | 40 | signed_consensus_shiftnorm | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.639 | 8.399 | 3.665 | 0.0860 |  |  | 1.829 |
| `h34_pure_official_stage02_highsop_s150k_act2p0_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | None/None | 0 | 1.677 | 8.849 | 3.259 | 0.0764 |  |  | 1.861 |
| `h34_pure_official_highsop_s300k_act4p0_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | None/None | 0 | 1.734 | 9.091 | 3.296 | 0.0773 |  |  | 1.924 |
| `h34_pure_official_highsop_s150k_act2p0_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | None/None | 0 | 1.781 | 9.161 | 3.146 | 0.0738 |  |  | 1.966 |

### h36_lr_strategy_sweep_20260521_020530

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h36_stage02_highsop_conservative_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.074 | 6.374 | 3.801 | 0.0892 |  |  | 1.229 |
| `h36_stage02_highsop_neuronfast_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.085 | 6.488 | 3.758 | 0.0882 |  |  | 1.241 |
| `h36_stage02_highsop_backbone2x_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.087 | 6.523 | 3.761 | 0.0882 |  |  | 1.244 |
| `h36_stage02_highsop_threshfast_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.091 | 6.526 | 3.824 | 0.0897 |  |  | 1.251 |
| `h36_highsop_sparse_cur_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.093 | 6.632 | 3.719 | 0.0872 |  |  | 1.251 |
| `h36_stage02_highsop_conservative_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.551 | 7.610 | 3.623 | 0.0850 |  |  | 1.724 |
| `h36_stage02_highsop_threshfast_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.557 | 7.655 | 3.630 | 0.0852 |  |  | 1.731 |
| `h36_stage02_highsop_neuronfast_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.562 | 7.602 | 3.592 | 0.0843 |  |  | 1.734 |
| `h36_highsop_sparse_cur_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.568 | 7.813 | 3.534 | 0.0829 |  |  | 1.742 |
| `h36_stage02_highsop_backbone2x_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.583 | 7.891 | 3.586 | 0.0841 |  |  | 1.760 |

### h37_reviewed_attention_20260521_022527

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h37_strict_bsa_qkv_sqrt_signv_neuronfast_steps120` |  | 120 | 10 | strict_bsa_qkv_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.081 | 6.675 | 3.682 | 0.0864 |  |  | 1.238 |
| `h37_strict_bsa_qkv_sqrt_signv_conservative_steps120` |  | 120 | 10 | strict_bsa_qkv_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.099 | 6.792 | 3.790 | 0.0889 |  |  | 1.262 |
| `h37_strict_bsa_qkv_sqrt_signv_neuronfast_steps120_valid40` |  | 120 | 40 | strict_bsa_qkv_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.535 | 7.580 | 3.498 | 0.0821 |  |  | 1.702 |
| `h37_strict_bsa_qkv_sqrt_signv_conservative_steps120_valid40` |  | 120 | 40 | strict_bsa_qkv_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.567 | 7.799 | 3.584 | 0.0841 |  |  | 1.743 |

### h40_p4_ang05_screen160_bs4x2_00_h40_p4_SNS02_ang05_dlr_20260522_020555

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SNS02_ang05_dlr_steps160` | pass | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | dlr | 0.05/2.0 | 0.5 | 1.084 | 7.447 | 3.018 | 0.0708 | 5.47 | 1 | 1.180 |

### h40_p4_ang05_screen160_bs4x2_01_h40_p4_SNS02_ang05_warm_20260522_020555

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SNS02_ang05_warm_steps160` | pass | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | warm | 0.05/2.0 | 0.5 | 1.059 | 7.530 | 3.229 | 0.0757 | 4.83 | 1 | 1.200 |

### h40_p4_ang05_screen160_bs4x2_02_h40_p4_SNS02_ang05_slowbb_20260522_020956

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SNS02_ang05_slowbb_steps160` | pass | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | slowbb | 0.05/2.0 | 0.5 | 1.051 | 7.317 | 3.151 | 0.0739 | 5.42 | 1 | 1.164 |

### h40_p4_ang05_screen160_bs4x2_03_h40_p4_SNS02_ang05_warm_slowbb_20260522_020956

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SNS02_ang05_warm_slowbb_steps160` | pos_neg_ratio>40.0 | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | warm_slowbb | 0.05/2.0 | 0.5 | 1.087 | 7.724 | 3.230 | 0.0758 | 200938.79 | 1 | 403.111 |

### h40_p4_ang05_screen160_bs4x2_04_h40_p4_TXS02_ang05_dlr_20260522_021356

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_TXS02_ang05_dlr_steps160` | AAE>7.9 | 160 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s2_half | dlr | 0.05/2.0 | 0.5 | 1.080 | 7.962 | 3.138 | 0.0736 | 4.65 | 0 | 1.250 |

### h40_p4_ang05_screen160_bs4x2_05_h40_p4_TXS02_ang05_warm_20260522_021356

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_TXS02_ang05_warm_steps160` | pass | 160 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s2_half | warm | 0.05/2.0 | 0.5 | 1.041 | 7.682 | 3.078 | 0.0722 | 5.57 | 1 | 1.183 |

### h40_p4_ang05_screen160_bs4x2_06_h40_p4_TXS02_ang05_slowbb_20260522_021756

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_TXS02_ang05_slowbb_steps160` | AAE>7.9 | 160 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s2_half | slowbb | 0.05/2.0 | 0.5 | 1.138 | 7.955 | 3.350 | 0.0786 | 5.03 | 1 | 1.421 |

### h40_p4_ang05_screen160_bs4x2_07_h40_p4_TXS02_ang05_warm_slowbb_20260522_021756

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_TXS02_ang05_warm_slowbb_steps160` | pos_neg_ratio>40.0 | 160 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s2_half | warm_slowbb | 0.05/2.0 | 0.5 | 1.107 | 7.667 | 3.213 | 0.0754 | 66979.60 | 1 | 135.206 |

### h40_p4_ang05_screen160_bs4x2_08_h40_p4_HTS02_ang02_dlr_20260522_022147

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_HTS02_ang02_dlr_steps160` | pass | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | dlr | 0.05/2.0 | 0.2 | 1.046 | 7.525 | 3.105 | 0.0728 | 4.69 | 1 | 1.150 |

### h40_p4_ang05_screen160_bs4x2_09_h40_p4_HTS02_ang02_warm_20260522_022147

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_HTS02_ang02_warm_steps160` | pos_neg_ratio>40.0 | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | warm | 0.05/2.0 | 0.2 | 1.073 | 7.715 | 3.135 | 0.0735 | 133959.19 | 1 | 269.109 |

### h40_p4_ang05_screen160_bs4x2_10_h40_p4_HTS02_ang02_slowbb_20260522_022547

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_HTS02_ang02_slowbb_steps160` | pass | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | slowbb | 0.05/2.0 | 0.2 | 1.024 | 7.453 | 3.039 | 0.0713 | 4.64 | 1 | 1.120 |

### h40_p4_ang05_screen160_bs4x2_11_h40_p4_HTS02_ang02_warm_slowbb_20260522_022547

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_HTS02_ang02_warm_slowbb_steps160` | pass | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | warm_slowbb | 0.05/2.0 | 0.2 | 0.993 | 7.276 | 3.186 | 0.0747 | 4.73 | 1 | 1.115 |

### h40_p4_ang05_screen160_bs4x2_12_h40_p4_SCS02_ang05_dlr_20260522_022947

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SCS02_ang05_dlr_steps160` | pass | 160 | 5 | signed_consensus_shiftmax | s0_ffn+s2_half | dlr | 0.05/2.0 | 0.5 | 1.089 | 7.798 | 3.136 | 0.0736 | 5.45 | 1 | 1.249 |

### h40_p4_ang05_screen160_bs4x2_13_h40_p4_SCS02_ang05_warm_20260522_022947

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SCS02_ang05_warm_steps160` | pass | 160 | 5 | signed_consensus_shiftmax | s0_ffn+s2_half | warm | 0.05/2.0 | 0.5 | 1.085 | 7.163 | 3.213 | 0.0754 | 5.12 | 1 | 1.213 |

### h40_p4_ang05_screen160_bs4x2_14_h40_p4_SCS02_ang05_slowbb_20260522_023348

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SCS02_ang05_slowbb_steps160` | pass | 160 | 5 | signed_consensus_shiftmax | s0_ffn+s2_half | slowbb | 0.05/2.0 | 0.5 | 1.068 | 7.767 | 3.186 | 0.0747 | 5.19 | 1 | 1.243 |

### h40_p4_ang05_screen160_bs4x2_15_h40_p4_SCS02_ang05_warm_slowbb_20260522_023348

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SCS02_ang05_warm_slowbb_steps160` | SOPs>3.35G | 160 | 5 | signed_consensus_shiftmax | s0_ffn+s2_half | warm_slowbb | 0.05/2.0 | 0.5 | 0.989 | 6.933 | 3.357 | 0.0788 | 66979.60 | 1 | 135.123 |

### h40_p4_ang05_screen160_bs4x2_16_h40_p4_SCS012_ang05_dlr_20260522_023748

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SCS012_ang05_dlr_steps160` | pass | 160 | 5 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | dlr | 0.05/2.0 | 0.5 | 1.021 | 6.550 | 2.948 | 0.0691 | 5.07 | 1 | 1.094 |

### h40_p4_ang05_screen160_bs4x2_17_h40_p4_SCS012_ang05_warm_20260522_023748

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SCS012_ang05_warm_steps160` | pass | 160 | 5 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | warm | 0.05/2.0 | 0.5 | 1.043 | 7.308 | 2.866 | 0.0672 | 5.00 | 1 | 1.135 |

### h40_p4_ang05_screen160_bs4x2_18_h40_p4_SCS012_ang05_slowbb_20260522_024159

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SCS012_ang05_slowbb_steps160` | pass | 160 | 5 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | slowbb | 0.05/2.0 | 0.5 | 1.085 | 6.877 | 3.153 | 0.0740 | 5.40 | 1 | 1.187 |

### h40_p4_ang05_screen160_bs4x2_19_h40_p4_SCS012_ang05_warm_slowbb_20260522_024159

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SCS012_ang05_warm_slowbb_steps160` | pass | 160 | 5 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | warm_slowbb | 0.05/2.0 | 0.5 | 1.050 | 6.768 | 2.926 | 0.0686 | 5.16 | 1 | 1.130 |

### h40_p4_ang05_screen160_bs4x2_20_h40_p4_SNS012_ang05_dlr_20260522_024609

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SNS012_ang05_dlr_steps160` | pass | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | dlr | 0.05/2.0 | 0.5 | 1.030 | 6.690 | 2.901 | 0.0680 | 5.10 | 1 | 1.107 |

### h40_p4_ang05_screen160_bs4x2_21_h40_p4_SNS012_ang05_warm_20260522_024609

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SNS012_ang05_warm_steps160` | pass | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | warm | 0.05/2.0 | 0.5 | 1.068 | 7.268 | 2.918 | 0.0685 | 5.28 | 1 | 1.160 |

### h40_p4_ang05_screen160_bs4x2_22_h40_p4_SNS012_ang05_slowbb_20260522_025020

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SNS012_ang05_slowbb_steps160` | pass | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | slowbb | 0.05/2.0 | 0.5 | 1.064 | 7.404 | 2.901 | 0.0681 | 5.31 | 1 | 1.159 |

### h40_p4_ang05_screen160_bs4x2_23_h40_p4_SNS012_ang05_warm_slowbb_20260522_025020

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SNS012_ang05_warm_slowbb_steps160` | pass | 160 | 5 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | warm_slowbb | 0.05/2.0 | 0.5 | 1.079 | 6.898 | 2.987 | 0.0701 | 5.55 | 1 | 1.162 |

### h40_p4_ang05_screen160_bs4x2_24_h40_p4_TXS012_ang05_dlr_20260522_025430

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_TXS012_ang05_dlr_steps160` | pos_neg_ratio>40.0 | 160 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s1_ffn+s2_half | dlr | 0.05/2.0 | 0.5 | 1.035 | 6.993 | 3.009 | 0.0706 | 66979.60 | 1 | 135.039 |

### h40_p4_ang05_screen160_bs4x2_25_h40_p4_TXS012_ang05_warm_20260522_025430

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_TXS012_ang05_warm_steps160` | pass | 160 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s1_ffn+s2_half | warm | 0.05/2.0 | 0.5 | 0.980 | 7.046 | 2.974 | 0.0698 | 5.09 | 1 | 1.066 |

### h40_p4_ang05_screen160_bs4x2_26_h40_p4_TXS012_ang05_slowbb_20260522_025840

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_TXS012_ang05_slowbb_steps160` | pass | 160 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s1_ffn+s2_half | slowbb | 0.05/2.0 | 0.5 | 1.018 | 6.856 | 2.924 | 0.0686 | 5.64 | 1 | 1.100 |

### h40_p4_ang05_screen160_bs4x2_27_h40_p4_TXS012_ang05_warm_slowbb_20260522_025840

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_TXS012_ang05_warm_slowbb_steps160` | pass | 160 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s1_ffn+s2_half | warm_slowbb | 0.05/2.0 | 0.5 | 1.092 | 7.159 | 2.948 | 0.0691 | 5.24 | 1 | 1.181 |

### h40_p4_ang05_screen160_bs4x2_28_h40_p4_HTS012_ang05_dlr_20260522_030251

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_HTS012_ang05_dlr_steps160` | pass | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s1_ffn+s2_half | dlr | 0.05/2.0 | 0.5 | 1.055 | 7.345 | 2.643 | 0.0620 | 4.55 | 1 | 1.149 |

### h40_p4_ang05_screen160_bs4x2_29_h40_p4_HTS012_ang05_warm_20260522_030251

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_HTS012_ang05_warm_steps160` | pass | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s1_ffn+s2_half | warm | 0.05/2.0 | 0.5 | 1.096 | 7.589 | 2.672 | 0.0627 | 4.63 | 1 | 1.196 |

### h40_p4_ang05_screen160_bs4x2_30_h40_p4_HTS012_ang05_slowbb_20260522_030701

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_HTS012_ang05_slowbb_steps160` | pass | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s1_ffn+s2_half | slowbb | 0.05/2.0 | 0.5 | 1.050 | 7.155 | 2.947 | 0.0691 | 4.40 | 1 | 1.139 |

### h40_p4_ang05_screen160_bs4x2_31_h40_p4_HTS012_ang05_warm_slowbb_20260522_030701

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_HTS012_ang05_warm_slowbb_steps160` | pass | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s1_ffn+s2_half | warm_slowbb | 0.05/2.0 | 0.5 | 1.058 | 7.281 | 2.853 | 0.0669 | 4.07 | 0 | 1.150 |

### h40_p4_ang05_screen160_bs4x2_32_h40_p4_HTS02_ang05_dlr_20260522_031111

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_HTS02_ang05_dlr_steps160` | pass | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | dlr | 0.05/2.0 | 0.5 | 1.136 | 7.884 | 3.138 | 0.0736 | 4.58 | 1 | 1.299 |

### h40_p4_ang05_screen160_bs4x2_33_h40_p4_HTS02_ang05_warm_20260522_031111

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_HTS02_ang05_warm_steps160` | pass | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | warm | 0.05/2.0 | 0.5 | 1.053 | 7.664 | 3.190 | 0.0748 | 4.64 | 1 | 1.226 |

### h40_p4_ang05_screen160_bs4x2_34_h40_p4_HTS02_ang05_slowbb_20260522_031452

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_HTS02_ang05_slowbb_steps160` | pass | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | slowbb | 0.05/2.0 | 0.5 | 0.987 | 7.683 | 3.036 | 0.0712 | 4.48 | 1 | 1.129 |

### h40_p4_ang05_screen160_bs4x2_35_h40_p4_HTS02_ang05_warm_slowbb_20260522_031452

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_HTS02_ang05_warm_slowbb_steps160` | AAE>7.9 | 160 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | warm_slowbb | 0.05/2.0 | 0.5 | 1.096 | 7.933 | 3.270 | 0.0767 | 133959.19 | 1 | 269.231 |

### h40_p4_ang05_screen160_bs4x2_36_h40_p4_SLS02_ang05_dlr_20260522_031852

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SLS02_ang05_dlr_steps160` | pos_neg_ratio>40.0 | 160 | 5 | signed_consensus_popcount_l1 | s0_ffn+s2_half | dlr | 0.05/2.0 | 0.5 | 1.114 | 7.491 | 3.215 | 0.0754 | 66979.60 | 1 | 135.170 |

### h40_p4_ang05_screen160_bs4x2_37_h40_p4_SLS02_ang05_warm_20260522_031852

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SLS02_ang05_warm_steps160` | pass | 160 | 5 | signed_consensus_popcount_l1 | s0_ffn+s2_half | warm | 0.05/2.0 | 0.5 | 1.111 | 7.283 | 3.230 | 0.0758 | 5.26 | 1 | 1.247 |

### h40_p4_ang05_screen160_bs4x2_38_h40_p4_SLS02_ang05_slowbb_20260522_032252

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SLS02_ang05_slowbb_steps160` | pass | 160 | 5 | signed_consensus_popcount_l1 | s0_ffn+s2_half | slowbb | 0.05/2.0 | 0.5 | 1.034 | 7.557 | 3.291 | 0.0772 | 5.17 | 1 | 1.244 |

### h40_p4_ang05_screen160_bs4x2_39_h40_p4_SLS02_ang05_warm_slowbb_20260522_032252

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p4_SLS02_ang05_warm_slowbb_steps160` | AAE>7.9 | 160 | 5 | signed_consensus_popcount_l1 | s0_ffn+s2_half | warm_slowbb | 0.05/2.0 | 0.5 | 1.118 | 8.377 | 3.028 | 0.0710 | 5.18 | 1 | 1.316 |

### rapid_screen_h18_direct_h13fix_20260519_232711

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h13v_target05_lower_lr_guard120_steps120` |  | 120 | 10 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 0.961 | 5.903 | 3.829 | 0.0898 |  |  | 1.108 |
| `h13w_sparse_feedback_stronger_guard120_steps120` |  | 120 | 10 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.035/1.8 | 0 | 0.992 | 5.735 | 3.729 | 0.0875 |  |  | 1.132 |
| `h13x_threshold_frozen_guard120_steps120` |  | 120 | 10 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 0.994 | 5.954 | 3.932 | 0.0922 |  |  | 1.146 |
| `h18e_a2os2a_direct_l1_guard120_steps120` |  | 120 | 10 | a2os2a_direct | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.047 | 6.292 | 4.325 | 0.1015 |  |  | 1.222 |
| `h18c_alpha_xnor_direct_shiftmax_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.088 | 6.732 | 3.814 | 0.0895 |  |  | 1.251 |
| `h18d_alpha_xnor_direct_l1_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_l1 | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.125 | 6.933 | 4.233 | 0.0993 |  |  | 1.309 |
| `h13w_sparse_feedback_stronger_guard120_steps120_valid40` |  | 120 | 40 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.035/1.8 | 0 | 1.535 | 7.557 | 3.581 | 0.0840 |  |  | 1.705 |
| `h13x_threshold_frozen_guard120_steps40` |  | 40 | 10 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 7.102 | 86.236 | 2.443 | 0.0573 |  |  | 8.826 |
| `h13w_sparse_feedback_stronger_guard120_steps40` |  | 40 | 10 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.035/1.8 | 0 | 7.197 | 91.899 | 2.373 | 0.0557 |  |  | 9.035 |
| `h18e_a2os2a_direct_l1_guard120_steps40` |  | 40 | 10 | a2os2a_direct | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 7.299 | 90.642 | 2.897 | 0.0680 |  |  | 9.112 |
| `h18c_alpha_xnor_direct_shiftmax_guard120_steps40` |  | 40 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 7.273 | 93.273 | 2.448 | 0.0574 |  |  | 9.139 |
| `h18d_alpha_xnor_direct_l1_guard120_steps40` |  | 40 | 10 | alpha_xnor_matrix_l1 | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 7.349 | 94.689 | 2.802 | 0.0657 |  |  | 9.242 |
| `h13v_target05_lower_lr_guard120_steps40` |  | 40 | 10 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 7.374 | 96.301 | 2.400 | 0.0563 |  |  | 9.300 |

### rapid_screen_h18_h13fix_20260519_232157

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h18a_alpha_xnor_shiftmax_guard120_steps120` |  | 120 | 10 | ternary_alpha_xnor_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.088 | 6.406 | 3.624 | 0.0850 |  |  | 1.237 |
| `h18a_alpha_xnor_shiftmax_guard120_steps120_valid40` |  | 120 | 40 | ternary_alpha_xnor_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.683 | 7.935 | 3.429 | 0.0804 |  |  | 1.855 |
| `h18a_alpha_xnor_shiftmax_guard120_steps40` |  | 40 | 10 | ternary_alpha_xnor_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 7.268 | 94.588 | 2.305 | 0.0541 |  |  | 9.160 |

### rapid_screen_h21_hamming_20260519_235214

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h21b_hamming_ternary_active_direct_guard120_steps120` |  | 120 | 10 | hamming_ternary_active_direct | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.101 | 6.367 | 3.743 | 0.0878 |  |  | 1.254 |
| `h21c_hamming_binary_signv_guard120_steps120` |  | 120 | 10 | hamming_binary_direct | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.102 | 7.033 | 4.102 | 0.0962 |  |  | 1.282 |
| `h21a_spikevideo_hamming_binary_direct_guard120_steps120` |  | 120 | 10 | hamming_binary_direct | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.111 | 6.895 | 4.037 | 0.0947 |  |  | 1.286 |
| `h21b_hamming_ternary_active_direct_guard120_steps120_valid40` |  | 120 | 40 | hamming_ternary_active_direct | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.677 | 8.424 | 3.586 | 0.0841 |  |  | 1.865 |
| `h21a_spikevideo_hamming_binary_direct_guard120_steps40` |  | 40 | 10 | hamming_binary_direct | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 7.311 | 93.840 | 2.767 | 0.0649 |  |  | 9.188 |
| `h21b_hamming_ternary_active_direct_guard120_steps40` |  | 40 | 10 | hamming_ternary_active_direct | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 7.369 | 96.120 | 2.273 | 0.0533 |  |  | 9.292 |
| `h21c_hamming_binary_signv_guard120_steps40` |  | 40 | 10 | hamming_binary_direct | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 7.402 | 94.900 | 2.760 | 0.0647 |  |  | 9.300 |

### rapid_screen_h22_h18c_hparam_20260520_001015

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h22c_h18c_target035_eta08_act10_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.035/1.8 | 0 | 1.027 | 6.046 | 3.709 | 0.0870 |  |  | 1.173 |
| `h22g_h18c_alpha001_penalty05_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.058 | 6.549 | 3.818 | 0.0896 |  |  | 1.218 |
| `h22f_h18c_alpha0_penalty025_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.063 | 6.484 | 3.847 | 0.0902 |  |  | 1.223 |
| `h22e_h18c_score1p5_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.067 | 6.527 | 3.807 | 0.0893 |  |  | 1.226 |
| `h22a_h18c_target045_eta04_act07_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.045/1.8 | 0 | 1.076 | 6.322 | 3.846 | 0.0902 |  |  | 1.232 |
| `h22j_h18c_sign_value_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.067 | 6.871 | 3.810 | 0.0894 |  |  | 1.233 |
| `h22k_h18c_lr3em05_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.069 | 7.025 | 3.874 | 0.0909 |  |  | 1.241 |
| `h22e_h18c_score0p75_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.090 | 6.314 | 3.795 | 0.0890 |  |  | 1.244 |
| `h22i_h18c_active_norm_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.090 | 6.333 | 3.811 | 0.0894 |  |  | 1.245 |
| `h22e_h18c_score0p5_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.088 | 6.997 | 3.800 | 0.0891 |  |  | 1.256 |
| `h22k_h18c_lr1em05_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.107 | 6.608 | 3.928 | 0.0921 |  |  | 1.272 |
| `h22b_h18c_target040_eta05_act08_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/1.8 | 0 | 1.106 | 7.065 | 3.760 | 0.0882 |  |  | 1.274 |
| `h22d_h18c_target030_eta10_act12_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.03/1.8 | 0 | 1.108 | 7.199 | 3.728 | 0.0874 |  |  | 1.277 |
| `h22e_h18c_score0p5_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.540 | 7.879 | 3.604 | 0.0845 |  |  | 1.717 |
| `h22g_h18c_alpha001_penalty05_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.551 | 7.814 | 3.632 | 0.0852 |  |  | 1.729 |
| `h22i_h18c_active_norm_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.552 | 7.778 | 3.642 | 0.0854 |  |  | 1.730 |
| `h22a_h18c_target045_eta04_act07_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.045/1.8 | 0 | 1.558 | 7.915 | 3.657 | 0.0858 |  |  | 1.738 |
| `h22j_h18c_sign_value_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.572 | 7.578 | 3.592 | 0.0843 |  |  | 1.744 |
| `h22b_h18c_target040_eta05_act08_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/1.8 | 0 | 1.579 | 7.665 | 3.566 | 0.0836 |  |  | 1.751 |
| `h22e_h18c_score0p75_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.576 | 7.919 | 3.588 | 0.0842 |  |  | 1.754 |
| `h22f_h18c_alpha0_penalty025_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.585 | 7.731 | 3.647 | 0.0856 |  |  | 1.762 |
| `h22c_h18c_target035_eta08_act10_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.035/1.8 | 0 | 1.594 | 7.755 | 3.515 | 0.0825 |  |  | 1.766 |
| `h22e_h18c_score1p5_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.586 | 8.044 | 3.629 | 0.0851 |  |  | 1.768 |
| `h22d_h18c_target030_eta10_act12_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.03/1.8 | 0 | 1.633 | 8.023 | 3.504 | 0.0822 |  |  | 1.809 |
| `h22k_h18c_lr3em05_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.05/1.8 | 0 | 1.642 | 8.651 | 3.649 | 0.0856 |  |  | 1.837 |

### rapid_screen_h23_low_lr_sparse_combo_20260520_005352

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h23e_h13v_lr1e5_target035_guard120_steps120` |  | 120 | 10 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.035/1.8 | 0 | 0.999 | 6.032 | 3.734 | 0.0876 |  |  | 1.145 |
| `h23d_h13v_lr1e5_target040_guard120_steps120` |  | 120 | 10 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/1.8 | 0 | 1.007 | 6.002 | 3.807 | 0.0893 |  |  | 1.156 |
| `h23a_h18c_lr1e5_target040_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/1.8 | 0 | 1.050 | 6.194 | 3.818 | 0.0896 |  |  | 1.203 |
| `h23b_h18c_lr1e5_target035_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.035/1.8 | 0 | 1.083 | 6.774 | 3.766 | 0.0883 |  |  | 1.245 |
| `h23c_h18c_lr1e5_target040_score075_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/1.8 | 0 | 1.111 | 6.707 | 3.866 | 0.0907 |  |  | 1.276 |
| `h23e_h13v_lr1e5_target035_guard120_steps120_valid40` |  | 120 | 40 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.035/1.8 | 0 | 1.503 | 7.371 | 3.586 | 0.0841 |  |  | 1.670 |
| `h23a_h18c_lr1e5_target040_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/1.8 | 0 | 1.515 | 7.470 | 3.630 | 0.0852 |  |  | 1.686 |
| `h23b_h18c_lr1e5_target035_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.035/1.8 | 0 | 1.553 | 7.634 | 3.532 | 0.0829 |  |  | 1.723 |
| `h23d_h13v_lr1e5_target040_guard120_steps120_valid40` |  | 120 | 40 | signed_consensus_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/1.8 | 0 | 1.560 | 7.814 | 3.643 | 0.0855 |  |  | 1.738 |
| `h23c_h18c_lr1e5_target040_score075_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/1.8 | 0 | 1.565 | 7.828 | 3.677 | 0.0863 |  |  | 1.745 |

### rapid_screen_h24_h9ascope_axnor_hparam_20260520_010832

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h24b_h9ascope_axnor_lr1e5_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | /0.13 | 0 | 1.096 | 6.073 | 3.746 | 0.0879 |  |  | 1.244 |
| `h24a_h9ascope_axnor_base_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | /0.13 | 0 | 1.135 | 5.993 | 3.731 | 0.0875 |  |  | 1.280 |
| `h24c_h9ascope_axnor_sparse040_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.126 | 6.288 | 3.882 | 0.0911 |  |  | 1.283 |
| `h24b_h9ascope_axnor_lr1e5_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | /0.13 | 0 | 1.564 | 7.442 | 3.575 | 0.0839 |  |  | 1.732 |
| `h24c_h9ascope_axnor_sparse040_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.565 | 7.674 | 3.705 | 0.0869 |  |  | 1.743 |
| `h24a_h9ascope_axnor_base_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | /0.13 | 0 | 1.662 | 7.754 | 3.552 | 0.0833 |  |  | 1.835 |

### rapid_screen_h24_remaining_after_cleanup_20260520_013701

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h24d_h9ascope_axnor_sparse035_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.035/0.13 | 0 | 1.043 | 5.779 | 3.813 | 0.0894 |  |  | 1.187 |
| `h24e_h9ascope_axnor_ang002_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0.02 | 1.085 | 6.225 | 3.845 | 0.0902 |  |  | 1.239 |
| `h24g_h9ascope_axnor_flowreg0003_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.092 | 6.213 | 3.807 | 0.0893 |  |  | 1.245 |
| `h24f_h9ascope_axnor_ang005_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0.05 | 1.096 | 6.617 | 3.735 | 0.0876 |  |  | 1.253 |
| `h24f_h9ascope_axnor_ang005_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0.05 | 1.551 | 7.944 | 3.567 | 0.0837 |  |  | 1.729 |
| `h24d_h9ascope_axnor_sparse035_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.035/0.13 | 0 | 1.559 | 7.771 | 3.620 | 0.0849 |  |  | 1.735 |
| `h24e_h9ascope_axnor_ang002_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0.02 | 1.571 | 8.058 | 3.684 | 0.0864 |  |  | 1.755 |
| `h24g_h9ascope_axnor_flowreg0003_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.608 | 7.833 | 3.611 | 0.0847 |  |  | 1.785 |

### rapid_screen_h25_module_combinations_20260520_014859

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h25e_no_ffn_downsample_only_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | down02_binary | single_lr | 0.04/0.13 | 0 | 1.045 | 6.141 | 3.915 | 0.0918 |  |  | 1.201 |
| `h25g_ffn_all_binary_no_downsample_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn | single_lr | 0.04/0.13 | 0 | 1.058 | 5.910 | 3.874 | 0.0909 |  |  | 1.207 |
| `h25a_ffn_sn1_only_binary_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | ffn_sn1_only_binary+down02_binary | single_lr | 0.04/0.13 | 0 | 1.077 | 6.025 | 3.950 | 0.0927 |  |  | 1.232 |
| `h25b_ffn_sn2_only_binary_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | ffn_sn2_only_binary+down02_binary | single_lr | 0.04/0.13 | 0 | 1.095 | 5.927 | 3.899 | 0.0915 |  |  | 1.246 |
| `h25f_ffn_all_ternary_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.098 | 6.322 | 3.862 | 0.0906 |  |  | 1.255 |
| `h25c_ffn_sn1_ternary_sn2_binary_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | ffn_sn1_ternary+ffn_sn2_binary+down02_binary | single_lr | 0.04/0.13 | 0 | 1.105 | 6.197 | 3.864 | 0.0906 |  |  | 1.260 |
| `h25d_ffn_sn1_binary_sn2_ternary_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | ffn_sn1_binary+ffn_sn2_ternary+down02_binary | single_lr | 0.04/0.13 | 0 | 1.145 | 6.188 | 3.892 | 0.0913 |  |  | 1.301 |
| `h25g_ffn_all_binary_no_downsample_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn | single_lr | 0.04/0.13 | 0 | 1.523 | 7.759 | 3.720 | 0.0873 |  |  | 1.703 |
| `h25c_ffn_sn1_ternary_sn2_binary_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | ffn_sn1_ternary+ffn_sn2_binary+down02_binary | single_lr | 0.04/0.13 | 0 | 1.584 | 7.870 | 3.686 | 0.0865 |  |  | 1.765 |
| `h25b_ffn_sn2_only_binary_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | ffn_sn2_only_binary+down02_binary | single_lr | 0.04/0.13 | 0 | 1.604 | 8.003 | 3.725 | 0.0874 |  |  | 1.789 |
| `h25f_ffn_all_ternary_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.624 | 7.475 | 3.628 | 0.0851 |  |  | 1.794 |
| `h25d_ffn_sn1_binary_sn2_ternary_guard120_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | ffn_sn1_binary+ffn_sn2_ternary+down02_binary | single_lr | 0.04/0.13 | 0 | 1.622 | 8.156 | 3.727 | 0.0874 |  |  | 1.810 |

### rapid_screen_h26_attention_revisit_20260520_020913

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h26h_hamming_ternary_sparse035_guard120_steps120` |  | 120 | 10 | hamming_ternary_active_direct | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.035/0.13 | 0 | 1.081 | 6.458 | 3.683 | 0.0864 |  |  | 1.233 |
| `h26a_axnor_l1_sparse040_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_l1 | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.107 | 6.299 | 4.088 | 0.0959 |  |  | 1.273 |
| `h26b_a2os2a_sparse040_guard120_steps120` |  | 120 | 10 | a2os2a_direct | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.103 | 6.458 | 4.164 | 0.0977 |  |  | 1.275 |
| `h26d_hamming_binary_sparse040_guard120_steps120` |  | 120 | 10 | hamming_binary_direct | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.138 | 6.319 | 3.822 | 0.0897 |  |  | 1.293 |
| `h26f_axnor_l1_ffn_ternary_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_l1 | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.122 | 6.449 | 4.168 | 0.0978 |  |  | 1.294 |
| `h26i_axnor_l1_flowreg0003_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_l1 | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.127 | 6.463 | 4.121 | 0.0967 |  |  | 1.297 |
| `h26c_hamming_ternary_sparse040_guard120_steps120` |  | 120 | 10 | hamming_ternary_active_direct | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.138 | 6.614 | 3.787 | 0.0888 |  |  | 1.298 |
| `h26e_axnor_shiftmax_signv_sparse040_guard120_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.151 | 6.349 | 4.475 | 0.1050 |  |  | 1.333 |
| `h26g_a2os2a_ffn_sn1_ternary_guard120_steps120` |  | 120 | 10 | a2os2a_direct | ffn_sn1_ternary+ffn_sn2_binary+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.169 | 7.028 | 4.017 | 0.0942 |  |  | 1.346 |
| `h26h_hamming_ternary_sparse035_guard120_steps120_valid40` |  | 120 | 40 | hamming_ternary_active_direct | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.035/0.13 | 0 | 1.597 | 7.701 | 3.514 | 0.0824 |  |  | 1.768 |
| `h26c_hamming_ternary_sparse040_guard120_steps120_valid40` |  | 120 | 40 | hamming_ternary_active_direct | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.612 | 8.332 | 3.648 | 0.0856 |  |  | 1.800 |
| `h26d_hamming_binary_sparse040_guard120_steps120_valid40` |  | 120 | 40 | hamming_binary_direct | stage0_ffn+stage3_block0_ffn+downsample_stage0_stage2 | single_lr | 0.04/0.13 | 0 | 1.626 | 8.272 | 3.646 | 0.0855 |  |  | 1.813 |

### rapid_screen_h27_strict_bsa_standard_20260520_100940

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h27d_strict_bsa_thetav_head_sparse040_guard120_steps120` |  | 120 | 10 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/0.13 | 0 | 1.107 | 5.913 | 3.887 | 0.0912 |  |  | 1.257 |
| `h27a_strict_bsa_signv_sqrt_sparse040_guard120_steps120` |  | 120 | 10 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/0.13 | 0 | 1.095 | 6.614 | 3.872 | 0.0908 |  |  | 1.258 |
| `h27b_strict_bsa_thetav_sqrt_sparse040_guard120_steps120` |  | 120 | 10 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/0.13 | 0 | 1.112 | 6.301 | 3.834 | 0.0899 |  |  | 1.268 |
| `h27e_strict_bsa_signv_active_sparse040_guard120_steps120` |  | 120 | 10 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/0.13 | 0 | 1.154 | 6.692 | 3.880 | 0.0910 |  |  | 1.319 |
| `h27c_strict_bsa_signv_head_sparse040_guard120_steps120` |  | 120 | 10 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/0.13 | 0 | 1.159 | 6.574 | 3.858 | 0.0905 |  |  | 1.320 |
| `h27f_strict_bsa_signv_sqrt_sparse035_guard120_steps120` |  | 120 | 10 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.035/0.13 | 0 | 1.150 | 6.989 | 3.881 | 0.0910 |  |  | 1.321 |
| `h27a_strict_bsa_signv_sqrt_sparse040_guard120_steps120_valid40` |  | 120 | 40 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/0.13 | 0 | 1.543 | 7.865 | 3.712 | 0.0871 |  |  | 1.724 |
| `h27b_strict_bsa_thetav_sqrt_sparse040_guard120_steps120_valid40` |  | 120 | 40 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/0.13 | 0 | 1.572 | 7.869 | 3.640 | 0.0854 |  |  | 1.751 |
| `h27d_strict_bsa_thetav_head_sparse040_guard120_steps120_valid40` |  | 120 | 40 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/0.13 | 0 | 1.614 | 7.921 | 3.713 | 0.0871 |  |  | 1.797 |
| `h27c_strict_bsa_signv_head_sparse040_guard120_steps120_valid40` |  | 120 | 40 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/0.13 | 0 | 1.643 | 8.571 | 3.693 | 0.0866 |  |  | 1.838 |
| `h27f_strict_bsa_signv_sqrt_sparse035_guard120_steps120_valid40` |  | 120 | 40 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.035/0.13 | 0 | 1.644 | 8.591 | 3.706 | 0.0869 |  |  | 1.840 |
| `h27e_strict_bsa_signv_active_sparse040_guard120_steps120_valid40` |  | 120 | 40 | strict_bsa_shiftmax | stage0_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | single_lr | 0.04/0.13 | 0 | 1.647 | 8.635 | 3.703 | 0.0869 |  |  | 1.844 |

### rapid_screen_h28_h29_h30_lr_binary_bsa_20260520_174434

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h28b_diff_lr_newfast_steps360` |  | 360 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.038 | 6.261 | 3.666 | 0.0860 |  |  | 1.185 |
| `h30b_strict_bsa_thresholdv_diff_lr_steps360` |  | 360 | 10 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.050 | 6.295 | 3.730 | 0.0875 |  |  | 1.201 |
| `h29b_diff_lr_binary_target_strong_steps360` |  | 360 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.055 | 6.583 | 3.706 | 0.0869 |  |  | 1.211 |
| `h29a_diff_lr_binary_target_mild_steps360` |  | 360 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.075 | 6.634 | 3.697 | 0.0867 |  |  | 1.231 |
| `h30a_strict_bsa_signv_diff_lr_steps360` |  | 360 | 10 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.082 | 6.526 | 3.803 | 0.0892 |  |  | 1.240 |
| `h28c_diff_lr_balanced_steps360` |  | 360 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.081 | 6.719 | 3.751 | 0.0880 |  |  | 1.241 |
| `h29a_diff_lr_binary_target_mild_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.086 | 6.354 | 3.902 | 0.0915 |  |  | 1.245 |
| `h28a_diff_lr_safe_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.085 | 6.670 | 3.814 | 0.0895 |  |  | 1.247 |
| `h28a_diff_lr_safe_steps360` |  | 360 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.095 | 6.508 | 3.807 | 0.0893 |  |  | 1.253 |
| `h30b_strict_bsa_thresholdv_diff_lr_steps120` |  | 120 | 10 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.101 | 6.833 | 3.779 | 0.0886 |  |  | 1.265 |
| `h28c_diff_lr_balanced_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.107 | 6.768 | 3.756 | 0.0881 |  |  | 1.268 |
| `h30a_strict_bsa_signv_diff_lr_steps120` |  | 120 | 10 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.120 | 6.730 | 3.809 | 0.0894 |  |  | 1.283 |
| `h28b_diff_lr_newfast_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.122 | 6.980 | 3.776 | 0.0886 |  |  | 1.289 |
| `h29b_diff_lr_binary_target_strong_steps120` |  | 120 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.143 | 6.943 | 3.836 | 0.0900 |  |  | 1.312 |
| `h30b_strict_bsa_thresholdv_diff_lr_steps120_valid40` |  | 120 | 40 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.535 | 7.633 | 3.595 | 0.0843 |  |  | 1.707 |
| `h29b_diff_lr_binary_target_strong_steps360_valid40` |  | 360 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.548 | 7.817 | 3.517 | 0.0825 |  |  | 1.721 |
| `h28b_diff_lr_newfast_steps360_valid40` |  | 360 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.557 | 7.548 | 3.499 | 0.0821 |  |  | 1.724 |
| `h28c_diff_lr_balanced_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.563 | 7.835 | 3.574 | 0.0838 |  |  | 1.739 |
| `h28b_diff_lr_newfast_steps120_valid40` |  | 120 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.580 | 7.848 | 3.560 | 0.0835 |  |  | 1.756 |
| `h30b_strict_bsa_thresholdv_diff_lr_steps360_valid40` |  | 360 | 40 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.589 | 7.821 | 3.562 | 0.0836 |  |  | 1.764 |
| `h29a_diff_lr_binary_target_mild_steps360_valid40` |  | 360 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.590 | 8.051 | 3.530 | 0.0828 |  |  | 1.768 |
| `h28c_diff_lr_balanced_steps360_valid40` |  | 360 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.589 | 8.186 | 3.559 | 0.0835 |  |  | 1.771 |

### rapid_screen_h31_sparse_lr_binary_bsa_20260520_192841

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h31c_h29b_lower_threshold_lr_steps360` |  | 360 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.035/1.8 | 0 | 1.056 | 6.399 | 3.762 | 0.0882 |  |  | 1.211 |
| `h31b_newfast_sparse028_bin045_steps360` |  | 360 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.028/2.0 | 0 | 1.064 | 6.316 | 3.698 | 0.0868 |  |  | 1.214 |
| `h31d_h29b_high_binary_eta_steps360` |  | 360 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.033/1.9 | 0 | 1.066 | 6.372 | 3.709 | 0.0870 |  |  | 1.218 |
| `h31e_strict_bsa_sparse030_bin055_steps360` |  | 360 | 10 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.03/1.9 | 0 | 1.074 | 6.822 | 3.674 | 0.0862 |  |  | 1.233 |
| `h31f_strict_bsa_sparse028_bin045_steps360` |  | 360 | 10 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.028/2.0 | 0 | 1.085 | 6.560 | 3.601 | 0.0845 |  |  | 1.237 |
| `h31a_newfast_sparse030_bin055_steps360` |  | 360 | 10 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.03/1.9 | 0 | 1.098 | 6.697 | 3.744 | 0.0878 |  |  | 1.257 |
| `h31e_strict_bsa_sparse030_bin055_steps360_valid40` |  | 360 | 40 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.03/1.9 | 0 | 1.559 | 7.858 | 3.492 | 0.0819 |  |  | 1.732 |
| `h31a_newfast_sparse030_bin055_steps360_valid40` |  | 360 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.03/1.9 | 0 | 1.576 | 7.977 | 3.539 | 0.0830 |  |  | 1.754 |
| `h31d_h29b_high_binary_eta_steps360_valid40` |  | 360 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.033/1.9 | 0 | 1.585 | 7.812 | 3.500 | 0.0821 |  |  | 1.758 |
| `h31f_strict_bsa_sparse028_bin045_steps360_valid40` |  | 360 | 40 | strict_bsa_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.028/2.0 | 0 | 1.608 | 7.918 | 3.376 | 0.0792 |  |  | 1.777 |
| `h31b_newfast_sparse028_bin045_steps360_valid40` |  | 360 | 40 | alpha_xnor_matrix_shiftmax | stage0_all_ffn_binary+stage1_half_even_ffn_binary+stage2_half_even_ffn_binary+stage3_block0_ffn_binary+downsample_stage0_stage2_binary | differential_lr | 0.028/2.0 | 0 | 1.622 | 7.990 | 3.490 | 0.0819 |  |  | 1.797 |

### rapid_screen_h37_main_batch_20260521_152904

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h37_binary_axnor_l1_neuronfast_steps360` | SOPs>3.35G | 360 | 10 | binary_alpha_xnor_matrix_l1 | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.072 | 6.349 | 3.542 | 0.0831 | 5.45 | 0 | 1.339 |
| `h37_strict_bsa_qkv_sqrt_signv_conservative_steps360` | SOPs>3.35G | 360 | 10 | strict_bsa_qkv_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.059 | 6.438 | 3.615 | 0.0848 | 6.87 | 0 | 1.368 |
| `h36_signed_consensus_shiftmax_conservative_steps360` | SOPs>3.35G | 360 | 10 | signed_consensus_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.019 | 6.174 | 3.718 | 0.0872 | 6.17 | 1 | 1.381 |
| `h36_strict_bsa_signv_conservative_steps360` | SOPs>3.35G | 360 | 10 | strict_bsa_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.087 | 6.458 | 3.679 | 0.0863 | 6.42 | 1 | 1.433 |
| `h36_strict_bsa_signv_conservative_steps120` | SOPs>3.35G | 120 | 10 | strict_bsa_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.072 | 6.482 | 3.740 | 0.0877 | 6.12 | 1 | 1.454 |
| `h37_binary_axnor_shiftmax_conservative_steps360` | SOPs>3.35G | 360 | 10 | binary_alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.071 | 6.514 | 3.816 | 0.0895 | 6.33 | 1 | 1.497 |
| `h37_binary_axnor_l1_conservative_steps360` | SOPs>3.35G | 360 | 10 | binary_alpha_xnor_matrix_l1 | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.080 | 6.714 | 3.816 | 0.0895 | 6.46 | 1 | 1.511 |
| `h37_binary_axnor_shiftmax_conservative_steps120` | SOPs>3.35G | 120 | 10 | binary_alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.086 | 6.301 | 3.826 | 0.0897 | 6.00 | 1 | 1.513 |
| `h37_a2os2a_qkv_signv_neuronfast_steps360` | SOPs>3.35G | 360 | 10 | a2os2a_qkv_l1 | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.037 | 6.415 | 3.905 | 0.0916 | 6.07 | 2 | 1.513 |
| `h36_signed_consensus_shiftmax_conservative_steps120` | SOPs>3.35G | 120 | 10 | signed_consensus_shiftmax | highsop_official | differential_lr | 0.035/1.8 | 0 | 1.040 | 6.045 | 3.952 | 0.0927 | 5.53 | 1 | 1.534 |
| `h37_a2os2a_qkv_signv_conservative_steps360` | SOPs>3.35G | 360 | 10 | a2os2a_qkv_l1 | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.075 | 6.387 | 4.084 | 0.0958 | 5.78 | 1 | 1.653 |
| `h37_a2os2a_qkv_signv_neuronfast_steps120` | SOPs>3.35G | 120 | 10 | a2os2a_qkv_l1 | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.056 | 6.143 | 4.205 | 0.0986 | 6.25 | 0 | 1.698 |
| `h37_a2os2a_qkv_signv_conservative_steps120` | SOPs>3.35G | 120 | 10 | a2os2a_qkv_l1 | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.042 | 6.276 | 4.256 | 0.0998 | 6.21 | 2 | 1.717 |
| `h37_strict_bsa_qkv_sqrt_signv_neuronfast_steps360` | SOPs>3.35G | 360 | 10 | strict_bsa_qkv_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.053 | 6.511 | 3.516 | 0.0825 | 33489.80 | 1 | 68.249 |
| `h37_binary_axnor_l1_neuronfast_steps120` | SOPs>3.35G | 120 | 10 | binary_alpha_xnor_matrix_l1 | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.067 | 6.524 | 3.730 | 0.0875 | 33489.80 | 1 | 68.384 |
| `h37_binary_axnor_l1_conservative_steps120` | SOPs>3.35G | 120 | 10 | binary_alpha_xnor_matrix_l1 | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.049 | 6.345 | 3.804 | 0.0892 | 33489.80 | 1 | 68.404 |
| `h37_strict_bsa_qkv_sqrt_signv_conservative_steps120` | SOPs>3.35G | 120 | 10 | strict_bsa_qkv_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.108 | 6.440 | 3.798 | 0.0891 | 33489.80 | 1 | 68.462 |
| `h37_binary_axnor_shiftmax_neuronfast_steps360` | SOPs>3.35G | 360 | 10 | binary_alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.034 | 5.878 | 3.630 | 0.0851 | 66979.60 | 1 | 135.256 |
| `h37_strict_bsa_qkv_sqrt_signv_neuronfast_steps120` | SOPs>3.35G | 120 | 10 | strict_bsa_qkv_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.054 | 6.138 | 3.646 | 0.0855 | 66979.60 | 1 | 135.292 |
| `h37_binary_axnor_shiftmax_neuronfast_steps120` | SOPs>3.35G | 120 | 10 | binary_alpha_xnor_matrix_shiftmax | stage02_highsop_official | differential_lr | 0.035/1.8 | 0 | 1.088 | 6.664 | 3.671 | 0.0861 | 66979.60 | 1 | 135.353 |

### rapid_screen_h40_confirm_360_20260521_230503

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p2_SNS012_F_steps360` | pass | 360 | 10 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.182 | 7.092 | 3.059 | 0.0718 | 5.01 | 1 | 1.270 |
| `h40_p2_SNS02_F_steps360` | pass | 360 | 10 | signed_consensus_shiftnorm | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.101 | 6.263 | 3.262 | 0.0765 | 5.70 | 1 | 1.271 |
| `h40_p2_SCS02_F_steps360` | pass | 360 | 10 | signed_consensus_shiftmax | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.091 | 6.185 | 3.314 | 0.0777 | 4.88 | 1 | 1.274 |
| `h40_p2_SCS012_F_steps360` | pass | 360 | 10 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.195 | 6.835 | 3.020 | 0.0709 | 5.30 | 1 | 1.276 |
| `h40_p2_SNS02_F_steps360_valid40` | AEE>1.58 | 360 | 40 | signed_consensus_shiftnorm | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.776 | 8.391 | 3.072 | 0.0721 | 5.70 | 1 | 2.152 |
| `h40_p2_SCS02_F_steps360_valid40` | AEE>1.58 | 360 | 40 | signed_consensus_shiftmax | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.767 | 8.642 | 3.134 | 0.0735 | 4.88 | 1 | 2.175 |
| `h40_p2_SNS012_F_steps360_valid40` | AEE>1.58 | 360 | 40 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.866 | 8.841 | 2.889 | 0.0678 | 5.01 | 1 | 2.369 |
| `h40_p2_SCS012_F_steps360_valid40` | AEE>1.58 | 360 | 40 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.879 | 9.099 | 2.842 | 0.0667 | 5.30 | 1 | 2.421 |
| `h40_p2_TXS012_F_steps360` | pos_neg_ratio>40.0 | 360 | 10 | ternary_alpha_xnor_shiftmax | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.184 | 7.088 | 3.089 | 0.0725 | 33489.80 | 1 | 68.212 |
| `h40_p2_HTS012_F_steps360` | pos_neg_ratio>40.0 | 360 | 10 | hamming_ternary_active_direct | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.276 | 7.709 | 3.041 | 0.0713 | 33489.80 | 1 | 68.358 |

### rapid_screen_h40_p2_N_20260521_225711

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p2_SCN_F_steps80` | SOPs>3.35G | 80 | 5 | signed_consensus_shiftmax | no FFN extra replacement | differential_lr | 0.05/2.0 | 0 | 0.869 | 6.248 | 3.897 | 0.0914 | 6.35 | 1 | 1.336 |
| `h40_p2_SLN_F_steps80` | SOPs>3.35G | 80 | 5 | signed_consensus_popcount_l1 | no FFN extra replacement | differential_lr | 0.05/2.0 | 0 | 0.887 | 6.992 | 3.888 | 0.0912 | 6.90 | 1 | 1.367 |
| `h40_p2_SNN_F_steps80` | SOPs>3.35G | 80 | 5 | signed_consensus_shiftnorm | no FFN extra replacement | differential_lr | 0.05/2.0 | 0 | 0.914 | 7.014 | 3.889 | 0.0912 | 7.53 | 1 | 1.395 |

### rapid_screen_h40_p2_N_20260522_011630

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p2_TXN_F_steps80` | SOPs>3.35G | 80 | 5 | ternary_alpha_xnor_shiftmax | no FFN extra replacement | differential_lr | 0.05/2.0 | 0 | 0.884 | 6.450 | 3.999 | 0.0938 | 6.24 | 1 | 1.416 |
| `h40_p2_CPN_F_steps80` | SOPs>3.35G | 80 | 5 | compat_qk_product | no FFN extra replacement | differential_lr | 0.05/2.0 | 0 | 1.029 | 7.587 | 3.782 | 0.0887 | 5.71 | 1 | 1.463 |
| `h40_p2_HTN_F_steps80` | SOPs>3.35G | 80 | 5 | hamming_ternary_active_direct | no FFN extra replacement | differential_lr | 0.05/2.0 | 0 | 0.944 | 7.202 | 3.832 | 0.0899 | 33489.80 | 1 | 68.337 |

### rapid_screen_h40_p2_S012_20260521_201531

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p2_TXS012_F_steps80` | pass | 80 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 0.982 | 7.041 | 2.921 | 0.0685 | 5.95 | 1 | 1.068 |
| `h40_p2_SNS012_F_steps80` | pass | 80 | 5 | signed_consensus_shiftnorm | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.037 | 6.893 | 2.984 | 0.0700 | 5.71 | 0 | 1.119 |
| `h40_p2_HTS012_F_steps80` | pass | 80 | 5 | hamming_ternary_active_direct | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.111 | 7.297 | 2.955 | 0.0693 | 4.67 | 1 | 1.203 |
| `h40_p2_CPS012_F_steps80` | AAE>7.9 | 80 | 5 | compat_qk_product | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.151 | 8.182 | 2.816 | 0.0660 | 4.16 | 1 | 1.328 |
| `h40_p2_SCS012_F_steps80` | pos_neg_ratio>40.0 | 80 | 5 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.009 | 6.514 | 3.115 | 0.0731 | 33489.80 | 1 | 68.031 |
| `h40_p2_SLS012_F_steps80` | pos_neg_ratio>40.0 | 80 | 5 | signed_consensus_popcount_l1 | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.091 | 7.493 | 2.992 | 0.0702 | 33489.80 | 1 | 68.128 |

### rapid_screen_h40_p2_S02_20260521_194431

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p2_SNS02_F_steps80` | pass | 80 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 0.963 | 7.061 | 3.231 | 0.0758 | 5.88 | 1 | 1.093 |
| `h40_p2_SCS02_F_steps80` | pass | 80 | 5 | signed_consensus_shiftmax | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 0.997 | 7.076 | 3.285 | 0.0771 | 5.87 | 1 | 1.194 |
| `h40_p2_SLS02_F_steps80` | SOPs>3.35G | 80 | 5 | signed_consensus_popcount_l1 | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.011 | 7.314 | 3.354 | 0.0787 | 5.52 | 1 | 1.234 |
| `h40_p2_CPS02_F_steps80` | AAE>7.9 | 80 | 5 | compat_qk_product | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.078 | 8.110 | 3.087 | 0.0724 | 4.15 | 1 | 1.248 |
| `h40_p2_HTS02_F_steps80` | AAE>7.9 | 80 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.107 | 8.069 | 3.145 | 0.0738 | 4.48 | 0 | 1.290 |
| `h40_p2_TXS02_F_steps80` | pos_neg_ratio>40.0 | 80 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 1.021 | 7.574 | 3.326 | 0.0780 | 66979.60 | 1 | 135.162 |

### rapid_screen_h40_p3_angular_20260522_012202

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p2_SNS02_F_steps80` | pass | 80 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0.2 | 0.977 | 7.011 | 3.274 | 0.0768 | 5.63 | 1 | 1.169 |

### rapid_screen_h40_p3_angular_v2_20260522_013500

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p3_SNS02_ang05_steps80` | pass | 80 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0.5 | 0.948 | 7.288 | 3.227 | 0.0757 | 5.68 | 1 | 1.083 |
| `h40_p3_TXS02_ang05_steps80` | pass | 80 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0.5 | 1.004 | 7.403 | 3.266 | 0.0766 | 5.93 | 1 | 1.204 |
| `h40_p3_HTS02_ang02_steps80` | pass | 80 | 5 | hamming_ternary_active_direct | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0.2 | 1.040 | 7.563 | 3.262 | 0.0765 | 4.85 | 1 | 1.242 |
| `h40_p3_SNS02_ang02_steps80` | pos_neg_ratio>40.0 | 80 | 5 | signed_consensus_shiftnorm | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0.2 | 1.050 | 7.528 | 3.299 | 0.0774 | 33489.80 | 1 | 68.202 |
| `h40_p3_TXS02_ang02_steps80` | pos_neg_ratio>40.0 | 80 | 5 | ternary_alpha_xnor_shiftmax | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0.2 | 0.961 | 7.082 | 3.279 | 0.0769 | 133959.19 | 1 | 269.035 |

### rapid_screen_h40_phase1_fast_20260521_183134

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_r1_HTS0_F_steps80` | SOPs>3.35G | 80 | 5 | hamming_ternary_active_direct | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.001 | 6.868 | 4.229 | 0.0992 | 16.62 | 0 | 1.675 |
| `h40_r1_CPS0_F_steps80` | SOPs>3.35G | 80 | 5 | compat_qk_product | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.028 | 7.018 | 4.281 | 0.1004 | 14.35 | 0 | 1.736 |
| `h40_r1_TXS0_F_steps80` | SOPs>3.35G | 80 | 5 | ternary_alpha_xnor_shiftmax | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 0.945 | 6.930 | 4.446 | 0.1043 | 11.10 | 0 | 1.747 |
| `h40_r1_SNS0_F_steps80` | SOPs>3.35G | 80 | 5 | signed_consensus_shiftnorm | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 0.951 | 6.542 | 4.509 | 0.1058 | 11.67 | 0 | 1.779 |
| `h40_r1_SCS0_F_steps80` | SOPs>3.35G | 80 | 5 | signed_consensus_shiftmax | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 0.960 | 6.462 | 4.510 | 0.1058 | 15.49 | 0 | 1.786 |
| `h40_r1_SLS0_F_steps80` | SOPs>3.35G | 80 | 5 | signed_consensus_popcount_l1 | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.051 | 6.829 | 4.363 | 0.1024 | 17.01 | 0 | 1.802 |
| `h40_r1_BSS0_F_steps80` | SOPs>3.35G | 80 | 5 | strict_bsa_shiftmax | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.176 | 7.532 | 5.116 | 0.1200 | 17.07 | 0 | 2.380 |
| `h40_r1_BQS0_F_steps80` | SOPs>3.35G | 80 | 5 | strict_bsa_qkv_shiftmax | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.014 | 7.130 | 5.432 | 0.1274 | 18.20 | 0 | 2.390 |
| `h40_r1_ADS0_F_steps80` | SOPs>3.35G | 80 | 5 | a2os2a_direct | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.097 | 7.448 | 5.295 | 0.1242 | 13.39 | 0 | 2.402 |
| `h40_r1_HBS0_F_steps80` | AAE>7.9 | 80 | 5 | hamming_binary_direct | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.299 | 8.513 | 4.804 | 0.1127 | 19.31 | 0 | 2.436 |
| `h40_r1_TLS0_F_steps80` | SOPs>3.35G | 80 | 5 | alpha_xnor_matrix_l1 | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.216 | 7.771 | 5.336 | 0.1252 | 9.19 | 0 | 2.593 |
| `h40_r1_AQS0_F_steps80` | SOPs>3.35G | 80 | 5 | a2os2a_qkv_l1 | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.184 | 7.448 | 5.558 | 0.1304 | 16.34 | 0 | 2.641 |

### rapid_screen_h40_phase2_b_20260521_193913

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_p2_SCS012_F_steps80` | pass | 80 | 5 | signed_consensus_shiftmax | s0_ffn+s1_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 0.977 | 6.849 | 2.977 | 0.0698 | 5.53 | 1 | 1.058 |
| `h40_p2_SCS02_F_steps80` | SOPs>3.35G | 80 | 5 | signed_consensus_shiftmax | s0_ffn+s2_half | differential_lr | 0.05/2.0 | 0 | 0.950 | 6.962 | 3.373 | 0.0791 | 5.66 | 1 | 1.171 |

### rapid_screen_h40_round1_120only_20260521_181356

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_r1_SCS012_F_steps120` | SOPs>3.35G | 120 | 10 | signed_consensus_shiftmax | stage0_ffn+stage1_half+stage2_highsop | differential_lr | 0.05/0.5 | 0 | 1.193 | 6.668 | 4.256 | 0.0998 | 16.42 | 0 | 1.878 |
| `h40_r1_SCS02_F_steps120` | SOPs>3.35G | 120 | 10 | signed_consensus_shiftmax | stage0_ffn+stage2_highsop | differential_lr | 0.05/0.5 | 0 | 1.171 | 6.550 | 4.356 | 0.1022 | 14.08 | 0 | 1.911 |
| `h40_r1_SCS0_F_steps120` | SOPs>3.35G | 120 | 10 | signed_consensus_shiftmax | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.099 | 6.221 | 4.565 | 0.1071 | 9.90 | 0 | 1.952 |
| `h40_r1_SCN_F_steps120` | SOPs>3.35G | 120 | 10 | signed_consensus_shiftmax | no FFN extra replacement | differential_lr | 0.05/0.5 | 0 | 1.067 | 6.357 | 5.105 | 0.1198 | 8.52 | 0 | 2.235 |

### rapid_screen_h40_round1_20260521_172610

| 实验 | gate | steps | samples | attention | FFN/范围 | LR策略 | target/maxTh | ang | AEE | AAE | SOPs(G) | firing | worstPN | zeroNeg | score |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `h40_r1_SCS012_F_steps120` | SOPs>3.35G | 120 | 10 | signed_consensus_shiftmax | stage0_ffn+stage1_half+stage2_highsop | differential_lr | 0.05/0.5 | 0 | 1.178 | 6.727 | 4.107 | 0.0964 | 11.02 | 0 | 1.779 |
| `h40_r1_SCS0_F_steps120` | SOPs>3.35G | 120 | 10 | signed_consensus_shiftmax | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.070 | 6.086 | 4.465 | 0.1047 | 10.65 | 0 | 1.861 |
| `h40_r1_SCS012_F_steps360` | SOPs>3.35G | 360 | 10 | signed_consensus_shiftmax | stage0_ffn+stage1_half+stage2_highsop | differential_lr | 0.05/0.5 | 0 | 1.232 | 7.043 | 4.197 | 0.0985 | 23.17 | 0 | 1.898 |
| `h40_r1_SCS02_F_steps120` | SOPs>3.35G | 120 | 10 | signed_consensus_shiftmax | stage0_ffn+stage2_highsop | differential_lr | 0.05/0.5 | 0 | 1.186 | 6.659 | 4.319 | 0.1013 | 9.90 | 0 | 1.907 |
| `h40_r1_SCS02_F_steps360` | SOPs>3.35G | 360 | 10 | signed_consensus_shiftmax | stage0_ffn+stage2_highsop | differential_lr | 0.05/0.5 | 0 | 1.204 | 6.496 | 4.332 | 0.1016 | 15.21 | 0 | 1.929 |
| `h40_r1_SNS0_F_steps120` | SOPs>3.35G | 120 | 10 | signed_consensus_shiftnorm | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.085 | 6.192 | 4.578 | 0.1074 | 10.75 | 0 | 1.945 |
| `h40_r1_SNS0_F_steps360` | SOPs>3.35G | 360 | 10 | signed_consensus_shiftnorm | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.099 | 6.226 | 4.581 | 0.1075 | 17.14 | 0 | 1.960 |
| `h40_r1_SCS0_F_steps360` | SOPs>3.35G | 360 | 10 | signed_consensus_shiftmax | stage0_ffn | differential_lr | 0.05/0.5 | 0 | 1.129 | 6.193 | 4.536 | 0.1064 | 14.41 | 0 | 1.964 |
| `h40_r1_SCN_F_steps360` | SOPs>3.35G | 360 | 10 | signed_consensus_shiftmax | no FFN extra replacement | differential_lr | 0.05/0.5 | 0 | 1.072 | 6.226 | 4.929 | 0.1156 | 14.52 | 0 | 2.135 |
| `h40_r1_SCN_F_steps120` | SOPs>3.35G | 120 | 10 | signed_consensus_shiftmax | no FFN extra replacement | differential_lr | 0.05/0.5 | 0 | 1.146 | 7.084 | 4.907 | 0.1151 | 16.62 | 0 | 2.218 |
