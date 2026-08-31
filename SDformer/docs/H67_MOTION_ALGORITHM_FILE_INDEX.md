# H67 Motion 主线算法文件索引（写论文用）

日期：2026-08-27。仓库根：`sdformer_codex/SDformer`。  
身份冻结：full-resolution **H67 ep35**，AEE **1.329678**（valid825）。  
H67 = 重训的 all-12-block **H60 Motion-XOR + Shiftmax gated-K**，不是公开 SDformerFlow 的 SDSA。

符号：`T_snn=10`，时间窗 `T_w=2`，空间窗 `15×15`，`N_tok=450`，`N_pair=225`。  
部署：Q7 score、Q8 LUT Shiftmax、Q1.7 gate、K 复用为 V。  
`binary_motion_xor_alpha=0.25`。checkpoint 在算法服务器上，本机 git 不强制带 `.pth`。

写论文只读本索引的「主线」列；「旁路」列不要写进 DATE 主模型。

---

## 0. 先读的身份合同（算法侧冻结）

| 文件 | 作用 |
|---|---|
| `neuron_autoresearch/H67_PAPER_IDENTITY_CONTRACT_20260813.md` | 论文身份一页纸 |
| `neuron_autoresearch/H67_PAPER_IDENTITY_CONTRACT_20260813.json` | 同上机器可读 |
| `neuron_autoresearch/H67_FULLRES_LINEAGE_RECEIPT_20260805.md` | fullres 血统 |
| `neuron_autoresearch/DATE_FINAL_MAINLINE_DECISION_20260812.md` | Motion vs Local5 主线裁决 |
| `neuron_autoresearch/DATE_FOUR_LINE_PAPER_FIT_20260817.md` | H67/H81/Local5/NB0 四线拟合 |
| `neuron_experiments/H9_bipolar_self_attention/entrypoints/generate_h67_paper_identity_contract.py` | 合同生成器 |

主 checkpoint 约定路径（本机可能没有权重文件）：

`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth`

---

## 1. 网络骨架（第三方主干，H67 不改这些文件）

H9 README 规定：**不修改** `third_party/SDformerFlow`，H67 用 overlay 挂上去。

| 路径 | 内容 |
|---|---|
| `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py` | `MS_SpikingformerFlowNet_en4` 顶层 |
| `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py` | 3D Swin encoder、attention 载体 |
| `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py` | 脉冲模块、BN、残差 |
| `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_submodules.py` | 子模块 |
| `third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py` | 网络装配 |
| `third_party/SDformerFlow/models/STSwinNet/PatchEmbed.py` | patch embed / 卷积前端 |
| `third_party/SDformerFlow/models/unet.py` | 多分辨率 U-Net 外壳 |
| `third_party/SDformerFlow/DSEC_dataloader/` | DSEC 数据 |
| `third_party/SDformerFlow/eval_DSEC_flow_SNN.py` | DSEC 评估入口（被 H9 train/eval 调用） |
| `third_party/SDformerFlow/eval_MV_flow_SNN.py` | MVSEC 评估 |
| `third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py` | 原版训练脚本（H9 `entrypoints/train.py` 打补丁后用） |
| `third_party/SDformerFlow/configs/` | 上游 yaml，H67 不用这些当身份 |

`src/` 下的 `sdformer/`、`sparse_ops/`、`external_inspirations/` 是另一套研究骨架，**不是** H67 训练路径。

---

## 2. H67 真正改动的算法代码（overlay）

根目录：`neuron_experiments/H9_bipolar_self_attention/overlay/`

训练时把该目录插到 `sys.path`，替换 `models.STSwinNet_SNN.*`。

### 2.1 主线必读（写方法节）

| 文件 | 论文对应 |
|---|---|
| `overlay/models/STSwinNet_SNN/bsa_attention.py` | **核心**。H60/H67 分数：TTX 重叠 + `16*motion` XOR 项、Shiftmax、gated-K、Q7 统计、`_binary_temporal_pair_stats`。`binary_motion_xor_alpha` 即 Motion-XOR。约 1000+ 行 |
| `overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py` | ATLIF+PSN 神经元；H67 部署是 **binary** `official_atlif`，输出 `{0,≈1}` |
| `overlay/models/STSwinNet_SNN/atlif_ternary_psn/installer.py` | 按 yaml 路径换成 ATLIF（Q/K `sn2_q` + `all_non_qk`） |
| `overlay/models/STSwinNet_SNN/atlif_ternary_psn/training.py` | 阈值训练/冻结（`threshold_freeze_after_step: 1224`） |
| `overlay/models/STSwinNet_SNN/h9_load_audit.py` | overlay 210 key、missing=0 的加载审计 |
| `overlay/models/STSwinNet_SNN/h9_losses.py` | 训练损失挂钩 |

`bsa_attention.py` 里和硬件 RQTB 对齐的公式注释：

- TTX：`(64*overlap + same_zero)/16`
- H67：TTX 分子 `+ 16*motion`（K 时域 XOR）

### 2.2 overlay 里有、但不是 H67 主模型（附录/消融才引用）

| 文件 | 为什么旁路 |
|---|---|
| `pattern_paft.py` | PAFT 微调，checkpoint 身份会变 |
| `near_match_residual_elision*.py` | 有损 near-match，已多次 NO-GO |
| `bounded_destination_group_pruning.py` | G11 静态 beta |
| `shared_fc1_patch_group_pruning.py` | 同上 |
| `simple_ternary_psn.py` / `simple_ternary_installer.py` | 早期 ternary，H67 是 binary |
| `h28_optimizer.py` / `h55_teacher.py` | 别的 H 实验 |

---

## 3. 冻结配置（yaml 即论文超参）

目录：`neuron_experiments/H9_bipolar_self_attention/configs/generated/`

### 3.1 必须引用

| 文件 | 用途 |
|---|---|
| `dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml` | **训练身份**。window `[2,15,15]`，`mode: h60`，`binary_motion_xor_alpha: 0.25`，ATLIF binary official |
| `dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml` | **部署/硬件序** Q7/Q1.7 |
| `dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_dyadic_q7q17_deploy.yml` | dyadic Shiftmax 对照 |

训练 yaml 关键字段：

- `model.name: MS_SpikingformerFlowNet_en4`
- `swin_depths: [2,2,6,2]`，heads `[3,6,12,24]`，`base_num_channels: 96`
- `spiking_neuron.num_steps: 10`，`neuron_type: psn`
- `atlif_ternary_psn.output_mode: binary`，`threshold_mode: official_atlif`
- `bsa_attention.mode: h60`，全部 12 个 block

### 3.2 同线但非 ep35 主锚

- `dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_seed{1,2}.yml` 多种子
- `dsec_fullres_w15_H67_ep35_score_qf{5,6,7,8}_gate_q17_sensitivity.yml` 分数位宽敏感性
- `dsec_fullres_w15_H67_crop_bb1e4_resume_ep30_*_deploy.yml` crop 血统
- `dsec_fullres_w15_NB0_equal_plus10_ep40.yml` 无 Motion 对照（NB0）

该目录有 1200+ yaml，大量是 Local5/H81/H82/rescue。写论文不要扫全目录，只用上表。

---

## 4. 训练 / 评估 / profile 入口

目录：`neuron_experiments/H9_bipolar_self_attention/entrypoints/`

### 4.1 主线

| 文件 | 用途 |
|---|---|
| `train.py` | H9 训练入口：装 overlay、seed、调上游 `train_flow_parallel_supervised_SNN.py` |
| `run_h9_standard_valid825_eval.py` | DSEC valid825（主表 AEE） |
| `run_h9_standard_mvsec_eval.py` | MVSEC 四序列 |
| `profile_nts11_hardware_p0.py` | 硬件 P0 profile（空 Q、K-zero、TTX 相等率） |
| `run_dsec_fullres_w15_equal_plus10_convergence.py` | equal+10 收敛（ep35 所在 run） |
| `run_dsec_fullres_w15_h67_bb1e4_resume{10,15,30}.py` | fullres 续训 |
| `run_date11_deploy_quant_eval.py` | 部署量化评估 |
| `generate_h67_paper_identity_contract.py` | 身份合同 |
| `generate_h67_fullres_lineage_receipt_20260805.py` | 血统收据 |
| `h67_bit_trace.py` / `run_h67_ep35_profile100_bit_trace_20260818.py` | INT8 Q/K bit trace |
| `capture_h67_full_network_binary_inputs.py` | 全网二值输入抓取 |
| `profile_sops.py` | SOP 统计 |

### 4.2 算法对照实验入口（可作消融，不是主模型）

`make_h67_motion_xor_ttx_config.py`，`audit_h67_h81_nomotion_result_20260812.py`，`audit_h67_h81_training_fairness_20260812.py`，MVSEC CICC 配置脚本。

---

## 5. 测试（方法节可引用「与硬件序对齐」）

`neuron_experiments/H9_bipolar_self_attention/tests/`

| 文件 | 覆盖 |
|---|---|
| `test_binary_temporal_pair_arch.py` | TTX/H67 时域对 |
| `test_bsa_attention.py` | Shiftmax / H60 |
| `test_atlif_ternary_psn.py` | ATLIF |
| `test_h67_bit_trace.py` | bit trace 身份 |
| `test_equal_plus10_config_provenance.py` | equal+10 配置血统 |
| `test_date_algorithm_closure_audit.py` | DATE 算法闭合 |
| `test_mvsec_direct_protocol.py` | MVSEC 协议 |
| `test_motion_t5_quotient_*.py` | T=5 商（**扩展实验，非 ep35**） |
| `test_motion_t4_pad_quotient_*.py` | T=4 pad（非 ep35） |
| `test_motion_sw12_overlap_*.py` | stride-12 窗（非 ep35） |

---

## 6. 算法文档（写相关工作/实验表）

### 6.1 必读

| 路径 | 内容 |
|---|---|
| `neuron_experiments/H9_bipolar_self_attention/README.md` | H9 = PSN+ATLIF+Shiftmax；引用 BSA NeurIPS'25 |
| `neuron_experiments/H9_bipolar_self_attention/docs/design.md` | overlay 设计 |
| `neuron_experiments/H_SERIES_SUMMARY.md` | H1–H9 神经元谱系 |
| `neuron_autoresearch/H67_H81_NOMOTION_RESULT_20260812.md` | 去 Motion 消融 |
| `neuron_autoresearch/H67_H81_TRAINING_FAIRNESS_20260812.md` | 与 H81 训练公平 |
| `neuron_autoresearch/H67_NB0_FULLRES_HEAD_TO_HEAD_20260805.json` | vs NB0 |
| `neuron_autoresearch/CLAUDE_DATE_EXPERIMENT_GAPS_20260818.md` | 算法实验缺口（P0 多项已完成） |
| `docs/H67_MOTION_ALGORITHM_FILE_INDEX.md` | 本索引 |

### 6.2 不要当成 H67 主贡献

- `D1_MOTION_T5_*`、`D2_MOTION_SW12_*`、`B2_MOTION_T4_PAD_*`：合同草案，未替换 ep35
- `DATE_NOVELTY_4_C8_H82/H86`：另一条 novelty，硬件未跟
- Local5 / H81 全套 yaml 与 `run_h66d_*`

---

## 7. 数据与 checkpoint（本机通常不全）

| 项 | 约定位置 | git |
|---|---|---|
| DSEC `saved_flow_data` | `data/Datasets/DSEC/saved_flow_data`（yaml 相对路径） | 不入库 |
| ep35 `.pth` | `.../results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth` | 不入库 |
| valid825 `spike_profile.json` | 同上 `standard_valid825/epoch35/` | 收据 JSON 在 `neuron_autoresearch/` |

写论文用合同里的 AEE/SHA，不必把 ckpt 拷进稿件仓库。

---

## 8. 方法节建议对应关系

| 论文模块 | 读这些文件 |
|---|---|
| 网络整体（Swin U-Net, T=10） | `Spiking_STSwinNet.py` + 训练 yaml `model`/`swin_transformer`/`spiking_neuron` |
| Patch embed / Conv | `PatchEmbed.py` + `Spiking_modules.py` |
| ATLIF 神经元 | `atlif_ternary_psn.py` + yaml `atlif_ternary_psn` |
| Motion-XOR 注意力分数 | `bsa_attention.py`：`_binary_temporal_pair_stats`、`binary_motion_xor_alpha` |
| Shiftmax / gated-K | `bsa_attention.py`：`install_shiftmax_attention`，yaml `bsa_attention.mode: h60` |
| 量化部署 | `*_hardware_order_q7q17_deploy.yml` |
| 任务指标 | `run_h9_standard_valid825_eval.py`、`run_h9_standard_mvsec_eval.py` |
| 与硬件对齐的统计 | `profile_nts11_hardware_p0.py` |

---

## 9. 一句话地图

```
third_party/SDformerFlow/models/STSwinNet_SNN/     # 主干，勿改
        ↑ overlay 安装
neuron_experiments/H9_bipolar_self_attention/
        overlay/.../bsa_attention.py               # H67 分数 + Shiftmax
        overlay/.../atlif_ternary_psn/             # ATLIF binary
        configs/generated/dsec_fullres_w15_H67_*   # 身份 yaml
        entrypoints/train.py + run_h9_standard_*   # 训评
neuron_autoresearch/H67_PAPER_IDENTITY_CONTRACT_*  # 冻结数字
```
