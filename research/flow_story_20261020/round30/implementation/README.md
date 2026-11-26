# R30：普通联合模式恢复的实现

本轮只实现普通 Q4/Q8/mixed 部署与恢复，不是光流新评分器。新核心由冻结 R28 core `f4e4dae1…fe9a` 派生；旧文件不变。独立4bit尺度、RNE等普通量化竞争解释继续保留。

`mixed_integer_core.py` 保持完整 T10/B/空间/通道和原 current BN/PSN。量化范围仍为24个MLP FC和4个patch residual 3×3卷积。M模式的低4个位平面仅对 `true_tail ∪ selected_complete_tiles` 做真实整数矩阵乘；其他完整tile的低plane没有执行。高4planes覆盖所有行，一次整数累加后按原output-channel尺度缩放。negative floor truncation、signplane−128和原bias规则不变。

外部 mask 为16组的CPU bool[H,W]，12个MLP的FC1/FC2共用同一对象，4个卷积各用输出tile mask。只能选完整9×9，真尾不能被表示为所选tile而是强制Q8。STE input-gradient使用同一high8/low4有效权重；卷积按输出grad域分组反传并汇合输入halo，weight-gradient仍为原identity STE。dense整数参考按相同行mask直接乘有效整数权重。没有完整Q8计算后再清零的伪跳过。

外部 `flow_r30_mask_contract_v1` 必需字段：`seed=20261019`、`tile_size=9`、`tail_mode=Q8`、原`members_sha256`、`training_selection=sha256_seed_step_group_row_col_topk_v1`；`geometries.full9/crop_center9`各包含16组的`H/W/Q8_complete_tiles`；`evaluation_tiles`给出full9的固定有序tile坐标。配额由硬件方/root冻结，本实现不选k，合法支持零配额。训练M的hash输入为紧凑JSON `[seed,step,group,row,col]`，按SHA及坐标升序取固定quota，不读取GT或crop内容。两几何真实尺寸由原图下采样关系核验。

正式训练沿原256成员/两遍512顺序和crop，原W271482起点、固定原生teacher、同477允许参数、AdamW和监督/蒸馏loss。模式为 `['Q4','Q4','Q8','M'][(visit-1+sequence_rank+2*(pass-1))%4]`，各256/128/128次；每序列每遍16/8/8，且每身份恰好一次Q4加一次其他模式。每步只有一次学生和一次教师前向，没有额外多图loss。无dev择权、无smoke续接。

正式模型前向 **1289**：初始M64 + 学生512 + 教师512 + 最终三模式各(3路径parity+64开发预测)。旧原W的初始Q4/Q8只引用已审来源，新增前向为零；CPU另核新旧uniform整数数学逐字节一致。末同W的Q4/Q8/M分别保存64个head3原数组和全4head SHA，GT仅哈希不读数值。全部78 BN与28实际affine调用逐前向记录；三路径的整数/PSN中间值在CPU内存逐字节比较，四head原数组各模式单独落盘。

唯一smoke使用首个训练crop强制M做一次更新，教师1+学生1+完整分辨率M的STE/deploy/dense三路径=**5**前向，不计算开发质量。smoke与formal各自重新原W和新optimizer，分别20/60分钟硬alarm、共用原GPU0锁；root另加外层进程时限。此前吞吐不能代替本轮实测，训练秒数/峰值显存会实际记录。

CPU入口：

```bash
CUDA_VISIBLE_DEVICES='' /opt/anaconda3/envs/pytorch310_cpu/bin/python -B \
  test_mixed_cpu.py /fresh/path/cpu_checks.json
```

运行入口由root部署，所有路径显式传入，默认仅CPU preflight：

```bash
python -B run_joint_recovery.py \
  --r28-root R28_ROOT --r26-root R26_ROOT --members ORIGINAL_R27_MEMBERS \
  --mask-contract EXTERNAL_MASK_JSON --mask-contract-sha256 MASK_SHA \
  --runner-sha256 RUNNER_SHA --reference-contract INITIAL_REFERENCE_JSON \
  --output FRESH_OUTPUT --preflight-output FRESH_PREFLIGHT --wall-minutes 60
```

正式加 `--execute`；smoke改为 `--smoke --wall-minutes 20`（可不传reference-contract），使用独立输出目录。`INITIAL_REFERENCE_JSON` schema为`flow_r30_original_initial_references_v1`，states.initial_Q4/initial_Q8各含旧result/audit的path和SHA；只加载旧metadata，不读旧GT数值或质量。

结果schema `flow_r30_joint_recovery_v1`，graph `flow_r30_joint_Q4_Q8_M_all12mlp_patch4_currentBN_v1`。正式成功状态 `JOINT512_AND256_DEVELOPMENT_PREDICTIONS_COMPLETE`；评估键为 `initial_M/endpoint_Q4/endpoint_Q8/endpoint_M`。`final_parity[Q4|Q8|M]`含`training_deployment/dense_integer/heads_file/heads_sha256`。初/末权重、实际teacher恒定性、全部训练ledger、每次M的坐标和mask SHA、每模式首步与最后一步完整kernel记录保留。此实现不包含硬件成本模型或质量门计算。
