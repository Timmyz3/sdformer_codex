# S1 全候选筛查记分板

日期 2026-09-07。不选赢家。QK 普查身份=**ep35 路径 100 样本**，不是 ep34。

| ID | 最高阶 | 状态 | 关键数字 | 下一步 |
|---|---|---|---|---|
| A1 | S1 | SEALED_REPLAY | `{"c1_full_layer": {"path": "/home/zhumd/work/ideafromai/research/complete_transfer_20260907/c1_full_layer_r1.json", "n_points": 2, "first_two": [{"config": "artifact_default_256x16…` | Optional: dual-port parent SRAM model; do not call it a new mechanism until vs official run_fc same-resource |
| A2 | S1 | SEALED_REPLAY | `{"result": {"path": "/home/zhumd/work/ideafromai/research/c1_retained_parent_promotion_20260907/result_r1.json", "status": "EXPLORATORY_LOCAL_SERVICE_OPPORTUNITY_ONLY", "excerpt_ke…` | Only retry if layout/ports change; +1.58% add was the increment |
| A3 | S3_OLD_IDENTITY | SEALED_REPLAY | `{"m247_path": "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/results/m247_paft_vs_control_paired_valid825_r1_20260825/m247_paft_vs_control_paired_valid825_r1.json"…` | Must retrain on ep34; PAFT-ep4 running AEE ~1.47 already fails 1.259 floor |
| A4 | S1 | SEALED_REPLAY_EP34 | `{"path": "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/results/tsbg_ep34_same_io_b2_b4_b8_quickkill_r1_20260902/result.json", "checkpoint_sha_prefix": "4bbaf7fc",…` | CPU premodel GO at 1.15 cycle gate; RTL still separate. Under new contract this is retry-not-killed. |
| A5 | S1 | SEALED_REPLAY_EP34 | `{"path": "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/results/m1713_ep34_s2_fc_patch_zero_cost_upper_bound_fastkill_r1_20260901/result.json", "decision": {"fc1":…` | FC2 zero-cost UB<1.15 NO-GO; FC1/patch only if drop remaining work 39.5%/22.8% AND paired AEE |
| A6 | S4_FUNC | SEALED_RTL_FUNC | `{"rtl": {"path": "/home/zhumd/work/ideafromai/research/c2_temporal_shared_protocol_20260907/records/functional_r2/result.json", "status": "PASS_FUNCTIONAL_ONLY", "keys": ["status",…` | Wire bank-return + persistent Y; compare same-resource vs direct FTP |
| A7 | S1 | SEALED_REPLAY | `{"bank": "BANK_LOCAL_SINGLE_MODE_ARITHMETIC_OPPORTUNITY_ONLY"}` | Remap capture address==bank or stop claiming physical request equality |
| A8 | S4_FUNC | SEALED_RTL_FUNC | `{"rtl": {"path": "/home/zhumd/work/ideafromai/research/hardware_mechanisms_20260906/records/functional_r3/result.json", "status": "PASS_FIVE_CONFIGURATIONS", "keys": ["status", "re…` | Feed real BN/PSN intervals; measure restore rate |
| A9 | S0 | NO_RTL_NO_AEE | `{}` | Needs DSEC train; skip until AEE probe |
| B1 | S1 | NEW_QK_CENSUS_EP35 | `{"overlap_mean": 0.012507870370370364, "motion_xor_mean": 1.5262000000000007, "same_zero_mean": 30.62226481481481, "q_bit_density": 0.0172676731078905, "k_bit_density": 0.038265453…` | Need ep34 QK census before paper table; leaf object viable |
| B2 | S1 | NEW_QK_CENSUS_EP35 | `{"dirty_or_frac": 0.6089288244766505, "score_leaf_needed_frac": 0.43902834138486313, "ideal_skip": 0.5609716586151369, "per_stage": {"S0": {"tokens": 135000, "k_zero_frac": 0.74333…` | Row-level Shiftmax denom still unmeasured; do not treat token skip as row skip |
| B3 | S1 | NEW_QK_CENSUS_EP35 | `{"k_zero_both_t_frac": 0.5604312399355877, "k0_zero_frac": 0.682156843800322, "k1_zero_frac": 0.6957004830917874}` | Lossless if K-as-V holds; combine with B2 |
| B4 | S0 | NEEDS_OVERLAY_FINETUNE | `{}` | S2: one stage2 block swap, 10-frame AEE from ep34 |
| B5 | S0 | NEEDS_OVERLAY_FINETUNE | `{"tw": 2}` | T_w=2 may be too short; S2 after kernel swap |
| B6 | S0 | NEEDS_OVERLAY_FINETUNE | `{}` | Add depthwise beside SSA; A800 only after S2 |
| B7 | S0 | CONTRACT_CONFLICT_RISK | `{}` | Align with PSN/ATLIF BN before any train |
| B8 | S0 | NEEDS_OVERLAY_FINETUNE | `{}` | Plugin on ep34; S2 10-frame |
| B9 | S1 | MAPPED_FROM_QK_CENSUS | `{"three_pop_means": [0.012507870370370364, 30.62226481481481, 1.5262000000000007]}` | AND-PopCount is only overlap term; must add K_peer XOR |
| B10 | S1 | NEW_QK_CENSUS_EP35 | `{"both_qk_zero_t1_frac": 0.5569458937198067, "q_bit_density": 0.0172676731078905, "k_bit_density": 0.03826545390499195}` | Q already very sparse; dual-spike skip is almost Q-skip |
| C1 | S0 | NEEDS_ISLAND_SPEC | `{"t_attn": 2, "t_neuron": 10}` | Reuse C3 coverage RTL as rate converter |
| C2 | S0 | NEEDS_TRAIN | `{}` | AEE-sensitive; S2 with T=1/2/4 on subset |
| C3 | S0 | NEEDS_RETRAIN_FROM_SDFORMERFLOW | `{}` | Changes input identity |
| C4 | S1 | IDENTITY_LOCKED_AS_THETA_G | `{"z": "theta*g", "ep34_aee": 1.199514}` | Changing contract requires S3 AEE |
| D1 | S1 | SEALED_CPU_REF | `{"source_order": {"status": "EXPLORATORY_SOURCE_ORDER_IMPLEMENTATION_REFERENCE", "layers": [{"module": "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.fc2", "slots": {…` | Complete T FTP is the strong baseline; candidate must beat it same-resource |
| D2 | S1 | SEALED_CPU_REF | `{"fc2_signatures": "/home/zhumd/work/ideafromai/research/fusion_delivery_20260907/fc2_signatures_r1.json"}` | Keep as comparator, not auto title |
| D3 | S1 | PARTIAL_EP34_SHARDS | `{"shard_dir_glob": "results/m1681_ep34_decoder_d0_shard_*"}` | Need complete decoder Table-A before system claim |
| D4 | S0 | NEEDS_RETRAIN | `{"old_nm_audit": "FP32 weights have no exact zero blocks"}` | Retrain prune+4bit then S3 |
| D5 | S0 | ARCHITECTURE_OPTION | `{}` | Only after A1/D1/D3 share a parent-forest ISA |
| E1 | S0 | NEEDS_OVERLAY_FINETUNE | `{}` | S2 10-frame |
| E2 | S0 | NEEDS_OVERLAY_FINETUNE | `{"stage2_blocks": 6}` | Cheapest network swap |
| E3 | S0 | NEEDS_OVERLAY_FINETUNE | `{}` | Keep mul-free attention |
| E4 | S0 | NEEDS_RETRAIN_FROM_SDFORMERFLOW | `{}` | Geometry change |

## Attention census (ep35 QK, 100×12)

- tokens=3105000
- K 双时间全零 0.5604
- dirty(Q或K变化) 0.6089
- 仍需打分叶 0.4390 → 理想跳过 0.5610
- t1 上 Q且K 全零 0.5569
- Q bit 密度 0.01727  K bit 密度 0.03827
- overlap/motion/same_zero 均值 0.013 / 1.526 / 30.622

### per stage
{
  "S0": {
    "tokens": 135000,
    "k_zero_frac": 0.7433333333333333,
    "dirty_frac": 0.34115555555555555,
    "leaf_needed_frac": 0.2544666666666667
  },
  "S1": {
    "tokens": 270000,
    "k_zero_frac": 0.9518629629629629,
    "dirty_frac": 0.061174074074074075,
    "leaf_needed_frac": 0.048107407407407404
  },
  "S2": {
    "tokens": 1620000,
    "k_zero_frac": 0.6914024691358025,
    "dirty_frac": 0.5799308641975308,
    "leaf_needed_frac": 0.3082666666666667
  },
  "S3": {
    "tokens": 1080000,
    "k_zero_frac": 0.2432537037037037,
    "dirty_frac": 0.8228361111111111,
    "leaf_needed_frac": 0.7559712962962963
  }
}

## 新精度门对旧 PAFT

PAFT-ep4 running-BN 对照 AEE≈1.47 **已经高于 1.259**。要套 A3 必须在 **ep34** 上重训，不能引用 1.47 身份。

## TSBG ep34 CPU premodel
{
  "2": {
    "conservative_serialized_speedup": 1.5937566438434225,
    "roofline_speedup": 1.6781725316560163,
    "weight_fetch_ratio": 1.6678118472442658,
    "weight_byte_reduction": 0.4004119819317118,
    "cycle_gate_ge_1p15": true,
    "energy_branch_weight_reduction_ge_30pct": true
  },
  "4": {
    "conservative_serialized_speedup": 2.533808244015755,
    "roofline_speedup": 2.9035609271900737,
    "weight_fetch_ratio": 2.8733780344677022,
    "weight_byte_reduction": 0.6519775720408291,
    "cycle_gate_ge_1p15": true,
    "energy_branch_weight_reduction_ge_30pct": true
  },
  "8": {
    "conservative_serialized_speedup": 3.893370078010493,
    "roofline_speedup": 5.1217395129030105,
    "weight_fetch_ratio": 5.049365334698687,
    "weight_byte_reduction": 0.8019553084962759,
    "cycle_gate_ge_1p15": true,
    "energy_branch_weight_reduction_ge_30pct": true
  }
}
