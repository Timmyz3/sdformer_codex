run_id:      architecture_20260718_typed_slot_selective_residency
design_name: H67_GateStack_typed_adaptive_selective_residency
tool:        architecture_skill+rtl_evidence_audit+open_structure_ablation
start_time:  2026-07-18T17:00:00Z
last_stage:  complete_c0_onchip_builder_segmented_walker_pass
overall:     CONDITIONAL_ARCHITECTURE_FREEZE_SIGNOFF_NOT_ACHIEVED
recommended_candidate: C_RUNTIME_TYPED_ADAPTIVE_IPD_RESIDENT_FADC_RAW_COLD_EXACT
fallback_candidate: C0_single_context_gatestack_v1
deferred_candidate: C1_dual_context_until_cold_overlap_is_proven
previous_run_id: architecture_20260715_gatestack
artifact: docs/110_TypedSlotMetadata与IPD选择性驻留架构闭环_20260718.md
review: docs/111_DATE独立复审与TypedResidency第四轮整改_20260718.md
latest_artifact: docs/115_完整C0片上Builder与分段精确Walker架构迭代_20260720.md
latest_review: docs/113_DATE独立审稿人第五轮_Builder前端后评估_20260718.md
completed_scope:
  - paper_system_boundary_frozen
  - must_should_nice_requirements_defined
  - conservative_balanced_aggressive_candidates_defined
  - fair_baseline_and_ablation_contract_defined
  - real_trace_contract_defined
  - memory_throughput_ppa_targets_defined
  - risks_and_elimination_gates_defined
  - real_qk_gate_four_stage_first_window_rtl_ablation_passed
  - residency_demoted_to_energy_auxiliary
  - fadc24_real_trace_roundtrip_and_profile100_bounds_completed
  - fadc24_streaming_decoder_leaf_rtl_and_assertions_passed
  - fadc24_four_stage_real_trace_fulltop_icarus_verilator_passed
  - fadc24_s3_terms_reduced_30960_to_12888
  - fadc24_s3_speedup_1p527x_vs_ipd_no_residency
  - raw_payload_bits_and_slot_capacity_bits_split
  - adaptive_csr_runtime_header_select_rtl_passed
  - adaptive_csr_four_stage_single_configuration_zero_mismatch
  - adaptive_csr_mixed_ipd_fadc_raw_single_context_passed
  - adaptive_trace_bundle_1p407x_vs_gatestack
  - adaptive_leaf_yosys_generic_cells_1496
  - physically_stripped_direct_raw_projection_slice_passed
  - direct_ipd_adaptive_generic_cells_1293_2188_3183
  - temporary_adaptive_residency_failfast_superseded
  - adaptive_selector_invariant_sva_passed
  - head_major_spill_transaction_scheduler_passed
  - head_major_real_trace_lower_bound_completed
  - commit_time_raw_ipd_fadc_slot_format_metadata
  - replay_plan_format_and_atomic_route_ownership_offset_contract
  - ipd_only_descriptor_cache_lookup_fill_and_release
  - fadc_raw_noncacheable_exact_replay
  - mixed_ipd_fadc_raw_residency_real_trace_zero_mismatch
  - typed_residency_trace_bundle_195149_cycles
  - adaptive_open_structure_ablation_4191_4958_5249_cells
  - adaptive_residency_open_lec_4832_of_4832
  - tag_qualified_stale_release_protection_and_sva
  - typed_trace_provenance_hashes_and_tool_versions
  - independent_date_rereview_3p2_weak_reject_borderline
  - capacity_first_ipd_fadc_rescue_raw_policy_frontend
  - onchip_term_fanout_metadata_accumulator
  - builder_error_and_counter_overflow_raw_failsafe
  - exact_four_stage_runtime_policy_decisions
  - typed_builder_frontend_dual_sim_sva_yosys_erie_pass
  - long_descriptor_cache_release_refill_shadow_stress
  - independent_date_fifth_review_3p3_weak_reject
  - raw41_ipd32w_fadc24_payload_serializer_word_exact
  - typed_slot_atomic_commit_inspect_replay_release
  - serializer_slot_dual_sim_sva_yosys_erie_pass
  - complete_final_gate_k_to_typed_slot_c0_rtl
  - automatic_ipd_fadc_raw_policy_and_payload_path
  - segmented16_exact_destination_walker
  - linear_scan_cycles_10832_to_segmented_941
blocking_signoff_gates:
  - target_node_library_pvt_missing
  - absolute_area_budget_missing
  - absolute_power_budget_missing
  - real_100frame_allblock_bit_trace_missing
  - full_bitexact_head_major_baseline_missing
  - target_library_physically_stripped_baselines_missing
  - expanded_real_trace_and_fadc24_tail_distribution_missing
  - explicit_banked_bitmap_write_microarchitecture_missing
  - all_45_head_rtl_latency_ledger_missing
  - c1_dual_workspace_overlap_missing
  - private_payload_buffer_copy_not_eliminated
  - valid825_projection_deployment_contract_missing
  - mapped_sram_saif_sta_lec_missing
  - full_encoder_throughput_and_data_movement_not_closed
write_scope:
  - docs/98_H67_GateStack_DATE补强架构签核规格_20260717.md
  - docs/101_H67真实四Stage消融与GateStack架构再冻结_20260717.md
  - docs/103_FADC24流式Decoder与四Stage同顶层RTL迭代_20260718.md
  - docs/105_统一AdaptiveCSR运行时双格式架构与RTL验证_20260718.md
  - docs/107_PhysicallyStripped_Direct_RAW41投影基线_20260718.md
  - docs/108_AdaptiveCSR配置合同与SelectorSVA整改_20260718.md
  - docs/109_HeadMajor_PSUM_Spill公平下界与架构决策_20260718.md
  - docs/110_TypedSlotMetadata与IPD选择性驻留架构闭环_20260718.md
  - docs/111_DATE独立复审与TypedResidency第四轮整改_20260718.md
  - memory/architecture/run_state.md
# 2026-07-20 BPB 架构决策

- BPB 晋级为 C0 默认路径：只对 `FADC24 && fanout>21` 生效，保持 IPD/list/RAW 语义。
- 45-head RTL 总周期降低 3.42%，p99/max 降低 31.90%，完整 C0 开放结构代理增加 2.65%。
- C1 stage-bounded 模型更新为 9992 cycle、1.409x；仍需 RTL 复现。
- 论文定位：BPB 是三格式驻留数据流的 representation-preserving co-design，不单独冒充系统架构贡献。
# 2026-07-20 C1 模型到 RTL 晋级

- C1 stage-bounded 模型 9992 cycle，真实 RTL 10035 cycle，偏差 43 cycle（0.43%）。
- C1 相对 BPB-C0 实测减少 28.72%，1.403x；该收益已从 `[模型]` 晋级 `[rtl]`。
- 开放结构代理 C0→C1 为 3181→5576 cells（+75.29%）；必须用 SRAM macro/DC/SAIF 判断是否主推。
- 论文写法：sequence-ordered dual-workspace/shared-backend pipeline；不能把普通双缓冲单独包装成创新。
# 2026-07-20 bitmap 物理端口候选淘汰

- H67 canonical bitmap 的访问本质冻结为 token-major 32-lane 写、term-major 162-token 读的在线 bit-matrix transpose。
- 352 tiny-bank 全展开候选功能 bit-exact，但开放综合 180 秒超时，不能作为可交 DC 主线。
- 默认回退已签过开放综合的隐式 bitmap；后续只评估 lane 分组或后台转置，并在完整 projection slice 下比较 EDP。

# 2026-07-20 HATF96 架构晋级

- 完整 projection 的 Builder 占比仅 7.17%，C1 系统加速仅 1.021x，因此主优化对象转向 projection backend。
- HATF96 真实四 stage RTL 周期相对 32 lane 为 2.611x；开放 Nangate45 逻辑面积归一吞吐为 1.180x，当前四宽度中最佳。
- 96 lane 冻结为 3×32 物理权重 bank；独立 ready/valid、错峰返回与原子 join 叶模块双模拟器/SVA/Yosys 通过。
- 默认研究候选更新为 C0+BPB+HATF96；目标库 SRAM/STA/SAIF/完整集成未完成，仍不具备架构签核。

# 2026-07-20 同步 Bias SRAM 后重评

- bias 已改为带 tag/output_tile/token 的 request 与带 tag/token、OUT_TILE×ACC_W 的同步 response，关闭原 P0 接口缺口。
- 新合同下真实 C0/C1 四 stage 为 211303/207213 cycle，Builder 占比 6.68%，C1 系统加速 1.020x。
- HATF32/64/96/128 为 211303/113546/80223/64410 cycle；HATF96 相对32为2.634x。
- 开放 Nangate45 逻辑面积归一吞吐为1.000/1.142/1.164/1.133；HATF96仍是DSE候选，不是物理最优签核。
- 独立DATE复审仍为Weak Reject/Borderline前；下一项只保留DCTF分布式窄命令fabric与BSF bias-stationary finalizer。

# 2026-07-20 BSF 条件降级

- BSF在HATF96真实S0-S3上将总周期80223降至75197，降低6.265%，bias请求2430降至15，逐元素0 mismatch。
- 同顶层开放Nangate45逻辑代理面积204042.748增至225972.852（+10.748%），面积归一吞吐仅0.963x。
- 当前flop-based BSF不能列为面积效率贡献；仅在bank-local SRAM/register-file和SAIF证明总能量下降至少10%或projection EDP改善至少15%后晋级。
- 论文当前只能报告exact transaction reduction，不能声称功耗或EDP下降。

# 2026-07-20 等并行度96-Lane结构基线

- 实现真实3xIndependent32 wrapper：三套独立replay/decoder/weight/product/Acc，只共享clock/reset；小规模三事务384元素0 mismatch、0串扰。
- 同一Nangate45开放逻辑映射中，HATF96-Central/3xIndependent32 area为203921.452/270665.640，Central减少24.659%；cells减少22.319%。
- 该结果只证明共享前端的logic-area潜力；未映射mem_v2为3/9且宽度不同，不能推导memory面积或总PPA。
- 真实H67 S0-S3三读口slot wall-time、六个统一Acc SRAM、STA/SAIF仍是主贡献晋级门槛。

# 2026-07-20 DCTF真实Term语义前端

- 新增完整term验证后发射adapter，默认缓存1458bit，阻止重复/越界/元数据错误term产生partial command。
- DCTF entry扩展term_issue_seq/first/last/head_last，dispatch-retire与bank compute-complete明确分离。
- Adapter+Q2 fabric开放逻辑代理合计10206.420，adapter另有1个未映射mem_v2；真实周期、memory和后端未计。
- 前端仍是rtl-leaf；只有接入三路bank-local weight/product/Acc并过真实S0-S3后才晋级架构原型。

# 2026-07-20 DCTF单Bank计算边界

- DCTF32 executor已把窄command接到真实weight request、32-lane product和两路Acc update接口，term内product只生成一次并按destination复用。
- `fabric retire`继续只表示三bank已接收command；`term_done`严格绑定最后destination的Acc update握手，关闭了把dispatch误报为compute complete的架构风险。
- late response使用epoch隔离并通过同身份ABA定向验证；默认4-bit epoch需要系统级迟到寿命合同。
- 当前仍是单bank叶模块，不能据此声称DCTF96系统加速、能耗或PPA收益。

# 2026-07-20 DCTF96三Bank中间顶层

- adapter、Q2 fabric和三套32-lane executor已集成，暴露三路独立weight和六路Acc update；不存在中央96-lane weight/product join。
- 修复跨term supertile串扰风险：logical_supertile现随每条command驻留于fabric entry；不同supertile重叠TB通过。
- 非法channel/supertile在adapter前精确拒绝，零issued/weight/Acc副作用；三bank ABA旧epoch响应各drop一次。
- `head_compute_done`绑定head-last term三个bank的最后Acc握手；当前最后command retire可与最慢bank compute同拍，尚非时间解耦dispatch pipeline。
- 证据晋级为rtl-integration；三个Acc、bias/final、真实S0-S3和目标PPA仍缺失。

# 2026-07-20 DCTF96完整Projection后端

- DCTF已从中间term datapath推进到三套真实Acc、三路bias和六路final的完整projection后端。
- 共享边界仍为窄term command，weight/product/Acc/bias/final保持bank-local；不存在中央768-bit weight join。
- 完整生命周期与flush/epoch恢复已通过定向动态SVA，证据仍为rtl-integration。
- 下一判定点是相同term/event输入边界下Central96、3xIndependent32与DCTF96的真实H67 S0-S3 wall-time和访问分账。

# 2026-07-22 DCTF96真实H67四Stage回放

- H67 sample0/window0 S0-S3完整projection-only回放为822/718/5652/55072周期，总62264周期。
- 四stage共5010逻辑term、15030次物理32-lane weight访问、7290次bias请求和233280个acc32逐元素检查，0 mismatch。
- S1真实K全零，仍完成972次bias请求和31104个final检查，证明zero-work head/tile生命周期没有被错误跳过。
- 该结果冻结DCTF工作量，但Central96/Independent32同边界结果未完成前不计算speedup或EDP。

# 2026-07-20 Projection Acc位宽冻结

- 使用H67真实S0-S3 gate/K与dyadic INT8 weight/bias重算124416个最终元素，四stage与expected_output_acc32均0 mismatch。
- S3实际final/中间部分和最大绝对值为55035/55015；真实激活绝对和界85572。
- DIM768、gate511、全K激活、所有INT8权重幅值128的配置级上界为50233425，只需27-bit signed，int32仍有42.75x裕量。
- 正常路径不采用整tile final quarantine；overflow定位为非法配置/损坏防护，成功tile完成仍必须禁止在overflow时发生。

# 2026-07-22 DCTF-2C架构晋级

- 同一term/event到projection-final边界下，Central96、3xIndependent32、DCTF96-1C、DCTF96-2C真实四stage周期分别为59853/59945/62264/53910。
- 1C负结果确认前端相序串行是S3瓶颈；2C whole-term原子提交将collect/validate与前一term emit重叠，S3周期降低14.526%，相对Central实测1.110x。
- 2C命令发射保持全局顺序，input-channel base与logical supertile按context驻留，malformed term在任何command可见前原子丢弃。
- 等边界开放Nangate45逻辑面积为189141.960/224987.588/182119.826/181392.050；2C面积归一吞吐代理为1.158x，但另有3138 bit架构状态且mem_v2未计面积。
- 架构贡献冻结为bank-local distributed command execution、whole-term atomic dual-context commit和H67稀疏term软硬件协同；普通双缓冲不能单独作为创新。
- 证据仍受单sample/window、行为存储、无SDC/STA/SAIF/SRAM宏/目标DC和full-encoder闭环限制，整体状态仍非架构签核。
- 最新文档：docs/142_等边界四架构对照与DCTF2C原子双上下文迭代_20260722.md。

# 2026-07-22 PPDI候选

- 发现DCTF标量command使每bank偶/奇两路Acc每拍只使用一条；同term product可安全广播给一个偶token和一个奇token。
- 真实sample0/window0逐term奇偶分布显示command work由23175降至16160，理论降低30.270%；S3由21616降至14952，降低30.829%。
- profile100含7101034个G1 term、40560225个destination，平均fanout 5.712；无奇偶约束M2降低44.308%仅作乐观参照。
- PPDI定义为whole-term原子提交后的奇偶分区、双destination窄命令和三bank双Acc同拍更新；与2C的term间重叠正交。
- 当前仅为profile候选，未实现RTL、周期或PPA，不进入已完成贡献；晋级合同见docs/144_PPDI奇偶配对目的令牌架构候选与RTL晋级合同_20260722.md。

# 2026-07-22 非法Metadata活性整改

- DCTF96前端非法term从“拒绝握手”改为“握手后隔离drain”，不改变合法数据流或架构性能主张。
- 首轮独立审阅的两个P1和三个P2均已形成RTL/SVA/TB整改与四组1C/2C回归证据，正在等待独立复审。
- 此整改只关闭接口正确性风险，不提升DATE架构新颖性，也不关闭PPA、多trace或系统边界缺口。
- 第二次独立复审保留的2C恢复P1已补充完整合法结果核对与四路径回归，等待第三次独立复审确认关闭。
- 第三次独立复审PASS，非法Metadata协议整改关闭；该项不增加架构新颖性。

# 2026-07-22 PPDI Executor晋级为RTL叶模块

- PPDI从profile候选推进到单bank executor rtl-leaf；仍未成为完整架构贡献。
- 采用两位exactly-once commit mask支持偶/奇Acc端口独立反压，避免valid依赖ready和快端口重复累加；command保持整体retire。
- 叶模块双模拟器/SVA/Yosys/Erie通过，真实H67的30.270% command-work仍不能等价为周期收益。
- 下一门槛是独立审阅通过后实现parity-partition adapter与dual-destination ordered fabric。
