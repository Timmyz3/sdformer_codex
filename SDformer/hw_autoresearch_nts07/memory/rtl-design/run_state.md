# 2026-08-13 Q-silent exact cascade
- Added sidecar `qfit_local5_qsilent_score_leaf` with ENABLE_QSILENT default-off passthrough.
- ep29 100-group TCFM5 L1 324605->191424 (1.6957x), Acc32 270000 compare, 0 mismatch.
- Motion naive dual-core LAWS model ANT 0.936: not a contribution.
run_id:      rtl-design_20260718_typed_metadata_selective_residency
design_name: gatestack_typed_adaptive_single_context_execution
tool:        rtl-design+erie+iverilog+verilator_sva+yosys_lec
start_time:  2026-07-18T17:00:00Z
last_stage:  complete_c0_onchip_builder_segmented_walker_pass
overall:     PARTIAL_IMPLEMENTATION_NOT_SIGNED_OFF
report:      docs/110_TypedSlotMetadata与IPD选择性驻留架构闭环_20260718.md
latest_report: docs/115_完整C0片上Builder与分段精确Walker架构迭代_20260720.md
write_scope:
  - docs/99_GateStack公平基线与真实Trace_RTL实施规格_20260717.md
  - docs/110_TypedSlotMetadata与IPD选择性驻留架构闭环_20260718.md
  - memory/rtl-design/run_state.md
constraints:
  clk_mhz:   UNFROZEN
  area_um2:  UNFROZEN
  power_mw:  UNFROZEN
  pvt:       UNFROZEN
audit_scope:
  - Direct RAW41-only 同接口公平基线
  - IPD no-residency 同接口公平基线
  - head-major spill 同接口公平基线
  - H67 四 stage 真实 bit trace 文件与 TB 接入
  - SVA、覆盖率、综合消融矩阵
  - 第一批代码改动与文件所有权
explicit_non_claims:
  - 当前RAW41-only是同顶层运行路径，不是物理裁剪面积基线
  - 当前真实trace只覆盖一个样本、首block、首window
  - Adaptive CSR已完成提交期格式元数据和IPD-only驻留，但仍不等于FADC residency或在线编码决策
  - 未执行目标库DC、mapped SAIF、STA或mapped-netlist LEC/Formality
completed_scope:
  - IPD_no_residency_compile_variant
  - RAW41_only_same_top_runtime_path
  - real_H67_QK_gate_weight_bias_vector_generation
  - four_stage_three_path_icarus_verilator_sva_zero_mismatch
  - pow2safe_regression
  - fadc24_streaming_decoder_and_exact_raw_fallback_format
  - fadc24_leaf_random_backpressure_and_corrupt_bitmap_detection
  - fadc24_decoder_sva
  - fadc24_four_stage_real_trace_fulltop_zero_mismatch
  - raw_payload_6642_and_slot_capacity_6656_contract_split
  - adaptive_csr_header_steered_dual_decoder
  - adaptive_four_stage_single_configuration_zero_mismatch
  - adaptive_same_context_ipd_fadc_raw_zero_mismatch
  - adaptive_external_and_child_specific_sva_passed
  - physically_stripped_direct_raw_projection_top
  - direct_raw_tail_and_accumulator_zero_mismatch_tb
  - direct_ipd_adaptive_yosys_structure_ablation
  - adaptive_illegal_residency_failfast_superseded_by_typed_residency
  - adaptive_selector_control_sva
  - head_major_spill_transaction_scheduler
  - head_major_spill_directed_dual_sim_regression
  - commit_time_typed_slot_metadata
  - planner_static_unsupported_format_bounded_reject
  - atomic_route_format_cache_ownership_offset_checks
  - non_ipd_cache_fill_bypass
  - idempotent_cache_release
  - typed_residency_four_stage_and_mixed_context_dual_sim_sva
  - adaptive_residency_open_structure_lec
  - tag_qualified_cache_release_refill_protection
  - typed_trace_provenance_bundle
  - typed_format_metadata_accumulator
  - capacity_first_typed_format_policy
  - upstream_builder_error_raw_failsafe
  - typed_builder_frontend_307_head_random_boundary_regression
  - descriptor_cache_2006_operation_release_refill_stress
  - raw41_ipd32w_fadc24_payload_serializer
  - serializer_to_typed_slot_atomic_commit
  - four_real_case_word_exact_commit_inspect_replay_release
  - serializer_and_slot_sva_dual_sim_yosys_erie_pass
  - canonical_raw_term_bitmap_workspace
  - full_onchip_builder_automatic_policy_to_serializer_path
  - exact_ipd_fadc_raw_automatic_selection_word_exact
  - segmented16_destination_walker
  - linear_vs_segmented_dual_sim_sva_yosys_ablation
remaining_scope:
  - target_library_physically_stripped_baselines
  - full_bitexact_head_major_projection_baseline
  - expanded_real_trace_and_tail_latency
  - long_multicontext_release_refill_stress_and_coverage
  - explicit_class_and_segment_banked_bitmap_write_path
  - all_45_head_rtl_latency_ledger
  - streaming_commit_without_private_payload_copy
  - c1_dual_workspace_overlap
  - target_library_ppa
# 2026-07-20 BPB 收口

- 完成 FADC24 Bitmap-Preserving Bypass：大扇出 canonical bitmap 不再逐 token 展开后重建。
- Serializer 使用 8-bit/拍累计 popcount，并在 21-byte 输出末尾原子校验；否决 162-bit 组合 popcount 版本。
- 45-head 双模拟器/SVA：861 word 零失配，逻辑 destination 3226 不变，扫描握手 3226→2728。
- S3/H4：1561→1063 cycle；45-head 总和 14576→14078 cycle。
- 开放综合代理：Serializer 492→534 cells；完整 C0 3099→3181 cells；不是目标库 PPA。
- 下一阶段：实现两个 workspace、共享单 Serializer、严格有序提交的 C1 RTL。
# 2026-07-20 C1 双 Workspace RTL

- 新增 `gatestack_onchip_typed_builder_c1_top.sv`：两个 canonical workspace，共享单 Serializer/typed slot。
- sequence-ordered issue：45-head 顺序等待异常为 0，done sequence 逐项匹配。
- 真实 45-head stage-bounded：10035 cycle；C0 14078；加速 1.403x；模型误差 0.43%。
- 861 word 逐 word零失配；terms=762、destinations=3226、scan/BPB work=2728。
- Verilator+SVA 全规模 lint/elaboration 0 warning；全规模动态执行因工具性能超时，不标动态 PASS。
- Yosys check 0 problem；C1 5576 generic cells，仅作结构代理。
# 2026-07-20 转置分段 bitmap 端口化

- 新增可选 `gatestack_transposed_bitmap_bank.sv`，显式描述 token-major 写入、term-major 读取的在线转置端口。
- 隐式/显式存储与线性/16-bit walker 四组合均逐 destination PASS；C0/C1 45-head 计数与周期不变。
- 显式 11×32 个 4×16b tiny-bank 在模块级 Yosys 180 秒超时，未形成可用开放综合结果。
- 默认 `EXPLICIT_BITMAP_BANK_ENABLE=0`；显式版只保留为 macro/generator 端口候选，不进入 DC 主线。

# 2026-07-20 HATF96 三 Bank 权重合并器

- 新增逻辑 96-lane 到 3×32-lane 权重 bank 的 request fanout/response atomic join。
- 修复同拍多 bank 计数覆盖与跨事务 issued bitmap 清零优先级问题。
- 覆盖错峰、逆序、同拍、反压和身份错配；Icarus、Verilator+SVA、Yosys check 全部通过。
- 当前为叶模块，尚未接入同步 bias SRAM 修订后的完整 Builder→projection 顶层。

# 2026-07-20 Bias SRAM P0 关闭与完整回归

- projection bias 接口改为单 outstanding 同步 req/rsp；response 为OUT_TILE×ACC_W，错tag/token拒绝且重请求。
- 定向覆盖+200000/-300000、4拍response stall、身份错配和overflow quarantine。
- single-head到Builder完整真实S0-S3，以及Direct RAW、G1、32/64/162 token主线回归通过。
- 真实C0/C1双模式共466560个acc32比较，0 mismatch；ASIC sign-off仍受参数边界、macro、STA/SAIF和部署后处理阻塞。

# 2026-07-20 DCTF 叶模块

- 新增三消费者多读者有序term command队列，支持bank独立前进、consume mask、全局有序retire和atomic flush。
- Q=2/3/4各260条命令回归0 mismatch；Q=4 Verilator动态SVA、Yosys、Erie通过。
- Q2/Q4 synthetic周期402/389，开放Nangate45叶模块面积4123.798/5522.692；默认先用Q2。
- 尚未接真实weight/product/Acc，证据等级仅为rtl-leaf。

# 2026-07-20 BSF 精确Bias驻留终结器

- 新增参数化BSF：每个tag/output supertile只请求一次OUT_TILE×ACC_W bias，复用于162 token，默认关闭保持baseline。
- 单头覆盖错tag/token重试、32-bit大偏置、响应延迟和final反压；真实HATF96 S0-S3共233280元素0 mismatch。
- 专用runner通过Icarus、Verilator动态SVA、S3层级SVA lint、Yosys和Erie；Erie错误匹配已包含标准[ERROR]前缀。
- 开放逻辑面积代价+10.748%，因此RTL保留为条件候选，不冻结为默认实现。

# 2026-07-20 3xIndependent32结构基线

- 新增三套独立gatestack_multihead_decoder_projection_top的packed wrapper，固定每套OUT_TILE32、总96 product lane。
- 三路payload、weight、bias、final、done/error接口独立，未使用payload/term广播或中央weight join。
- 小规模Icarus、Verilator+SVA、Yosys、Erie均PASS；384元素exact，cross-talk为0，独立weight/final反压命中。
- 当前未接真实Builder/typed-slot三读口与S0-S3 trace，不能用48/49个小TB周期做论文性能结论。

# 2026-07-20 DCTF Term/Event Adapter

- 新增term/event adapter：完整term校验后按destination串化，输出cmd_sequence与term_issue_seq/first/last/head_last。
- 修复审阅发现的EVENT_WAYS硬编码4问题，增加EVENT_WAYS=2 Yosys elaboration检查。
- Fabric Q2/3/4扩展并逐bank保序携带term边界；原flush、full retire+accept和consume-mask语义保持。
- Adapter与fabric双模拟器/SVA/Yosys/Erie通过；尚未接bank executor。

# 2026-07-20 DCTF32 Bank Executor

- 新增单物理32-lane bank executor：首destination触发一次weight request，term内后续destination复用本地product寄存器，最后一次Acc握手才释放product并产生term_done。
- 固定物理tile映射为`3*logical_supertile+bank_id`，destination按token奇偶路由到两路Acc update，不存在中央96-lane product join。
- flush后epoch递增；同tag/channel/tile的旧epoch响应会被ready/drop并计数，不能进入product、Acc或term_done。ABA定向测试命中`stale_rsp=1`。
- Icarus、Verilator动态SVA、Yosys、Erie均通过；剩余约束是`EPOCH_W=4`有限回绕，系统必须保证迟到响应寿命小于16次flush，或在完整顶层增大epoch宽度。

# 2026-07-20 DCTF96三Bank Term Datapath

- 新增`gatestack_dctf96_term_datapath_top`，集成adapter、Q2 fabric与三套bank executor，保留三路weight和六路Acc物理接口。
- fabric entry新增logical_supertile sideband并通过跨term重叠、反压稳定和full retire+accept复用检查。
- 顶层拒绝input-channel和physical-tile溢出term，非法输入不进入adapter且不产生任何计算副作用。
- Icarus/Verilator结果为91/89周期，issued4、每bank completed3/weight4/update6/stale1，六parity通道各3次，0 mismatch。
- Yosys hierarchy/check/stat与Erie 0 error/warning通过；下一层必须接flushable Acc和bank-local bias/final。

# 2026-07-20 Banked Accumulator同步Flush

- `hitflow_banked_accumulator`新增显式同步flush；当拍屏蔽start/update/final/finish，下一拍清group、busy和valid状态，acc_mem本体不清。
- flush清除group-local overflow但保留五类性能累计计数器；同tag重启依赖valid bitmap隔离旧数据。
- 定向覆盖普通update中断、final反压中断、overflow恢复和旧final隔离；Icarus、Verilator动态SVA、Yosys、Erie通过。
- Central single/multihead与G1实例显式tie-off flush并回归PASS；DCTF完整顶层将连接真实flush。

# 2026-07-20 DCTF96完整Bank-Local Projection

- 新增`gatestack_dctf96_banklocal_projection_top.sv`，接入三套双bank Acc、三路同步bias和六路final。
- tile/head FSM、三Acc原子start/finish、每bank bias single-outstanding与epoch迟到隔离完成。
- 独立审阅修复source_done同term提前完成、wrong-current响应死锁、长flush epoch回绕、overflow后新tile和done payload反压稳定五类问题。
- 小规模Icarus/Verilator动态SVA/Yosys/Erie完整PASS；真实H67 S0-S3仍在进行。

# 2026-07-22 DCTF-2C RTL

- 新增`gatestack_dctf_term_event_adapter_2c.sv`，两个有序context在whole-term验证后提交，支持collect/emit重叠、原子错误丢弃和flush清空。
- `gatestack_dctf96_term_datapath_top`与`gatestack_dctf96_banklocal_projection_top`新增`ADAPTER_CONTEXTS`参数，默认1C回归不变，2C sideband随context保存。
- 2C adapter通过Icarus、Verilator动态SVA、Yosys两种EVENT_WAYS和Erie；1C term datapath与完整projection参数化集成后重新PASS。
- H67真实S0-S3 2C完整projection为764/718/5356/47072周期，233280个acc32零失配；S0 Verilator全顶层SVA与Icarus计数一致。
- 开放Nangate45映射0 process、无未映射非memory `$` 单元；logic area 181392.050、cells 134524、mem_v2 20，只作为结构代理。
- 当前RTL可作为DC候选输入，但目标SDC、SRAM替换、STA、SAIF、门级LEC和P&R尚未完成，状态仍为PARTIAL_IMPLEMENTATION_NOT_SIGNED_OFF。

# 2026-07-22 非法Metadata活性修复

- 独立DATE审稿发现非法term用ready=0拒绝会锁死标准ready/valid source。
- 顶层新增隔离drain状态：非法term正常握手，非零payload持续消费至event term-last，禁止进入adapter/fabric/weight/Acc。
- 合法term仍进入原1C/2C adapter；datapath idle显式排除illegal drain。
- 更新动态SVA覆盖非法握手隔离、非零term进入drain、drain接口状态和last-event释放。
- term datapath Icarus/Verilator/Yosys/Erie、完整projection及2C H67 S0-S3重新PASS；合法主路径53910周期和233280 acc32零失配不变。
- 首轮独立审阅的P1/P2整改已落地：新增不依赖内部fire信号的解锁SVA、new-error-wins优先级、零destination、drain中flush及2C合法context/非法drain重叠测试。
- 第二次独立复审保留的P1已整改：2C在途合法context经过非法drain后继续执行，逐项核对tag、issue sequence、token12/13/14、三路weight、九次Acc更新和三路term-done。
- 最终1C/2C四组回归为Icarus 95/126周期、Verilator+SVA 93/121周期，均mismatch=0；2C新增合法恢复term使issued=5、completed=4/4/4。
- 工具版本、输入SHA256、日志索引和中文结果包已更新；本项等待第三次独立复审，动态SVA仍不等于形式化liveness证明。
- 第三次独立复审PASS，非法Metadata整改P0/P1/P2 OPEN均为0；该结论只针对协议活性与隔离。

# 2026-07-22 PPDI双目的Bank Executor

- 新增sidecar PPDI executor，不修改标量DCTF基线；一条command携带一偶一奇两个可选目的。
- 两位exactly-once commit mask允许两Acc端口分拍握手，已完成端口不再发射，全部有效目的完成后command才retire。
- 两个连续command复用一次term product；部分提交时flush清状态并以epoch隔离迟到weight响应。
- Icarus/Verilator动态SVA均为71个非reset周期、5 command、5 weight、偶奇各4次提交、4 done、1 stale、0 mismatch；Yosys/Erie通过。
- 当前仅为rtl-leaf，等待独立审阅；尚无adapter/fabric、完整projection、真实H67周期或PPA结论。
- 首轮独立审阅为CONDITIONAL，提出stale撤回valid、child clear、partial Acc flush、epoch ABA四个P1及两个覆盖P2。
- stale现与Acc commit解耦；parent只在非clear拍采样child旧sticky，本地新错误保持new-error-wins。
- 新增pending-generation bitmap：只分配无未决请求的epoch，满表阻止新term，drain后恢复；3-bit满8项回归通过。
- 真实Banked Acc共同flush集成在Icarus/Verilator+双模块SVA通过，旧partial写在同tag/token重启final中不可见。
- 扩展回归为175周期、7 command、16 weight、Acc 6/6、5 done、2 stale；等待独立复审。
- 第二次复审指出pending generation仅按epoch清位；现为每个pending epoch保存tag/channel/tile，错误身份只drop+报错不清位，完整身份才释放。
- 最新叶回归为177周期、7 command、16 weight、Acc 6/6、5 done、3 stale；结果包改为相对路径并固化12份运行/构建日志哈希，等待第三次复审。

# 2026-07-29 Local5 TARE/Direct 同顶层消融

- `score_gate_term/window/linebuf` 新增同顶层 `USE_TARE` 参数，发布默认值为 Direct。
- TARE 与 Direct 的 MFEP 后端、刺激和 command 数相同；window4/window16/linebuf
  周期分别为 471/1896/1232 与 420/1662/1118。
- TARE 周期开销为 10.20%--14.08%；同一 Yosys `abc -fast` 下只减少 1.40%
  通用单元，面积归一吞吐代理为 0.889--0.920x。
- TARE 保留为待 fullres 活动率与 SAIF/DC 证明的精确低功耗候选，不是默认吞吐路径。
- 全 32 种 Local5 mask、空 mask 零 issue、`DEST_W=9/token449`、随机反压、
  TARE/Direct SVA、Icarus、Verilator 与四点 Yosys 均已通过。
- Shiftmax 端口扁平化、function 写法和 MFEP 常数界循环已完成开放综合可移植性整改。
- 当前仍未签核目标工艺 DC、STA、SAIF、LEC、SRAM macro 和 fullres T450 真实 trace。
- 主报告：`docs/178_Local5_TARE同顶层消融与双线第六轮RTL收口_20260729.md`。
# 2026-07-29 Local5 完成反压整改

- 独立 RTL 审阅发现 `stencil_done` 反压期间可能提前接收下一 anchor；已同时门控对外 ready 和对内 valid，并锁存 done tag。
- score-to-term TB 对每个 stencil 施加 1 至 3 拍完成反压；Direct/TARE Verilator、SVA、Icarus 和 lint 均通过。
- T450 edge/term/naive 性能计数从 16 bit 拓宽为参数化 17 bit。
- Local5 研究原型 RTL 可条件冻结；完整 DC handoff 仍缺 T450 Acc、filelist/SDC、SRAM/STA/SAIF/LEC。

# 2026-07-30 Local5 T450 全链与逐行清零

- `local5_banklocal_projection_top` 新增 `ST_CLEAR`，每拍清一个目的行，替代不可扩展的单周期 `MAX_DEST x OUT_DIM` 全阵列清零。
- `run_busy` 覆盖 CLEAR/RUN，清零期间命令反压，末行后进入 RUN；清零成本显式计入周期。
- 去除数组索引 `int'(...)`，bank-local 投影在 `MAX_DEST=450, DEST_W=9` 下通过 Yosys hierarchy/proc/check/stat。
- 新增 T450 全链 TB：目的449经过score、Shiftmax5、MFEP、bridge、projection后正确读回，目的0无混叠；Verilator/Icarus均为gate128、cmd32、cycle485。
- 当前 PERF 计数宽度为32位，不是17位；72000 term 边界已通过。
- 尚未完成 epoch-tag 惰性清零、真实 SRAM、DC/STA/SAIF/LEC 和 fullres ordered trace。

# 2026-07-30 Local5 T450 接口最终整改

- 顶层通过内部 `proj_acc_*` 对齐读口与 `ST_FINISH/run_done`，不再提前一拍暴露projection读回。
- projection对读目的/输出范围做fail-closed检查，非法读不握手、不返回数据并置协议错误。
- wrapper跟踪合法`w_load_last`；权重未完成时run_start保持IDLE并置错，避免永久反压。
- 独立Python golden驱动双窗口，6+3次输入、3600项Acc全部一致；完整27项依赖与向量hash已固化。
- 独立最终接口复审确认三项Closed、无新P0/P1；DATE评分仍2.3/5，下一步是fullres post-G0 ordered trace。
