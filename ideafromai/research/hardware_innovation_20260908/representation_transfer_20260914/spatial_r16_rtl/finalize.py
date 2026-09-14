from pathlib import Path
import json
H=Path(__file__).resolve().parent
raw=json.loads((H/'raw_verification.json').read_text());stream=json.loads((H/'stream_verification.json').read_text());direct=json.loads((H.parent/'spatial_r16_direct/SUMMARY.json').read_text());os=json.loads((H.parent/'spatial_r16_direct/OS_SUMMARY.json').read_text())
resource={
 'producer':{'alu32_carry_chains':8,'q1_carry_cut_bit':15,'signed19x13_multipliers':8,'external_weight_service_bits':256,'q1_vector_payload_bits':64,'q2_vector_payload_bits':104,'source_service_payload_bits':10,'z_vector_bits':256,'z_scalar_bank_bits':32,'psum_vector_bits':256,
 'storage_bytes':{'source':1920,'native_window':20,'source_masks':10,'q1_static':4608,'q2_static':7488,'q1_q2_static_live':144,'z_physical':1280,'z_hold':32,'psum':15360,'accumulator':32,'position_live':80,'rank_live':1,'q2_cache':312,'q2_block_live':3,'q2_remaining':3,'q1_pending':5,'q1_hold':8,'raw_result_hold':32,'origin':4},
 'storage_scope':'listed data/support registers; FSM, integer loop/address registers, counters and mux/priority/control combinational logic are additional, not area-free',
 'weight_port':'Q1/Q2 separate narrow banks, phase-exclusive enables/address and one 256-bit service permission; no simultaneous Q1/Q2 reads; no physical SRAM/area inference claimed',
 'z_port':'one address, Q1 read then write in separate states; ZSCAN full 8-bank vector; Q2 selected bank only; final support fully overwritten before use',
 'psum_port':'one common address for all 8 banks, read or write per cycle; first stripe overwrites all 480 rows before second stripe loads; drain after all 960 stores',
 'local_access':'16x10 source window supplies eight register-mux gathers; position metadata is distributed 80x8 bits with three selected-position lookups; qcache reads one 8x13 vector at selected term',
 'resident_model_lifecycle':'all 1152 Q1/Q2 vectors configured before first start; every configuration writes associated support; sequential no-reset different-source and repeat commands tested; no online weight versioning or overlapping commands'},
 'consumer':{'coefficient_bytes':768,'p_j_a_b_hold_bytes':128,'wide_hold_bytes':64,'output_hold_bytes':32,'signed32x32_multipliers':8,'wide64_carry_chains':8,'fp32_to_q20_lanes':8,'rne26_saturation_lanes':8,'identity_input_bits':256,'coefficient_read_bits':256,'raw_join_buffer_vectors':1,'identity_join_buffer_vectors':1,'scope':'same ordinary single-context consumer reused; producer and consumer arithmetic are separately instantiated; conversion shifts/increments, RNE increments, counters/control are additional'},
 'model_admission':{'q1_min':-127,'q1_max':127,'q2_min':-4095,'q2_max':4095,'z_signed15_bound':[-8562,7427],'p_any_prefix_abs_bound':504162009,'intermediate_rounding':False,'full_static_admission':'admission.json, fixed q1/q2 only; arbitrary signed8 q1 not accepted'},
 'service_convention':{'source_configuration_beats_per_tile':1536,'origin_configuration_beats_per_tile':1,'start_cycles_per_tile':1,'factor_configuration_beats_per_model':1152,'consumer_configuration_beats_per_model':24,'configuration_bus_bits':256,'cold_stream':'configure model once, then 64 tile input/origin/start and full output service','resident_stream':'same model already configured, input/origin/start per tile','per_tile_reset_cold':'a separate metric; all model config paid anew each tile, not the reported cold64'},
 'validation':'Verilator 4.028 cycle RTL only, no synthesis, area, timing, energy or PPA evidence; debug monitors and assertions are nonfunctional verification taps'}
(H/'resource_contract.json').write_text(json.dumps(resource,indent=2,ensure_ascii=False)+'\n')
summary={'passed':True,'raw_commands':raw['commands'],'stream_commands':stream['commands'],'unique_fixtures':142,'real_fixtures':135,'synthetic_fixtures':7,'raw':{},'complete_i24':{},'same_function_raw_comparison':{}}
for n in ['small','held','disjoint']:
 names=(H/f'{n}.txt').read_text().splitlines();summary['raw'][n]=[]
 for stall in [0,1]:
  rs=[json.loads(x) for x in (H/f'results_{n}_{stall}.jsonl').read_text().splitlines()][:len(names)]
  c=sum(r['cycles'] for r in rs);states=[sum(r['state_cycles'][i] for r in rs) for i in range(19)]
  costs={'z_clear':states[1],'native_source_load_setup':states[2]+states[3],'q1_gather_check_advance':states[4]+states[5]+states[10],'q1_weight_read':states[6],'q1_event_select':states[7],'q1_z_read_add':states[8]+states[9],'final_z_support_scan':states[11],'q2_weight_prefetch':states[12],'q2_position_load_store':states[13]+states[15],'q2_multiply_accumulate':states[14],'raw_drain_finish':sum(states[16:19])};assert sum(costs.values())==c
  summary['raw'][n].append(dict(stall=stall,tiles=len(names),core_cycles=c,cold_stream_service=c+1538*len(names)+1152,resident_stream_service=c+1538*len(names),state_costs=costs))
 summary['complete_i24'][n]=stream['sets'][n]['summaries']
 summary['same_function_raw_comparison'][n]={'primary_os':[r for r in os['summaries'] if r['set']==n],'retained_gustav_rmw':[r for r in direct['summaries'] if r['set']==n]}
(H/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Spatial R16：完整普通空间因子 RTL','',
'已经完成原生 96×4×4×T10 source → Q1 竖 3×1 → 精确 Z15 → Q2 横 1×3 → raw p32 → FP32 identity/J20/wide/I24。A 是普通空间低秩分解的完整迁移；双 P、最终 Z 支持、三 tap 权重复用与普通消费者为实现底座，不据此声称新颖 X。网络质量由根代理的同网实测报告承担，本目录不借旧浮点/R8 AEE。','',
'| 64 tile 集 | 直接 OS raw core | 因子 raw core | 直接 OS raw 冷服务 | 因子 raw 冷服务 | 因子完整 I24 冷服务 |','|---|---:|---:|---:|---:|---:|']
for n in ['held','disjoint']:
 d=next(r for r in os['summaries'] if r['set']==n and r['stall']==0);r=summary['raw'][n][0];s=summary['complete_i24'][n][0]
 lines.append(f"| {n} | {d['core_cycles']:,} | {r['core_cycles']:,} | {d['cold_stream_service']:,} | {r['cold_stream_service']:,} | {s['cold_stream_service']:,} |")
lines+=['','冷服务 = 每组首次模型配置 + 每 tile 的 1536 source 配置、1 原点配置、1 启动拍 + 全部计算/输出；权重配置不在 64 tile 内重复收费。直接展开每模型 10,368 拍、因子 1,152 拍；完整消费者另 24 拍。ready raw 的同函数冷服务分别下降 **10.9380% / 17.3764%**（core 11.0528% / 17.7297%）。主分母已转为 `../spatial_r16_direct/OS_SUMMARY.json` 的 output-stationary 同预算 bitmap 强臂，已独立 572 命令与完整逐状态验证；其输出逐值等于同一整数因子函数。旧 `SUMMARY.json` 的 Gustav event/psum RMW 臂仍保留，29.5488% / 38.4415% 只属于该旧臂，不能作为主增益。此处没有推测直接展开的完整 I24 周期。',
'', '资源并非等面积：双方执行资源合同为 8×32 ALU、单 256 bit 权重服务和同 p_mem；因子另实际使用 8 个 19×13 乘法，直接展开可用但闲置。直接展开 W32 为 331,776 B，因子静态 Q1/Q2 为 12,096 B；双方均有 1,280 B 八 bank 中间存储，直接 OS 实存 1,080 B 位图，因子实存 Z。因子另有 **Q2 cache 312 B、静态支持 144 B、最终位置支持 80 B**。完整资源和端口/寄存器边界见 `resource_contract.json`，无综合面积、时序或能耗结论。',
'', 'Q1 将相邻 x 两个 Z15 放在同一 32 bit bank 字，通过 bit15 carry cut 复用八条 ALU；每个合法输入先读取 16 字局部窗口，再由真实门字生成 2×4 个 T10 掩码。两个 R8 条带重读 source 的费用全部计入。Q2 在完整 Z 扫描之后使用最终支持，每个 N8 输出组实际预取 R8×3tap×N8 到 312 B cache；Z15 符号扩展到 19，再乘 signed13。stripe0 写 p_mem，stripe1 实读并累加；最后才向消费者按原 og/P/T/lane 顺序输出，任何阶段没有中间 RNE。',
'', '位宽仅对固定模型 admission：Z ∈ [−8562, 7427]，任意 Q2 前缀绝对界 504,162,009；因此 signed15 与 signed32 足够。`prepare.py` 对每个 fixture 重算原生门、Z、p，并与独立展开 W 卷积及导出 real gold 比对。全零/全一、随机、尾项、合法 q1 正负方向和 padding 污染均保留固定因子，未用截断伪造宽度支持。',
'', '验证：raw 与完整消费者各 **572 命令**，均含 15 小集、held128–191、disjoint4000–4063，ready/BP 和不 reset 连续两遍；总计 142 个独立 fixture（135 真实、7 合成）。每一路 raw/J/wide/I24 分别检查 2,196,480 值、Z 732,160 值；raw-only 再独立检查同量 raw/Z。所有数值、逐状态与资源计数、消费者成本恒等式、跨遍同值/同周期通过。BP 在实际 source/weight/raw/identity/output 服务上发生。报告见 `raw_verification.json`、`stream_verification.json`，逐命令 `results_*.jsonl`、`stream_*.jsonl`。',
'', '当前 ready Q2 乘加占 core 的 held 56.49%、disjoint 55.24%，每 tile 平均 18,353.8125 / 19,471.3125 个实际 scalar-Z×N8 服务；Q1双P分别 1,604.65625 / 2,110.28125。固定 source 装载阶段 3,264 拍，Q2 cache 装载 576 拍、最终支持扫描 80 拍、两条带位置装载/存储 1,920 拍，均未藏在 oracle 中。完整普通消费者 ready 每 tile 增加 2,425 拍。最值得另开有界接口的是连续 Q2 的三 tap 空间重用/变换，并同时收费输入变换、重建和新增位宽；本目录保持普通 A，新的函数/位宽必须使用自己的 gold。',
'', '复现入口：`run.py --stage all --prepare`；也可 `--stage raw` 或 `stream`。`spatial_core.sv` 为生产者，`spatial_stream.sv` 接复用的单上下文 `i24_consumer.sv` / `wide_phase_alu.sv`；后者仅追加 wide debug 端点，算术与原消费者一致。配置接口冻结于 PLAN，source origin.hex 是物理 source 窗口原点（output origin−1），不能再减一次。只在本目录写入，无生产、训练、量化调整、EDA 或 Git 提交。','']
(H/'README.md').write_text('\n'.join(lines))
print(json.dumps({'passed':True,'written':['SUMMARY.json','resource_contract.json','README.md']}))
