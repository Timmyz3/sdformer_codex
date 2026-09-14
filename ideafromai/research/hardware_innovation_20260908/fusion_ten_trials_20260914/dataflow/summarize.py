"""Generate final concise tables from completed receipts; no simulation/model calls."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent;O=H.parents[1]/'r8_consumer_fusion_20260914'
def rows(d,n):return json.loads((H/d/n).read_text())
def cold(rs):return {r['mode']:r for r in rs if not r['stall'] and not r['command']}
D1=cold(rows('d1_forward','results_64.json'));D2=cold(rows('d2_halo','results_full.json'));D3=cold(rows('d3_interleave','results_full.json'));P3=cold(rows('d3_interleave','results_64.json'))
assert set(D3)=={0,1,2},'Final ordinary-RR full-frame receipt required'
audit=json.loads((H/'audit_results.json').read_text())
r=json.loads((O/'consumer_packed/resource_contract.json').read_text())
resource={k:r[k] for k in ['consumer_contract','consumer_common_resources']}
resource['consumer_contract']['same_function']='All D3 modes exactly the same frozen deployment function; no z or p RNE and no change to real IEEE32 identity conversion.'
resource.update(scope='two contexts, single shared producer arithmetic/weight array, complete native frame plus real FP32-to-I24 consumer',modes={'0':'sequential compute, next context may compute while first drains','1':'phase-offset: context1 starts when context0 Q1 complete','2':'ordinary both-ready round-robin; both start after two source tiles loaded'},authoritative_resources='RESOURCE_CONTRACT.md',shared_producer={'ALU32':8,'signed19_by_13_multipliers':8,'Q1_bytes':2592,'Q2_bytes':1536,'source_grants_per_cycle':1,'weight_grants_per_cycle':1,'z_grants_per_cycle':1,'psum_grants_per_cycle':1,'ALU_grants_per_cycle':1,'grant_rule':'disjoint resources can proceed together; overlap RR. ZADD/MAC require atomic z+ALU'},private_common={'contexts':2,'source_bytes_total':3840,'local_source_window_bits_each':160,'local_10bit_read_muxes_each':4,'z_bytes_total':1040,'psum_bytes_total':30720,'Q2_block_bytes_total':256,'other_state':'both context declarations in thread_context.sv, including metadata, holds, accumulators, pointers and counters'},static_beats_per_cold_command=1848,source_write_beats_per_tile=1536,origin_beats_per_tile=1,identity_and_output_beats_each_per_tile=480,consumer_has_separate_wide_arithmetic=True,total_cycles='window+launch+static+parameter_stalls+source_load+origin+source_load_stalls+1',window_cycles='consumer_cycles+tiles+floor(tiles/2)',no_cross_family_same_area_claim=True,no_EDA=True,full_frame_external_stall_or_warm=False)
(H/'d3_interleave/resource_contract.json').write_text(json.dumps(resource,indent=2)+'\n')
summary={'complete':True,'trials':{},'audit':audit,'not_combined':True}
summary['trials']['D1']={'strong_control_mode':1,'candidate_mode':2,'continuous_tiles':64,'control_cycles':D1[1]['total_cycles'],'candidate_cycles':D1[2]['total_cycles'],'saved_psum_writes':D1[1]['core_psum_writes'],'cycle_benefit':0,'conclusion':'common forwarding removes the claimed cycle delta; only p_mem write transactions removed'}
summary['trials']['D2']={'full_frame_tiles':19200,'control_cycles':D2[0]['total_cycles'],'candidate_cycles':D2[1]['total_cycles'],'saved_cycles':D2[0]['total_cycles']-D2[1]['total_cycles'],'saved_source_writes':D2[1]['retained_source_words'],'conclusion':'ordinary halo A; positive complete-frame service cycles'}
summary['trials']['D3']={'full_frame_tiles':19200,'sequential_cycles':D3[0]['total_cycles'],'phase_offset_cycles':D3[1]['total_cycles'],'ordinary_RR_cycles':D3[2]['total_cycles'],'phase_penalty_against_RR':D3[1]['total_cycles']-D3[2]['total_cycles'],'RR_saved_against_sequential':D3[0]['total_cycles']-D3[2]['total_cycles'],'conclusion':'ordinary both-ready RR is stronger than phase offset; generic interleave A, no phase-offset X'}
(H/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
for d in ['d1_forward','d2_halo','d3_interleave']:
 (H/d/'SUMMARY.json').write_text(json.dumps({'complete':True,'coverage':audit[d],'result':summary['trials'][{'d1_forward':'D1','d2_halo':'D2','d3_interleave':'D3'}[d]]},indent=2)+'\n')
labels=[('total_cycles','总周期'),('window_cycles','计算/消费窗口拍'),('static_words','静态配置词'),('source_load_words','source写词'),('external_source_words','图内source读词'),('origin_words','origin配置词'),('core_weight_words','Q1/Q2权重词'),('core_z_vector_reads','z向量读'),('core_z_writes','z向量写'),('core_z_scalar_reads','z标量读 / Q2MAC'),('core_psum_reads','psum读'),('core_psum_writes','psum写'),('shared_alu_grants','producer共享ALU grant'),('conflict_cycles','资源冲突/仲裁等待拍'),('both_compute_cycles','两个context均计算中的拍'),('core_output_stalls','producer合计等待输出拍'),('output_beats','I24输出向量')]
table='\n'.join('|'+label+'|'+'|'.join(str(D3[m][key]) for m in [0,1,2])+'|' for key,label in labels)
a=D3[0]['total_cycles'];b=D3[1]['total_cycles'];c=D3[2]['total_cycles'];n=audit['d3_interleave']['values_each_checkpoint']
(H/'d3_interleave/REPORT.md').write_text(f'''# D3：双context有限共享执行，普通RR胜过阶段错位

**最强对照改变了结论。** 完整19200tile单go：seq（已允许下一context与前一drain重叠）**{a:,}拍**，Q1→Q2阶段错位 **{b:,}拍**，普通双ready/RR **{c:,}拍**。阶段错位比普通RR慢 **{b-c:,}拍（{100*(b/c-1):.3f}%）**；不能用seq作唯一分母称阶段错位为X。普通RR相对seq省{a-c:,}拍（{100*(1-c/a):.3f}%），属于已有多context共享执行A。

三者同一个SV，两个完整tile source/z/psum/局部Q2状态，但只有top一套8×19×13乘法、8×32加法和Q1/Q2阵列。每context显式请求source/W/z/psum/ALU；有交集才RR，互不冲突可以同时推进，ZADD/BASE_MAC必须同时获得z+ALU。实际数据算术在top，child只输出操作数，获准后才能提交。I24后端另外一套8×32×32/8×64，不把它免费计作producer资源。详见[完整资源](RESOURCE_CONTRACT.md)。

mode0在context0计算完整480个rawp后释放context1，允许其计算与context0 drain/consumer重叠；mode1在context0完成全部Q1后释放；mode2两个source加载完同拍start，完全相同仲裁。source按序真实收费、两tile一batch、没有loader/compute重叠；一个共同消费者按tile顺序消费。每job首tile/静态加载、奇数尾tile、最后输出与done均包括。没有从两个独立实例的周期离线相加得到stream。

|全帧冷命令|seq mode0|阶段错位 mode1|普通RR mode2|
|---|---:|---:|---:|
{table}

全帧外部allow和输出ready均开放；表中的仲裁/consumer等待是真实内部费用。普通RR可能有更多冲突，却能更早释放两个任务并缩短整个窗口；减少冲突数不等价于减少总周期。三个模式source/W/z/psum/乘加总事务相同，收益来自时序重叠，不是操作数剪枝。两个core的cycles会含重叠及等待，不与consumer_cycles相加。

首64（128..191跨行）冷拍为{P3[0]['total_cycles']} / {P3[1]['total_cycles']} / {P3[2]['total_cycles']}，真实3tile（159..161）另覆盖跨行及奇数尾。16fixture包含8真实块、零/一/padding poison/正负极值/FP转换边界/tie/saturation；这部分每命令单tile，双context碰撞由3/64/full真实连续数据覆盖。小块和3/64都运行两种外部背压及无reset二次命令。最终192+12+12+3=219命令，rawp、实际J20、I24各 **{n:,}值** 全绿。三个整帧仿真墙时依次{D3[0]['job_wall_seconds']:.2f}/{D3[1]['job_wall_seconds']:.2f}/{D3[2]['job_wall_seconds']:.2f}s，不是硬件Fmax。

补普通RR只改变start条件和mode编码，原mode0/1的小块、3、64全部整数计数与补齐前一致；原mode0/1完整帧收据保留，mode2补独立完整帧。`prior_two_mode/`仅来源回归，不重复计入219命令。作者audit核对全grant/工作/周期总账和回归；root独立读代码审阅另列，作者审计不自称独立review；见[root独立审核](../../REVIEW_DATAFLOW.md)。

此布局没有证明新调度X，但完成了真实共享执行强A底座。未跑全帧背压/warm、滚动任意context加载、EDA/PPA或D1/D2/root其他融合组合。两tile状态比单tile多，禁止跨核同面积比较。本项停止于固定三个公平调度，不扫描队列/上下文容量。
''')
(H/'d3_interleave/README.md').write_text('''# D3有限双context共享执行

[REPORT.md](REPORT.md)给最终三臂结论；[RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md)说明唯一共享producer算术/权重和双方双tile状态。最强普通RR对照胜过阶段错位，保留负差分。独立审阅见[root REVIEW_DATAFLOW](../../REVIEW_DATAFLOW.md)。

```bash
/opt/anaconda3/bin/python3.12 prepare.py
/opt/anaconda3/bin/python3.12 run.py
/opt/anaconda3/bin/python3.12 run_stream.py --count 3 --first 159
/opt/anaconda3/bin/python3.12 run_stream.py --count 64 --first 128
/opt/anaconda3/bin/python3.12 run_stream.py --full
```

Verilator4.028，严格-Wall、--cc --exe后make。run_stream默认三模式；`--modes 2 --append`可在已有两个完整收据后仅补mode2，不重复其它模式。SV与C++为唯一执行源，不依赖bootstrap生成脚本。父目录`audit.py`/`summarize.py`分别检查和汇总最终收据；需要旧冻结data中真实NPY/NPZ，按脚本路径读取。
''')
(H/'REPORT.md').write_text(f'''# 三项数据流/消费者接口的完整RTL筛选

全部使用同一固定Q1/Q2和真实FP32 identity→J20→I24出口，C96/K864/R8/N96/T10；未改数值函数、训练或生产树。三项各有实际SV/TB与最强共同资源控制，不把stall/重启或第三调度控制另计新点。

|本项|实际范围|强控制→候选|结论|
|---|---|---|---|
|[D1完整rawp旁路](d1_forward/REPORT.md)|8真+8边界，64连续|968747→968747拍|共同STORE+forward消除周期差；候选少30720个p_mem写词。|
|[D2两列halo环形复用](d2_halo/REPORT.md)|8真+8边界，64与19200连续|324217349→309563909拍|全帧省14653440源装入拍，4.5196%；普通halo A。|
|[D3有限双context](d3_interleave/REPORT.md)|8真+8边界，3/64与19200连续|普通RR {c}→阶段错位 {b}拍|普通RR更强；阶段错位失败。generic interleave胜seq，但不称X。|

各项内部同状态/端口/输出backpressure权限。D3真实外置共享producer八ALU/乘法和单Q1/Q2阵列，I24仍独立宽算术；两tile存储比D1/D2多，不能跨项宣称同面积。三项没有在同一模块叠加，所以不能把百分比相乘或许诺组合全帧收益。

所有checkpoints为真实rawp、SV实际转换J、最终I24；gold不用于决定动态门、地址或调度。静态冷配一次、新源/origin/identity每次实计，padding由RTL产生。背压、warm同参数重启、tile身份/末输出都在小块及连续测试；整帧是单go无外部stall。完整收据与作者审计见[SUMMARY.json](SUMMARY.json)、[audit_results.json](audit_results.json)。固定数值函数已有旧stage真实825任务评价，本轮不重跑质量，也不声称与原浮点消费者逐位无损。

本页是作者结果汇总；独立实现审阅见[root审核](../REVIEW_DATAFLOW.md)。没有EDA、Fmax/面积/能耗数据，仿真墙时只说明实际可执行范围。三项均在当前固定工作点收口。
''')
(H/'README.md').write_text('''# fusion_ten_trials：dataflow独占三项

[总结果](REPORT.md) · [机器可读汇总](SUMMARY.json) · [原始假说/公平控制](PLAN.md) · [root独立实现审核](../REVIEW_DATAFLOW.md)

独立三个目录各有可编译SV/C++、准备脚本与实际JSON收据；各README给重现命令。先完成对应run，再在此目录运行`/opt/anaconda3/bin/python3.12 audit.py`与`/opt/anaconda3/bin/python3.12 summarize.py`。不运行一次性源编辑脚本；它们已移除。build与可由prepare重新产生的fixture binary按本地.gitignore忽略，最终SV/JSON/CSV/文档保留。
''')
print(json.dumps(summary['trials'],indent=2))
