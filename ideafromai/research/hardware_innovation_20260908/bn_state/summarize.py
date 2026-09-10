"""Derive compact final tables and Gram production limits from completed runs."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
j=json.loads((ROOT/'final_comparison.json').read_text())
rows=[]
for l in j['layers']:
    m=l['meta']; d=np.load(ROOT/f"trace_s{m['sample']}_stage{m['stage']}.npz")
    r=next(r for r in l['rows'] if r['axis']=='gram')
    pairs=int(np.triu(d['G']).sum(dtype=np.uint64))
    contraction=(m['C']**2+2*m['C'])*m['H']
    rows.append(dict(sample=m['sample'],stage=m['stage'],
        exact_row_cooccurrence_counter_increments=pairs,
        eight_increment_lane_ideal_lower_bound_beats=(pairs+7)//8,
        lower_bound_scope='Only a row-sparse producer issuing at most eight scalar counter increments per beat, before extraction/ports/read-modify-write. Not a measured producer; pattern aggregation is outside this bound.',
        implemented_bit_intersections=r['counters']['gram_pair_popcounts'],
        implemented_counter_service_beats=r['busy']['gram'],
        full_source_statistics_ready=r['statistics_all_ready'],
        online_W_contraction_macs=contraction,
        online_W_contraction_96_lane_ideal_lower_bound=(contraction+95)//96,
        ideal_contraction_scope='Mathematical service lower bound for this explicitly implemented online contraction; it is not a simulated schedule or a bound on every possible algebraic alternative.',
        strongest_baseline_component_service=l['strong_baseline_beats'],
        state_G_bytes=m['C']*(m['C']+1)//2*4,
        state_v_bytes=m['C']*16*8,state_tau_bytes=m['H']*m['T']*8,
        charged_G_reads=r['bytes']['gram_contraction_bank_reads'],
        charged_v_write_read=r['bytes']['gram_v_write_read'],
        charged_tau_write=r['bytes']['gram_threshold_write'],
        charged_tau_read=r['bytes']['gram_threshold_read'],
        timeline=r['execution_milestones']))
(ROOT/'gram_cost_analysis.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
external_names={'source_retention_write','source_backing_read','weight_backing_read',
                'wide_external_write','wide_external_read','record_external_write','record_external_read',
                'gram_source_read','gram_static_parameter_read','static_parameter_read'}
resource_rows=[]
for l in j['layers']:
    for r in l['rows']:
        resource_rows.append(dict(sample=l['meta']['sample'],stage=l['meta']['stage'],axis=r['axis'],
            external_read_write_bytes=sum(v for name,v in r['bytes'].items() if name in external_names),
            external_busy_beats=r['busy']['external'],
            external_summed_queue_wait_beats=r['queued_wait']['external'],
            numeric_busy_beats=r['busy']['numeric'],numeric_summed_queue_wait_beats=r['queued_wait']['numeric'],
            first_h_block_statistics_ready=r['statistics_first_ready'],
            last_h_block_statistics_ready=r['statistics_all_ready'],
            first_confirmed_output=r['first_confirmed_output'],
            shared_state_live_peak_bytes=r['state_live_peak_bytes'],
            external_mutable_live_peak_bytes=r['external_mutable_peak_bytes'],
            wide_work_live_max_bytes=r['wide_work_live_max_bytes']))
(ROOT/'resource_summary.json').write_text(json.dumps(dict(rows=resource_rows,
    waiting_definition='Queue wait is summed over all requests; overlapping waits must not be added to end-to-end service time. Statistics readiness is separately reported for the first and last h blocks, each using complete N.',
    capacity_definition='Shared state excludes the explicitly provisioned common weight/source caches, wide-work/PSN registers, moments and metadata arrays.'),indent=2)+'\n')

text='''本轮四轴比较已完成。Gram 在两个 S0 样本中保留明确的组件服务收益，S3 因在线权重收缩而失败；A8 的收益不跨样本稳定。固定 S0 BN 的完整 valid825 算法对照已由主线完成，因此这里的正结果只支持“必须保留动态统计”这一受限条件下的后续研究，本轮不进入 RTL。

以下数字是有限资源、真实支持驱动的 **Float64 模型服务拍**，不是 RTL 周期、频率、ASIC PPA 或整网加速比。四条路线都完成每个 h 的整个 N=T×P 统计域，再交付最终 θg；普通通道分块、源驻留、权重广播和源窗口顺序已经纳入对照。保存路线同时考察存 Y 与将满秩 PSN 前移后存 U，最终基线取存 Y、存 U、保源重算中最快者。

范围是 `layers.{0,3}.swin_blocks.0.mlp.fc1` 两个实际模块。S0 为 T=10、P=19,200、C=96、H=384、N=192,000；S3 为 T=10、P=300、C=768、H=3,072、N=3,000。每点包含该模块完整统计域，并不覆盖该 stage 的其他块。sample0 是 `zurich_city_09_a_0001`，sample10 是 `interlaken_01_a_0000`；二者都是已有硬件捕获，无本轮重训或重新运行网络。

| 样本 / stage | 存 Y | 存 U | 保源重算 | A8 B32/K4 | θ 加权 Gram | 最强普通基线 |
|---|---:|---:|---:|---:|---:|---|
'''
for l in j['layers']:
    r={r['axis']:r for r in l['rows']}; m=l['meta']
    text+=f"| {m['sample']} / S{m['stage']} | "+' | '.join(f"{r[a]['beats']/1e6:.6f} M" for a in ('save_y','save_u','recompute','packet','gram'))+f" | {l['strong_baseline_axis']} |\n"
text+='''
| 样本 / stage | A8 服务时间变化 | Gram 服务时间变化 | A8 必要 h-tile 恢复数 / 实际重算放大 |
|---|---:|---:|---:|
'''
for l in j['layers']:
    a=next(r for r in l['rows'] if r['axis']=='packet'); g=next(r for r in l['rows'] if r['axis']=='gram'); m=l['meta']
    text+=f"| {m['sample']} / S{m['stage']} | {(a['relative_service_time_to_strong_baseline']-1)*100:+.3f}% | {(g['relative_service_time_to_strong_baseline']-1)*100:+.3f}% | {a['counters']['required_failed_h_tiles']} / {a['counters']['repair_channel_amplification']:.3f}× |\n"
text+='''
S0 两点逐点下降率的均值：Gram 31.711%，A8 为 −3.063%（即平均增加 3.063%）。按两个组件总服务时间加权，Gram 下降 31.711%，A8 增加 3.067%。它们不是网络层间加权或 valid825 的性能结果。S3 必须相对更快的存 U 基线比较，不能用较弱的保源重算制造优势。

sample0 的带宽、首个通道组统计等待与峰值状态如下，另一个样本及请求队列累计等待详见 `resource_summary.json`。MB 为十进制，KiB 为二进制；共享状态峰值不包含下文另外列出的公共缓存和工作寄存器。首组统计就绪与首个最终输出不同，不能把所有 h 组的最后统计就绪时间当作全域初始停顿。

| stage / 轴 | 外部读写 MB | 首组统计就绪 M拍 | 首个最终输出 M拍 | 共享状态峰值 KiB | 外部可变状态峰值 MB |
|---|---:|---:|---:|---:|---:|
'''
for r in resource_rows:
    if r['sample']!=0: continue
    text+=f"| S{r['stage']} / {r['axis']} | {r['external_read_write_bytes']/1e6:.3f} | {r['first_h_block_statistics_ready']/1e6:.6f} | {r['first_confirmed_output']/1e6:.6f} | {r['shared_state_live_peak_bytes']/1024:.3f} | {r['external_mutable_live_peak_bytes']/1e6:.3f} |\n"
text+='''

公共资源是一个显式的宽资源平台：96 个 Float64 数值 lane，FMA II=1、依赖延迟 4；128 KiB、8×128-bit 读/写分路权重缓存；2 MiB、8×128-bit 1RW 状态服务；96 KiB 转置源缓存；两块各 245,760 B 的 96-lane 宽工作寄存器，共 480 KiB；16 KiB PSN 寄存器；64 KiB moments/参数/双阈值 scratch；32 KiB 恢复 bitmap/缓存目录。另有 8-lane moments、8 个 128-bit popcount、8 个 K4 包编码器及平方根服务。外部总线每拍 32 B，突发最长 4 KiB，每个突发另付 40 拍；输出总线每拍 16 B。所有路线提供同一资源集合，即使辅助单元闲置。特别是 480 KiB 宽寄存器和其端口**未做物理映射**，这不是等面积或 SRAM 宏可实现性证明。

实际 FC1 按原 c 顺序发射捕获的活动位置，显式处理 lane-bank 冲突、重复累加的四拍依赖、两级系数/源 staging 和权重缓存容量；没有用 MAC/96 直接作为 FC1 时间。PSN 使用完整 T=10、秩 10、100 个非零 A 元素；将 (位置,h,t) 展开以避免 q1 人为低利用率，全部 T 输入和累加器在明确容量中。每个 h 通道组覆盖完整 N；组间的 FIFO 调度不是全局最优调度证明。队列等待在 JSON 中按所有请求求和，可能重叠，不能加到总服务时间上。

本次最终表只重算首轮已选出的五个组织，未再次搜参：S0 的普通最快组织与 Gram 都为 q96；A8 用 q96 首遍、q1 精细恢复和有限驻留窗口顺序。S3 最快普通基线为 q96 存 U；sample0 A8 为 q1 窗口恢复，sample10 为 q96 tile 恢复。最后一种宁可多重算约 6.04 倍 h-tile 也比狭窄读取更快，这正是“失败少”不等于“恢复便宜”的例子。

Gram 的计算是 S∈{0,1}，z=θS；G=SᵀS，k=diag(G)，μ=θWk/N，var=θ² diag(WGWᵀ)/N−μ²。读取的两层源 θ 与出口 θ 恰好都是 1.0，这是 checkpoint 数据，不是把 ATLIF 普遍定义成 θ=1。一般每通道 θ 需要 Dθ 形式；本次只评价这两个真实标量阈值层。W 来自 ep34，未使用诊断权重或预计算 W 外积 ROM。

G 用 128 行转置双缓冲、固定 i 列寄存器和 8 路 j 列读取构造，上三角 32-bit 计数器的 1RW 读改写收费。收缩按 q16×6 个 i 行使用 96 数值 lane；G 的真实上三角地址产生读 bank 冲突。每个 q16 在线计算 v=Gw，再计算 wᵀv 与 kᵀw；G 读、v 写读、权重读取、平方根、仿射阈值和 τ 写读均付费。全部 H×T×8 B 的 τ 占用共享状态，S0 为 30,720 B，S3 为 245,760 B。旧表曾仅为 τ 留 H×16 B，现已修正，并因此减少 S0 可固定驻留的源包。

sample0/S0 的时间线：源保留和首批读取后，G 更新从 94,894 拍至 2,119,894 拍；在线收缩和阈值保存至 2,203,102 拍；首次最终消费者在 2,219,907 拍完成，最终输出在 25,455,358 拍完成。计数服务占 2,025,000 拍；6,984,000 次位交集、55,872,000 B 计数器读写、114,048,000 B 转置列读取均计入。后续在线 W 收缩为 3,612,672 MAC，G 读 2,431,872 B，v 写读 589,824 B。详细首末事件见 `final_comparison.json` 的 `execution_milestones`。

G 的稀疏 co-occurrence 生产没有被当成不存在：四点真实逐行计数增量分别为 38,354,975、8,897,761、38,130,077、8,870,235。对于每拍至多 8 个标量计数更新的逐行稀疏实现，S0 即使不付任何提取、端口或读改写也至少要约 4.79 M 拍，已高于所用位交集的 2.025 M 计数服务。这个下界只针对上述逐行实现；不否定另外的模式聚合结构。S3 的稀疏生产可能比当前 1.806 M 拍位交集更快，但当前在线 W 收缩自身就有 1,816,657,920 MAC，96-lane 无冲突理想下界为 18.924 M 拍，已经超过约 10 M 拍的完整存 U 基线；仅优化 G 生产不能挽救此深层组织。实际 S3 G 读取达 1,245,699,072 B，统计全部就绪在 32,597,620 拍。没有把这一下界写成实测周期。

A8 使用每 (B32,t,h) 的候选 bits、两侧证书和 K4 边界值，Float64 包按 16 B 对齐后 64 B；最终统计之后核验，失败对该 h 的整个 T 组恢复，只有最终确认后才交付消费者。服务表补足 32 B 输出描述符、失败 bitmap 写/扫描、两个在途恢复描述符和选择器费用。编码前缀统计仅使用当前与此前空间包的完整 T 数据；没有免费未来全域统计。首个恢复的源、权重、FC1、PSN、输出事件存于 `first_repair`。q1 精细恢复避免误重算其他 h，但同一源包必须多次供不同失败消费者；有限源窗口排序已是强普通控制，不应单独命名为新机制。

最值得保留的电路问题是：统计就绪之后，有限源转置驻留怎样为稀疏失败 h 供数，同时限制位图扫描、权重 sector 读取和全 T 重算放大。本轮已包含 tile 顺序、h-sector 顺序和有限驻留窗口内 h 顺序，任何新组织都必须超过这些简单对照。现有 A8 两个浅层样本的一胜一负不足以启动 RTL；动态 BN 可低代价替换为固定统计后，该等待/恢复问题的重要性还需重新评估。

数值边界：全四域的 Float64 内部比较中，BN 后 PSN、移阈值、sum/sumsq moments 和 θ 加权 Gram 的最终 bits 一致。但 Float64 参考与存档捕获比对在 sample10/S0 **差 2 位**，其余三域为 0；最小边界约 2.61e−8。独立 A800 真 FC1 FP32 输出捕获中，FP32 移阈值相对实际 sn2 bits 的差异按 Z/S0、Z/S3、I/S0、I/S3 为 0、157、1、185；在同一 Y32 上用 FP64 变换则四点为 0。这些探针使用 torch.var_mean，不能冒充 cuDNN 内部保存统计。模型没有逐拍仿真每次 IEEE-754 舍入，也没有产生原 FP32 的区间等价保证，因此不能称“冻结 ep34 完全无损 RTL”。sample10 不在 valid825 有 GT 的 AEE 子集中，仅用于硬件和数值分析。

主线算法强对照：训练集 32 帧校准、无需梯度训练，仅固定 S0 两个 FC1 BN 后完整 valid825 AEE 为 1.201095，当次 dynamic 为 1.198610，增加 0.002485。该结果由主线算法目录提供，不由本服务模型推导。允许这种模型改变时，应优先研究固定统计之后的执行行为，不能仅凭这里 31.7% 的受限组件模型收益保住复杂动态统计电路。数学矩传播和源 Gram 也已有先验，剩余研究价值在具体有限资源电路组织，本轮没有形成录用或 PPA 结论。

执行入口：`build_traces.py` 读取已有真实捕获与 checkpoint；`service_model.py` 是首轮普通组织比较；`finish_comparison.py` 对原先选出的组织作修正复算；`summarize.py` 生成本说明与 `gram_cost_analysis.json`。当前最终结果以 `final_comparison.json` 为准，首轮 `service_result.json` 保留原值便于看清修正，不能混用。
'''
(ROOT/'README.md').write_text(text)
print('WROTE README.md gram_cost_analysis.json')
