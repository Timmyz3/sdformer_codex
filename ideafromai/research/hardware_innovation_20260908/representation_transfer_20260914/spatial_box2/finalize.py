from pathlib import Path
import json
H=Path(__file__).resolve().parent
s=json.loads((H/'SUMMARY.json').read_text());c=json.loads((H/'comparison.json').read_text())
lines=['# Box2 普通强控制：同 moment 函数，完整 raw→I24 实测','',
'Box2 保持冻结 moment 的全部系数、RNE 和输出，完成了源侧共同 `(1+x)` 的实际 RTL。它在 36 个跨序列 tile 上比普通三 tap 快 9.95%，但比 moment3 慢 2.07%；两套 64 tile 分别比 moment3 慢 7.24% 和 7.93%。因此小权重表与免恢复阶段并未取代 moment3：源计数使 Q1 支持扩大，普通两 tap 又需要更多 Q2 MAC。这是成熟代数分解的强控制，不是新机制或新公式。','',
'| 同函数完整消费者冷服务 | 普通三 tap | 通用 4M | moment3 | Box2 ready | Box2 BP |',
'|---|---:|---:|---:|---:|---:|']
for stage,label in [('held','原帧 128–191 / 64'),('disjoint','原帧 4000–4063 / 64'),('sequences','18 序列 × 2 tile / 36')]:
 r=next(x for x in c['comparisons'] if x['stage']==stage and x['consumer'] and not x['stall']);b=next(x for x in c['comparisons'] if x['stage']==stage and x['consumer'] and x['stall'])
 lines.append(f"| {label} | {r['ordinary3tap']:,} | {r['general4M']:,} | {r['moment3']:,} | {r['box2']:,} | {b['box2']:,} |")
lines += ['', '冷服务从配置原始源/权重至消费者最后输出，包含每 tile source 1536 拍、origin 1 拍、start 1 拍，及每次 replay 首命令静态 Q1 576 + Q2 384 + a/b 24 = 984 拍；第二遍沿用权重，分别减 984 拍。原生 FP32 identity 供数、J20 转换和全部消费者等待均实际执行。BP 使用与对照相同的 source/weight/raw/identity/output 停顿日历。raw-only 的 cold ready 三集为 1,881,012 / 2,036,282 / 795,806 拍；完整消费者 ready 固定再付 2,425 拍/tile 和首个 a/b 24 拍。每条原始结果保存在 `raw_*_s*.jsonl`、`consumer_*_s*.jsonl`，每记录一行。','',
'冻结公式为 `g=[a,a+b,b]`，源邻接计数 `c=s_j+s_(j+1)∈{0,1,2}` 经原 Q1 得到 `E_j=Z_j+Z_(j+1)`，再做 `y0=aE0+bE1, y1=aE1+bE2`。实际门由原 source_mem 读取进 C16 窗口；六组 T10 OR/AND 构造非零和 count2，CHECK/TIMESEL/ZREAD/ZADD 都收费。count2 通过 signed `W<<1` 接线和 0/W/2W mux，在同一 AAC 拍完成，不进行第二次加法。唯一八个 32 位 ALU 在 bit16 截断 carry 做 dual16。静态所有 binary 源及全部 Q1 前缀 E∈[-17124,14854]；两 tap Q2 任意归约前缀绝对值≤140,756,698，分别严格容于 signed16/32。原 p32→a_q40、FP32 identity→J20、wide64→RNE/sat24 I24 无中间 RNE 或 scale 改动。','',
'| 资源/服务 | Box2 实现 |', '|---|---|',
'| 原 source / 窗口 / source-mask holding | 1920 B / 20 B / 10 B；后者是独立保持寄存器 |',
'| E / p SRAM | 原 1280 B 八 bank E，2Y×3 列加第四零 padding；p 15360 B |',
'| Q1 / Q2 / Q2 cache | 4608 B / 4992 B / 208 B；Q2 仅 [a,b]，无第二份 g 或变换表 |',
'| 支持 | Q1 live 72 B，Q2 live 48 B，position 80 B，block+remaining 32 bit，rank 8 bit |',
'| holding | acc 32 B、E read 32 B、Q1 8 B、output 32 B、pending 40 bit；两个 active count 共 4 bit，比原门选择多 2 FF |',
'| 组合新增逻辑 | 六组 10-bit OR/AND，八 lane × 两字段 0/W/2W mux，dual16 carry cut；无 count SRAM |',
'| 算术 | producer 8×32 ALU + 8×signed19×13 mult，E16 符号扩展；原 consumer 8×32×32 mult + 8×64 ALU，单 context |',
'| 实际端口 | 单原 source 10 bit；单 Q1/Q2 权重授权 256 bit（Q1 有效64 / Q2有效104）；E 共地址八 bank 256-bit 向量或单 bank 32-bit scalar，读写互斥；p 共地址256-bit读写互斥 |',
'',
'Q2 表比 moment3 少 2496 B、cache 少 104 B，普通 acc 32 B 也无需 moment3 的额外 64 B M 累加器和 32 B transform tail；E 与 p 容量不增加。这仅说明资源合同与显式状态，并非综合等面积或等 Fmax 已证。mux/OR/AND 非免费；未运行 EDA，异步读+组合 ALU 时序模型与原 factor 相同。原三 tap、通用 Winograd、moment3 目录均未修改。','',
'36 tile 对 moment3 的实付差异可精确分解：Q1 共享 issue 39,550→45,894，新增 6,344 个事件各付 TIMESEL/ZREAD/ZADD 三拍，共 +19,032；Q2 MAC 217,296→282,288，+64,992；省固定 1,832×36=65,952 和首配置192，净 +17,880。每行新增 Q1 事件恰为 s0=s1=0 且 s2=1，脚本直接从原源独立计数复核，count2 本身没有第二拍税。源词96,768与 Q1 权重词8,248均不变；Q2 权重词16,092→10,728、cache写20,736→13,824，却不能补偿更密的 E 支持和两输出重复读取/乘法。两64的对应净差137,530 / 160,960也逐命令完全复现。','',
'验证覆盖 178 个独立 fixture（small15 含八个真实边缘/内部 tile、zero/one/random/tail、正/负 rank 支持、padding poison；两64与36跨序列有重叠 fixture）。每集 ready/BP、连续两遍无 reset：raw 与完整消费者各716命令；各检查2,749,440个完整 raw 输出，消费者另同数 J/wide/I24，每命令还比较1280个 E16物理字段（960有效、320零padding），末包地址与背压稳定性全部通过。独立逐FSM/服务检查158,968项；每个ready命令满足 `Box2−moment3 = ΔQ2_MAC +3ΔQ1_issue−1832`。CPU同时检查直接 source-count Q1、原 Z 邻和、两tap/原三tap/expanded P 与 I24。跨序列只复用不变上游 source/FP32 identity 前缀，按新 moment 重算输出，未把母 Q11 P 当 gold，也没有声称新36帧网络 capture。网络质量由父任务已有同 moment 函数评估负责，这里没有重训/重新定点。','',
'复现入口：`prepare.py` → `run.py --stage small` 与 `run.py --consumer --stage small`，后续 `--stage held|disjoint|sequences --skip-build`；最后 `verify.py`、`compare.py`、`finalize.py`。Verilator 4.028，编译用 `-Wall`，Python `/opt/anaconda3/bin/python3.12`。检查结果见 [SUMMARY.json](SUMMARY.json)、[comparison.json](comparison.json)、[admission.json](admission.json)，源/数值生成见 [prepare.py](prepare.py)，实际 core 见 [spatial_core.sv](spatial_core.sv)。']
(H/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(README=str(H/'README.md'),passed=s['passed'],commands=s['commands'],checks=s['checks'])))
