from pathlib import Path
import csv,json
H=Path(__file__).resolve().parent
rows=json.loads((H/'results.json').read_text());checks=json.loads((H/'checks.json').read_text());assert checks['all_pass']
summary={'complete':True,'runs':len(rows),'checked_outputs':sum(r['values_checked'] for r in rows),'real_tiles':8,'functional_tiles':5,'summary':{}}
for kind in ('direct','one_axis'):
 summary['summary'][kind]={}
 for stress in (0,1):
  rs=[r for r in rows if r['fixture'].startswith('real_') and r['kind']==kind and r['stress']==stress]
  fields=('cycles','configuration_cycles','weight_requests','weight_responses','cr_bytes','cache_hits','alu_vector_cycles','input_stall','output_stall','request_stall')
  s={f:sum(r[f] for r in rs) for f in fields};s['state_cycles']=[sum(r['state_cycles'][i] for r in rs) for i in range(19)]
  summary['summary'][kind][str(stress)]=s
summary['one_axis_slowdown_percent']=100*(summary['summary']['one_axis']['0']['cycles']/summary['summary']['direct']['0']['cycles']-1)
(H/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
with (H/'costs.csv').open('w') as f:
 fields=[k for k in rows[0] if k!='state_cycles'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
contract={'scope':'Same one_axis_tile module, runtime mode0 original-W direct AAC and mode1 horizontalF(2,3)+verticaldirect; C96,N96,T10; 4x4 input to2x2 output. Not the native mode6 resource point.',
 'arithmetic':{'shared_signed48_add_sub_chains':8,'multipliers':0,'shared_by':['input transform','AAC','inverse','output exact arithmetic shift'],'address_control_arithmetic_is_additional':True},
 'external_ports':{'input_bits':128,'metadata_bits':128,'coefficient_request_response_bits':256,'coefficient_outstanding':1,'output_bits':256,'origin':'two signed16 fields in the paid start command'},
 'local_arrays':{
 'source':{'entries':960,'bits':16,'bytes':1920,'permissions':'8 writes/load; fixed10-word T-gather network forSCAN and fixed8-channel transform read network. These are register/mux permissions, not a claimed single-read SRAM.'},
 'V':{'entries':15360,'bits':3,'bytes':5760,'permissions':'8 writes/TR1; at most10 V reads perSCAN orSCAN2, vertical consumers gathered in separate cycles'},
 'M':{'entries':1280,'bits':48,'bytes':7680,'permissions':'AAC8read+8write in one cycle; inverseA16read; inverseB8read+8write. Explicitly wider access than native_sparse.'},
 'work':{'entries':8,'bits':48,'bytes':48},'static_support':{'entries':6912,'bits':2,'bytes':1728,'permissions':'64-entry finite successor and half-zero selection;64 entries/128bit meta beat'},
 'decoded_weights':{'bytes':64},'retained_three_registers':{'bytes':64,'note':'Inherited common state; PRE3 never executes for actual one-axis alphabet'},
 'assembled_lines':{'bytes':64},'cache_data':{'bytes':64,'lines':2,'line_bits':256},'cache_tags':{'bytes':8},'V_holds':{'entries':20,'bits':3,'bytes':7.5},'events':{'bits':40,'bytes':5}},
 'listed_array_bytes':17412.5,'not_in_array_total':'FSM,integer address/loop counters,valid flags,origin32bits,out_data256bits,parity diagnostic; no synthesized area estimate',
 'coefficient_format':{'direct_bits':16,'one_axis_bits':18,'direct_bytes':165888,'one_axis_bytes':248832,'one_axis_N16_vector_bits':288,'one_axis_logical_growth':1.5,'physical_rule':'Global bit-tight stream, address=floor(bit_offset/256), assembled at most2 CR lines; cache misses generate real requests/responses'},
 'metadata':{'direct_bytes':1296,'one_axis_bytes':1728},
 'common_strong_permissions':['completeP4/T10 coefficient reuse in direct','finite64-entry static successor','whole-source zero fast path','finite-value dynamic zero skip','O8 half-zero CR cancellation','same2-line cache','same8adders and same largest M/V/source state'],
 'unsupported_claims':['No full-frame one-axis stream run','No whole-network bittrue or new AEE','No EDA/Fmax/energy','No cross-resource speedup against native mode6','No novelty from ordinary1D Winograd']}
(H/'resource_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
a=summary['summary']['direct']['0'];b=summary['summary']['one_axis']['0'];stress=summary['summary']['one_axis']['1']
report=f'''# 横向一维Winograd挑战者：固定接口功能通过，真实稀疏r0未获益

2026-09-14，最后一个挑战者已完成并封口，不增加后续布局实验。**52 runs、199,680个输出逐值全匹配**：原八个完整C96/N96/T10真实tile，加全零、全一、图外毒值、孤立事件和半组零五个固定控制，分别运行direct/one_axis及两种背压。无mask新增近似；整个比较均为原始dense Q16函数。[完整结果](results.json)、[汇总](SUMMARY.json)、[逐case费用](costs.csv)、[资源合同](resource_contract.json)。

真实八块总周期为 **direct178,948、one_axis305,875，后者慢{summary['one_axis_slowdown_percent']:.2f}%**。同模块direct保留前轮已审的完整P4/T10复用、有限64项静态successor、半组零skip和两行CR缓存；16个真实direct运行的160项计数/状态数组与[旧强direct](../../r0_execution_trials_20260913/winograd/REPORT.md)完全复现。这里的源读取/CR/psum端口比native mode6宽，**不能将178,948与native mode6的周期跨资源相除。** 这一负点无需通过额外的pair布局证明；不宣称当前direct是所有可能设计的上界。

## 实际不同的分解接口

沿[定义的第二候选](../novelty/REPORT.md)，源每行的四个位通过Bt变为四个有符号值，纵向保持原3tap直接归约：

`V[c,t,y,i]=Σx Bt[i,x]S[c,t,y,x]`

`U2[n,c,ky,i]=Σkx G2[i,kx]Wq[n,c,ky,kx]`

`M[n,t,py,i]=Σc,ky U2[n,c,ky,i]V[c,t,py+ky,i]`

`Y[n,t,py,px]=(Σi At[px,i]M[n,t,py,i])/2`。

G2为2G，V仅取−1/0/1/2，signed3存储。全C96和全部ky归约后执行横向逆变换，结果必整除2；RTL在输出检查低bit奇偶，出现非整除即报错，不用RNE掩盖错误。负数用有符号右移一位得到精确整数。首逆变换状态以16个M读数完成八组两项和，第二状态再读第三项并写回；py0输出先写M0/M1，之后py1才写M2/M3，因此原位写回不覆盖尚需的第二行M4..M7。

本核没有执行第二维输入/逆变换：输入只有一次横向TR1；第二纵向消费者通过付费SCAN2读取另一源行。M是8个坐标而非2D的16个。旧基座的PRE3及第二逆变换状态保留为共同未到达编码，实际52次运行均为0拍；没有原样重跑2D而更名。

## signed18紧排与有限实际端口

U2最坏包含三个signed16系数之和，必须保留signed18；物理文件按18bit全局紧排，没有24bit填充。每N16向量是288bit，可能跨两个256bit CR行；RTL计算位地址、发实际miss请求、组装两行并解码，已有两行cache的hit也实测。原W16为165,888B，U2为248,832B，增为1.5倍。不是只报12/9个系数；也没有把名义36→24乘积槽变成周期速度。

两侧共用128bit源装入、128bit静态metadata、256bit单outstanding CR与两行缓存、256bit输出、八条signed48加减链。源是原生4×4的T/C空间字；origin在start命令锁存，LOAD用RTL图界掩码处理毒值padding。source的T10读取与8通道transform读取是明确的固定寄存器gather网络；V每次最多读10个值，两个纵向消费者分SCAN/SCAN2两拍；M的AAC可8R8W，逆变换首步需要16R。这些宽访问双方同有，未称为native单10bit源口，也未假定普通单读SRAM能免费实现。

M共享最大容量为1280×48bit（7,680B），V为15360×3bit（5,760B），源1,920B，支持metadata1,728B，两行cache64B、组装64B、权重64B等均列在资源合同。动态源支持、变换值、逆变换、码/半组选择和缓存全部在SV；TB只供普通原生输入及静态常量，发起一份有限未完成CR事务并施加响应延迟。

## 完整周期和瓶颈

八真实tile、无背压：

| 项目 | direct | one_axis |
|---|---:|---:|
| 总周期，含metadata/input/output | 178,948 | 305,875 |
| metadata配置拍 | 648 | 864 |
| 输入128bit装入拍 | 960 | 960 |
| CR256请求 | 11,790 | 20,580 |
| 请求搬运字节 | 377,280 | 658,560 |
| AAC八lane更新拍 | 77,172 | 75,180 |
| 输入变换状态拍（含零组判定） | 0 | 7,335 |
| 逆变换拍 | 0 | 7,680 |
| 第一SCAN拍 | 41,472 | 55,296 |
| 第二纵向SCAN2拍 | 0 | 55,296 |
| M清零拍 | 3,840 | 7,680 |
| 共享ALU向量拍 | 81,012 | 93,500 |

实际AAC更新只少2.58%，输入扩散、两行V供数与系数宽度引起的代价超过节省。每个实际U2向量占两次LOOKUP，两行cache命中10,572次，仍需20,580个外部CR请求。输入变换状态7,335拍中包括535个全零8通道组的一拍判断，实际算术为6,800拍，未把判断漏掉。两模式输出ROUND/SEND各3,840拍；one_axis的ROUND执行精确/2。

带输入阻塞、权重请求阻塞、1→4拍响应延迟及输出阻塞后，direct219,807、one_axis374,787，结论不变。所有字段来自TB对实际FSM/握手的计数，总周期等于全部状态拍之和；缓存/背压并非离线周期代算。

全一功能分布则one_axis100,070拍低于direct442,283，因为V集中于sum坐标；孤立事件控制反而17,414对7,007。它们说明不同支持分布的行为，均不进入真实八块主分母，也不据负点否定一维Winograd家族。

## 验证范围与停止结论

[检查脚本](verify.py)/[检查结果](checks.json)另核144个原生源/卷积tap基恒等式、全部16种四bit源行、2,515,968个signed16/18紧排系数逐值解码，覆盖83,520个跨CR行的U2向量及1,152个half-only U2向量。这些跨行/half-only数量属于静态格式解码检查。半组零RTL控制的源为全一，仅ξ1活动，实际高半组起始偏移为32/160，分别覆盖两字/单字读取；其他半组对齐未宣称已动态穷举，低半组单独存活无专用动态fixture。RTL用随机全signed16范围负权、全一、原生边界毒值和真实输入共同验证；每结果beat地址和八lane逐值核对，输出受阻时保持。每run完整C/N/T/K，非局部加法模型。

该一维挑战者实际范围为八个原真实tile和五个功能tile，未接整帧stream wrapper，也未声称完整层/全帧结果；本阶段的整帧证据只属于[mode5/6 stream](../stream_rtl/REPORT.md)。本点在相同模块强direct下已明确负，按授权停止扩展。普通一维Winograd、常量编译、有限值skip和缓存均归A，不由本次实现赋予X；没有新AEE、EDA、PPA或频率声明，原八块不能替代完整任务泛化评估。

复现：`/opt/anaconda3/bin/python3.12 prepare.py`、`run.py`、`verify.py`、`summarize.py`；run使用Verilator4.028 `--cc --exe -CFLAGS -O3`后独立make。旧目录未修改。

独立[结构、整数与词布局审阅](../novelty/REVIEW_ONE_AXIS.md)无阻断发现，另核1,336,320个存活半组系数槽与176项旧direct标量计数。
'''
(H/'REPORT.md').write_text(report)
(H/'README.md').write_text('# One-axis F(2,3) challenger\n\n固定接口已封口：[REPORT.md](REPORT.md)、[资源合同](resource_contract.json)。\n\n- 原八真实tile+五功能tile，52runs/199,680输出全绿。\n- 同模块强direct178,948拍，one_axis305,875拍；不跨资源比较native mode6。\n- signed18系数真实紧排/两行CR组装，输入与逆变换、纵向第二次V读取均付费。\n- `one_axis_tile.sv` / `tb.cpp` / `prepare.py` / `run.py`：完整C96/N96/T10实施。\n- `verify.py` / `checks.json`：恒等式、字布局与旧direct不弱化核对。\n- 仅tile范围，未声称整帧；不再追加布局实验。\n\n独立审阅：[REVIEW_ONE_AXIS.md](../novelty/REVIEW_ONE_AXIS.md)，无阻断。静态字节覆盖与动态半组对齐覆盖分别报告。\n')
print(json.dumps(summary,indent=2))
