from pathlib import Path
import json,csv
H=Path(__file__).resolve().parent;s=json.loads((H/'SUMMARY.json').read_text());d={m:json.loads((H/f'quality_diverse_{m}.json').read_text()) for m in range(7)}
def quality(m):
 path=H/f'quality_valid_{m}.json'
 if path.exists():
  q=json.loads(path.read_text())
  if q.get('complete'):return f"{q['result']['AEE_frame_mean']:.12f}（{'过门' if q['below_same_set_NB0'] else '未过门'}）"
 return '未完成，正在运行' if m in [1,2,5,6] else ('1.327635022608（同环境封存值，未重跑）' if m==0 else '未运行')
def table(modes):
 t='| 执行模式 | 暖计算总拍 | 冷配置+计算 | 比exact暖拍减少 | diverse10 AEE | valid825 AEE |\n|---|---:|---:|---:|---:|---|\n'
 for m in modes:
  r=s['real'][str(m)]['0'];f={7:6,8:5}.get(m,m);t+=f"| {m} {s['modes'][str(m)]} | {r['total_cycles']} | {r['total_cycles']+r['configuration_cycles']} | {100*(107044-r['total_cycles'])/107044:.3f}% | {d[f]['result']['AEE_frame_mean']:.12f} | {quality(f)} |\n"
 return t
common='''共同合同：固定Q1[8,864]、Q2[96,8]、θ=1和aQ40/bQ20；完整Q1、T10、C96、N96、P4都在RTL。入口是实际r0.sn2脉冲与真实FP32 identity，出口完整I24。所有模式每tile480个8lane消费者字，包含FP32转Q20、乘加、RNE/saturation和背压。九执行模式共用全部编码状态、端口与8路加法器/8路signed16×14乘法器；固定核中的完整R8是mode0。参见[资源合同](RESOURCE_CONTRACT.md)、[执行账](EXECUTION_LEDGER.md)和[原始收据](results.json)。396次运行，三个检查点raw/J/I24各1,520,640值零差；独立源公式11088项通过。

训练校准使用实际train列表的thun_00_a_0002，336规则栅格窗口，输入183322个spikes；不属于valid825。硬件评价是另一帧zurich_city_09_a_0001的8个固定内部/边界块，没有活动筛选。每块连续4×4源、2×2输出；不是整层。校准参数在AEE之前冻结，不训练模型、不扫描比例。完整I24消费者同环境diverse10 baseline精确复现1.3903315401317662；同集合NB0十帧1.4699681144337489，官方825门1.447936665574317。十帧过门不等同825过门；共同全零encoder旁路前的数只存在pre_zero_bypass快照，最终396runs没有重复计数。旧历史十帧门1.45460286107也单独保留。AEE是此整数线性与I24函数接后续浮点网络，不能称全网bittrue。

表中暖拍从go到core+consumer全部退休；冷拍另加每tile3450拍完整source/参数配置，共8tile27600拍。跨其他代理的源快照、端口或整层结果不直接拼周期。背压/二次命令数据见SUMMARY.json。

'''
text1='''本项完成了“有损完成latent整向量关停”的实际接口：在所有864个K输入累加完后，对每P/T的8个signed latent一起决策，使整组Q2乘加可以关闭；实际I24、identity和后续PSN照常执行。8块暖拍比相同完整R8减少8.75%，但相对逐rank幅值强控制仅快1.202%，且十帧和825 AEE均更差（825：1.351391218079484对1.331155075046491）。当前证据是有限的速度/质量折衷，尚不足作为新标题。

A是[Bishop](https://arxiv.org/html/2505.12281v1)的结构组与误差约束剪枝；B是本网R8完成值存在signed抵消、Q2按实际nonzero rank收费，逐rank删除无法保证整个向量关停。本次X候选是以真实I24消费者训练校准出的轻量score驱动完成向量的执行单位，尚未显示强理论/算法新颖性；没有继承Bishop二值attention的误差界。

校准对每向量实际计算完整I24与将线性p置零后的I24，NNLS拟合8个|z_r|到I24 RMS变化，再将权重固定成2^[2,2,2,2,1,1,1,0]。score≤25×nnz(z)时整vector置零；τ×nnz以9项控制表实现。训练校准Q2 MAC从504336降至376032，削25.44%。强控制逐rank |z|≤[4,4,4,4,4,4,3,3]置零，削25.29%；两个训练预算接近，并未对验证集再对齐。真实评价分布下分别6360/8376次Q2向量MAC，不能把训练的25%当硬件评价比例。

RTL整组encoder每向量8拍：原生z读、共享减法、绝对值、三拍归约、score比较、原位写；非零向量每tile基准320拍。rank控制非零4拍/向量；两者全零向量都在同一次读出检测后直接写回，共2拍/向量。本8tile113/320个向量为全零，encoder实际分别1882/1054拍。完整source/Q1计算和20字z清零没有减少。其训练I24误差proxy不是严格上界，signed抵消会令L1高估/失真。

'''
text2='''本项完成了“完成latent K4原型+单rank残差”的实际表示接口，编码、weighted距离、最近原型选择、最大加权残差选择、原型Q2表读、残差乘加均在RTL。8块暖拍比完整R8少7.69%；diverse10为1.471894678700，略差同集合NB0门1.469968114434。按预先约定的≥10%净周期拓展条件，本工作点不再跑825，也不靠换K/残差数/重选格式续命。它比zero+单rank控制质量好，但并未因此胜过完整R8。

A是[LUT-DLA](https://arxiv.org/html/2501.10658v1)的在线距离+查表与[Phi](https://arxiv.org/html/2505.10909v1)的pattern/weight product加双向残差。B是这里完整Q1照付，Q2只有8rank，encoder成本很容易吃掉算子削减。本次仅测试从二值模式迁至完成signed latent，并只保留一个最大加权残差的新有损工作点；这种迁移与限定残差本身没有建立足够X。

code0固定全零，另3原型用训练窗口weighted L1最远初始化与8次固定kmedian获得，不使用验证或GT。运行argmin Σ_r 2^shift_r |z_r−c_r|，低index解tie；残差rank用argmax同weighted绝对值。输出p=Q2*c+Q2[:,r]*(z_r−c_r)，其余七个残差删除。z_hat各坐标仍来自真z或训练原型，signed13可存；RTL按早先抽象界给残差保守留signed14，共同乘法器为16×14；固定Q1精确L1界最大667，实际13bit已经足够。完整整数gold同时核对“重建z_hat后dot”和“prototype结果+残差”相等。

非零vector encoder29拍，全零同读检测后2拍；本8块encoder实际6229拍；共660个256bit原型结果字读、2484次Q2 MAC。zero-prototype+单rank控制非零vector5拍/全零2拍，encoder实际1261拍、无非零原型读、同2484 MAC，但AEE1.714547411881。原型结果表1536B，codebook52B及元数据另计。所有参数冷装入也计；没有把encoder放TB或把表当免费组合ROM。codebook小寄存器有独立8lane读口，公平资源中明确列出。

'''
text3='''本项完成了同帧T10“完成latent整向量保持”的有损接口。每P的t0全算，以当前z对上次保留近似z计算weighted L1，≤206时复用近似vector及Q2线性结果，否则刷新。每T仍消费其真实identity，执行固定BN折叠/I24与后续网络；不缓存后继PSN，不用最终flow作同帧先验。

A是[DeltaCNN](https://arxiv.org/html/2203.03996v2)所代表的阈值更新/状态缓存/稀疏delta；B是本网只有8rank，单rank迟滞可能少改值却关不掉整Q2输出。候选完成vector保持将有损决策和线性输出生命周期绑定，在这个短T10布局得到实测折衷；并不是首次阈值保持，也不是旧无损Δ+prefix的重复结果。本次阈值保留了不同函数，不能称lossless。

门206来自训练相邻真实z变化weighted L1的正值25分位一次统计，未递归扫描阈值。参考逐rank死区[6,5,4,4,4,4,4,4]使用相同训练集一次约25%变化rank预算。所有比较对“上次实际保留的近似值”，t0/空间P/命令都会重启，持续小变化不会因只比较相邻真值而永久无界遗漏。

强控制进一步给双方相同编译权限：计算Δ=z_hat(t)−z_hat(t−1)；可装signed13且变化rank少于当前nonzero rank时，只对变化rank执行prev p+Q2Δ，否则完整Q2重算。shared EDIFF已经形成差值，原位z半字存Δ，104bit参考寄存器保留近似值，use_delta40bit控制留存prev p；没有第二套完整z表。mode7/6与mode8/5各自gold相同，质量不重复测。范围不满足不得截断。

8块，整向量两执行选项5→8为100916→100808拍；rank强控制6→7为107612→102596拍，说明不给局部差分权限会明显削弱控制。最终整vector比强rank少1.743%暖拍，但diverse10为1.386801509411，强rank1.345721957989更好。完整825也为整组1.3520299033733638、强rank1.3269470346487395；后者略低于原R8固定消费者1.3276350226079938。不是全面胜出。两侧额外encoder、metadata、full identity/I24全付。

'''
for name,txt,modes in [('AS1_GROUP',text1,[0,1,2]),('AS2_PROTOTYPE',text2,[0,3,4]),('AS3_TEMPORAL',text3,[0,5,6,7,8])]:
 (H/(name+'.md')).write_text(txt+table(modes)+'\n'+common+'实现与重现：[reference.py](reference.py)、[lossy_r8.sv](lossy_r8.sv)、[run.py](run.py)、[verify.py](verify.py)。完整primary来源边界见[SOURCES.md](SOURCES.md)。\n')
completed=sum((H/f'quality_valid_{m}.json').exists() and json.loads((H/f'quality_valid_{m}.json').read_text()).get('complete',False) for m in [1,2,5,6])
status=f'四个候选/强控制的官方825已完成{completed}/4；剩余评价仍在运行，未完成项不能写PASS。' if completed<4 else '四个候选/强控制的官方825全部完成，四臂均通过同环境NB0严格门。'
(H/'README.md').write_text('三项不同有损接口已完成完整K864→R8→N96→FP32 identity/I24的小fixture RTL，以及各自diverse10。'+status+'AS2固定K4+1工作点十帧未过，净周期收益小于约定10%扩展门，停止本点。没有模型训练、格式/比例扫描、生产修改或EDA。\n\n'+table(list(range(9)))+'\n结果入口：[AS1组关停](AS1_GROUP.md)、[AS2原型残差](AS2_PROTOTYPE.md)、[AS3时间保持](AS3_TEMPORAL.md)、[周期与位宽](RESOURCE_CONTRACT.md)、[独立公式](EXECUTION_LEDGER.md)、[primary来源](SOURCES.md)。\n\n本地复现使用Python3.12：`/opt/anaconda3/bin/python3.12 prepare.py`，`/opt/anaconda3/bin/python3.12 run.py`，`/opt/anaconda3/bin/python3.12 verify.py`。SV已提交独立文件；make_rtl.py随后strengthen.py可重建当前核心，make_wrapper.py重建共同消费者wrapper。NPZ大源保留本地，常量hex和frozen_parameters.json/parameters.json含整数实值。GPU入口evaluate_sparse.py，沿既有A800 env312、同合法模型及数据协议；不需要重建venv。\n')
with (H/'comparison.csv').open('w') as out:
 w=csv.writer(out);w.writerow(['mode','name','warm_cycles','cold_cycles','diverse10_AEE','valid825'])
 for m in range(9):
  r=s['real'][str(m)]['0'];func={7:6,8:5}.get(m,m);w.writerow([m,s['modes'][str(m)],r['total_cycles'],r['total_cycles']+r['configuration_cycles'],d[func]['result']['AEE_frame_mean'],quality(func)])
