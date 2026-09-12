from pathlib import Path
import json
import numpy as np
import integrated as stage

HERE=Path(__file__).resolve().parent
r32=json.loads((HERE/'integrated_r32.json').read_text())
ranks=json.loads((HERE/'rank_fusion.json').read_text())
maskpath=HERE/'mask_fusion_global_group2.json'
mask=json.loads(maskpath.read_text()) if maskpath.exists() else None
rows=[]
for window in ['corner','interior']:
    a,b=[next(r for r in r32['rows'] if r['axis']==axis and r['window']==window) for axis in ['ordinary','lifting_raw']]
    rows.append(dict(window=window,mode='R32',ordinary_service=a['service_slots'],lifting_service=b['service_slots'],
        lifting_net_service_reduction=1-b['service_slots']/a['service_slots'],
        ordinary_source_service=a['source_program']['service_slots'],lifting_source_service=b['source_program']['service_slots'],
        source_service_reduction=1-b['source_program']['service_slots']/a['source_program']['service_slots']))
    for mode in ['original_ordered24','activation_whitened24']:
        ar,br=[next(r for r in ranks['rows'] if r['axis']==axis and r['window']==window and r['mode']==mode) for axis in ['ordinary','lifting_raw']]
        rows.append(dict(window=window,mode=mode,ordinary_service=ar['service_slots'],lifting_service=br['service_slots'],
            lifting_net_service_reduction=1-br['service_slots']/ar['service_slots'],
            common_rank24_saving_ordinary=a['service_slots']-ar['service_slots'],
            common_rank24_saving_lifting=b['service_slots']-br['service_slots']))
    if mask and len(mask['rows'])==4:
        ar,br=[next(r for r in mask['rows'] if r['axis']==axis and r['window']==window) for axis in ['ordinary','lifting_raw']]
        rows.append(dict(window=window,mode='global_group2_R32',ordinary_service=ar['service_slots'],lifting_service=br['service_slots'],
            lifting_net_service_reduction=1-br['service_slots']/ar['service_slots'],
            common_mask_reduction_ordinary=1-ar['service_slots']/a['service_slots'],
            common_mask_reduction_lifting=1-br['service_slots']/b['service_slots']))
inputchecks=[]
for label in ['corner','interior']:
    values=[stage.common.read_npz(stage.common.FULL/'capture'/axis/'000_zurich_city_09_a_0001.npz')[label+'_I24'] for axis in ['ordinary','lifting_raw']]
    inputchecks.append(dict(window=label,shape=list(values[0].shape),values=int(values[0].size),differences=int(np.count_nonzero(values[0]!=values[1]))))
    assert inputchecks[-1]['differences']==0
checks=dict(R32_integer_values=0,R32_integer_differences=0,R24_all_variants_integer_values=0,R24_all_variants_integer_differences=0)
for r in r32['rows']:
    for c in r['consumer']['checks'].values():checks['R32_integer_values']+=c['values'];checks['R32_integer_differences']+=c['differences']
for r in ranks['rows']:
    for c in r['checks'].values():checks['R24_all_variants_integer_values']+=c['values'];checks['R24_all_variants_integer_differences']+=c['differences']
stress=json.loads((HERE/'rank_fusion_ordinary_interior_stress.json').read_text())
for r in stress['rows']:
    ready=next(x for x in ranks['rows'] if (x['axis'],x['window'],x['mode'])==(r['axis'],r['window'],r['mode']))
    assert r['checks']==ready['checks'] and r['candidate_PED_delta']==ready['candidate_PED_delta']
result=dict(evidence='CPU integrated finite payload model; isolated RTL evidence is separately owned/reported in rtl_source.',
    no_system_or_RTL_speedup=True,no_native_projection_or_globalBN=True,rows=rows,input_comparison=inputchecks,checks=checks,
    pressure=dict(axis='ordinary',window='interior',same_ready_integer_output=True,
        services={r['mode']:r['service_slots'] for r in stress['rows']}),
    quality=dict(source='../../accuracy_baseline/valid825_summary.json',ordinary_full825=1.219801338299,
        lifting_full825=1.232979367919,NB0_full825=1.445352534681,
        caveat='These are prior paired full825 algorithm validations of original fixed R32, not full825 execution of this hardware model. New R24/masks retain their distinct AEE scopes.'),
    attribution='Both get identical coalescing, CSE privileges, 96RF, ROM budget and resident MAC. R24 and static masks are common ablations; candidate X is source structure after this complete local consumer cost.')
(HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# 结构T10源：同机器完整局部集成','',
    '已把两套真实源程序接入一台共享端口/状态的机器，经过真实门缓存后完成完整K864整数双消费者。共同R32下，lifting在corner/interior分别减少7.784%/7.996%总服务；双方都享R24后仍减少约7.95%/8.13%。这是局部模型服务，不是整网或RTL加速比。','',
    '|共同执行底座|窗口|ordinary服务槽|lifting服务槽|lifting净减少|','|---|---|---:|---:|---:|']
for r in rows:lines.append(f"|{r['mode']}|{r['window']}|{r['ordinary_service']:,}|{r['lifting_service']:,}|{100*r['lifting_net_service_reduction']:.3f}%|")
lines += ['',
    '计量起点是两学生完全相同的真实I24 halo：corner 77,760值，interior116,160值，逐值0差。源按原two_stage_writeback程序执行，再经过完整K864 preview U32/V、固定BN1、非因果T10 sn2、驻留门缓存、完整K864 Conv2 U16/F/BN2+raw I24，最后生成projection gate和真实PED U32/V96（或已导出U24/V96参数）。','',
    '边界是真实同Machine状态交接：不外写/重装sn2门。前级124,544B系数及后级44,096B R32系数在同128KiB池里实际依次替换，raw I24后继重读、输出DMA、地址/目录和所有RNE/sat仍收费。R24后级少3,072B系数，并在实际24维循环中减少计算/状态访问。R24分支从已实际执行后的完整Machine状态复制；这是替代方案仿真复用，未导入独立时间表或以gold喂入门缓存。','',
    '普通源282条程序含260次加减；lifting216条含159次加减和35次norm24（原位置的RNE+signed24饱和）。两者同8192B ROM、96×8×48 RF、128KiB state/coef、SR64/SW64/CR256。源本身减少16.369%服务，进入真实消费者后成为上表约8%的净值，不能把260→159写成硬件收益。','',
    '双方的原排列R24与白化R24在这四窗费用相同，每窗均减少45,184槽；其精度不同，不能以相同服务推断相同输出。global-H8只作已固定R32掩码消融，metadata在同state池冷填一次并供前后级复用，未混入未经评价的R24+mask网络。该地址/剪枝底座不单独构成X。','',
    f"检查：R32整数消费者{checks['R32_integer_values']:,}值0差；R32+两R24分支共{checks['R24_all_variants_integer_values']:,}值对各自独立gold 0差。源sn1和sn2门均符合原捕获。preview的raw/BN1 FP32与CUDA存在原来已有的微差，本轮逐项复现旧模型差异；没有把全链所有浮点值误称逐位相等。ordinary/interior固定压力的R32、两R24均通过，服务见summary.pressure。",
    '',
    '质量归属：原fixed R32完整825帧的ordinary AEE1.219801、lifting1.232979均优于同人口本地SDformerFlow NB0的1.445353；该结论来自accuracy_baseline中已逐帧配对的算法验证。R24与mask不继承原R32的825结论，后续新验证由独立算法任务记录。本模型没有回放完整帧native projection/global BN。',
    '',
    '创新归属：CSE、Gustav式稀疏供数、普通word coalescing、resident MAC、R24本身都归借入或共同底座；当前保留的候选是可学习T10源结构经过真实双消费者后仍剩的净服务。约8%是实测线索，尚不足以宣称TCAS-II强接收。本阶段没有轻率否掉源结构家族，也没有把条件消融改成新标题。',
    '',
    '文件：PLAN.md记录先写的B/A/X与强对照；integrated.py执行共同R32；rank_fusion.py与consumer_ranked.py执行真实R24；mask_fusion.py为固定global R32消融；source_rtl_inputs提供原程序、真实输入/gold和实际sink事件；RTL_REVIEW.md独立检查共同源RTL。rtl_source结果与CPU表分开，不相乘或互相替换。']
(HERE/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
