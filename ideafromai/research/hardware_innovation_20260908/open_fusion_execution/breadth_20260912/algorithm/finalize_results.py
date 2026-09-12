"""Summarize completed matched endpoints, without inventing pending results."""
from pathlib import Path
import csv,json
import numpy as np
HERE=Path(__file__).resolve().parent


def arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def main():
    parent=HERE/'matched_training';run=json.loads((parent/'run.json').read_text())
    base=HERE.parents[1];nbfile=base/'accuracy_baseline/source_nb0_valid825.csv'
    with nbfile.open() as f:nb={r['file']:r for r in csv.DictReader(f)}
    table=[];details={}
    for axis,entry in run['axes'].items():
        init=arrays(HERE/'initialization'/(axis+'_constants.npz'))
        for stage,r in entry['stages'].items():
            if not r.get('complete'):continue
            directory=parent/axis/('stage'+stage);q=arrays(directory/'deployed_constants.npz')
            rows=json.loads((directory/(axis+'_frames.json')).read_text())
            assert len(rows)==10 and all(int(float(nb[x['file']]['valid_pixels']))==x['valid_pixels'] for x in rows)
            key='lifting_q12' if axis=='lifting40' else 'As_q16'
            changes={k:dict(elements=q[k].size,changed=int(np.count_nonzero(q[k]!=init[k])),max_integer_change=int(np.max(np.abs(q[k].astype(np.int64)-init[k].astype(np.int64)))))
                for k in [key,'source_threshold','consumer_threshold','U_conv2_theta_q16','F_q16','U_ped_q16','V_ped_q16','BN2_constant_q24','PED_bias_q24']}
            interface=dict(structure=axis,actual_source_coefficients=key,
                source_program='Four layers, two q12 lifting half-steps per layer, RNE/saturate24 after each; fixed output permutation then inclusive threshold' if axis=='lifting40' else 'Signed16 source matrix dot, RNE/saturate24, inclusive threshold',
                authoritative_cutoff_fields=['source_threshold','source_direction','source_constant','consumer_threshold','consumer_direction','consumer_constant'],
                inherited_readout_metadata='source_tau_real, mathematical cutoff and native source bias/readout metadata are historical; do not regenerate learned literal cutoffs from them.',
                original_I24_producer_unchanged=True,
                literal_biases={k:dict(elements=q[k].size,integer_min=int(q[k].min()),integer_max=int(q[k].max())) for k in ['BN2_constant_q24','PED_bias_q24']},
                note='The raw superclass static_constant report describes its parent compilation; literal_biases here are recomputed from the actual exported arrays. Per-frame arithmetic ranges and clip counts are actual new deployment observations.')
            (directory/'compiled_interface.json').write_text(json.dumps(interface,indent=2)+'\n')
            if axis=='contiguous34':assert np.count_nonzero(q[key]*(1-q['source_mask']))==0
            activity=json.loads((directory/'activity_ranges.json').read_text());gates={}
            for frame in activity['frames']:
                for gate in ['source_gate','consumer_gate']:
                    a=gates.setdefault(gate,dict(nonzero=0,elements=0))
                    for k in a:a[k]+=frame[gate][k]
            aee=float(np.mean([x['AEE'] for x in rows]));h9=float(np.mean([x['AEE'] for x in rows[1:]]))
            b10=float(np.mean([float(nb[x['file']]['AEE']) for x in rows]));b9=float(np.mean([float(nb[x['file']]['AEE']) for x in rows[1:]]))
            item=dict(structure=axis,new_GT_steps=int(stage),AEE_diverse10=aee,AEE_holdout9=h9,
                delta_NB0_diverse10=aee-b10,delta_NB0_holdout9=h9-b9,
                better_than_NB0=aee<b10 and h9<b9,valid_pixels=sum(x['valid_pixels'] for x in rows),
                source_nonzero=int(np.count_nonzero(q[key])),source_quantized_changed=changes[key]['changed'],
                source_gate_density=gates['source_gate']['nonzero']/gates['source_gate']['elements'],
                consumer_gate_density=gates['consumer_gate']['nonzero']/gates['consumer_gate']['elements'],
                deployed_flow_difference=r['trained_QAT_vs_reloaded_literal_flow_max_abs'])
            table.append(item);details[axis+'/'+stage]=dict(parameters=str(directory/'deployed_constants.npz'),
                changes=changes,actual_gate_totals=gates,onepass_bn=True,full_valid825=False)
    result=dict(complete=False,matched_training_complete=run['complete'],common_prior_GT_steps=320,moment_init_steps_per_arm=1024,
        scheduled_new_GT_steps_per_arm=320,paired_frames_and_masks=True,full_valid825=False,rows=table,details=details,
        numeric='QAT hard forward and reloaded literal fixed deployment; no inherited old precision/service metrics.')
    result['new_training_vs_valid825_overlap']={k:len(set(s)&set(nb)) for k,s in run['training']['schedules'].items()}
    assert not any(result['new_training_vs_valid825_overlap'].values())
    full_path=HERE/'valid825/run.json';full_rows=[]
    if full_path.exists():
        full=json.loads(full_path.read_text())
        for axis,v in full['axes'].items():
            if v.get('complete'):
                s=v['summary'];assert s['frames']==825 and s['valid_pixels']==48152523
                full_rows.append(dict(structure=axis,AEE_frame_mean=s['AEE_frame_mean'],AEE_pixel_mean=s['AEE_pixel_mean'],
                    NB0_frame_mean=v['NB0_AEE'],delta_NB0=v['delta_NB0'],better_than_NB0=v['better_than_NB0'],frames=825,valid_pixels=s['valid_pixels']))
        result['fresh_valid825']=dict(complete=full['complete'],rows=full_rows)
        result['complete']=bool(run['complete'] and full['complete']);result['full_valid825']=bool(full['complete'])
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if table:
        with (HERE/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=table[0].keys());w.writeheader();w.writerows(table)
    lines=['# 同父、同预算三结构实测','',
        '每臂共享普通R24＋onepass父的旧320步历史，再接受相同1024步TRAIN输入矩初始化及64＋256步真实GT恢复。最终选择只看320步端点；64步为预定中间收据。',
        '', '同预算不等于同参数量：初始化有效拟合参数为dense100、masked普通34、lifting40＋10个gain；34分配100槽张量但仅34项可更新。初始化均值补偿只属于拟合域，部署逐处RNE后的均值由真实推理决定。',
        '', '新定点常量需重新编译与重新计费，不继承旧lifting节点数、RTL周期或PPA。这里是10帧功能/精度比较；未完成项不填数。','',
        '|结构|新增GT步|10帧AEE|9帧AEE|相对NB0（10帧）|源非零系数|实际改变系数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    lines.extend(f"|{r['structure']}|{r['new_GT_steps']}|{r['AEE_diverse10']:.9f}|{r['AEE_holdout9']:.9f}|{r['delta_NB0_diverse10']:+.9f}|{r['source_nonzero']}|{r['source_quantized_changed']}|" for r in table)
    lines+=['','原始source I24的上游生产函数全程冻结；可用旧ordinary父同位置I24作为新source接口输入。消费者矩阵也经训练更新，完整下游服务需使用新参数和重新产生的门。',
        '', '34项结构共享固定输出排列；有效mask始终34项。lifting40使用八个实际RNE/sat24半步，旧As矩阵仅为父元数据，不是其运行时源矩阵。']
    if full_rows:
        lines+=['','## 新学生完整valid825','',
            '下表均为本轮最终320步常量的独立重新加载推理，825帧/48,152,523有效像素逐帧匹配本地upstream复现NB0。NB0原最终头与本学生粗头的区别保留。','',
            '|结构|825帧均AEE|像素加权AEE|相对NB0帧均|','|---|---:|---:|---:|']
        lines.extend(f"|{r['structure']}|{r['AEE_frame_mean']:.9f}|{r['AEE_pixel_mean']:.9f}|{r['delta_NB0']:+.9f}|" for r in full_rows)
    (HERE/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=result['complete'],rows=table),indent=2))


if __name__=='__main__':main()
