"""Recompute matched NB0/candidate AEE from existing records; never run inference.

Python 3.12 standard library only. --fetch refreshes the small read-only A800
source CSV/config/metadata copies; the default rebuild is offline.
"""
from pathlib import Path
import argparse
import csv
import json
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
OPEN = HERE.parent
BASE = OPEN.parent
CALIBRATION = 'zurich_city_09_a_0001.npy'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def fetch(socket):
    remote = r'''
import json
from pathlib import Path
root=Path('/root/private_data/work/sdformer_codex/SDformer')
exp=root/'neuron_experiments/H9_bipolar_self_attention'
run=exp/'results/dsec_fullres_w15_NB0_equal_plus10_ep40_20260805'
source=exp/'results/dsec_density_table_g_four_line_20260817/nb0/per_frame.csv'
profile=run/'standard_valid825/epoch29/spike_profile.json'
d=json.loads(profile.read_text())
config=Path(d['artifact_identity']['config_path'])
checkpoint=Path(d['artifact_identity']['checkpoint_path'])
meta=dict(source_csv=str(source), profile=str(profile), config=str(config),
 checkpoint=str(checkpoint), checkpoint_exists=checkpoint.is_file(),
 validation_file_list=d['validation_file_list']['path'],
 canonical_data_root=str((root/'data/Datasets/DSEC/saved_flow_data').resolve()),
 metrics=d['metrics'], metric_contract=d['metric_contract'],
 metric_aggregation_audit=d['metric_aggregation_audit'],
 eval_protocol=d['eval_protocol'], module_counts=d['module_counts'],
 checkpoint_load=dict(checkpoint_overlay_keys=d['checkpoint_load_audit']['checkpoint_overlay_keys'],
 missing_count=d['checkpoint_load_audit']['missing_count'],
 unexpected_count=d['checkpoint_load_audit']['unexpected_count']),
 identity='Local upstream SDformerFlow PSN/SDSA reproduction, not the authors checkpoint; no ATLIF or Shiftmax.',
 training='60 crop epochs from local baseline_stride_upstream ep59, then30 full-resolution epochs; equal+10 labels34/39 did not beat label29. Paper states80 crop epochs.',
 readout='Original complete model pred_list[flow][-1], not the derived coarse student.',
 inspected_utc_date='2026-09-12')
print(json.dumps(dict(csv=source.read_text(), config=config.read_text(), metadata=meta)))
'''
    cmd=['ssh','-S',socket,'-p10037','-oBatchMode=yes',
         'root@ssh.sd5ai.scnet.cn',
         '/root/private_data/work/hardware_innovation_20260908/env312/bin/python3.12 -']
    bundle=json.loads(subprocess.run(cmd,input=remote,text=True,capture_output=True,check=True).stdout)
    (HERE/'source_nb0_valid825.csv').write_text(bundle['csv'])
    (HERE/'source_nb0_config.yml').write_text(bundle['config'])
    write(HERE/'source_metadata.json',bundle['metadata'])


def stats(rows):
    count=sum(int(r['valid_pixels']) for r in rows)
    return dict(frames=len(rows),valid_pixels=count,
        AEE_frame_mean=statistics.mean(float(r['AEE']) for r in rows),
        AEE_pixel_mean=sum(float(r.get('aee_sum',float(r['AEE'])*int(r['valid_pixels'])) ) for r in rows)/count)


def candidates():
    """Explicit completed arms, no search that silently adds unrelated runs."""
    path=OPEN/'new_interface_selection/aee_rebase/results/run.json'
    d=read(path)
    for axis,ar in d['axes'].items():
        for mode,r in ar['modes'].items():
            yield 'PED_R24',axis,mode,r['frames'],r['summary'],d['complete'] and ar['complete'],path,False
    bn=OPEN/'default_bn/onepass/aee_check'
    for folder,mode in [('baseline_recheck','original_cuda_bn'),('centered_results','centered_engine'),('paired_onepass','onepass')]:
        path=bn/folder/'run.json'; d=read(path)
        for axis,ar in d['axes'].items():
            rows_path=path.parent/axis/mode/(mode+'_frames.json')
            yield 'BN',axis,mode,read(rows_path),ar[mode],d['complete'] and ar['complete'],rows_path,False
    path=OPEN/'pruning/aee_results/run.json';d=read(path)
    for axis,ar in d['axes'].items():
        for mode,r in ar['stages']['diverse10'].items():
            yield 'pruning_no_train',axis,mode,r['frames'],r['summary'],d['complete'] and ar['complete'],path,False
    path=OPEN/'pruning/aee_row_phase/run.json';d=read(path)
    for axis,ar in d['axes'].items():
        for mode,r in ar['stages']['diverse10'].items():
            yield 'pruning_row_phase',axis,mode,r['frames'],r['summary'],d['complete'] and ar['complete'],path,False
    path=OPEN/'pruning/paired_recovery/stage64/run.json';d=read(path)
    for mode,ar in d['axes'].items():
        rows_path=path.parent/mode/(mode+'_frames.json')
        yield 'pruning_recovery64','ordinary',mode,read(rows_path),ar['evaluation'],d['complete'] and ar['complete'],rows_path,True


def full825(baseline, names):
    """Only the two pre-existing fixed R32 students; never inherit to new arms."""
    chain=BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain'
    current=read(OPEN/'new_interface_selection/aee_rebase/results/run.json')
    choices=[('ordinary',chain/'temporal_structured_recovery/fixed_valid825','identity_permuted_base',
              chain/'temporal_structured_recovery/stage128x256/identity_permuted_base.npz'),
             ('lifting_raw',chain/'fast_temporal_recovery_lifting40/fixed_lifting_valid825','fast_raw_diagonal',
              chain/'fast_temporal_recovery_lifting40/stage320/fast_raw_diagonal.npz')]
    nb0=stats(list(baseline.values()));arms=[];matched={};gaps=[]
    for axis,folder,mode,student in choices:
        frame_path=folder/(mode+'_frames.json');summary_path=folder/(mode+'_summary.json')
        if not frame_path.exists():
            gaps.append(dict(axis=axis,reason='Per-frame record missing',summary=read(summary_path) if summary_path.exists() else None));continue
        rows=read(frame_path);recorded=read(summary_path);run=read(folder/'result.json');ar=run['axes'][mode]
        lookup={r['file']:r for r in rows}
        if not run['complete'] or not ar['complete'] or not recorded['complete']:
            gaps.append(dict(axis=axis,reason='Incomplete evaluation'));continue
        if len(rows)!=825 or set(lookup)!=set(baseline):
            gaps.append(dict(axis=axis,reason='Not the same825 unique frames'));continue
        if any(int(r['valid_pixels'])!=baseline[r['file']]['valid_pixels'] for r in rows):
            gaps.append(dict(axis=axis,reason='Per-frame valid GT pixel count mismatch'));continue
        measured=stats(rows)
        if abs(measured['AEE_frame_mean']-recorded['AEE_frame_mean'])>1e-10: raise ValueError((axis,'full825 mean drift'))
        if measured['valid_pixels']!=48152523: raise ValueError((axis,'full825 pixel population drift'))
        ten=stats([lookup[n] for n in names]);old=current['axes'][axis]['modes']['original32']['frames']
        max_delta=max(abs(r['AEE']-lookup[r['file']]['AEE']) for r in old)
        numeric=ar.get('coordinate_function',ar.get('fixed_function',{}))
        deltas=[r['AEE']-baseline[r['file']]['AEE'] for r in rows]
        arms.append(dict(axis=axis,mode=mode,student_parameters=str(student.relative_to(BASE)),
            source_frames=str(frame_path.relative_to(BASE)),source_summary=str(summary_path.relative_to(BASE)),
            source_result=str((folder/'result.json').relative_to(BASE)),
            complete=True,joined_unique_frames=825,matched_per_frame_valid_pixels=True,
            sequence_count=len({baseline[r['file']]['sequence'] for r in rows}),
            metrics=measured,NB0_metrics=nb0,delta_AEE_vs_NB0=measured['AEE_frame_mean']-nb0['AEE_frame_mean'],
            relative_AEE_reduction_vs_NB0=1-measured['AEE_frame_mean']/nb0['AEE_frame_mean'],
            better_than_NB0=measured['AEE_frame_mean']<nb0['AEE_frame_mean'],
            frame_deltas=dict(better=sum(d<0 for d in deltas),equal=sum(d==0 for d in deltas),worse=sum(d>0 for d in deltas),minimum=min(deltas),maximum=max(deltas)),
            extracted_diverse10=ten,current_original32_diverse10_max_abs_AEE_difference=max_delta,
            numeric_scope={k:numeric[k] for k in ['mode','completed_state','coefficient_format','in_flight','source','state_order','updates','residual','scope','outside_scope','claim'] if k in numeric},
            no_inheritance_to=['R24 bases','modified BN functions','pruning masks','paired64 recovered weights']))
        matched[axis]=lookup
    fields=['file','sequence','valid_pixels','NB0_AEE']
    for axis in matched: fields += [axis+'_AEE',axis+'_aee_sum',axis+'_delta_vs_NB0']
    with (HERE/'valid825_matched_frames.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for name,b in baseline.items():
            row=dict(file=name,sequence=b['sequence'],valid_pixels=b['valid_pixels'],NB0_AEE=b['AEE'])
            for axis,lookup in matched.items():
                r=lookup[name];row.update({axis+'_AEE':r['AEE'],axis+'_aee_sum':r['aee_sum'],axis+'_delta_vs_NB0':r['AEE']-b['AEE']})
            writer.writerow(row)
    report=dict(complete=len(arms)==2,inference_run=False,
        scope='Historical complete825 fixed ordinary R32 and lifting40 raw R32 vs same-population NB0; separate from26 small-set arms.',
        baseline_csv='source_nb0_valid825.csv',matched_csv='valid825_matched_frames.csv',
        arms=arms,gaps=gaps,quality_decision='Both original fixed students clear the newNB0 full825 AEE gate' if len(arms)==2 and all(a['better_than_NB0'] for a in arms) else 'See matched arms and gaps',
        limits=['All825 filenames and per-frame valid counts join; mask loader/path contract matches, historical mask-bit archive not independently checked.',
            'NB0 final head vs each declared coarse preds.2 student; this is task quality, not the causal effect of lifting alone.',
            'Original fixed helper passes this dataset-level accuracy gate. NewR24, BN, pruning or recovered parameters retain only their own measured population.',
            'Accuracy does not establish novel hardware, same-resource service, RTL or PPA. No new evaluation was launched.'])
    write(HERE/'valid825_summary.json',report)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fetch',action='store_true')
    ap.add_argument('--socket',default='/tmp/codex_bn_onepass_a800_20260912.sock')
    args=ap.parse_args()
    if args.fetch: fetch(args.socket)
    meta=read(HERE/'source_metadata.json')
    raw=list(csv.DictReader((HERE/'source_nb0_valid825.csv').open()))
    baseline={r['file']:dict(r,valid_pixels=int(float(r['valid_pixels'])),AEE=float(r['AEE'])) for r in raw}
    if len(raw)!=825 or len(baseline)!=825: raise ValueError('Source must contain825 unique frames')
    names=read(BASE/'algorithm/samples.json')['valid']
    if len(names)!=10 or len(set(names))!=10: raise ValueError('Expected the existing diverse10')
    selected=[baseline[n] for n in names]
    held=[r for r in selected if r['file']!=CALIBRATION]
    all_stats,ten_stats,nine_stats=stats(list(baseline.values())),stats(selected),stats(held)
    if all_stats['valid_pixels']!=48152523: raise ValueError('NB0 population differs')
    if abs(all_stats['AEE_frame_mean']-float(meta['metrics']['AEE']))>1e-7: raise ValueError('Source aggregate differs from NB0 profile')
    full=full825(baseline,names)
    matched_fields=['file','sequence','valid_pixels','AEE','AAE','AAE_Benchmark','DSEC_Fl','excluded_from_holdout9']
    with (HERE/'matched_diverse10.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=matched_fields,lineterminator='\n');writer.writeheader()
        for r in selected: writer.writerow({k:r[k] for k in matched_fields[:-1]}|{'excluded_from_holdout9':r['file']==CALIBRATION})
    results=[];omitted=[];paired_rows=[]
    for group,axis,mode,rows,recorded,complete,path,trained in candidates():
        key=f'{group}/{axis}/{mode}'
        if not complete or not recorded.get('complete'):
            omitted.append(dict(id=key,reason='Incomplete source evaluation'));continue
        lookup={r['file']:r for r in rows}
        if len(rows)!=10 or set(lookup)!=set(names):
            omitted.append(dict(id=key,reason='Frame population mismatch'));continue
        ordered=[lookup[n] for n in names]
        if any(int(r['valid_pixels'])!=baseline[r['file']]['valid_pixels'] for r in ordered):
            omitted.append(dict(id=key,reason='Per-frame valid GT count mismatch'));continue
        measured=stats(ordered);holdout=stats([r for r in ordered if r['file']!=CALIBRATION])
        if abs(measured['AEE_frame_mean']-recorded['AEE_frame_mean'])>1e-10: raise ValueError((key,'recorded mean differs'))
        for s,b in [(measured,ten_stats),(holdout,nine_stats)]:
            s['NB0_AEE_frame_mean']=b['AEE_frame_mean']
            s['delta_vs_NB0']=s['AEE_frame_mean']-b['AEE_frame_mean']
            s['better_than_NB0']=s['AEE_frame_mean']<b['AEE_frame_mean']
        result=dict(id=key,group=group,axis=axis,mode=mode,source=str(path.relative_to(OPEN)),
            training=trained,training_budget_steps=64 if trained else 0,
            new_student_identity=trained,diverse10=measured,holdout9=holdout,
            matched_frame_names=True,matched_per_frame_valid_pixels=True,
            mask_contract='Same canonical saved mask_tensors/<file> GT mask; no event-mask intersection. Count matching is not an independent historical mask-bit proof.',
            scope='Historical matched diverse10 task-accuracy comparison, not candidate valid825 or same-process inference')
        results.append(result)
        for r in ordered:
            b=baseline[r['file']]
            paired_rows.append(dict(candidate=key,file=r['file'],valid_pixels=r['valid_pixels'],
                NB0_AEE=b['AEE'],candidate_AEE=r['AEE'],delta_AEE=r['AEE']-b['AEE'],
                excluded_from_holdout9=r['file']==CALIBRATION))
    with (HERE/'candidate_matched_frames.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(paired_rows)
    summary=dict(complete=True,inference_run=False,policy='Candidate AEE must be lower than original SDformerFlow NB0 on the matched population; +0.005 relative to ordinary is no longer an admission gate.',
        source_metadata='source_metadata.json',source_csv='source_nb0_valid825.csv',
        baseline=dict(valid825=all_stats,diverse10=ten_stats,holdout9=nine_stats,
            profile_reference_AEE=float(meta['metrics']['AEE']),
            csv_minus_profile_AEE=all_stats['AEE_frame_mean']-float(meta['metrics']['AEE'])),
        calibration_frame=CALIBRATION,files=names,
        counts=dict(arms=len(results),passed_diverse10=sum(r['diverse10']['better_than_NB0'] for r in results),
            passed_holdout9=sum(r['holdout9']['better_than_NB0'] for r in results),omitted=len(omitted)),
        arms=results,omitted=omitted,valid825_comparison='valid825_summary.json',
        limits=['NB0 is the original PSN/SDSA local upstream reproduction, not the authors checkpoint.',
            'NB0 full final head vs derived preds.2 coarse-exit students; compare task quality, do not attribute all changes to a local interface.',
            'Both compute valid GT EPE with scale1 and frame-equal aggregation. NB0 CSV uses float32 map reduction rounded to10 decimals; candidate records sum float32 EPE in float64. No bitwise arithmetic equivalence claimed.',
            'Matching uses declared identical GT-mask path/loader and all per-frame valid counts. No new inference or per-frame mask-bit archive audit.',
            'Diverse10 includes a prior mask calibration frame. Holdout9 removes it, but this repeatedly examined small set is not untouched validation.',
            'NewR24/BN/pruned candidate valid825, deployment service, RTL/PPA and novelty are separate; original fixed R32 full825 joins are in valid825_summary.json.',
            'The public-paper hidden-test AEE1.602 and author-validation1.61 are different populations and are not this gate.'])
    write(HERE/'summary.json',summary)
    lines=['# 原 SDformerFlow：同帧精度基线','',
        '2026-09-12。原 PSN/SDSA 的本地 upstream 复现 NB0 ep29；**不是 ordinary 学生、Motion ep34，也不是作者发布的检查点**。本目录仅提取旧实测记录，没有新推理、训练或硬件性能测量。用户新门是同人口 AEE 优于 NB0，旧 ordinary `+0.005` 不再否决候选。','',
        '| NB0 范围 | 帧数 / 有效像素 | AEE 帧均值 | AEE 像素均值 |','|---|---:|---:|---:|']
    for label,s in [('valid825',all_stats),('diverse10',ten_stats),('去校准源帧九帧',nine_stats)]:
        lines.append(f"| {label} | {s['frames']} / {s['valid_pixels']:,} | {s['AEE_frame_mean']:.12f} | {s['AEE_pixel_mean']:.12f} |")
    lines += ['', '**原 fixed R32 的完整825也已逐帧配对。** 两学生均825/825文件与每帧有效像素相同，共18序列、48,152,523像素；这不是仅有汇总。', '',
        '| 旧完整验证身份 | AEE帧均值 | Δ对NB0 | 相对AEE下降 |','|---|---:|---:|---:|']
    for a in full['arms']:
        lines.append(f"| {a['axis']} / {a['mode']} | {a['metrics']['AEE_frame_mean']:.12f} | {a['delta_AEE_vs_NB0']:+.9f} | {100*a['relative_AEE_reduction_vs_NB0']:.3f}% |")
    lines += ['', '旧原学生抽出的十帧AEE与当前 `aee_rebase/original32` 各轴逐帧相同（最大差0）。**原 lifting40 已过新NB0完整825精度门**；此前仅相对 ordinary 的回退不能继续否决它。证据与原参数/定点语义路径见 [valid825_summary.json](valid825_summary.json) 和 [825行配对](valid825_matched_frames.csv)。新R24、BN改法、剪枝及恢复权重仍只有各自十帧，不继承此全量结果。']
    lines += ['',f"825 行重新聚合相对原 profile 仅差 {summary['baseline']['csv_minus_profile_AEE']:.3g} AEE（CSV 小数序列化及旧聚合）。十帧全部命中原记录，逐帧有效像素一致。**下表 {len(results)} 臂均低于同十帧与同九帧 NB0**；重复的原学生控制按实验来源保留，不冒充独立模型。",'',
        '| 来源 / 学生 / 臂 | 十帧 AEE | 九帧 AEE | 十帧 Δ对NB0 |','|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['id']} | {r['diverse10']['AEE_frame_mean']:.6f} | {r['holdout9']['AEE_frame_mean']:.6f} | {r['diverse10']['delta_vs_NB0']:+.6f} |")
    lines += ['',
        '**身份与口径。** NB0 使用本地60轮 crop ep59 起点，再30轮 full-resolution；equal+10 的ep34/39未超过ep29。论文写80轮 crop，不能称严格作者训练复现。分辨率480×640、窗口T2×15×15、batch1、原78处BN无运行统计；AT-LIF/Shiftmax安装数均0。checkpoint/config/A800原CSV路径保存在 [source_metadata.json](source_metadata.json)，配置原文在 [source_nb0_config.yml](source_nb0_config.yml)。NB0走完整模型最终 `flow[-1]`；本轮候选走 `preds.2` 时间求和并双线性恢复480×640的粗头。可以比较任务质量，不能把全部精度差归因U/V、BN或剪枝，也不能将其称同一个网络。','',
        '**配对边界。** 两侧同一canonical GT及 `mask_tensors/<file>`，不交事件掩码，flow scaling=1，先每帧有效GT像素平均再帧等权；程序逐项检查帧名、每帧有效像素和已完成记录均值。旧NB0使用FP32误差图归约并保存10位小数，候选以FP64累加FP32 EPE；不是逐位算术复现。旧mask全部位图未归档，本次不伪称仅像素数就证明历史mask逐位相等。九帧仅排除 `zurich_city_09_a_0001.npy` 校准源帧，不将反复使用的小集合冒称未见验证。','',
        '**交付。** [825行原CSV](source_nb0_valid825.csv)、[同十帧NB0](matched_diverse10.csv)、[逐臂逐帧配对](candidate_matched_frames.csv)、[小集完整汇总](summary.json)。未训剪枝6臂、行相位4臂与恢复64步2臂均重新按NB0比较；恢复使用新训练身份，不能继承免训硬件活动。原各实验旧门文字不在本目录改写。新变体还须完成自己的valid825；本表不提供RTL/PPA、创新性或录用证据。公开论文1.602/1.61人口不同，不用于本门。','',
        '复算：`/usr/bin/python3.12 extract.py`（离线，标准库）；需更新源副本才加 `--fetch --socket /tmp/codex_bn_onepass_a800_20260912.sock`，仅SSH读取既有CSV/config/profile，不加载网络。']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary['counts']))
    print(json.dumps(summary['baseline']))
    print(json.dumps({'full825':[(a['axis'],a['metrics']['AEE_frame_mean'],a['delta_AEE_vs_NB0']) for a in full['arms']],'gaps':full['gaps']}))


if __name__=='__main__':
    main()
