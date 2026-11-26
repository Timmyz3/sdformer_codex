#!/usr/bin/env python3
"""Summarize complete fixed32 breadth cases; no ranking or rejection criteria."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def require(ok,message):
    if not ok:raise ValueError(message)


def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def mean(values):return sum(values)/len(values) if values else None


def regions_for(frames):
    keys=list(frames[0]['regions'])
    require(all(set(f['regions'])==set(keys) for f in frames),'Region definitions differ within case')
    out={}
    for region in keys:
        parts=[f['regions'][region] for f in frames]
        for p in parts:
            require(type(p.get('valid_pixels')) is int and p['valid_pixels']>=0 and finite(p.get('error_sum')) and p['error_sum']>=0,
                    'Invalid region counts/error_sum')
            if p['valid_pixels']:
                require(finite(p.get('AEE')) and math.isclose(p['AEE'],p['error_sum']/p['valid_pixels'],rel_tol=1e-10,abs_tol=1e-10),
                        'Region AEE differs from sum/count')
            else:require(p.get('AEE') is None and p['error_sum']==0,'Empty region must be null, not zero AEE')
        values=[p['AEE'] for p in parts if p['valid_pixels']];pixels=sum(p['valid_pixels'] for p in parts)
        out[region]={'frames_with_region':len(values),'empty_region_frames':len(parts)-len(values),'valid_pixels':pixels,
            'error_sum':sum(p['error_sum'] for p in parts),'frame_mean_AEE':mean(values),
            'pixel_weighted_AEE':sum(p['error_sum'] for p in parts)/pixels if pixels else None}
    require(out['all_valid']['frames_with_region']==len(frames),'All-valid frames cannot be empty')
    return out


def pair_regions(frames,baseline):
    reference={f['target_id']:f for f in baseline};out={}
    for region in frames[0]['regions']:
        deltas=[];sum_difference=0.;pixels=0
        for f in frames:
            require(f['target_id'] in reference,'Pair target missing')
            a=f['regions'][region];b=reference[f['target_id']]['regions'].get(region)
            require(b is not None and a['valid_pixels']==b['valid_pixels'],'Paired GT/region counts differ')
            if a['valid_pixels']:
                deltas.append(a['AEE']-b['AEE']);pixels+=a['valid_pixels'];sum_difference+=a['error_sum']-b['error_sum']
        out[region]={'paired_nonempty_frames':len(deltas),'frame_mean_delta_AEE':mean(deltas),
                     'pixel_weighted_delta_AEE':sum_difference/pixels if pixels else None}
    return out


def execution_probes(arms):
    actual=[];factor=[];allkeep=[]
    for case_id,entry in arms.items():
        if 'all_keep_equivalence' in entry:allkeep.append({'case_id':case_id,**entry['all_keep_equivalence']})
        for frame in entry.get('frames',[]):
            pair=frame.get('actual_skip_pair')
            if pair:
                row={'case_id':case_id,'target_id':frame['target_id'],'max_abs_error':pair['max_abs_error'],
                    'allclose':pair['allclose_atol1e-5_rtol1e-5'],'finite':pair['finite'],
                    'same_tile_masks':pair['same_tile_masks'],'seconds':pair['seconds']}
                for key in ('fc1_rows','fc2_rows'):
                    row['actual_'+key]=sum(x['observed'][key] for x in pair['tile_execution'])
                    row['regular_'+key]=sum(x['observed'][key] for x in frame.get('tile_execution',[]))
                actual.append(row)
            for probe in frame.get('tp_factorization_probe',[]):
                c=probe['comparison']
                factor.append({'case_id':case_id,'target_id':frame['target_id'],'layer_name':probe['layer_name'],
                    'score_max_abs':c['score_max_abs'],'score_bitwise_equal':c['score_bitwise_equal'],
                    'score_allclose':c['score_allclose_atol2e_6_rtol2e_5'],
                    'tile_rank_different_positions':c['tile_rank_different_positions'],'seconds':probe['seconds'],
                    'different_tiles_by_ratio':{str(r['keep_fraction']):r['different_tiles'] for r in c['masks']},
                    'different_pixels_by_ratio':{str(r['keep_fraction']):r['different_real_spatial_pixels'] for r in c['masks']}})
    return {'all_keep':{'checks':len(allkeep),'max_abs_error':max((r['max_abs_error'] for r in allkeep),default=None),
                'bitwise_equal_checks':sum(r['bitwise_equal'] for r in allkeep),'records':allkeep},
        'actual_skip':{'paired_frames':len(actual),'allclose_frames':sum(r['allclose'] for r in actual),
            'same_mask_frames':sum(r['same_tile_masks'] for r in actual),'finite_frames':sum(r['finite'] for r in actual),
            'max_abs_error':max((r['max_abs_error'] for r in actual),default=None),'records':actual,
            'scope':'Observed FC rows and numerical pairing only; Python/GPU wall time is not accelerator cycles'},
        'TP_factorization':{'layer_sample_checks':len(factor),'allclose_checks':sum(r['score_allclose'] for r in factor),
            'bitwise_score_checks':sum(r['score_bitwise_equal'] for r in factor),
            'max_score_abs_error':max((r['score_max_abs'] for r in factor),default=None),
            'checks_with_any_changed_tile_mask':sum(any(r['different_tiles_by_ratio'].values()) for r in factor),
            'records':factor,'scope':'Actual incoming u; score closeness does not imply identical ranking or masks'}}


def summarize(value,source_sha):
    require(value.get('schema')=='flow_breadth_exploration_v1','Wrong breadth schema')
    targets=value['population']['target_ids']
    require(len(targets)==len(set(targets))==32 and value['population']['population']=='train_gate_development','Expected fixed train gate-dev32')
    complete={};incomplete=[];errors=[]
    for case in value['cases']:
        name=case['id'];entry=value.get('arms',{}).get(name)
        if entry is None:incomplete.append({'case_id':name,'status':'NOT_EXECUTED'});continue
        if entry.get('status')!='PASS':
            incomplete.append({'case_id':name,'status':entry.get('status'),'frames':len(entry.get('frames',[])),'error':entry.get('error')});continue
        try:
            frames=entry['frames'];require([f['target_id'] for f in frames]==targets,'PASS does not contain exactly fixed32 in declared order')
            require(entry.get('BN_buffers_unchanged') is True,'PASS lacks unchanged BN proof')
            calculated=regions_for(frames)
            # Refuse to silently propagate stale/mismatched stored aggregate evidence.
            stored=entry.get('summary',{}).get('regions',{})
            for name_r,r in calculated.items():
                require(stored.get(name_r,{}).get('valid_pixels')==r['valid_pixels'],'Stored summary count differs')
                for key in ('frame_mean_AEE','pixel_weighted_AEE'):
                    old=stored[name_r].get(key);new=r[key]
                    require((old is None and new is None) or (finite(old) and finite(new) and math.isclose(old,new,rel_tol=1e-10,abs_tol=1e-10)),
                            'Stored summary differs from per-frame evidence')
            complete[name]={'case':case,'regions':calculated,'frames':frames,'entry':entry}
        except (ValueError,KeyError,TypeError) as exc:errors.append({'case_id':name,'error':str(exc)})
    baseline=complete.get('dense_D');rows=[];region_rows=[];same_bn_baselines={}
    for bn in ('P_paper','P_code','D'):
        candidates=[name for name,item in complete.items() if item['case']['method']=='dense' and item['case']['bn']==bn]
        preferred='dense_'+bn
        if preferred in candidates:same_bn_baselines[bn]=preferred
        elif len(candidates)==1:same_bn_baselines[bn]=candidates[0]
    for name,item in complete.items():
        case,entry,stats=item['case'],item['entry'],item['regions'];paired=None
        if baseline:
            try:paired=pair_regions(item['frames'],baseline['frames'])
            except (ValueError,KeyError) as exc:errors.append({'case_id':name,'pair_error':str(exc)})
        same_bn_id=same_bn_baselines.get(case['bn']);same_bn_paired=None
        if same_bn_id:
            try:same_bn_paired=pair_regions(item['frames'],complete[same_bn_id]['frames'])
            except (ValueError,KeyError) as exc:errors.append({'case_id':name,'same_BN_pair_error':str(exc)})
        allv=stats['all_valid'];delta=paired['all_valid'] if paired else {}
        same_bn_delta=same_bn_paired['all_valid'] if same_bn_paired else {}
        row={'case_id':name,'bn':case['bn'],'method':case['method'],'policy_or_static':case.get('policy',case.get('static_method','dense')),
             'keep_fraction':case.get('keep_fraction',case.get('keep_ratio')),'bits':case.get('bits'),'frames':32,
             'frame_mean_AEE':allv['frame_mean_AEE'],'pixel_weighted_AEE':allv['pixel_weighted_AEE'],
             'paired_frame_delta_AEE_vs_dense_D':delta.get('frame_mean_delta_AEE'),
             'paired_pixel_delta_AEE_vs_dense_D':delta.get('pixel_weighted_delta_AEE'),
             'same_bn_baseline_id':same_bn_id,
             'delta_same_bn_frame_AEE':same_bn_delta.get('frame_mean_delta_AEE'),
             'delta_same_bn_pixel_AEE':same_bn_delta.get('pixel_weighted_delta_AEE'),
             'valid_pixels':allv['valid_pixels'],'wall_seconds':entry.get('wall_seconds_including_build_data_metrics'),
             'max_allocated_bytes':entry.get('max_allocated_bytes'),'max_reserved_bytes':entry.get('max_reserved_bytes'),
             'parameter_count':entry.get('parameter_count')}
        rows.append(row)
        for name_r,r in stats.items():region_rows.append({'case_id':name,'bn':case['bn'],'region':name_r,**r,
            'paired_frame_delta_AEE_vs_dense_D':paired[name_r]['frame_mean_delta_AEE'] if paired else None,
            'paired_pixel_delta_AEE_vs_dense_D':paired[name_r]['pixel_weighted_delta_AEE'] if paired else None,
            'same_bn_baseline_id':same_bn_id,
            'delta_same_bn_frame_AEE':same_bn_paired[name_r]['frame_mean_delta_AEE'] if same_bn_paired else None,
            'delta_same_bn_pixel_AEE':same_bn_paired[name_r]['pixel_weighted_delta_AEE'] if same_bn_paired else None})
    return {'schema':'flow_breadth_summary_v1','input_sha256':source_sha,'input_status':value.get('status'),
        'checkpoint_sha256':value['checkpoint_sha256'],'checkpoint_step':value['checkpoint_sidecar']['completed_optimizer_steps'],
        'geometry':value['geometry'],'target_ids':targets,'target_ids_sha256':value['population']['target_ids_sha256'],
        'case_order':'Original manifest order, not ranked','completed_cases':len(rows),'incomplete_or_pending':incomplete,
        'evidence_errors':errors,'dense_D_complete':baseline is not None,'same_bn_baselines':same_bn_baselines,'cases':rows,'regions':region_rows,
        'execution_probes':execution_probes(value.get('arms',{})),
        'interpretation':'Interim canonical6000 weights, train-side32, zero recovery unless separately documented. No paper reproduction, final-valid, method rejection, hardware speedup or causal flow-specific claim.'}


def fmt(x):return '—' if x is None else f'{x:.6f}' if isinstance(x,float) else str(x)


def write_csv(path,rows):
    if not rows:path.write_text('');return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def render(result):
    p=result['execution_probes'];lines=['# 训练侧广度探索汇总','',result['interpretation'],'',
        f"Checkpoint step={result['checkpoint_step']}；geometry={result['geometry']}；完整 case={result['completed_cases']}；未完成/待准备={len(result['incomplete_or_pending'])}。",
        '只纳入 PASS、固定32成员及逐帧记录完整的 case；按声明顺序列出，不排名。ΔAEE 与同一 target 的 dense_D 配对；P-paper 的差值同时包含 BN 语义差异，不能归因于压缩。','',
        '压缩质量的主配对列是 ΔAEE vs same BN，使用本次运行中相同 BN 语义的 dense 基线；旧 ΔvsD 保留为诊断，跨 BN 时包含统计语义差异。','',
        '| Case | 全GT逐帧AEE | 像素加权AEE | ΔAEE vs same BN | ΔAEE vs dense_D | 秒 |','|---|---:|---:|---:|---:|---:|']
    for r in result['cases']:lines.append('| '+ ' | '.join([r['case_id'],fmt(r['frame_mean_AEE']),fmt(r['pixel_weighted_AEE']),fmt(r['delta_same_bn_frame_AEE']),fmt(r['paired_frame_delta_AEE_vs_dense_D']),fmt(r['wall_seconds'])])+' |')
    lines+=['', '所有分区（B、L、B_unknown、其余已知区域、voxel_zero/nonzero 等）的像素数、空帧数和配对差值见 regions.csv。没有将无事件/平坦区域从全部有效GT指标中排除。','',
        f"全开：{p['all_keep']['checks']} 次，max abs={fmt(p['all_keep']['max_abs_error'])}。",
        f"actual_skip：{p['actual_skip']['paired_frames']} 个配对帧，allclose={p['actual_skip']['allclose_frames']}，mask相同={p['actual_skip']['same_mask_frames']}，max abs={fmt(p['actual_skip']['max_abs_error'])}。",
        f"TP 实际u：{p['TP_factorization']['layer_sample_checks']} 个层/样本检查，allclose={p['TP_factorization']['allclose_checks']}，存在tile mask变化={p['TP_factorization']['checks_with_any_changed_tile_mask']}，score max abs={fmt(p['TP_factorization']['max_score_abs_error'])}。",
        '', '执行行数、mask变化、TP排序差异详见 summary.json。GPU/Python 时间不等于加速器周期；假量化仍是FP32，静态方法未恢复训练，不能据此宣判方法失败。']
    if result['incomplete_or_pending']:
        lines+=['','未完成/待准备：']+['- '+r['case_id']+': '+str(r['status']) for r in result['incomplete_or_pending']]
    if result['evidence_errors']:lines+=['','证据不一致（已排除对应完整统计或配对）：',json.dumps(result['evidence_errors'],ensure_ascii=False)]
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--input-sha256',required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    require(not a.output_dir.exists(),'Use new output directory; no report overwrite')
    raw=a.input.read_bytes();checksum=hashlib.sha256(raw).hexdigest();require(checksum==a.input_sha256,'Input result SHA differs')
    result=summarize(json.loads(raw),checksum);result['summary_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();a.output_dir.mkdir(parents=True)
    (a.output_dir/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_csv(a.output_dir/'cases.csv',result['cases']);write_csv(a.output_dir/'regions.csv',result['regions'])
    (a.output_dir/'SUMMARY.md').write_text(render(result))
    print(json.dumps({'complete_cases':result['completed_cases'],'evidence_errors':result['evidence_errors']}))


if __name__=='__main__':main()
