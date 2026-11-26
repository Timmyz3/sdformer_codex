#!/usr/bin/env python3
"""Frozen exploratory analysis, including two-sequence leave-one-sequence-out.

No fresh model execution. k1 selections index already executed singleton flows.
Ridge protocol fixed before reading probe GT: five input features, alpha=1,
train-fold standardization, two targets; includes a learned position-only control.
Reused dev8 is not an independent validation set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

FEATURES=('amplitude_abs_mean','feature_rms','pooling_residual_mse',
          'spatial_neighbor_abs_difference','temporal_neighbor_abs_difference')
SCORES=FEATURES+('psn1_activity','psn1_spatial_neighbor_hamming','offline_branch_mse')
BASE='coarse_stage02'
TILES=tuple((r,c) for r in range(3) for c in range(4))


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x,m):
    if not x: raise ValueError(m)
def arm(tile): return 'fine_r%d_c%d'%tile
def reg(frame,name,region='all_valid'): return frame['arms'][name]['metrics']['regions'][region]
def gain(frame,tile): return reg(frame,BASE)['AEE']-reg(frame,arm(tile))['AEE']


def ranks(a):
    a=np.asarray(a); order=np.argsort(a,kind='stable'); out=np.zeros(len(a)); start=0
    while start<len(a):
        end=start+1
        while end<len(a) and a[order[end]]==a[order[start]]: end+=1
        out[order[start:end]]=(start+end-1)/2.; start=end
    return out


def spearman(x,y):
    a=ranks(x); b=ranks(y); a-=a.mean(); b-=b.mean()
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/den) if den else None


def choose(values): return min(TILES,key=lambda t:(-float(values[TILES.index(t)]),t))


def ridge_fit_predict(x,y,test):
    mean=x.mean(0); std=x.std(0); std=np.where(std>1e-12,std,1.)
    z=(x-mean)/std; yy=y-y.mean()
    beta=np.linalg.solve(z.T@z+np.eye(x.shape[1]),z.T@yy)
    pred=(test-mean)/std@beta+y.mean()
    return pred,{'alpha':1.,'standardization_mean':mean.tolist(),'standardization_std':std.tolist(),
                 'coefficient':beta.tolist(),'intercept_target_mean':float(y.mean()),
                 'target_scaling':'center only; standardization only uses training fold'}


def summarize(frames,choices):
    regions=tuple(frames[0]['arms'][BASE]['metrics']['regions'])
    result={}
    for region in regions:
        vals=[reg(f,choices[f['target_id']],region) for f in frames]
        nonempty=[x for x in vals if x['valid_pixels']>0]
        n=sum(x['valid_pixels'] for x in vals); err=sum(x['error_sum'] for x in vals)
        result[region]={'frames_with_region':len(nonempty),'valid_pixels':n,'error_sum':err,
                        'frame_mean_AEE':float(np.mean([x['AEE'] for x in nonempty])) if nonempty else None,
                        'pixel_weighted_AEE':err/n if n else None}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--input-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    require(sha(a.input)==a.input_sha256,'Input SHA differs')
    require(not a.output_dir.exists(),'New output directory required')
    raw=json.loads(a.input.read_text());require(raw['status']=='TILE_INTERVENTIONS_DEV8_COMPLETE','Complete input required')
    frames=raw['frames'];require(len(frames)==8,'Fixed dev8 required')
    seqs=sorted(set(f['sequence'] for f in frames));require(len(seqs)==2,'Exactly two exploratory sequences')
    data={};choices={key:{} for key in ('coarse_baseline','posthoc_GT_best_singleton',*SCORES,
                    'LOSO_task_ridge','LOSO_branch_mse_ridge','LOSO_position_task_mean')}
    frame_diagnostics=[]
    for f in frames:
        ordinary={tuple(x['tile']):x for x in f['ordinary_scores']}
        branch={tuple(x['tile']):x['MSE'] for x in f['offline_fine_coarse_branch_mse']}
        y=np.array([gain(f,t) for t in TILES]);base=reg(f,BASE)['AEE']
        x=np.array([[ordinary[t][k] for k in FEATURES] for t in TILES],dtype=np.float64)
        data[f['target_id']]={'x':x,'gain':y,'relative_gain':y/base,'mse':np.array([branch[t] for t in TILES])}
        choices['coarse_baseline'][f['target_id']]=BASE
        choices['posthoc_GT_best_singleton'][f['target_id']]=arm(choose(y))
        scores={k:np.array([branch[t] if k=='offline_branch_mse' else ordinary[t][k] for t in TILES]) for k in SCORES}
        for k,v in scores.items():choices[k][f['target_id']]=arm(choose(v))
        frame_diagnostics.append({'target_id':f['target_id'],'sequence':f['sequence'],'singleton_gains_AEE':y.tolist(),
                                  'positive_gains':int((y>0).sum()),'negative_gains':int((y<0).sum()),
                                  'best_gain':float(y.max()),'worst_gain':float(y.min()),
                                  'spearman_score_vs_task_gain':{k:spearman(v,y) for k,v in scores.items()}})
    folds=[]
    for check_seq in seqs:
        train=[f for f in frames if f['sequence']!=check_seq];check=[f for f in frames if f['sequence']==check_seq]
        require(len(train)==len(check)==4,'4/4 fixed sequence split')
        x=np.concatenate([data[f['target_id']]['x'] for f in train])
        tx=np.concatenate([data[f['target_id']]['x'] for f in check])
        fold={'training_sequence':train[0]['sequence'],'check_sequence':check_seq,
              'training_target_ids':[f['target_id'] for f in train],'check_target_ids':[f['target_id'] for f in check],
              'training_samples':len(x),'models':{}}
        for key,target in [('LOSO_task_ridge','relative_gain'),('LOSO_branch_mse_ridge','mse')]:
            y=np.concatenate([data[f['target_id']][target] for f in train])
            pred,model=ridge_fit_predict(x,y,tx);pred=pred.reshape(len(check),12)
            fold['models'][key]=dict(model,target=target,predictions=pred.tolist())
            for f,v in zip(check,pred):choices[key][f['target_id']]=arm(choose(v))
        position=np.mean([data[f['target_id']]['relative_gain'] for f in train],axis=0)
        fold['position_task_mean']=position.tolist()
        for f in check:choices['LOSO_position_task_mean'][f['target_id']]=arm(choose(position))
        folds.append(fold)
    summaries={name:{'regions':summarize(frames,c),
                     'per_sequence':{seq:summarize([f for f in frames if f['sequence']==seq],c) for seq in seqs},
                     'selected_arms':c} for name,c in choices.items()}
    for name,c in choices.items():
        summaries[name]['all_GT_wins_ties_losses_vs_coarse']={
            'wins':sum(reg(f,c[f['target_id']])['AEE']<reg(f,BASE)['AEE'] for f in frames),
            'ties':sum(reg(f,c[f['target_id']])['AEE']==reg(f,BASE)['AEE'] for f in frames),
            'losses':sum(reg(f,c[f['target_id']])['AEE']>reg(f,BASE)['AEE'] for f in frames)}
    joint=[]
    for f in frames:
        for name,tiles in f['combination_masks'].items():
            expected=sum(gain(f,tuple(t)) for t in tiles)
            actual=reg(f,BASE)['AEE']-reg(f,name)['AEE']
            joint.append({'target_id':f['target_id'],'sequence':f['sequence'],'arm':name,'tiles':tiles,
                          'singleton_gain_sum':expected,'actual_joint_gain':actual,
                          'joint_minus_sum':actual-expected})
    result={'schema':'flow_tile_intervention_analysis_v1','input_sha256':a.input_sha256,'script_sha256':sha(__file__),
            'features':FEATURES,'scope':'Reused dev8 two-sequence exploratory replay. k1 policies select executed singleton outputs; no new forward. GT best singleton is by definition an upper reference; k2 GT singleton ranking is not a joint optimum.',
            'singleton_count':96,'negative_singletons':sum(f['negative_gains'] for f in frame_diagnostics),
            'frame_diagnostics':frame_diagnostics,'LOSO_folds':folds,'k1_summaries':summaries,
            'actual_k2_summaries':{k:v for k,v in raw['summaries'].items() if k.startswith('k2_')},
            'joint_nonadditivity':joint,'deployable_predictor_validated':False,
            'cost_boundary':'Same k1 restored producer service, scorer costs differ and are not measured cycles. Offline MSE and GT consume counterfactual computation. Ridge feature statistics and inference must be priced before any hardware claim.'}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# 固定单块干预的探索性排序与两序列预测检查','','输入SHA `'+a.input_sha256+'`。固定12候选、全GT主排序；原始全分区与例外保留。',
           '', '## k=1：直接索引已执行的真实单块结果','',
           '| 规则 | 全GT | B | L | 全GT胜/平/负 vs coarse |','|---|---:|---:|---:|---|']
    for k,v in summaries.items():
        vals=[v['regions'][r]['frame_mean_AEE'] for r in ('all_valid','B','L')]
        win=v['all_GT_wins_ties_losses_vs_coarse']
        lines.append('| '+k+' | '+' | '.join(f'{x:.9f}' for x in vals)+f" | {win['wins']}/{win['ties']}/{win['losses']} |")
    lines+=['','LOSO仅两序列互换、训练侧dev8探索；训练成员48条，5个特征，ridge alpha=1且不调参。任务目标为相对全GT单块改善，特征目标为branch MSE；同输入特征/容量/训练成员。position表是不看当前特征的已拟合强位置控制。','',
            '## k=2：真实组合前向','','| 规则 | 全GT | B | L |','|---|---:|---:|---:|']
    for k,v in result['actual_k2_summaries'].items():lines.append('| '+k+' | '+' | '.join(f"{v['regions'][r]['frame_mean_AEE']:.9f}" for r in ('all_valid','B','L'))+' |')
    lines+=['','GT top1是定义上最优；GT单块top2并非联合最优。离线特征MSE和GT都消费反事实结果，不能当免费在线预测。原始[analysis.json](analysis.json)保存逐序列、负收益、相关性、组合非加性与完整拟合参数；这是探索，不是独立确认、论文复现失败或已测硬件收益。']
    (a.output_dir/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'ANALYSIS_COMPLETE','negative_singletons':result['negative_singletons'],'output':str(a.output_dir)},ensure_ascii=False))


if __name__=='__main__':main()
