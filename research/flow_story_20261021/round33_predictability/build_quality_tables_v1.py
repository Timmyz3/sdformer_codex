"""Report every frozen learned head; no fitting, selection, or new inference."""
from pathlib import Path
import csv
import hashlib
import json

B = Path(__file__).resolve().parent
METHODS = ['direct_raw','signed_g1','signed_g4','direct_aug_g1','direct_aug_g4','direct_residual_g1','direct_residual_g4']


def main():
    out=B/'report';out.mkdir(exist_ok=False)
    rows=[];sources={};text=['# Round33 完整冻结头结果表','',
        '正号表示相对同数据 Q4 的 AEE 退化；负号表示改善。固定六动作与允许 Q4 分开。',
        '仅仓库内划分：tune16 与 confirm40，不是官方 DSEC test。每行均为同一冻结头，无结果后重训或择参。',
        '周期列只含四个受控卷积的模型服务之和，未含预测器，不是整网延迟。尾数是 allGT/B/L 的触发项数，不是独立帧数。','']
    cache={}
    for phase in ['tune','confirm']:
        for dim in [128,160]:
            p=B/f'evaluations/{phase}_{dim}_v1/EVALUATION_V1.json'
            r=json.loads(p.read_text());cache[phase,dim]=r;sources[str(p.relative_to(B))]=hashlib.sha256(p.read_bytes()).hexdigest()
            base=r['quality_summaries']['Q4']['overall']
            for name,scopes in r['quality_summaries'].items():
                q=scopes['overall'];d=r['decision_summaries'][name]['overall'];c=r['comparisons'].get(name+'__vs__Q4',{})
                row=dict(phase=phase,feature_dimension=dim,policy=name,observable=r['policy_definitions'][name]['observable'],
                         upgrade_fraction=d['upgrade_fraction'],four_conv_operator_only_mean_ticks=d['operator_only_service']['mean_four_conv_ticks'],
                         tail_review_items_vs_Q4=len(c.get('high_support_tail_review',[])))
                for region in ['all_valid','B','L','B_and_L','voxel_nonzero','voxel_zero']:
                    a=q[region]['pixel_weighted_AEE'];z=base[region]['pixel_weighted_AEE']
                    row[region+'_pixel_AEE']=a
                    row[region+'_relative_percent_vs_Q4']=100*(a/z-1) if z else None
                    row[region+'_frame_AEE']=q[region]['frame_mean_AEE']
                    row[region+'_support']=q[region]['valid_pixels']
                rows.append(row)
            for mode,title in [('six','固定六选一'),('allow_Q4','允许不升级')]:
                text.extend([f'## {phase} / X{dim} / {title}','',
                    '|方法|全GT AEE|相对Q4 %|B变化 %|L变化 %|B∩L变化 %|升级比例|四conv模型ticks|尾部触发项|',
                    '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
                names=['Q4','fit_static__'+mode]+(['uniform_six_expectation'] if mode=='six' else [])+[m+'__'+mode for m in METHODS]+['oracle__'+mode,'ideal_signed_g1__'+mode,'ideal_signed_g4__'+mode]
                for name in names:
                    x=next(v for v in rows if v['phase']==phase and v['feature_dimension']==dim and v['policy']==name)
                    text.append(f"|{name}|{x['all_valid_pixel_AEE']:.6f}|{x['all_valid_relative_percent_vs_Q4']:+.4f}|{x['B_relative_percent_vs_Q4']:+.4f}|{x['L_relative_percent_vs_Q4']:+.4f}|{x['B_and_L_relative_percent_vs_Q4']:+.4f}|{x['upgrade_fraction']:.3f}|{x['four_conv_operator_only_mean_ticks']:.1f}|{x['tail_review_items_vs_Q4']}|")
                text.append('')
    (out/'QUALITY_TABLES_V1.md').write_text('\n'.join(text)+'\n')
    with (out/'ALL_POLICIES_V1.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
    labels=['Direct raw','Signed 1x1','Signed 4x4','Direct augmented 1x1','Direct augmented 4x4','Direct residual 1x1','Direct residual 4x4']
    for ax,dim in zip(axes,[128,160]):
        ax.axvline(0,color='#777777',linewidth=1)
        for j,m in enumerate(METHODS):
            vals=[]
            for phase in ['tune','confirm']:
                r=cache[phase,dim];q=r['quality_summaries'];vals.append(100*(q[m+'__six']['overall']['all_valid']['pixel_weighted_AEE']/q['Q4']['overall']['all_valid']['pixel_weighted_AEE']-1))
            ax.plot(vals,[j,j],color='#bbbbbb',linewidth=1)
            ax.scatter(vals[0],j,s=55,marker='s',color='#c77725',label='Tune 16' if j==0 else None)
            ax.scatter(vals[1],j,s=55,marker='o',color='#21618c',label='Confirm 40' if j==0 else None)
        ax.set_title(f'X{dim}: all seven frozen heads')
        ax.set_yticks(range(7),labels);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('All-valid pixel AEE change vs Q4 (%)\nLower is better; fixed six-action budget')
        ax.legend(loc='lower left')
    axes[0].invert_yaxis()
    fig.suptitle('Round33: small gains and a tune / confirmation ranking reversal',fontsize=13)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'QUALITY_TRANSFER_V1.{ext}',dpi=180,bbox_inches='tight')
    freeze=dict(schema='flow_r33_reporting_artifacts_v1',sources=sources,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},new_fit=False,new_model_inference=False,GPU_used=False,
                note='All frozen methods and all diagnostic policies retained in CSV; no method selected by this report.')
    (out/'REPORT_FREEZE_V1.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print('REPORT_COMPLETE',len(rows))


if __name__=='__main__':main()
