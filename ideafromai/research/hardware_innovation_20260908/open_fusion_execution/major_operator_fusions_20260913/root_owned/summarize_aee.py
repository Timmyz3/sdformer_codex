"""Recompute the actual per-frame AEE of this stage, including fusion arms."""
from pathlib import Path
import csv
import json
from statistics import mean

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
NB=BASE/'accuracy_baseline/source_nb0_valid825.csv'
if not NB.exists():NB=BASE/'stage_20260912/algorithm/source_nb0_valid825.csv'


def main():
    with NB.open() as f:nb={r['file']:r for r in csv.DictReader(f)}
    parent=json.loads((HERE/'aee/parent_frames.json').read_text())
    names=[r['file'] for r in parent]
    baseline=mean(float(nb[n]['AEE']) for n in names)
    parent_mean=mean(r['AEE'] for r in parent)
    lifting_frames=HERE/'temporal_fusion_aee/lifting_parent_frames.json'
    lifting_mean=(mean(r['AEE'] for r in json.loads(lifting_frames.read_text())) if lifting_frames.exists() else None)
    rows=[]
    for batch in ['aee','count_aee','r0_sparse_aee','control_aee','cohort_aee','temporal_fusion_aee']:
        path=HERE/batch/'summary.json'
        if not path.exists():continue
        summary=json.loads(path.read_text())
        for name,axis in summary['axes'].items():
            frames=json.loads((HERE/batch/(name+'_frames.json')).read_text())
            assert [r['file'] for r in frames]==names
            for r in frames:
                assert r['valid_pixels']==int(float(nb[r['file']]['valid_pixels']))
                assert abs(r['AEE']-r['aee_sum']/r['valid_pixels'])<1e-10
            aee=mean(r['AEE'] for r in frames)
            assert abs(aee-axis['summary']['AEE_frame_mean'])<1e-10
            paired_parent=(lifting_mean if axis['config'].get('structure')=='lifting40' else parent_mean)
            rows.append(dict(batch=batch,arm=name,frames=len(frames),AEE=aee,
                AEE_nine_excluding_common_calibration_frame=mean(r['AEE'] for r in frames[1:]),
                paired_parent_AEE=paired_parent,delta_from_paired_parent=aee-paired_parent,delta_from_NB0=aee-baseline,
                better_than_NB0=aee<baseline))
    with (HERE/'aee_all.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    report=dict(actual_axes=len(rows),actual_frame_forwards=sum(r['frames'] for r in rows),
        new_optimizer_updates=0,NB0_same_ten_frames=baseline,parent=parent_mean,
        full_valid825=False,frames=names,rows=rows,
        scope='Exploratory common forward/evaluation harness across new arms, with coarse preds.2. NB0 uses its original final head and FP32 reduction; new arms use FP64 EPE summation. Same files, GT and metric comparison, not identical network execution or bit-identical numerical protocol. First frame contributes source calibration to some arms; nine-frame column excludes it, but all frames have been observed in prior project work. No held-out test generalization, statistical superiority or hardware speed claim.')
    (HERE/'aee_all.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 本轮实际整网 AEE（探索性 diverse10）','',
        f'共 {len(rows)} 个完整十帧运行、{len(rows)*10} 次前向，0 次新增训练更新。NB0={baseline:.10f}；dense父={parent_mean:.10f}；lifting父={lifting_mean:.10f}。各臂相对自己的结构父比较。',
        '','同帧名、同有效像素已逐帧核对。新臂用共同粗头preds.2与FP64 EPE求和；NB0用原最终头与FP32归约。比较任务质量，不声称同网络执行或逐位同数值协议。第一帧参与部分候选的源统计校准；CSV另列剔除它的九帧均值。所有帧都是项目中已看过的验证样本，没有把十帧评估写成新的 valid825。','',
        '|批次 / 运行|AEE|相对对应结构父|低于NB0|','|---|---:|---:|---|']
    for r in rows:lines.append(f"|{r['batch']} / {r['arm']}|{r['AEE']:.10f}|{r['delta_from_paired_parent']:+.7f}|{'是' if r['better_than_NB0'] else '否'}|")
    lines+=['','融合组合均实际重新运行，没有把单项质量或周期比相乘。新的融合不因小幅相对父退化淘汰，但过 NB0 本身不足以证明优于普通2:4/3:4，也不代表硬件费用通过。']
    (HERE/'aee_all.md').write_text('\n'.join(lines)+'\n')
    print('AEE_RECOMPUTED',len(rows),baseline)


if __name__=='__main__':main()
