#!/usr/bin/env python3
"""Independent integer geometry/MAC inventory, no models, GT, quality or GPU."""
import ast
import hashlib
import itertools
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DATE=HERE.parents[1]
REPO=DATE.parents[1]
GEOMETRY=REPO/'research/flow_story_20261016/coverage_recovery_round13/independent_execution_audit/audit_execution.py'
G_SHA='dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8'
NATIVE=DATE/'dense_reference_round17/actual_v1/run_v1/results.json'
N_SHA='0b4ef2744c9da8fd3513b65141b4133f381cf267db68083e4bb854165ae07a2e'
AUDIT=DATE/'dense_reference_round17/independent_execution_audit/ACTUAL_EXECUTION_AUDIT_V1.json'
A_SHA='6ed07d00330df7824e4f8cc754a1c735bded4b3bf5bf89565f9114506594276f'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cells(h,w,selected=None):
    # Deliberately independent of primitive.pack/service_model.points.
    out=[]
    for ty in range((h+8)//9):
        for tx in range((w+8)//9):
            y0,x0=9*ty,9*tx;y1,x1=min(y0+9,h),min(x0+9,w)
            stride=1 if selected==(ty,tx) else 3
            for y in range(y0,y1,stride):
                for x in range(x0,x1,stride):
                    out.append((min(y+stride,y1)-y)*(min(x+stride,x1)-x))
    assert sum(out)==h*w
    return out


def main():
    assert sha(GEOMETRY)==G_SHA and sha(NATIVE)==N_SHA and sha(AUDIT)==A_SHA
    # Read only literal source geometry; never import the old auditor/runner.
    parsed=ast.parse(GEOMETRY.read_text())
    node=next(n.value for n in parsed.body if isinstance(n,ast.Assign) and
              any(isinstance(t,ast.Name) and t.id=='GEOMETRY' for t in n.targets))
    stages=ast.literal_eval(node.generators[0].iter.args[0])
    audit=json.loads(AUDIT.read_text());assert audit['status']=='PASS' and audit['source_result_sha256']==N_SHA
    data=json.loads(NATIVE.read_text());record=data['forward_records'][0]
    assert record['arm']=='native' and record['execution']=='native'
    rows=[]
    for stage,(depth,h,w,c) in enumerate(stages):
        native=h*w;coarse=len(cells(h,w));complete=(h//9)*(w//9)
        assert complete>0
        counts={len(cells(h,w,(y,x))) for y in range(h//9) for x in range(w//9)}
        assert counts=={coarse+72}
        k1=coarse+72
        for block in range(depth):
            name=f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.{block}.mlp'
            assert record['all12_observed_fc_rows'][name]==dict(fc1_rows=10*native,fc2_rows=10*native)
            for bn,width in (('bn1',4*c),('bn2',c)):
                shapes=record['native_BN_inputs'][name+'.'+bn+'.norm_layer']
                assert len(shapes)==1 and shapes[0]['shape']==[10,1,width,h,w]
            policies={}
            for arm,reps in (('native',native),('CC',coarse),('k1',k1)):
                per_fc=10*reps*c*(4*c)
                policies[arm]=dict(spatial_representatives=reps,rows_per_FC=10*reps,
                                   FC1_MAC=per_fc,FC2_MAC=per_fc,FC_pair_MAC=2*per_fc)
            rows.append(dict(layer=name,short_name=f's{stage}b{block}',stage=stage,block=block,
                        shape=[10,1,h,w,c],hidden_channels=4*c,complete_9x9_tiles=complete,
                        total_tiles=((h+8)//9)*((w+8)//9),tail_tiles=((h+8)//9)*((w+8)//9)-complete,
                        policies=policies))
    assert len(rows)==12 and sum(x['policies']['native']['rows_per_FC'] for x in rows)==558000
    total=sum(x['policies']['native']['FC_pair_MAC'] for x in rows)
    assert total==169869312000 and len({x['policies']['native']['FC_pair_MAC'] for x in rows})==1
    for row in rows:
        native=row['policies']['native']['FC_pair_MAC']
        for arm,p in row['policies'].items():
            p.update(retained_fraction_of_all12_native_FC_MAC=p['FC_pair_MAC']/total,
                     cancelled_FC_MAC=native-p['FC_pair_MAC'],
                     cancelled_fraction_of_all12_native_FC_MAC=(native-p['FC_pair_MAC'])/total)
    ranges={}
    for arm in ('CC','k1'):
        cancelled=[a['policies'][arm]['cancelled_FC_MAC']+b['policies'][arm]['cancelled_FC_MAC']
                   for a,b in itertools.combinations(rows,2)]
        ranges[arm]=dict(pair_count=len(cancelled),cancelled_MAC_min=min(cancelled),cancelled_MAC_max=max(cancelled),
                        cancelled_fraction_min=min(cancelled)/total,cancelled_fraction_max=max(cancelled)/total,
                        interpretation='Only additive disjoint-layer FC service counts; no layers selected, no quality gains added, no execution admission')
    out=dict(schema='flow_r19_independent_nominal_MLP_scope_v1',status='GEOMETRY_INVENTORY_ONLY',
             source_sha256={str(p):sha(p) for p in (Path(__file__),GEOMETRY,NATIVE,AUDIT)},
             layers=rows,all12_native_FC_pair_MAC=total,all12_native_rows_each_FC=558000,
             nominal_two_layer_ranges=ranges,full_model_forwards=0,GPU_calls=0,GT_or_quality_read=False,
             limitations=['Only FC1+FC2 MAC; not PE cycles or full-network denominator.',
                          'CC/k1 at layers other than R19 target are hypothetical geometry, not tested interventions.',
                          'Do not select layers or launch larger graph from these counts; ordinary controls/quality and a complete finite-resource schedule remain required.'])
    dest=HERE/'NOMINAL_MLP_SCOPE_V1.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
    lines=['|层|H×W×C|完整/尾tile|代表点 native/CC/k1|FC1+FC2 GMAC native/CC/k1|CC/k1取消占十二MLP原FC MAC|',
           '|---|---|---:|---|---|---|']
    for r in rows:
        pol=r['policies'];dims=r['shape'][2:]
        lines.append('|%s|%s|%d/%d|%s|%s|%s|'%(r['short_name'],'×'.join(map(str,dims)),r['complete_9x9_tiles'],r['tail_tiles'],
          '/'.join(str(pol[a]['spatial_representatives']) for a in ('native','CC','k1')),
          '/'.join('%.8f'%(pol[a]['FC_pair_MAC']/1e9) for a in ('native','CC','k1')),
          '/'.join('%.6f%%'%(100*pol[a]['cancelled_fraction_of_all12_native_FC_MAC']) for a in ('CC','k1'))))
    table=HERE/'NOMINAL_MLP_TABLE_V1.md';assert not table.exists();table.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(result_sha256=sha(dest),all12_native_FC_MAC=total,two_layer_ranges=ranges),indent=2))


if __name__=='__main__':main()
