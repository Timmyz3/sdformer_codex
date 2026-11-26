#!/usr/bin/env python3
"""One frozen legacy-fixture group-opportunity census, never a cycle simulator."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DS=((0,0),(0,1),(0,-1),(1,0),(-1,0))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def require(ok,msg):
    if not ok:raise AssertionError(msg)


def ratio(saved,base):return saved/base if base else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    adm=json.loads(args.admission.read_text())
    require(adm['schema']=='flow_r41_delta_group_admission_v1','admission schema')
    require(adm['status']=='PASS' and adm['reviewer']!='hardware-implementer','independent PASS required')
    for name in ('PROTOCOL_V1.md','ASSETS_V1.json','screen_delta_groups_v1.py'):
        require(sha(HERE/name)==adm['sha256'][name],'reviewed hash '+name)
    assets=json.loads((HERE/'ASSETS_V1.json').read_text())
    require(sha(ROOT/assets['old_results_path'])==assets['old_results_sha256'],'legacy results hash')
    require(assets['graph_id']=='r26_W271482_all78current_s2b2_fullspatial_weightplanes_H8','legacy graph')
    require(len(assets['records'])==8 and len({r['target_id'] for r in assets['records']})==8,'all eight unique identities')
    require(not args.output_dir.exists(),'fresh output directory required')
    args.output_dir.mkdir(parents=True)
    import numpy as np
    started=time.monotonic();raw={};records=[]
    result={'schema':'flow_r41_delta_group_result_v1','status':'INCOMPLETE','graph_id':assets['graph_id'],
            'admission_sha256':sha(args.admission),'device':'CPU_NUMPY','numpy':np.__version__,
            'model_forwards':0,'GPU_calls':0,'GT_reads':0,'training_steps':0,
            'cost_unit':'128-input-group x weight-plane x output-channel; NOT cycles',
            'free_costs':['reference lookup','reference current residency','comparison','mode selection',
                          'current initialization','data movement','control'],
            'reference_domain':'same saved tensor, previous PSN lane; NOT consecutive data windows',
            'displacements_dy_dx':[list(d) for d in DS],'records':records}

    def count_groups(a,width):
        return a.reshape(*a.shape[:-1],a.shape[-1]//width,width).any(axis=-1).sum(axis=-1,dtype=np.int16)

    try:
        for index,rec in enumerate(assets['records']):
            require(time.monotonic()-started<115,'internal time cap')
            path=ROOT/rec['path']
            require(sha(path)==rec['sha256'],'legacy archive SHA '+rec['target_id'])
            id_record={'target_id':rec['target_id'],'archive_sha_verified':True,'FC':{}}
            with np.load(path,allow_pickle=False) as archive:
                for name,fc,cin,cout in (('sn1','fc1',384,1536),('sn2','fc2',1536,384)):
                    trace=rec['FC_execution'][fc]
                    require(trace['actual_input_shape']==[10,1,30,40,cin] and trace['input_rows']==12000,'legacy FC input geometry')
                    calls=trace['int_mm_calls']
                    require(len(calls)==8 and [x['bit'] for x in calls]==list(range(8)),'eight source bitplane calls')
                    for bit,call in enumerate(calls):
                        require(call['Cin']==cin and call['Cout']==cout and call['rows']==12000,'source Cout/rows')
                        require(call['coefficient']==(1<<bit if bit<7 else -128),'signed source plane')
                        require(call['row_scope']=='all_spatial_rows' and call['explicit_padding_rows']==0
                                and call['explicit_padding_channels']==0,'source all-real-row domain')
                    a=archive[name]
                    require(a.shape==(10,1,30,40,cin) and a.dtype==np.float32,'binary dump shape/dtype')
                    require(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==rec['tensors'][name]['sha256'],'legacy payload SHA')
                    require(bool(np.isfinite(a).all()) and bool(((a==0)|(a==1)).all()),'strict finite binary dump')
                    s=a[:,0].astype(np.bool_);del a
                    g0=count_groups(s,128);g16=count_groups(s,16)
                    gp=np.full((10,30,40,5),-1,dtype=np.int16)
                    gn=np.full_like(gp,-1);valid=np.zeros(gp.shape,dtype=np.bool_)
                    for di,(dy,dx) in enumerate(DS):
                        y0,y1=max(0,-dy),min(30,30-dy)
                        x0,x1=max(0,-dx),min(40,40-dx)
                        cur=s[1:,y0:y1,x0:x1,:]
                        ref=s[:-1,y0+dy:y1+dy,x0+dx:x1+dx,:]
                        gp[1:,y0:y1,x0:x1,di]=count_groups(cur & ~ref,128)
                        gn[1:,y0:y1,x0:x1,di]=count_groups(~cur & ref,128)
                        valid[1:,y0:y1,x0:x1,di]=True
                    d0=g0.copy();d0_choice=np.zeros(g0.shape,dtype=np.int8)
                    delta=gp[...,0]+gn[...,0]
                    take=valid[...,0] & (delta<d0)
                    d0[take]=delta[take];d0_choice[take]=1
                    best=g0.copy();choice=np.zeros(g0.shape,dtype=np.int8)
                    for di in range(5):
                        delta=gp[...,di]+gn[...,di]
                        take=valid[...,di] & (delta<best)
                        best[take]=delta[take];choice[take]=di+1
                    require(bool((best<=d0).all()) and bool((d0<=g0).all()),'direct/d0 dominance')
                    require(np.array_equal(best[0],g0[0]) and not bool(valid[0].any()),'t0 cold direct')
                    factor=8*cout
                    by_time=[]
                    for t in range(10):
                        totals={'ordinary128':int(g0[t].sum())*factor,'zero_shift_direct':int(d0[t].sum())*factor,
                                'free5d_direct':int(best[t].sum())*factor,'ordinary16_nonzero_groups':int(g16[t].sum())}
                        by_time.append({'t':t,'grouped_service':totals,
                                        'choice_counts_direct_then_DS':np.bincount(choice[t].ravel(),minlength=6).tolist(),
                                        'd0_choice_counts':np.bincount(d0_choice[t].ravel(),minlength=2).tolist(),
                                        'valid_rows_each_d':valid[t].sum(axis=(0,1)).tolist(),
                                        'GP_sums_valid_each_d':np.where(valid[t],gp[t],0).sum(axis=(0,1)).tolist(),
                                        'GN_sums_valid_each_d':np.where(valid[t],gn[t],0).sum(axis=(0,1)).tolist(),
                                        'zero_current_128_groups':int((cin//128)*1200-g0[t].sum())})
                    total={key:sum(t['grouped_service'][key] for t in by_time) for key in by_time[0]['grouped_service']}
                    base=total['ordinary128']
                    id_record['FC'][fc]={'Cin':cin,'Cout':cout,'plane_count':8,'factor':factor,'rows':12000,
                        'payload_sha_verified':True,'strict_binary_verified':True,'total':total,'by_time':by_time,
                        'potential':ratio(base-total['free5d_direct'],base),
                        'displacement_increment':ratio(total['zero_shift_direct']-total['free5d_direct'],base)}
                    prefix=f'{index:02d}_{fc}_'
                    for key,val in [('G0',g0),('ordinary16',g16),('GP',gp),('GN',gn),('valid',valid),
                                    ('d0_cost',d0),('d0_choice',d0_choice),('free5d_cost',best),('free5d_choice',choice)]:raw[prefix+key]=val
                    del s
            id_record['combined']={k:sum(v['total'][k] for v in id_record['FC'].values())
                                   for k in ('ordinary128','zero_shift_direct','free5d_direct')}
            records.append(id_record)
        summed={k:sum(r['combined'][k] for r in records) for k in ('ordinary128','zero_shift_direct','free5d_direct')}
        base=summed['ordinary128'];potential=ratio(base-summed['free5d_direct'],base)
        increment=ratio(summed['zero_shift_direct']-summed['free5d_direct'],base)
        first=base>0 and 10*(base-summed['free5d_direct'])>=base
        second=base>0 and 20*(summed['zero_shift_direct']-summed['free5d_direct'])>=base
        result['aggregate']={'summed_grouped_service':summed,'mean_grouped_service':{k:v/8 for k,v in summed.items()},
            'potential':potential,'displacement_increment':increment,
            'gate1_at_least_10_percent':first,'gate2_at_least_5_percentage_points':second,
            'decision':'ELIGIBLE_ONLY_FOR_SAME_OPERATOR_NEW_VERSION_CHECK' if first and second else 'STOP_DISPLACEMENT_ROUTE_ON_THIS_LEGACY_FIXTURE',
            'ordinary_zero_shift_potential':ratio(base-summed['zero_shift_direct'],base)}
        require(len(records)==8,'all eight identities complete')
        result['status']='LEGACY_FIXTURE_GROUP_SCREEN_COMPLETE'
    except Exception:result['exception']=traceback.format_exc()
    result['elapsed_seconds']=time.monotonic()-started
    np.savez_compressed(args.output_dir/'ROWS_V1.npz',**raw)
    result['rows_sha256']=sha(args.output_dir/'ROWS_V1.npz')
    (args.output_dir/'RESULTS_V1.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'identities_complete':len(records),'elapsed_seconds':result['elapsed_seconds'],
                      'aggregate':result.get('aggregate')},sort_keys=True))
    if result['status']!='LEGACY_FIXTURE_GROUP_SCREEN_COMPLETE':raise SystemExit(2)


if __name__=='__main__':main()
