#!/usr/bin/env python3
"""Independently aggregate M361r4 partition records and audit geometry."""
import argparse, hashlib, json, math
from pathlib import Path

def need(x, msg):
    if not x: raise RuntimeError(msg)

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    def hook(items):
        d={}
        for k,v in items:
            need(k not in d, 'duplicate key '+k); d[k]=v
        return d
    return json.loads(Path(p).read_text(), object_pairs_hook=hook,
                      parse_constant=lambda x: (_ for _ in ()).throw(RuntimeError(x)))

def signed_bits(k):
    b=1
    while (1 << (b-1)) < 128*k: b += 1
    return b

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--contract',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); need(not a.output_dir.exists(),'output exists')
    c=load(a.contract); need(c['schema']=='m372_m361r4_independent_hammer_contract_v1','schema')
    root=a.contract.resolve().parents[1]; paths={}
    for n,x in c['inputs'].items():
        p=root/x['path']; need(p.is_file() and digest(p)==x['sha256'],'SHA '+n); paths[n]=p
    wide=load(paths['m361r4_result']); k16=load(paths['m339_result']); m338=load(paths['m338_result']); m354=load(paths['m354_review'])
    qv=(16,32,64,128); runtime_base=k16['exact_runtime_working_set'][0]['bit_sparse_vector_ops_per_block']
    need(all(r['bit_sparse_vector_ops_per_block']==runtime_base for r in k16['exact_runtime_working_set']),'k16 runtime baseline')
    train_base=m338['calibration_observations'][0]['bit_sparse_vector_ops_per_block']
    need(all(r['bit_sparse_vector_ops_per_block']==train_base for r in m338['calibration_observations']),'train baseline')
    rows=[]; partition_records=0
    geo={g['partition_bits']:g for g in wide['geometry_dse']}
    for k in (32,64):
        g=geo[k]; parts=6912//k; need(g['partitions_per_operator']==parts,'partitions')
        sums={q:{'train':0,'runtime':0} for q in qv}
        for op in wide['catalogs'][str(k)]:
            need(len(op['partitions'])==parts,'partition population')
            for p in op['partitions']:
                partition_records += 1; centers=p['nested_patterns']
                need(len(centers)==len(set(centers)) and len(centers)<=128,'centers')
                need(all(len(x)==k//4 and int(x,16)>0 and
                         bin(int(x,16)).count('1')>=2 for x in centers),
                     'center encoding')
                previous_train=previous_runtime=None
                for q in qv:
                    o=p['observations'][str(q)]
                    need(o['active_patterns']==min(q,len(centers)),'nested prefix count')
                    tr=o['train_candidate_vector_ops_per_block']; rt=o['runtime_candidate_vector_ops_per_block']
                    if previous_train is not None:
                        need(tr<=previous_train and rt<=previous_runtime,'partition monotonicity')
                    previous_train,previous_runtime=tr,rt; sums[q]['train']+=tr; sums[q]['runtime']+=rt
        byq={r['q_capacity']:r for r in g['q_rows']}; bits=signed_bits(k); vec=(96*bits+7)//8
        need(bits==g['signed_int8_pwp_bits'] and vec==g['signed_pwp_vector_bytes_per_output_block'],'PWP width')
        for q in qv:
            r=byq[q]; train_speed=train_base/float(sums[q]['train']); runtime_speed=runtime_base/float(sums[q]['runtime'])
            need(sums[q]['runtime']==r['runtime_candidate_vector_ops_per_block'],'runtime sum')
            need(abs(train_speed-r['train_exact_vector_work_speedup'])<1e-15,'train divide')
            need(abs(runtime_speed-r['runtime_exact_vector_work_speedup'])<1e-15,'runtime divide')
            need(r['runtime_pwp_vector_ops_per_block']+r['runtime_correction_vector_ops_per_block']==sums[q]['runtime'],'work conservation')
            pattern=4*parts*q*(k//8); pwp=4*parts*q*8*vec
            need(pattern==r['pattern_table_capacity_bytes'] and pwp==r['full_signed_pwp_capacity_bytes'],'capacity')
            rows.append({'k':k,'q':q,'train_candidate_work':sums[q]['train'],'train_speedup':train_speed,'runtime_candidate_work':sums[q]['runtime'],'runtime_speedup':runtime_speed,'signed_pwp_bits':bits,'pwp_vector_bytes':vec,'pattern_capacity_bytes':pattern,'full_pwp_capacity_bytes':pwp})
    for k in (32,64):
        rr=[r for r in rows if r['k']==k]; need(all(rr[i]['runtime_candidate_work']<=rr[i-1]['runtime_candidate_work'] for i in range(1,4)),'global monotonic')
    k16q={r['q_capacity']:r for r in k16['exact_runtime_working_set']}; ref=k16q[128]
    ref_speed=runtime_base/float(ref['candidate_vector_ops_per_block']); need(abs(ref_speed-2.043940269355372)<1e-15,'k16 ref')
    matched=[]
    for row in rows:
        k16_speed=runtime_base/float(k16q[row['q']]['candidate_vector_ops_per_block'])
        need(row['runtime_speedup'] < k16_speed, 'wide point not below matched-q k16')
        matched.append({'k':row['k'],'q':row['q'],
                        'wide_speedup':row['runtime_speedup'],
                        'k16_speedup':k16_speed,
                        'wide_over_k16_candidate_work':
                            row['runtime_candidate_work']/float(k16q[row['q']]['candidate_vector_ops_per_block'])})
    q32cap=next(r for r in m354['capacity_recompute']['rows'] if r['q']==32 and r['output_block_tile']==4)
    result={'schema':'m372_m361r4_independent_hammer_v1','status':'PASS_M372_KEY_ROWS_RECOMPUTED','partition_records_aggregated':partition_records,'train_baseline_work':train_base,'runtime_baseline_work':runtime_base,'rows':rows,'matched_q_k16_comparison':matched,'k16_reference':{'q128_candidate_work':ref['candidate_vector_ops_per_block'],'q128_speedup':ref_speed,'q32_candidate_work':k16q[32]['candidate_vector_ops_per_block'],'q32_speedup':runtime_base/float(k16q[32]['candidate_vector_ops_per_block'])},'k16_q32_o4_context':q32cap,'decision':{'close_wide_partition_active_line':True,'continue_k16_q32_o4_executable_scheduler':True,'reason':'Every matched-q wide point has worse exact work than k16; even k32/q128 remains below k16/q128 while requiring 13-bit PWP. k16 q32/O4 has a sealed two-context capacity fit and is the more mature executable-scheduler target.','wide_partition_cycle_or_area_dominance_proven':False},'claim_boundary':{'exact_vector_work':True,'cycle_speedup':False,'system_speedup':False,'headline':False}}
    a.output_dir.mkdir(parents=True); (a.output_dir/'m372_m361r4_independent_hammer_r1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('M372_PASS rows=8 partition_records={} k32q128={:.6f} k64q128={:.6f} k16q128={:.6f}'.format(partition_records,[r for r in rows if r['k']==32 and r['q']==128][0]['runtime_speedup'],[r for r in rows if r['k']==64 and r['q']==128][0]['runtime_speedup'],ref_speed))
if __name__=='__main__': main()
