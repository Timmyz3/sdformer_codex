#!/usr/bin/env python3
"""Fit-only real streaming count and packed-tree replay; no GPU or quality read."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

P=Path(__file__).resolve().parent
ROOT=P.parents[2]
R33=ROOT/'flow_story_20261021/round33_predictability'
R34=ROOT/'flow_story_20261021/round34_deployability'
OUT=P/'actual_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
NAMES=['count32_tree_direct','count32_tree_projection']

def pack(tree):
    words=np.zeros(16,dtype='<u4');nodes=[]
    def visit(t,depth=0):
        if 'feature' not in t:return 0x80000000|int(t['action'])
        assert depth<2 and len(nodes)<3
        i=len(nodes);nodes.append(None)
        left=visit(t['left'],depth+1);right=visit(t['right'],depth+1)
        nodes[i]=[int(t['feature']),int(t['threshold']),left,right]
        return i
    words[12]=visit(tree)
    for i,node in enumerate(nodes):words[4*i:4*i+4]=node
    return words

def traverse(counts,words):
    # Finite 192B physical register image, same counted port per word as C++.
    mem=np.concatenate([np.asarray(counts,dtype='<u4'),words]);events=[];ticks=0
    def load(address):
        nonlocal ticks
        events.append(dict(start=ticks,end=ticks+1,address=4*address,op='read'))
        ticks+=1;return int(mem[address])
    node=load(44);depth=0
    while not node&0x80000000:
        assert node<3 and depth<2
        a=32+4*node;feature=load(a);threshold=load(a+1);left=load(a+2);right=load(a+3)
        value=load(feature);events.append(dict(start=ticks,end=ticks+1,op='compare'));ticks+=1
        node=left if value<=threshold else right;events.append(dict(start=ticks,end=ticks+1,op='branch'));ticks+=1;depth+=1
    ticks+=1
    assert ticks==2+7*depth
    return node&7,depth,ticks,events

def main():
    OUT.mkdir(exist_ok=True)
    models_path=P.parent/'algorithm/actual_v1/FINAL_FIT_MODELS_V1.json'
    pred_path=P.parent/'algorithm/actual_v1/PREDICTIONS_BEFORE_QUALITY_V1.json'
    models=read(models_path);pred=read(pred_path)
    configs={n:pack(models['models'][n]) for n in NAMES}
    for n,c in configs.items():c.tofile(OUT/(n+'.u32'))
    rows=[r for r in read(R33/'data/MEMBERSHIP_V1.json')['records'] if r['role']=='fit']
    assert len(rows)==64 and [r['target_id'] for r in rows]==pred['ids']==models['fit_ids']
    reference=np.load(R34/'numeric/actual_v1/FEATURES_FP32_V1.npz')
    ref_ids=list(reference['fit_ids']);ref_x=reference['fit_x160']
    cols=[8*c+p for c in range(16) for p in (2,3)]
    def job(item):
        index,r=item;tid=r['target_id'];s=next(f for f in r['source_files'] if f['category']=='input')
        source=R33/'data/fit_tune_source_snapshot'/s['relative_path'];assert sha(source)==s['sha256']
        cmd=[str(P/'count_stream'),str(source),str(OUT/'count32_tree_direct.u32'),str(OUT/(tid+'.u32')),str(OUT/(tid+'.json')),str(OUT/'first.trace') if index==0 else '-','16']
        subprocess.run(cmd,check=True)
        raw=np.load(source,allow_pickle=False);expected=[]
        for cy in range(4):
            for cx in range(4):
                z=raw[:,cy*120:(cy+1)*120,cx*160:(cx+1)*160]
                expected.extend([int(np.count_nonzero(z>0)),int(np.count_nonzero(z<0))])
        count=np.fromfile(OUT/(tid+'.u32'),dtype='<u4');back=np.fromfile(OUT/(tid+'.u32.rf'),dtype='<u4')
        assert np.array_equal(count,expected) and np.array_equal(back[:32],count) and np.array_equal(back[32:],configs[NAMES[0]])
        assert np.array_equal(count,np.asarray(pred['counts'][index]))
        fractions=count.astype(np.float32)/np.float32(192000)
        assert np.array_equal(fractions,ref_x[ref_ids.index(tid),cols])
        service=read(OUT/(tid+'.json'));action,depth,head_ticks,_=traverse(count,configs[NAMES[0]])
        assert (action,depth)==(service['action'],service['tree_depth_visited'])
        assert service['feature_ready_ticks']==2310098 and service['action_ready_ticks']==2310098+head_ticks
        return dict(target_id=tid,source=str(source),input_sha256=s['sha256'],count=count.tolist(),service=service,
                    exact_raw_numpy_counts=True,exact_algorithm_counts=True,bitwise_R34_FP32_fraction_subset=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(job,enumerate(rows)))
    byid={r['target_id']:r for r in result};replays=[]
    for fi,fold in enumerate(pred['folds']):
        for name in NAMES:
            words=pack(fold['fit']['models'][name]);words.tofile(OUT/(f'fold{fi}_{name}.u32'))
            for j,tid in enumerate(fold['held_ids']):
                action,depth,ticks,events=traverse(byid[tid]['count'],words)
                assert action==fold['predictions'][name]['actions'][j] and depth==fold['predictions'][name]['comparisons'][j]
                index=pred['ids'].index(tid);assert action==pred['actions'][name][index]
                replays.append(dict(fold=fi,held_group=fold['held_group'],target_id=tid,policy=name,action=action,
                    visited_nodes=depth,head_ticks=ticks,full_prefix_ticks=2310098+ticks,head_events=events))
    assert len(replays)==128
    first=result[0]
    subprocess.run([str(P/'count_stream'),first['source'],str(OUT/'count32_tree_direct.u32'),str(OUT/'scalar_classifier.u32'),str(OUT/'scalar_classifier.json'),str(OUT/'scalar_classifier.trace'),'1'],check=True)
    assert np.array_equal(np.fromfile(OUT/'scalar_classifier.u32',dtype='<u4'),first['count'])
    np.savez(OUT/'COUNT32_V1.npz',ids=np.asarray([r['target_id'] for r in result]),counts=np.asarray([r['count'] for r in result],dtype=np.uint32))
    output=dict(schema='r37_real_count32_stream_execution_v1',status='PASS',fit_identities=64,OOF_integer_tree_replays=128,
        input_payload_bytes=12288000,features_equivalent_only_to_X128_count_fraction_subset=True,
        old_X128_policy_equivalence=False,old_accuracy_reused=False,GPU_used=False,quality_computed=False,
        rows=result,oof_replays=replays,countercondition=read(OUT/'scalar_classifier.json'),
        bindings={str(p):sha(p) for p in [P/'count_stream.cpp',P/'run_cpu.py',models_path,pred_path,P/'IMPLEMENTATION_CONTRACT_V1.md']})
    (OUT/'CPU_EXECUTION_V1.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('rows','oof_replays','bindings')},indent=2))

if __name__=='__main__':main()
