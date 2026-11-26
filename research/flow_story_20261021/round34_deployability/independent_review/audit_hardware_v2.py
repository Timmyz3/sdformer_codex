"""Independent repaired storage lifetime, full recorded bounds and head bridge."""
from pathlib import Path
import csv,json,os
import numpy as np
import audit_numeric_v1 as N
B=N.B;R33=N.R33;D=B/'hardware';A=D/'actual_v2'
def oplat(k):return 3 if k in ['add','sub','mul'] else 16 if k in ['div','sqrt'] else 1
def ridge(m,x):
    x=np.asarray(x,np.float32);mean,scale,coef,bias=[np.asarray(m[k],np.float32) for k in ['mean','scale','coef','intercept']]
    z=(x-mean)/scale;y=np.zeros((len(x),len(bias)),np.float32)
    for k in range(z.shape[1]):y+=z[:,k,None]*coef[k]
    return y+bias
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';out=B/'independent_review/HARDWARE_V2_INDEPENDENT_AUDIT_V1.json';assert not out.exists()
    dtype=np.dtype([('start','<u8'),('end','<u8'),('address','<u4'),('words','<u4'),('mem','u1'),('write','u1'),('vector','u1'),('reserved','u1')]);assert dtype.itemsize==28
    streams=N.read(A/'STREAM_EXECUTION_V1.json');heads=N.read(A/'HEAD_PREFIX_RESULTS_V1.json');feat=N.load(A/'STREAM_FEATURES_V1.npz');refs=N.load(B/'numeric/actual_v1/FEATURES_FP32_V1.npz');scores=N.load(A/'STREAM_HEAD_SCORES_V1.npz')
    assert streams['identity_count']==120 and len(streams['rows'])==120;assert heads['feature_source_sha256']==N.sha(A/'STREAM_FEATURES_V1.npz')
    protected=[];tracecounts={};servicereceipts={}
    for name,meta in [('rowband160',streams['rows'][0]['service_file']),('rowband128','rowband128.json'),('cell160','cell160.json')]:
        p=A/(name+'.trace');m=np.memmap(p,mode='r',dtype=dtype);r=N.read(A/meta)
        assert len(m)==r['trace_records'] and np.all(m['end']>=m['start']) and np.all(m['start'][1:]>=m['end'][:-1])
        cap=np.array([524288,4194304,2048]);assert np.all(m['mem']<3) and np.all(m['reserved']==0)
        assert np.all(m['address']%4==0) and np.all(m['address'].astype(np.uint64)+m['words'].astype(np.uint64)*4<=cap[m['mem']])
        expected=np.where(m['vector'],(m['words']+15)//16,m['words']);assert np.array_equal(m['end']-m['start'],expected)
        writes=(m['mem']==0)&(m['write']==1);intersect=writes&(m['address']<400640)&(m['address'].astype(np.uint64)+m['words'].astype(np.uint64)*4>400000)
        ii=np.flatnonzero(intersect);assert len(ii)==(32 if name!='rowband128' else 16)
        expected_writes={(400000+i*32,8) for i in range(16)}
        if name!='rowband128':expected_writes|={(400512+i*8,2) for i in range(16)}
        assert {(int(m['address'][i]),int(m['words'][i])) for i in ii}==expected_writes
        # Each physical output interval is written exactly once; no later DMA,
        # scratch write or output overlaps it, unlike the retained V1 failure.
        for i in ii:
            lo=int(m['address'][i]);hi=lo+4*int(m['words'][i]);later=ii[ii>i]
            assert not np.any((m['address'][later]<hi)&(m['address'][later]+4*m['words'][later]>lo))
        dmarows=list(csv.DictReader((A/(name+'.trace.dma.csv')).open()));coverage=np.zeros(12288000//64,np.uint8);duration=0
        for row in dmarows:
            start,end,addr,size=[int(row[k]) for k in ['start','end','external_byte_address','bytes']]
            assert addr%64==0 and size%64==0 and end-start==32+size//64 and addr+size<=12288000
            coverage[addr//64:(addr+size)//64]+=1;duration+=end-start
        assert np.all(coverage==1) and len(dmarows)==r['DMA_transactions'] and duration==r['DMA_ticks'] and r['DMA_bytes']==12288000
        for mem in range(3):
            mm=m[m['mem']==mem];saved=r['memory'][mem]
            assert int(mm['address'].astype(np.uint64).__add__(mm['words'].astype(np.uint64)*4).max(initial=0))==saved['high_water_address']
            for write,key in [(0,'read_bytes'),(1,'write_bytes')]:assert int(mm['words'][mm['write']==write].astype(np.uint64).sum())*4==saved[key]
            ticks=int((mm['end']-mm['start']).sum())-(192000 if mem==0 else 0);assert ticks==saved['serial_service_ticks']
        recomputed=r['DMA_ticks']+sum(v['serial_service_ticks'] for v in r['memory'])+sum(oplat(k)*v for k,v in r['ops'].items());assert recomputed==r['feature_ready_tick']
        assert r['memory'][2]['high_water_address']==0
        tracecounts[name]=len(m);protected.append(name);servicereceipts[name]=dict(trace_sha256=N.sha(p),feature_ready_tick=recomputed,DMA_transactions=len(dmarows),IOS_high_water=r['memory'][0]['high_water_address'])
    x=[];ids=[];maxfeature=0.
    for row in streams['rows']:
        payload=np.fromfile(A/row['feature_file'],np.float32);image=np.fromfile(A/(row['feature_file']+'.ios'),np.float32);assert image.size==524288//4
        N.eq(image[400000//4:400000//4+160],payload);ix=feat['ids'].tolist().index(row['id']);N.eq(feat['x160'][ix],payload)
        role=row['role'];j=refs[role+'_ids'].tolist().index(row['id']);ref=refs[role+'_x160'][j];err=float(np.max(np.abs(payload-ref)));N.close(err,row['max_abs_feature_error']);maxfeature=max(maxfeature,err)
        ids.append(row['id']);x.append(payload)
    x=np.stack(x);N.eq(ids,feat['ids']);N.eq(ids,scores['ids']);assert len(heads['models'])==17
    independently_replayed={}
    for dim,branch in [(128,'voxel128_v1'),(160,'directional160_v1')]:
        model=N.read(R33/f'fitted/{branch}/MODEL_V1.json');s,_,_=N.suite(model,x[:,:dim],'fp32',{},trace=None)
        for name,value in s.items():independently_replayed[f'X{dim}_{name}']=value
        cal=N.read(B/f'ordinary/actual_v1/MODEL_{dim}_V1.json');aux=ridge(cal['family']['auxiliary'],x[:,:dim]);pp=N.parts(aux,1).transpose(0,2,1).reshape(len(x),-1)
        augmented=np.concatenate([x[:,:dim],aux,pp],axis=1)
        independently_replayed[f'X{dim}_calibrated_residual_g1']=N.signed(aux,1,np.float32)+np.float32(cal['selection']['shrink'])*ridge(cal['family']['residual'][str(cal['selection']['alpha'])],augmented)
    folded=N.read(B/'numeric/folded_constant_v/actual_v1/FOLDED_MODEL_V1.json')['ridge'];independently_replayed['X128_folded_constant_v']=ridge(folded,x[:,:128])
    modelreceipts={}
    for name,value in independently_replayed.items():
        N.eq(value,scores[name]);r=heads['models'][name];N.eq(np.asarray(r['scores'],np.float32),value[0]);events=N.read(A/(name+'_HEAD_EVENTS_V1.json'))
        operand=[e for e in events if e['space']=='IOS' and e['write'] is False];assert operand[0]['address']==400000
        for e in events:
            if e['space']=='EXTERNAL_TO_WSLOT':assert e['end']-e['start']==32+e['bytes']//64;continue
            cap={'IOS':524288,'WRF':32768,'WSLOT':1048576}[e['space']];assert e['address']%4==0 and e['address']+e['words']*4<=cap
            assert e['end']-e['start']==(e['words']+e['words_per_tick']-1)//e['words_per_tick']
        paid=[e for e in events if e['kind']!='weight_refill_destination'];total=sum(e['end']-e['start'] for e in paid)+sum(oplat(k)*v for k,v in r['arithmetic_ops'].items());assert total==r['action_ready_tick']
        assert r['serial_prefix_action_ready_first_identity']==r['feature_ready_tick_first_identity']+total
        assert r['feature_stream_action_flips_fixed']==r['feature_stream_action_flips_allow']==0
        modelreceipts[name]=dict(parameter_file_sha256=N.sha(A/(name+'_PARAMETERS_FP32_V1.bin')),event_file_sha256=N.sha(A/(name+'_HEAD_EVENTS_V1.json')),head_action_ready_tick=total,serial_prefix_action_ready_tick=r['serial_prefix_action_ready_first_identity'])
    report=dict(schema='flow_r34_hardware_v2_independent_audit_v1',status='PASS_REPAIRED_STORAGE_WITH_EXPLICIT_RESOURCE_ASSUMPTIONS',auditor_sha256=N.sha(__file__),
        stream_source_sha256=N.sha(D/'stream_features_v2.cpp'),head_source_sha256=N.sha(D/'head_prefix_v2.py'),resource_scope_sha256=N.sha(D/'RESOURCE_SCOPE_V2.md'),
        stream_receipt_sha256=N.sha(A/'STREAM_EXECUTION_V1.json'),head_receipt_sha256=N.sha(A/'HEAD_PREFIX_RESULTS_V1.json'),trace_records=tracecounts,
        repaired_protected_region=[400000,400640],IOS_images_actual_readback_verified=120,replayed_heads=17,max_feature_difference_vs_frozen_FP32=maxfeature,feature_services=servicereceipts,head_services=modelreceipts,
        v1_failure_receipt='HARDWARE_LIFETIME_FAILURE_V1.json',scope='IOS/RES/DMA recorded ranges and feature-to-head storage now connect. Scalar RF 2R1W, routing/control and blocking FP latencies remain unsynthesized assumptions. Unknown shared stem, no total-network or mixed-int16 cycle claim.',GPU_calls=0)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'trace_records':tracecounts,'head_count':17,'max_feature_error':maxfeature}))
if __name__=='__main__':main()
