#!/usr/bin/env python3
"""Independently compare returned CPU reconstructions to frozen R13 input arrays."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATE = HERE.parents[1]
SOURCE = DATE / 'event_source_round14'
SNAP = DATE / 'coverage_recovery_round13/input_snapshot'
EXPECTED = {'zurich_city_10_a_0554':'dc47271d7ffb5945236ba5b0c939c1a8b3b0240ca069354bc297cd4cffee3951',
            'zurich_city_10_a_0475':'df939c9efd8cd8786bd9fa4ed73e3b528950d502b660248923f8da53d17663be'}
RAW = {'train_events/zurich_city_10_a/events/left/events.h5':'9e9b62d167735a67f97c4480c504f3ade378702733a9f23b9022bed1c82e77b7',
       'train_events/zurich_city_10_a/events/left/rectify_map.h5':'7d3efd1c2745f8967a2d9e88e2f954ed56853b826ba488a7f6c3072c051d9038',
       'train_optical_flow/zurich_city_10_a/flow/forward_timestamps.txt':'67df5b6c6a028b7c12f7271e861b8cacc4db27733830e504ebea8cc59d227f43'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def main():
    script=SOURCE/'reconstruct_two_inputs.py'
    assert sha(script)=='29e9105b6345e1213f344bde2b1161e8bad1b6dde76d6d5c39d343d0d7fd96c8'
    assert sha(SNAP/'receipt.json')=='2188cce3a09235dc5c96b793e94c025433b4154bd09ffd8ee15c2dcd9a3df8b6'
    result_path=SOURCE/'run_v1/result.json';result=read(result_path)
    observed_path=SOURCE/'ORIGINAL_SOURCE_FULLHASH_V1.json';observed=read(observed_path)
    assert result['schema']=='flow_fixed_two_event_source_reconstruction_v1' and result['status']=='PASS'
    assert result['script_sha256']==sha(script) and result['representation_sha256']=='f02493e7440e12828b487b272d0c17f96a1a2a6c785fd8a6764981b09825403a'
    assert result['cuda_initialized'] is False and result['full_model_forwards']==0
    assert observed['GPU_used'] is False and observed['sha256']==RAW
    assert {k:v['sha256'] for k,v in result['raw_source_files'].items()}==RAW
    assert [r['target_id'] for r in result['records']]==list(EXPECTED)
    snapshot={r['target_id']:r for r in read(SNAP/'receipt.json')['all_files'] if r['kind']=='events'}
    rows=[]
    for rec in result['records']:
        tid=rec['target_id'];frozen=snapshot[tid]
        relative='saved_flow_data/event_tensors/10bins/left/zurich_city_10_a/'+tid+'.npy'
        assert frozen['relative_path']==relative
        old=SNAP/relative;new=SOURCE/'run_v1'/(tid+'.npy')
        assert sha(old)==sha(new)==frozen['sha256']==frozen['expected_sha256']==EXPECTED[tid]
        assert rec['stored_file_sha256']==rec['reconstructed_file_sha256']==EXPECTED[tid]
        x,y=np.load(old,allow_pickle=False),np.load(new,allow_pickle=False)
        assert x.shape==y.shape==(10,480,640) and x.dtype==y.dtype==np.float32
        assert np.isfinite(x).all() and np.isfinite(y).all() and np.array_equal(x,y)
        payload=hashlib.sha256(y.tobytes(order='C')).hexdigest()
        assert payload==rec['payload_sha256'] and rec['array_equal'] is True and rec['max_absolute_difference']==0
        lo,hi=rec['event_index_range_half_open'];assert 0<=lo<hi
        assert set(rec['raw_slice_sha'])=={'p','x','y','t'} and all(v['count']==hi-lo for v in rec['raw_slice_sha'].values())
        rows.append({'target_id':tid,'event_count':hi-lo,'timestamp_pair_us':rec['timestamp_pair_us'],
                     'full_file_sha256':sha(new),'payload_sha256':payload,'array_equal':True,
                     'elements':int(y.size),'maximum_absolute_difference':float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))))})
    out={'schema':'flow_two_event_input_independent_verification_v1','status':'PASS','auditor_sha256':sha(Path(__file__)),
         'reconstruction_result_sha256':sha(result_path),'original_source_observation_sha256':sha(observed_path),
         'frozen_snapshot_receipt_sha256':sha(SNAP/'receipt.json'),'raw_source_sha256':RAW,'records':rows,'GPU_used':False,
         'scope':'Independent local whole-array comparison of two returned CPU reconstructions to frozen experiment inputs; source H5 slice execution evidenced by root original-source hashes and reviewed pinned reconstruction script. Shared upstream voxel math, no whole-dataset or representation-design proof, no quality/gate change.'}
    path=HERE/'EVENT_SOURCE_TWO_INPUT_AUDIT_V1.json'
    with path.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({'status':'PASS','records':rows,'receipt_sha256':sha(path)},indent=2))


if __name__=='__main__':main()
