#!/usr/bin/env python3
"""Bind RTL fixtures to the exact accepted dense/pruned model artifacts."""
from pathlib import Path
import json
import numpy as np
from analyze import readhex

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'data_and_quality'

def verify():
    capture=np.load(DATA/'r0_contiguous_t10.npz',allow_pickle=False)
    records={}
    for arm,file in [('dense','dense_q16.npz'),('physical25','physical25_q16.npz'),('magnitude25','magnitude25_q16.npz')]:
        model=np.load(DATA/file,allow_pickle=False)
        assert np.array_equal(model['source_bits'],capture['source_bits'])
        assert np.array_equal(model['input_origin_yx'],capture['input_origin_yx'])
        mask=np.repeat(np.repeat(model['live'],8,axis=0),4,axis=1)[:,:,None,None]
        assert np.array_equal(model['weight_q16'],capture['weight_q16']*mask)
        for tile in range(8):
            path=HERE/'fixtures'/f'real_{arm}_tile{tile}'
            assert np.array_equal(readhex(path/'mask.hex').reshape(12,24),model['live'])
            assert np.array_equal(readhex(path/'origin.hex').view(np.int32),model['input_origin_yx'][tile])
            gold=readhex(path/'gold.hex').view(np.int32)
            official=model['golden_accum'][tile].reshape(10,12,8,4).transpose(1,3,0,2).reshape(-1)
            assert np.array_equal(gold,official)
        records[arm]={'model_artifact':str((DATA/file).resolve()),'native_source_identity':True,
            'masked_weight_identity':True,'origin_identity':True,'all_official_integer_gold_equal':True,
            'gold_values_checked':30720,'live_groups':int(model['live'].sum()),
            'coefficient_nonzero':int(np.count_nonzero(model['weight_q16']))}
    (HERE/'input_identity.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'artifacts_checked':3,'official_gold_values_checked':92160,'all_equal':True}))

if __name__=='__main__':verify()
