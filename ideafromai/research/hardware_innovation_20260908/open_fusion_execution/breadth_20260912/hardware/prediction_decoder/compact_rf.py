from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prediction_decoder_base',HERE/'execute.py')
dec=importlib.util.module_from_spec(spec);spec.loader.exec_module(dec)


def preload(m):
    m.phase='compact_table_RF_preload'
    for start in range(0,640,8):
        payload=bytearray();cursor=start
        while cursor<start+8:
            row,t=divmod(cursor,10);count=min(start+8-cursor,10-t)
            m.advance(tag='compact_table_segment_address')
            m.coefficient(row*32+t*3)
            offset=t*3
            payload.extend(m.cword[offset:offset+count*3]);cursor+=count
            m.count['compact_table_segments']+=1
        assert len(payload)==24
        m.pred_unpack=bytes(payload);dst=16+start//8
        m.wait_reg(dst);m.advance(op=('IPRED_UNPACK24',dst,None),tag='compact_table_packed24_RF_load')
        m.wait_reg(dst)


def run(words,D,c,stress):
    image,meta,proof=dec.constant_image(D,c,'LUT5plus5')
    m=dec.DecoderMachine(stress)
    gatebytes=words.astype('<u2').tobytes();m.state[:len(gatebytes)]=gatebytes
    m.phase='predictor_constant_cold_fill';m.dma_input(image,0,True)
    preload(m)
    halves,_=dec.all_patterns(D,c)
    expected_table=np.concatenate(halves).reshape(-1)
    assert np.array_equal(m.rf[16:96].astype(np.int64).reshape(-1),expected_table)
    setup=m.time;group_counts=[]
    for group in range(len(words)//8):
        m.phase='predictor_gate_input';m.load_gates(group*16)
        m.phase='compact_RF_prediction'
        for t in range(10):
            m.wait_reg(t);m.advance(op=('ILOAD',t,np.zeros(8,np.int64)),tag='predictor_clear')
        for half in range(2):
            covered=0;groups=0
            while covered!=255:
                m.advance(tag='LUT_next_uncovered_lane')
                lane=next(l for l in range(8) if not covered&(1<<l))
                key=(int(m.rf[10,lane])>>(half*5))&31
                m.select_mask(half=half,key=key)
                covered|=sum(int(v)<<l for l,v in enumerate(m.predictor_mask));groups+=1
                row=half*32+key
                if row in meta['zero_rows']:
                    m.advance(tag='LUT_static_zero_row_skip');continue
                m.count['compact_RF_row_lookups']+=1
                for t in range(10):
                    # Existing scalar controller computes one address, then
                    # existing two-read-port masked broadcast consumes it.
                    # There is never8 simultaneous different RF addresses.
                    m.advance(tag='compact_RF_scalar_address')
                    index=row*10+t;r=16+index//8;l=index%8
                    assert 16<=r<96
                    m.wait_reg(t)
                    m.advance(op=('IPRED_MASKED_CACHED_ADD',t,(r,l)),tag='compact_RF_prediction_add')
            group_counts.append(groups)
        m.phase='predictor_output_store'
        for t in range(10):m.store_i24(t,dec.OUTPUT+(group*10+t)*24)
    actual=dec.shared.integer.read24(m,dec.OUTPUT,(len(words)//8,10,8)).transpose(0,2,1).reshape(-1,10)
    expected=np.array([D@((w>>np.arange(10))&1)+c for w in words],np.int64)
    assert np.array_equal(actual,expected)
    assert np.array_equal(m.rf[16:96].astype(np.int64).reshape(-1),expected_table),'table overwritten by working state'
    assert sum(m.stages.values())==m.time
    return dict(method='LUT5plus5_compact_RF',stress=stress,service_slots=m.time,
        cold_fill_and_cache_slots=setup,post_setup_service_slots=m.time-setup,stages=dict(m.stages),counts=dict(m.count),
        coefficient_bytes=2048,RF_table_vectors=80,RF_live_vectors=91,RF_budget=96,
        new_ISA_operations=[],arbitrary_lane_gather=False,
        reads_per_prediction_issue='acc RF plus ONE table RF; existing scalar selector/broadcast, masked subgroup only',
        per_scalar_address_control_slots=m.count['compact_RF_scalar_address'],
        preload_segments=m.count['compact_table_segments'],
        actual_cache_values=640,actual_cache_differences=0,cache_values_preserved_after_all_outputs=True,
        output_values=int(actual.size),differences=0,
        port_bytes=dict(SR64=m.count['SR64_reads']*8,SW64=m.count['SW64_writes']*8,
            CR256=m.count['CR256_reads']*32,CW256=m.count['CW256_writes']*32),
        half_groups=group_counts,all_patterns=proof)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');a=ap.parse_args()
    original=json.loads((HERE/('stress.json' if a.stress else 'ready.json')).read_text())
    result=dict(scope='Same ready-gate decoder leaf. Existing ISA and one-index-subgroup-at-a-time RF addressing; actual compact table cold preload.',rows=[])
    path=HERE/('compact_stress.json' if a.stress else 'compact_ready.json')
    for axis in (['ordinary'] if a.stress else ['ordinary','lifting_raw']):
        p=dict(np.load(dec.PARAMETERS/(axis+'.npz')))
        for label in (['interior'] if a.stress else ['corner','interior']):
            row=run(dec.actual_words(axis,label),p['full_g_q8_D'],p['full_g_q8_c'],a.stress)
            row.update(axis=axis,window=label)
            same=[x for x in original['rows'] if x['axis']==axis and x['window']==label and x['mode']=='full_g_q8']
            seq=next(x for x in same if x['method']=='condition_add')
            cr=next(x for x in same if x['method']=='LUT5plus5')
            row.update(condition_add_service=seq['service_slots'],CR_table_service=cr['service_slots'],
                change_vs_condition_add_percent=100*(row['service_slots']/seq['service_slots']-1),
                change_vs_CR_table_percent=100*(row['service_slots']/cr['service_slots']-1))
            result['rows'].append(row);path.write_text(json.dumps(result,indent=2)+'\n')
            print(axis,label,row['service_slots'],row['change_vs_condition_add_percent'],row['port_bytes'],flush=True)
    result['complete']=True;path.write_text(json.dumps(result,indent=2)+'\n')
    print('COMPACT_RF_COMPLETE',flush=True)


if __name__=='__main__':main()
