"""Stream complete preselected gate windows through one finite preview machine.

Weights and input halo are loaded once per window, not once per P2. Odd width
uses a real one-position last group; no invented in-image zero padding.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from run_chain import *


def window(data,p,label,compact,stress=False,handoff=False,source_axis=None):
    geo=json.loads(str(data['window_geometry_json']))[label]
    source=data[label+'_sn1_gate'];words=np.zeros(source.shape[1:],np.uint16)
    for t in range(10): words|=source[t].astype(np.uint16)<<t
    blob,base,info=coefficients(p,compact)
    m=Machine(stress)
    m.phase='coefficient_cold_fill';m.dma_input(blob,0,True)
    source_report=None
    if source_axis is None:
        m.phase='sn1_halo_input';m.dma_input(words.astype('<u2').tobytes(),SRC)
    else:
        from source_program import run as source_run
        source_report=source_run(m,data,source_axis,label)
    h,w=geo['gate_shape'];oy,ox=geo['gate_origin']
    z=np.empty((10,32,h,w),np.float32);raw=np.empty((10,96,h,w),np.float32);y=np.empty_like(raw)
    rows=[]
    for ly in range(h):
        for lx in range(0,w,2):
            count=min(2,w-lx)
            positions=[(oy+ly,ox+lx+i) for i in range(count)]
            start=m.time
            n,occ,live=build_nrv(m,words,geo['source_origin'],positions)
            zz=execute_u(m,n,base,compact,count)
            rr=execute_v_bn(m,base,compact,live,count)
            yy=np.frombuffer(m.state,'<f4',count=count*10*96,offset=Y).reshape(count,10,96).copy()
            execute_sn2(m,base,p['preview_A'],count,GATE+(ly*w+lx)*192)
            for ip in range(count):
                z[:,:,ly,lx+ip]=zz.reshape(count,10,32)[ip]
                raw[:,:,ly,lx+ip]=rr.reshape(count,10,96)[ip]
                y[:,:,ly,lx+ip]=yy[ip]
            rows.append(dict(y=ly,x=lx,positions=count,NRV_rows=n,source_occurrences=occ,
                nonempty_TP=live.bit_count(),start=start,gate_ready=m.time))
    actual=np.frombuffer(m.state,'<u2',count=h*w*96,offset=GATE).reshape(h,w,96).copy()
    if not handoff:
        m.phase='sn2_gate_egress'
        for off in range(0,h*w*192,32):
            payload=b''.join(m.read_word(GATE+off+j) for j in range(0,32,8))
            assert len(payload)==32
            for _ in range(5):m.advance(tag='DMA_output_slots')
    gate=np.stack([(actual>>t)&1 for t in range(10)]).transpose(0,3,1,2).astype(bool)
    checks=dict(Z=difference(z,data[label+'_preview_Z_shared']),raw=difference(raw,data[label+'_preview_shared_raw']),
        BN1=difference(y,data[label+'_preview_BN1_Y']),sn2=difference(gate,data[label+'_sn2_gate']))
    report=dict(service_slots=m.time,stages=dict(m.stages),counts=dict(m.count),parameters=info,
        actual_gate_shape=list(gate.shape),source_halo_shape=list(source.shape),checks=checks,
        P2_rows=rows,max_pending=m.max_pending,state_high_water=GATE+h*w*192,
        common_source_empty_skip='Only skip V for TP whose COMPLETE K864 source has no events; mandatory NRV construction discovers this, BN1 bias and all sn2 dependents still run.',
        interface='Single resident coefficient/halo fill per entire captured window. Gate cache fully materialized before downstream integer/native consumer. Whole BN domain not included.')
    if source_report is not None:report['source_program']=source_report
    value=dict(z=z,raw=raw,y=y,gate=gate)
    return (value,report,m) if handoff else (value,report)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stress',action='store_true');args=parser.parse_args()
    result=dict(scope=__doc__,axes={},complete_chain=False,new_training=False,new_quantization=False)
    for axis in ('ordinary','lifting_raw'):
        path=HERE.parent/'capture'/axis
        data=read_npz(path/'000_zurich_city_09_a_0001.npz');p=read_npz(path/'live_parameters.npz')
        result['axes'][axis]={}
        for label in ('corner','interior'):
            values={};reports={}
            for encoding in ('expanded_fp32','existing_u8_shift'):
                values[encoding],reports[encoding]=window(data,p,label,encoding=='existing_u8_shift',args.stress)
                r=reports[encoding]
                print(axis,label,encoding,r['service_slots'],'gate_diff',r['checks']['sn2']['differences'],flush=True)
            delta={key:difference(values['expanded_fp32'][key],values['existing_u8_shift'][key]) for key in values['expanded_fp32']}
            assert all(d['differences']==0 for d in delta.values())
            reports['encoding_value_comparison']=delta
            result['axes'][axis][label]=reports
            # Reusable local payload for the next consumer; Git ignores NPZ.
            if not args.stress:
                np.savez_compressed(HERE/f'{axis}_{label}_preview_output.npz',**values['existing_u8_shift'])
    name='windows_stress.json' if args.stress else 'windows.json'
    (HERE/name).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
