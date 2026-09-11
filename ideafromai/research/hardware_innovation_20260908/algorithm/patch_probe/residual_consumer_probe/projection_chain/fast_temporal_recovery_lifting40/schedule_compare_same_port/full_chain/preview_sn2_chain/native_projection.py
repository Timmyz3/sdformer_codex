"""Full K864/H96 native projection after actual resident projection gates.

Three H32 coefficient tiles each fit128KiB. Build and retain eight complete
P2 NRV directories, reuse across all three tiles; they occupy at most55296B.
Projection gates remain in SRAM at65536. FP32 output staging starts57344.
All raw projection values are written to the external full-BN-domain buffer.
This local window never substitutes captured global BN statistics as free data.
"""
import json
import numpy as np
from integer_chain import directory,PROJ
from run_chain import tf32_round,difference

OUTPUT=57344


def run(m,data,p,label):
    geo=json.loads(str(data['window_geometry_json']))[label]
    out_y,out_x=geo['output_origin'];start=m.time
    counts=dict(m.count);stages=dict(m.stages)
    groups=[]
    for dy in range(4):
        for dx in (0,2):
            positions=[(2*(out_y+dy),2*(out_x+dx+i)) for i in range(2)]
            dir_base=len(groups)*6912
            n,live=directory(m,geo,positions,PROJ,dir_base,'native_NRVs_from_actual_projection_gate')
            groups.append(dict(positions=positions,base=dir_base,n=n,live=live))
    assert float(p['proj_theta_output'])==1.0
    weight=tf32_round(p['proj_weight_fp32'].reshape(96,864))
    value=np.empty((10,96,4,4),np.float32)
    for h0 in (0,32,64):
        blob=weight[h0:h0+32].T.astype('<f4').tobytes()
        bias_base=len(blob)
        if bool(p['proj_has_bias']):blob+=p['proj_bias_fp32'][h0:h0+32].astype('<f4').tobytes()
        assert len(blob)<=131072
        m.phase='native_H32_coefficient_replace';m.dma_input(blob,0,True)
        for group in groups:
            m.phase='native_projection_full_K864'
            for r in range(80):m.advance(op=('clear',r,None),tag='native_clear')
            for i in range(group['n']):
                raw=m.read_word(group['base']+i*8)
                k=int.from_bytes(raw[:4],'little');mask=int.from_bytes(raw[4:],'little')
                m.advance(tag='native_NRV_select')
                for hg in range(4):
                    m.coefficient((k*32+hg*8)*4)
                    if not np.frombuffer(m.cword,'<f4').any():continue
                    for tp in range(20):
                        if mask&(1<<tp):
                            r=tp*4+hg;m.wait_reg(r)
                            m.advance(op=('FMA',r,(None,None)),tag='native_active_FMA')
            m.drain()
            if bool(p['proj_has_bias']):
                for hg in range(4):
                    m.coefficient(bias_base+hg*32)
                    for tp in range(20):m.advance(op=('add_coef',tp*4+hg,None),tag='native_bias')
                m.drain()
            for ip,(y,x) in enumerate(group['positions']):
                dy,dx=y//2-out_y,x//2-out_x
                for t in range(10):
                    for hg in range(4):
                        r=(ip*10+t)*4+hg
                        value[t,h0+hg*8:h0+hg*8+8,dy,dx]=m.rf[r]
                        m.store_reg(r,OUTPUT+((ip*10+t)*32+hg*8)*4)
            m.phase='native_projection_to_global_BN_buffer'
            for off in range(0,2560,32):
                payload=b''.join(m.read_word(OUTPUT+off+j) for j in range(0,32,8));assert len(payload)==32
                for _ in range(5):m.advance(tag='native_DMA_output_slots')
    report=dict(service_slots=m.time-start,counts={k:v-counts.get(k,0) for k,v in m.count.items() if v-counts.get(k,0)},
        stages={k:v-stages.get(k,0) for k,v in m.stages.items() if v-stages.get(k,0)},
        checks=difference(value,data[label+'_proj_conv_fp32']),K=864,H=96,T=10,spatial_outputs=16,
        coefficient_tiles=3,bytes_per_tile=110592,native_raw_output_bytes=61440,
        directory_allocation_bytes=55296,full_global_BN_domain=False,
        FP32_order='Ascending original K, correctly rounded fmaf for every retained term; CUDA reduction order can differ.')
    return value,report
