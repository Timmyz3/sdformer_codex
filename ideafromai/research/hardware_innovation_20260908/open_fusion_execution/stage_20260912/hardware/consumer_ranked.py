"""Rank-parameterized direct PED continuation, copied from the existing integer runner.

Only PED latent widths are parameters; all original RNE/sat, complete-K,
raw-I24, gate order, port functions and true output egress are retained.
The old integer_chain.py stays unchanged. No late-V path is measured here.
"""
from integer_chain import *
from consumer_resident import dense_adapter as dense
from execute import directory

def run(data,q,label,preview_gate,stress=False,machine=None,handoff_native=False,late_v=False,rank=32):
    assert rank in (24,32) and not late_v, "This stage exercises direct completed PED only."
    geo=json.loads(str(data['window_geometry_json']))[label]
    h,w=geo['gate_shape'];oy,ox=geo['gate_origin'];out_y,out_x=geo['output_origin']
    m=Machine(stress) if machine is None else machine;begin=m.time
    before=dict(m.count);prior_stages=dict(m.stages)
    if machine is None:
        words=sum(preview_gate[t].astype(np.uint16)<<t for t in range(10)).transpose(1,2,0)
        m.phase='sn2_continuation_input';m.dma_input(words.astype('<u2').tobytes(),GATE)
    blob,base,bounds=coeffs(q,late_v)
    m.phase='integer_coefficient_replace';m.dma_input(blob,0,True)
    if late_v:
        # Both placements fit the same pool; native H32 tiles end at110592.
        # Relocate at the original coefficient load, not an extra late refill.
        m.dma_input(q['V_ped_q16'].T.astype('<i2').tobytes(),base['V_ped'],True)
        m.dma_input(q['PED_bias_q24'].astype('<i4').tobytes(),base['PED_bias'],True)
    sy,sx=geo['source_origin']
    updated=data[label+'_I24'][:,:,oy-sy:oy-sy+h,ox-sx:ox-sx+w].astype(np.int64).copy()
    continuous=np.empty((10,96,4,4),np.int64);u_output=np.empty((10,rank,4,4),np.int64)
    spill_width=rank if late_v else 96
    spill=bytearray(16*10*spill_width*3)
    anchors={(2*(out_y+dy),2*(out_x+dx)) for dy in range(4) for dx in range(4)}
    rows=[]
    for y in range(oy,oy+h):
        for anchor in (True,False):
            xs=[x for x in range(ox,ox+w) if ((y,x) in anchors)==anchor]
            for i in range(0,len(xs),2):
                positions=[(y,x) for x in xs[i:i+2]];count=len(positions);start=m.time
                feed_identity(m,data,label,geo,positions)
                if anchor:
                    n,live=directory(m,geo,positions)
                    sparse_u(m,n,count,base,q)
                    dense(m,base,'F',16,96,LAT16,UPDATED,count,int(q['F_exponent']),live)
                    merge(m,base,count)
                input_base=UPDATED if anchor else I
                actual_updated=read24(m,input_base,(count,10,96))
                projection_gates(m,base,geo,positions,input_base)
                gate_ready=m.time
                if anchor:
                    dense(m,base,'U_ped',96,rank,UPDATED,PED_U,count,int(q['U_ped_exponent']))
                    u_ready=m.time;actual_u=read24(m,PED_U,(count,10,rank))
                    if not late_v:
                        dense(m,base,'V_ped',rank,96,PED_U,PED_V,count,int(q['V_ped_exponent']),bias='PED_bias')
                        actual_ped=read24(m,PED_V,(count,10,96))
                    m.phase='latent_U_PED_egress' if late_v else 'continuous_PED_egress'
                    tile_payload=bytearray()
                    for off in range(0,count*10*spill_width*3,32):
                        payload=b''.join(m.read_word((PED_U if late_v else PED_V)+off+j) for j in range(0,32,8));assert len(payload)==32
                        tile_payload.extend(payload)
                        for _ in range(5):m.advance(tag='PED_DMA_output_slots')
                for ip,(yy,xx) in enumerate(positions):
                    updated[:,:,yy-oy,xx-ox]=actual_updated[ip]
                    if anchor:
                        dy,dx=yy//2-out_y,xx//2-out_x
                        if not late_v:continuous[:,:,dy,dx]=actual_ped[ip]
                        u_output[:,:,dy,dx]=actual_u[ip]
                        stride=10*spill_width*3;offset=(dy*4+dx)*stride
                        spill[offset:offset+stride]=tile_payload[ip*stride:(ip+1)*stride]
                rows.append(dict(positions=positions,anchor=anchor,start=start,gate_ready=gate_ready,
                    U_ready=u_ready if anchor else None,end=m.time))
    if not handoff_native:
        m.phase='projection_gate_egress'
        for off in range(0,h*w*192,32):
            payload=b''.join(m.read_word(PROJ+off+j) for j in range(0,32,8));assert len(payload)==32
            for _ in range(5):m.advance(tag='gate_DMA_output_slots')
    words=np.frombuffer(m.state,'<u2',count=h*w*96,offset=PROJ).reshape(h,w,96).copy()
    gate=np.stack([(words>>t)&1 for t in range(10)]).transpose(0,3,1,2).astype(bool)
    checks=dict(updated=difference(updated,data[label+'_updated_I24']),
        projection_gate=difference(gate,data[label+'_proj_gate']))
    if not late_v:checks['PED']=difference(continuous,data[label+'_continuous_q24'])
    assert all(v['differences']==0 for v in checks.values()),checks
    report=dict(PED_rank=rank,service_slots=m.time-begin,stages={k:v-prior_stages.get(k,0) for k,v in m.stages.items() if v-prior_stages.get(k,0)},
        counts={k:v-before.get(k,0) for k,v in m.count.items() if v-before.get(k,0)},checks=checks,
        coefficients_bytes=len(blob)+(6528 if late_v else 0),legal_accumulator48_bounds=bounds,state_high_water=PROJ+h*w*192,rows=rows,
        input_contract='Each raw I24 pixel is 2880 contiguous bytes, spatial/T/C order; source-side packing not covered.',
        common_baseline='Defer original V/bias/sat until global BN readiness; preserve actual U24 spill.' if late_v else 'Direct V completion, actual V24 spill for post-BN reread.',
        PED_spill_bytes=len(spill),PED_spill_width=spill_width,
        V_coefficients_retained_across_native=late_v,
        full_chain_closed=False)
    return dict(updated=updated,continuous=continuous if not late_v else np.empty(0,np.int64),gate=gate,U_ped=u_output,
        PED_spill=np.frombuffer(spill,np.uint8).copy()),report
