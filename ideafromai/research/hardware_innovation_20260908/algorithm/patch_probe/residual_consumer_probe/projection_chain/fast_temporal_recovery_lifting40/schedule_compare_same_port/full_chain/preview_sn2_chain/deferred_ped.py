"""Local PED completion using actual spill bytes and computed full-domain BN.

The external mu/rsqrt record is produced by global_bn.cpp, not captured CUDA
statistics. Its full-domain projection inputs outside these windows are still
captured; missing global producer service/wait is explicitly not invented.
"""
import json
from pathlib import Path
import numpy as np
from integer_chain import dense,read24,PED_U,PED_V
from run_chain import difference

AFFINE,RAW,FINAL=24576,81920,90112


def computed_stats(axis,stress):
    mode='pairwise_stress' if stress else 'pairwise_ready'
    path=Path('/tmp/tcasii_global_bn_work_20260911')/mode/(axis+'_stats.f32')
    return np.fromfile(path,'<f4').reshape(3,96),str(path)


def run(m,data,p,q,label,values,stats,late_v):
    begin=m.time;counts=dict(m.count);prior=dict(m.stages)
    # The independent full-domain kernel uses low coefficient addresses only;
    # the common128KiB pool can preserve the deferred V/bias high allocation.
    base={'V_ped':114688,'PED_bias':120832}
    if late_v:
        assert bytes(m.coef[114688:120832])==q['V_ped_q16'].T.astype('<i2').tobytes()
        assert bytes(m.coef[120832:121216])==q['PED_bias_q24'].astype('<i4').tobytes()
    blob=np.concatenate([stats[0],stats[2],p['proj_bn_gamma'],p['proj_bn_beta'],np.full(8,2**-14)]).astype('<f4').tobytes()
    m.phase='BN_computed_statistics_and_affine_input';m.dma_input(blob,0,True)
    for hg in range(12):
        # Same separated FP32 scale and bias operations as the full BN unit.
        for reg,offset in ((0,768),(1,0),(2,1152)):
            m.coefficient(offset+hg*32)
            m.advance(op=('load',reg,np.frombuffer(m.cword,'<f4').copy()),tag='BN_affine_coefficient_load')
            m.wait_reg(reg)
        m.coefficient(384+hg*32)
        m.advance(op=('mul_coef',0,None),tag='BN_gamma_rsqrt');m.wait_reg(0)
        m.advance(op=('mul_reg',1,0),tag='BN_mean_scale');m.wait_reg(1)
        m.advance(op=('sub_reg',2,1),tag='BN_beta_bias');m.wait_reg(2)
        m.store_reg(0,AFFINE+hg*32);m.store_reg(2,AFFINE+384+hg*32)
    spill=values['PED_spill'].tobytes();width=32 if late_v else 96
    raw=values['native_projection'];continuous=np.empty_like(data[label+'_continuous_q24'])
    final=np.empty_like(data[label+'_ped_output_fp32']);norm=np.empty_like(final)
    for dy in range(4):
        for dx in (0,2):
            m.phase='PED_post_BN_spill_read'
            first=(dy*4+dx)*10*width*3
            m.dma_input(spill[first:first+2*10*width*3],PED_U if late_v else PED_V)
            if late_v:
                dense(m,base,'V_ped',32,96,PED_U,PED_V,2,int(q['V_ped_exponent']),bias='PED_bias')
            actual=read24(m,PED_V,(2,10,96))
            m.phase='native_raw_post_BN_read'
            raw_tile=raw[:,:,dy,dx:dx+2].transpose(2,0,1).copy()
            m.dma_input(raw_tile.astype('<f4').tobytes(),RAW)
            m.phase='post_BN_normalize_and_PED_merge'
            for ip in range(2):
                for t in range(10):
                    for hg in range(12):
                        tp=ip*10+t;h=hg*8
                        m.load_reg(0,RAW+(tp*96+h)*4)
                        m.load_reg(1,AFFINE+hg*32)
                        m.advance(op=('mul_reg',0,1),tag='native_BN_scale');m.wait_reg(0)
                        m.load_reg(2,AFFINE+384+hg*32)
                        m.advance(op=('add_reg',0,2),tag='native_BN_bias');m.wait_reg(0)
                        norm[t,h:h+8,dy,dx+ip]=m.rf[0]
                        m.load_i24(3,PED_V+(tp*96+h)*3)
                        m.advance(op=('i24_to_float',3,None),tag='PED_I24_to_FP32');m.wait_reg(3)
                        m.coefficient(1536)
                        m.advance(op=('mul_coef',3,None),tag='PED_Q14_scale');m.wait_reg(3)
                        m.advance(op=('add_reg',0,3),tag='final_PED_add');m.wait_reg(0)
                        m.store_reg(0,FINAL+(tp*96+h)*4)
                continuous[:,:,dy,dx+ip]=actual[ip]
            m.phase='final_PED_output'
            for off in range(0,2*10*96*4,32):
                payload=b''.join(m.read_word(FINAL+off+j) for j in range(0,32,8))
                tp,h=divmod(off//4,96);ip,t=divmod(tp,10)
                final[t,h:h+8,dy,dx+ip]=np.frombuffer(payload,'<f4')
                for _ in range(5):m.advance(tag='final_PED_DMA_output_slots')
    checks=dict(continuous_q24=difference(continuous,data[label+'_continuous_q24']),
        normalized_vs_CUDA=difference(norm,data[label+'_proj_norm_fp32']),
        final_vs_CUDA=difference(final,data[label+'_ped_output_fp32']))
    assert checks['continuous_q24']['differences']==0,checks
    return dict(continuous=continuous,normalized=norm,final=final),dict(
        service_slots=m.time-begin,counts={k:v-counts.get(k,0) for k,v in m.count.items() if v-counts.get(k,0)},
        stages={k:v-prior.get(k,0) for k,v in m.stages.items() if v-prior.get(k,0)},checks=checks,
        spill_read_bytes=len(spill),native_raw_read_bytes=61440,final_write_bytes=61440,
        BN_statistics_origin='Actual full-domain pairwise Engine output; input projection outside local windows remains captured',
        BN_full_producer_wait_slots=None,full_layer_closed=False)
