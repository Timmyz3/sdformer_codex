"""Freeze one spatial-R16 integer chain and emit real-input reference outputs.

CPU NumPy only. No fitting, training, GPU or RTL. Existing trees are read-only.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent;B=H.parents[1]
D=B/'open_fusion_execution/major_operator_fusions_20260913/decomposition_owned/results'
R=B/'r8_consumer_fusion_20260914/data'

def rne_shift(x,shift,bits):
 x=np.asarray(x,np.int64);q=x//(1<<shift);r=x-q*(1<<shift)
 q=q+((r>(1<<(shift-1)))|((r==(1<<(shift-1)))&((q&1)!=0)))
 return np.clip(q,-(1<<(bits-1)),(1<<(bits-1))-1).astype(np.int64)

def identity_to_j(x):
 x=np.asarray(x,np.float32)
 if not np.all(np.isfinite(x)):raise ValueError('NaN/Inf identity is outside finite deployment contract')
 return np.clip(np.rint(x.astype(np.float64)*(1<<20)),-(1<<31),(1<<31)-1).astype(np.int64)

def freeze():
 with np.load(D/'spatial_r16_w8.npz') as z:f={k:z[k].copy() for k in z.files}
 with np.load(D/'spatial_r16_w32.npz') as z:w32={k:z[k].copy() for k in z.files}
 frozen=json.loads((R/'frozen_parameters.json').read_text())['arrays']
 old={k:np.asarray(a['values'],dtype=a['dtype']).reshape(a['shape']) for k,a in frozen.items()}
 theta=float(f['theta']);s1=f['scale_first'].reshape(16).astype(np.float64)
 q1=np.rint(f['first'].astype(np.float64)/f['scale_first']).astype(np.int64)[:,:,:,0]
 assert np.max(abs(q1))<=127 and np.array_equal((q1[:,:,:,None]*f['scale_first']).astype(np.float32),f['first'])
 # Keep W8's exact codes. Absorb all per-rank scale and source theta into Q2.
 # Signed13 is chosen once to fit the existing13-bit multiplier operand;
 # static bounds below admit signed32 p for any partial reduction order.
 effective=f['second'].astype(np.float64)[:,:,0,:]*s1[None,:,None]*theta
 scale=np.max(abs(effective),axis=(1,2))/4095.0
 assert np.all(scale>0)
 q2=np.rint(effective/scale[:,None,None]).astype(np.int64)
 assert np.max(abs(q2))<=4095
 bias=f['bias'].astype(np.float64)
 gain=old['BN_gain'].astype(np.float64);offset=old['BN_offset'].astype(np.float64)
 aq=np.rint(scale*gain*(1<<40)).astype(np.int64)
 bq=np.rint((offset+gain*bias)*(1<<20)).astype(np.int64)
 zlo=np.minimum(q1,0).sum((1,2));zhi=np.maximum(q1,0).sum((1,2));za=np.maximum(-zlo,zhi)
 expanded=np.einsum('orx,rcy->ocyx',q2,q1)
 plo=np.minimum(expanded,0).sum((1,2,3));phi=np.maximum(expanded,0).sum((1,2,3))
 pprefix=(abs(q2)*za[None,:,None]).sum((1,2));product=abs(q2)*za[None,:,None]
 wide=np.maximum(-plo,phi)*abs(aq)+((1<<31)+abs(bq))*(1<<20)
 assert zlo.min()>=-(1<<14) and zhi.max()<(1<<14)
 assert pprefix.max()<(1<<31) and wide.max()<(1<<63)
 assert np.max(abs(aq))<(1<<31) and np.max(abs(bq))<(1<<31)
 result=dict(q1=q1.astype(np.int8),q2=q2.astype(np.int16),output_scale=scale,a_q40=aq.astype(np.int32),b_q20=bq.astype(np.int32),
  theta=np.array(theta),bias=bias,BN_gain=gain,BN_offset=offset,first_scale=s1,
  q1_bits=np.array(8),z_bits=np.array(15),q2_bits=np.array(13),p_bits=np.array(32),wide_bits=np.array(64),
  expanded_int32=expanded.astype(np.int32),z_lower=zlo,z_upper=zhi,p_lower=plo,p_upper=phi,p_any_prefix_abs=pprefix,wide_abs_bound=wide)
 np.savez_compressed(H/'factors.npz',**result)
 fields=['q1','q2','output_scale','a_q40','b_q20','theta','bias','BN_gain','BN_offset']
 js=dict(schema='spatial_r16_q8_q13_i24_v1',target='sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0',
  geometry='q1[r,c,ky] vertical3x1 pad(1,0); q2[o,r,kx] horizontal1x3 pad(0,1); stride1',
  arrays={k:dict(shape=list(result[k].shape),dtype=str(result[k].dtype),values=result[k].tolist()) for k in fields})
 (H/'frozen_parameters.json').write_text(json.dumps(js,separators=(',',':'))+'\n')
 bounds=dict(q1_range=[int(q1.min()),int(q1.max())],Z_range=[int(zlo.min()),int(zhi.max())],q2_range=[int(q2.min()),int(q2.max())],
  product_abs_bound=int(product.max()),p_final_range=[int(plo.min()),int(phi.max())],p_any_second_stage_prefix_abs=int(pprefix.max()),
  expanded_range=[int(expanded.min()),int(expanded.max())],a_q40_range=[int(aq.min()),int(aq.max())],b_q20_range=[int(bq.min()),int(bq.max())],
  wide_all_finite_saturated_J32_abs=int(wide.max()),output_scale_range=[float(scale.min()),float(scale.max())],
  first_QDQ_as_float32_exactly_recovered=True,intermediate_RNE=False,coefficient_choice='one signed13 Q2 choice, admitted by exact static source/rank bounds; no sweep')
 return result,f,w32,bounds

def gather(source,tile):
 oy,ox=2*(tile//160),2*(tile%160);words=np.zeros((96,4,4),np.uint16)
 for y in range(4):
  for x in range(4):
   yy,xx=oy-1+y,ox-1+x
   if 0<=yy<240 and 0<=xx<320:words[:,y,x]=source[:,yy,xx]
 return words,np.array([oy,ox],np.int32)

def factor_chain(g,first,second):
 z=np.stack([np.einsum('tcyx,rcy->trx',g[:,:,py:py+3,:],first) for py in range(2)],axis=2)
 p=np.stack([np.einsum('trpx,orx->top',z[:,:,:,px:px+3],second) for px in range(2)],axis=3)
 return z,p

def error(a,b):
 d=np.asarray(a,np.float64)-np.asarray(b,np.float64)
 return dict(relative_L2=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(np.asarray(b).ravel()),1e-300)),RMSE=float(np.sqrt(np.mean(d*d))),max_abs=float(np.max(abs(d))))

def main():
 f,w8,w32,bounds=freeze()
 source=np.load(R/'first_source_words.npy',mmap_mode='r');identity=np.load(R/'identity_fp32_full.npy',mmap_mode='r')
 oldJ=np.load(R/'identity_q20_full.npy',mmap_mode='r')
 with np.load(R/'consumer_integer_first8.npz') as z:original8=((z['output_origin_yx'][:,0]//2)*160+z['output_origin_yx'][:,1]//2).tolist()
 ids=list(dict.fromkeys(original8+list(range(128,192))+list(range(4000,4064))))
 arrays={k:[] for k in ['source_words','output_origin_yx','z_halo_int','p_int','identity_fp32_bits','J_q20','wide_int64','i24','qdq64_i24','integer_ideal_i24']}
 ref8=[];ref32=[];newraw=[];per=[]
 for tile in ids:
  words,origin=gather(source,tile);g=((words[None,:,:,:]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
  z,p=factor_chain(g,f['q1'].astype(np.int64),f['q2'].astype(np.int64))
  direct=np.stack([np.stack([g[:,:,py:py+3,px:px+3].reshape(10,864)@f['expanded_int32'].reshape(96,864).astype(np.int64).T for px in range(2)],axis=2) for py in range(2)],axis=2)
  assert np.array_equal(p,direct),(tile,'factor/direct')
  # Float64 evaluates the frozen FP32 QDQ factors with unquantized latent state.
  _,raw8=factor_chain(g.astype(np.float64)*float(w8['theta']),w8['first'][:,:,:,0].astype(np.float64),w8['second'][:,:,0,:].astype(np.float64));raw8+=w8['bias'][None,:,None,None]
  _,raw32=factor_chain(g.astype(np.float64)*float(w32['theta']),w32['first'][:,:,:,0].astype(np.float64),w32['second'][:,:,0,:].astype(np.float64));raw32+=w32['bias'][None,:,None,None]
  raw=p*f['output_scale'][None,:,None,None]+f['bias'][None,:,None,None]
  ident=np.asarray(identity[tile],np.float32);J=identity_to_j(ident);assert np.array_equal(J,oldJ[tile])
  wide=p*f['a_q40'][None,:,None,None].astype(np.int64)+(J+f['b_q20'][None,:,None,None].astype(np.int64))*(1<<20)
  i24=rne_shift(wide,26,24)
  floatcheck=np.clip(np.rint(wide.astype(np.float64)/(1<<26)),-(1<<23),(1<<23)-1).astype(np.int64)
  assert np.array_equal(i24,floatcheck)
  def float_i24(v):return np.clip(np.rint((v*f['BN_gain'][None,:,None,None]+f['BN_offset'][None,:,None,None]+ident.astype(np.float64))*(1<<14)),-(1<<23),(1<<23)-1).astype(np.int64)
  qdq_i24=float_i24(raw8);ideal_i24=float_i24(raw)
  vals=dict(source_words=words,output_origin_yx=origin,z_halo_int=z.astype(np.int16),p_int=p.astype(np.int32),identity_fp32_bits=ident.view(np.uint32),J_q20=J.astype(np.int32),wide_int64=wide,i24=i24.astype(np.int32),qdq64_i24=qdq_i24.astype(np.int32),integer_ideal_i24=ideal_i24.astype(np.int32))
  for k,v in vals.items():arrays[k].append(v)
  ref8.append(raw8);ref32.append(raw32);newraw.append(raw)
  per.append(dict(tile=tile,Z_min=int(z.min()),Z_max=int(z.max()),p_min=int(p.min()),p_max=int(p.max()),wide_min=int(wide.min()),wide_max=int(wide.max()),integer_vs_W8_QDQ_raw=error(raw,raw8),I24_vs_QDQ64_different=int(np.count_nonzero(i24!=qdq_i24)),I24_vs_QDQ64_max_delta=int(np.max(abs(i24-qdq_i24))),consumer_only_I24_different=int(np.count_nonzero(i24!=ideal_i24))))
 arrays={k:np.stack(v) for k,v in arrays.items()};arrays['tile_ids']=np.array(ids,np.int32)
 np.savez_compressed(H/'gold_tiles.npz',**arrays)
 # Integer rounding edge checks, separate from any selected real data.
 half=1<<25;edges=np.array([q*(1<<26)+d for q in [-8388609,-8388608,-3,-2,-1,0,1,2,8388606,8388607,8388608] for d in [0,half-1,half,half+1,(1<<26)-1]],np.int64)
 assert np.array_equal(rne_shift(edges,26,24),np.clip(np.rint(edges.astype(np.float64)/(1<<26)),-(1<<23),(1<<23)-1))
 special=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),np.nextafter(np.float32(0),np.float32(-1)),np.finfo(np.float32).max,-np.finfo(np.float32).max,0.5/(1<<20),1.5/(1<<20),-0.5/(1<<20),-1.5/(1<<20)],np.float32)
 assert identity_to_j(special).tolist()==[0,0,0,0,(1<<31)-1,-(1<<31),0,2,0,-2]
 report=dict(passed=True,bounds=bounds,tiles=len(ids),tile_sets=dict(original8=original8,held64=[128,191],disjoint64=[4000,4063],deduplicated=True),raw_values=int(arrays['p_int'].size),Z_gold_values=int(arrays['z_halo_int'].size),source_shape=list(source.shape),identity_shape=list(identity.shape),
  same_W8_QDQ_reference='Float64 evaluation of saved FP32 factor values, with no latent quantization; not CUDA FP32 reduction-order equality or network AEE',
  integer_vs_W8_QDQ_raw=error(np.stack(newraw),np.stack(ref8)),W8_QDQ_vs_W32_raw=error(np.stack(ref8),np.stack(ref32)),integer_vs_W32_raw=error(np.stack(newraw),np.stack(ref32)),
  I24_vs_QDQ64=dict(different=int(np.count_nonzero(arrays['i24']!=arrays['qdq64_i24'])),max_delta=int(np.max(abs(arrays['i24'].astype(np.int64)-arrays['qdq64_i24'])))),
  consumer_only_I24=dict(different=int(np.count_nonzero(arrays['i24']!=arrays['integer_ideal_i24'])),max_delta=int(np.max(abs(arrays['i24'].astype(np.int64)-arrays['integer_ideal_i24'])))),
  factor_vs_expanded_integer_differences=0,FP32_J_vs_existing_gold_differences=0,integer_RNE_vs_exact_range_FP64_differences=0,rounding_edge_cases=len(edges)+len(special),network_AEE='not run; W32/QDQ AEE is not inherited')
 (H/'stats.json').write_text(json.dumps(report,separators=(',',':'))+'\n');(H/'tile_stats.jsonl').write_text(''.join(json.dumps(v,separators=(',',':'))+'\n' for v in per))
 manifest=dict(factors='factors.npz',gold='gold_tiles.npz',input_fields=['tile_ids','source_words','output_origin_yx','identity_fp32_bits'],oracle_only_fields=['z_halo_int','p_int','J_q20','wide_int64','i24','qdq64_i24','integer_ideal_i24'],source_original=str(R/'first_source_words.npy'),identity_original=str(R/'identity_fp32_full.npy'),array_shapes={k:list(v.shape) for k,v in arrays.items()},array_dtypes={k:str(v.dtype) for k,v in arrays.items()})
 (H/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
