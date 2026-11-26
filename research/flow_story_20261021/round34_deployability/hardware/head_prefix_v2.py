#!/usr/bin/env python3
"""One finite FP32 head at a time, actual weights; no GPU or RTL timing claim."""
from pathlib import Path
import hashlib,importlib.util,json,math
import numpy as np
P=Path(__file__).resolve().parent;R34=P.parent;R33=R34.parent/'round33_predictability';O=P/'actual_v2'
spec=importlib.util.spec_from_file_location('r34_numeric_reference',R34/'numeric/core.py');num=importlib.util.module_from_spec(spec);spec.loader.exec_module(num)
F=np.float32
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def align(n,a=64):return ((n+a-1)//a)*a

class Head:
 def __init__(self,name):
  self.name=name;self.ios=np.fromfile(O/'zurich_city_07_a_0010.f32.ios',dtype=F);assert self.ios.size==524288//4;self.tick=0;self.ops={};self.events=[];self.wslot=0;self.wrf_peak=0;self.ios_peak=0;self.parameter_payload=bytearray();self.blocks=[]
 def op(self,k,n=1):
  self.ops[k]=self.ops.get(k,0)+n;self.tick+=n*(3 if k in ['add','sub','mul'] else 16 if k in ['div','sqrt'] else 1)
 def access(self,space,write,address,words,vector=1,kind='operand'):
  cap={'IOS':524288,'WRF':32768,'WSLOT':1048576}[space];assert address%4==0 and address+4*words<=cap
  start=self.tick;self.tick+=(words+vector-1)//vector;self.events.append(dict(kind=kind,space=space,write=write,address=address,words=words,start=start,end=self.tick,words_per_tick=vector))
  if space=='WRF':self.wrf_peak=max(self.wrf_peak,address+4*words)
  if space=='IOS':self.ios_peak=max(self.ios_peak,address+4*words)
 def pack(self,a):
  a=np.ascontiguousarray(a,dtype=F);address=len(self.parameter_payload);self.parameter_payload.extend(a.tobytes());self.parameter_payload.extend(bytes(align(a.nbytes)-a.nbytes));return address,a
 def prepare(self,ridges):
  for r in ridges:
   mean,scale,coef,bias=[np.asarray(r[k],dtype=F) for k in ['mean','scale','coef','intercept']];n,o=coef.shape;d={'i':n,'o':o,'mean':self.pack(mean),'scale':self.pack(scale),'tiles':[]}
   for start in range(0,o,16):d['tiles'].append({'start':start,'coef':self.pack(coef[:,start:start+16]),'bias':self.pack(bias[start:start+16])})
   self.blocks.append(d)
  self.wslot=len(self.parameter_payload);assert self.wslot<=1048576
  # Shared DMA to weight slot, one contiguous offline-packed representation.
  self.tick=32+self.wslot//64;self.events.append(dict(kind='parameter_DMA',space='EXTERNAL_TO_WSLOT',write=True,address=0,bytes=self.wslot,start=0,end=self.tick))
 def refill(self,source,wrf):
  address,a=source;n=align(a.nbytes)//4
  # Same existing 256B/tick weight-SRAM-to-RF path. Source/dest ports operate
  # together, so one transfer is charged once, not two serialized copies.
  start=self.tick;self.access('WSLOT',False,address,n,64,'weight_refill');end=self.tick
  assert wrf+n*4<=32768;self.wrf_peak=max(self.wrf_peak,wrf+n*4)
  self.events.append(dict(kind='weight_refill_destination',space='WRF',write=True,address=wrf,words=n,start=start,end=end,words_per_tick=64))
 def affine(self,index,x,xaddr,yaddr):
  d=self.blocks[index];n,o=d['i'],d['o'];x=np.asarray(x,dtype=F);assert x.shape==(n,);assert np.array_equal(x,self.ios[xaddr//4:xaddr//4+n]);x=self.ios[xaddr//4:xaddr//4+n]
  mean,scale=d['mean'][1],d['scale'][1];mb=0;sb=align(n*4);cb=2*align(n*4)
  self.refill(d['mean'],mb);self.refill(d['scale'],sb);z=np.empty(n,F)
  for k in range(n):
   self.access('IOS',False,xaddr+k*4,1);self.access('WRF',False,mb+k*4,1);self.access('WRF',False,sb+k*4,1)
   self.op('sub');v=F(x[k]-mean[k]);self.op('div');z[k]=F(v/scale[k]);self.access('IOS',True,8192+k*4,1);self.ios[8192//4+k]=z[k]
  out=np.zeros(o,F)
  for tile in d['tiles']:
   co=tile['coef'][1];bi=tile['bias'][1];width=co.shape[1];bb=cb+align(co.nbytes);self.refill(tile['coef'],cb);self.refill(tile['bias'],bb)
   for k in range(n):
    self.access('IOS',False,8192+k*4,1)
    for j in range(width):
     self.access('WRF',False,cb+(k*width+j)*4,1);self.op('mul');v=F(z[k]*co[k,j]);self.op('add');out[tile['start']+j]=F(out[tile['start']+j]+v)
   for j in range(width):
    self.access('WRF',False,bb+j*4,1);self.op('add');idx=tile['start']+j;out[idx]=F(out[idx]+bi[j]);self.access('IOS',True,yaddr+idx*4,1);self.ios[yaddr//4+idx]=out[idx]
  return out
 def products(self,aux,g):
  dim=2*g*g;p=np.empty((6,dim),F);assert np.array_equal(aux,self.ios[1024//4:1024//4+len(aux)]);aux=self.ios[1024//4:1024//4+len(aux)]
  for a in range(6):
   for k in range(dim):
    self.access('IOS',False,1024+k*4,1);self.access('IOS',False,1024+(dim+a*dim+k)*4,1);self.op('mul');p[a,k]=F(aux[k]*aux[dim+a*dim+k]);self.access('IOS',True,4096+(a*dim+k)*4,1);self.ios[4096//4+a*dim+k]=p[a,k]
  return p
 def signed(self,p):
  s=np.zeros(6,F)
  for a in range(6):
   for k in range(p.shape[1]):self.access('IOS',False,4096+(a*p.shape[1]+k)*4,1);self.op('add');s[a]=F(s[a]+p[a,k])
  return s
 def finish(self,scores):
  # Both fixed-six and allow-Q4 use one argmin, optional zero comparison.
  self.op('compare',5);self.op('compare',1);head_ready=self.tick;self.op('action_latch',1)
  return dict(head_ready_tick=head_ready,action_ready_tick=self.tick,first_controlled_conv_join='max(action_ready,shared_stem_ready), stem unmodeled',scores=scores.tolist(),fixed_action=int(num.choose(scores[None])[0]),allow_action=int(num.choose(scores[None],True)[0]))

def execute(model,x,name,calibrated=None,folded=None):
 h=Head(name);g=0
 if folded is not None:ridges=[folded]
 elif name=='direct_raw':ridges=[model['models'][name]]
 else:
  g=int(name[-1]);ridges=[model['auxiliary'][f'g{g}']]
  if calibrated is not None:ridges=[calibrated['family']['auxiliary'],calibrated['family']['residual'][str(calibrated['selection']['alpha'])]]
  elif not name.startswith('signed'):ridges.append(model['models'][name])
 h.prepare(ridges)
 if g==0:s=h.affine(0,x,400000,12288)
 else:
  aux=h.affine(0,x,400000,1024);p=h.products(aux,g)
  if name.startswith('signed'):s=h.signed(p)
  else:
   # Copy from actual IOS source fields into one contiguous augmented vector.
   for address,words in [(400000,len(x)),(1024,len(aux)),(4096,p.size)]:h.access('IOS',False,address,words)
   aug=np.concatenate((h.ios[400000//4:400000//4+len(x)],h.ios[1024//4:1024//4+len(aux)],h.ios[4096//4:4096//4+p.size]));h.access('IOS',True,16384,len(aug));h.ios[16384//4:16384//4+len(aug)]=aug;r=h.affine(1,aug,16384,12288)
   if 'residual' in name:
    s=h.signed(p)
    for a in range(6):
     if calibrated is not None:h.op('mul');r[a]=F(F(calibrated['selection']['shrink'])*r[a])
     h.op('add');s[a]=F(s[a]+r[a])
   else:s=r
 out=h.finish(s);out.update(name=name,parameter_bytes_aligned=h.wslot,WRF_peak_bytes=h.wrf_peak,IOS_peak_bytes=h.ios_peak,arithmetic_ops=h.ops,service_events=h.events,scalar_register_file_bytes=2048,one_blocking_FPU=True,RTL_latency_verified=False)
 return out,bytes(h.parameter_payload)

def main():
 stream=np.load(O/'STREAM_FEATURES_V1.npz');x160=stream['x160'];ref=np.load(R34/'numeric/actual_v1/FEATURES_FP32_V1.npz');ids=list(stream['ids']);roles=list(stream['roles'])
 refx=np.stack([ref[str(role)+'_x160'][list(ref[str(role)+'_ids']).index(i)] for i,role in zip(ids,roles)])
 report={'schema':'r34_actual_single_head_prefix_v1','models':{},'feature_source_sha256':sha(O/'STREAM_FEATURES_V1.npz'),'new_RTL':False,'whole_network':False,'model_arithmetic_not_RTL':True}
 all_scores={}
 for width,branch in [(128,'voxel128_v1'),(160,'directional160_v1')]:
  model=read(R33/'fitted'/branch/'MODEL_V1.json');x=x160[:,:width];rref=refx[:,:width];suite,_,_=num.run_suite(model,x);baseline,_,_=num.run_suite(model,rref)
  for name in num.METHODS:
   key=f'X{width}_{name}';record,blob=execute(model,x[0],name);expected=suite['scores'][name][0];assert np.array_equal(np.asarray(record['scores'],F),expected),(key,record['scores'],expected)
   record['feature_stream_action_flips_fixed']=int(np.count_nonzero(suite['fixed_six_choices'][name]!=baseline['fixed_six_choices'][name]));record['feature_stream_action_flips_allow']=int(np.count_nonzero(suite['allow_Q4_choices'][name]!=baseline['allow_Q4_choices'][name]));record['actual_120_scores_computed']=True;record['first_identity_service_scores_bitwise_equal_to_numeric_reference']=True
   events=record.pop('service_events');(O/(key+'_HEAD_EVENTS_V1.json')).write_text(json.dumps(events,indent=2)+'\n');(O/(key+'_PARAMETERS_FP32_V1.bin')).write_bytes(blob);report['models'][key]=record;all_scores[key]=suite['scores'][name]
  calibration=read(R34/f'ordinary/actual_v1/MODEL_{width}_V1.json');key=f'X{width}_calibrated_residual_g1';record,blob=execute(model,x[0],'direct_residual_g1',calibration)
  def calibrated_scores(features):
   a,_=num.affine_fp32(calibration['family']['auxiliary'],features);p=num.products(a,1);aug=np.concatenate((features,a,p.reshape(len(features),-1)),axis=1);r,_=num.affine_fp32(calibration['family']['residual'][str(calibration['selection']['alpha'])],aug);return num.signed(a,1)+F(calibration['selection']['shrink'])*r
  actual=calibrated_scores(x);old=calibrated_scores(rref);assert np.array_equal(np.asarray(record['scores'],F),actual[0]);record['feature_stream_action_flips_fixed']=int(np.count_nonzero(num.choose(actual)!=num.choose(old)));record['feature_stream_action_flips_allow']=int(np.count_nonzero(num.choose(actual,True)!=num.choose(old,True)));all_scores[key]=actual
  events=record.pop('service_events');(O/(key+'_HEAD_EVENTS_V1.json')).write_text(json.dumps(events,indent=2)+'\n');(O/(key+'_PARAMETERS_FP32_V1.bin')).write_bytes(blob);report['models'][key]=record
 folded=read(R34/'numeric/folded_constant_v/actual_v1/FOLDED_MODEL_V1.json')['ridge'];key='X128_folded_constant_v';record,blob=execute(None,x160[0,:128],key,folded=folded);expected,_=num.affine_fp32(folded,x160[:,:128]);assert np.array_equal(np.asarray(record['scores'],F),expected[0]);orig,_=num.affine_fp32(folded,refx[:,:128]);record['feature_stream_action_flips_fixed']=int(np.count_nonzero(num.choose(expected)!=num.choose(orig)));record['feature_stream_action_flips_allow']=int(np.count_nonzero(num.choose(expected,True)!=num.choose(orig,True)));events=record.pop('service_events');(O/(key+'_HEAD_EVENTS_V1.json')).write_text(json.dumps(events,indent=2)+'\n');(O/(key+'_PARAMETERS_FP32_V1.bin')).write_bytes(blob);report['models'][key]=record;all_scores[key]=expected
 feat128=read(O/'rowband128.json');feat160=read(O/(str(ids[0])+'.json'))
 for key,r in report['models'].items():
  f=feat128 if key.startswith('X128') else feat160;r['feature_ready_tick_first_identity']=f['feature_ready_tick'];r['serial_prefix_action_ready_first_identity']=f['feature_ready_tick']+r['action_ready_tick'];r['shared_stem_service_included']=False
 np.savez(O/'STREAM_HEAD_SCORES_V1.npz',ids=stream['ids'],roles=stream['roles'],**all_scores)
 (O/'HEAD_PREFIX_RESULTS_V1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:{q:v[q] for q in ['action_ready_tick','serial_prefix_action_ready_first_identity','parameter_bytes_aligned']} for k,v in report['models'].items()},indent=2))
if __name__=='__main__':main()
