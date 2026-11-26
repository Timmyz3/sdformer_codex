#!/usr/bin/env python3
"""CPU metadata/real PT shape audit; no model forward, GT or cycle claim.

Recounts an affine-product subset of the current native graph. Old full-shape
records supply named constructor geometry only; no old aggregate is reused.
"""
import argparse, hashlib, json, math
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE=Path(__file__).resolve().parent
R26=HERE.parent
REPO=HERE.parents[3]
OLD=REPO/'research/flow_story_20261011/preflight/execution_contract/execution_services_results.json'
PT=REPO/'research/flow_story_20261016/baseline_checkpoint_verification/source_snapshot/checkpoint_step_000271482.pt'
RESULT=R26/'actual_v1/run_v1/results.json'
AUDIT=R26/'independent_execution_audit/ACTUAL_EXECUTION_AUDIT_V1.json'
SOURCE=REPO/'SDformer/third_party/SDformerFlow'
P='sttmultires_unet.'
E=P+'encoders.swin3d.'

def require(v,m):
 if not v:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def valid_axis(n,m,k,s,p,transpose):
 # Count true input/output coordinate pairs, including no padded-zero products.
 return sum(0 <= (i*s-p+j) < (m if transpose else n)
            for i in range(n if transpose else m) for j in range(k))

def run():
 require(sha(OLD)=='3cc67c3c331dc8475cd1647003a684bb6a36843e8b6ce00faa5b41f1f30968e8','Old same-source fullshape metadata')
 require(sha(PT)=='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977','Actual original PT')
 require(sha(RESULT)=='ca16d2b0c491841f45e43a9ace4010de83706585950903ac22da4d2a4a42ce5c','Actual completed R26 result')
 require(sha(AUDIT)=='4f4812a4d7b8418ccdbcce5a026082c720d85fa35a115c7d0497ef7907b2b3f8','Independent complete execution audit')
 old=read(OLD);actual=read(RESULT);a=read(AUDIT)
 require(a['status']=='PASS' and a['source_result_sha256']==sha(RESULT),'Native metadata is within actual audited graph')
 for n,h in old['source_sha256'].items():require(sha(SOURCE/n)==h,'Same unchanged native source '+n)
 config=next(x for x in old['configurations'] if x['name']=='full_480x640_window9_current_config')
 require(config['input_HWT']==[480,640,10],'No crop scaling')
 record=actual['forward_records'][0];require(record['arm']=='native','Current graph actual native record')
 bn=record['native_BN_inputs'];require(len(bn)==78 and all(len(v)==1 for v in bn.values()),'78 observed BN shapes')
 import torch
 state=torch.load(PT,map_location='cpu',weights_only=False)['model_state_dict']
 require(len(state)==711,'Actual full original checkpoint')
 rows=[];mapped=[];bn_checked=set()
 def bshape(name,expected):
  require(bn[name][0]['shape']==list(expected),'Actual BN geometry '+name);bn_checked.add(name)
 def conv(name,wname,bname,expectedin,expectedout,k,s,transpose=False):
  src=next(x for x in config['retained_conv_geometry'] if x['name']==name)
  require(src['input_HWC']==expectedin and src['output_HWC']==expectedout and src['kernel']==k and src['stride']==s and src['operation']==('ConvTranspose2d' if transpose else 'Conv2d'),'Named constructor geometry '+name)
  h,w,ci=expectedin;oh,ow,co=expectedout;p=k//2
  require([*state[wname].shape]==([ci,co,k,k] if transpose else [co,ci,k,k]),'Actual PT convolution shape '+name)
  if bname:
   expected=[10,1,co,oh,ow] if not name.startswith('patch.PED') else [10,co,oh,ow]
   bshape(bname,expected)
  vy=valid_axis(h,oh,k,s,p,transpose);vx=valid_axis(w,ow,k,s,p,transpose)
  require(vy*vx==src['valid_spatial_taps'],'Independent valid-tap coordinate enumeration '+name)
  terms=10*vy*vx*ci*co
  rows.append(dict(name=name,kind='conv',weight=wname,weight_shape=list(state[wname].shape),input_HWC=expectedin,output_HWC=expectedout,valid_spatial_taps=vy*vx,scalar_weighted_product_terms=terms,actual_output_BN_name=bname))
  mapped.append(wname)
 pe=E+'patch_embed.'
 conv('patch.head',pe+'head.conv.0.weight',pe+'head.norm_layer.norm_layer',[480,640,2],[480,640,48],3,1)
 conv('patch.conv',pe+'conv.conv.0.weight',pe+'conv.norm_layer.norm_layer',[480,640,48],[240,320,96],3,2)
 for i in range(2):
  for j in range(2):
   q=pe+f'residual_encoding.resblocks.{i}.'
   conv(f'patch.res{i}.conv{j}',q+f'conv{j+1}.0.weight',q+f'norm{j+1}.norm_layer',[240,320,96],[240,320,96],3,1)
 conv('patch.PED.conv_res',pe+'proj.conv_res.weight',pe+'proj.norm_layer',[240,320,96],[120,160,96],1,2)
 conv('patch.PED.conv_spike',pe+'proj.conv.weight',pe+'proj.norm_layer',[240,320,96],[120,160,96],3,2)
 for i in range(2):
  for j in range(2):
   q=P+f'resblocks.{i}.'
   conv(f'bottleneck.res{i}.conv{j}',q+f'conv{j+1}.0.weight',q+f'norm{j+1}.norm_layer',[15,20,768],[15,20,768],3,1)
 for i,(h,w,ci,co) in enumerate([(15,20,1536,384),(30,40,770,192),(60,80,386,96),(120,160,194,96)]):
  conv(f'decoder{i}.deconv',P+f'decoders.{i}.deconv.0.weight',P+f'decoders.{i}.norm_layer.norm_layer',[h,w,ci],[2*h,2*w,co],3,2,True)
  conv(f'pred{i}.conv',P+f'preds.{i}.conv.0.weight',None,[2*h,2*w,co],[2*h,2*w,2],1,1)
 require(set(mapped)=={k for k,v in state.items() if k.endswith('.weight') and v.ndim==4},'All20 current convolution weights covered')
 layers=[]
 for si,(depth,h,w,c) in enumerate([(2,120,160,96),(2,60,80,192),(6,30,40,384),(2,15,20,768)]):
  # Each actual complete9x9 tile is eligible; every partial tile retainsQ8.
  low=sum(min(9,h-y)==9 and min(9,w-x)==9 for y in range(0,h,9) for x in range(0,w,9))*81
  for bi in range(depth):
   q=E+f'layers.{si}.swin_blocks.{bi}.';f=q+'mlp.'
   obs=record['all12_original_fc_observations'][f[:-1]]
   require(obs==dict(fc1_rows=10*h*w,fc2_rows=10*h*w,fc1_calls=1,fc2_calls=1),'Actual fullFC records')
   terms=0
   for name,ci,co,bnn in [('fc1',c,4*c,'bn1'),('fc2',4*c,c,'bn2')]:
    wn=f+name+'.weight';require(list(state[wn].shape)==[co,ci],'Original FC shape');bshape(f+bnn+'.norm_layer',[10,1,co,h,w])
    n=10*h*w*ci*co;terms+=n;rows.append(dict(name=f+name,kind='MLP_FC',weight=wn,scalar_weighted_product_terms=n));mapped.append(wn)
   require(terms==14155776000,'Equal actual affine work perMLP')
   layers.append(dict(stage=si,block=bi,shape=[10,1,h,w,c],complete_tile_pixels=low,tail_pixels=h*w-low,complete_tile_fraction=float(Fraction(low,h*w)),FC_terms=terms,all_layer_Q6_cancelled_Q8_equivalent_terms=terms*low//(h*w*4),all_layer_Q4_cancelled_Q8_equivalent_terms=terms*low//(h*w*2)))
   # All three projection BN records include padding in the actual currentBN graph.
   for linear,bnn in [('linear_q','bn_q'),('linear_k','bn_k'),('proj','proj_bn')]:
    wn=q+'attn.'+linear+'.weight';name=q+'attn.'+bnn+'.norm_layer';shape=bn[name][0]['shape']
    require(list(state[wn].shape)==[c,c] and shape[2]==c and shape[0]==2 and shape[-2:]==[9,9],'Actual padded attention projection shape')
    bn_checked.add(name);n=math.prod(shape)//c*c*c
    rows.append(dict(name=q+'attn.'+linear,kind='attention_projection',weight=wn,actual_output_BN_name=name,actual_output_BN_shape=shape,scalar_weighted_product_terms=n));mapped.append(wn)
  if si<3:
   q=E+f'layers.{si}.downsample.';name=q+'norm.norm_layer';shape=bn[name][0]['shape'];wn=q+'reduction.weight'
   require(list(state[wn].shape)==[2*c,4*c],'Actual patchmerge weight');bshape(name,[10,1,2*c,(h+1)//2,(w+1)//2])
   rows.append(dict(name=q+'reduction',kind='patch_merge',weight=wn,scalar_weighted_product_terms=math.prod(shape)//(2*c)*8*c*c));mapped.append(wn)
 require(len(bn_checked)==78,'All78 recorded output-BN shapes accounted for')
 require(len(rows)==83 and len(set(mapped))==83,'20conv+24FC+36attention+3merge affine tensors')
 sums=Counter()
 for r in rows:sums[r['kind']]+=r['scalar_weighted_product_terms']
 total=sum(sums.values());fc=sums['MLP_FC'];common=total-fc
 require(fc==169869312000,'Independent actual full12 MLP terms')
 screens=[]
 for bits in (6,4):
  delta=sum(x[f'all_layer_Q{bits}_cancelled_Q8_equivalent_terms'] for x in layers)
  # This is only a uniform completed-Q8-weighted-term accounting convention.
  # Different bitserial/native throughputs must use the symbolic condition.
  frac=Fraction(delta,fc)
  screens.append(dict(mode=f'Q{bits}_all12_complete_tiles_only',quality_tested_all12=False,weighted_average_MLP_plane_fraction_cancelled=float(frac),cancelled_Q8_equivalent_affine_terms=delta,uniform_Q8_equivalent_affine_subset_reduction=float(Fraction(delta,total)),minimum_whole_PE_fraction_in_MLP_FC_for_10percent_at_same_layer_proportions=float(Fraction(1,10)/frac),rho_break_even_ignoring_all_non_affine_and_overhead=float(Fraction(common,10*delta-fc)),rho_definition='Cost per completed MLP Q8 weighted connection / average cost per common affine weighted connection. Same scalar terms do not establish rho=1.',actual_PE_cycles=None,actual_10percent_gate=None))
 return dict(schema='flow_r26_current_graph_affine_scope_v1',status='SAME_SOURCE_REAL_SHAPES_STATIC_PRODUCTS_NOT_PE_CYCLES',source_sha256={str(p):sha(p) for p in (Path(__file__),OLD,PT,RESULT,AUDIT)},native_source_sha256=old['source_sha256'],independent_actual_metadata_checks=dict(native_record_forward_id=1,source_weight_shapes=83,convolution_weights=20,observed_BN_shapes=78,FC_layers=12),affine_records=rows,stage_layer_precision_domain=layers,affine_scalar_product_subtotals=dict(sums),affine_scalar_product_total=total,MLP_fraction_of_affine_product_subset=float(Fraction(fc,total)),all12_precision_screens=screens,
  actual_GPU_used=False,model_forwards=0,GT_values_read=False,old_crop_aggregate_reused=False,old_issue_cycles_reused=False,
  limitations=['These are dense scalar weighted-connection terms, not operation-equivalent cycles. Native common work and MLP bitplanes require a shared resource model.', 'Only affine products counted. All78currentBN, PSN temporal real products, nonlinear/reduction/control, precision conversion/scaling, memory and predictor are omitted rather than declared free.', '20conv geometry is independently enumerated from unchanged source constructors and original PT weight shapes, with 16 output domains corroborated by current native BN (PED shares oneBN); four prediction shapes follow actual decoder outputs. This is not a new complete conv-hook trace.', 'All36 attention projections use actual padded BN domains. No old dead-padding cancellation assumed under currentBN.', 'The hypothetical all12 lowprecision extension has no multi-layer quality result. Single-target evidence cannot validate it.', 'Uniform-cost affine subset percentage is an optimistic arithmetic screen only under equal completed-Q8-weighted-term costs and nonnegative common overhead. No universal hardware impossibility or whole-net speedup is proved.', 'Actual PE fraction, shape/rate/stall trace, bitserial throughput, ordinary baseline optimization and added predictor/packing/BN service remain to be re-traced/modelled on one implementable machine.'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Exclusive new output')
 d=run();a.output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:d[k] for k in ('status','affine_scalar_product_subtotals','affine_scalar_product_total','MLP_fraction_of_affine_product_subset','all12_precision_screens')}))
