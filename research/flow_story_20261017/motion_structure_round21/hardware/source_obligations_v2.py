#!/usr/bin/env python3
"""Source-bound operation obligations only; zero head/model calls or fitting."""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent;DATE=HERE.parents[1];DESIGN=HERE.parent/'design'
SNAPSHOT=DESIGN/'DESIGN_REVIEW_SNAPSHOT_V1.json'
SNAPSHOT_SHA='d05c1d5a61dd44349876567dfbe1fd9ae061c39f1eaf2004cd0b8ba8e41742eb'
CORE=DESIGN/'structure_models.py';CORE_SHA='efbfa8f5bea04cc96f967241c7655a31a969bc40f17650ab446e2122cc567528'
PRIOR=DATE/'action_domain_round20/hardware/R21_EXACT_SOURCE_LEDGER_V1.json'
PRIOR_SHA='a359a2ddf6b01c384f8fd07b183609f5bbad380f700b8502f63db44fc6f44921'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,m):
 if not ok:raise ValueError(m)
def pool(c):
 return {'fullgrid_scalar_input_elements':1200*c,'mean_add':(1200-140)*c,'mean_divide':140*c,
  'representative_elements':140*c,'expanded_write_elements':1200*c,'residual_subtract':1200*c,
  'residual_write_elements':1200*c,'true_pool_cells':140}
def cnn():
 return {'conv_and_head_bias_add':2*972*8+12*8+12,'ReLU_compare':2*972*8+12*8,
  'final_avgpool_add':12*9*8*8,'final_avgpool_divide':12*9*8,
  'final_avgpool_output_elements':12*9*8,'two_conv_output_elements':2*972*8}
def report():
 require(sha(SNAPSHOT)==SNAPSHOT_SHA and sha(CORE)==CORE_SHA and sha(PRIOR)==PRIOR_SHA,'Exact reviewed design/source and predecessor MAC inventory')
 for v in json.loads(SNAPSHOT.read_text())['files']:require(sha(Path(v['path']))==v['sha256'],'Design snapshot asset')
 old=json.loads(PRIOR.read_text());methods={};channels={'generic_raw':386,'generic_group':34,'same_motion':6,'structured':6}
 for name,ch in channels.items():
  r=dict(old['methods'][name]);r.update({
   'spatial_scale_subtract':972*ch,'spatial_scale_divide':972*ch,
   'spatial_scaler_constant_elements':2*ch,'context_scale_subtract':408,'context_scale_divide':408,
   'context_scaler_constant_elements':68,'output_denormalize_multiply':12,'output_denormalize_add':12,'target_scaler_constants':2,
   'top1_strict_compare':11,'score_elements':12,
   'spatial_scaled_output_elements':972*ch,'context_scaled_output_elements':408,
   'MAC_is_not_cycles':True,'full_resource_schedule':None,'net_hardware_win':None})
  if name in ('same_motion','structured'):
   r['motion_proxy_bias_add']=2400;r['motion_fullgrid_residual']=pool(2)
   r['scaling_semantics']='q/r share one positive scalar; q translation only, r and xy zero-centered constants. Literal scale still subtracts and divides all6 channels; no free zero/one simplification credited.'
  if name!='structured':r['head_non_MAC']=cnn()
  if name=='generic_group':r['generic_fullgrid_residual']=pool(16)
  if name=='generic_raw':
   r['head_non_MAC']['projection_bias_add']=972*8
   r['head_non_MAC']['projected_avgpool_add']=12*9*8*8
   r['head_non_MAC']['projected_avgpool_divide']=12*9*8
   r['head_non_MAC']['projected_repeat_write_elements']=972*8
   r['head_non_MAC']['projected_residual_subtract']=972*8
   r['head_non_MAC']['projected_residual_write_elements']=972*8
  if name=='structured':r['head_non_MAC']={
   'node_edge_head_bias_add':972*4+12*144*4+12*8+12,'ReLU_compare':972*4+12*144*4+12*8,
   'node_avgpool_add':12*9*4*8,'node_avgpool_divide':12*9*4,
   'directed_edges':12*144,'endpoint_gather_scalar_reads':12*144*4*2,
   'q_edge_difference_subtract':12*144*2,'edge_concat_write_elements':12*144*12,
   'node_output_elements':972*4,'edge_output_elements':12*144*4,
   'dense_membership_output_elements':12*4*9,'dense_membership_MAC_included_in_prior_MAC':62208,
   'sparse_membership_optimization':'NOT_CREDITED; changing reduction order needs its own numerical contract'}
  methods[name]=r
 return {'schema':'flow_r21_uniform_vector_scaling_source_obligations_v2','status':'DESIGN_OBLIGATIONS_ONLY_NOT_FITTED',
  'source_sha256':{str(p):sha(p) for p in (Path(__file__),CORE,SNAPSHOT,PRIOR)},
  'scope_per_inference':'One original T10/B1/H30/W40/C384 input, twelve complete9x9 candidates, full-grid context/q include true tails.',
  'new_neural_forwards':0,'fit_calls':0,'GPU_calls':0,'RTL_calls':0,
  'common_time_mean':old['common_time_mean'],
  'common_generic_context':{'group_mean_add':1200*16*23,'group_mean_divide':1200*16,
   'group_output_elements':19200,'global_mean_calls':2,'global_mean_reduction_add':2*16*1199,'global_mean_divide':32,
   'global_variance_center_subtract':19200,'global_variance_square':19200,'global_variance_reduce_add':16*1199,
   'global_variance_divide':16,'global_std_sqrt':16,'global_output_elements':32,'context_repeat_plus_position_elements':408,
   'convention':'Source calls z.mean plus z.std; std includes its own mean. Counts use mathematical N-1 reduction adds, not NumPy instructions or physical traffic.'},
  'methods':methods,
  'storage_and_scope_obligations':[
   'All operation counts are logical scalar/source-shape obligations, not physical bus transactions or timed instruction traces. Bias/ReLU/pooling/scaling are separate from nominal MAC.',
   'The common time mean reads18432000B and writes1843200B. Every materialized reread of the mean adds1843200B, not3686400B per direction. Projection/context producer fusion may avoid rereads only with a lifetime/port proof.',
   'q full-grid output9600B, grouped z76800B; vector residual buffers, scaled spatial/context arrays, intermediate head outputs and score12 must be stored/streamed and their accesses charged. Element fields allow exact FP32 byte conversion but do not imply all arrays simultaneously resident.',
   'Mean1843200B plus existing2MiB target working set cannot both be assumed freely resident. Explicit lifetime reuse, spills or producer fusion is unresolved. No new finite-port head schedule or area measurement exists.',
   'The batched CPU spatial_inputs function constructs all four feature families. Per-method obligations here count only that method plus common context; a literal unpruned shared CPU workflow also computes unused q/z residuals and copies. Neither path is measured hardware.',
   'Scaler fit/statistics, ridge solve/OOF fitting, training loss and labels are offline; inference uses frozen scalers and folded q. Vector-scaler square root is a fit cost, while global generic-context std16 sqrt is a runtime cost.',
   'For equally informed G2/F, q, globalcontext and vector scaling are common. Strong generic_raw retains learned384->8; all controls may use the same FP32 lanes/producer fusion/PE resource reuse. OldFP64score implementation is not a mandatory generic baseline.',
   'All methods observe preMLP u after attention, and can only decide the targetMLP. Any whole-window attention cancellation needs an earlier validated tap and actual dependency closure.',
   'No quality or feasibility implication: R21 structure fitting is paused. This preserves a concrete cost obligation if later information tests justify a new candidate, and cannot declare a net hardware winner.']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Exclusive new receipt')
 out=report();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'sha256':sha(a.output),'new_forwards':0,'fit_calls':0}))
