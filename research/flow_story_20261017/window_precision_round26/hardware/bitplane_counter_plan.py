#!/usr/bin/env python3
"""Independent signed-plane/true-tail accounting; no torch/model/GT/RTL."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
MODEL=REPO/'SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py'
MODEL_SHA='8d551eac17e2272813dd0238ea4aee2e84cfe0b7e9435a1407d0787fb5407768'
def require(v,msg):
 if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def precision(y,x,bits):return bits if y<27 and x<36 else 8
def rowcounts(bits):
 return {k:10*sum(precision(y,x,bits)>=(8-k) for y in range(30) for x in range(40)) for k in range(8)}
def report():
 require(sha(MODEL)==MODEL_SHA,'Actual native model source')
 source=MODEL.read_text();require('self.fc1 =  sj_layer.Linear(in_features, hidden_features,bias=False)' in source and 'self.fc2 = sj_layer.Linear(hidden_features, out_features,bias=False)' in source,'Actual FC bias false')
 checked=0
 for q in range(-127,128):
  for bits in (8,6,4):
   drop=8-bits;trunc=(q>>drop)<<drop
   decoded=sum(((q>>k)&1)*(-128 if k==7 else 1<<k) for k in range(drop,8))
   require(trunc==decoded,'Signed arithmetic shift and signplane decoding');checked+=1
 require((-127>>2)<<2==-128 and (-1>>4)<<4==-16,'Negative truncation must not reclamp/midpoint-correct')
 # Exhaustive short binary dot vectors verify integer plane accumulation and sign.
 import itertools
 products=0
 for weights in itertools.product((-127,-65,-1,0,1,64,127),repeat=3):
  for spikes in itertools.product((0,1),repeat=3):
   for bits in (8,6,4):
    drop=8-bits
    reference=sum(x*((q>>drop)<<drop) for x,q in zip(spikes,weights))
    serial=sum((-128 if k==7 else 1<<k)*sum(x*((q>>k)&1) for x,q in zip(spikes,weights)) for k in range(drop,8))
    require(reference==serial,'Binary signed integer reference equals retained-plane sum');products+=1
 native_MAC=12000*(384*1536+1536*384);graphs={}
 for bits in (8,6,4):
  rows=rowcounts(bits);rp=sum(rows.values())
  require(rows[7]==12000 and rows[0]==(12000 if bits==8 else 2280),'Sign/all rows and low/tail rows')
  graphs['BS'+str(bits)]={'complete_tile_precision_bits':bits,'tail_tile_precision_bits':8,'fulltile_spatial_points':972,'tail_spatial_points':228,'fullT_rows_each_FC':12000,'per_plane_rows':rows,'row_plane_sum_each_FC':rp,'integer_plane_contraction_terms_two_FC':rp*(384*1536+1536*384),'relative_target_Q8_plane_work_cancelled':1-rp/(8*12000),'hypothetical_whole12_uniform_Q8_FC_plane_work_cancelled':(1-rp/(8*12000))/12,'scaling_FP32_multiplies':12000*(1536+384),'int32_to_FP32_elements':12000*(1536+384),'production_FC_bias_adds':0,'full_target_BN_elements':23040000,'all78_BN_elements':1414344960,'whole12_FC_rows_each':558000,'whole12_FP32_shape_MAC_invariant':12*native_MAC,'actual_mixed_engine_whole_network_cycles':None}
 serial_rows=8*graphs['BS8']['row_plane_sum_each_FC']+10*graphs['BS6']['row_plane_sum_each_FC']+8*graphs['BS4']['row_plane_sum_each_FC']
 require(serial_rows==1990560,'Fixed42 serial-plane rows')
 return {'schema':'flow_r26_independent_bitplane_counter_plan_v1','status':'SOURCE_GEOMETRY_AND_INTEGER_IDENTITY_ONLY','source_sha256':{str(MODEL):MODEL_SHA,str(Path(__file__)):sha(__file__)},'integer_checks':{'single_weight_precision_cases':checked,'binary_three_term_dot_cases':products,'status':'PASS'},'graphs':graphs,
  'fixed42_plan_counts':{'forward_count':42,'quality_endpoints':40,'Q6_replays':2,'native_float_target_forwards':8,'Q8_int_reference_forwards':8,'bitserial_target_forwards':26,'native_target_rows_each_FC':96000,'reference_integer_target_rows_each_FC':96000,'serial_target_rows_each_FC':312000,'serial_row_plane_sum_each_FC':serial_rows,'all12_geometric_rows_each_FC':42*558000,'BN_calls':42*78,'all78_BN_input_elements':42*1414344960},
  'static_weight_and_activation_obligations':{'two_FC_weight_elements':2*384*1536,'original_FP32_weight_bytes':4*2*384*1536,'q8_int8_weight_bytes':2*384*1536,'all8_onebit_planes_packed_total_bytes':2*384*1536,'all8_binary_planes_int8_tensor_bytes_if_all_materialized':8*2*384*1536,'per_output_scale_elements':1536+384,'per_output_scale_FP32_bytes':4*(1536+384),'source_weights_have_bias':False,'binary_input_check_elements_per_forward':12000*(384+1536),'two_FC_int32_output_bytes_per_forward':4*12000*(1536+384),'maximum_abs_integer_dot_conservative':128*1536,'fullT_u_FP32_bytes':4*10*30*40*384},
  'limits':['The real graph retains all12000 targetFC rows, allT/allchannels and full78 currentBN. Only low-plane contractions on fulltile rows may disappear.','PyTorch _int_mm uses int8 tensors/int32 output. Onebit-packed SRAM, AND/popcount hardware and signed-shift accumulation are a proposed execution mapping, not measured RTL/DC.','Rows per plane are a workload counter, not PE cycles; reductions, packing, row-gather/scatter, scale, conversion, buffers and control remain necessary.','All8 weight planes are still needed for8-bit tails. Spatially shared weights are not pruned; no automatic whole-matrix external-read/storage reduction follows.','Whole12 percentages require the explicit hypothetical equal8-plane engine across all12FC pairs. The actual11 FP32 MLPs cannot be summed with target integer plane work as a hardware cycle denominator.','Original dense target14.155776G MAC and whole12 169.869312G MAC describe invariant shape; lowbit approximation changes operation type, not floatMAC counts.','No quality, GPU, model execution, trained scorer, SRAM schedule, physical FP order, RTL/DC, energy or whole-network speedup is established by this script.']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Exclusive receipt');d=report();a.output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':d['status'],'counts':d['fixed42_plan_counts'],'tests':d['integer_checks'],'sha256':sha(a.output)}))
