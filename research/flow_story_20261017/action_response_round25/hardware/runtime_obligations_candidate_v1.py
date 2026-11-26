#!/usr/bin/env python3
"""Static source/geometry obligations only: no model import, tensors, GT, or fit."""
import argparse,ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;DESIGN=HERE.parent/'design'
PINS={'response_models.py':'686e2a565683d843af293a0cf61d48c8c9533a67c6ad0c25465dce3741caa21c','fit_response_screen.py':'abee518d769498ad208bc404a8b30b568784ee364082eb1dc27c75536e838d4c','PROTOCOL_PRE_LABELS_V1.md':'892158c0722bc9f27ebc9bd813ae0171eea1fefa0d535a3cfa8fa5e2797f84ae'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def report():
 for name,digest in PINS.items():require(sha(DESIGN/name)==digest,'Exact candidate source '+name)
 tree=ast.parse((DESIGN/'response_models.py').read_text());conv=[];linear=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('Conv2d','Linear'):
   args=[ast.literal_eval(x) if isinstance(x,ast.Constant) else ast.unparse(x) for x in n.args]
   (conv if n.func.attr=='Conv2d' else linear).append(args)
 require(sorted(conv)==sorted([[2,8,3],[8,8,3],[2,4,1]]),'Actual Conv source dimensions')
 require([8,4] in linear and [16,1] in linear and len(linear)==3,'Actual Linear source dimensions')
 H,W,C,T=30,40,384,10;P=H*W;A=12;X=1186;R=8;D=P*2
 tiles=[(r,c,min(9,H-9*r)*min(9,W-9*c)) for r in range(4) for c in range(5)]
 edges=[(y,x,y+dy,x+dx,(y//9)*5+x//9) for y in range(H) for x in range(W) for dy,dx in ((0,1),(1,0)) if y+dy<H and x+dx<W]
 E=len(edges);groups=[sum(e[-1]==i for e in edges) for i in range(20)]
 require(sum(v[2] for v in tiles)==P and E==2330 and groups[-1]==17,'All20 real tails and all2330 edges')
 feature={
  'fullT_mean':{'input_elements':T*P*C,'input_bytes_FP32':4*T*P*C,'adds':(T-1)*P*C,'scales':P*C,'output_elements':P*C},
  'group24_mean':{'original_prefix_operand_elements':P*C,'adds':P*16*23,'scales':P*16,'output_elements':P*16},
  'global_group_context':{'mean_adds':16*(P-1),'mean_scales':16,'population_std_subs':16*P,'population_std_squares':16*P,'population_std_adds':16*(P-1),'population_std_scales':16,'sqrt':16,'output_elements':32},
  'complete12_tile_mean':{'input_elements':A*81*C,'adds':A*80*C,'scales':A*C,'output_elements':A*C},
  'complete12_tile_std':{'subtractions':A*81*C,'squares':A*81*C,'sum_adds':A*80*C,'scales':A*C,'sqrt':A*C,'output_elements':A*C},
  'complete12_H1_pool_expand':{'mean_input_elements':A*81*C,'mean_adds':A*9*(2+5)*C,'scales':A*18*C,'representative_elements':A*18*C,'expanded_written_elements':A*81*C},
  'H1_residual_RMS':{'subtractions':A*81*C,'squares':A*81*C,'sum_adds':A*80*C,'scales':A*C,'sqrt':A*C,'output_elements':A*C},
  'shared_X':{'elements_per_action':X,'all12_elements':A*X,'all12_bytes_FP32':4*A*X,'global_context_replication_elements':A*32,'coordinate_constants':A*2}}
 require(3*C+32+2==X,'1186 descriptor source geometry')
 producer={'rank':R,'MAC_per_action':X*R+R*D,'MAC_per_frame_all12':A*(X*R+R*D),'bias_adds_per_frame':A*(R+D),'weight_bias_basis_mean_elements':X*R+R+R*D+D,'weight_bias_basis_mean_bytes_FP32':4*(X*R+R+R*D+D),'coefficients_all12_elements':A*R,'response_all12_elements':A*D,'response_all12_bytes_FP32':4*A*D,'source_inference_dtype':'FP32','online_true_action_outputs':0}
 direct_head=X*16+16;response_head=(X+160)*16+16
 methods={
  'Gdirect':{'head_MAC_per_action':direct_head,'body_MAC_per_action':0,'parameters':(X+1)*16+17,'bias_adds_per_action':17,'ReLU_per_action':16,'pool_adds_per_action':0,'pool_scales_per_action':0,'requires_response_producer':False},
  'Gresponse':{'head_MAC_per_action':response_head,'body_MAC_per_action':P*(2*8*9+8*8*9),'parameters':(2*9+1)*8+(8*9+1)*8+(X+160+1)*16+17,'bias_adds_per_action':P*16+17,'ReLU_per_action':P*16+16,'pool_adds_per_action':8*(P-20),'pool_scales_per_action':160,'requires_response_producer':True},
  'Fresponse':{'head_MAC_per_action':response_head,'body_MAC_per_action':P*2*4+E*8*4,'parameters':(2+1)*4+(8+1)*4+(X+160+1)*16+17,'bias_adds_per_action':P*4+E*4+17,'ReLU_per_action':P*4+E*4+16,'pool_adds_per_action':4*(P-20)+4*(E-20),'pool_scales_per_action':160,'edge_delta_subtractions_per_action':E*2,'edge_endpoint_scalar_gathers_per_action':E*4,'edge_cat_written_elements_per_action':E*8,'edge_output_elements_per_action':E*4,'requires_response_producer':True}}
 for name,v in methods.items():
  v['network_MAC_per_frame_all12']=A*(v['head_MAC_per_action']+v['body_MAC_per_action'])
  v['network_plus_response_MAC_per_frame']=v['network_MAC_per_frame_all12']+(producer['MAC_per_frame_all12'] if v['requires_response_producer'] else 0)
  v['weights_bytes_FP32']=4*v['parameters']
  v['X_scaler_subtractions_per_frame']=A*X;v['X_scaler_divisions_per_frame']=A*X
  v['needed_response_scaler_subtractions_per_frame']=A*D if v['requires_response_producer'] else 0
  v['needed_response_scaler_divisions_per_frame']=A*D if v['requires_response_producer'] else 0
  v['target_unscale_multiplies_per_frame']=A;v['target_unscale_adds_per_frame']=A;v['top1_comparisons_per_frame']=A-1
  v['selected_action_cost_included']=False
 require([methods[k]['parameters'] for k in ('Gdirect','Gresponse','Fresponse')]==[19009,22305,21617],'Independent parameter formulas match published CPU architecture')
 return {'schema':'flow_r25_candidate_runtime_obligations_v1','status':'STATIC_SOURCE_WORK_ONLY_NO_INFERENCE_OR_FIT','source_sha256':{str(DESIGN/k):v for k,v in PINS.items()}|{str(Path(__file__)):sha(__file__)},'source_dimensions':{'Conv2d':conv,'Linear':linear},'feature_producer_per_frame':feature,'response_producer':producer,'methods':methods,
  'geometry':{'tiles':tiles,'edge_count':E,'edge_groups':groups,'cross_tile_edges_included':True},
  'storage_elements_or_bytes':{'full_T_u_FP32_bytes':T*P*C*4,'time_mean_FP32_bytes':P*C*4,'global_grouped_FP32_bytes':P*16*4,'all12_X_FP32_bytes':A*X*4,'all12_response_FP32_bytes':A*D*4,'one_response_FP32_bytes':D*4,'X_scaler_mean_std_FP32_bytes':X*2*4,'response_scaler_mean2_std1_FP32_bytes':12,'structured_source_dst_and_group_indices_int64_bytes':3*E*8,'structured_direction_FP32_bytes':2*E*4,'one_generic_8ch_map_FP32_bytes':P*8*4,'one_structured4ch_node_FP32_bytes':P*4*4,'one_structured8input_edge_FP32_bytes':E*8*4,'one_structured4output_edge_FP32_bytes':E*4*4},
  'offline_only':{'response_target_formula':'action_head3 - T2_head3, all480x640 pixels reduced to30x40x2; no runtime counterfactual','PCA_ridge_dtype':'FP64; 8 LOO folds plus final full16 fit; FP32 frozen coefficients for inference','gain_head_optimization':'FP32 CPU400 steps per each of3 heads; full batch192 actions; no real fit run by this ledger','OOF_fit_rows_each':168,'final_fit_rows':192,'response_target_dimensions':2400,'rank':8,'ridge_alpha':10},
  'limits':['These are nominal source algebra / tensor obligations, not physical instructions, DRAM transactions, cycle or net-resource equivalence. NumPy std counted as centered population two-pass algebra; actual kernel order/temporary bytes unmeasured.','All methods need shared X; direct deployment does not require delta. Current infer_head wrapper nevertheless evaluates and scales d even for direct, a benchmark-only avoidable operation which is not silently charged to an optimized direct head.','FullT input or a streaming mean accumulator with proven producer/lifetime schedule must be retained; full-time u is18.432MB, mean1.8432MB. No 2MiB fit claim for all tensors together.','All12 actions processed; true20 tile and2330 global edges retained. A future streaming address implementation can avoid explicit index arrays, but current Torch has those buffers.','The ordinary seven-score controls also merit FP32/producer fusion alternatives; prior FP64 one-lane implementation is not a mandatory ordinary baseline.','Both response methods share the same predicted delta, fit supervision and full-domain costs; smaller nominal MAC is not same-net-budget or better quality.','Current source is a candidate, not final freeze. Execution depends on R24 all8 feasible gate; no data/GT/GPU/RTL/model forward or training executed here.']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New receipt only');out=report();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'methods':{k:v['network_plus_response_MAC_per_frame'] for k,v in out['methods'].items()},'sha256':sha(a.output)}))
