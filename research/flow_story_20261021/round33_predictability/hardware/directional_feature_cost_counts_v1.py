#!/usr/bin/env python3
"""X160 prototype arithmetic/storage ledger, never a cycle or RTL model."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;S=P.parent/'directional_branch/predictors.py';B=P.parent/'algorithm/predictors.py'
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=module(S,'r33_directional_cost_source');base=module(B,'r33_128_cost_reference')
assert m.N_FEATURES==160 and base.N_FEATURES==128 and m.N_ACTIONS==6
assert m.ALPHA==base.ALPHA==0.1 and m.METHODS==base.METHODS and len(m.METHODS)==7
z=np.zeros((10,12,12),dtype=np.float32);f=m.voxel_features(z)
assert f.shape==(160,) and f.dtype==np.float64 and np.array_equal(f[:128],base.voxel_features(z)) and np.array_equal(f[128:],np.zeros(32))
D=160;shape={}
for g in (1,4):
 aux=np.zeros((1,14*g*g));x=np.zeros((1,D));products=m.component_products(aux,g);aug=m.augmented_features(x,aux,g)
 assert products.size==12*g*g and aug.shape==(1,D+26*g*g)
 shape['g'+str(g)]={'aux_width':aux.shape[1],'component_products':products.size,'augmented_width':aug.shape[1]}
def block(i,o):
 return {'input_width':i,'output_width':o,'dense_product_terms':i*o,'dense_MAC_equivalent_terms':i*o,
 'dot_reduction_adds':(i-1)*o,'bias_adds':o,'centering_subtractions':i,'scaling_divisions':i,
 'FP64_parameter_values':i*o+2*i+o,'FP64_parameter_bytes':8*(i*o+2*i+o),
 'FP64_folded_parameter_bytes_UNVERIFIED':8*(i*o+o)}
blocks={'raw':block(D,6),'aux_g1':block(D,14),'aux_g4':block(D,224),
 'aug_g1':block(D+26,6),'aug_g4':block(D+416,6),'residual_g1':block(D+26,6),'residual_g4':block(D+416,6)}
heads={}
for name in m.METHODS:
 if name=='direct_raw':keys=['raw'];g=0
 else:
  g=int(name[-1]);keys=['aux_g'+str(g)]
  if name.startswith('direct_aug'):keys+=['aug_g'+str(g)]
  if name.startswith('direct_residual'):keys+=['residual_g'+str(g)]
 heads[name]={'ridge_blocks':keys,'dense_MAC_equivalent_terms':sum(blocks[k]['dense_product_terms'] for k in keys),
 'component_products_once_if_shared':12*g*g if g else 0,'signed_reduction_adds':6*(2*g*g-1) if g and (name.startswith('signed') or 'residual' in name) else 0,
 'residual_score_adds':6 if 'residual' in name else 0,
 'FP64_parameter_bytes':sum(blocks[k]['FP64_parameter_bytes'] for k in keys),
 'FP64_folded_parameter_bytes_UNVERIFIED':sum(blocks[k]['FP64_folded_parameter_bytes_UNVERIFIED'] for k in keys),
 'centering_subtractions':sum(blocks[k]['centering_subtractions'] for k in keys),
 'scaling_divisions':sum(blocks[k]['scaling_divisions'] for k in keys)}
suite={k:sum(v[k] for v in blocks.values()) for k in ['dense_product_terms','dense_MAC_equivalent_terms','dot_reduction_adds','bias_adds','centering_subtractions','scaling_divisions','FP64_parameter_values','FP64_parameter_bytes','FP64_folded_parameter_bytes_UNVERIFIED']}
suite.update(component_products_literal=408,component_products_if_shared=204,signed_reduction_adds=192,residual_score_adds=12)
old=json.loads((P/'PREDICTOR_COST_COUNTS_V1.json').read_text())
suite['increment_over_X128_dense_terms']=suite['dense_product_terms']-old['literal_predict_suite_shared_aux_all7_methods']['dense_product_terms']
suite['increment_over_X128_FP64_parameter_bytes']=suite['FP64_parameter_bytes']-old['literal_predict_suite_shared_aux_all7_methods']['deployment_FP64_parameter_bytes']
C,T,H,W=16,10,120,160;mid=C*(T-1)*H*W;n=C*(T-1)*(H-2)*(W-2)
assert n==2684736 and 5*n==13423680
arithmetic={'full_voxel_shape':[10,480,640],'cell_shape':[T,H,W],'mid_shape_per_cell':[T-1,H,W],
 'It_Ix_Iy_shape_per_cell':[T-1,H-2,W-2],'cells':C,'samples_per_cell':n//C,'samples_total':n,
 'midpoint_values':mid,'midpoint_additions':mid,'midpoint_half_scalings':mid,
 'It_subtractions':n,'Ix_Iy_subtractions':2*n,'Ix_Iy_half_scalings':2*n,
 'all_half_scalings':mid+2*n,
 'five_correlation_product_streams':['Ix*It','Iy*It','Ix*Ix','Iy*Iy','It*It_shared'],
 'sample_products':5*n,'five_mathematical_reduction_adds':5*(n-C),
 'denominator_scalar_products':2*C,'sqrt_operations':2*C,'denominator_guard_comparisons':2*C,
 'correlation_divisions_max':2*C,'guard':'sqrt(sum(spatial^2)*sum(It^2)) <= 1e-12 -> 0',
 'new_feature_values':2*C,'new_feature_FP64_bytes':2*C*8,'full_X160_FP64_bytes':D*8,
 'base128_cost_still_required':True,'source_It_square_reused_between_axes':True,
 'raw_FP32_input_bytes':10*480*640*4,'extra_physical_input_read_bytes':None,
 'literal_wrapper_separate_base128_and_directional_calls':True,
 'FP32_input_literal_two_conversion_passes_useful_read_bytes':2*10*480*640*4,
 'two_conversion_passes_not_offchip_traffic_measurement':True,
 'half_scaling_is_not_counted_as_free_shift_or_one_cycle':True}
arrays={'mid_per_cell_FP64_bytes':(T-1)*H*W*8,'each_It_Ix_Iy_per_cell_FP64_bytes':(T-1)*(H-2)*(W-2)*8,
 'mid_plus_three_derivatives_per_cell_FP64_bytes':((T-1)*H*W+3*(T-1)*(H-2)*(W-2))*8,
 'one_temporary_product_array_per_cell_FP64_bytes':(T-1)*(H-2)*(W-2)*8,
 'scope':'sizes of literal arrays; allocation/liveness and measured peak not evaluated'}
stream=[]
for name,width,old_state,cells_active in [('cell',160,230400,1),('row_band',640,921600,4),('full_plane',640,3686400,16)]:
 stream.append({'candidate':name,'old_previous_FP32_and_signed_sum_FP64_bytes':old_state,
  'candidate_three_mid_rows_FP64_bytes':3*width*8,'candidate_two_It_rows_FP64_bytes':2*width*8,
  'candidate_extra_gradient_line_bytes':5*width*8,'five_moment_accumulators_FP64_bytes':5*cells_active*8,
  'old_temporal_previous_shared_only_if_fusion_proven':True,'not_a_minimum_or_implemented_buffer':True,
  'extra_port_and_source_reread_costs_unknown':True,'cell_borders_must_reset_gradient_validity':True})
out={'schema':'round33_directional_X160_theoretical_cost_scope_v1','status':'PROTOTYPE_ARITHMETIC_NO_CYCLES_NO_RTL',
 'source':str(S),'source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),
 'source_freeze_sha256':hashlib.sha256((S.parent/'SOURCE_FREEZE_V1.json').read_bytes()).hexdigest(),'original_X128_source_sha256':hashlib.sha256(B.read_bytes()).hexdigest(),
 'base128_cost_ledger_sha256':hashlib.sha256((P/'PREDICTOR_COST_COUNTS_V1.json').read_bytes()).hexdigest(),
 'actual_small_array_shape_checks':shape,'small_zero_voxel_X160_and_original128_prefix':'PASS',
 'directional_increment_arithmetic':arithmetic,'literal_array_sizes':arrays,'candidate_streaming_state':stream,
 'ridge_blocks':blocks,'single_candidate_heads':heads,'literal_suite_all7_methods':suite,
 'gradient_buffer_schedule_proven':False,'cycle_claim':False,'new_RTL':False,'GPU_used':False,'real_data_read':False,
 'trained_coefficients_loaded':False,'FP32_or_fixedpoint_validated':False,'standardization_folding_verified':False,
 'old128_and_controller_freezes_modified':False}
(P/'DIRECTIONAL_FEATURE_COST_SCOPE_V1.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':out['status'],'shape_checks':shape,'samples':n,'five_products':5*n,'dense_suite_terms':suite['dense_product_terms'],'suite_FP64_parameters':suite['FP64_parameter_bytes']},indent=2))
