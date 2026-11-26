#!/usr/bin/env python3
"""Independent arithmetic/storage ledger; no real data, fitting, GPU or timing."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;S=P.parent/'algorithm/predictors.py'
spec=importlib.util.spec_from_file_location('r33_source_predictors_cost',S);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert(m.FEATURE_GRID,m.FEATURES_PER_CELL,m.N_FEATURES,m.N_ACTIONS)==(4,8,128,6)
assert len(m.METHODS)==7 and len(m.FEATURE_NAMES)==8
D,A=128,6
# These source functions are executed only on small zero arrays to verify widths.
shape_checks={}
for g in (1,4):
 aux=np.zeros((1,14*g*g),dtype=np.float64);x=np.zeros((1,D),dtype=np.float64)
 v,d=m.split_aux(aux,g);products=m.component_products(aux,g);aug=m.augmented_features(x,aux,g)
 assert v.shape==(1,2*g*g) and d.shape==(1,6,2*g*g)
 assert products.shape==(1,6,2*g*g) and aug.shape==(1,128+26*g*g)
 shape_checks['g'+str(g)]={'v_width':v.shape[1],'six_d_width':d.size,'aux_width':aux.shape[1],'component_product_width':products.size,'augmented_width':aug.shape[1]}

def ridge(i,o):
 c=i*o;stored=c+2*i+o;folded=c+o
 return dict(input_width=i,output_width=o,dense_product_terms=c,MAC_equivalent_terms=c,
             mathematical_dot_reduction_adds=o*(i-1),intercept_adds=o,
             runtime_center_subtractions=i,runtime_scale_divisions=i,
             coefficient_values=c,mean_values=i,scale_values=i,intercept_values=o,
             deployment_parameter_values=stored,deployment_FP64_parameter_bytes=stored*8,
             real_arithmetic_folded_values=folded,folded_FP64_parameter_bytes=folded*8,
             folded_numerical_equivalence_verified=False)
blocks={'raw':ridge(128,6),'aux_g1':ridge(128,14),'aux_g4':ridge(128,224),
        'aug_g1':ridge(154,6),'aug_g4':ridge(544,6),'residual_g1':ridge(154,6),'residual_g4':ridge(544,6)}
heads={}
for name in m.METHODS:
 if name=='direct_raw':keys=['raw'];g=0
 else:
  g=int(name[-1]);keys=['aux_g'+str(g)]
  if name.startswith('direct_aug'):keys+=['aug_g'+str(g)]
  if name.startswith('direct_residual'):keys+=['residual_g'+str(g)]
 p=12*g*g if g else 0;reduce=6*(2*g*g-1) if g and (name.startswith('signed') or name.startswith('direct_residual')) else 0
 heads[name]={'ridge_blocks':keys,'dense_MAC_equivalent_terms':sum(blocks[k]['MAC_equivalent_terms'] for k in keys),
  'component_products_once_if_shared':p,'signed_reduction_adds':reduce,'residual_score_adds':6 if 'residual' in name else 0,
  'centering_subtractions_unfolded':sum(blocks[k]['runtime_center_subtractions'] for k in keys),
  'scale_divisions_unfolded':sum(blocks[k]['runtime_scale_divisions'] for k in keys),
  'FP64_deployment_parameter_bytes':sum(blocks[k]['deployment_FP64_parameter_bytes'] for k in keys),
  'FP64_folded_parameter_bytes_unverified':sum(blocks[k]['folded_FP64_parameter_bytes'] for k in keys),
  'bias_adds':sum(blocks[k]['intercept_adds'] for k in keys),
  'scope':'single selected method algebra; reuse identical component products for residual is prospective, current suite recomputes them'}
suite={k:sum(b[k] for b in blocks.values()) for k in ['dense_product_terms','MAC_equivalent_terms','mathematical_dot_reduction_adds','intercept_adds','runtime_center_subtractions','runtime_scale_divisions','coefficient_values','deployment_parameter_values','deployment_FP64_parameter_bytes','folded_FP64_parameter_bytes']}
suite.update(component_products_literal=2*(12+192),component_products_if_shared=12+192,signed_reduction_adds=6+186,residual_score_adds=12,seven_scores_returned=True,single_deployed_method_cost=False)
T,H,W=10,480,640;sp=H*W;N=T*sp;cells=16;cellsp=sp//cells
features={'T':T,'H':H,'W':W,'cell_H':120,'cell_W':160,'cells':16,'features':128,
 'raw_FP32_values':N,'raw_FP32_bytes_one_useful_read':N*4,'input_finiteness_tests':N,'current_full_FP64_conversion_bytes':N*8,
 'adjacent_pairs':(T-1)*sp,'adjacent_subtractions':(T-1)*sp,'adjacent_absolute_values':(T-1)*sp,
 'signed_temporal_sum_adds_mathematical':(T-1)*sp,'signed_temporal_sum_abs_values':sp,
 'signed_abs_spatial_reduce_adds':sp-cells,
 'per_positive_or_negative_mean_reduce_adds':N-cells,
 'per_positive_or_negative_predicate_values':N,
 'absolute_amplitude_values':N,
 'mass_t_spatial_reduce_adds':N-cells*T,'mass_temporal_reduce_adds':cells*(T-1),
 'two_moment_numerator_products':2*cells*T,'two_moment_numerator_reduce_adds':2*cells*(T-1),
 'literal_edges_squared_products_in_cell_loop':cells*T,
 'data_dependent_mass_divisions_max':cells*2,'mass_zero_tests':cells,
 'adjacent_mean_reduce_adds':(T-1)*sp-cells,
 'feature_FP64_output_bytes':128*8,
 'constant_normalization_scalar_divisions_literal':cells*7,
 'normalization_detail':'positive/negative means2, sign fractions2, signed-sum spatial mean and /T2, adjacent mean1 per cell; plus up to2 mass divisions',
 'count_update_implementation':'positive/negative uint counters; no invented FP MAC or cycle cost',
 'reduction_counts_scope':'mathematical operations; numpy reduction tree and hardware FP64 rounding not validated',
 'raw_input_dtype_contract':'FP32 full T10 input supplied by task; collect.py explicitly checks shape, not dtype',
 'physical_read_traffic_proven':False}
buffers=[]
for name,height,width,transactions,layout in [
 ('time_major_full_plane',480,640,10,'10 full-frame contiguous slabs; whole-plane state'),
 ('row_band_120_by_640',120,640,40,'4 spatial row bands x10 time slabs; each slab contiguous'),
 ('cell_120_by_160',120,160,19200,'16 cells x10 time slabs x120 row segments;640 useful bytes per strided row, unless a2D DMA handles them')]:
 n=height*width
 buffers.append({'candidate':name,'previous_FP32_bytes':n*4,'signed_sum_FP64_bytes':n*8,
 'two_main_buffers_bytes':n*12,'useful_input_bytes_one_pass':N*4,
 'contiguous_source_segments_without_2D_DMA':transactions,
 'layout':layout,'small_statistics_excluded':True,'read_modify_write_ports_not_scheduled':True,'existing_RF_or_input_buffers_not_assumed_free':True})
report={'schema':'round33_predictor_arithmetic_storage_scope_v1','source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),
 'source':str(S),'executed_shape_checks':shape_checks,'ridge_blocks':blocks,'single_candidate_heads':heads,
 'literal_predict_suite_shared_aux_all7_methods':suite,'voxel_feature_arithmetic':features,'candidate_streaming_buffers':buffers,
 'logical_head_arrays_FP64_bytes_not_peak':{'raw_features':1024,'aux_g1':112,'aux_g4':1792,'products_g1':96,'products_g4':1536,'augmented_g1':1232,'augmented_g4':4352,'one_six_score_vector':48,'normalization_temporaries':'current NumPy creates vector intermediates; no liveness/port schedule or peak claim'},
 'action_selection':{'scores':6,'lower_is_better':True,'argmin_logical_comparisons_for_one_reduction':5,'allow_no_upgrade_zero_comparison':1,'tie_order':'no-upgrade first at minimum>=0; otherwise lower upgrade id','interface_map':'software -1 -> controller action6','literal_suite_returns_both_fixed6_and_allowQ4':True,'no_cycle_claim':True},
 'cycle_claim':False,'new_RTL':False,'actual_trained_coefficients_loaded':False,'FP32_or_fixedpoint_validated':False,
 'folding_verified':False,'frozen_controller_modified':False,'real_data_read':False,'GPU_used':False}
(P/'PREDICTOR_COST_COUNTS_V1.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':'COUNT_LEDGER_COMPLETE_NO_CYCLE_CLAIM','shape_checks':shape_checks,'suite_dense_terms':suite['MAC_equivalent_terms'],'suite_FP64_parameters':suite['deployment_FP64_parameter_bytes'],'raw_FP32_bytes':N*4},indent=2))
