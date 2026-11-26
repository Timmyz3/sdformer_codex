#!/usr/bin/env python3
"""Frozen ROUND6 source binding and synthetic CPU shape/dtype checks only.

Never imports/runs the analyzer or collector, reads data/results/GT, fits a
predictor, uses a network/GPU, or overwrites the declared-count snapshot.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="2"
sys.dont_write_bytecode=True
import numpy as np

HERE=Path(__file__).resolve().parent
SNAPSHOT_SHA="b48f996804501200d4b7f108231632cf342caf469065cf44b33ce70374cae78a"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok,message):
    if not ok:
        raise ValueError(message)


def functions(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-freeze",type=Path,default=HERE/"analysis_freeze_v1.json")
    freeze=parser.parse_args().analysis_freeze
    frozen=json.loads(freeze.read_text())
    for name,pin in frozen["source_sha256"].items():
        require(sha(HERE/name)==pin,"Analysis freeze differs: "+name)
    collector_freeze=HERE/"collector_freeze_v1.json"
    collector=json.loads(collector_freeze.read_text())
    require(sha(HERE/"collect_singletons.py")==collector["source_sha256"]["collect_singletons.py"],"Collector freeze differs")
    snapshot=HERE/"predictor_cost_contract.json"
    require(sha(snapshot)==SNAPSHOT_SHA,"Declared snapshot changed")
    declared=json.loads(snapshot.read_text())
    path=HERE/"predictor_models.py"
    spec=importlib.util.spec_from_file_location("_round6_predictor_source_bound",path)
    pm=importlib.util.module_from_spec(spec);spec.loader.exec_module(pm)
    pf=functions(path);af=functions(HERE/"analyze_predictors.py");cf=functions(HERE/"collect_singletons.py")
    std_calls=[n for n in ast.walk(pf["vector_statistics"]) if isinstance(n,ast.Call) and
               isinstance(n.func,ast.Attribute) and n.func.attr=="std"]
    require(len(std_calls)==1 and any(k.arg=="ddof" and ast.literal_eval(k.value)==0 for k in std_calls[0].keywords),"Population std source")
    require(tuple(pm.BASE_FEATURES)==('amplitude_abs_mean','feature_rms','pooling_residual_mse',
                                     'spatial_neighbor_abs_difference','temporal_neighbor_abs_difference'),"B5 definitions")
    require(tuple(pm.TILES)==tuple((r,c) for r in range(3) for c in range(4)),"Legal candidate order")
    checks=[]
    rng=np.random.RandomState(20261014)
    cells=rng.standard_normal((10,14,384)).astype(np.float32)
    head={'mean':np.linspace(-.2,.2,384),'std':np.linspace(.5,1.5,384),
          'beta':rng.standard_normal((384,2))*.01,'intercept':np.array([.25,-.5])}
    weight,bias=pm.folded(head)
    output=pm.predict_fp32(head,cells)
    expected=cells @ weight.astype(np.float32)+bias.astype(np.float32)
    require(output.shape==(10,14,2) and output.dtype==np.float32 and np.array_equal(output,expected),"Folded FP32 head")
    np.testing.assert_allclose(output,pm.predict(head,cells),atol=1e-6,rtol=2e-5)
    checks.append("synthetic_140x384_head_folded_FP32_shape_and_explicit_affine_exact")

    vectors=np.stack(np.meshgrid(np.arange(3),np.arange(3)),axis=-1).astype(np.float32)
    statistics=pm.vector_statistics(vectors)
    require(statistics.shape==(6,) and statistics.dtype==np.float32,"Vector feature dtype/shape")
    mag=np.sqrt((vectors.astype(np.float64)**2).sum(-1))
    reference=np.array([mag.mean(),mag.max(),mag.std(ddof=0),1.,1.,
                        np.sqrt((vectors.astype(np.float64).mean((0,1))**2).sum())/(mag.mean()+1e-6)])
    np.testing.assert_allclose(statistics,reference,rtol=2e-6,atol=1e-7)
    require(abs(float(statistics[2])-mag.std(ddof=1))>1e-3,"Population vs sample std indistinguishable test")
    np.testing.assert_array_equal(pm.vector_statistics(np.zeros((3,3,2),dtype=np.float32)),np.zeros(6,dtype=np.float32))
    checks.append("six_FP32_vector_statistics_population_std_edges_and_zero_guard")

    grouped_input=np.tile(np.repeat(np.arange(16),24),(3,3,1)).astype(np.float32)
    rich=pm.richer_statistics(grouped_input)
    require(rich.shape==(32,) and rich.dtype==np.float32,"Rich subfeatures dtype/shape")
    np.testing.assert_array_equal(rich[:16],np.arange(16));np.testing.assert_array_equal(rich[16:],np.zeros(16))
    ordinary=[dict(tile=list(tile),**{name:float(i+j)/20 for j,name in enumerate(pm.BASE_FEATURES)})
              for i,tile in enumerate(pm.TILES)]
    families=pm.feature_families(ordinary,cells,output,output)
    for name,width in (("B5",5),("rich37",37),("PCA11",11),("flow11",11)):
        xx=families[name]
        require(xx.shape==(12,width) and xx.dtype==np.float32,"Family dtype/width "+name)
        selector={'mean':np.zeros(width),'std':np.ones(width),
                  'beta':np.linspace(-.1,.2,width),'intercept':np.array(.125)}
        score=pm.predict_fp32(selector,xx)
        require(score.shape==(12,) and score.dtype==np.float32 and 0<=pm.top1(score)<12,"One-selector interface")
    np.testing.assert_array_equal(families["PCA11"],families["flow11"])
    checks.append("rich_contiguous24_groups_and_12x5_37_11_11_FP32_single_selector_interfaces")
    require(pm.top1(np.ones(12,dtype=np.float32))==0,"Lexicographic top1 tie")
    checks.append("top1_deterministic_lexicographic_tie")

    rich_cost=declared["rich_grouped_feature_cost"]
    projected=declared["six_projected_features_all12"]
    # Two explicit mag.mean calls, plus np.std's own centering mean. The
    # ordinary arithmetic ledger reuses one mean for all three consumers.
    literal=dict(projected);literal["add"]+=2*12*8;literal["constant_scale"]+=2*12
    evidence=[]
    for filename,names,mapping in (
        ("predictor_models.py",("folded","predict_fp32","vector_statistics","richer_statistics","feature_families","top1"),pf),
        ("analyze_predictors.py",("load_input","fit_projections","family_for","main"),af),
        ("collect_singletons.py",("cell_descriptors",),cf)):
        evidence.extend(dict(file=filename,function=name,line=mapping[name].lineno) for name in names)
    result=dict(schema="flow_round6_predictor_source_binding_v1",status="SOURCE_BINDING_AND_SYNTHETIC_CPU_PASS",
        source_sha256={str(p):sha(p) for p in (freeze,collector_freeze,snapshot,path,HERE/"analyze_predictors.py",
                                             HERE/"collect_singletons.py",Path(__file__))},
        function_locations=evidence,CPU_checks=checks,CPU_check_count=len(checks),numpy_version=np.__version__,
        quality_results_or_GT_read=False,model_fitting_or_network_inference=False,real_weights_tested=False,
        source_contract=dict(cell_pool_then_full_T_mean=True,private_PSN_T_retained=10,
                             projection_arithmetic="folded coefficients cast to FP32; FP32 x@weight+bias",
                             projected_statistics_dtype="float32",std_ddof=0,
                             feature_widths=dict(B5=5,rich37=37,PCA11=11,flow11=11),
                             scalar_selector_per_isolated_deployment_policy=1),
        source_cost_adjustments=dict(
            rich_grouping_actual_source_cells=108,rich_grouping_actual_ops=rich_cost["candidate108_grouping"],
            projection_actual_source_cells=140,projection_per_policy=declared["flow_and_PCA_same_projector_full140"],
            six_stats_ordinary_shared_mean_ops=projected,six_stats_unshared_centering_mean_schedule=literal,
            numpy_kernel_instruction_count_measured=False,
            literal_difference="Two explicit mag.mean calls plus np.std centering; a direct unshared schedule adds 192 adds and 24 constant scalings vs one shared mean. This is not a NumPy kernel trace."),
        deployment_boundary=dict(
            analyzer_is_multimethod_offline_evaluation=True,
            analyzer_also_computes_FP64_selector_reference=True,
            folding_currently_recomputed_in_predict_fp32_call=True,
            folded_FP32_weights_can_be_precomputed_for_fixed_models=True,
            persistent_single_policy_deployment_adapter_implemented_here=False,
            original_main_policy_count=8,additional_B5_small4_controls=2,
            one_policy_cost_is_not_sum_of_offline_candidates=True,
            actual_hardware_cycles_or_net_saving_measured=False),
        fixed_k1=dict(all12_fc_rows_each=154320,target_fc_rows_each=2120,
                      producer_services_identical_across_selectors=True,
                      selector_does_not_itself_add_cancelled_PE_services=True,
                      condition_for_net_gain="Need matched-quality lower legal producer budget or a verified same-function total service reduction, including predictor and current-BN cost"),
        history=dict(declared_snapshot_unchanged_SHA256=SNAPSHOT_SHA,
                     source_binding_does_not_assert_completed_collection=True))
    out=HERE/"predictor_source_binding.json"
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    require(sha(snapshot)==SNAPSHOT_SHA,"Declared snapshot was modified")
    print(json.dumps(dict(status=result["status"],checks=len(checks),output=out.name,
                         no_GT_or_quality_read=True,rich_source_cells=108,projection_cells=140)))


if __name__=="__main__":
    main()
