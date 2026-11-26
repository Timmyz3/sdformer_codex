#!/usr/bin/env python3
"""Declared ROUND6 arithmetic/memory volumes; stdlib only, no model or quality.

This is a source-bound shape ledger for a proposed interface. It is not a
runtime trace, a cycle simulator, a norm-kernel implementation or synthesis.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREV = HERE.parent/"damage_allocation_probe"
PINS = {
    PREV/"hardware_contract_complete_bn_audit.json": "9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567",
    PREV/"hardware_contract_complete_services.json": "0375388e5b6aa777842819d228f239f3283772d5d7a0bc8a7582bc396c212db1",
    PREV/"run_tile_interventions.py": "68ca81dc2dada3c6ce17f1d7edd7b762e992808c6f2fc359f5fe104cc20e214f",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scale(d, n):
    return {k:v*n for k,v in d.items()}


def sum_ops(items):
    result={}
    for item in items:
        for k,v in item.items():
            result[k]=result.get(k,0)+v
    return result


def projection(cells, in_channels=384, out_channels=2):
    return dict(real_dot_products=cells*in_channels*out_channels,
                dot_reduction_additions=cells*(in_channels-1)*out_channels,
                bias_additions=cells*out_channels,
                parameters_with_bias=(in_channels+1)*out_channels,
                parameters_FP32_bytes=4*(in_channels+1)*out_channels,
                input_elements=cells*in_channels,output_elements=cells*out_channels)


def main():
    for p,pin in PINS.items():
        require(sha(p)==pin,"Source changed: "+str(p))
    previous=json.loads((PREV/"hardware_contract_complete_services.json").read_text())
    bn=json.loads((PREV/"hardware_contract_complete_bn_audit.json").read_text())
    t,h,w,c,tile,stride,n=10,30,40,384,9,3,12
    cells=((h+stride-1)//stride)*((w+stride-1)//stride)
    cell_areas=[min(stride,h-y)*min(stride,w-x) for y in range(0,h,stride) for x in range(0,w,stride)]
    require(cells==140 and sum(cell_areas)==1200 and cell_areas.count(9)==130 and cell_areas.count(3)==10,"Tail cells")
    per_tile_e=t*tile*tile*c
    candidate_cells=n*(tile//stride)**2
    shared=dict(pool_input_elements=t*h*w*c,pool_additions=t*c*(h*w-cells),
                pool_constant_scalings=t*c*cells,pooled_T10_elements=t*c*cells,
                temporal_mean_additions=(t-1)*c*cells,temporal_mean_constant_scalings=c*cells,
                descriptor_elements=c*cells,descriptor_shape_HWC=[10,14,c])
    require(shared["pool_additions"]==4070400 and shared["pooled_T10_elements"]==537600,"Source pool geometry")
    k1=previous["services_by_number_of_fine_candidates"]["1"]
    require(k1["target"]["fc_rows_each"]==2120 and k1["all12_fc_rows_each"]==154320,"Prior actual k1")

    # Same five declared R5 statistics, computed on the 12 complete raw u tiles.
    b5_per_tile={
        "amplitude_abs_mean":dict(absolute=per_tile_e,add=per_tile_e-1,constant_scale=1),
        "feature_rms":dict(square=per_tile_e,add=per_tile_e-1,constant_scale=1,sqrt=1),
        "pooling_residual_MSE_excluding_shared_pool":dict(subtract=per_tile_e,square=per_tile_e,
                                                        add=per_tile_e-1,constant_scale=1),
        "spatial_neighbor_abs_difference":dict(subtract=2*t*c*tile*(tile-1),absolute=2*t*c*tile*(tile-1),
                                                add=2*(t*c*tile*(tile-1)-1)+1,constant_scale=3),
        "temporal_neighbor_abs_difference":dict(subtract=(t-1)*tile*tile*c,absolute=(t-1)*tile*tile*c,
                                                 add=(t-1)*tile*tile*c-1,constant_scale=1),
    }
    b5_all={key:scale(v,n) for key,v in b5_per_tile.items()}
    # Explicit direct centered population-std schedule for six features of 9 2D vectors.
    flow_six_per_tile={
        "nine_vector_L2_norms":dict(square=18,add=9,sqrt=9),
        "norm_mean_max":dict(add=8,constant_scale=1,compare=8),
        "norm_population_std":dict(subtract=9,square=9,add=8,constant_scale=1,sqrt=1),
        "twelve_neighbor_vector_L2_differences":dict(subtract=24,square=24,add=12,sqrt=12),
        "neighbor_mean_max":dict(add=11,constant_scale=1,compare=11),
        "coherence":dict(add=18,constant_scale=2,square=2,sqrt=1,divide=1),
    }
    require(sum_ops(flow_six_per_tile.values())==dict(square=53,add=66,sqrt=23,constant_scale=5,compare=19,subtract=33,divide=1),"Six-feature arithmetic")
    group=dict(groups=16,channels_each=24,
        full140_grouping=dict(add=cells*16*23,constant_scale=cells*16,output_elements=cells*16),
        candidate108_grouping=dict(add=candidate_cells*16*23,constant_scale=candidate_cells*16,output_elements=candidate_cells*16),
        tile_mean_all12=dict(add=n*16*8,constant_scale=n*16,output_elements=n*16),
        tile_neighbor_abs_mean_all12=dict(subtract=n*16*12,absolute=n*16*12,
                                         add=n*16*11,constant_scale=n*16,output_elements=n*16))
    selectors={}
    for name,d in (("B5",5),("rich37",37),("PCA11",11),("flow11",11)):
        selectors[name]=dict(dimension=d,one_deployed_target=projection(n,d,1),
                              both_targets_offline=projection(n,d,2),
                              folded_standardization=True)
    largest_bn=max(bn["external_BN"]["records"],key=lambda x:x["elements"])
    memory=dict(raw_u_FP32_bytes=4*t*h*w*c,candidate_u_FP32_bytes=4*n*per_tile_e,
                complete_tile_u_FP32_bytes=4*per_tile_e,
                ordinary_streaming_best_plus_incoming_tile_u_FP32_bytes=8*per_tile_e,
                pooled_T10_FP32_bytes=4*t*c*cells,descriptor_FP32_bytes=4*c*cells,
                flow_or_PCA_map_FP32_bytes=4*cells*2,group_map_FP32_bytes=4*cells*16,
                top1_scores_FP32_bytes=n*4,
                B5_features_FP32_bytes=n*5*4,rich37_features_FP32_bytes=n*37*4,
                projected11_features_FP32_bytes=n*11*4,
                largest_external_BN_input_FP32_bytes=4*largest_bn["elements"],
                largest_external_BN_name=largest_bn["name"],
                physical_transactions_or_peak_live_bytes=None)
    out=dict(schema="flow_round6_predictor_declared_cost_v1",status="DECLARED_COUNTS_PASS_NOT_RUNTIME_MEASUREMENT",
        source_sha256={str(p):sha(p) for p in (*PINS,Path(__file__))},
        input_contract=dict(T=t,B=1,H=h,W=w,C=c,cell_stride=3,cell_grid=[10,14],
                            full_tile_candidates=12,candidate_cells=candidate_cells,k=1,
                            private_PSN_T_retained=10,selector_target_names=["task_damage","feature_MSE"],
                            definition_status="Parent-declared ROUND6 interface; actual implementation source binding pending",
                            std_ddof_assumption=0,neighbor_rule="12 within-tile horizontal/vertical edges; no cross-tile edges",
                            group_rule="contiguous nonoverlapping groups of 24 channels; arithmetic mean"),
        shared_pool_and_time_mean=shared,k1_actual_producer_reference=k1,
        fine_winner_prepaid_pool=dict(add=t*c*(81-9),constant_scale=t*c*9,
                                     reason="All candidate cell descriptors and B5 pooling residual require this before selection"),
        B5_direct_operations_all12=b5_all,B5_direct_total_excluding_shared_pool=sum_ops(b5_all.values()),
        rich_grouped_feature_cost=group,
        flow_and_PCA_same_projector_full140=projection(cells),
        optional_common_candidate108_projection=projection(candidate_cells),
        optional_common_candidate108_time_mean=dict(add=(t-1)*c*candidate_cells,constant_scale=c*candidate_cells),
        six_projected_features_per_tile=flow_six_per_tile,
        six_projected_features_all12=scale(sum_ops(flow_six_per_tile.values()),n),selectors=selectors,
        top1=dict(candidate_count=12,min_pair_comparisons=11,
                  tie_rule="higher score then lexicographic tile coordinate; finite guard required",
                  decision_barrier="Winner fine cannot be finalized before all candidate scores; ordinary loser coarse retirement may stream",
                  cycles=None),logical_memory_volumes=memory,
        BN_closure_known=dict(external_count=54,private_frozen_count=24,
            external_elements=bn["external_BN"]["total_elements"],
            declared_old_arithmetic_subtotal=bn["conditional_service_denominator"]["known_arithmetic_issue_subtotal"],
            complete_issue_denominator=None,cycle_denominator=None),
        exclusions=["No quality ranks read; no native inference, fit, GPU, RTL, synthesis or server access.",
                    "Dot products/MAC terms, adds, square roots and bytes are separate; never summed as PE cycles.",
                    "Eight offline fitted selectors are alternatives, not eight simultaneous deployment heads.",
                    "Shared pool/mean and ordinary fusion/reassociation opportunities apply equally to controls.",
                    "FP32 equivalence and numerical conversion cost of folded weights/reassociation remain to be checked."])
    destination=HERE/"predictor_cost_contract.json"
    destination.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(status=out["status"],cells=cells,projection_MACs=projection(cells)["real_dot_products"],
                         selectors_MACs={k:v["one_deployed_target"]["real_dot_products"] for k,v in selectors.items()},
                         B5_direct_ops=out["B5_direct_total_excluding_shared_pool"],output=destination.name)))


if __name__=="__main__":
    main()
