#!/usr/bin/env python3
"""Pure-stdlib source/shape accounting; never imports torch or runs a model.

Optional --results validates already-produced BN pre-hooks. BN arithmetic is a
declared two-pass mathematical schedule, not a PyTorch bit-exact implementation
or a characterized cycle model. Writes only its own receipt in this directory.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESEARCH = REPO / "research"
NATIVE = REPO / "SDformer/third_party/SDformerFlow"
STATIC = RESEARCH / "flow_story_20261011/preflight/execution_contract/execution_services_results.json"
CMA = RESEARCH / "flow_story_20261013/breadth/results_bn/bn_cma176_full9.json"
RUNNER = HERE / "run_tile_interventions.py"
MERGE = HERE.parent / "structured_merge/structured_merge_mlp.py"
FREEZE = HERE / "freeze_v1.json"
STATIC_SHA = "3cc67c3c331dc8475cd1647003a684bb6a36843e8b6ce00faa5b41f1f30968e8"
NATIVE_FILES = ("models/STSwinNet_SNN/Spiking_modules.py",
                "models/STSwinNet_SNN/Spiking_swin_transformer3D.py",
                "models/STSwinNet_SNN/Spiking_submodules.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prod(values):
    result = 1
    for v in values:
        result *= v
    return result


def ceil16(n):
    return (n+15)//16


def native_bn_shape(name, channel, stages):
    prefix = "sttmultires_unet."
    tail = name[len(prefix):]
    m = re.fullmatch(r"encoders.swin3d.layers.(\d).swin_blocks.(\d).attn.(bn_q|bn_k|proj_bn).norm_layer", tail)
    if m:
        s, block, _ = m.groups()
        stage = stages[int(s)]
        require(int(block) < stage["depth"], "attention block")
        h, w, c = stage["feature_HWC"]
        require(c == channel, "attention C")
        windows = 5*((h+8)//9)*((w+8)//9)
        return [2, windows, c, 9, 9], "padded_attention", "window_partition_v2 + attention pre-unpad BN"
    m = re.fullmatch(r"encoders.swin3d.layers.(\d).downsample.norm.norm_layer", tail)
    if m:
        stage = int(m.group(1))+1
        h, w, c = stages[stage]["feature_HWC"]
        require(c == channel, "merge C")
        return [10, 1, c, h, w], "patch_merge", "MS_SpikingPatchMerging after reduction"
    if tail.startswith("encoders.swin3d.patch_embed."):
        sub = tail[len("encoders.swin3d.patch_embed."):]
        if sub == "head.norm_layer.norm_layer":
            return [10, 1, 48, 480, 640], "patch", "MS_PED patch head stride1"
        if sub == "conv.norm_layer.norm_layer" or re.fullmatch(r"residual_encoding.resblocks.[01].norm[12].norm_layer", sub):
            return [10, 1, 96, 240, 320], "patch", "patch conv stride2 / residual unchanged"
        if sub == "proj.norm_layer":
            return [10, 96, 120, 160], "patch", "SpikingPEDLayer flattens T,B before ordinary BatchNorm2d"
    if re.fullmatch(r"resblocks.[01].norm[12].norm_layer", tail):
        return [10, 1, 768, 15, 20], "bottleneck", "native bottleneck residual blocks"
    m = re.fullmatch(r"decoders.(\d).norm_layer.norm_layer", tail)
    if m:
        h, w, c = ((30,40,384), (60,80,192), (120,160,96), (240,320,96))[int(m.group(1))]
        return [10, 1, c, h, w], "decoder", "MS_SpikingTransposeDecoderLayer after stride2 deconv"
    raise ValueError("Unmapped external BN: " + name)


def bn_ops(elements, channels):
    # mu=sum(x)/m; v=sum((x-mu)^2)/m; a=gamma*rsqrt(v+eps);
    # b=beta-a*mu; y=a*x+b. Population variance, no running-stat update.
    sizes = dict(mean_sum=elements-channels, mean_scale=channels,
                 variance_center=elements, variance_square=elements,
                 variance_sum=elements-channels, variance_scale=channels,
                 epsilon_add=channels, rsqrt=channels, scale_gamma=channels,
                 offset_mul=channels, offset_sub=channels,
                 output_mul=elements, output_add=elements)
    return dict(real_add_or_sub=4*elements, real_multiply=2*elements+4*channels,
                rsqrt=channels, named_scalar_ops=sizes,
                hypothetical_vector16_issue_excluding_rsqrt=sum(ceil16(v) for k,v in sizes.items() if k != "rsqrt"),
                complete_issue_steps=None,
                mathematical_two_pass_input_reads_elements=3*elements,
                mathematical_output_writes_elements=elements)


def attention_full_issue(stage):
    h, w, c = stage["feature_HWC"]
    heads = stage["heads"]
    rows = 10*((h+8)//9)*9*((w+8)//9)*9
    pairs = rows//2
    def psn(t, f):
        return t*ceil16(t)*ceil16(f)+ceil16(t*f)
    # Same issue geometry as the old packed Engine, but every padded producer
    # needed by current-stat BN remains. The external residual stays unpadded.
    result = (3*psn(2,pairs*c) + 2*rows*ceil16(c)*ceil16(c) + ceil16(rows*c)
              + ceil16(rows*heads)*2 + psn(2,pairs*heads)
              + ceil16(rows*c) + rows*ceil16(c)*ceil16(c) + ceil16(10*h*w*c))
    old = stage["variants"][0]["cross_group_packed_dense_attention_issue_steps"]
    require(all(v["cross_group_packed_dense_attention_issue_steps"] == old for v in stage["variants"]), "shift old cost differs")
    return dict(stage=stage["stage"], depth=stage["depth"], full_padded_rows=rows,
                old_live_QK_rows=stage["variants"][0]["baseline_attention_producers"]["QK_projection_rows_kept"],
                old_projection_rows=10*h*w, new_packed_attention_issue_per_block=result,
                old_packed_attention_issue_per_block=old, restored_issue_per_block=result-old)


def tile_counts(points):
    t, c, f = 10, 384, 1536
    return dict(fc_rows_each=t*points, two_fc_weighted_terms=2*t*points*c*f,
                two_psn_real_macs=t*t*points*(c+f), two_bn_elements=t*points*(c+f),
                two_threshold_elements=t*points*(c+f))


def tile_private_issue(points):
    # Identical old rectangular16x16 dot / vector16 contract; not cycles.
    t, c, f = 10, 384, 1536
    return (2*t*points*ceil16(c)*ceil16(f)
            +t*ceil16(t)*(ceil16(points*c)+ceil16(points*f))
            +ceil16(t*points*c)+ceil16(t*points*f))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path)
    ap.add_argument("--output", type=Path, default=HERE/"hardware_contract_audit.json")
    a = ap.parse_args()
    require(a.output.resolve().parent == HERE, "Receipt output must remain in this directory")
    require(sha(STATIC) == STATIC_SHA, "Static source changed")
    cal = json.loads(CMA.read_text())
    require(cal["status"] == "COMPLETE" and cal["BN_count"] == 78 and cal["geometry"] == "full9", "CMA identity")
    for rel in NATIVE_FILES:
        require(sha(NATIVE/rel) == cal["source_sha256"][rel], "Native source mismatch " + rel)
    static = json.loads(STATIC.read_text())["configurations"][1]
    stages = static["stages"]
    freeze = json.loads(FREEZE.read_text())
    require(sha(RUNNER) == freeze["source_sha256"][RUNNER.name], "Frozen probe source changed")
    require(sha(MERGE) == freeze["frozen_module_sha256"], "Frozen representative module changed")
    require(stages[2]["feature_HWC"] == [30,40,384] and stages[2]["depth"] == 6,
            "Target geometry differs")
    records = []
    for name, buf in cal["buffers"].items():
        if ".mlp." in name:
            continue
        c = len(buf["running_mean"])
        shape, family, basis = native_bn_shape(name,c,stages)
        axis = 2 if len(shape) == 5 else 1
        require(shape[axis] == c, "BN C/shape")
        elements = prod(shape)
        records.append(dict(name=name, channels=c, native_input_shape=shape, family=family,
                            channel_axis=axis, reduction_elements_per_channel=elements//c,
                            elements=elements, shape_evidence="source_derived_not_observed_yet",
                            derivation=basis, operations=bn_ops(elements,c)))
    require(len(records) == 54, "External BN inventory")
    observed = dict(status="PENDING_REAL_BN_HOOKS", actual_shapes_verified=0)
    if a.results:
        raw = json.loads(a.results.read_text())
        require(raw["binding"]["runner_sha256"] == sha(RUNNER), "Probe source binding")
        require(raw["binding"]["module_sha256"] == sha(MERGE) and
                raw["binding"]["checkpoint_sha256"] == freeze["checkpoint_sha256"],
                "Probe module/checkpoint identity differs")
        hooks = raw["frames"][0]["arms"]["coarse_stage02"]["execution"]["batch_norm_inputs"]
        require(set(hooks) == set(cal["buffers"]), "BN hook inventory")
        private_bn_elements = 0
        private_bn_count = 0
        for name, hook in hooks.items():
            if ".mlp." not in name:
                continue
            require(not hook["training"] and hook["track_running_stats"] and hook["running_mean_exists"],
                    "Private MLP BN must stay frozen: " + name)
            require(hook["input_calls"] and all(call["dtype"] == "torch.float32" for call in hook["input_calls"]),
                    "Private MLP BN must execute in FP32")
            private_bn_count += 1
            private_bn_elements += sum(prod(call["shape"]) for call in hook["input_calls"])
        expected_private = sum(10*s["depth"]*s["feature_HWC"][2]*5*
            (((s["feature_HWC"][0]+2)//3)*((s["feature_HWC"][1]+2)//3) if s["stage"] in (0,2)
             else s["feature_HWC"][0]*s["feature_HWC"][1]) for s in stages)
        require(private_bn_count == 24 and private_bn_elements == expected_private,
                "Private frozen BN observed element service differs")
        for item in records:
            hook = hooks[item["name"]]
            require(len(hook["input_calls"]) == 1, "External BN must execute once")
            call = hook["input_calls"][0]
            require(call["shape"] == item["native_input_shape"], "Observed BN shape differs " + item["name"])
            require(call["dtype"] == "torch.float32" and not hook["training"] and
                    not hook["track_running_stats"] and
                    not hook["running_mean_exists"], "External BN mode/dtype differs")
            item.update(shape_evidence="real_first_frame_pre_hook_matches_source_derivation", observed=hook)
        observed = dict(status="54_EXTERNAL_BN_SHAPES_VERIFIED", actual_shapes_verified=54,
                        private_frozen_BN_count=private_bn_count, private_frozen_BN_elements=private_bn_elements,
                        result_status=raw["status"], source_results_sha256=sha(a.results),
                        target_id=raw["frames"][0]["target_id"], source_model_or_quality_reexecuted=False)
    by_family = {}
    for item in records:
        row = by_family.setdefault(item["family"],dict(count=0,elements=0,channels_sum=0))
        row["count"] += 1; row["elements"] += item["elements"]; row["channels_sum"] += item["channels"]
    elements = sum(v["elements"] for v in records)
    channels = sum(v["channels"] for v in records)
    vector_issue = sum(v["operations"]["hypothetical_vector16_issue_excluding_rsqrt"] for v in records)
    repair = [attention_full_issue(s) for s in stages]
    pad_issue = sum(v["depth"]*v["restored_issue_per_block"] for v in repair)
    baseline = static["cross_group_packed_dense_baseline_issue_steps"]
    fine, coarse = tile_counts(81),tile_counts(9)
    delta = {k:fine[k]-coarse[k] for k in fine}
    coarse_points = {s["stage"]:((s["feature_HWC"][0]+2)//3)*((s["feature_HWC"][1]+2)//3) for s in stages}
    stage02_rows = sum(10*coarse_points[s["stage"]]*s["depth"] for s in stages if s["stage"] in (0,2))
    baseline_all12_rows = stage02_rows + sum(10*s["feature_HWC"][0]*s["feature_HWC"][1]*s["depth"] for s in stages if s["stage"] not in (0,2))
    require((stage02_rows,baseline_all12_rows,coarse_points[2]) == (51600,153600,140), "Known actual scope row counts differ")
    e, reps, candidates = 10*384*81, 10*384*9, 12
    scores = dict(
        amplitude_abs_mean=dict(input_elements=e, absolute_values=e, reduction_adds=e-1, scalar_divisions=1),
        pooling_residual_mse=dict(input_elements=e, pool_adds=e-reps, pool_divisions=reps,
                                 residual_subtractions=e, squares=e, final_reduction_adds=e-1,
                                 final_scalar_divisions=1, temporary_representative_elements=reps,
                                 restored_dense_elements_logical=e),
        psn1_activity=dict(input_elements=e, real_temporal_MACs=10*e, thresholds=e,
                           spike_sum_adds=e-1, scalar_divisions=1,
                           checkpoint_temporal_weight_shape=[10,10]),
        psn1_spatial_neighbor_hamming=dict(shared_psn1_with_activity=True,
                    binary_neighbor_comparisons=2*10*384*8*9, reduction_adds=2*(10*384*8*9-1)+1,
                    scalar_divisions=3, bitpacking_or_popcount_mapping_cycles=None),
        offline_branch_MSE=dict(fine_producer=fine,coarse_producer=coarse,
                    differences=e,squares=e,reduction_adds=e-1,scalar_divisions=1,
                    admissible_runtime_predecision_feature=False))
    result = dict(schema="damage_allocation_hardware_contract_v1", status="SOURCE_COUNTS_COMPLETE",
        audit_model_or_GPU_execution=False, source_sha256={str(p):sha(p) for p in (Path(__file__),STATIC,CMA,RUNNER,MERGE,FREEZE)},
        native_source_sha256={rel:sha(NATIVE/rel) for rel in NATIVE_FILES},
        external_BN=dict(inventory=54,by_family=by_family,records=records,observed=observed,
            total_elements=elements,channels_sum=channels,real_add_or_sub=4*elements,
            real_multiply=2*elements+4*channels,rsqrt_elements=channels,
            declared_vector16_issue_excluding_rsqrt=vector_issue,
            variance_contract="population variance; stable centered two-pass mathematical schedule; runtime affine coefficients; no running updates",
            precision_scope="Not claimed bit-exact with native FP kernels; no characterization of rsqrt/reduction/ports"),
        attention_padding_repair=repair,
        conditional_service_denominator=dict(old_folded_issue=baseline,padding_producers_restored_issue=pad_issue,
            declared_BN_vector16_issue_excluding_rsqrt=vector_issue,
            known_arithmetic_issue_subtotal=baseline+pad_issue+vector_issue,
            complete_issue_denominator=None,rsqrt_issue=None,cycle_denominator=None,
            formula="old + restored padded attention producers + specified BN schedule + unknown rsqrt issue; not a calibrated cycle total",
            omitted="real BW/ports, statistics barriers, SRAM lifetime, fusion/reassociation validation, frequency, voxelization and output copies"),
        target=dict(shape=[10,1,30,40,384],hidden=1536,
            candidate_tiles=[[r,c] for r in range(3) for c in range(4)],tails="always coarse; not excluded from GT",
            full_tile_fine=fine,full_tile_coarse=coarse,coarse_to_fine_delta=delta,
            avoided_pool=dict(adds=e-reps,constant_divisions=reps,repeat_dense_write_elements_logical=e),
            residual_elements_unchanged=e,old_private_issue_delta=tile_private_issue(81)-tile_private_issue(9),
            k2_delta={k:2*v for k,v in delta.items()},k2_target_fc_rows_each=10*coarse_points[2]+2*delta["fc_rows_each"],
            k2_stage02_fc_rows_each=stage02_rows+2*delta["fc_rows_each"],
            k2_all12_fc_rows_each=baseline_all12_rows+2*delta["fc_rows_each"]),
        scores_per_candidate=scores,score_candidate_count=candidates,
        upfront_psn1_all12={},
        constraints=["All decisions before private fine/coarse producer issue; complete T10, all C; residual full grid retained.",
            "GT-singleton-gain and branch MSE are offline diagnostics, not deployable free scores.",
            "Pool-based score forces pooled statistics before selection even for selected fine tiles.",
            "Fine PSN spikes cannot be averaged into coarse PSN spikes. Prethreshold linear reuse is a separate ordinary option requiring precision and storage validation.",
            "Same MLP service cost does not imply equal predictor/control or net PE cost."])
    result["upfront_psn1_all12"] = dict(real_temporal_MACs=10*e*candidates,thresholds=e*candidates,
        k2_selected_fine_possible_reuse_MACs=2*10*e,
        extra_real_MACs_after_only_selected_fine_reuse=(candidates-2)*10*e,
        coarse_prethreshold_reuse_included=False)
    if a.results:
        result["source_sha256"][str(a.results)] = sha(a.results)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(status="PASS", BN_observed=observed["status"], BN_elements=elements,
         BN_vector16_excluding_rsqrt=vector_issue,pad_restoration_issue=pad_issue,
         conditional_known_subtotal=baseline+pad_issue+vector_issue,one_tile_delta=delta),sort_keys=True))


if __name__ == "__main__":
    main()
