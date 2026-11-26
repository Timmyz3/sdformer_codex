#!/usr/bin/env python3
"""Read completed R5 service hooks only; no quality ranks, torch, model or GPU.

Preserves the earlier RUNNING-snapshot audit. Writes new complete BN and service
receipts; compiles only a pure TP counting function, never its torch module.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE/"probe_v1/results.json"
SUPPLEMENT = HERE/"supplement_v1/results.json"
RAW_SHA = "bfcf633fb09c05520a252e47f57c4b38060ad50f6fba0f7f1b89e06481f2bf5a"
GEOMETRY = HERE.parent/"structured_merge/audit_structured_merge_service.py"
PINS = {
    GEOMETRY: "abbcf5d93e0062cbb2ca128fde184a51e91cbcd9aa5947cc6efb2880bc704d10",
    HERE/"audit_hardware_contract.py": "ec78f0b5193178b4764fdb71d87f0c9d33b81b81573ecbbb717040eb53e787cb",
    HERE/"hardware_contract_audit.json": "d27b27a9305d23334327ada3cb5789543fe26ee727674b783e9c09a1ddd0bf85",
    HERE/"frozen_tp_score.py": "7d0db322ac276cfaa459cb6ece4b41722b85714bd25feb4f53ddd4529093108a",
    HERE/"supplemental_snn_scores.py": "3d525e3fb21745d6ef3652e3b3906fc16608a2be26911951b86dc681320e0c36",
    SOURCE: RAW_SHA,
    SUPPLEMENT: "9e8bb5e450126f66521232cfe8f6bcc2f981b0a8d7c26b91b7a77878b562f81e",
}
TARGET = "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    for p, pin in PINS.items():
        require(sha(p) == pin, "Frozen source changed: "+str(p))
    spec = importlib.util.spec_from_file_location("_pure_R3_geometry", GEOMETRY)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    ledger = g.extracted_source_ledger()
    constants = g.runner_constants()
    shapes = {f"sttmultires_unet.encoders.swin3d.layers.{s}.swin_blocks.{b}.mlp": list(shape)
              for s, shape in constants["STAGE_SHAPES"].items()
              for b in range(constants["DEPTHS"][s])}
    # Original constants are indexed by stage; accept its exact mapping form.
    if len(shapes) != 12 or not all(isinstance(v,list) and len(v)==5 for v in shapes.values()):
        raise ValueError("Unexpected stage shape schema")
    scoped = {n for n in shapes if ".layers.0." in n or ".layers.2." in n}
    raw = json.loads(SOURCE.read_text())
    require(raw["status"] == "TILE_INTERVENTIONS_DEV8_COMPLETE" and raw["completed_targets"] == 8 and
            len(raw["frames"]) == 8 and raw["full_forward_count"] == 143, "Incomplete R5")
    require(all(raw[k] for k in ("original_model_unchanged","BN_buffers_unchanged","original_modules_restored")), "State restore")
    require(raw["binding"]["module_sha256"] == g.MODULE_SHA and sha(g.MODULE) == g.MODULE_SHA, "Module binding")
    require([f["target_id"] for f in raw["frames"]] == raw["binding"]["dev_ids"], "Frame identity/order")
    candidates = {(r,c) for r in range(3) for c in range(4)}
    singles = {f"fine_r{r}_c{c}":((r,c),) for r,c in candidates}
    k2_names = {"k2_posthoc_singleton_task_gain", "k2_amplitude", "k2_pooling_residual",
                "k2_fixed_uniform", "k2_offline_branch_mse"}
    expected_arms = set(singles)|k2_names|{"coarse_stage02"}
    actual_records = all12_checks = aliases = 0
    class_services = {}
    frame_rows = []
    for frame in raw["frames"]:
        require(set(frame["arms"]) == expected_arms, "Arm inventory")
        require(set(frame["combination_masks"]) == k2_names, "k2 inventory")
        require(frame["ordinary_score_cost"]["diagnostic_only_extra_psn1_temporal_MACs_nominal"] == 37324800 and
                frame["ordinary_score_cost"]["diagnostic_only_extra_psn1_elements"] == 3732480, "Predecision PSN fee")
        for arm, item in frame["arms"].items():
            selection = () if arm == "coarse_stage02" else singles.get(arm)
            if selection is None:
                selection = tuple(tuple(v) for v in frame["combination_masks"][arm])
                require(len(selection) == len(set(selection)) == 2, "k2 cardinality")
            require(set(selection) <= candidates, "Tail selected")
            k = len(selection)
            execution = item["execution"]
            require(set(execution["layers"]) == scoped and set(execution["all12_observed_fc_rows"]) == set(shapes), "Layer inventory")
            for name, shape in shapes.items():
                rows = 10*shape[2]*shape[3]
                if name in scoped:
                    rec = execution["layers"][name]
                    require(rec["mode"] == "actual", "Reference used as deployment")
                    mask = [[(r,c) in selection if name == TARGET else False
                             for c in range((shape[3]+8)//9)] for r in range((shape[2]+8)//9)]
                    require(rec["fine_mask"] == mask, "Routing changed outside selected target")
                    g.audit_record(rec, shape, 4*shape[4], ledger)
                    actual_records += 1
                    rows = rec["executed_fc_rows_per_layer"]
                require(execution["all12_observed_fc_rows"][name] == {"fc1_rows":rows,"fc2_rows":rows}, "Independent FC hook")
                all12_checks += 1
            target = execution["layers"][TARGET]
            require(target["input_sha256"] == frame["predecision_u_sha256"], "Target u differs")
            total_rows = 153600+720*k
            require(execution["summed_fc1_rows"] == execution["summed_fc2_rows"] == total_rows and
                    sum(v["fc1_rows"] for v in execution["all12_observed_fc_rows"].values()) == total_rows, "All12 sum")
            geom = g.geometry(shapes[TARGET],target["fine_mask"])
            service = dict(target=g.counts(shapes[TARGET],1536,geom["fine"]+geom["reps"]),
                           target_added=g.added_counts(shapes[TARGET],geom),
                           target_coarse_representatives=geom["reps"],target_fine_points=geom["fine"],
                           target_coarse_cell_area_histogram=geom["coarse_cell_area_histogram"],
                           stage02_fc_rows_each=51600+720*k,all12_fc_rows_each=total_rows)
            key=str(k)
            if key in class_services:
                require(class_services[key] == service, "Unequal same-k service")
            class_services[key] = service
            alias=item.get("identical_mask_execution_alias")
            if alias is not None:
                require(alias in k2_names and frame["combination_masks"][alias] == frame["combination_masks"][arm] and
                        execution == frame["arms"][alias]["execution"], "Alias has different mask/service")
                aliases += 1
            frame_rows.append(dict(target_id=frame["target_id"],arm=arm,k=k,all12_fc_rows_each=total_rows,
                                   target_fc_rows_each=target["executed_fc_rows_per_layer"],execution_alias=alias))
    dual_records=0
    require(set(raw["first_frame_pairs"]) == {"coarse_stage02","fine_r0_c0"}, "Reference pairs")
    for pair in raw["first_frame_pairs"].values():
        for key in ("actual_execution","dual_execution"):
            for name, record in pair[key]["layers"].items():
                g.audit_record(record,shapes[name],4*shapes[name][4],ledger)
                dual_records += 1
    unique_k2=sum(f["combination_unique_forward_count"] for f in raw["frames"])
    require(unique_k2 == 40-aliases == 37 and raw["full_forward_count"] == 8*13+unique_k2+2, "Forward accounting")

    # Preserve earlier receipt. Re-run its frozen stdlib BN path into a NEW file.
    completed_bn = HERE/"hardware_contract_complete_bn_audit.json"
    subprocess.run([sys.executable,"-B",str(HERE/"audit_hardware_contract.py"),"--results",str(SOURCE),
                    "--output",str(completed_bn)],check=True,stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,universal_newlines=True)
    bn=json.loads(completed_bn.read_text())
    old=json.loads((HERE/"hardware_contract_audit.json").read_text())
    require(bn["external_BN"]["observed"]["result_status"] == raw["status"] and
            bn["external_BN"]["records"] == old["external_BN"]["records"], "Completed first-frame BN differs from snapshot")

    tp_path=HERE/"frozen_tp_score.py"
    node=next(n for n in ast.parse(tp_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="scoring_cost")
    tree=ast.Module(body=[node],type_ignores=[]); env={}
    exec(compile(tree,str(tp_path),"exec"),env)
    tp=env["scoring_cost"]((10,1,384,30,40))
    require(tp["official_dense_conv_mac_including_padded_taps"] == 1592524800,"TP nominal convolution")
    supplementary=json.loads(SUPPLEMENT.read_text())
    require(supplementary["status"] == "SUPPLEMENTAL_SNN_SCORES_DEV8_COMPLETE" and
            supplementary["completed_targets"] == len(supplementary["frames"]) == 8 and
            supplementary["new_full_forward_count"] == 14 and supplementary["TP_full_map_score_calls"] == 8,
            "Incomplete supplement")
    sb=supplementary["binding"]
    require(sb["parent_probe_results_sha256"] == RAW_SHA and sb["runner_sha256"] == sha(HERE/"supplemental_snn_scores.py") and
            sb["tp_adaptation_sha256"] == sha(tp_path) and sb["probe_runner_sha256"] == raw["binding"]["runner_sha256"] and
            sb["module_sha256"] == raw["binding"]["module_sha256"] and
            sb["checkpoint_sha256"] == raw["binding"]["checkpoint_sha256"],"Supplement identity/source binding")
    require(all(supplementary[k] for k in ("original_model_unchanged","BN_buffers_unchanged","original_modules_restored")),"Supplement state restore")
    require(supplementary["runtime"] == raw["runtime"] and supplementary["original_state_sha256"] == raw["original_state_sha256"],"Supplement numerical runtime/state")
    costnode=next(n for n in ast.parse((HERE/"supplemental_snn_scores.py").read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=="logical_score_cost")
    costenv={}; exec(compile(ast.Module(body=[costnode],type_ignores=[]),"logical_score_cost","exec"),costenv)
    expected_tp_cost=costenv["logical_score_cost"](tp)
    supplement_records=supplement_all12=0
    source_counts={}
    for frame,parent in zip(supplementary["frames"],raw["frames"]):
        require(frame["target_id"] == parent["target_id"] and
                frame["parent_predecision_u_sha256"] == parent["predecision_u_sha256"],"Supplement u/frame binding")
        require(frame["PSN_control_score_cost"] == parent["ordinary_score_cost"],"PSN diagnostic fees lost")
        require(set(frame["selected_masks"]) == set(sb["arms"]) == set(frame["scores"]) and
                set(frame["arms"]) == set(sb["arms"])|{"coarse_stage02"},"Supplement arm inventory")
        require(frame["scores"]["k2_tp_official_sdt_tile"]["cost"] == expected_tp_cost,"TP semantic service volumes")
        for arm in sb["arms"]:
            require(frame["scores"][arm]["GT_used_for_selection"] is False,"Supplement GT selection")
            selection=tuple(tuple(v) for v in frame["selected_masks"][arm])
            require(len(selection)==len(set(selection))==2 and set(selection)<=candidates,"Supplement legal k2")
            item=frame["arms"][arm]; execution=item["execution"]
            require(set(execution["layers"])==scoped and set(execution["all12_observed_fc_rows"])==set(shapes),"Supplement layer inventory")
            for name,shape in shapes.items():
                rows=10*shape[2]*shape[3]
                if name in scoped:
                    rec=execution["layers"][name]
                    mask=[[(r,c) in selection if name==TARGET else False for c in range((shape[3]+8)//9)]
                          for r in range((shape[2]+8)//9)]
                    require(rec["mode"]=="actual" and rec["fine_mask"]==mask,"Supplement routes")
                    g.audit_record(rec,shape,4*shape[4],ledger)
                    rows=rec["executed_fc_rows_per_layer"]; supplement_records+=1
                require(execution["all12_observed_fc_rows"][name]=={"fc1_rows":rows,"fc2_rows":rows},"Supplement FC hook")
                supplement_all12+=1
            require(execution["summed_fc1_rows"]==execution["summed_fc2_rows"]==155040 and
                    execution["layers"][TARGET]["input_sha256"]==parent["predecision_u_sha256"],"Supplement actual service/u")
            source=item["execution_source"]; source_counts[source]=source_counts.get(source,0)+1
            if source=="pinned_parent_identical_mask":
                ref=item["source_arm"]
                require(parent["combination_masks"][ref]==frame["selected_masks"][arm] and
                        parent["arms"][ref]["execution"]==execution,"Supplement parent alias")
            elif source=="supplement_identical_mask":
                ref=item["source_arm"]
                require(frame["selected_masks"][ref]==frame["selected_masks"][arm] and
                        frame["arms"][ref]["execution"]==execution,"Supplement local alias")
            else:
                require(source=="new_actual_full_forward" and item["source_arm"] is None,"Unknown supplement source")
    require(source_counts.get("new_actual_full_forward")==14 and sum(source_counts.values())==24 and
            sum(f["new_unique_forward_count"] for f in supplementary["frames"])==14,"Supplement unique forward ledger")
    tp["ordinary_real_arithmetic_factorization_candidate"]={
        "channel_sum_additions":1200*383,"single_channel_box_valid_taps":(3*30-2)*(3*40-2),
        "single_channel_box_additions":(3*30-2)*(3*40-2)-1200,"constant_scale_multiplications":1200,
        "formula":"z[p]=(1/9)*sum_valid_3x3_q(sum_c time_mean[q,c]); neighborhood[p,c_out]=z[p]",
        "FP32_bitwise_or_top2_equivalence_proven_here":False,
        "cosine_eps_and_two_global_softmaxes_retained":True,
        "scope":"Ordinary algebraic optimization candidate; no implemented scorer, timing or hardware result"}
    result=dict(schema="damage_allocation_completed_service_audit_v1",status="PASS",quality_ranks_read=False,
        model_or_GPU_executed=False,raw_status=raw["status"],raw_full_forwards=143,
        source_sha256={str(p):sha(p) for p in (*PINS,Path(__file__),completed_bn)},
        history_preserved=dict(snapshot_receipt="hardware_contract_audit.json",snapshot_status="RUNNING",
                               completed_bn_receipt=completed_bn.name),
        actual_wrapper_records_checked=actual_records,all12_layer_hook_checks=all12_checks,
        reference_side_wrapper_records_checked=dual_records,arm_records=len(frame_rows),
        unique_deployment_forwards=104+unique_k2,reference_forwards=2,k2_actual_unique_forwards=unique_k2,
        k2_identical_mask_aliases=aliases,services_by_number_of_fine_candidates=class_services,
        per_arm_service_records=frame_rows,TP_source_formula_cost=tp,
        supplement=dict(status=supplementary["status"],source_execution_counts=source_counts,
            actual_wrapper_records_checked=supplement_records,all12_layer_hook_checks=supplement_all12,
            score_source_and_full_map_service_verified=True,PSN_predecision_fee_preserved=True,
            all_policies_all12_fc_rows_each=155040),
        limits=["Pool and PSN service are logical counts, not physical cycles/traffic.",
                "Score work is additional to wrapper service; precomputed pooling statistics cannot also be credited as avoided.",
                "No quality ranks, GT oracle values or policy superiority read or derived.",
                "TP literal convolution fee is not the cost of its strongest ordinary algebraic implementation."])
    out=HERE/"hardware_contract_complete_services.json"
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(status="PASS",actual_wrapper_records=actual_records,all12_checks=all12_checks,
                         arm_records=len(frame_rows),unique_deployment_forwards=141,dual_forwards=2,
                         aliases=aliases,output=out.name)))


if __name__ == "__main__":
    main()
