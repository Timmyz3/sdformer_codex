#!/usr/bin/env python3
"""R10 block2 execution-only audit. No model, quality analysis, or GPU.

Default preflight validates source/membership contracts only. A complete-result
audit needs explicit raw SHA and a fresh output path; no running file accepted.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parents[1] / "flow_story_20261014"
R6 = OLD / "damage_predictor_round6"
R5 = OLD / "damage_allocation_probe"
GEOMETRY = OLD / "structured_merge/audit_structured_merge_service.py"
HELPERS = R6 / "audit_complete_collection.py"
TARGET = "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp"
PREFIX = tuple("sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp".format(s,b)
               for s,b in ((0,0),(0,1),(2,0),(2,1),(2,2)))
PINS = {
    HERE / "source_pins.json": "ad9065cec7ac1dca0b24e2bb8e3d518829a737d8e1ef063536c48be9de1916a5",
    HERE / "membership_v1.json": "28c6dc4ffb2197301c8d8a763e65fa45b6cf753768b15d45750de49e3228a9df",
    HERE / "selection_manifest_v1.json": "a22097a54a8f272fe920bb61ac59006bf891745808f4b4360ae1cffe3ff00c0d",
    HERE / "collector_cpu_results.json": "b3289a1bff5a781b8071fb2d6f80dce964c3fed2cfc8c8f7fb337e1b14fe1e50",
    GEOMETRY: "abbcf5d93e0062cbb2ca128fde184a51e91cbcd9aa5947cc6efb2880bc704d10",
    HELPERS: "bf37de4584054a86cdc6f7480596b9d75948c0b665d494b5947be396cb3986a0",
    R6 / "collect_v1/results.json": "ffcaf053717b886cf535f542f394ce1cf939fdfaee62c4da878146f3a3edfad5",
    R5 / "hardware_contract_complete_bn_audit.json": "9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567",
    R5 / "hardware_contract_complete_services.json": "0375388e5b6aa777842819d228f239f3283772d5d7a0bc8a7582bc396c212db1",
}


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def preflight():
    for path,pin in PINS.items():require(sha(path)==pin,"Pinned source changed: "+str(path))
    sources=json.loads((HERE/"source_pins.json").read_text())["sources"]
    for pin in sources.values():require(sha(ROOT/pin["path"])==pin["sha256"],"Inherited source changed")
    cpu=json.loads((HERE/"collector_cpu_results.json").read_text())
    require(cpu["passed"] and cpu["tests_run"]==6 and not cpu["GPU_initialized"] and
            not cpu["full_model_run"] and not cpu["errors"] and not cpu["failures"],"CPU contract")
    for path,pin in cpu["source_sha256"].items():require(sha(path)==pin,"CPU source pin")
    require(cpu["membership_sha256"]==sha(HERE/"membership_v1.json") and
            cpu["inherited_source_pins_sha256"]==sha(HERE/"source_pins.json"),"CPU membership/source binding")
    member=json.loads((HERE/"membership_v1.json").read_text())
    manifest=json.loads((HERE/"selection_manifest_v1.json").read_text())
    groups=member["groups"]
    require(manifest["groups"]==groups and manifest["compatibility_only"] and
            manifest["r10_membership_sha256"]==sha(HERE/"membership_v1.json"),"Compatibility membership")
    require(len(groups["fit"])==20 and len(groups["check"])==8 and len(set(groups["fit"]+groups["check"]))==28,"Membership size/overlap")
    g=load(GEOMETRY,"_r10_pure_geometry");h=load(HELPERS,"_r10_pure_archive_helpers")
    require(sha(g.MODULE)==g.MODULE_SHA,"Wrapper source")
    constants=g.runner_constants()
    shapes={"sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp".format(s,b):list(shape)
            for s,shape in constants["STAGE_SHAPES"].items() for b in range(constants["DEPTHS"][s])}
    require(shapes[TARGET]==[10,1,30,40,384] and all(n in shapes for n in PREFIX),"Target/prefix geometry")
    return g,h,g.extracted_source_ledger(),shapes,groups,sources


def audit(result_path,result_sha,output):
    require(not output.exists(),"Fresh output path required")
    require(len(result_sha)==64 and sha(result_path)==result_sha,"Explicit complete raw SHA")
    g,h,ledger,shapes,groups,sources=preflight()
    raw=json.loads(result_path.read_text())
    require(raw["schema"]=="flow_round10_block2_singletons_v1" and
            raw["status"]=="ROUND10_BLOCK2_SINGLETON_COLLECTION_COMPLETE" and
            raw["completed_targets"]==len(raw["frames"])==28 and raw["full_forward_count"]==366,"Complete R10 required")
    require(raw["inference_executed"] and not raw["training_performed"] and all(raw[k] for k in
            ("original_model_unchanged","BN_buffers_unchanged","original_modules_restored")),"Native restoration")
    binding=raw["binding"]
    require(binding["target"]==TARGET and binding["prefix_layer_names"]==list(PREFIX) and
            binding["old_target"].endswith("layers.2.swin_blocks.0.mlp") and binding["full_forward_budget"]==366,"Target/budget")
    for field,path in (("runner_sha256",HERE/"collect_block2.py"),("runner_checks_sha256",HERE/"collector_cpu_results.json"),
                       ("inherited_source_pins_sha256",HERE/"source_pins.json"),("r10_membership_sha256",HERE/"membership_v1.json"),
                       ("selection_manifest_sha256",HERE/"selection_manifest_v1.json"),
                       ("inherited_collector_source_sha256",R6/"collect_singletons.py")):
        require(binding[field]==sha(path),"Runtime binding "+field)
    require(binding["checkpoint_sha256"]=="d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30","Checkpoint")
    old=json.loads((R6/"collect_v1/results.json").read_text())
    require(raw["runtime"]==old["runtime"] and raw["original_state_sha256"]==old["original_state_sha256"],"Inherited runtime/state")
    sig=raw["BN_scope"]["BN_signature_before_inference"]
    require(sig==old["BN_scope"]["BN_signature_before_inference"],"BN signature")
    frozen={n for n in sig if ".mlp." in n};external=set(sig)-frozen;bn=raw["BN_scope"]
    require(len(frozen)==24 and len(external)==54 and bn["scope"]=="MLP_CMA176" and bn["BN_count"]==78 and
            bn["MLP_BN_count"]==24 and set(bn["frozen_BN_names"])==frozen and set(bn["current_stat_BN_names"])==external
            and bn["non_bn_state_unchanged"],"BN scope")
    for name,s in sig.items():
        require(not s["training"] and s["tracking"]==(name in frozen) and
                (s["running_mean"] is not None)==(name in frozen) and (s["running_var"] is not None)==(name in frozen),"BN mode")
    scope={n for n in shapes if ".layers.0." in n or ".layers.2." in n}
    counts=dict(actual_forwards=0,dual_forwards=0,wrapper_records=0,all12_FC_records=0,prefix_input_hashes=0,prefix_output_hashes=0,FC_rows_each_total=0)
    snapshots=[]

    def execution_check(execution,selected,mode,baseline):
        require(set(execution["layers"])==scope and set(execution["all12_observed_fc_rows"])==set(shapes),"Layer inventory")
        require(execution["r10_target_layer"]==TARGET and execution["r10_prefix_layer_names_checked"]==list(PREFIX) and
                execution["r10_current_CC_prefix_input_exact"] and execution["r10_earlier_prefix_output_exact"],"New prefix guard")
        total=0
        for name,shape in shapes.items():
            rows=10*shape[2]*shape[3]
            if name in scope:
                rec=execution["layers"][name]
                expected=[[(r,c) in selected if name==TARGET else False for c in range((shape[3]+8)//9)]
                          for r in range((shape[2]+8)//9)]
                require(rec["mode"]==mode and rec["fine_mask"]==expected,"Block2 mask/mode/tails")
                g.audit_record(rec,shape,4*shape[4],ledger);rows=rec["executed_fc_rows_per_layer"]
                counts["wrapper_records"]+=1
            require(execution["all12_observed_fc_rows"][name]==dict(fc1_rows=rows,fc2_rows=rows),"All12 FC hooks")
            total+=rows;counts["all12_FC_records"]+=1
        expected_total=609600 if mode=="dual_reference" else 153600+720*len(selected)
        require(execution["summed_fc1_rows"]==execution["summed_fc2_rows"]==total==expected_total,"Real total row budget")
        for name in PREFIX:
            require(execution["layers"][name]["input_sha256"]==baseline["layers"][name]["input_sha256"],"Same current-CC prefix input")
            counts["prefix_input_hashes"]+=1
        for name in PREFIX[:-1]:
            require(execution["layers"][name]["output_sha256"]==baseline["layers"][name]["output_sha256"],"Same earlier-prefix output")
            counts["prefix_output_hashes"]+=1
        counts["actual_forwards" if mode=="actual" else "dual_forwards"]+=1
        counts["FC_rows_each_total"]+=total
        if execution.get("batch_norm_inputs"):snapshots.append(execution["batch_norm_inputs"])

    arms={"coarse_stage02":()}
    arms.update({"fine_r{}_c{}".format(r,c):((r,c),) for r in range(3) for c in range(4)})
    archives=[];prior_services=json.loads((R5/"hardware_contract_complete_services.json").read_text())["services_by_number_of_fine_candidates"]
    for frame in raw["frames"]:
        require(frame["target_id"] in groups[frame["group"]] and set(frame["arms"])==set(arms),"Member/arm inventory")
        baseline=frame["arms"]["coarse_stage02"]["execution"]
        for arm,selected in arms.items():
            execution=frame["arms"][arm]["execution"]
            execution_check(execution,selected,"actual",baseline)
            target=execution["layers"][TARGET]
            require(target["input_sha256"]==frame["predecision_u_sha256"],"Fresh block2 captured u")
            geo=g.geometry(shapes[TARGET],target["fine_mask"])
            require(g.counts(shapes[TARGET],1536,geo["fine"]+geo["reps"])==prior_services[str(len(selected))]["target"] and
                    g.added_counts(shapes[TARGET],geo)==prior_services[str(len(selected))]["target_added"],"Same target geometry arithmetic")
        require(frame["ordinary_score_cost"]["diagnostic_only_extra_psn1_calls"]==12 and
                frame["ordinary_score_cost"]["diagnostic_only_extra_psn1_temporal_MACs_nominal"]==37324800,"Diagnostic PSN cost")
        for key in ("predecision_input_file","offline_branch_file","prediction_file"):
            h.archive_check(result_path.parent,frame,key,archives,raw_u=key=="predecision_input_file")
    require([f["target_id"] for f in raw["frames"]]==binding["target_ids"] and
            set(binding["target_ids"])==set(groups["fit"]+groups["check"]),"Completed members")
    require(set(raw["first_frame_pairs"])=={"coarse_stage02","fine_r0_c0"},"Two explicit dual references")
    first=raw["frames"][0];baseline=first["arms"]["coarse_stage02"]["execution"]
    for arm,pair in raw["first_frame_pairs"].items():
        require(pair["actual_execution"]==first["arms"][arm]["execution"],"Reused actual side binding")
        execution_check(pair["dual_execution"],arms[arm],"dual_reference",baseline)
    expected=dict(actual_forwards=364,dual_forwards=2,wrapper_records=2928,all12_FC_records=4392,
                  prefix_input_hashes=1830,prefix_output_hashes=1464,FC_rows_each_total=57371520)
    require(counts==expected,"Complete service inventory")
    require(len(snapshots)==1,"Only first baseline captures full BN shapes")
    hooks=snapshots[0];require(set(hooks)==set(sig),"78 BN prehooks")
    prior_bn=json.loads((R5/"hardware_contract_complete_bn_audit.json").read_text())
    for record in prior_bn["external_BN"]["records"]:
        hook=hooks[record["name"]]
        require(not hook["training"] and not hook["track_running_stats"] and not hook["running_mean_exists"] and
                len(hook["input_calls"])==1 and hook["input_calls"][0]["shape"]==record["native_input_shape"] and
                hook["input_calls"][0]["stride"]==record["observed"]["input_calls"][0]["stride"] and
                hook["input_calls"][0]["dtype"]=="torch.float32","External BN actual shape/mode")
    private_elements=0
    for name in frozen:
        hook=hooks[name]
        require(not hook["training"] and hook["track_running_stats"] and hook["running_mean_exists"] and
                all(call["dtype"]=="torch.float32" for call in hook["input_calls"]),"Private BN mode")
        private_elements+=sum(h.product(call["shape"]) for call in hook["input_calls"])
    require(private_elements==152064000,"Private CC BN elements")
    receipt=dict(schema="flow_round10_block2_execution_audit_v1",status="PASS",counts=counts,
        source_sha256={str(p):sha(p) for p in tuple(PINS)+(result_path,HERE/"collect_block2.py",Path(__file__))},
        target=TARGET,full_forward_count=366,member_groups=groups,prefix=list(PREFIX),
        singleton_actions=336,baseline_FC_rows_each=153600,singleton_FC_rows_each=154320,dual_FC_rows_each=609600,
        per_target_nominal_services={k:prior_services[k] for k in ("0","1")},
        BN=dict(private_frozen=24,external_current_stat=54,full_input_snapshot_count=1,
                private_CC_elements=private_elements,external_elements=976584960,state_and_buffers_unchanged=True),
        archives=dict(files=len(archives),bytes=sum(r["bytes"] for r in archives),records=archives,
                      raw_u_payload_SHA_checked=28,descriptor_headers_checked=28,prediction_or_GT_values_loaded=False),
        scope=dict(quality_fields_analyzed=False,model_or_GPU_executed_by_auditor=False,chip_or_full_network_cycle_claim=False,
                   new_BN_shape_snapshot_every_frame=False,learned_policy_hardware_gain_claim=False),
        boundary="Only target changed to block2; completed same-u/k1 producer services are audited, not predictor quality, scoring cost or net cycle gain. Exact task/quality comparisons remain separate.")
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return dict(status="PASS",counts=counts,archives=len(archives),output=str(output))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path)
    parser.add_argument("--results-sha256")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    if args.results is None:
        require(args.results_sha256 is None and args.output is None,"Preflight takes no result/output arguments")
        _,_,_,shapes,groups,_=preflight()
        print(json.dumps(dict(status="PREFLIGHT_EXPECTATIONS_ONLY",actual_results_audited=False,target=TARGET,
            target_shape=shapes[TARGET],members={k:len(v) for k,v in groups.items()},
            expected_forwards=366,expected_actual_FC_rows=[153600,154320],expected_dual_FC_rows=609600)))
    else:
        require(args.results_sha256 and args.output,"Result SHA and fresh output required")
        print(json.dumps(audit(args.results,args.results_sha256,args.output)))


if __name__=="__main__":main()
