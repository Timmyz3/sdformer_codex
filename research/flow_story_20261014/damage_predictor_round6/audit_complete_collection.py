#!/usr/bin/env python3
"""Independent ROUND6 execution-only audit: stdlib, no quality/GT values.

Reads execution/membership/source fields, hashes NPZ files as opaque bytes, and
checks only NPY metadata plus the raw predecision-u payload hash. Never loads
prediction or GT arrays, imports torch, fits models, or runs inference.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zipfile
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
R5=HERE.parent/"damage_allocation_probe"
GEOMETRY=HERE.parent/"structured_merge/audit_structured_merge_service.py"
PINS={
    HERE/"collect_v1/results.json":"ffcaf053717b886cf535f542f394ce1cf939fdfaee62c4da878146f3a3edfad5",
    HERE/"native_check_v1/results.json":"48af99c1984d089ef0618789dc38898da0dbf4314320cce5fc18cce5114c4540",
    GEOMETRY:"abbcf5d93e0062cbb2ca128fde184a51e91cbcd9aa5947cc6efb2880bc704d10",
    HERE/"predictor_cost_contract.json":"b48f996804501200d4b7f108231632cf342caf469065cf44b33ce70374cae78a",
    HERE/"predictor_source_binding.json":"f1b46b120770242a8389a97b649f877093cb2df20be01d233068e856ea7c1090",
    R5/"hardware_contract_complete_bn_audit.json":"9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567",
    R5/"hardware_contract_complete_services.json":"0375388e5b6aa777842819d228f239f3283772d5d7a0bc8a7582bc396c212db1",
}
TARGET="sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp"


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()


def product(xs):
    r=1
    for x in xs:r*=x
    return r


def read_header(stream):
    require(stream.read(6)==b"\x93NUMPY","NPY magic")
    version=stream.read(2)
    require(version in (b"\x01\x00",b"\x02\x00",b"\x03\x00"),"NPY version")
    length=struct.unpack("<H" if version==b"\x01\x00" else "<I",stream.read(2 if version==b"\x01\x00" else 4))[0]
    require(length<65536,"NPY header bound")
    return ast.literal_eval(stream.read(length).decode("utf-8"))


def archive_check(root,frame,key,records,raw_u=False):
    path=root/frame[key]
    require(path.resolve().parent==root.resolve(),"Archive path escapes run")
    digest=sha(path);require(digest==frame[key+"_sha256"],"Archive checksum "+str(path))
    record=dict(path=str(path),sha256=digest,bytes=path.stat().st_size)
    if raw_u:
        with zipfile.ZipFile(path) as archive:
            for name,shape in (("predecision_u.npy",(10,1,30,40,384)),
                               ("no_GT_cell_descriptor.npy",(1,10,14,384))):
                with archive.open(name) as stream:
                    header=read_header(stream)
                    require(header["shape"]==shape and header["descr"]=="<f4" and not header["fortran_order"],"No-GT input header")
                    if name=="predecision_u.npy":
                        hashed=hashlib.sha256();nbytes=0
                        for block in iter(lambda:stream.read(1024*1024),b""):
                            hashed.update(block);nbytes+=len(block)
                        require(nbytes==4*product(shape) and hashed.hexdigest()==frame["predecision_u_sha256"],"Captured raw-u hash")
                record[name]=dict(shape=list(shape),dtype="float32",payload_hashed=name=="predecision_u.npy")
    records.append(record)


def main():
    for path,pin in PINS.items():require(sha(path)==pin,"Pinned input changed: "+str(path))
    for freeze_name in ("collector_freeze_v1.json","native_freeze_v1.json"):
        freeze=json.loads((HERE/freeze_name).read_text())
        for name,pin in freeze["source_sha256"].items():require(sha(HERE/name)==pin,"Runner freeze differs "+name)
    spec=importlib.util.spec_from_file_location("_r6_pure_service_geometry",GEOMETRY)
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    require(sha(g.MODULE)==g.MODULE_SHA,"Native wrapper source differs")
    constants=g.runner_constants();ledger=g.extracted_source_ledger()
    shapes={f"sttmultires_unet.encoders.swin3d.layers.{s}.swin_blocks.{b}.mlp":list(shape)
            for s,shape in constants["STAGE_SHAPES"].items() for b in range(constants["DEPTHS"][s])}
    scoped={name for name in shapes if ".layers.0." in name or ".layers.2." in name}
    main_raw=json.loads((HERE/"collect_v1/results.json").read_text())
    native=json.loads((HERE/"native_check_v1/results.json").read_text())
    manifest=json.loads((HERE/"selection_manifest_v1.json").read_text())
    groups=manifest["groups"]
    require(len(groups["fit"])==20 and len(groups["check"])==8,"Manifest sizes")
    require(len(set(groups["fit"]+groups["check"]))==28,"Manifest duplicate or overlap")
    require(main_raw["status"]=="ROUND6_SINGLETON_COLLECTION_COMPLETE" and main_raw["completed_targets"]==28 and
            len(main_raw["frames"])==28 and main_raw["full_forward_count"]==28*13+2==366,"Incomplete collection")
    require(native["status"]=="ROUND6_NATIVE_CHECK8_COMPLETE" and native["completed_targets"]==8 and
            len(native["frames"])==8 and native["full_forward_count"]==8,"Incomplete native anchor")
    for raw in (main_raw,native):
        require(raw["inference_executed"] and not raw["training_performed"] and
                all(raw[k] for k in ("original_model_unchanged","BN_buffers_unchanged","original_modules_restored")),"State/BN restore")
        require(raw["binding"]["checkpoint_sha256"]=="d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30","Checkpoint")
    require(main_raw["binding"]["runner_sha256"]==sha(HERE/"collect_singletons.py") and
            native["binding"]["runner_sha256"]==sha(HERE/"collect_native_check.py"),"Runner source")
    require(main_raw["binding"]["inherited_numerical_and_source_contract"]["module_sha256"]==g.MODULE_SHA and
            main_raw["binding"]["frozen_R5_probe_sha256"]==sha(R5/"run_tile_interventions.py") and
            native["binding"]["collector_source_sha256"]==sha(HERE/"collect_singletons.py"),"Inherited module/probe binding")
    require(main_raw["binding"]["selection_manifest_sha256"]==sha(HERE/"selection_manifest_v1.json"),"Membership binding")
    require(main_raw["runtime"]==native["runtime"] and main_raw["original_state_sha256"]==native["original_state_sha256"],"Native numerical runtime/state")
    require(main_raw["BN_scope"]["BN_signature_before_inference"]==native["BN_scope"]["BN_signature_before_inference"],"Native BN signature")
    signatures=main_raw["BN_scope"]["BN_signature_before_inference"]
    frozen={n for n in signatures if ".mlp." in n};external=set(signatures)-frozen
    require(len(frozen)==24 and len(external)==54,"BN partition")
    for raw in (main_raw,native):
        scope=raw["BN_scope"]
        require(scope["scope"]=="MLP_CMA176" and scope["BN_count"]==78 and scope["MLP_BN_count"]==24 and
                set(scope["frozen_BN_names"])==frozen and set(scope["current_stat_BN_names"])==external and
                scope["non_bn_state_unchanged"],"BN scope")
    for name,sig in signatures.items():
        require(not sig["training"] and sig["tracking"]==(name in frozen),"BN mode")
        require((sig["running_mean"] is not None)==(name in frozen) and
                (sig["running_var"] is not None)==(name in frozen),"BN buffers")
    candidates={(r,c) for r in range(3) for c in range(4)}
    arms={"coarse_stage02":()} ; arms.update({f"fine_r{r}_c{c}":((r,c),) for r,c in candidates})
    wrapper_records=all12_checks=0;archives=[];frame_records=[];bn_snapshots=[]
    prior_services=json.loads((R5/"hardware_contract_complete_services.json").read_text())["services_by_number_of_fine_candidates"]
    for frame in main_raw["frames"]:
        require(frame["target_id"] in groups[frame["group"]],"Frame group identity")
        require(set(frame["arms"])==set(arms),"Singleton inventory")
        for arm,selection in arms.items():
            execution=frame["arms"][arm]["execution"]
            require(set(execution["layers"])==scoped and set(execution["all12_observed_fc_rows"])==set(shapes),"Layer inventory")
            for name,shape in shapes.items():
                rows=10*shape[2]*shape[3]
                if name in scoped:
                    rec=execution["layers"][name]
                    expected=[[(r,c) in selection if name==TARGET else False for c in range((shape[3]+8)//9)]
                              for r in range((shape[2]+8)//9)]
                    require(rec["mode"]=="actual" and rec["fine_mask"]==expected,"Fine route/tail")
                    g.audit_record(rec,shape,4*shape[4],ledger);wrapper_records+=1
                    rows=rec["executed_fc_rows_per_layer"]
                require(execution["all12_observed_fc_rows"][name]==dict(fc1_rows=rows,fc2_rows=rows),"Independent FC hook")
                all12_checks+=1
            k=len(selection);target=execution["layers"][TARGET]
            require(target["input_sha256"]==frame["predecision_u_sha256"],"Same-u")
            expected_rows=153600+720*k
            require(execution["summed_fc1_rows"]==execution["summed_fc2_rows"]==expected_rows and
                    sum(r["fc1_rows"] for r in execution["all12_observed_fc_rows"].values())==expected_rows,"FC totals")
            geo=g.geometry(shapes[TARGET],target["fine_mask"])
            require(g.counts(shapes[TARGET],1536,geo["fine"]+geo["reps"])==prior_services[str(k)]["target"] and
                    g.added_counts(shapes[TARGET],geo)==prior_services[str(k)]["target_added"],"Target arithmetic/pool vs source geometry")
            if execution["batch_norm_inputs"]:bn_snapshots.append(execution["batch_norm_inputs"])
        require(frame["ordinary_score_cost"]["diagnostic_only_extra_psn1_temporal_MACs_nominal"]==37324800 and
                frame["ordinary_score_cost"]["diagnostic_only_extra_psn1_calls"]==12,"Extra diagnostic PSN")
        for key in ("predecision_input_file","offline_branch_file","prediction_file"):
            archive_check(HERE/"collect_v1",frame,key,archives,raw_u=key=="predecision_input_file")
        frame_records.append(dict(target_id=frame["target_id"],group=frame["group"],singleton_count=12,
                                  baseline_rows_each=153600,singleton_rows_each=154320,same_u_all13=True))
    require([f["target_id"] for f in main_raw["frames"]]==main_raw["binding"]["target_ids"] and
            set(f["target_id"] for f in main_raw["frames"])==set(groups["fit"]+groups["check"]),"Complete member order")
    require(len(bn_snapshots)==1,"Unexpected BN sampling count")
    hooks=bn_snapshots[0];require(set(hooks)==set(signatures),"78 BN pre-hooks")
    prior_bn=json.loads((R5/"hardware_contract_complete_bn_audit.json").read_text())
    for rec in prior_bn["external_BN"]["records"]:
        name=rec["name"];hook=hooks[name]
        require(not hook["training"] and not hook["track_running_stats"] and not hook["running_mean_exists"] and
                len(hook["input_calls"])==1 and hook["input_calls"][0]["shape"]==rec["native_input_shape"] and
                hook["input_calls"][0]["dtype"]=="torch.float32","External BN actual shape/mode")
    private_elements=0
    for name in frozen:
        hook=hooks[name]
        require(not hook["training"] and hook["track_running_stats"] and hook["running_mean_exists"] and
                all(c["dtype"]=="torch.float32" for c in hook["input_calls"]),"Private BN actual mode")
        private_elements+=sum(product(c["shape"]) for c in hook["input_calls"])
    require(private_elements==152064000,"Private baseline BN elements")
    dual_records=0
    require(set(main_raw["first_frame_pairs"])=={"coarse_stage02","fine_r0_c0"},"Two first-frame pairs")
    for pair in main_raw["first_frame_pairs"].values():
        for key in ("actual_execution","dual_execution"):
            require(set(pair[key]["layers"])==scoped,"Reference layer inventory")
            for name,rec in pair[key]["layers"].items():g.audit_record(rec,shapes[name],4*shapes[name][4],ledger);dual_records+=1
    native_checks=0
    require([f["target_id"] for f in native["frames"]]==native["binding"]["target_ids"]==groups["check"],"Native fixed check8")
    for frame in native["frames"]:
        require(frame["group"]=="check" and set(frame["arms"])=={"native_dense"},"Native arm")
        execution=frame["arms"]["native_dense"]["execution"]
        require(execution["layers"]=={} and set(execution["all12_observed_fc_rows"])==set(shapes),"Native has no wrappers")
        for name,shape in shapes.items():
            rows=10*shape[2]*shape[3]
            require(execution["all12_observed_fc_rows"][name]==dict(fc1_rows=rows,fc2_rows=rows),"Native FC full support")
            native_checks+=1
        require(execution["summed_fc1_rows"]==execution["summed_fc2_rows"]==558000,"Native total rows")
        archive_check(HERE/"native_check_v1",frame,"prediction_file",archives)
    result=dict(schema="flow_round6_complete_execution_audit_v1",status="PASS",
        source_sha256={str(p):sha(p) for p in (*PINS,HERE/"collector_freeze_v1.json",HERE/"native_freeze_v1.json",
                HERE/"collect_singletons.py",HERE/"collect_native_check.py",HERE/"selection_manifest_v1.json",Path(__file__))},
        quality_fields_or_GT_values_inspected=False,quality_or_cycle_conclusion=False,model_or_GPU_executed=False,
        collection=dict(status=main_raw["status"],frames=28,actual_arm_forwards=364,dual_forwards=2,total_forwards=366,
                        wrapper_records_checked=wrapper_records,all12_layer_hook_checks=all12_checks,
                        actual_dual_side_wrapper_records_checked=dual_records,singleton_actions_checked=336,
                        per_frame=frame_records,services_by_k={k:prior_services[k] for k in ("0","1")}),
        BN=dict(private_frozen=24,external_current_stat=54,actual_full_BN_snapshot_frames=1,
                external_elements_per_sampled_full_forward=prior_bn["external_BN"]["total_elements"],
                private_baseline_BN_elements=private_elements,native_and_collection_signatures_exact=True),
        native=dict(status=native["status"],frames=8,full_forwards=8,all12_layer_hook_checks=native_checks,
                    each_FC_total_rows=558000,wrappers_used=False,BN_scope="MLP_CMA176",
                    native_tiled_numerical_equivalence_claim=False),
        archive_integrity=dict(files=len(archives),total_bytes=sum(x["bytes"] for x in archives),records=archives,
                              raw_u_payload_sha_verified_frames=28,descriptor_NPY_header_verified_frames=28,
                              prediction_or_GT_arrays_loaded=False),
        boundary="Completed GPU collection services are audited read-only; candidate selector cost remains separately bound. Fixed k1 supplies equal producer work, not predictor net cycle savings. Original count/source snapshots unchanged.")
    out=HERE/"complete_collection_service_audit.json"
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(status="PASS",wrapper_records=wrapper_records,all12_checks=all12_checks,
                         dual_side_records=dual_records,native_checks=native_checks,archive_files=len(archives),output=out.name)))


if __name__=="__main__":main()
