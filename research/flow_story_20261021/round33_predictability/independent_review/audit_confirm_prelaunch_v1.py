"""Verify both actual fitted branches and their joint pre-confirmation freeze."""
import datetime
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROUND=HERE.parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def digest_arrays(arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.asarray(a);h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main():
    output=HERE/"CONFIRM_PRELAUNCH_REVIEW_V1.json";assert not output.exists()
    packaudit=read(HERE/"FIT_TUNE_PACK_INDEPENDENT_AUDIT_V1.json");assert packaudit["status"]=="PASS"
    membershippath=ROUND/"data/MEMBERSHIP_V1.json";membership=read(membershippath)
    membershipsha=sha(membershippath);assert membershipsha=="d860612a3ae3e4620fa2a9a616df9f8e93774efadcf7dc564881e27b7e7da8cc"
    ids={role:[r["target_id"] for r in membership["records"] if r["role"]==role] for role in ["fit","tune","confirm"]}
    seq={r["target_id"]:r["sequence"] for r in membership["records"]}
    methods=["direct_raw","signed_g1","signed_g4","direct_aug_g1","direct_aug_g4","direct_residual_g1","direct_residual_g4"]
    executionpath=ROUND/"fitted/CPU_FIT_EXECUTION_RECEIPT_V1.json";execution=read(executionpath)
    assert execution["alpha"]==.1 and execution["fit_identities_per_branch"]==64
    assert not execution["changed_settings_after_loso"] and not execution["tune_quality_decoded"] and not execution["confirmation_decoded"] and not execution["GPU_used"]
    jobs={j["branch"]:j for j in execution["jobs"]};assert len(jobs)==2
    branches=[];checkedfolds=0;normal_residuals={}
    def folds(trace,available):
        nonlocal checkedfolds
        groups=sorted({seq[i] for i in available})
        assert [r["held_group"] for r in trace]==groups
        for r in trace:
            held=r["held_group"];expectedfit=[i for i in available if seq[i]!=held];expectedheld=[i for i in available if seq[i]==held]
            assert r["fit_ids"]==expectedfit and r["held_ids"]==expectedheld
            assert not set(r["fit_ids"])&set(r["held_ids"])
            assert not (set(r["fit_ids"]+r["held_ids"])&set(ids["tune"]+ids["confirm"]))
            if "fit_groups" in r:assert r["fit_groups"]==sorted({seq[i] for i in expectedfit})
            checkedfolds+=1
            if "auxiliary_crossfit" in r:
                assert set(r["auxiliary_crossfit"])=={"g1","g4"}
                for inner in r["auxiliary_crossfit"].values():folds(inner,expectedfit)
    for branch,width,src,packname,schema in [
            ("voxel128_v1",128,"algorithm","fit_tune_128_v1","flow_r33_voxel_small_predictors_v1"),
            ("directional160_v1",160,"directional_branch","fit_tune_160_v1","flow_r33_directional_small_predictors_v1")]:
        directory=ROUND/"fitted"/branch;source=ROUND/src;packdir=ROUND/"packs"/packname
        modelpath=directory/"MODEL_V1.json";receiptpath=directory/"FIT_RECEIPT_V1.json";headpath=directory/"HEADS_FREEZE_V1.json"
        m,r,h=read(modelpath),read(receiptpath),read(headpath);job=jobs[branch]
        assert job["exit_code"]==0 and job["environment_overrides"]["CUDA_VISIBLE_DEVICES"]==""
        for key in ["stdout","stderr"]:assert sha(job[key+"_path"])==job[key+"_sha256"]
        assert job["source_freeze_sha256"]==sha(source/"SOURCE_FREEZE_V1.json")
        assert job["pack_freeze_sha256"]==sha(packdir/"PACK_FREEZE_V1.json")
        assert job["fit_pack_sha256"]==sha(packdir/"fit.npz")
        assert job["model_sha256"]==sha(modelpath) and job["fit_receipt_sha256"]==sha(receiptpath)
        command=job["command"]
        assert "fit" in command and "--include-loso" in command and "-O" not in command
        assert Path(command[command.index("--fit-pack")+1])==packdir/"fit.npz"
        assert Path(command[2])==source/"run_predictors.py"
        assert m["schema"]==schema and m["alpha"]==r["alpha"]==h["alpha"]==.1
        assert m["methods"]==r["methods"]==h["methods"]==methods
        meta=m["metadata"]
        assert meta["input_width"]==width and meta["fit_ids"]==r["fit_ids"]==ids["fit"]
        assert meta["fit_groups"]==r["fit_groups"]==sorted({seq[i] for i in ids["fit"]})
        assert meta["membership_sha256"]==r["membership_sha256"]==h["membership_sha256"]==membershipsha
        assert not meta["alpha_search"] and meta["no_upgrade_threshold"]==0 and meta["higher_is_better"] is False
        assert r["identity_count"]==64 and r["auxiliary_training"]=="inner_leave_one_sequence_out"
        assert not r["tune_or_confirm_read"] and not r["GPU_used"] and r["real_flow_network_forwards"]==0
        for name in ["predictor","runner"]:
            filename="predictors.py" if name=="predictor" else "run_predictors.py"
            assert meta[name+"_source_sha256"]==r[name+"_source_sha256"]==sha(source/filename)
        assert meta["fit_pack_sha256"]==r["fit_pack_sha256"]==h["fit_pack_sha256"]==sha(packdir/"fit.npz")
        assert r["model_sha256"]==h["model_sha256"]==sha(modelpath) and h["fit_receipt_sha256"]==sha(receiptpath)
        assert h["execution_receipt_sha256"]==sha(executionpath)
        for role in ids:assert h[role+"_ids"]==ids[role]
        for name,digest in h["files"].items():assert Path(name).name==name and sha(directory/name)==digest
        assert datetime.datetime.fromisoformat(job["finished_UTC"])<=datetime.datetime.fromisoformat(h["frozen_UTC"])
        with np.load(packdir/"fit.npz",allow_pickle=False) as z:
            fit={k:z[k].copy() for k in ["x","delta_aee","valid_pixels","ids","groups","aux_g1","aux_g4"]}
        fit["valid_pixels"]=fit["valid_pixels"].astype(np.float64)
        assert digest_arrays(fit[k] for k in sorted(fit))==meta["fit_input_and_label_sha256"]
        for trace in m["crossfit_trace"].values():folds(trace,ids["fit"])
        nestedpath=directory/"FIT_NESTED_LOSO_V1.json";nested=read(nestedpath)
        assert sha(nestedpath)==job["fit_loso_sha256"]
        assert nested["ids"]==ids["fit"] and nested["uses_fit_only"] and not nested["confirmation_read"] and not nested["model_selection"]
        folds(nested["results"]["trace"],ids["fit"])
        # Validate actual raw/aux coefficients by their weighted normal-equation
        # residuals; no refit, inference or quality selection is performed.
        X=fit["x"];w=fit["valid_pixels"]/fit["valid_pixels"].sum()
        for key,rm,Y in [("direct_raw",m["models"]["direct_raw"],fit["delta_aee"]),
                         ("aux_g1",m["auxiliary"]["g1"],fit["aux_g1"]),
                         ("aux_g4",m["auxiliary"]["g4"],fit["aux_g4"])]:
            mean=(w[:,None]*X).sum(axis=0);scale=np.sqrt((w[:,None]*(X-mean)**2).sum(axis=0));scale=np.where(scale>1e-12,scale,1)
            intercept=(w[:,None]*Y).sum(axis=0)
            assert np.allclose(rm["mean"],mean,rtol=1e-12,atol=1e-12)
            assert np.allclose(rm["scale"],scale,rtol=1e-12,atol=1e-12)
            assert np.allclose(rm["intercept"],intercept,rtol=1e-12,atol=1e-12)
            A=(X-mean)/scale*np.sqrt(w[:,None]);B=(Y-intercept)*np.sqrt(w[:,None]);coef=np.asarray(rm["coef"])
            residual=A.T@(A@coef-B)+.1*coef
            maxres=float(np.abs(residual).max());assert maxres<1e-8
            normal_residuals[branch+"/"+key]=maxres
        for rm in list(m["models"].values())+list(m["auxiliary"].values()):
            assert rm["alpha"]==.1 and rm["fit_rows"]==64
            for key in ["mean","scale","intercept","coef"]:assert np.isfinite(rm[key]).all()
        expected=np.average(fit["delta_aee"],axis=0,weights=fit["valid_pixels"])
        assert np.allclose(expected,meta["fit_static_true_delta_means"],rtol=0,atol=1e-15)
        assert int(np.argmin(expected))==meta["fit_static_six_action"]
        assert (-1 if expected.min()>=0 else int(np.argmin(expected)))==meta["fit_static_allow_Q4_action"]
        branches.append(dict(branch=branch,model_sha256=sha(modelpath),fit_receipt_sha256=sha(receiptpath),
                             heads_freeze_sha256=sha(headpath),source_freeze_sha256=sha(source/"SOURCE_FREEZE_V1.json"),
                             fit_pack_sha256=sha(packdir/"fit.npz"),fit_identities=64,heads=7,input_width=width))
    jointdir=ROUND/"confirmation_freeze_v1";jointpath=jointdir/"HEADS_FREEZE_V1.json";joint=read(jointpath)
    assert sha(jointpath)=="ebad14475bcbbfd8c17185d6debc9e1696c1661b8aa81501357b54ae2956255b"
    for name,digest in joint["files"].items():assert Path(name).name==name and sha(jointdir/name)==digest
    assert set(joint["files"])=={"MODEL_V1.json","FIT_RECEIPT_V1.json","DIRECTIONAL_MODEL_V1.json","DIRECTIONAL_FIT_RECEIPT_V1.json","DIRECTIONAL_HEADS_FREEZE_V1.json"}
    assert joint["joint_branches"]==[b["branch"] for b in branches]
    assert joint["base_branch_heads_freeze_sha256"]==branches[0]["heads_freeze_sha256"]
    for idx,prefix in [(0,""),(1,"DIRECTIONAL_")]:
        assert sha(jointdir/(prefix+"MODEL_V1.json"))==branches[idx]["model_sha256"]
        assert sha(jointdir/(prefix+"FIT_RECEIPT_V1.json"))==branches[idx]["fit_receipt_sha256"]
    relocated=read(jointdir/"DIRECTIONAL_HEADS_FREEZE_V1.json")
    original=read(ROUND/"fitted/directional160_v1/HEADS_FREEZE_V1.json")
    for key,value in original.items():
        if key!="files":assert relocated[key]==value
    assert relocated["relocated_from_heads_freeze_sha256"]==branches[1]["heads_freeze_sha256"]
    for name,digest in relocated["files"].items():assert sha(jointdir/name)==digest
    # Exercise precisely the already-frozen collector-auditor128 gate against
    # the actual joint directory; additional160 files must all be byte-bound.
    import audit_collector_actual_v1 as collector_audit
    collector_audit.audit_heads(jointpath,{"heads_freeze_sha256":sha(jointpath)},membership)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    out=dict(schema="flow_r33_confirmation_prelaunch_independent_review_v1",status="PASS_FOR_FIXED_280_CONFIRMATION_FORWARDS",
        reviewed_UTC=now,joint_heads_freeze_sha256=sha(jointpath),membership_sha256=membershipsha,
        collector_sha256=sha(ROUND/"collect.py"),branches=branches,
        source_execution_receipt_sha256=sha(executionpath),pack_independent_audit_sha256=sha(HERE/"FIT_TUNE_PACK_INDEPENDENT_AUDIT_V1.json"),
        folds_checked=checkedfolds,normal_equation_max_abs_residual=normal_residuals,
        exact_fit_tune_confirm_ids_bound=True,all_fourteen_heads_frozen=True,
        confirmation_identities=40,confirmation_sequences=5,expected_confirmation_model_forwards=280,
        additional_dense_references=0,GT_confirmation_read=False,new_fitting=False,model_changed=False,GPU_used=False,
        limitations=["Fit-only OOF trace and coefficient checks verify implementation and data scope, not accuracy or generalization.",
                     "An independent auditor has decoded fit/tune arrays for integrity after the two fitted model artifacts were written; no changes or model selection followed.",
                     "The actual remote launch must still use this joint freeze SHA and the frozen collector, then pass execution-only audit before confirmation quality."])
    output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"status":out["status"],"joint_heads_freeze_sha256":sha(jointpath),"folds_checked":checkedfolds,"heads":14}))


if __name__=="__main__":main()
