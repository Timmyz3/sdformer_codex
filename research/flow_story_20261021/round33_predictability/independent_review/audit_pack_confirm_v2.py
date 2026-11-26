"""Independent original-array confirmation-v2 pack audit; CPU local or near-data."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
ROOT = ROUND.parents[2]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def frozen_pack(path):
    freeze=read(path/"PACK_FREEZE_V1.json")
    assert freeze["schema"]=="flow_r33_pack_freeze_v1" and freeze["phase"]=="confirm"
    for name,digest in freeze["files"].items():
        assert Path(name).name==name and sha(path/name)==digest
    return freeze


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def masks_from_sources(gt,valid,events,boundary,large):
    assert gt.dtype==np.float32 and valid.dtype==np.bool_ and events.dtype==np.float32
    assert gt.shape==(2,480,640) and valid.shape==(480,640) and events.shape==(10,480,640)
    assert np.isfinite(gt).all() and np.isfinite(events).all() and valid.any()
    f=gt.astype(np.float64)
    known=np.zeros(valid.shape,dtype=bool)
    known[1:-1,1:-1]=np.logical_and.reduce([valid[1:-1,1:-1],valid[:-2,1:-1],valid[2:,1:-1],valid[1:-1,:-2],valid[1:-1,2:]])
    boundary_mask=np.zeros_like(known)
    for other in [f[:,:-2,1:-1],f[:,2:,1:-1],f[:,1:-1,:-2],f[:,1:-1,2:]]:
        diff=f[:,1:-1,1:-1]-other
        boundary_mask[1:-1,1:-1]|=np.sqrt(np.sum(diff*diff,axis=0))>boundary
    B=known&boundary_mask
    L=valid&(np.sqrt(np.sum(f*f,axis=0))>large)
    active=np.count_nonzero(events,axis=0)>0
    result=dict(all_valid=valid,B=B,L=L,B_and_L=B&L,B_only=B&~L,
                L_only_B_known=L&known&~B,neither_B_known=known&~B&~L,
                B_unknown=valid&~known,voxel_nonzero=valid&active,voxel_zero=valid&~active)
    for key in ["B","L","B_unknown"]:
        result[key+"_voxel_nonzero"]=result[key]&active
        result[key+"_voxel_zero"]=result[key]&~active
    assert len(result)==16 and all(not np.any(m&~valid) for m in result.values())
    return result


def independent_features(events):
    z=events.astype(np.float64);first=[];extra=[]
    times=np.linspace(-1.,1.,10)
    for cy in range(4):
        for cx in range(4):
            a=z[:,cy*120:(cy+1)*120,cx*160:(cx+1)*160]
            mass=np.sum(np.abs(a),axis=(1,2));den=mass.sum()
            first.extend([np.sum(a[a>0])/a.size,-np.sum(a[a<0])/a.size,
                          np.sum(a>0)/a.size,np.sum(a<0)/a.size,
                          np.sum(np.abs(np.sum(a,axis=0)))/(10*120*160),
                          np.sum(np.abs(a[1:]-a[:-1]))/(9*120*160),
                          np.sum(times*mass)/den if den else 0,
                          np.sum(times*times*mass)/den if den else 0])
            temporal=a[1:,1:-1,1:-1]-a[:-1,1:-1,1:-1]
            dx=(a[1:,1:-1,2:]+a[:-1,1:-1,2:]-a[1:,1:-1,:-2]-a[:-1,1:-1,:-2])/4
            dy=(a[1:,2:,1:-1]+a[:-1,2:,1:-1]-a[1:,:-2,1:-1]-a[:-1,:-2,1:-1])/4
            for spatial in [dx,dy]:
                den=np.sqrt(np.sum(spatial**2)*np.sum(temporal**2))
                extra.append(np.sum(spatial*temporal)/den if den>1e-12 else 0.)
    return np.asarray(first),np.asarray(extra)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ["run","source","pack128","pack160","execution-audit","output"]:
        ap.add_argument("--"+name,type=Path,required=True)
    args=ap.parse_args()
    output=args.output;assert not output.exists()
    audit=read(args.execution_audit)
    assert audit["schema"]=="flow_r33_confirm_actual_execution_audit_v2"
    assert audit["status"]=="PASS" and audit["phase"]=="confirm" and audit["GT_values_read"] is False
    assert audit["collector_sha256"]=="1d2b80e6f7fc1c34c9a185086cce3ebd92096424b5ef3dcb77780d5ce8a03af9"
    assert audit["native_verification_sha256"]=="769671e6e3754f068587eaed576ce51c0d0d6f939aeabf033a006f9aec9e6c5c"
    run=args.run;source=args.source;p128=args.pack128;p160=args.pack160
    assert sha(run/"results.json")==audit["source_result_sha256"]
    result=read(run/"results.json")
    assert result["phase"]=="confirm" and result["runner_sha256"]==audit["collector_sha256"]
    assert result["native_verification_sha256"]==audit["native_verification_sha256"]
    assert sha(ROUND/"data/MEMBERSHIP_V1.json")=="d860612a3ae3e4620fa2a9a616df9f8e93774efadcf7dc564881e27b7e7da8cc"
    membership=read(ROUND/"data/MEMBERSHIP_V1.json")
    rows=[r for r in membership["records"] if r["role"]=="confirm"]
    assert len(rows)==40 and [r["target_id"] for r in rows]==result["selected_ids"]
    f128,f160=frozen_pack(p128),frozen_pack(p160)
    assert sha(p128/"QUALITY_V1.json")==sha(p160/"QUALITY_V1.json")
    quality=read(p128/"QUALITY_V1.json")
    assert quality["source_result_sha256"]==audit["source_result_sha256"]
    assert quality["phase"]=="confirm"
    assert quality["runner_sha256"]==sha(ROUND/"make_pack_confirm_v2.py")=="97a56b032d7fd6915cdf8596f0f158823b6f0e731093bb90c39bea4c3897b9d5"
    assert quality["predictors_sha256"]=="35663dd6b999621652d9d185d72b967e33c380e1064a09ec75540d2759cd61c3"
    regionpath=ROOT/"research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json"
    assert sha(regionpath)==quality["regions_sha256"]=="2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b"
    region_contract=read(regionpath)
    frame_quality={r["target_id"]:r for r in quality["frames"]}
    assert len(frame_quality)==40 and set(frame_quality)=={r["target_id"] for r in rows}
    augmentation=read(p160/"DIRECTIONAL_AUGMENT_RECEIPT_V1.json")
    assert augmentation["original_pack_freeze_sha256"]==sha(p128/"PACK_FREEZE_V1.json")
    assert augmentation["predictors_sha256"]==sha(ROUND/"directional_branch/predictors.py")
    assert augmentation["runner_sha256"]==sha(ROUND/"augment_directional_pack.py")
    packs={};loc={}
    for role,n in [("confirm",40)]:
        base,extra=arrays(p128/(role+".npz")),arrays(p160/(role+".npz"))
        assert set(base)==set(extra) and len(base["ids"])==n
        assert base["x"].shape==(n,128) and extra["x"].shape==(n,160)
        for key in base:
            y=extra[key][:,:128] if key=="x" else extra[key]
            assert base[key].dtype==y.dtype and base[key].shape==y.shape
            assert np.ascontiguousarray(base[key]).tobytes()==np.ascontiguousarray(y).tobytes()
        assert base["ids"].tolist()==[r["target_id"] for r in rows if r["role"]==role]
        assert set(base["roles"])=={role}
        packs[role]=(base,extra)
        for j,tid in enumerate(base["ids"]):loc[str(tid)]=(role,j)
    states=["background_Q4"]+[f"upgrade_{a}" for a in range(6)]
    assert quality["states"]==states
    by_state={s:{r["target_id"]:r for r in result["evaluations"][s]["frames"]} for s in states}
    maxdiff={};counts=dict(identities=0,source_files=0,predictions=0,region_metrics=0,auxiliary_grids=0)
    def compare(name,a,b,atol=2e-11,rtol=2e-12):
        a,b=np.asarray(a),np.asarray(b)
        assert a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)),name
        diff=float(np.max(np.abs(a-b))) if a.size else 0.
        maxdiff[name]=max(maxdiff.get(name,0.),diff)
        assert np.allclose(a,b,atol=atol,rtol=rtol),(name,diff)
    for rec in rows:
        tid=rec["target_id"];role,j=loc[tid];base,aug=packs[role]
        assert str(base["groups"][j])==rec["sequence"] and frame_quality[tid]["role"]==role
        original={}
        for item in rec["source_files"]:
            path=source/item["relative_path"]
            assert sha(path)==item["sha256"]
            original[item["category"]]=np.load(path,allow_pickle=False)
            counts["source_files"]+=1
        gt,valid,events=original["gt"],original["mask"],original["input"]
        masks=masks_from_sources(gt,valid,events,region_contract["boundary_threshold_pixels"],region_contract["large_displacement_threshold_pixels"])
        predictions=[];aees=[]
        for state in states:
            row=by_state[state][tid];path=run/row["prediction_file"]
            assert sha(path)==row["prediction_file_sha256"]
            with np.load(path,allow_pickle=False) as z:pred=z["head3"].copy()
            assert hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest()==row["prediction_sha256"]
            pred=pred.astype(np.float64);predictions.append(pred)
            error=np.sqrt(np.sum((pred-gt.astype(np.float64))**2,axis=0))
            measured=frame_quality[tid]["states"][state]
            assert set(measured)==set(masks)
            for name,mask in masks.items():
                n=int(np.count_nonzero(mask));s=float(np.sum(error[mask]))
                saved=measured[name]
                assert n==saved["valid_pixels"]
                compare("raw_error_sum",s,saved["error_sum"],atol=1e-8)
                if n:compare("raw_AEE",s/n,saved["AEE"])
                else:assert saved["AEE"] is None
                counts["region_metrics"]+=1
            aees.append(float(np.mean(error[valid])))
            counts["predictions"]+=1
        n=int(np.count_nonzero(valid));assert n==int(base["valid_pixels"][j])
        compare("background_AEE",aees[0],base["background_AEE"][j])
        compare("delta_aee",np.asarray(aees[1:])-aees[0],base["delta_aee"][j])
        e=predictions[0]-gt.astype(np.float64)
        unit=e/np.maximum(np.sqrt(np.sum(e*e,axis=0)),1e-12)[None]
        delta=np.stack(predictions[1:])-predictions[0][None]
        compare("ideal_signed_full",np.asarray([np.sum(unit[:,valid]*d[:,valid])/n for d in delta]),base["ideal_signed_full"][j])
        for g in [1,4]:
            v=np.zeros((2,g,g));dmean=np.zeros((6,2,g,g));support=np.zeros((g,g),dtype=np.int64)
            for cy in range(g):
                for cx in range(g):
                    ys=slice(cy*480//g,(cy+1)*480//g);xs=slice(cx*640//g,(cx+1)*640//g)
                    mask=valid[ys,xs];nc=int(np.count_nonzero(mask));support[cy,cx]=nc
                    if nc:
                        v[:,cy,cx]=unit[:,ys,xs][:,mask].sum(axis=1)/n
                        dmean[:,:,cy,cx]=delta[:,:,ys,xs][:,:,mask].mean(axis=2)
            assert np.array_equal(support,base[f"support_counts_g{g}"][j])
            compare(f"aux_g{g}",np.concatenate([v.reshape(-1),dmean.reshape(-1)]),base[f"aux_g{g}"][j])
            compare(f"ideal_signed_g{g}",(v[None]*dmean).sum(axis=(1,2,3)),base[f"ideal_signed_g{g}"][j])
            counts["auxiliary_grids"]+=1
        x,extra=independent_features(events)
        compare("base128",x,base["x"][j])
        compare("directional32",extra,aug["x"][j,128:])
        counts["identities"]+=1
        if counts["identities"]%10==0:print("independent_recomputed",counts["identities"],flush=True)
    assert counts==dict(identities=40,source_files=120,predictions=280,region_metrics=4480,auxiliary_grids=80)
    out=dict(schema="flow_r33_confirm_actual_pack_independent_audit_v2",status="PASS",counts=counts,
        execution_audit_sha256=sha(args.execution_audit),
        native_verification_sha256=audit["native_verification_sha256"],
        source_result_sha256=sha(run/"results.json"),pack128_freeze_sha256=sha(p128/"PACK_FREEZE_V1.json"),
        pack160_freeze_sha256=sha(p160/"PACK_FREEZE_V1.json"),augmentation_receipt_sha256=sha(p160/"DIRECTIONAL_AUGMENT_RECEIPT_V1.json"),
        quality_sha256=sha(p128/"QUALITY_V1.json"),independent_auditor_sha256=sha(__file__),
        max_abs_recompute_difference=maxdiff,all_nonX_arrays_and_base128_prefix_byte_identical=True,
        region_source="Frozen R13 thresholds; masks independently reconstructed from GT+events.",
        auxiliary_source="Independent explicit cell slicing/sums; neither make_labels nor pooled nor signed_scores called.",
        feature_source="Independent formulas directly from source raw signed T10; no GT support in either input feature branch.",
        GT_read_roles=["confirm"],confirmation_read=True,model_fit_or_changed=False,GPU_used=False,
        conclusion="Actual source-bound labels, all16 quality regions, both input branches and role separation verified. This is data integrity evidence, not selector effectiveness.")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"status":"PASS",**counts},indent=2))


if __name__=="__main__":main()
