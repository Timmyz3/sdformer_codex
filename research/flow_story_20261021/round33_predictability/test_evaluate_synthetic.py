#!/usr/bin/env python3
"""Synthetic scalar evaluation checks; no real quality/confirmation file read."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import evaluate as e


def fixture():
    n=4
    true=np.asarray([[-.2,.1,0,.2,.5,-.1],[.1,.2,.3,.4,.5,.6],[-.1,-.1,0,.3,.4,.2],[.5,.4,.3,.2,.1,0]])
    counts=np.asarray([1000,400,128,64])
    pack=dict(ids=np.asarray([f"synthetic_{i}" for i in range(n)]),groups=np.asarray(["a","a","b","b"]),
              valid_pixels=counts,delta_aee=true)
    rng=np.random.default_rng(3302)
    for g in (1,4):pack[f"aux_g{g}"]=rng.normal(scale=.03,size=(n,14*g*g))
    result=dict(scores={name:true.copy() for name in e.models.METHODS},
        predicted_auxiliary={f"g{g}":pack[f"aux_g{g}"].copy() for g in (1,4)},
        fit_static_six_choices=[0]*n,fit_static_allow_Q4_choices=[-1]*n)
    # Distinguishable stronger/weaker ordinary references, without any fitting.
    result["scores"]["direct_raw"]=-true
    result["scores"]["direct_aug_g1"]=np.full_like(true,.5)
    frames=[]
    for i in range(n):
        states={}
        for a,state in enumerate(e.STATES):
            delta=0 if a==0 else true[i,a-1]
            states[state]={}
            for j,region in enumerate(e.REGIONS):
                support=int(counts[i]) if region=="all_valid" else (int(counts[i])//2 if j%3 else 0)
                aee=(i+1)+delta*(1 if region=="all_valid" else j+.5)
                # Keep synthetic errors nonnegative while producing clear tail flags.
                aee=max(0,aee)
                states[state][region]=dict(valid_pixels=support,error_sum=float(support*aee),AEE=float(aee) if support else None)
        frames.append(dict(target_id=str(pack["ids"][i]),sequence=str(pack["groups"][i]),states=states))
    hw=dict(candidates=[dict(action=i,static_affine_BN_ticks=100+i,dynamic_affine_BN_ticks=110+i) for i in range(6)]
         +[dict(action=6,static_affine_BN_ticks=80,dynamic_affine_BN_ticks=80)])
    return pack,dict(results=result),{},frames,hw


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    assert not args.output.exists()
    pack,pred,model,frames,hw=fixture()
    out=e.evaluate_data(pack,pred,model,frames,hw)
    assert len(out["policy_definitions"])==32
    # Manual oracle choices and support-weighted whole-frame means.
    oracle=[0,0,0,5]
    selected=[frames[i]["states"][f"upgrade_{a}"]["all_valid"]["AEE"] for i,a in enumerate(oracle)]
    expected=float(np.dot(pack["valid_pixels"],selected)/sum(pack["valid_pixels"]))
    assert np.isclose(out["quality_summaries"]["oracle__six"]["overall"]["all_valid"]["pixel_weighted_AEE"],expected)
    assert np.isclose(out["quality_summaries"]["oracle__six"]["overall"]["all_valid"]["frame_mean_AEE"],np.mean(selected))
    actual=[r["policies"]["oracle__allow_Q4"]["action"] for r in out["per_identity"]]
    assert actual==[0,-1,0,-1]
    assert all(r["policies"]["oracle__six"]["regret_six_action_AEE"]==0 for r in out["per_identity"])
    assert all(r["policies"]["oracle__allow_Q4"]["regret_seven_action_AEE"]==0 for r in out["per_identity"])
    assert out["per_identity"][1]["policies"]["Q4"]["regret_six_action_AEE"]<0
    for policy in out["fixed_six_policy_names"]:
        assert out["decision_summaries"][policy]["overall"]["upgrade_fraction"]==1
    # Every policy selects using allGT first; unsupported regions remain null.
    assert out["quality_summaries"]["oracle__six"]["overall"]["B_and_L"]["pixel_weighted_AEE"] is None
    uniform_error=sum(sum(frames[i]["states"][s]["all_valid"]["error_sum"] for s in e.STATES[1:])/6 for i in range(4))
    assert np.isclose(out["quality_summaries"]["uniform_six_expectation"]["overall"]["all_valid"]["error_sum"],uniform_error)
    # A mean-quality flag cannot replace risk of drawing one of six actions.
    risk=out["uniform_six_tail_risk"]
    manual_bad_counts={r:0 for r in ("all_valid","B","L")};manual_any=0
    for i,frame in enumerate(frames):
        union=set()
        for region in manual_bad_counts:
            base=frame["states"]["background_Q4"][region];bad=[]
            for a in range(6):
                val=frame["states"][f"upgrade_{a}"][region]
                if base["valid_pixels"]>=256 and (val["AEE"]>1.2*base["AEE"] or val["AEE"]-base["AEE"]>1):bad.append(a)
            assert risk["frames"][i]["regions"][region]["triggering_actions"]==bad
            assert risk["frames"][i]["regions"][region]["trigger_probability"]==len(bad)/6
            manual_bad_counts[region]+=len(bad);union.update(bad)
        manual_any+=len(union)
        assert risk["frames"][i]["any_primary_region_trigger_probability"]==len(union)/6
    for region,count in manual_bad_counts.items():
        assert np.isclose(risk["summaries"]["overall"]["regions"][region]["expected_bad_frame_count"],count/6)
    assert np.isclose(risk["summaries"]["overall"]["expected_any_primary_region_bad_frame_count"],manual_any/6)
    assert risk["frames"][0]["regions"]["all_valid"]["trigger_probability"]>0
    meanrow=out["comparisons"]["uniform_six_expectation__vs__Q4"]["frames"][0]
    assert not meanrow["regions"]["all_valid"]["high_support_tail_review_flag"]
    # Operator-only action mapping: algorithm Q4=-1 -> hardware6, multiplied4.
    q4cost=out["decision_summaries"]["Q4"]["overall"]["operator_only_service"]
    assert q4cost["mean_four_conv_ticks"]==320
    staticcost=out["decision_summaries"]["fit_static__six"]["overall"]["operator_only_service"]
    assert staticcost["mean_four_conv_ticks"]==400
    uniformcost=out["decision_summaries"]["uniform_six_expectation"]["overall"]["operator_only_service"]
    assert np.isclose(uniformcost["mean_four_conv_ticks"],450)
    assert uniformcost["max_possible_four_conv_ticks"]==460
    # Every required strong ordinary and simplified reference exists.
    for regime in ("six","allow_Q4"):
        for ref in ("direct_aug_g1","direct_aug_g4","direct_residual_g1","direct_residual_g4","signed_g1","fit_static"):
            key=f"signed_g4__{regime}__vs__{ref}__{regime}"
            assert key in out["comparisons"]
            assert "aggregate_pass" not in out["comparisons"][key]
    assert out["comparisons"]["direct_raw__six__vs__Q4"]["high_support_tail_review"]
    assert out["comparisons"]["direct_raw__six__vs__Q4"]["aggregate_2pct_review"]["allGT_pixel"]["review_flag"]
    # Fully true replacement predictions collapse to the corresponding ideal.
    for g in (1,4):
        for regime in ("six","allow_Q4"):
            for row in out["per_identity"]:
                base=row["policies"][f"ideal_signed_g{g}__{regime}"]["action"]
                assert base==row["policies"][f"offline_predv_trued_g{g}__{regime}"]["action"]
                assert base==row["policies"][f"offline_truev_predd_g{g}__{regime}"]["action"]
    result=dict(schema="flow_r33_evaluator_synthetic_checks_v1",status="PASS",policies_checked=32,
        checks=["manual pixel/frame oracle", "Q4-inclusive oracle ties and negative six-regret", "fixed budget excludes Q4",
                "all16 and empty support", "uniform exact expected errors and separate maximum service",
                "static/dynamic four-operator mapping", "all mandatory ordinary comparisons", "tail and 2pct review only",
                "uniform single-action tail probabilities and union expected counts, not threshold of expectation",
                "offline replacements collapse when fields identical"],
        real_quality_read=False,confirmation_read=False,head_fitting=False,GPU_used=False,
        source_sha256={Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (__file__,e.__file__)})
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status="PASS",policies=32,real_quality_read=False)))


if __name__=="__main__":main()
