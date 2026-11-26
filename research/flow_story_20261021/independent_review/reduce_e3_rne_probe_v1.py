"""E3-only execution binding and raw16-region single-upgrade headroom analysis.

Does not call the unexecuted E2 binder. All actions are independent RNE4/Q8
at the actual completed E0 step512 endpoint, never recovered old Q6/drop4.
"""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reduce_e1_quality_v1 as b
import audit_e0_work_v1 as ew
ROOT, R27, raw, mc = b.ROOT, b.R27, b.raw, b.mathcore
sha, read, child, prediction = b.sha, b.read, b.child, b.prediction
E3 = HERE.parent/"e3_rne_probe"
UPGRADES = [f"upgrade_{i}" for i in range(6)]
ACTUAL = ["background_Q4"] + UPGRADES
REFS = ["endpoint_Q8", "endpoint_M", "original_native", "recovered_native"]


def independent_layouts(contract):
    assert contract["schema"] == "flow_e3_rne_patch6_probe_contract_v1"
    assert contract["status"] == "FROZEN_ENDPOINT_STEP512_PENDING_ROOT_EXECUTION_DECISION"
    assert set(contract["geometries"]) == ew.TARGETS and set(contract["family_targets"]) == ew.CONVS
    assert [g["id"] for g in contract["groups"]] == list(range(6))
    assert (contract["formal_forwards"], contract["engineering_dense_reference_forwards"],
            contract["total_forwards"]) == (112,6,118)
    for i,g in enumerate(contract["groups"]):
        cells = [[j//35,j%35] for j in range(i*151,(i+1)*151)]
        assert set(g["tiles"]) == ew.CONVS and all(v == cells for v in g["tiles"].values())
        assert g["added_core_ticks_vs_Q4"] == 82192320
    geometries = {}
    for n,g in contract["geometries"].items():
        group = n.rsplit(".",1)[0] if n in ew.LINEARS else n
        geometries[group] = dict(H=g["H"], W=g["W"])
    out = {}
    for action in range(-1,6):
        selected = {n:[] for n in ew.GROUPS}
        if action >= 0:
            selected.update(contract["groups"][action]["tiles"])
        masks = {}
        for n,g in geometries.items():
            m = np.zeros((g["H"],g["W"]),dtype=bool)
            for y,x in selected[n]:
                m[y*9:y*9+9,x*9:x*9+9] = True
            masks[n] = m
        # Translate only declared geometry/tiles into the independent E0 work
        # checker's contract vocabulary. No old R30 choices are reused.
        wc = dict(seed=20261019, tile_size=9,
                  geometries={"full9":{n:dict(**g,Q8_complete_tiles=len(selected[n])) for n,g in geometries.items()}},
                  evaluation_tiles=selected)
        out["background_Q4" if action < 0 else f"upgrade_{action}"] = (masks,wc)
    return out


def bind(a):
    r = read(a.run/"results.json")
    assert r["schema"] == "flow_e3_rne_probe_v1" and r["status"] == "E3_RNE112_AND6_DENSE_REFERENCES_COMPLETE"
    assert (r["model_forwards"],r["formal_model_forwards"],r["engineering_model_forwards"]) == (118,112,6)
    assert (r["expected_model_forwards"],r["expected_formal_forwards"],r["expected_engineering_forwards"]) == (118,112,6)
    assert r["completed_optimizer_steps"] == 0 and r["background_replay_bitwise_equal_all16"]
    for flag in ("training_performed","development_GT_arrays_read","quality_computed","quality_selection",
                 "predictor_implemented","resume_allowed"):
        assert r[flag] is False
    assert r["runtime"]["AMP"] is False and r["runtime"]["TF32"] is False
    freeze_path = E3/"SOURCE_FREEZE_V1.json"
    freeze = read(freeze_path)
    assert all(sha(E3/n) == h for n,h in freeze["files"].items())
    sources = dict(runner=E3/"collect_probe.py", contract_tool=E3/"probe_contract.py",
                   core=HERE.parent/"e0_ordinary/mixed_integer_core.py",
                   source_freeze=freeze_path, action_contract=a.contract,
                   selections=HERE.parent/"selections_v1.json",
                   members=R27/"independent_review/membership_v1.json",
                   smoke=ROOT/"research/flow_story_20261019/round28/implementation/run_smoke_v2.py",
                   helper=ROOT/"research/flow_story_20261019/round28/implementation/run_recovery_v2.py")
    for k,p in sources.items():
        assert r["sources"][k] == sha(p), k
    cpu = read(HERE/"E3_INDEPENDENT_CPU_V1.json")
    assert cpu["status"] == "PASS" and cpu["source_freeze_sha256"] == sha(freeze_path)
    contract = read(a.contract)
    assert r["action_contract"] == contract
    layouts = independent_layouts(contract)
    e0 = read(a.e0_run/"results.json")
    ea = read(a.e0_audit)
    assert ea["schema"] == "flow_e0_independent_actual_execution_audit_v1"
    assert ea["status"] == "PASS" and ea["quality_analysis_allowed"]
    assert ea["counts"] == dict(optimizer_updates=512,quality_predictions=320,formal_model_forwards=1353)
    assert ea["source_result_sha256"] == sha(a.e0_run/"results.json")
    ep = contract["endpoint"]
    assert ep == {k:r["sources"][k] for k in ep}
    assert ep["role"] == "formal_step512_only" and ep["completed_optimizer_steps"] == 512
    assert ep["result_sha256"] == sha(a.e0_run/"results.json")
    assert ep["weights_sha256"] == ea["checkpoint_sha256"] == e0["weights_sha256"]
    assert ep["state_sha256"] == ea["final_state_sha256"] == e0["final_state_sha256"] == r["source_state_sha256"] == r["final_state_sha256"]
    roles, members = read(sources["selections"]), read(sources["members"])
    ids = roles["e2_reserved_ids"]
    assert ids == r["selected_ids"] == contract["target_ids"] and len(ids) == len(set(ids)) == 16
    assert not set(ids) & set(roles["localization_ids"] + roles["confirmation_reserved_ids"])
    records = {x["target_id"]:x for x in members["records"]}
    regions_path = ROOT/"research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json"
    assert sha(regions_path) == b.REGIONS_SHA
    regions = read(regions_path)
    assert not set(ids) & set(regions["fit_target_ids"] + members["sealed_ids"] + members["H8_historical_ids"])
    raw_sources, evidence = {}, {}
    for tid in ids:
        raw_sources[tid], evidence[tid] = {}, {}
        for f in records[tid]["source_files"]:
            k = {"input":"events","gt":"gt","mask":"mask"}[f["category"]]
            p = child(R27/"dev64_source_snapshot",f["relative_path"])
            assert sha(p) == f["sha256"] and p.stat().st_size == int(f["bytes"])
            raw_sources[tid][k] = p
            evidence[tid][k] = dict(relative_path=f["relative_path"],sha256=f["sha256"])
    q4 = {f["target_id"]:f for f in e0["evaluations"]["endpoint_Q4"]["frames"]}
    bound, inputs = {}, {}
    assert set(r["evaluations"]) == set(ACTUAL)
    for action,name in enumerate(ACTUAL,start=-1):
        ev = r["evaluations"][name]
        assert ev["action"] == action and ev["state_sha256"] == ep["state_sha256"]
        assert ev["model_forwards"] == 16 and ev["GT_arrays_read"] is False
        assert [f["target_id"] for f in ev["frames"]] == ids
        masks,wc = layouts[name]
        mode = "Q4" if action < 0 else "M"
        ml = ev["mask_layout"]
        assert ml["action"] == action and ml["mode"] == mode
        assert ml["mask_sha256"] == {n:raw.payload_sha(m) for n,m in masks.items()}
        assert ml["selected_tiles"] == {n:int(m.sum())//81 for n,m in masks.items()}
        assert ml["added_core_ticks_vs_Q4"] == (0 if action < 0 else 82192320)
        for row in ev["frames"]:
            tid = row["target_id"]
            assert row["sequence"] == records[tid]["sequence"] and row["action"] == action
            assert row["input_evidence"] == evidence[tid] == q4[tid]["input_evidence"]
            assert row["actual_BN_calls"] == q4[tid]["actual_BN_calls"]
            assert len(row["actual_BN_calls"]) == 78 and set(row["actual_BN_calls"].values()) == {1}
            assert set(row["actual_affine_call_counts"]) == ew.TARGETS and set(row["actual_affine_call_counts"].values()) == {1}
            if action == -1:
                assert row["heads"] == q4[tid]["heads"]
                inputs[tid] = row["actual_input"]
                assert inputs[tid]["shape"] == [1,10,2,480,640] and inputs[tid]["dtype"] == "torch.float32"
            assert row["actual_input"] == inputs[tid]
            assert all("kernel_calls" in w for w in row["work"].values())
            ew.check_calls(q4[tid]["actual_affine_calls"],row["work"],mode,(480,640),wc)
            for n,w in row["work"].items():
                assert w["representation"] == q4[tid]["work"][n]["representation"]
            bound[(name,tid)] = prediction(a.run,row)
    assert set(r["parity"]) == set(UPGRADES)
    for name,p in r["parity"].items():
        assert p["target_id"] == ids[0] and p["comparison"]["pass"]
        checks = p["comparison"]["checks"]
        assert len(checks) == 137
        assert sum(n.startswith("integer::") for n in checks) == 28
        assert sum(n.startswith("PSN::") for n in checks) == 105
        assert sum(n.startswith("head") for n in checks) == 4
        assert all(x["bitwise_equal"] and x["different_bytes"] == 0 and x["actual_sha256"] == x["reference_sha256"] for x in checks.values())
        for key,dense in (("actual_record",False),("dense_record",True)):
            rec = p[key]
            assert (rec["integer_count"],rec["PSN_count"],rec["head_count"]) == (28,105,4)
            assert rec["ste_entry"] is False and rec["dense_integer_reference"] is dense
            assert rec["actual_BN_calls"] == q4[ids[0]]["actual_BN_calls"]
            assert set(rec["actual_target_calls"]) == ew.TARGETS and set(rec["actual_target_calls"].values()) == {1}
            ew.check_calls(q4[ids[0]]["actual_affine_calls"],rec["work"],"M",(480,640),layouts[name][1],dense=dense)
        heads = r["evaluations"][name]["frames"][0]["heads"]
        assert all(heads[f"head{i}"]["sha256"] == checks[f"head{i}"]["actual_sha256"] for i in range(4))
    audited = {(x["state"],x["target_id"]):x for x in ea["predictions"]}
    for state in ("endpoint_Q8","endpoint_M"):
        for row in e0["evaluations"][state]["frames"]:
            tid = row["target_id"]
            if tid not in ids:
                continue
            key = ("recovered_rne_"+state.removeprefix("endpoint_"),tid)
            assert row["prediction_file_sha256"] == audited[key]["file_sha256"]
            assert row["prediction_sha256"] == audited[key]["head3_sha256"]
            assert row["input_evidence"] == evidence[tid]
            bound[(state,tid)] = prediction(a.e0_run,row)
    native_run = R27/"actual_v1/formal_native_v1"
    native = read(native_run/"results.json")
    na = read(R27/"actual_execution_audit_v1.json")
    assert na["status"] == "PASS" and na["quality_analysis_allowed"]
    assert na["source_results"]["native"] == sha(native_run/"results.json")
    for stage,state in (("initial","original_native"),("endpoint","recovered_native")):
        rows = {x["target_id"]:x for x in native["evaluations"][stage]["frames"]}
        for tid in ids:
            assert rows[tid]["input_evidence"] == evidence[tid]
            bound[(state,tid)] = prediction(native_run,rows[tid])
    audit = dict(status="PASS",result_sha256=sha(a.run/"results.json"),model_forwards=118,
                 new_quality_predictions=112,new_action_labels=96,background_replays=16,
                 engineering_dense_references=6,optimizer_steps=0,E0_reference_predictions=32,
                 native_reference_predictions=32,source_arrays=48,source_binding_before_GT_values=True,
                 E0_audit_sha256=sha(a.e0_audit),endpoint=ep,action_contract_sha256=sha(a.contract),
                 limitation="Intermediate parity values are runtime receipts, not separately saved offline arrays.")
    return r,roles,regions,raw_sources,bound,layouts,audit


def average_errors(metrics):
    out = {}
    for region in mc.REGIONS:
        n = metrics[0][region]["valid_pixels"]
        assert all(m[region]["valid_pixels"] == n for m in metrics)
        err = math.fsum(m[region]["error_sum"] for m in metrics)/len(metrics)
        out[region] = dict(valid_pixels=n,error_sum=err,AEE=err/n if n else None)
    return out


def policy_analysis(frames):
    seqs = list(dict.fromkeys(f["sequence"] for f in frames))
    assert len(frames) == 16 and len(seqs) == 8 and all(sum(f["sequence"] == s for f in frames) == 2 for s in seqs)
    error = lambda f,s: f["states"][s]["all_valid"]["error_sum"]
    fit = {seq:min(UPGRADES,key=lambda s:(math.fsum(error(f,s) for f in frames if f["sequence"] != seq),s)) for seq in seqs}
    glob = min(UPGRADES,key=lambda s:(math.fsum(error(f,s) for f in frames),s))
    choices = {}
    for f in frames:
        chosen = dict(fit_fold_static=fit[f["sequence"]],
                      single_action_oracle=min(UPGRADES,key=lambda s:(error(f,s),s)),
                      in_sample_static_diagnostic=glob)
        choices[f["target_id"]] = chosen
        for policy,state in chosen.items():
            f["states"][policy] = f["states"][state]
        f["states"]["uniform_upgrade_expectation"] = average_errors([f["states"][s] for s in UPGRADES])
    return choices,fit,glob


def uniform_tail_risk(frames,reference):
    """Exact action-frequency risk; average errors alone can conceal rare tails."""
    output = []
    for f in frames:
        probabilities, any_actions = {}, set()
        for region in ("all_valid","B","L"):
            ref = f["states"][reference][region]
            bad = []
            for s in UPGRADES:
                c = f["states"][s][region]
                assert c["valid_pixels"] == ref["valid_pixels"]
                if ref["valid_pixels"] >= 256 and (c["AEE"] > ref["AEE"]*1.2 or c["AEE"]-ref["AEE"] > 1):
                    bad.append(s); any_actions.add(s)
            probabilities[region] = dict(valid_pixels=ref["valid_pixels"],bad_actions=bad,probability=len(bad)/6)
        output.append(dict(target_id=f["target_id"],sequence=f["sequence"],regions=probabilities,
                           probability_any_main_region_tail=len(any_actions)/6))
    return dict(reference=reference,frames=output,
                expected_flagged_frame_region_cells=math.fsum(v["probability"] for f in output for v in f["regions"].values()),
                expected_frames_with_any_main_region_tail=math.fsum(f["probability_any_main_region_tail"] for f in output),
                definition="Uniform choice among six actions. Flag: support>=256 and (>20% or >1px injury). This is exact frequency over actions, not thresholding averaged predictions or averaged errors.")


def coverage(arrays,masks,layouts):
    valid,events = arrays["mask"],arrays["events"]
    bins = np.count_nonzero(events,axis=0); active = bins>0
    name = sorted(ew.CONVS)[0]
    out = {}
    for state in UPGRADES:
        footprint = np.repeat(np.repeat(layouts[state][0][name],2,axis=0),2,axis=1)
        n,v = int(footprint.sum()),int((footprint&valid).sum())
        out[state] = dict(image_pixels=n,valid_GT_pixels=v,valid_GT_fraction_of_selected=v/n,
            valid_GT_fraction_of_frame=v/int(valid.sum()),
            voxel_active_pixels=int((footprint&active).sum()),voxel_zero_pixels=int((footprint&~active).sum()),
            nonzero_voxel_bins=int(bins[footprint].sum()),
            valid_GT_voxel_active_pixels=int((footprint&valid&active).sum()),
            valid_GT_voxel_zero_pixels=int((footprint&valid&~active).sum()),
            valid_GT_nonzero_voxel_bins=int(bins[footprint&valid].sum()),
            region_support={k:int((footprint&m).sum()) for k,m in masks.items()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("run","e0-run","e0-audit","contract","output"):
        ap.add_argument("--"+name,type=Path,required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    r,roles,regions,sources,bound,layouts,audit = bind(a)
    frames,covered = [],{}
    for tid in roles["e2_reserved_ids"]:
        arrays = {k:np.load(p,allow_pickle=False) for k,p in sources[tid].items()}
        masks = raw.region_masks(arrays["gt"],arrays["mask"],arrays["events"],
                                 regions["boundary_threshold_pixels"],regions["large_displacement_threshold_pixels"])
        measured = {}
        for state in ACTUAL+REFS:
            with np.load(bound[(state,tid)],allow_pickle=False) as z:
                measured[state] = raw.measure(z["head3"],arrays["gt"],masks)
        frames.append(dict(target_id=tid,sequence=tid.rsplit("_",1)[0],states=measured))
        covered[tid] = coverage(arrays,masks,layouts)
    choices,fit,glob = policy_analysis(frames)
    policies = ["fit_fold_static","single_action_oracle","in_sample_static_diagnostic","uniform_upgrade_expectation"]
    states = ACTUAL+REFS+policies
    seqs = list(dict.fromkeys(f["sequence"] for f in frames))
    summaries = {s:{scope:{reg:mc.aggregate([f["states"][s][reg] for f in frames if scope=="overall" or f["sequence"]==scope])
                           for reg in mc.REGIONS} for scope in ["overall"]+seqs} for s in states}
    pairs = [(c,ref) for c in UPGRADES+policies for ref in ["background_Q4"]+REFS]
    pairs += [("single_action_oracle","fit_fold_static"),("single_action_oracle","uniform_upgrade_expectation"),
              ("fit_fold_static","uniform_upgrade_expectation")]
    pairs += [("background_Q4",ref) for ref in REFS]
    comparisons = {c+"__vs__"+ref:mc.comparison(frames,summaries,c,ref,seqs) for c,ref in pairs}
    output = dict(schema="flow_e3_rne_independent_raw_quality_v1",status="COMPLETE",execution_review=audit,
                  states=states,frames=frames,summaries=summaries,comparisons=comparisons,
                  policy_choices=choices,fit_fold_static=fit,in_sample_static_diagnostic=glob,
                  uniform_action_tail_risk={ref:uniform_tail_risk(frames,ref) for ref in ["background_Q4"]+REFS},
                  action_coverage=covered,
                  selected_action_coverage={tid:{p:covered[tid][s] for p,s in c.items()} for tid,c in choices.items()},
                  selection_objective="All-valid pixel-pooled AEE only, one common action for all16 regions; lexicographic tie break. Fit static excludes target sequence.",
                  uniform_definition="Exact average of six actual error sums/AEE, never average flow vectors; action tail probabilities separately reported.",
                  oracle_definition="Six same-core-cost upgrades only; background excluded from oracle and separately shown as cheaper Pareto point.",
                  scope="All16 are exposed training-domain development. E0 evaluated these identities; LOSO is head development only, no independent pipeline confirmation.",
                  background_definition="E0 Q4 means all28 complete tiles use RNE4 while true tails remain Q8; it is not single-representation all-rows Q4.",
                  known_tail_scope="E0 Q4 high-support L tail zurich_city_06_a_0363 is outside these fixed16 identities. This probe cannot demonstrate repair of that tail.",
                  unexcluded_ordinary_alternatives=["True single-representation all-rows Q4 has no matched quality result yet.",
                                                   "Static154-tile neighbor is only an arithmetic budget candidate under the current conservative controller; actual service and quality are unmeasured and the budget must be recomputed for the final controller."],
                  coverage_definition="2x geometric footprint of patch output tiles, not downstream causal support; voxel bins are not raw event counts.",
                  selector_learning_demonstrated=False,pair_effects_tested=False,
                  cost=dict(added_core_ticks_vs_Q4=82192320,fraction_native=82192320/3652955059,
                            sign="additional expense to buy precision, not savings",net_cycles=False,
                            supply_control_selector_costs_deducted=False,
                            control_scope="Measured conservative span/group comparisons are not a universal unavoidable dynamic-control lower bound. Precomputed masks/base-pointer alternatives need their own storage/port/service evidence.",
                            reference_costs="E0 Q8/M are differing-cost ordinary Pareto neighbors; no equal-net-cost claim."),
                  legacy_lines="3%/5% legacy references only; no new compulsory threshold.",
                  reducer_sha256=sha(__file__),raw_math_sha256=sha(raw.__file__),work_math_sha256=sha(ew.__file__))
    a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    table = {s:{reg:summaries[s]["overall"][reg]["pixel_weighted_AEE"] for reg in ("all_valid","B","L")}
             for s in ["background_Q4","endpoint_M","endpoint_Q8"]+policies}
    print(json.dumps(dict(status="COMPLETE",output=str(a.output),pixel_AEE=table),indent=2))


if __name__=="__main__":
    main()
