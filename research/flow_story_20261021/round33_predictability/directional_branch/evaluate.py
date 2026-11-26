#!/usr/bin/env python3
"""R33 frozen-choice CPU evaluator. Never fits heads or launches a flow model.

Bind every input byte before reading numerical labels. Exact old scalar
aggregation is reused; old gates are NOT imported as R33 pass/fail criteria.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
BRANCH = Path(__file__).resolve().parent
HERE = BRANCH.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(BRANCH))
import predictors as models

CORE = ROOT/"research/flow_story_20261019/round27/independent_review/analysis_core.py"
spec = importlib.util.spec_from_file_location("r33_old_scalar_math", CORE)
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)
REGIONS = tuple(mc.REGIONS)
STATES = ["background_Q4"]+[f"upgrade_{i}" for i in range(6)]
MEMBERSHIP_SHA = "d860612a3ae3e4620fa2a9a616df9f8e93774efadcf7dc564881e27b7e7da8cc"
SOURCE_FREEZE_SHA = "fd69f637525925213725f0a9a2545531806790c869bb6776c19bc57b810e6d31"
LABEL_PREDICTOR_SHA = "35663dd6b999621652d9d185d72b967e33c380e1064a09ec75540d2759cd61c3"
LABEL_FREEZE_SHA = "5c6c47334c700b9a6f109e7cb768942905563d898312cfe69b0ffb3d57007f54"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite {path}")
    path.write_text(json.dumps(models.jsonable(data), indent=2, allow_nan=False)+"\n")


def source_record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def bind(args):
    # No NPZ array or quality JSON is decoded in this operation.
    paths = dict(pack=args.pack_dir/(args.role+".npz"), quality=args.pack_dir/"QUALITY_V1.json",
        pack_freeze=args.pack_dir/"PACK_FREEZE_V1.json", predictions=args.predictions,
        model=args.model, fit_receipt=args.model.parent/"FIT_RECEIPT_V1.json",
        membership=HERE/"data/MEMBERSHIP_V1.json", hardware=args.hardware,
        algorithm_freeze=BRANCH/"SOURCE_FREEZE_V1.json", predictor=Path(models.__file__),
        predictor_runner=BRANCH/"run_predictors.py", scalar_math=CORE,
        label_predictor=HERE/"algorithm/predictors.py", label_algorithm_freeze=HERE/"algorithm/SOURCE_FREEZE_V1.json",
        augmentation_source=HERE/"augment_directional_pack.py",
        augmentation_receipt=args.pack_dir/"DIRECTIONAL_AUGMENT_RECEIPT_V1.json",
        original_pack=args.original_pack_dir/(args.role+".npz"),
        original_quality=args.original_pack_dir/"QUALITY_V1.json",
        original_pack_freeze=args.original_pack_dir/"PACK_FREEZE_V1.json",
        evaluator=Path(__file__))
    if args.heads_freeze is not None:
        paths["heads_freeze"] = args.heads_freeze
    require(args.role != "confirm" or "heads_freeze" in paths,
            "confirmation evaluation requires the prior heads freeze")
    write_new(args.output, dict(schema="flow_r33_directional_evaluation_bindings_v1", role=args.role,
        files={k: source_record(p) for k,p in paths.items()},
        numerical_quality_read=False, real_data_fitting=False, GPU_used=False))
    print(json.dumps(dict(status="INPUT_BYTES_BOUND", bindings_sha256=sha(args.output))))


def check_inputs(binding_path):
    b = read(binding_path)
    require(b["schema"] == "flow_r33_directional_evaluation_bindings_v1", "binding schema")
    require(b["role"] in ("tune", "confirm"), "evaluation role")
    for key, item in b["files"].items():
        path = Path(item["path"])
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                f"changed bound file {key}")
    files = {k: Path(v["path"]) for k,v in b["files"].items()}
    require(sha(files["evaluator"]) == sha(__file__), "evaluator source binding")
    require(sha(files["predictor"]) == sha(models.__file__), "predictor import binding")
    require(sha(files["scalar_math"]) == sha(CORE), "scalar import binding")
    require(sha(files["membership"]) == MEMBERSHIP_SHA, "membership binding")
    require(sha(files["algorithm_freeze"]) == SOURCE_FREEZE_SHA, "algorithm freeze binding")
    freeze = read(files["algorithm_freeze"])
    require(sha(files["predictor"]) == freeze["files"]["predictors.py"], "frozen predictor")
    require(sha(files["predictor_runner"]) == freeze["files"]["run_predictors.py"], "frozen runner")
    pack_freeze = read(files["pack_freeze"])
    require(pack_freeze["schema"] == "flow_r33_pack_freeze_v1", "pack freeze schema")
    require(pack_freeze["files"][files["pack"].name] == sha(files["pack"]), "pack SHA")
    require(pack_freeze["files"]["QUALITY_V1.json"] == sha(files["quality"]), "quality SHA")
    require(pack_freeze["model_forwards"] == 0 and not pack_freeze["GPU_used"], "CPU pack provenance")
    require(sha(files["label_predictor"]) == LABEL_PREDICTOR_SHA and
            sha(files["label_algorithm_freeze"]) == LABEL_FREEZE_SHA, "original label source binding")
    original_freeze = read(files["original_pack_freeze"])
    require(original_freeze["schema"] == "flow_r33_pack_freeze_v1", "original pack schema")
    require(original_freeze["files"][files["original_pack"].name] == sha(files["original_pack"]), "original NPZ SHA")
    require(original_freeze["files"]["QUALITY_V1.json"] == sha(files["original_quality"]), "original quality SHA")
    require(sha(files["quality"]) == sha(files["original_quality"]), "QUALITY original bytes preserved")
    require(pack_freeze["files"]["DIRECTIONAL_AUGMENT_RECEIPT_V1.json"] == sha(files["augmentation_receipt"]),
            "augmentation receipt pack binding")
    augmentation = read(files["augmentation_receipt"])
    require(augmentation["schema"] == "flow_r33_directional_pack_v1", "augmentation schema")
    require(augmentation["predictors_sha256"] == sha(files["predictor"]) and
            augmentation["runner_sha256"] == sha(files["augmentation_source"]), "new feature source binding")
    require(augmentation["membership_sha256"] == MEMBERSHIP_SHA, "augmentation membership")
    require(augmentation["original_pack_freeze_sha256"] == sha(files["original_pack_freeze"]), "original freeze link")
    require(augmentation["original_quality_json_sha256"] == sha(files["quality"]), "original quality link")
    require(augmentation["labels_unchanged"] and not augmentation["GPU_used"] and augmentation["model_forwards"] == 0,
            "augmentation only changes CPU input features")
    require(augmentation["phase"] == original_freeze["phase"] == pack_freeze["phase"], "dual pack phases")
    model = models.load_model(files["model"])
    receipt = read(files["fit_receipt"])
    require(receipt["schema"] == "flow_r33_directional_fit_receipt_v1", "fit receipt schema")
    require(receipt["model_sha256"] == sha(files["model"]), "fit receipt model SHA")
    require(receipt["membership_sha256"] == model["metadata"]["membership_sha256"] == MEMBERSHIP_SHA,
            "fit membership binding")
    require(receipt["fit_pack_sha256"] == model["metadata"]["fit_pack_sha256"], "fit pack binding")
    require(receipt["fit_ids"] == model["metadata"]["fit_ids"], "fit identities binding")
    require(receipt["alpha"] == model["alpha"] == .1 and not receipt["tune_or_confirm_read"],
            "fixed fit-only training")
    require(receipt["predictor_source_sha256"] == model["metadata"]["predictor_source_sha256"] == sha(files["predictor"]),
            "fit predictor source")
    require(receipt["runner_source_sha256"] == model["metadata"]["runner_source_sha256"] == sha(files["predictor_runner"]),
            "fit runner source")
    members = read(files["membership"])
    role_rows = [r for r in members["records"] if r["role"] == b["role"]]
    fit_ids = [r["target_id"] for r in members["records"] if r["role"] == "fit"]
    require(len(receipt["fit_ids"]) == 64 and set(receipt["fit_ids"]) == set(fit_ids), "entire64 fit")
    if b["role"] == "confirm":
        require("heads_freeze" in files, "missing confirmation heads freeze")
        heads = read(files["heads_freeze"])
        require(heads["schema"] == "flow_r33_heads_freeze_v1", "heads freeze schema")
        require(heads["membership_sha256"] == MEMBERSHIP_SHA, "heads membership SHA")
        require(heads["model_sha256"] == sha(files["model"]), "heads model SHA")
        require(heads["fit_receipt_sha256"] == sha(files["fit_receipt"]), "heads receipt SHA")
        require(heads["fit_pack_sha256"] == receipt["fit_pack_sha256"], "heads fit pack SHA")
        for role in ("fit", "tune", "confirm"):
            expected = [r["target_id"] for r in members["records"] if r["role"] == role]
            require(len(heads[role+"_ids"]) == len(expected) and set(heads[role+"_ids"]) == set(expected),
                    "heads role identities")
    prediction = read(files["predictions"])
    require(prediction["schema"] == "flow_r33_directional_predicted_choices_v1" and prediction["role"] == b["role"],
            "prediction role/schema")
    require(prediction["model_sha256"] == sha(files["model"]), "predicted model binding")
    require(prediction["membership_sha256"] == MEMBERSHIP_SHA, "predicted membership")
    require(prediction["input_pack_sha256"] == sha(files["pack"]), "predicted pack binding")
    require(not prediction["GT_or_counterfactual_fields_read"], "runtime input information")
    with np.load(files["pack"], allow_pickle=False) as z:
        pack = {k: z[k].copy() for k in z.files}
    with np.load(files["original_pack"], allow_pickle=False) as z:
        original_pack = {k: z[k].copy() for k in z.files}
    validate_augmented_arrays(pack, original_pack)
    checked = augmentation["source_checks"]
    by_target = {r["target_id"]: r for r in members["records"]}
    checked_ids = [r["target_id"] for r in checked]
    phase_roles = {"fit","tune"} if augmentation["phase"] == "fit_tune" else {"confirm"}
    expected_checked = {r["target_id"] for r in members["records"] if r["role"] in phase_roles}
    require(len(checked_ids) == len(set(checked_ids)) == len(expected_checked) and set(checked_ids) == expected_checked,
            "augmentation checked complete phase identities")
    for row in checked:
        source = next(f for f in by_target[row["target_id"]]["source_files"] if f["category"] == "input")
        require(row["input_sha256"] == source["sha256"] and row["first128_bitwise_equal"], "augmentation input/prefix proof")
    ids = pack["ids"].astype(str).tolist()
    groups = pack["groups"].astype(str).tolist()
    require(len(ids) == len(set(ids)) == len(role_rows), "unique role count")
    expected = {r["target_id"]: r["sequence"] for r in role_rows}
    require(set(ids) == set(expected) and all(expected[i] == g for i,g in zip(ids,groups)), "pack roles")
    require(set(pack["roles"].astype(str)) == {b["role"]}, "single-role pack only")
    require(ids == prediction["ids"] and groups == prediction["groups"], "pack/prediction row order")
    require(not set(ids) & set(model["metadata"]["fit_ids"]), "fit/evaluation overlap")
    require(prediction["features_sha256"] == models.digest_arrays(pack["x"]), "input feature binding")
    # Verify stored decisions from frozen model and X without using GT labels.
    replay = models.predict_suite(model, pack["x"])
    for key in ("scores", "predicted_auxiliary", "fixed_six_choices", "allow_Q4_choices"):
        require(set(replay[key]) == set(prediction["results"][key]), f"predicted {key} methods")
        for name in replay[key]:
            require(np.array_equal(replay[key][name], np.asarray(prediction["results"][key][name])),
                    f"frozen inference replay {key}/{name}")
    for key, meta in (("fit_static_six_choices", "fit_static_six_action"),
                      ("fit_static_allow_Q4_choices", "fit_static_allow_Q4_action")):
        require(prediction["results"][key] == [model["metadata"][meta]]*len(ids), "fit static binding")
    quality = read(files["quality"])
    require(quality["schema"] == "flow_r33_source_bound_raw_quality_v1" and quality["status"] == "COMPLETE",
            "source raw quality status")
    require(quality["membership_sha256"] == MEMBERSHIP_SHA and quality["predictors_sha256"] == LABEL_PREDICTOR_SHA,
            "unchanged quality uses original128 labelmaker, not new160 feature source")
    require(quality["scalar_math_sha256"] == sha(CORE), "quality scalar binding")
    require(quality["states"] == STATES, "quality seven state order")
    expected_phase = "fit_tune" if b["role"] == "tune" else "confirm"
    require(quality["phase"] == pack_freeze["phase"] == expected_phase, "raw/pack phase")
    role_quality_rows = [r for r in quality["frames"] if r["role"] == b["role"]]
    quality_rows = {r["target_id"]: r for r in role_quality_rows}
    require(len(role_quality_rows) == len(quality_rows) == len(ids) and set(quality_rows) == set(ids), "raw role identities")
    frames = [quality_rows[i] for i in ids]
    validate_raw_pack(frames, pack)
    hardware = read(files["hardware"])
    require(hardware["schema"] == "round33_bitmap_control_result_v1", "hardware schema")
    require(not hardware["whole_network"] and not hardware["predictor_execution_included"] and
            not hardware["feature_reads_included"], "operator-only hardware scope")
    require(sorted(c["action"] for c in hardware["candidates"]) == list(range(7)), "hardware action ids")
    return b, model, prediction, pack, frames, hardware


def validate_augmented_arrays(pack, original_pack):
    require(set(pack) == set(original_pack), "unchanged label/metadata array keys")
    for key in pack:
        if key == "x":
            require(pack[key].shape == (len(pack[key]),160) and original_pack[key].shape == (len(pack[key]),128),
                    "base128/new160 feature dimensions")
            require(pack[key].dtype == original_pack[key].dtype and
                    np.ascontiguousarray(pack[key][:,:128]).tobytes() == np.ascontiguousarray(original_pack[key]).tobytes(),
                    "base128 prefix bytes preserved")
        else:
            require(pack[key].dtype == original_pack[key].dtype and pack[key].shape == original_pack[key].shape and
                    np.ascontiguousarray(pack[key]).tobytes() == np.ascontiguousarray(original_pack[key]).tobytes(),
                    "all non-feature label/metadata bytes preserved: "+key)


def validate_raw_pack(frames, pack):
    n = len(frames)
    require(pack["x"].shape == (n,160) and np.all(np.isfinite(pack["x"])), "finite common inputs")
    require(pack["delta_aee"].shape == (n,6), "six true delta labels")
    for i, f in enumerate(frames):
        require(set(f["states"]) == set(STATES), "raw seven states")
        require(f["sequence"] == str(pack["groups"][i]), "raw sequence")
        for state in STATES:
            require(set(f["states"][state]) == set(REGIONS), "all16 regions required")
            for reg in REGIONS:
                v = f["states"][state][reg]
                mc.validate(v)
                require(v["valid_pixels"] == f["states"]["background_Q4"][reg]["valid_pixels"],
                        "identical per-frame GT support")
        base = f["states"]["background_Q4"]["all_valid"]
        require(base["valid_pixels"] > 0 and base["valid_pixels"] == int(pack["valid_pixels"][i]), "allGT count")
        require(math.isclose(base["AEE"], float(pack["background_AEE"][i]), rel_tol=1e-12, abs_tol=1e-10), "Q4 label AEE")
        truth = [f["states"][s]["all_valid"]["AEE"]-base["AEE"] for s in STATES[1:]]
        require(np.allclose(truth, pack["delta_aee"][i], rtol=1e-12, atol=1e-10), "raw true delta labels")
    for grid in (1,4):
        aux = pack[f"aux_g{grid}"]
        require(aux.shape == (n,14*grid*grid) and np.all(np.isfinite(aux)), "finite auxiliary labels")
        count = pack[f"support_counts_g{grid}"]
        require(count.shape == (n,grid,grid) and np.all(np.isfinite(count)) and
                np.all(count >= 0) and np.all(count == np.floor(count)), "GT support grid")
        require(np.array_equal(count.sum(axis=(1,2)), pack["valid_pixels"]), "GT support grid totals")
        v,d = models.split_aux(aux,grid)
        vf = v.reshape(n,2,grid,grid)
        limit = count/pack["valid_pixels"][:,None,None]
        require(np.all(np.linalg.norm(vf,axis=1) <= limit+1e-12), "weighted direction physical bound")
        df = d.reshape(n,6,2,grid,grid)
        require(np.all(df*np.asarray(count == 0)[:,None,None] == 0), "unsupported response cells are zero")
        require(np.allclose(models.signed_scores(aux,grid),pack[f"ideal_signed_g{grid}"],rtol=1e-12,atol=1e-12),
                "ideal signed label readout")


def policy_weights(choices):
    choices = np.asarray(choices, dtype=np.int64)
    require(np.all((choices >= -1) & (choices < 6)), "action index range")
    result = np.zeros((len(choices),7),dtype=np.float64)
    result[np.arange(len(choices)),choices+1] = 1
    return result


def build_policies(pack, prediction, model):
    n = len(pack["ids"])
    out = {}
    def deterministic(name, choices, observable, regime, control="dynamic", label_source=None):
        out[name] = dict(weights=policy_weights(choices), choices=np.asarray(choices,dtype=np.int64),
            observable=observable, regime=regime, operator_control=control, label_source=label_source,
            stochastic_expectation=False)
    def scores_pair(name, scores, observable, label_source=None):
        deterministic(name+"__six", models.choose_actions(scores), observable, "fixed_six", label_source=label_source)
        deterministic(name+"__allow_Q4", models.choose_actions(scores,True), observable, "allow_Q4", label_source=label_source)
    for method in models.METHODS:
        scores_pair(method,np.asarray(prediction["results"]["scores"][method]),"runtime_voxel_only")
    deterministic("Q4",np.full(n,-1),"constant","Q4_only","static")
    deterministic("fit_static__six",prediction["results"]["fit_static_six_choices"],"fit_labels_then_constant","fixed_six","static")
    deterministic("fit_static__allow_Q4",prediction["results"]["fit_static_allow_Q4_choices"],"fit_labels_then_constant","allow_Q4","static")
    out["uniform_six_expectation"] = dict(weights=np.tile(np.r_[0,np.full(6,1/6)],(n,1)),choices=None,
        observable="input_independent_expected_random_rule",regime="fixed_six",operator_control="dynamic",
        label_source=None,stochastic_expectation=True)
    scores_pair("oracle",pack["delta_aee"],"offline_GT_and_six_counterfactuals","true_allGT_delta_AEE")
    for grid in (1,4):
        true = pack[f"aux_g{grid}"]
        predicted = np.asarray(prediction["results"]["predicted_auxiliary"][f"g{grid}"])
        replace = models.offline_replacement_scores(predicted,true,grid)
        scores_pair(f"ideal_signed_g{grid}",replace["all_true"],"offline_GT_and_six_counterfactuals","true_weighted_v_and_true_d")
        scores_pair(f"offline_predv_trued_g{grid}",replace["predicted_weighted_direction_true_response"],
                    "offline_true_response","predicted_support_weighted_v_true_d")
        scores_pair(f"offline_truev_predd_g{grid}",replace["true_weighted_direction_predicted_response"],
                    "offline_GT_weighted_direction","true_support_weighted_v_predicted_d")
    return out


def mixed_metrics(frame, weights):
    out = {}
    for region in REGIONS:
        n = frame["states"][STATES[0]][region]["valid_pixels"]
        error = math.fsum(float(w)*frame["states"][s][region]["error_sum"] for s,w in zip(STATES,weights))
        out[region] = dict(valid_pixels=n,error_sum=error,AEE=error/n if n else None)
    return out


def operator_cost(weights, control, hardware):
    by = {v["action"]:v for v in hardware["candidates"]}
    # Hardware action6 is Q4; algorithm -1/Q4 occupies weights column0.
    column_actions = [6,0,1,2,3,4,5]
    key = "static_affine_BN_ticks" if control == "static" else "dynamic_affine_BN_ticks"
    ticks = np.asarray([4*by[a][key] for a in column_actions],dtype=np.float64)
    mean_each = weights@ticks
    # Uniform expected cost and worst possible action cost are different.
    worst_each = np.asarray([max(ticks[w > 0]) for w in weights])
    return dict(per_identity_expected_ticks=mean_each,per_identity_max_possible_ticks=worst_each,
                four_operator_ticks_by_column=ticks)


def summarize_policies(frames,pack,policies,hardware):
    ids = pack["ids"].astype(str).tolist(); seqs = list(dict.fromkeys(pack["groups"].astype(str)))
    scoped = ["overall"]+seqs
    all_frames = [dict(target_id=i,sequence=str(g),states={}) for i,g in zip(ids,pack["groups"])]
    summary, decision, per_identity = {},{},[]
    base = np.asarray([f["states"]["background_Q4"]["all_valid"]["AEE"] for f in frames])
    true_delta = np.asarray(pack["delta_aee"],dtype=np.float64)
    all_delta = np.c_[np.zeros(len(frames)),true_delta]
    oracle6 = true_delta.min(axis=1); oracle7 = all_delta.min(axis=1)
    details = {i:{} for i in ids}
    for name,policy in policies.items():
        weights=policy["weights"]
        require(weights.shape==(len(frames),7) and np.all(weights>=0) and np.allclose(weights.sum(axis=1),1),"policy probabilities")
        if policy["regime"]=="fixed_six":require(np.all(weights[:,0]==0),"Q4 entered fixed six budget")
        for i,frame in enumerate(frames):
            all_frames[i]["states"][name]=mixed_metrics(frame,weights[i])
        chosen_delta=np.sum(weights*all_delta,axis=1)
        reg6=chosen_delta-oracle6;reg7=chosen_delta-oracle7
        require(np.all(reg7>=-1e-10),"seven-action regret nonnegative")
        if policy["regime"]=="fixed_six":require(np.all(reg6>=-1e-10),"six-action regret nonnegative")
        cost=operator_cost(weights,policy["operator_control"],hardware)
        summary[name]={};decision[name]={}
        for scope in scoped:
            mask=np.asarray([scope=="overall" or f["sequence"]==scope for f in all_frames])
            selected=[f for f,m in zip(all_frames,mask) if m]
            summary[name][scope]={r:mc.aggregate([f["states"][name][r] for f in selected]) for r in REGIONS}
            support=np.asarray(pack["valid_pixels"])[mask]
            upgrade=1-weights[mask,0]
            decision[name][scope]=dict(identity_count=int(mask.sum()),upgrade_fraction=float(upgrade.mean()),
                action_histogram_expected={str(a):float(weights[mask,j].sum()) for j,a in enumerate(range(-1,6))},
                histogram_is_expectation=policy["stochastic_expectation"],
                improved_Q4_identities=int(np.sum(chosen_delta[mask]<0)),
                worsened_Q4_identities=int(np.sum(chosen_delta[mask]>0)),
                unchanged_Q4_identities=int(np.sum(chosen_delta[mask]==0)),
                regret_six_action_pixel_weighted=float(np.average(reg6[mask],weights=support)),
                regret_six_action_frame_mean=float(reg6[mask].mean()),
                regret_seven_action_pixel_weighted=float(np.average(reg7[mask],weights=support)),
                regret_seven_action_frame_mean=float(reg7[mask].mean()),
                six_action_regret_can_be_negative=policy["regime"]!="fixed_six",
                operator_only_service=dict(mean_four_conv_ticks=float(cost["per_identity_expected_ticks"][mask].mean()),
                    max_expected_four_conv_ticks=float(cost["per_identity_expected_ticks"][mask].max()),
                    max_possible_four_conv_ticks=float(cost["per_identity_max_possible_ticks"][mask].max()),
                    summed_expected_four_conv_ticks=float(cost["per_identity_expected_ticks"][mask].sum()),
                    control=policy["operator_control"],feature_and_head_included=False,whole_network=False))
        for i,tid in enumerate(ids):
            details[tid][name]=dict(action=None if policy["choices"] is None else int(policy["choices"][i]),
                action_probabilities=weights[i].tolist(),upgrade_probability=float(1-weights[i,0]),
                selected_actual_delta_AEE=float(chosen_delta[i]),regret_six_action_AEE=float(reg6[i]),
                regret_seven_action_AEE=float(reg7[i]),four_operator_expected_ticks=float(cost["per_identity_expected_ticks"][i]),
                regions=all_frames[i]["states"][name])
    for i,tid in enumerate(ids):
        per_identity.append(dict(target_id=tid,sequence=str(pack["groups"][i]),
            valid_pixels=int(pack["valid_pixels"][i]),policies=details[tid]))
    return all_frames,summary,decision,per_identity,seqs


def comparison(frames,summaries,candidate,reference,seqs):
    """Descriptive quality comparison; no inherited aggregate_pass field."""
    out=dict(candidate=candidate,reference=reference,summary_differences={},
             aggregate_2pct_review={},high_support_tail_review=[],frames=[],support_strata={},
             review_flags_are_not_pass_fail=True)
    if candidate == "uniform_six_expectation":
        out["risk_scope_note"] = ("These flags apply to expected AEE, not tail risk of a realized random draw; "
                                  "see separately reported uniform_six_tail_risk from all six action outcomes.")
    for scope in ["overall"]+seqs:
        out["summary_differences"][scope]={}
        for region in REGIONS:
            c,r=summaries[candidate][scope][region],summaries[reference][scope][region]
            require(c["valid_pixels"]==r["valid_pixels"],"comparison support")
            row=dict(valid_pixels=r["valid_pixels"])
            for metric in ("pixel_weighted_AEE","frame_mean_AEE"):
                row[metric]=dict(candidate=c[metric],reference=r[metric],
                    signed_delta_AEE=mc.difference(c[metric],r[metric]),relative_change=mc.ratio(c[metric],r[metric]))
            out["summary_differences"][scope][region]=row
    for key,region,metric in (("allGT_pixel","all_valid","pixel_weighted_AEE"),
                              ("allGT_frame","all_valid","frame_mean_AEE"),
                              ("B_pixel","B","pixel_weighted_AEE"),("L_pixel","L","pixel_weighted_AEE")):
        row=out["summary_differences"]["overall"][region][metric]
        a,b=row["candidate"],row["reference"]
        out["aggregate_2pct_review"][key]=dict(**row,threshold=.02,
            valid_pixels=out["summary_differences"]["overall"][region]["valid_pixels"],
            positive_support=a is not None and b is not None,
            review_flag=bool(a is not None and b is not None and a>b*1.02),
            B_L_protection_reference=region in ("B","L"),not_a_pass_fail_gate=True)
    for frame in frames:
        rows={}
        for region in ("all_valid","B","L"):
            c,r=frame["states"][candidate][region],frame["states"][reference][region]
            n=r["valid_pixels"];delta=mc.difference(c["AEE"],r["AEE"])
            flag=bool(n>=256 and (c["AEE"]>r["AEE"]*1.2 or delta>1.0))
            legacy=.03 if region=="all_valid" else .05
            row=dict(valid_pixels=n,support_band=mc.band(n),candidate_AEE=c["AEE"],reference_AEE=r["AEE"],
                signed_delta_AEE=delta,relative_change=mc.ratio(c["AEE"],r["AEE"]),
                legacy_relative_reference=legacy,
                legacy_relative_reference_exceeded=bool(n and c["AEE"]>r["AEE"]*(1+legacy)),
                high_support_tail_review_flag=flag,not_a_hard_gate=True)
            rows[region]=row
            if flag:out["high_support_tail_review"].append(dict(target_id=frame["target_id"],sequence=frame["sequence"],region=region,**row))
        out["frames"].append(dict(target_id=frame["target_id"],sequence=frame["sequence"],regions=rows))
    for region in REGIONS:
        out["support_strata"][region]={}
        for band in mc.BANDS:
            selected=[f for f in frames if mc.band(f["states"][reference][region]["valid_pixels"])==band]
            c=[f["states"][candidate][region] for f in selected];r=[f["states"][reference][region] for f in selected]
            out["support_strata"][region][band]=dict(frame_count=len(selected),candidate=mc.aggregate(c),reference=mc.aggregate(r),
                signed_delta_AEE=mc.quantiles([mc.difference(a["AEE"],b["AEE"]) for a,b in zip(c,r)]),
                relative_change=mc.quantiles([mc.ratio(a["AEE"],b["AEE"]) for a,b in zip(c,r)]))
    # Sequence evidence is descriptive, with no pseudo-independent action rows.
    changes=[out["summary_differences"][s]["all_valid"]["pixel_weighted_AEE"]["signed_delta_AEE"] for s in seqs]
    out["sequence_gain_counts"]=dict(improved=sum(d<0 for d in changes if d is not None),
        worsened=sum(d>0 for d in changes if d is not None),unchanged=sum(d==0 for d in changes if d is not None),
        total=len(seqs))
    return out


def uniform_tail_risk(raw_frames):
    """Exact marginal bad-tail probabilities, not flags on mean predictions.

    Within each identity the same chosen action determines all region outcomes,
    so any-region probability uses a union of action ids. No independence among
    regions or among identities is assumed. Expected counts sum marginals.
    """
    primary=("all_valid","B","L")
    rows=[]
    for frame in raw_frames:
        regions={};any_bad=set()
        for region in primary:
            ref=frame["states"]["background_Q4"][region]
            bad=[]
            for action in range(6):
                cand=frame["states"][f"upgrade_{action}"][region]
                if ref["valid_pixels"]>=256 and (cand["AEE"]>ref["AEE"]*1.2 or cand["AEE"]-ref["AEE"]>1.0):
                    bad.append(action)
            any_bad.update(bad)
            regions[region]=dict(valid_pixels=ref["valid_pixels"],high_support=ref["valid_pixels"]>=256,
                triggering_actions=bad,trigger_probability=len(bad)/6)
        rows.append(dict(target_id=frame["target_id"],sequence=frame["sequence"],regions=regions,
                         any_primary_region_triggering_actions=sorted(any_bad),
                         any_primary_region_trigger_probability=len(any_bad)/6))
    seqs=list(dict.fromkeys(f["sequence"] for f in raw_frames));summary={}
    for scope in ["overall"]+seqs:
        selected=[r for r in rows if scope=="overall" or r["sequence"]==scope]
        summary[scope]=dict(identity_count=len(selected),regions={},
            expected_any_primary_region_bad_frame_count=math.fsum(r["any_primary_region_trigger_probability"] for r in selected),
            mean_any_primary_region_trigger_probability=math.fsum(r["any_primary_region_trigger_probability"] for r in selected)/len(selected))
        for region in primary:
            probs=[r["regions"][region]["trigger_probability"] for r in selected]
            summary[scope]["regions"][region]=dict(expected_bad_frame_count=math.fsum(probs),
                mean_trigger_probability=math.fsum(probs)/len(selected),
                high_support_identities=sum(r["regions"][region]["high_support"] for r in selected))
    return dict(reference="Q4",distribution="each of six upgrades probability1/6; Q4 excluded",
        criterion="N>=256 and (>20% relative AEE increase or >1px absolute increase), review only",
        frames=rows,summaries=summary,expectation_of_single_action_tail_indicator=True,
        thresholding_expected_AEE_is_not_this_quantity=True,
        independence_between_regions_or_identities_assumed=False)


def evaluate_data(pack,prediction,model,raw_frames,hardware):
    policies=build_policies(pack,prediction,model)
    frames,summaries,decisions,per_identity,seqs=summarize_policies(raw_frames,pack,policies,hardware)
    pairs={(name,"Q4") for name in policies if name!="Q4"}
    for regime in ("six","allow_Q4"):
        primary="signed_g4__"+regime
        for ref in ("direct_aug_g4","direct_residual_g4","direct_aug_g1","direct_residual_g1","signed_g1","fit_static"):
            pairs.add((primary,ref+"__"+regime))
        pairs.add((primary,"oracle__"+regime))
    comparisons={c+"__vs__"+r:comparison(frames,summaries,c,r,seqs) for c,r in sorted(pairs)}
    definitions={name:{k:v for k,v in policy.items() if k not in ("weights","choices")} for name,policy in policies.items()}
    return dict(policy_definitions=definitions,quality_summaries=summaries,decision_summaries=decisions,
        per_identity=per_identity,comparisons=comparisons,sequences=seqs,
        uniform_six_tail_risk=uniform_tail_risk(raw_frames),
        fixed_six_policy_names=[k for k,v in policies.items() if v["regime"]=="fixed_six"],
        allow_Q4_policy_names=[k for k,v in policies.items() if v["regime"]=="allow_Q4"],
        primary="signed_g4",necessary_simplification="signed_g1",all_seven_learned_methods_reported=list(models.METHODS))


def run(args):
    require(not args.output.exists(),"output exists")
    b,model,prediction,pack,frames,hardware=check_inputs(args.bindings)
    out=evaluate_data(pack,prediction,model,frames,hardware)
    out.update(schema="flow_r33_directional_frozen_prediction_evaluation_v1",status="COMPLETE",role=b["role"],
        bindings_sha256=sha(args.bindings),sources=b["files"],identity_count=len(frames),
        real_data_fitting=False,new_flow_network_forwards=0,GPU_used=False,
        single_global_action_then_all16_region_evaluation=True,
        offline_oracle_objective="all_valid_AEE_only_not_region_specific",
        runtime_uses_only_raw_voxel_features=True,feature_branch="X160_spatial_temporal_correlations",
        label_source_is_original128_and_feature_source_is_new160=True,offline_replacement_diagnostics_are_not_deployable=True,
        activity_control_run=False,activity_control_note="No exact macro activity field in frozen pack; never reconstructed from 128 coarse statistics.",
        numerical_contract="float64 CPU selector; no deployed fixed-point predictor validation",
        hardware_scope=dict(source_schema=hardware["schema"],service_multiplier=4,
            interpretation="sum of four identical controlled patch operator services only; not full-network latency",
            selector_action_to_hardware_action={"-1":6,**{str(i):i for i in range(6)}},
            feature_and_head_execution_included=False,feature_reads_included=False,
            head_weight_storage_included=False,whole_network=False,
            GT_oracle_costs_are_hypothetical_operator_service_only=True,
            uniform_cost_is_expected_service_and_separate_worst_possible_action=True,
            no_total_hardware_gain_claim=True),
        threshold_notes=dict(aggregate_2pct="B/L protection review; allGT also disclosed; not a pass/fail claim",
            high_support_tail="legacy N>=256 and (>20% relative or >1px absolute) for allGT/B/L; review only",
            legacy_relative="allGT3%, B/L5% are old reference annotations, not this round's pass criteria"),
        oracle_regret_notes="6-action regret may be negative when Q4 is allowed; 7-action regret is separately nonnegative.",
        scope="selector held-out identity/sequence experiment; no official test or backbone-unseen claim")
    write_new(args.output,out)
    print(json.dumps(dict(status="COMPLETE",role=b["role"],identities=len(frames),policies=len(out["policy_definitions"]))))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="command",required=True)
    bp=sub.add_parser("bind")
    bp.add_argument("--pack-dir",type=Path,required=True)
    bp.add_argument("--original-pack-dir",type=Path,required=True)
    bp.add_argument("--predictions",type=Path,required=True)
    bp.add_argument("--model",type=Path,required=True)
    bp.add_argument("--role",choices=("tune","confirm"),required=True)
    bp.add_argument("--hardware",type=Path,default=HERE/"hardware/results/SUMMARY_V1.json")
    bp.add_argument("--heads-freeze",type=Path)
    bp.add_argument("--output",type=Path,required=True)
    rp=sub.add_parser("run")
    rp.add_argument("--bindings",type=Path,required=True)
    rp.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    bind(args) if args.command=="bind" else run(args)


if __name__=="__main__":
    main()
