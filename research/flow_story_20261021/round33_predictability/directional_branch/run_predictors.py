#!/usr/bin/env python3
"""Root-controlled CPU CLI; fit and prediction have disjoint input interfaces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import predictors as p


MEMBERSHIP_SHA = "d860612a3ae3e4620fa2a9a616df9f8e93774efadcf7dc564881e27b7e7da8cc"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(p.jsonable(value), indent=2, allow_nan=False)+"\n")


def member_rows(path, role):
    if sha(path) != MEMBERSHIP_SHA:
        raise ValueError("membership hash differs from R33 frozen 64/16/40 identities")
    m = json.loads(Path(path).read_text())
    return [r for r in m["records"] if r["role"] == role]


def verify_ids(ids, groups, membership, role):
    rows = member_rows(membership, role)
    expected = {r["target_id"]: r["sequence"] for r in rows}
    if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError("pack identities do not equal the entire frozen role")
    if any(expected[i] != g for i, g in zip(ids, groups)):
        raise ValueError("pack sequence metadata mismatch")


def read_metadata(z):
    ids = np.asarray(z["ids"], dtype=str)
    group_key = "groups" if "groups" in z else "seqs"
    return ids, np.asarray(z[group_key], dtype=str)


def read_input(z):
    return p.finite(z["x" if "x" in z else "X"], "voxel X", 2)


def load_fit_pack(path, membership):
    # A FIT-ONLY file is mandatory. A combined file containing confirmation
    # arrays is rejected rather than silently slicing it after loading labels.
    with np.load(path, allow_pickle=False) as z:
        ids, groups = read_metadata(z)
        verify_ids(ids, groups, membership, "fit")
        if "roles" in z and set(np.asarray(z["roles"], dtype=str)) != {"fit"}:
            raise ValueError("fit pack must contain fit rows only")
        x = read_input(z)
        y = z["delta_aee" if "delta_aee" in z else "y_delta"].copy()
        if "valid_pixels" in z:
            w = z["valid_pixels"].copy()
        else:
            raise ValueError("explicit valid_pixels required; do not infer region column order")
        d = dict(ids=ids, groups=groups, x=x, delta_aee=y, valid_pixels=w)
        for grid in (1, 4):
            key = f"aux_g{grid}"
            if key in z:
                d[key] = z[key].copy()
            else:
                v, response = z[f"v{grid}"], z[f"delta{grid}"]
                d[key] = np.concatenate((v.reshape(len(x), -1), response.reshape(len(x), -1)), axis=1)
    return p.validate_dataset(d, require_groups=3)


def inference_pack(path, membership, role):
    # Deliberately access ONLY identity/sequence/role metadata and voxel X.
    # A label-bearing pack is permitted for root convenience, but its label
    # arrays are never materialized or handed to the runtime interface.
    with np.load(path, allow_pickle=False) as z:
        ids, groups = read_metadata(z)
        verify_ids(ids, groups, membership, role)
        if "roles" in z and set(np.asarray(z["roles"], dtype=str)) != {role}:
            raise ValueError("prediction pack role mismatch")
        x = read_input(z)
    return ids, groups, x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--fit-pack", type=Path, required=True)
    fit.add_argument("--membership", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--include-loso", action="store_true", help="fixed nested development diagnostic")
    infer = sub.add_parser("predict")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--model-sha256", required=True, help="freeze model SHA before confirmation collection")
    infer.add_argument("--input-pack", type=Path, required=True)
    infer.add_argument("--membership", type=Path, required=True)
    infer.add_argument("--role", choices=("tune", "confirm"), required=True)
    infer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        d = load_fit_pack(args.fit_pack, args.membership)
        model = p.fit_suite(d, p.ALPHA)
        model["metadata"].update(membership_sha256=MEMBERSHIP_SHA,
                                 fit_pack_sha256=sha(args.fit_pack),
                                 predictor_source_sha256=sha(p.__file__),
                                 runner_source_sha256=sha(__file__))
        # A fit-only static six-action and Q4-enabled counterpart have the
        # same no-upgrade opportunity as the learned heads.
        mean_delta = np.average(d["delta_aee"], axis=0, weights=d["valid_pixels"])
        model["metadata"]["fit_static_true_delta_means"] = mean_delta.tolist()
        model["metadata"]["fit_static_six_action"] = int(np.argmin(mean_delta))
        model["metadata"]["fit_static_allow_Q4_action"] = int(p.choose_actions(mean_delta[None], True)[0])
        args.output_dir.mkdir(parents=True)
        model_path = args.output_dir/"MODEL_V1.json"
        p.save_model(model_path, model)
        if args.include_loso:
            write(args.output_dir/"FIT_NESTED_LOSO_V1.json", dict(
                schema="flow_r33_directional_fit_loso_diagnostic_v1", ids=d["ids"], groups=d["groups"],
                results=p.loso_development(d, p.ALPHA), uses_fit_only=True,
                confirmation_read=False, model_selection=False))
        write(args.output_dir/"FIT_RECEIPT_V1.json", dict(
            schema="flow_r33_directional_fit_receipt_v1", model_sha256=sha(model_path),
            fit_pack_sha256=sha(args.fit_pack), membership_sha256=MEMBERSHIP_SHA,
            alpha=p.ALPHA, identity_count=len(d["x"]),
            fit_ids=d["ids"].tolist(), fit_groups=sorted(set(d["groups"])),
            predictor_source_sha256=sha(p.__file__), runner_source_sha256=sha(__file__),
            deterministic_closed_form=True, auxiliary_training="inner_leave_one_sequence_out",
            methods=list(p.METHODS), tune_or_confirm_read=False,
            real_flow_network_forwards=0, GPU_used=False))
        print(json.dumps(dict(status="FIT_COMPLETE", model_sha256=sha(model_path))))
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        if sha(args.model) != args.model_sha256:
            raise ValueError("frozen model hash mismatch")
        model = p.load_model(args.model)
        if model["metadata"]["membership_sha256"] != MEMBERSHIP_SHA:
            raise ValueError("model membership mismatch")
        if model["metadata"]["predictor_source_sha256"] != sha(p.__file__):
            raise ValueError("runtime predictor source changed since fit")
        ids, groups, x = inference_pack(args.input_pack, args.membership, args.role)
        if set(ids) & set(model["metadata"]["fit_ids"]):
            raise ValueError("fit/prediction identity overlap")
        pred = p.predict_suite(model, x)
        n = len(x)
        pred["fit_static_six_choices"] = [model["metadata"]["fit_static_six_action"]]*n
        pred["fit_static_allow_Q4_choices"] = [model["metadata"]["fit_static_allow_Q4_action"]]*n
        write(args.output, dict(schema="flow_r33_directional_predicted_choices_v1", role=args.role,
              ids=ids, groups=groups, results=pred, model_sha256=args.model_sha256,
              membership_sha256=MEMBERSHIP_SHA, input_pack_sha256=sha(args.input_pack),
              features_sha256=p.digest_arrays(x), GT_or_counterfactual_fields_read=False,
              input_feature_rule="frozen raw T10 voxel_features only", hardware_cost_evaluated=False))
        print(json.dumps(dict(status="PREDICTION_COMPLETE", role=args.role, identities=n)))


if __name__ == "__main__":
    main()
