#!/usr/bin/env python3
"""R33 small CPU predictors; runtime API receives only frozen voxel features.

Offline label construction uses GT and real counterfactuals. This module never
imports a flow network, torch, or GPU library. It contains no real-data execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "flow_r33_voxel_small_predictors_v1"
FEATURE_GRID = 4
FEATURES_PER_CELL = 8
N_FEATURES = FEATURE_GRID ** 2 * FEATURES_PER_CELL
N_ACTIONS = 6
EPS = 1e-12
ALPHA = 0.1
METHODS = ("direct_raw", "signed_g1", "signed_g4", "direct_aug_g1",
           "direct_aug_g4", "direct_residual_g1", "direct_residual_g4")
FEATURE_NAMES = ("positive_amplitude_mean", "negative_magnitude_mean",
                 "positive_bin_fraction", "negative_bin_fraction",
                 "absolute_temporal_signed_sum_mean_div_T",
                 "absolute_adjacent_time_difference_mean",
                 "absolute_amplitude_time_centroid",
                 "absolute_amplitude_time_second_moment")


def finite(x, name, ndim=None):
    x = np.asarray(x, dtype=np.float64)
    if ndim is not None and x.ndim != ndim:
        raise ValueError(f"{name}: expected rank {ndim}, got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name}: nonfinite values")
    return x


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.asarray(value)
        if a.dtype.kind == "O":
            raise ValueError("object arrays are not accepted")
        h.update(str(a.shape).encode())
        h.update(a.dtype.str.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def voxel_features(voxel):
    """Return 128 features using only the existing T10 raw signed voxel.

    Cell order is row-major; within a cell FEATURE_NAMES fixes statistic order.
    Denominators use full geometric cell area and T, never GT/event support.
    No sample-specific normalization or clipping is fitted here.
    """
    z = finite(voxel, "voxel", 3)
    t, h, w = z.shape
    if t != 10 or h % FEATURE_GRID or w % FEATURE_GRID:
        raise ValueError("voxel must have T=10 and spatial dimensions divisible by 4")
    if min(h, w) < FEATURE_GRID:
        raise ValueError("empty feature cells")
    edges_t = np.linspace(-1.0, 1.0, t, dtype=np.float64)
    rows = []
    for cy in range(FEATURE_GRID):
        for cx in range(FEATURE_GRID):
            a = z[:, cy*h//4:(cy+1)*h//4, cx*w//4:(cx+1)*w//4]
            absolute = np.abs(a)
            mass_t = absolute.sum(axis=(1, 2), dtype=np.float64)
            mass = float(mass_t.sum())
            rows.extend((
                np.maximum(a, 0).mean(), np.maximum(-a, 0).mean(),
                np.count_nonzero(a > 0)/a.size, np.count_nonzero(a < 0)/a.size,
                np.abs(a.sum(axis=0)).mean()/t,
                np.abs(np.diff(a, axis=0)).mean(),
                np.dot(edges_t, mass_t)/mass if mass else 0.0,
                np.dot(edges_t*edges_t, mass_t)/mass if mass else 0.0,
            ))
    return finite(rows, "voxel features", 1)


def pooled(vector, valid, grid):
    """GT-supported offline vector cell mean and support counts, like R32."""
    h, w = valid.shape
    if grid not in (1, 4) or h % grid or w % grid or vector.shape != (2, h, w):
        raise ValueError("unsupported grid/vector geometry")
    bh, bw = h//grid, w//grid
    count = valid.reshape(grid, bh, grid, bw).sum(axis=(1, 3), dtype=np.int64)
    sums = (vector*valid[None]).reshape(2, grid, bh, grid, bw).sum(
        axis=(2, 4), dtype=np.float64)
    mean = np.divide(sums, count[None], out=np.zeros_like(sums), where=count[None] > 0)
    return count, mean


def make_labels(background, action_predictions, gt, valid):
    """Offline labels. These inputs MUST NOT enter deployment/predict().

    e=f_Q4-GT, u=e/max(norm(e),eps). For each cell:
      v_c=sum_valid(u)/N_total, d_ac=sum_valid(delta_a)/N_c (0 if N_c=0).
    The fixed readout sum_c v_c dot d_ac includes GT support in a predicted
    quantity. Auxiliary order is v (2,G,G), then all six d (6,2,G,G), C-order.
    True delta sign is AEE_action - AEE_Q4; smaller is better.
    """
    f0 = finite(background, "background", 3)
    fa = finite(action_predictions, "action predictions", 4)
    target = finite(gt, "GT", 3)
    m = np.asarray(valid)
    if m.dtype != np.bool_ or m.ndim != 2:
        raise ValueError("valid must be a boolean H,W mask")
    if f0.shape != target.shape or f0.shape != (2, *m.shape):
        raise ValueError("background/GT/mask geometry mismatch")
    if fa.shape != (N_ACTIONS, *f0.shape):
        raise ValueError("exactly six action predictions are required")
    n = int(m.sum())
    if n <= 0:
        raise ValueError("no valid GT pixels: reject identity before any fitting")
    e = f0-target
    error = np.linalg.norm(e, axis=0)
    unit = e/np.maximum(error, EPS)[None]
    delta = fa-f0[None]
    aee0 = float(error[m].mean())
    true_delta = np.asarray([np.linalg.norm(f-target, axis=0)[m].mean()-aee0 for f in fa])
    out = dict(valid_pixels=n, background_AEE=aee0, delta_aee=true_delta,
               offline_GT_and_counterfactuals=True)
    for grid in (1, 4):
        count, ubar = pooled(unit, m, grid)
        v = ubar*(count/n)[None]
        d = np.stack([pooled(d_a, m, grid)[1] for d_a in delta])
        aux = np.concatenate((v.ravel(), d.ravel()))
        out[f"aux_g{grid}"] = aux
        out[f"ideal_signed_g{grid}"] = signed_scores(aux[None], grid)[0]
        out[f"support_counts_g{grid}"] = count
    out["ideal_signed_full"] = np.asarray([
        np.sum(unit*d_a, axis=0)[m].mean() for d_a in delta])
    return out


def split_aux(aux, grid):
    z = finite(aux, "auxiliary field", 2)
    dim = 2*grid*grid
    if grid not in (1, 4) or z.shape[1] != 7*dim:
        raise ValueError("auxiliary field width mismatch")
    return z[:, :dim], z[:, dim:].reshape(len(z), N_ACTIONS, dim)


def component_products(aux, grid):
    v, d = split_aux(aux, grid)
    return v[:, None, :]*d


def signed_scores(aux, grid):
    return component_products(aux, grid).sum(axis=2)


def augmented_features(x, aux, grid):
    # Giving the ordinary readout every component product makes the fixed
    # contraction representable by a linear readout (not merely six averages).
    return np.concatenate((x, aux, component_products(aux, grid).reshape(len(x), -1)), axis=1)


def train_ridge(x, y, sample_weight, alpha=ALPHA):
    """Weighted ridge: sum_i p_i ||Yc_i-Xs_i B||² + alpha ||B||².

    Input centering/scaling and target centering are fit-only and weighted.
    Targets remain in their physical units. The intercept is not penalized.
    Dual solve avoids a large feature Gram matrix in this small-N experiment.
    """
    x, y = finite(x, "ridge inputs", 2), finite(y, "ridge targets", 2)
    w = finite(sample_weight, "sample weights", 1)
    if len(x) != len(y) or len(x) != len(w) or len(x) == 0:
        raise ValueError("ridge row mismatch or empty fit")
    if np.any(w <= 0) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("strictly positive weights/alpha required")
    p = w/w.sum()
    mean = (x*p[:, None]).sum(axis=0)
    scale = np.sqrt(((x-mean)**2*p[:, None]).sum(axis=0))
    # Constant fit columns carry no information; retain a finite transform.
    scale = np.where(scale > EPS, scale, 1.0)
    intercept = (y*p[:, None]).sum(axis=0)
    a = ((x-mean)/scale)*np.sqrt(p)[:, None]
    b = (y-intercept)*np.sqrt(p)[:, None]
    dual = np.linalg.solve(a@a.T + float(alpha)*np.eye(len(x)), b)
    coef = a.T@dual
    return dict(mean=mean, scale=scale, intercept=intercept, coef=coef,
                alpha=float(alpha), fit_rows=len(x), input_width=x.shape[1], output_width=y.shape[1])


def ridge_predict(model, x):
    x = finite(x, "predict features", 2)
    if x.shape[1] != model["input_width"]:
        raise ValueError("predictor feature width mismatch")
    return ((x-model["mean"])/model["scale"])@model["coef"]+model["intercept"]


def validate_dataset(data, require_groups=2):
    x = finite(data["x"], "x", 2)
    y = finite(data["delta_aee"], "delta_aee", 2)
    w = finite(data["valid_pixels"], "valid pixels", 1)
    ids, groups = np.asarray(data["ids"], dtype=str), np.asarray(data["groups"], dtype=str)
    if x.shape != (len(x), N_FEATURES) or y.shape != (len(x), 6):
        raise ValueError("dataset X/Y dimensions")
    if any(a.shape != (len(x),) for a in (w, ids, groups)):
        raise ValueError("metadata row dimensions")
    if len(set(ids)) != len(ids) or any(not s for s in ids) or any(not s for s in groups):
        raise ValueError("ids must be unique and ids/groups nonempty")
    if len(set(groups)) < require_groups:
        raise ValueError("not enough sequence groups for complete nested cross-fitting")
    if np.any(w <= 0) or np.any(w != np.floor(w)):
        raise ValueError("GT supports must be positive integer counts")
    out = dict(x=x, delta_aee=y, valid_pixels=w, ids=ids, groups=groups)
    for grid in (1, 4):
        key = f"aux_g{grid}"
        aux = finite(data[key], key, 2)
        if aux.shape != (len(x), 14*grid*grid):
            raise ValueError(f"{key} dimensions")
        out[key] = aux
    return out


def subset(data, mask):
    return {k: np.asarray(v)[mask] for k, v in data.items()}


def crossfit_auxiliary(x, aux, weight, ids, groups, alpha=ALPHA):
    """Inner leave-one-sequence-out predictions for honest stacked training."""
    predicted = np.empty_like(aux, dtype=np.float64)
    trace = []
    for held_group in sorted(set(groups)):
        held = groups == held_group
        fit = ~held
        if not np.any(fit):
            raise ValueError("OOF auxiliary training would be empty")
        model = train_ridge(x[fit], aux[fit], weight[fit], alpha)
        predicted[held] = ridge_predict(model, x[held])
        trace.append(dict(held_group=str(held_group), fit_ids=ids[fit].tolist(),
                          held_ids=ids[held].tolist(), fit_groups=sorted(set(groups[fit]))))
    return predicted, trace


def fit_suite(data, alpha=ALPHA):
    """Offline fit only. Never pass confirmation identities/labels here.

    All objectives use identical GT-support sample weights on FIT identities.
    Weighted GT is training supervision only, not runtime input. Auxiliary heads
    are shared by fixed signed readout and both augmented ordinary heads.
    """
    d = validate_dataset(data, require_groups=2)
    x, y, w = d["x"], d["delta_aee"], d["valid_pixels"]
    model = dict(schema=SCHEMA, alpha=float(alpha), feature_names=list(FEATURE_NAMES),
                 methods=list(METHODS), models={}, auxiliary={},
                 metadata=dict(fit_ids=d["ids"].tolist(), fit_groups=sorted(set(d["groups"])),
                    fit_input_and_label_sha256=digest_arrays(*(d[k] for k in sorted(d))),
                    seed=None, deterministic_closed_form=True, input_width=N_FEATURES,
                    sample_weight="fit-only valid_GT_pixels", no_upgrade_threshold=0.0,
                    higher_is_better=False, alpha_search=False), crossfit_trace={})
    model["models"]["direct_raw"] = train_ridge(x, y, w, alpha)
    for grid in (1, 4):
        aux = d[f"aux_g{grid}"]
        oof, trace = crossfit_auxiliary(x, aux, w, d["ids"], d["groups"], alpha)
        model["crossfit_trace"][f"g{grid}"] = trace
        model["auxiliary"][f"g{grid}"] = train_ridge(x, aux, w, alpha)
        aug = augmented_features(x, oof, grid)
        model["models"][f"direct_aug_g{grid}"] = train_ridge(aug, y, w, alpha)
        # Strong ordinary residual control: zero residual is the structured
        # prediction itself; real delta-AEE labels may improve/calibrate it.
        model["models"][f"direct_residual_g{grid}"] = train_ridge(
            aug, y-signed_scores(oof, grid), w, alpha)
    return model


def choose_actions(scores, allow_no_upgrade=False):
    """Upgrade ids 0..5; -1 is Q4. Ties: Q4 first, then lower upgrade id."""
    s = finite(scores, "scores", 2)
    if s.shape[1] != N_ACTIONS:
        raise ValueError("six scores required")
    best = np.argmin(s, axis=1).astype(np.int64)
    if allow_no_upgrade:
        best[np.min(s, axis=1) >= 0.0] = -1
    return best


def predict_suite(model, x):
    """Deployment interface: ONLY X from voxel_features; no GT/identity API."""
    if model["schema"] != SCHEMA:
        raise ValueError("model schema mismatch")
    x = finite(x, "online X", 2)
    if x.shape[1] != N_FEATURES:
        raise ValueError("online X must have 128 columns")
    scores = {"direct_raw": ridge_predict(model["models"]["direct_raw"], x)}
    aux_predictions = {}
    for grid in (1, 4):
        aux = ridge_predict(model["auxiliary"][f"g{grid}"], x)
        aux_predictions[f"g{grid}"] = aux
        signed = signed_scores(aux, grid)
        scores[f"signed_g{grid}"] = signed
        aug = augmented_features(x, aux, grid)
        scores[f"direct_aug_g{grid}"] = ridge_predict(model["models"][f"direct_aug_g{grid}"], aug)
        scores[f"direct_residual_g{grid}"] = signed + ridge_predict(
            model["models"][f"direct_residual_g{grid}"], aug)
    return dict(scores=scores, predicted_auxiliary=aux_predictions,
                fixed_six_choices={k: choose_actions(v) for k, v in scores.items()},
                allow_Q4_choices={k: choose_actions(v, True) for k, v in scores.items()})


def offline_replacement_scores(predicted_aux, true_aux, grid):
    """GT/counterfactual idealized diagnostics, never deployment methods.

    The predicted/true direction is the already-support-weighted v, not unit u.
    This deliberately does not isolate support and direction prediction errors.
    """
    vp, dp = split_aux(predicted_aux, grid)
    vt, dt = split_aux(true_aux, grid)
    if vp.shape != vt.shape:
        raise ValueError("replacement row mismatch")
    return dict(all_true=(vt[:, None]*dt).sum(axis=2),
                predicted_weighted_direction_true_response=(vp[:, None]*dt).sum(axis=2),
                true_weighted_direction_predicted_response=(vt[:, None]*dp).sum(axis=2),
                all_predicted=(vp[:, None]*dp).sum(axis=2))


def loso_development(data, alpha=ALPHA):
    """Complete outer LOSO; stacker OOF generation is nested inside fit folds."""
    d = validate_dataset(data, require_groups=3)
    n = len(d["x"])
    scores = {name: np.empty((n, 6), dtype=np.float64) for name in METHODS}
    aux = {f"g{g}": np.empty((n, 14*g*g), dtype=np.float64) for g in (1, 4)}
    trace = []
    static = np.empty(n, dtype=np.int64)
    for held_group in sorted(set(d["groups"])):
        held = d["groups"] == held_group
        fit = ~held
        model = fit_suite(subset(d, fit), alpha)
        pred = predict_suite(model, d["x"][held])
        for key in scores:
            scores[key][held] = pred["scores"][key]
        for key in aux:
            aux[key][held] = pred["predicted_auxiliary"][key]
        weighted_true = np.average(d["delta_aee"][fit], axis=0,
                                   weights=d["valid_pixels"][fit])
        static[held] = int(np.argmin(weighted_true))
        trace.append(dict(held_group=str(held_group), fit_ids=d["ids"][fit].tolist(),
                          held_ids=d["ids"][held].tolist(),
                          auxiliary_crossfit=model["crossfit_trace"]))
    return dict(scores=scores, predicted_auxiliary=aux, trace=trace,
                fit_fold_static_six_choices=static,
                fixed_six_choices={k: choose_actions(v) for k, v in scores.items()},
                allow_Q4_choices={k: choose_actions(v, True) for k, v in scores.items()})


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_model(path, model):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(jsonable(model), allow_nan=False, separators=(",", ":"))+"\n")


def load_model(path):
    model = json.loads(Path(path).read_text())
    if model["schema"] != SCHEMA or model["methods"] != list(METHODS):
        raise ValueError("serialized model schema/method mismatch")
    for section in ("models", "auxiliary"):
        for ridge in model[section].values():
            for key in ("mean", "scale", "intercept", "coef"):
                ridge[key] = finite(ridge[key], key)
    return model
