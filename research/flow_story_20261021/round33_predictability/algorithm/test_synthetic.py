#!/usr/bin/env python3
"""Synthetic-only numerical/leakage checks; no dataset paths or torch imports."""
import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

import predictors as p


def close(a, b, **kw):
    np.testing.assert_allclose(a, b, rtol=kw.pop("rtol", 1e-10),
                               atol=kw.pop("atol", 1e-10), **kw)


def check_features():
    zero = p.voxel_features(np.zeros((10, 8, 8)))
    assert zero.shape == (128,) and np.array_equal(zero, np.zeros(128))
    one = p.voxel_features(np.ones((10, 8, 8))).reshape(16, 8)
    expected = [1, 0, 1, 0, 1, 0, 0, np.mean(np.linspace(-1, 1, 10)**2)]
    close(one, np.tile(expected, (16, 1)))
    z = np.zeros((10, 8, 8)); z[0] = 1
    f = p.voxel_features(z).reshape(16, 8)
    close(f, np.tile([.1, 0, .1, 0, .1, 1/9, -1, 1], (16, 1)))
    rng = np.random.default_rng(33)
    z = rng.normal(size=(10, 8, 12)); z[np.abs(z) < .8] = 0
    f = p.voxel_features(z).reshape(16, 8)
    neg = p.voxel_features(-z).reshape(16, 8)
    close(neg, f[:, [1, 0, 3, 2, 4, 5, 6, 7]])
    doubled = p.voxel_features(2*z).reshape(16, 8)
    close(doubled[:, [0, 1, 4, 5]], 2*f[:, [0, 1, 4, 5]])
    close(doubled[:, [2, 3, 6, 7]], f[:, [2, 3, 6, 7]])
    return dict(status="PASS", checks=["zero finite", "closed-form constant and one-bin input",
         "polarity swap", "amplitude scaling and scale-invariant geometry statistics"])


def independent_cell_labels(f0, fa, gt, valid, grid):
    h, w = valid.shape; n = int(valid.sum())
    v = np.zeros((2, grid, grid)); d = np.zeros((6, 2, grid, grid))
    count = np.zeros((grid, grid), dtype=int)
    delta_sums = np.zeros_like(d)
    for row in range(h):
        for col in range(w):
            if not valid[row, col]:
                continue
            cy, cx = row*grid//h, col*grid//w
            err = f0[:, row, col]-gt[:, row, col]
            unit = err/max(float(np.sqrt(err[0]**2+err[1]**2)), 1e-12)
            v[:, cy, cx] += unit/n
            count[cy, cx] += 1
            for action in range(6):
                delta_sums[action, :, cy, cx] += fa[action, :, row, col]-f0[:, row, col]
    for cy in range(grid):
        for cx in range(grid):
            if count[cy, cx]:
                d[:, :, cy, cx] = delta_sums[:, :, cy, cx]/count[cy, cx]
    score = []
    for action in range(6):
        score.append(sum(v[k, cy, cx]*d[action, k, cy, cx]
                         for k in range(2) for cy in range(grid) for cx in range(grid)))
    return v, d, np.asarray(score), count


def check_labels():
    rng = np.random.default_rng(20261021)
    f0 = rng.normal(size=(2, 8, 12)); gt = rng.normal(size=f0.shape)
    fa = f0[None] + rng.normal(scale=.1, size=(6, *f0.shape))
    mask = rng.random((8, 12)) > .45
    mask[:2, :3] = False  # a truly unsupported 4x4 cell
    f0[:, 7, 11] = gt[:, 7, 11]; mask[7, 11] = True  # zero-error unit convention
    labels = p.make_labels(f0, fa, gt, mask)
    for grid in (1, 4):
        v, d, score, count = independent_cell_labels(f0, fa, gt, mask, grid)
        close(labels[f"aux_g{grid}"], np.r_[v.ravel(), d.ravel()])
        close(labels[f"ideal_signed_g{grid}"], score)
        assert np.array_equal(labels[f"support_counts_g{grid}"], count)
        z = labels[f"aux_g{grid}"][None]
        # Independent check of old support-weighted R32 expression.
        uv, dd = p.split_aux(z, grid)
        vv = uv.reshape(2, grid, grid)
        unweighted_u = np.divide(vv*mask.sum(), count[None],
                                 out=np.zeros_like(vv), where=count[None] > 0)
        old = [np.sum(count*np.sum(unweighted_u*a, axis=0))/mask.sum()
               for a in dd.reshape(6, 2, grid, grid)]
        close(old, score)
    assert labels["valid_pixels"] == int(mask.sum())
    # An intentional case where averaging vectors loses a correlation term.
    f0 = np.zeros((2, 4, 4)); f0[0, :, :2] = 1; f0[0, :, 2:] = -1
    fa = np.repeat(f0[None], 6, axis=0); fa[0] = 0
    l = p.make_labels(f0, fa, np.zeros_like(f0), np.ones((4, 4), dtype=bool))
    close(l["delta_aee"][0], -1)
    close(l["ideal_signed_full"][0], -1)
    close(l["ideal_signed_g4"][0], -1)
    close(l["ideal_signed_g1"][0], 0)
    try:
        p.make_labels(f0, fa, np.zeros_like(f0), np.zeros((4, 4), dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("zero GT support accepted")
    return dict(status="PASS", checks=["independent pixel-loop label and ordering match",
        "GT support absorbed exactly into v", "empty cell and zero-error handling",
        "true versus first-order sign", "1x1 correlation loss preserved", "zero-support reject"])


def check_ridge_math():
    rng = np.random.default_rng(44)
    x = rng.normal(size=(13, 5)); x[:, 4] = 2
    y = rng.normal(size=(13, 3)); w = np.arange(1, 14)
    model = p.train_ridge(x, y, w)
    weights = w/w.sum()
    mu = np.sum(weights[:, None]*x, axis=0)
    scale = np.sqrt(np.sum(weights[:, None]*(x-mu)**2, axis=0)); scale[-1] = 1
    z = (x-mu)/scale
    center_y = np.sum(weights[:, None]*y, axis=0)
    primal = np.linalg.solve(z.T@(weights[:, None]*z)+.1*np.eye(5),
                             z.T@(weights[:, None]*(y-center_y)))
    close(model["coef"], primal)
    close(p.ridge_predict(model, x), z@primal+center_y)
    # A pure fixed contraction is in the ordinary augmented model's class.
    aux = rng.normal(size=(7, 224)); x = rng.normal(size=(7, 128))
    aug = p.augmented_features(x, aux, 4)
    coefficient = np.zeros((544, 6))
    for action in range(6):
        coefficient[128+224+action*32:128+224+(action+1)*32, action] = 1
    close(aug@coefficient, p.signed_scores(aux, 4))
    return dict(status="PASS", checks=["dual versus independent primal weighted ridge",
         "constant column", "fixed contraction expressible by augmented ordinary head"])


def synthetic_data():
    rng = np.random.default_rng(33001)
    n = 24
    return dict(x=rng.normal(size=(n, 128)), delta_aee=rng.normal(scale=.01, size=(n, 6)),
        valid_pixels=rng.integers(50, 200, size=n),
        aux_g1=rng.normal(scale=.1, size=(n, 14)),
        aux_g4=rng.normal(scale=.03, size=(n, 224)),
        ids=np.asarray([f"synthetic_{i:03d}" for i in range(n)]),
        groups=np.asarray([f"sequence_{i//6}" for i in range(n)]))


def check_crossfit_and_runtime():
    d = synthetic_data()
    held = d["groups"] == "sequence_0"
    changed = {k: v.copy() for k, v in d.items()}
    changed["aux_g1"][held] += 10000
    oof, trace = p.crossfit_auxiliary(d["x"], d["aux_g1"], d["valid_pixels"], d["ids"], d["groups"])
    oof_changed, _ = p.crossfit_auxiliary(changed["x"], changed["aux_g1"], changed["valid_pixels"], changed["ids"], changed["groups"])
    close(oof[held], oof_changed[held], rtol=0, atol=0)
    for item in trace:
        assert not set(item["fit_ids"]) & set(item["held_ids"])
        assert item["held_group"] not in item["fit_groups"]
    # Every kind of held-out supervised value is changed. Outer predictions
    # must be exactly invariant because the complete stack is fit inside fold.
    changed["aux_g4"][held] -= 10000
    changed["delta_aee"][held] += 10000
    changed["valid_pixels"][held] *= 100
    out = p.loso_development(d)
    other = p.loso_development(changed)
    for method in p.METHODS:
        close(out["scores"][method][held], other["scores"][method][held], rtol=0, atol=0)
    for outer in out["trace"]:
        outer_held = set(outer["held_ids"])
        for grid_trace in outer["auxiliary_crossfit"].values():
            for inner in grid_trace:
                assert not outer_held & set(inner["fit_ids"])
                assert not outer_held & set(inner["held_ids"])
    model = p.fit_suite(d)
    probe = d["x"][:3] + .15
    before = p.predict_suite(model, probe)
    with tempfile.TemporaryDirectory(prefix="flow_r33_synthetic_") as tmp:
        path = Path(tmp)/"model.json"
        p.save_model(path, model)
        after = p.predict_suite(p.load_model(path), probe)
    for method in p.METHODS:
        close(before["scores"][method], after["scores"][method], rtol=0, atol=0)
    # Runtime has no label/GT/identity argument. Explicitly zeroing an ordinary
    # residual model recovers the structured method without any retraining.
    zero_residual = copy.deepcopy(model)
    for grid in (1, 4):
        head = zero_residual["models"][f"direct_residual_g{grid}"]
        head["coef"][:] = 0; head["intercept"][:] = 0
    result = p.predict_suite(zero_residual, probe)
    for grid in (1, 4):
        close(result["scores"][f"direct_residual_g{grid}"], result["scores"][f"signed_g{grid}"], rtol=0, atol=0)
    return dict(status="PASS", checks=["inner sequence OOF label perturbation invariance",
         "outer LOSO all-label/support perturbation invariance",
         "nested identity/group separation", "JSON model exact prediction round trip",
         "zero ordinary residual equals signed", "runtime receives only X"])


def check_decisions_and_replacements():
    s = np.asarray([[0, 0, 1, 2, 3, 4], [-1, -1, 3, 2, 1, 0], [1, 2, 3, 4, 5, 6]], dtype=float)
    assert p.choose_actions(s).tolist() == [0, 0, 0]
    assert p.choose_actions(s, True).tolist() == [-1, 0, -1]
    rng = np.random.default_rng(333)
    aux = rng.normal(size=(2, 224))
    replacement = p.offline_replacement_scores(aux, aux, 4)
    for v in replacement.values():
        close(v, p.signed_scores(aux, 4))
    return dict(status="PASS", checks=["Q4 wins zero ties", "lower action id wins upgrade ties",
         "six fixed budget remains separate", "replacement all-equal identity"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checks = {"features": check_features(), "labels": check_labels(),
              "ridge_math": check_ridge_math(), "crossfit_and_runtime": check_crossfit_and_runtime(),
              "decisions_and_replacements": check_decisions_and_replacements()}
    out = dict(schema="flow_r33_synthetic_cpu_checks_v1", status="PASS", checks=checks,
        real_data_read=False, real_data_fitting=False, real_flow_network_forwards=0,
        GPU_used=False, seeds=[33, 20261021, 44, 33001, 333],
        numpy_version=np.__version__,
        source_sha256={Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest()
                       for f in (p.__file__, __file__)})
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status="PASS", families=len(checks), real_data_fitting=False)))


if __name__ == "__main__":
    main()
