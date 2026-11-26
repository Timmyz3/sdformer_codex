"""Independent synthetic math/leakage checks, no real labels or model forward."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/"algorithm/predictors.py"
spec = importlib.util.spec_from_file_location("r33_independent_predictors", SOURCE)
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
rng = np.random.default_rng(3361)
checks = []

# Physical labels and support-folding checked with explicit cell slices.
h, w = 8, 12
f0 = rng.normal(size=(2, h, w))
gt = rng.normal(size=(2, h, w))
actions = f0[None] + rng.normal(scale=.04, size=(6, 2, h, w))
valid = rng.random((h, w)) > .4
valid[:2, :3] = False
labels = p.make_labels(f0, actions, gt, valid)
e = f0-gt; u = e/np.maximum(np.hypot(e[0], e[1]), 1e-12)[None]
for g in (1, 4):
    v = np.zeros((2, g, g)); delta = np.zeros((6, 2, g, g))
    for y in range(g):
        for x in range(g):
            sy, sx = slice(y*h//g, (y+1)*h//g), slice(x*w//g, (x+1)*w//g)
            m = valid[sy, sx]
            v[:, y, x] = u[:, sy, sx][:, m].sum(axis=1)/valid.sum()
            if m.any():
                for a in range(6):
                    delta[a, :, y, x] = (actions[a]-f0)[:, sy, sx][:, m].mean(axis=1)
    expected = np.r_[v.ravel(), delta.ravel()]
    np.testing.assert_allclose(labels[f"aux_g{g}"], expected, rtol=1e-12, atol=1e-12)
    scores = np.array([sum(v[axis, y, x]*delta[a, axis, y, x] for axis in range(2) for y in range(g) for x in range(g)) for a in range(6)])
    np.testing.assert_allclose(labels[f"ideal_signed_g{g}"], scores, rtol=1e-12, atol=1e-12)
true_delta = [np.hypot(*(a-gt))[valid].mean()-np.hypot(*(f0-gt))[valid].mean() for a in actions]
np.testing.assert_allclose(labels["delta_aee"], true_delta, rtol=1e-12, atol=1e-12)
checks.append("independent labels: signed orientation, support folded into v, cell means, empty cell, exact AEE deltas")

# Ridge dual solution compared to independent weighted primal normal equations.
x = rng.normal(size=(24, 128)); x[:, 4] = 2
y = rng.normal(size=(24, 6)); weight = rng.integers(1, 1000, 24)
model = p.train_ridge(x, y, weight, .1)
q = weight/weight.sum(); xm = np.average(x, axis=0, weights=weight)
xs = np.sqrt(np.average((x-xm)**2, axis=0, weights=weight)); xs[xs <= 1e-12] = 1
ym = np.average(y, axis=0, weights=weight)
z = (x-xm)/xs
coef = np.linalg.solve((z.T*q)@z+.1*np.eye(128), (z.T*q)@(y-ym))
testx = rng.normal(size=(7, 128))
np.testing.assert_allclose(p.ridge_predict(model, testx), (testx-xm)/xs@coef+ym, rtol=1e-10, atol=1e-10)
checks.append("weighted ridge dual agrees with independent primal; fit-only standardization and unpenalized intercept")

# All sequence-level folds must isolate held labels, including stacked auxiliary fields.
n = 32
data = dict(x=rng.normal(size=(n, 128)), delta_aee=rng.normal(scale=.03, size=(n, 6)),
            valid_pixels=rng.integers(100, 10000, n), ids=np.array([f"synthetic_{i}" for i in range(n)]),
            groups=np.array([f"group_{i//4}" for i in range(n)]),
            aux_g1=rng.normal(size=(n, 14)), aux_g4=rng.normal(size=(n, 224)))
first = p.loso_development(data)
held = data["groups"] == "group_0"
changed = {k: v.copy() for k, v in data.items()}
for key in ("delta_aee", "aux_g1", "aux_g4"):
    changed[key][held] += 1e5
changed["valid_pixels"][held] *= 100
second = p.loso_development(changed)
for name in p.METHODS:
    np.testing.assert_array_equal(first["scores"][name][held], second["scores"][name][held])
for row in first["trace"]:
    held_ids = set(row["held_ids"])
    assert held_ids.isdisjoint(row["fit_ids"])
    for traces in row["auxiliary_crossfit"].values():
        for inner in traces:
            assert held_ids.isdisjoint(inner["fit_ids"]+inner["held_ids"])
            assert set(inner["held_ids"]).isdisjoint(inner["fit_ids"])
checks.append("changing outer held labels/aux/GT weights leaves all seven held predictions bitwise unchanged; inner stack folds exclude outer held IDs")

suite = p.fit_suite(data)
inference = p.predict_suite(suite, data["x"][:3])
assert list(inspect.signature(p.predict_suite).parameters) == ["model", "x"]
for g in (1, 4):
    aux = inference["predicted_auxiliary"][f"g{g}"]
    aug = p.augmented_features(data["x"][:3], aux, g)
    fixed = p.signed_scores(aux, g)
    residue = p.ridge_predict(suite["models"][f"direct_residual_g{g}"], aug)
    np.testing.assert_array_equal(inference["scores"][f"direct_residual_g{g}"], fixed+residue)
    assert aug.shape[1] == 128+26*g*g
np.testing.assert_array_equal(p.choose_actions(np.array([[0, 0, 1, 2, 3, 4], [-1, -1, 0, 2, 3, 4]])), [0, 0])
np.testing.assert_array_equal(p.choose_actions(np.array([[0, 0, 1, 2, 3, 4], [-1, -1, 0, 2, 3, 4]]), True), [-1, 0])
assert set(p.METHODS) == set(inference["scores"]) == set(inference["allow_Q4_choices"])
checks.append("online interface X-only; augmented ordinary gets all products; residual equals fixed score plus learned correction; Q4 ties/threshold zero and all seven reported")

voxel = np.zeros((10, 8, 12)); voxel[0, :2, :3] = 2
feature = p.voxel_features(voxel)
expected = [0.2, 0, 0.1, 0, 0.2, 2/9, -1, 1]
np.testing.assert_allclose(feature[:8], expected, atol=1e-12)
assert feature.shape == (128,) and np.count_nonzero(feature[8:]) == 0
checks.append("raw T10 4x4 feature geometry/counts and no GT support denominator")

out = dict(schema="flow_r33_predictors_independent_synthetic_v1", status="PASS",
           predictor_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           real_GT_arrays_read=False, real_quality_results_read=False, GPU_used=False,
           model_forwards=0, checks=checks, scope="Synthetic mathematical and fold-isolation checks; no claim of real predictor quality or hardware costs.")
(HERE/"PREDICTORS_INDEPENDENT_CPU_V1.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
