"""Independent actual R39 audit. No GT, fitting, GPU, or policy changes."""
from pathlib import Path
import ast, collections, hashlib, importlib.util, json, os, time
import numpy as np

HERE = Path(__file__).resolve().parent
B = HERE.parent
F = np.float32
METHODS = ('folded128', 'ordinary128')
BASE = {'folded128': 113947223, 'ordinary128': 113965270}

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_module(p, name):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def arr(x): return np.asarray(x, F)
def exact(a, b):
    a, b = arr(a), arr(b)
    assert a.shape == b.shape and a.tobytes() == b.tobytes()
def cells(z):
    for y in range(4):
        for x in range(4):
            yield np.ascontiguousarray(z[:, y*120:(y+1)*120, x*160:(x+1)*160])

def features(z):
    # Independent transcription of the frozen NumPy FP32 reduction order.
    result = []; zero = 0; edges = np.linspace(-1, 1, 10, dtype=F)
    for a in cells(z):
        masses = np.sum(np.abs(a), axis=(1, 2), dtype=F)
        mass = np.sum(masses, dtype=F); zero += int(mass == 0)
        temporal = np.abs(np.sum(a, axis=0, dtype=F))
        adjacent = np.abs(np.subtract(a[1:], a[:-1]))
        result.extend([
            np.sum(np.maximum(a, F(0)), dtype=F)/F(a.size),
            np.sum(np.maximum(-a, F(0)), dtype=F)/F(a.size),
            F(np.count_nonzero(a > 0))/F(a.size),
            F(np.count_nonzero(a < 0))/F(a.size),
            F(np.sum(temporal, dtype=F)/F(temporal.size))/F(10),
            np.sum(adjacent, dtype=F)/F(adjacent.size),
            np.sum(edges*masses, dtype=F)/mass if mass else F(0),
            np.sum((edges*edges)*masses, dtype=F)/mass if mass else F(0)])
    return arr(result), zero

def mul(a, b):
    candidates = np.stack([np.multiply(x, y) for x in a for y in b])
    return np.min(candidates, axis=0), np.max(candidates, axis=0)
def add(a, b): return np.add(a[0], b[0]), np.add(a[1], b[1])
def affine(r, interval):
    mean, scale, coef, bias = [arr(r[k]) for k in ('mean', 'scale', 'coef', 'intercept')]
    assert np.all(scale > 0)
    z = tuple(np.divide(np.subtract(x, mean), scale) for x in interval)
    out = (np.zeros(len(bias), F), np.zeros(len(bias), F))
    for k in range(len(mean)):
        out = add(out, mul((z[0][k], z[1][k]), (coef[k], coef[k])))
    return add(out, (bias, bias))
def head(method, fold, ordinary, x):
    if method == 'folded128': return affine(fold['ridge'], x)
    family, selection = ordinary['family'], ordinary['selection']
    aux = affine(family['auxiliary'], x)
    product = mul(tuple(a[:2][None, :] for a in aux), tuple(a[2:].reshape(6, 2) for a in aux))
    signed = (np.zeros(6, F), np.zeros(6, F))
    for k in range(2): signed = add(signed, (product[0][:, k], product[1][:, k]))
    augmented = tuple(np.concatenate([x[i], aux[i], product[i].reshape(-1)]) for i in (0, 1))
    residual = affine(family['residual'][str(selection['alpha'])], augmented)
    shrink = F(selection['shrink'])
    return add(signed, mul((shrink, shrink), residual))
def certify(lower, upper):
    # Low-ID tie semantics of numpy.argmin, checked independently.
    winners = [i for i in range(6) if all(upper[i] < lower[j] for j in range(i))
               and all(upper[i] <= lower[j] for j in range(i+1, 6))]
    assert len(winners) <= 1
    return winners[0] if winners else None

def main():
    start = time.monotonic()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    output = HERE/'ACTUAL_FIT64_INDEPENDENT_V1.json'; assert not output.exists()
    A = B/'algorithm'; run = A/'actual_fit64_v1'
    freeze = read(A/'SOURCE_FREEZE_V1.json'); manifest = read(A/'INPUT_MANIFEST_V1.json')
    summary = read(run/'SUMMARY_V1.json'); receipt = read(A/'RUN_RECEIPT_V1.json')
    for p, h in freeze['files'].items(): assert sha(A/p) == h
    for rec in freeze['bound'].values(): assert sha(rec['path']) == rec['sha256']
    assert summary['source_freeze_sha256'] == sha(A/'SOURCE_FREEZE_V1.json')
    assert summary['input_manifest_sha256'] == sha(A/'INPUT_MANIFEST_V1.json')
    assert receipt['exit_code'] == 0 and receipt['status'] == 'COMPLETE'
    assert receipt['review_sha256'] == summary['review_sha256'] == sha(HERE/'RUNNER_PRELAUNCH_REVIEW_V1.json')
    assert receipt['environment']['CUDA_VISIBLE_DEVICES'] == ''
    fold = read(freeze['bound']['folded_model']['path']); ordinary = read(freeze['bound']['ordinary_model']['path'])
    records = manifest['records']; ids = [r['target_id'] for r in records]
    assert len(ids) == len(set(ids)) == 64 and ids == fold['fit_ids'] == ordinary['fit_ids']
    assert sorted(p.stem for p in (run/'identities').glob('*.json')) == sorted(ids)
    assert [r['target_id'] for r in summary['identities']] == ids
    # The feature-bound construction was separately proved/read and checked on
    # rational/synthetic fixtures. Replay that frozen core only for stage-0
    # unknown bounds, not for actual features, scores, interval heads or decisions.
    core = load_module(A/'certified_core.py', '_reviewed_r39_unknown_bounds')
    tree = ast.parse((A/'run_fit64.py').read_text())
    runtime = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'runtime')
    assert [a.arg for a in runtime.args.args] == ['z', 'fold', 'ordinary']
    names = {n.id for n in ast.walk(runtime) if isinstance(n, ast.Name)}
    assert not names.intersection({'full_reference', 'reference_scores', 'reference_features', 'GT', 'cache', 'summary'})
    counts = collections.Counter(); results = []; payload_hashes = {}
    for idx, rec in enumerate(records):
        assert rec['role'] == 'fit' and sha(rec['path']) == rec['sha256']
        assert Path(rec['path']).stat().st_size == rec['bytes']
        p = run/'identities'/(rec['target_id']+'.json'); row = read(p); payload_hashes[p.name] = sha(p)
        assert row['target_id'] == rec['target_id'] and row['sequence'] == rec['sequence'] and row['input_sha256'] == rec['sha256']
        z = np.load(rec['path'], allow_pickle=False)
        assert z.shape == (10, 480, 640) and z.dtype == F and np.isfinite(z).all()
        x, zeros = features(z); exact(x, row['reference_features']); assert zeros == row['zero_mass_cells']
        full = {}
        for m in METHODS:
            lower, upper = head(m, fold, ordinary, (x, x)); exact(lower, upper)
            exact(lower, row['reference_scores'][m]); full[m] = lower
        initial_lower = np.zeros(128, F); initial_upper = np.zeros(128, F)
        for i, a in enumerate(cells(z)):
            cheap = core.cheap_cell(a); unknown = core.unknown_bounds(cheap)
            indexes = i*8 + np.array([0, 1, 2, 3, 6, 7])
            exact(cheap['known'], x[indexes])
            initial_lower[indexes] = x[indexes]; initial_upper[indexes] = x[indexes]
            initial_lower[i*8+4:i*8+6] = unknown[0]; initial_upper[i*8+4:i*8+6] = unknown[1]
        decisions = {}
        assert row['known_feature_error'] is False
        for stage, st in enumerate(row['stages']):
            assert st['stage'] == stage and 'numerical_guard' not in st
            l, u = arr(st['feature_lower']), arr(st['feature_upper'])
            expected_l, expected_u = initial_lower.copy(), initial_upper.copy()
            for i in range(stage): expected_l[i*8+4:i*8+6] = expected_u[i*8+4:i*8+6] = x[i*8+4:i*8+6]
            exact(l, expected_l); exact(u, expected_u)
            assert np.all(l <= x) and np.all(x <= u); counts['feature_containment'] += 128
            assert set(st['heads']) == set(METHODS)-set(decisions)
            for m, h in st['heads'].items():
                lo, hi = head(m, fold, ordinary, (l, u)); exact(lo, h['lower']); exact(hi, h['upper'])
                assert np.all(lo <= full[m]) and np.all(full[m] <= hi)
                counts['score_containment'] += 6; counts[m+'_head_calls'] += 1
                action = certify(lo, hi); assert action == h['certified_action']
                if action is not None:
                    assert action == int(np.argmin(full[m])); decisions[m] = (stage, action)
        assert len(decisions) == 2 and len(row['stages']) == max(d[0] for d in decisions.values())+1
        counts['raw_inputs'] += 1; counts['zero_mass_cells'] += zeros; counts['complete_actions'] += 2
        row_result = dict(target_id=rec['target_id'], sequence=rec['sequence'], zero_mass_cells=zeros, decisions={})
        for m in METHODS:
            k, action = decisions[m]; d = row['decisions'][m]
            assert d['certified'] and d['stage'] == k and d['action'] == d['reference_action'] == action and d['reason'] is None
            base = BASE[m]-32*zeros; cost = 55877120-32*zeros+4204844*k+144320*((k+3)//4)
            assert d['original_prefix_ticks'] == base and d['optimistic_prefix_ticks'] == cost
            assert d['optimistic_saved_fraction'] == 1-cost/base
            assert d['bound_head_control_metadata_cost_assumed_zero'] is True
            assert d == summary['identities'][idx]['decisions'][m]
            row_result['decisions'][m] = dict(stage=k, action=action, base=base, cost=cost)
        results.append(row_result)
    assert counts['feature_containment'] == summary['checked_feature_bounds']
    assert counts['score_containment'] == summary['checked_score_bounds']
    scopes = ['overall']+list(dict.fromkeys(r['sequence'] for r in results)); independently_aggregated = {}
    for m in METHODS:
        independently_aggregated[m] = {}
        for scope in scopes:
            ds = [r['decisions'][m] for r in results if scope == 'overall' or r['sequence'] == scope]
            n = len(ds); base = sum(d['base'] for d in ds)/n; cost = sum(d['cost'] for d in ds)/n
            a = dict(identities=n, stage_histogram=dict(collections.Counter(str(d['stage']) for d in ds)), mean_stage=sum(d['stage'] for d in ds)/n,
                     fallbacks=0, original_mean_prefix_ticks=base, optimistic_mean_prefix_ticks=cost,
                     optimistic_mean_saved_fraction=1-cost/base, optimistic_worst_prefix_ticks=max(d['cost'] for d in ds),
                     optimistic_worst_saved_fraction=min(1-d['cost']/d['base'] for d in ds))
            assert a == summary['results'][m][scope], (m, scope)
            independently_aggregated[m][scope] = a
    go = {m: independently_aggregated[m]['overall']['optimistic_mean_saved_fraction'] >= .1 for m in METHODS}
    assert go == summary['necessary_optimistic_10pct_screen'] == dict(folded128=False, ordinary128=False)
    assert summary['continue_exposed_transport_allowed'] is False and summary['not_deployability_GO'] is True
    assert not summary['GT_read'] and not summary['training'] and not summary['GPU_used']
    report = dict(status='PASS_INDEPENDENT_NUMERIC_AND_AGGREGATE_STOP_CONFIGURATION', auditor_sha256=sha(__file__),
        summary_sha256=sha(run/'SUMMARY_V1.json'), source_freeze_sha256=sha(A/'SOURCE_FREEZE_V1.json'),
        run_receipt_sha256=sha(A/'RUN_RECEIPT_V1.json'), core_math_preflight_sha256=sha(HERE/'CORE_IMPLEMENTATION_PREFLIGHT_V1.json'),
        source_cost_review_sha256=sha(HERE/'SOURCE_COST_INDEPENDENT_V1.json'), identity_payload_sha256=payload_hashes,
        checks=dict(counts), source_information_isolation_review_pass=True, independent_FP32_features_scores_heads_and_first_certificates=True,
        feature_unknown_interval_provenance='Frozen reviewed core replay plus prior independent rational/synthetic mathematical preflight; not advertised as a second independent proof implementation.',
        false_certificates=0, fallback_count=0, all_actual_features_and_scores_bitwise_match=True,
        results=independently_aggregated, necessary_optimistic_10pct_screen=go,
        decision='STOP fixed all-cheap-then-natural-order-late, general-bound, old-resource two-pass configuration. No transport, fit, new GPU or RTL.',
        cost_scope='Source/service relaxation with all bound/head/control/metadata cost zero; NOT a new exact cycle replay, whole-network timing, RTL, or architecture-independent lower bound.',
        GT_values_read=0, new_fits=0, GPU_calls=0, auditor_CPU_seconds=time.monotonic()-start)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], counts=dict(counts), necessary_optimistic_10pct_screen=go, output_sha256=sha(output))))

if __name__ == '__main__': main()
