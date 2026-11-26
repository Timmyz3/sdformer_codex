"""ROUND6 frozen small predictors; NumPy only, no data access on import."""
import numpy as np

BASE_FEATURES = ('amplitude_abs_mean', 'feature_rms', 'pooling_residual_mse',
                 'spatial_neighbor_abs_difference', 'temporal_neighbor_abs_difference')
TILES = tuple((r, c) for r in range(3) for c in range(4))


def standardize_fit(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty matrix required')
    mean, std = x.mean(0), x.std(0)
    std = np.where(std > 1e-12, std, 1.)
    return (x - mean) / std, mean, std


def ridge_fit(x, y, alpha):
    z, mean, std = standardize_fit(x)
    y = np.asarray(y, dtype=np.float64)
    if len(y) != len(z) or not np.isfinite(y).all() or alpha <= 0:
        raise ValueError('Finite paired targets and positive alpha required')
    center = y.mean(0)
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - center))
    return {'kind': 'ridge', 'alpha': float(alpha), 'mean': mean, 'std': std,
            'beta': beta, 'intercept': center, 'fit_rows': len(x)}


def pca_fit(x):
    z, mean, std = standardize_fit(x)
    values, vectors = np.linalg.eigh(z.T @ z / len(z))
    order = np.argsort(values)[::-1][:2]
    beta = vectors[:, order]
    # Sign fixes only archive determinism. The six vector statistics are
    # invariant to a common orthogonal basis change (including sign).
    for j in range(2):
        if beta[np.argmax(np.abs(beta[:, j])), j] < 0:
            beta[:, j] *= -1
    return {'kind': 'PCA2', 'mean': mean, 'std': std, 'beta': beta,
            'intercept': np.zeros(2), 'fit_rows': len(x), 'eigenvalues': values[order]}


def predict(model, x):
    return (np.asarray(x, dtype=np.float64) - model['mean']) / model['std'] @ model['beta'] + model['intercept']


def predict_fp32(model, x):
    weight, bias = folded(model)
    return np.asarray(x, dtype=np.float32) @ weight.astype(np.float32) + np.asarray(bias, dtype=np.float32)


def folded(model):
    beta = model['beta'] / (model['std'][:, None] if model['beta'].ndim == 2 else model['std'])
    return beta, model['intercept'] - model['mean'] @ beta


def vector_statistics(vectors):
    v = np.asarray(vectors, dtype=np.float32)
    if v.shape != (3, 3, 2) or not np.isfinite(v).all():
        raise ValueError('Finite 3x3x2 vectors required')
    mag = np.linalg.norm(v, axis=-1)
    edges = np.concatenate([(v[1:] - v[:-1]).reshape(-1, 2),
                            (v[:, 1:] - v[:, :-1]).reshape(-1, 2)])
    variation = np.linalg.norm(edges, axis=1)
    return np.asarray([mag.mean(), mag.max(), mag.std(ddof=0), variation.mean(),
                       variation.max(), np.linalg.norm(v.mean((0, 1))) / (mag.mean() + np.float32(1e-6))], dtype=np.float32)


def richer_statistics(cells):
    cells = np.asarray(cells, dtype=np.float32)
    if cells.shape != (3, 3, 384) or not np.isfinite(cells).all():
        raise ValueError('Finite 3x3x384 cells required')
    grouped = cells.reshape(3, 3, 16, 24).mean(-1)
    edges = np.concatenate([np.abs(grouped[1:] - grouped[:-1]).reshape(-1, 16),
                            np.abs(grouped[:, 1:] - grouped[:, :-1]).reshape(-1, 16)])
    return np.concatenate([grouped.mean((0, 1)), edges.mean(0)])


def feature_families(ordinary, cells, flow_projection, pca_projection):
    cells = np.asarray(cells)
    if cells.shape != (10, 14, 384):
        raise ValueError('Expected 10x14x384 cell descriptors including tail')
    by_tile = {tuple(s['tile']): s for s in ordinary}
    base = np.asarray([[by_tile[t][k] for k in BASE_FEATURES] for t in TILES], dtype=np.float32)
    rich, flow, pca = [], [], []
    for r, c in TILES:
        patch = cells[r*3:(r+1)*3, c*3:(c+1)*3]
        rich.append(richer_statistics(patch))
        flow.append(vector_statistics(flow_projection[r*3:(r+1)*3, c*3:(c+1)*3]))
        pca.append(vector_statistics(pca_projection[r*3:(r+1)*3, c*3:(c+1)*3]))
    return {'B5': base, 'rich37': np.c_[base, rich],
            'flow11': np.c_[base, flow], 'PCA11': np.c_[base, pca]}


def top1(values):
    a = np.asarray(values)
    if a.shape != (12,) or not np.isfinite(a).all():
        raise ValueError('Twelve finite scores required')
    return min(range(12), key=lambda i: (-float(a[i]), TILES[i]))


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value
