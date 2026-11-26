"""R21 candidate mathematics. No file access, GPU, GT lookup or fitting on import.

Local mechanism adaptation; neither an ADAPT implementation nor a novelty claim.
All tensors used by predictors precede the cancelled target MLP producer.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TILES = tuple((r, c) for r in range(3) for c in range(4))
SEED = 20261017
STEPS = 400


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pool_expand3(x):
    """N,H,W,C, true tails included; fixed float32 summation then division."""
    x = np.asarray(x, dtype=np.float32)
    require(x.ndim == 4 and np.isfinite(x).all(), 'Finite NHWC')
    out = np.empty_like(x)
    for y in range(0, x.shape[1], 3):
        for z in range(0, x.shape[2], 3):
            a = x[:, y:y+3, z:z+3]
            mean = a.sum((1, 2), dtype=np.float32) / np.float32(a.shape[1]*a.shape[2])
            out[:, y:y+3, z:z+3] = mean[:, None, None]
    return out


def pixel_labels(gt, valid):
    """Offline fit only. 16x16 output support, pixel displacement units unchanged."""
    require(gt.shape == (2, 480, 640) and valid.shape == (480, 640), 'DSEC target geometry')
    require(np.isin(valid, (0, 1)).all(), 'Binary original validity')
    m = valid.astype(bool)
    require(np.isfinite(gt[:, m]).all(), 'Finite valid GT')
    counts = m.reshape(30, 16, 40, 16).sum((1, 3))
    masked = np.where(m[None], gt, 0.).astype(np.float64)
    sums = masked.reshape(2, 30, 16, 40, 16).sum((2, 4)).transpose(1, 2, 0)
    out = sums / np.maximum(counts, 1)[..., None]
    require(int(counts.sum()) == int(m.sum()), 'All valid pixels retained')
    return out, counts


def ridge_motion(x, truth, counts, alpha=10.):
    """Fit-side standardization, count-weighted ridge, normalized mean weight 1.

    Equal effective pixel support; empty cells never enter the fit.
    Predictions use the exported folded float32 operator.
    """
    x = np.asarray(x, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    require(x.shape[:-1] == truth.shape[:-1] == counts.shape and x.shape[-1] == 384 and truth.shape[-1] == 2,
            'Full-grid projection dimensions')
    keep = counts > 0
    a, y, w = x[keep], truth[keep], counts[keep]
    require(len(a) > 384 and np.isfinite(a).all() and np.isfinite(y).all(), 'Adequate finite fit cells')
    w = w / w.mean()
    mean = (a*w[:, None]).sum(0) / w.sum()
    std = np.sqrt(((a-mean)**2*w[:, None]).sum(0)/w.sum())
    std = np.where(std > 1e-6, std, 1.)
    z = (a-mean)/std
    intercept = (y*w[:, None]).sum(0)/w.sum()
    beta = np.linalg.solve(z.T@(z*w[:, None])+float(alpha)*np.eye(384), z.T@((y-intercept)*w[:, None]))
    weight = beta/std[:, None]
    bias = intercept-mean@weight
    return dict(weight=weight.astype(np.float32), bias=bias.astype(np.float32),
                fit_cells=int(len(a)), fit_valid_pixels=int(counts.sum()), alpha=float(alpha))


def project(model, x):
    x = np.asarray(x, dtype=np.float32)
    out = x@model['weight']+model['bias']
    require(out.shape == x.shape[:-1]+(2,) and np.isfinite(out).all(), 'Finite FP32 motion proxy')
    return out.astype(np.float32)


def fixed_groups(x):
    x = np.asarray(x, dtype=np.float32)
    require(x.shape[-1] == 384, 'Fixed full prefix channels')
    return x.reshape(*x.shape[:-1], 16, 24).mean(-1, dtype=np.float32)


def spatial_inputs(x, q):
    """Input-only: generic full-grid groups and q/r, plus equal global prefix context.

    No loss, GT or validity argument. xy is fixed block-relative coordinate.
    q may be OOF or final fit-only; both flow methods consume identical arrays.
    """
    x, q = np.asarray(x, dtype=np.float32), np.asarray(q, dtype=np.float32)
    require(x.shape[1:] == (30, 40, 384) and q.shape == x.shape[:-1]+(2,), 'Full original grid')
    z = fixed_groups(x)
    zr, qr = z-pool_expand3(z), q-pool_expand3(q)
    # All methods get the same global *generic* context. This contains true tails.
    global_context = np.concatenate((z.mean((1, 2)), z.std((1, 2))), -1).astype(np.float32)
    ordinary, raw, motion, contexts = [], [], [], []
    yy, xx = np.meshgrid(np.linspace(-1, 1, 9, dtype=np.float32), np.linspace(-1, 1, 9, dtype=np.float32), indexing='ij')
    xy = np.stack((xx, yy), -1)
    for i in range(len(x)):
        for r, c in TILES:
            sl = (i, slice(9*r, 9*r+9), slice(9*c, 9*c+9))
            ordinary.append(np.concatenate((z[sl], zr[sl], xy), -1))
            raw.append(np.concatenate((x[sl], xy), -1))
            motion.append(np.concatenate((q[sl], qr[sl], xy), -1))
            contexts.append(np.r_[global_context[i], np.float32(r/2), np.float32(c/3)])
    return {'ordinary': np.asarray(ordinary, dtype=np.float32).transpose(0, 3, 1, 2),
            'raw': np.asarray(raw, dtype=np.float32).transpose(0, 3, 1, 2),
            'motion': np.asarray(motion, dtype=np.float32).transpose(0, 3, 1, 2),
            'context': np.asarray(contexts, dtype=np.float32)}


def fit_scaler(x, axes):
    x = np.asarray(x, dtype=np.float32)
    mean, std = x.mean(axes, keepdims=True), x.std(axes, keepdims=True)
    std = np.where(std > 1e-6, std, np.float32(1.))
    return {'mean': mean, 'std': std}


def fit_motion_scaler(x):
    """One shared scalar for qx/qy/rx/ry preserves vector direction and units.

    A constant translation centers q only. Residuals and fixed xy are not centered.
    G2 and F use byte-identical exported scaling from the same OOF fit tensors.
    """
    require(x.ndim == 4 and x.shape[1:] == (6, 9, 9), 'Motion q/r/xy tensor')
    mean = np.zeros((1, 6, 1, 1), np.float32)
    mean[:, :2] = x[:, :2].mean((0, 2, 3), keepdims=True)
    common = np.sqrt(np.square(x[:, :2]-mean[:, :2]).mean(dtype=np.float32))
    common = np.float32(common if common > 1e-6 else 1.)
    std = np.ones_like(mean); std[:, :4] = common
    return {'mean': mean, 'std': std}


def scale(x, scaler):
    return ((np.asarray(x, dtype=np.float32)-scaler['mean'])/scaler['std']).astype(np.float32)


class OrdinaryCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(channels, 8, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(72+34, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x, context):
        spatial = F.avg_pool2d(self.encoder(x), 3).flatten(1)
        return self.head(torch.cat((spatial, context), 1)).squeeze(1)


class OrdinaryPrefixCNN(nn.Module):
    """Strong local control: learn the 384->8 projection from identical gain labels."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(384, 8, 1)
        self.rest = OrdinaryCNN(18)

    def forward(self, x, context):
        z = self.projection(x[:, :384])
        coarse = F.avg_pool2d(z, 3).repeat_interleave(3, 2).repeat_interleave(3, 3)
        return self.rest(torch.cat((z, z-coarse, x[:, 384:]), 1), context)


class MotionEdges(nn.Module):
    """Shared nonlinear node/edge embedding before true3x3 grouping.

    Same raw q/r/xy and global context as OrdinaryCNN(6). Difference terms
    deterministic from that information, not privileged motion observations.
    """
    def __init__(self):
        super().__init__()
        self.node = nn.Linear(4, 4)
        self.edge = nn.Linear(12, 4)
        self.head = nn.Sequential(nn.Linear(72+34, 8), nn.ReLU(), nn.Linear(8, 1))
        # Directed right/down edges, bin by source coordinate, no wrapped edge.
        src, dst, direction, group = [], [], [], []
        for y in range(9):
            for x in range(9):
                for dy, dx in ((0, 1), (1, 0)):
                    if y+dy < 9 and x+dx < 9:
                        src.append(9*y+x); dst.append(9*(y+dy)+x+dx)
                        direction.append((dx, dy)); group.append((y//3)*3+x//3)
        self.register_buffer('src', torch.tensor(src, dtype=torch.long))
        self.register_buffer('dst', torch.tensor(dst, dtype=torch.long))
        self.register_buffer('direction', torch.tensor(direction, dtype=torch.float32))
        membership = F.one_hot(torch.tensor(group), num_classes=9).float()
        membership = membership/membership.sum(0, keepdim=True)
        self.register_buffer('membership', membership)

    def forward(self, x, context):
        raw = x[:, :4].flatten(2).transpose(1, 2)
        node = F.relu(self.node(raw)).transpose(1, 2).reshape(-1, 4, 9, 9)
        node = F.avg_pool2d(node, 3).flatten(1)
        p, q = raw[:, self.src], raw[:, self.dst]
        edge = torch.cat((p, q, p[..., :2]-q[..., :2], self.direction.expand(len(x), -1, -1)), -1)
        encoded = F.relu(self.edge(edge))
        edge = (encoded.transpose(1, 2)@self.membership).flatten(1)
        return self.head(torch.cat((node, edge, context), 1)).squeeze(1)


def fit_predictor(kind, spatial, context, target):
    """Fixed single-seed full-batch AdamW; no validation argument/early stopping."""
    require(kind in ('generic_raw', 'generic_group', 'same_motion', 'structured'), 'Declared variants only')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model = (MotionEdges() if kind == 'structured' else
             OrdinaryPrefixCNN() if kind == 'generic_raw' else OrdinaryCNN(spatial.shape[1]))
    model.train()
    xs = fit_motion_scaler(spatial) if kind in ('same_motion', 'structured') else fit_scaler(spatial, (0, 2, 3))
    cs = fit_scaler(context, (0,))
    x, c = torch.from_numpy(scale(spatial, xs)), torch.from_numpy(scale(context, cs))
    y = np.asarray(target, dtype=np.float32)
    require(y.shape == (len(x),) and np.isfinite(y).all() and float(y.std()) > 1e-10, 'Nonconstant signed gain')
    ym, ys = np.float32(y.mean()), np.float32(y.std())
    yy = torch.from_numpy((y-ym)/ys)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    losses = []
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        pred = model(x, c)
        loss = F.mse_loss(pred, yy)
        require(bool(torch.isfinite(loss)), 'Finite training objective')
        loss.backward()
        opt.step()
        if step in (0, 99, 199, 299, 399):
            losses.append({'step': step+1, 'normalized_fit_mse': float(loss.detach())})
    model.eval()
    return dict(model=model, spatial_scaler=xs, context_scaler=cs,
                target_mean=ym, target_std=ys, fit_loss=losses,
                parameter_count=sum(v.numel() for v in model.parameters()), steps=STEPS, seed=SEED)


def infer_predictor(fitted, spatial, context):
    with torch.inference_mode():
        y = fitted['model'](torch.from_numpy(scale(spatial, fitted['spatial_scaler'])),
                            torch.from_numpy(scale(context, fitted['context_scaler'])))
    out = y.numpy()*fitted['target_std']+fitted['target_mean']
    require(np.isfinite(out).all(), 'Finite signed prediction')
    return out


def top1(x):
    x = np.asarray(x)
    require(x.shape == (12,) and np.isfinite(x).all(), '12 finite scores')
    return int(np.argmax(x))  # exact first maximum; fixed row-major order.


def state_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().numpy().tobytes())
    return h.hexdigest()
