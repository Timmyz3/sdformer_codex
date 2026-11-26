"""R27 original-array metrics. Fixed R13 region definitions, no GT refitting.

Provenance: coverage_recovery_round13/independent_quality_review/
reduce_raw_quality.py SHA26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088.
The per-pixel event quantity below counts nonzero voxel bins, NOT raw events.
"""
import hashlib
import numpy as np
from analysis_core import REGIONS, require

def payload_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def source_checks(gt, valid, events):
    require(gt.dtype == np.float32 and valid.dtype == np.bool_ and events.dtype == np.float32,
            'Original float32 GT/events and bool mask')
    require(gt.ndim == 3 and gt.shape[0] == 2 and valid.shape == gt.shape[1:] and
            events.shape == (10,)+valid.shape, 'Full spatial original input shapes')
    require(np.isfinite(gt).all() and np.isfinite(events).all(), 'Finite original arrays')
    require(bool(valid.any()), 'Positive all-valid support; no replacement sampling')

def region_masks(gt, valid, events, boundary_threshold, large_threshold):
    source_checks(gt, valid, events)
    require(np.isfinite(boundary_threshold) and np.isfinite(large_threshold) and
            boundary_threshold >= 0 and large_threshold >= 0, 'Fixed finite positive thresholds')
    v = gt.astype(np.float64)
    h,w = valid.shape
    known = np.zeros((h,w), dtype=bool)
    jump = np.zeros((h,w), dtype=np.float64)
    if h >= 3 and w >= 3:
        known[1:-1,1:-1] = (valid[1:-1,1:-1] & valid[:-2,1:-1] & valid[2:,1:-1]
                            & valid[1:-1,:-2] & valid[1:-1,2:])
        center = v[:,1:-1,1:-1]
        for neighbor in (v[:,:-2,1:-1],v[:,2:,1:-1],v[:,1:-1,:-2],v[:,1:-1,2:]):
            diff = center-neighbor
            np.maximum(jump[1:-1,1:-1],np.sqrt(diff[0]**2+diff[1]**2),out=jump[1:-1,1:-1])
    B = known & (jump > boundary_threshold)
    L = valid & (np.sqrt(v[0]**2+v[1]**2) > large_threshold)
    active = np.any(events != 0,axis=0)
    masks = dict(all_valid=valid,B=B,L=L,B_and_L=B&L,B_only=B&~L,
                 L_only_B_known=L&known&~B,neither_B_known=valid&known&~B&~L,
                 B_unknown=valid&~known,voxel_nonzero=valid&active,voxel_zero=valid&~active)
    for key in ('B','L','B_unknown'):
        masks[key+'_voxel_nonzero']=masks[key]&active
        masks[key+'_voxel_zero']=masks[key]&~active
    require(set(masks)==set(REGIONS) and all(m.dtype==np.bool_ and not np.any(m&~valid) for m in masks.values()),
            'Complete16 GT-subset masks')
    return masks

def measure(pred, gt, masks):
    require(pred.dtype==np.float32 and pred.shape==gt.shape and np.isfinite(pred).all(),
            'Finite native-resolution pixel-unit prediction')
    d = pred.astype(np.float64)-gt.astype(np.float64)
    e = np.sqrt(d[0]*d[0]+d[1]*d[1])
    out = {}
    for key in REGIONS:
        n = int(masks[key].sum())
        s = float(np.sum(e[masks[key]],dtype=np.float64))
        out[key] = dict(valid_pixels=n,error_sum=s,AEE=s/n if n else None)
    return out

def artifact_arrays(gt, valid, events, masks):
    """Shared original GT + derived support, saved once perframe after admission."""
    return dict(gt=gt,mask=valid,
                voxel_nonzero_bins=np.count_nonzero(events,axis=0).astype(np.uint8),
                **{'region__'+name:masks[name] for name in REGIONS})
