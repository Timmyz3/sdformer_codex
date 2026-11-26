"""R18 fixed, label-free ordinary descriptors of cached R17 native tensors.

No model imports, device code, fitting, GT, endpoint loss or score-direction search.
"""
import hashlib
import math
import numpy as np

TILES = tuple((r, c) for r in range(3) for c in range(4))
SCORES = ('amplitude_abs_mean', 'feature_rms', 'pooling_residual_mse',
          'spatial_neighbor_abs_difference', 'temporal_neighbor_abs_difference',
          'psn1_activity', 'psn1_spatial_neighbor_hamming')
IDS = tuple('zurich_city_'+s for s in
            ('07_a_0095','09_a_0581','02_e_0061','06_a_0130',
             '11_c_0768','10_b_0038','05_b_0026','02_c_0284'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def payload_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def tile_scores(x, spikes):
    require(x.ndim == 5 and x.shape[0] >= 2 and x.shape[1] == 1 and
            x.shape[2:4] == (9, 9) and spikes.shape == x.shape, 'Whole T/B/9x9/C tile')
    require(x.dtype == np.float32 and spikes.dtype == np.float32, 'Original FP32 captures')
    require(np.isfinite(x).all() and np.isfinite(spikes).all(), 'Finite captures')
    require(((spikes == 0) | (spikes == 1)).all(), 'Binary native PSN capture')
    # Cast before all arithmetic; no FP32 intermediate pooling or squaring.
    a = np.ascontiguousarray(x, dtype=np.float64)
    s = np.ascontiguousarray(spikes, dtype=np.int64)
    t, b, _, _, c = a.shape
    representatives = a.reshape(t, b, 3, 3, 3, 3, c).mean(axis=(3, 5), dtype=np.float64)
    restored = representatives.repeat(3, axis=2).repeat(3, axis=3)
    spatial = ((np.abs(a[:, :, 1:] - a[:, :, :-1]).mean(dtype=np.float64) +
                np.abs(a[:, :, :, 1:] - a[:, :, :, :-1]).mean(dtype=np.float64))/2.)
    hamming = ((np.count_nonzero(s[:, :, 1:] != s[:, :, :-1]) / s[:, :, 1:].size +
                np.count_nonzero(s[:, :, :, 1:] != s[:, :, :, :-1]) / s[:, :, :, 1:].size)/2.)
    values = (np.abs(a).mean(dtype=np.float64), np.sqrt(np.square(a).mean(dtype=np.float64)),
              np.square(a-restored).mean(dtype=np.float64), spatial,
              np.abs(a[1:]-a[:-1]).mean(dtype=np.float64),
              int(s.sum(dtype=np.int64))/s.size, hamming)
    require(all(math.isfinite(float(v)) for v in values), 'Finite descriptor values')
    return dict(zip(SCORES, map(float, values)))


def operation_ledger(shape):
    """Algebraic scalar operations/operand reads, not NumPy/CPU instruction counts.

    Reduction of n values is counted as n-1 additions independent of tree order.
    Reads below are original descriptor operand reads after conversion, not DRAM.
    Temporary-array reductions/materialization are separately identified.
    """
    t, b, h, w, c = shape
    require(t >= 2 and b == 1 and (h, w) == (9, 9) and c > 0, 'Ledger tile geometry')
    n=t*b*h*w*c; q=t*b*9*c; v=t*b*8*9*c; e=(t-1)*b*81*c
    by = {
      SCORES[0]: dict(u_reads=n, sn1_reads=0, absolute=n, add=n-1, divide=1),
      SCORES[1]: dict(u_reads=n, sn1_reads=0, multiply=n, add=n-1, divide=1, sqrt=1),
      SCORES[2]: dict(u_reads=2*n, sn1_reads=0, representative_reads=n,
                     add=8*q+n-1, divide=q+1, subtract=n, multiply=n,
                     representative_elements=q, expanded_representative_write_elements=n),
      SCORES[3]: dict(u_reads=4*v, sn1_reads=0, subtract=2*v, absolute=2*v,
                     add=2*(v-1)+1, divide=3),
      SCORES[4]: dict(u_reads=2*e, sn1_reads=0, subtract=e, absolute=e, add=e-1, divide=1),
      SCORES[5]: dict(u_reads=0, sn1_reads=n, integer_add=n-1, divide=1),
      SCORES[6]: dict(u_reads=0, sn1_reads=4*v, not_equal=2*v,
                     boolean_count_elements=2*v, integer_add=2*(v-1), add=1, divide=3),
    }
    return dict(scope='Algebraic scalar operand/reduction ledger, NOT measured instructions, bus bytes or cycles',
                tile_shape=list(shape), input_elements_each=n, representative_elements=q,
                spatial_edges_each_direction=v, temporal_edges=e, by_score=by,
                conversion=dict(u_float32_to_float64_elements=n, sn1_float32_to_int64_elements=n,
                                original_read_bytes=8*n, converted_write_bytes=16*n),
                validation_excluded_from_production_score=dict(finite_scans=2*n, binary_tests=2*n),
                temporary_reduction_reads_not_in_original_operand_reads=True,
                additional_PSN_calls=0, additional_FC_calls=0, additional_model_forwards=0)


def policy(tile=None, uniform=False):
    if uniform:
        require(tile is None, 'Uniform distribution has no selected tile')
        return dict(kind='expected_uniform_randomized_k1', selected_tile=None, mask=None,
                    weights=[1./12]*12, rational_weight=dict(numerator=1, denominator=12),
                    endpoint_arms=['fine_%d_%d'%p for p in TILES],
                    interpretation='Expectation of losses of 12 actual singleton endpoints; never average predicted flows')
    tile=tuple(tile)
    require(tile in TILES, 'Only twelve complete tiles are selectable')
    mask=[[False]*5 for _ in range(4)]; mask[tile[0]][tile[1]]=True
    return dict(kind='deterministic_k1', selected_tile=list(tile), mask=mask,
                endpoint_arm='fine_%d_%d'%tile,
                weights=[int(p==tile) for p in TILES])


def select(scores, name):
    require(name in SCORES and [tuple(r['tile']) for r in scores] == list(TILES), 'Fixed score/tile order')
    require(all(set(r)=={'tile',*SCORES} and all(math.isfinite(r[n]) for n in SCORES)
                for r in scores), 'Seven finite scores for every tile')
    best=0
    # Strict > preserves the first row-major coordinate on exact ties. No epsilon.
    for i in range(1, 12):
        if scores[i][name] > scores[best][name]: best=i
    return policy(TILES[best])


def score_frame(u, spikes):
    require(u.shape == spikes.shape and u.ndim == 5 and u.shape[:4] == (10,1,30,40), 'R17 T/B/H/W capture')
    require(u.dtype == spikes.dtype == np.float32, 'R17 FP32 dtype')
    records=[]
    for r,c in TILES:
        x=u[:,:,r*9:(r+1)*9,c*9:(c+1)*9,:]
        s=spikes[:,:,r*9:(r+1)*9,c*9:(c+1)*9,:]
        records.append(dict(tile=[r,c], **tile_scores(x,s)))
    policies={name:select(records,name) for name in SCORES}
    policies.update({'fixed_%d_%d'%tile:policy(tile) for tile in TILES})
    policies['uniform_random_expected']=policy(uniform=True)
    return dict(scores=records, policies=policies,
                per_tile_operation_ledger=operation_ledger((10,1,9,9,u.shape[-1])),
                ledger_tile_multiplicity=12,
                selector=dict(score_comparisons=7*11, comparisons_per_score=11,
                              tie_rule='exact > scan in row-major order; no epsilon',
                              selected_score_policies=7, fixed_policies=12, expectation_policies=1))
