"""Predeclared Phi-basis / pure signed-residual operation and request screen.

This is a causal, native-row-order reference, not a cycle or PPA model. The
unchanged basis_residual_plan.json defines the cohort and configuration sweep.
Resolved details, agreed with the coordinating agent before execution:
* Reset RNG seed 1701 independently for each (operator, K, requested q).
  Cluster weighted distinct supports; merge duplicate updated centers.
* Compile each base's largest nonzero proper-subset center as its parent;
  ties choose the smallest numeric mask. A missing parent is not replayed.
* Base and residual are separate object types in ONE LRU value budget. A pure
  residual key contains both signed masks, but never its generating base ID.
  No extra cross-type value aliasing or full-output cache is introduced.
* Independent L2 constructs delta from zero and then joins the base. The first
  operand assignment is free arithmetically; its read and negation are charged.
* Zero/one-hot original rows bypass. An unassigned nontrivial row has base zero
  and delta=S, and can use the residual mechanism in the residual variants.
* Process the base first, then the residual, using two explicitly reserved
  vector registers outside the LRU cache. Register copies never alias cache
  entries. Every original destination, including zero rows, is committed.
* Zero centers are constants. A one-hot center always reads the original raw
  weight in ALL variants; it never creates a PWP fetch or cache entry.

All clustering and scheduling are CPU research operations. Exact diagnostic
values have denominator 10**6, theta_c=(999883+17*c)/10**6, and signed weights.
They are not checkpoint weights, frozen FP32 equivalence, or an AEE evaluation.
"""

import sys

sys.dont_write_bytecode = True

from collections import Counter, OrderedDict
from pathlib import Path
import hashlib
import itertools
import json

import numpy as np

from screen_joint_residual import LEDGER, LEDGER_SHA, digest


BASE = Path(__file__).resolve().parent
PLAN = BASE / 'basis_residual_plan.json'
OUT = BASE / 'basis_residual_r1.json'
MASK = (1 << 16) - 1
LANES = 8
POP = np.array([v.bit_count() for v in range(1 << 16)], dtype=np.uint8)
VARIANTS = (
    'pwp_independent_l2',
    'ondemand_base_independent_l2',
    'ondemand_base_exact_residual',
    'ondemand_base_subset_residual',
)
RESOLVED = {
    'rng': 'Seed 1701 reset independently for every operator/K/requested-q model.',
    'duplicate_centers': 'Merge and sort after every update; actual center count may shrink.',
    'base_graph': 'Largest nonzero proper-subset center; ties smallest numeric mask; no virtual bases; nonresident parent means direct current-base construction.',
    'cache_keys': 'Separate base and residual object types, one shared LRU capacity. R key=(positive_mask,negative_mask), independent of base. No cross-type aliases.',
    'lru': 'Only actual value reads and new writes update recency; tag lookups and failed subset comparisons do not. New entries evict the least recently used completed value.',
    'residual_parent_tie': 'Largest resident same-sign subset, ties lexicographic (positive_mask,negative_mask); exact hit is checked first.',
    'independent_l2': 'Construct delta from zero then join base; first operand assignment free but read/negation charged. No persistent residual admission in independent-L2 variants.',
    'unassigned_rows': 'Nontrivial unassigned rows use base=0 and delta=S; residual variants may cache/reuse these R objects.',
    'zero_and_singletons': 'Original zero/one-hot rows bypass. One-hot centers use one raw weight read in every variant; zero centers use constant zero. Neither becomes a cache object or external PWP request.',
    'registers': 'Exactly two reserved vector registers outside cache: base/join, then residual/compute. Base first; its copied value survives any subsequent cache eviction.',
    'admission': 'Cache every fetched/computed nontrivial base. Cache every newly computed residual of length >=2 only in residual variants; exact hits do not rewrite.',
    'weight_formula': 'Eight lanes: (((c+3)*(h+5)*13)%256)-128; overwrite lane0=-128 and lane1=127 to exercise range extremes.',
    'threshold_formula': 'theta_c=(999883+17*c)/1000000; numerator arithmetic in int64, with common denominator 1000000.',
    'auxiliary_metadata': 'Origin base in cached R entries is audit-only provenance and never affects selection, replacement, or arithmetic.',
    'pwp_scope': 'PWP values precomputed externally only for the first variant. Report their runtime requests and separate offline full-center build arithmetic. Unique touched nontrivial bases are also reported as a per-tile prefetch lower bound, not as measured traffic.',
    'calibration_sweep_order': 'operator, K partition, requested q; the same calibrated centers and row assignment are supplied to every variant/capacity.',
}


def diagnostic_weights():
    weights = np.array([
        [((c + 3) * (h + 5) * 13) % 256 - 128 for h in range(LANES)]
        for c in range(16)
    ], dtype=np.int64)
    weights[:, 0] = -128
    weights[:, 1] = 127
    theta_num = np.arange(16, dtype=np.int64) * 17 + 999883
    return weights, weights * theta_num[:, None]


WEIGHTS, PHI = diagnostic_weights()


def vector_value(positive, negative=0):
    """Uncounted oracle/PWP value; never used to hide an executed source read."""
    assert positive & negative == 0
    result = np.zeros(LANES, dtype=np.int64)
    for source in range(16):
        if positive >> source & 1:
            result += PHI[source]
        elif negative >> source & 1:
            result -= PHI[source]
    return result


def read_rows(stream, sample, operator, partition, start=0, count=3000):
    assert 0 <= start < 3000 and 0 < count <= 3000 - start
    phase = (sample * 4 + operator) * 432 + partition
    stream.seek((phase * 3000 + start) * 9)
    raw = stream.read(count * 9)
    lines = raw.splitlines()
    assert len(lines) == count and all(len(line) == 8 for line in lines)
    masks = [int(line, 16) for line in lines]
    assert all(0 <= mask <= MASK for mask in masks)
    return masks, raw


def calibrate(histogram, requested_q):
    supports = np.array(sorted(m for m in histogram if m.bit_count() >= 2),
                        dtype=np.uint16)
    frequency = np.array([histogram[int(m)] for m in supports], dtype=np.int64)
    metadata = {
        'requested_q': requested_q,
        'calibration_rows': int(sum(histogram.values())),
        'included_nontrivial_rows': int(frequency.sum()),
        'included_distinct_supports': len(supports),
        'excluded_zero_rows': int(histogram[0]),
        'excluded_onehot_rows': int(sum(n for m, n in histogram.items()
                                      if m.bit_count() == 1)),
    }
    if not len(supports):
        metadata.update(actual_q=0, iterations=0, converged=True,
                        centers=[], final_weighted_hamming_objective=0,
                        center_calibration_weight=[], zero_centers=0,
                        onehot_centers=0, offline_full_pwp_nontrivial_vectors=0,
                        offline_full_pwp_source_reads=0,
                        offline_full_pwp_binary_vector_adds=0)
        return [], metadata
    rng = np.random.default_rng(1701)
    chosen = rng.choice(len(supports), size=min(requested_q, len(supports)),
                        replace=False, p=frequency / frequency.sum())
    centers = np.sort(supports[chosen])
    bits = ((supports[:, None] >> np.arange(16, dtype=np.uint16)) & 1)
    converged = False
    for iteration in range(1, 31):
        distance = POP[np.bitwise_xor(supports[:, None], centers[None, :])]
        assignments = np.argmin(distance, axis=1)
        updated = []
        for j, center in enumerate(centers):
            member = assignments == j
            total = int(frequency[member].sum())
            if not total:
                updated.append(int(center))
                continue
            weighted_ones = (bits[member] * frequency[member, None]).sum(axis=0)
            # Exact half rounds to ZERO, with integer arithmetic throughout.
            updated.append(sum(1 << bit for bit in range(16)
                               if 2 * int(weighted_ones[bit]) > total))
        next_centers = np.array(sorted(set(updated)), dtype=np.uint16)
        if np.array_equal(next_centers, centers):
            centers = next_centers
            converged = True
            break
        centers = next_centers
    distance = POP[np.bitwise_xor(supports[:, None], centers[None, :])]
    assignments = np.argmin(distance, axis=1)
    objective = int((distance[np.arange(len(supports)), assignments] * frequency).sum())
    center_weight = [int(frequency[assignments == j].sum())
                     for j in range(len(centers))]
    result = [int(c) for c in centers]
    metadata.update(actual_q=len(result), iterations=iteration, converged=converged,
                    centers=result, final_weighted_hamming_objective=objective,
                    center_calibration_weight=center_weight,
                    zero_centers=sum(c == 0 for c in result),
                    onehot_centers=sum(c.bit_count() == 1 for c in result),
                    offline_full_pwp_nontrivial_vectors=sum(c.bit_count() >= 2 for c in result),
                    offline_full_pwp_source_reads=sum(c.bit_count() for c in result if c.bit_count() >= 2),
                    offline_full_pwp_binary_vector_adds=sum(c.bit_count() - 1 for c in result if c.bit_count() >= 2))
    return result, metadata


def compile_bases(centers):
    parents = {}
    for base in centers:
        candidates = [p for p in centers if p and p != base and p & base == p]
        parents[base] = (min(candidates, key=lambda p: (-p.bit_count(), p))
                         if candidates else 0)
    return parents


def assign_rows(masks, centers):
    descriptors = []
    stats = Counter(original_rows=len(masks),
                    nonzero_original_rows=sum(bool(m) for m in masks),
                    original_source_terms=sum(m.bit_count() for m in masks),
                    raw_binary_vector_adds=sum(max(0, m.bit_count() - 1) for m in masks))
    lengths = Counter()
    bases_used = set()
    for mask in masks:
        base = 0
        if mask.bit_count() >= 2 and centers:
            candidate = min(centers, key=lambda p: ((mask ^ p).bit_count(), p))
            if (mask ^ candidate).bit_count() < mask.bit_count():
                base = candidate
        positive, negative = mask & ~base & MASK, base & ~mask & MASK
        assert positive & negative == 0
        assert mask == ((base & ~negative) | positive)
        descriptors.append((base, positive, negative))
        if mask.bit_count() >= 2:
            stats['nontrivial_original_rows'] += 1
            stats['assigned_base_rows'] += bool(base)
            stats['assigned_nontrivial_base_rows'] += base.bit_count() >= 2
            stats['assigned_onehot_base_rows'] += base.bit_count() == 1
            stats['residual_terms'] += positive.bit_count() + negative.bit_count()
            stats['negative_residual_terms'] += negative.bit_count()
            lengths[positive.bit_count() + negative.bit_count()] += 1
            if base.bit_count() >= 2:
                bases_used.add(base)
    return descriptors, {
        **dict(stats),
        'residual_length_histogram': dict(sorted(lengths.items())),
        'unique_touched_nontrivial_bases_prefetch_lower_bound': len(bases_used),
    }


class WideCache:
    def __init__(self, capacity, counts):
        self.capacity = capacity
        self.counts = counts
        self.entries = OrderedDict()

    def read(self, key):
        if key not in self.entries:
            return None
        value, origin_base = self.entries.pop(key)
        self.entries[key] = (value, origin_base)
        self.counts['cache_reads'] += 1
        self.counts[f'cache_{key[0]}_reads'] += 1
        return value.copy(), origin_base

    def write(self, key, value, origin_base=0):
        assert key not in self.entries, 'An exact cache hit must not be rewritten.'
        if len(self.entries) == self.capacity:
            old, _ = self.entries.popitem(last=False)
            self.counts['cache_evictions'] += 1
            self.counts[f'cache_{old[0]}_evictions'] += 1
        self.entries[key] = (value.copy(), origin_base)
        self.counts['cache_writes'] += 1
        self.counts[f'cache_{key[0]}_writes'] += 1
        self.counts['peak_shared_cache_vectors'] = max(
            self.counts['peak_shared_cache_vectors'], len(self.entries))
        assert len(self.entries) <= self.capacity

    def signed_parent(self, positive, negative):
        candidates = [key for key in self.entries
                      if key[0] == 'residual' and key[1] & positive == key[1]
                      and key[2] & negative == key[2]]
        if not candidates:
            return None
        return min(candidates, key=lambda k: (-(k[1].bit_count() + k[2].bit_count()),
                                             k[1], k[2]))


def accumulate_sources(positive, negative, counts, label, initial=None):
    """One explicit vector register: cache initialization or source assignment."""
    assert positive & negative == 0
    result = np.zeros(LANES, dtype=np.int64) if initial is None else initial
    initialized = initial is not None
    for source in range(16):
        sign = 1 if positive >> source & 1 else (-1 if negative >> source & 1 else 0)
        if not sign:
            continue
        counts['source_weight_reads'] += 1
        counts[f'{label}_source_reads'] += 1
        counts['signed_source_negations'] += sign < 0
        if initialized:
            if sign > 0:
                result += PHI[source]
            else:
                result -= PHI[source]
            counts['binary_vector_adds'] += 1
            counts[f'{label}_source_adds'] += 1
        else:
            # No uncounted second vector register or cached source product.
            result[:] = PHI[source]
            if sign < 0:
                result *= -1
            initialized = True
    return result


COUNT_FIELDS = (
    'binary_vector_adds', 'base_source_adds', 'residual_source_adds',
    'bypass_source_adds', 'base_residual_join_adds',
    'source_weight_reads', 'base_source_reads', 'residual_source_reads',
    'bypass_source_reads', 'signed_source_negations',
    'external_pwp_vector_requests', 'cache_reads', 'cache_writes',
    'cache_evictions', 'cache_base_reads', 'cache_base_writes',
    'cache_base_evictions', 'cache_residual_reads', 'cache_residual_writes',
    'cache_residual_evictions', 'base_exact_hits', 'base_graph_parent_hits',
    'base_graph_parent_misses_direct_build', 'base_graph_root_builds',
    'base_onehot_raw_reads', 'base_direct_builds', 'base_graph_builds',
    'residual_exact_hits', 'residual_subset_hits', 'residual_builds',
    'residual_direct_builds', 'residual_reads_with_changed_origin_base',
    'residual_reads_across_distinct_nonzero_bases',
    'residual_source_terms_without_reuse',
    'zero_original_bypasses', 'onehot_original_bypasses',
    'all_original_destination_commits', 'nonzero_destination_commits',
    'diagnostic_rational_lane_values_verified', 'diagnostic_mismatches',
    'peak_shared_cache_vectors', 'final_shared_cache_vectors',
)


def simulate(masks, descriptors, parents, capacity, variant):
    assert variant in VARIANTS
    counts = Counter({field: 0 for field in COUNT_FIELDS})
    cache = WideCache(capacity, counts)
    residual_enabled = variant in VARIANTS[2:]
    for row, (mask, descriptor) in enumerate(zip(masks, descriptors)):
        base, positive, negative = descriptor
        if mask.bit_count() <= 1:
            if not mask:
                result = np.zeros(LANES, dtype=np.int64)
                counts['zero_original_bypasses'] += 1
            else:
                result = accumulate_sources(mask, 0, counts, 'bypass')
                counts['onehot_original_bypasses'] += 1
        else:
            # Register 1: construct or fetch B, then retain it during all R work.
            base_register = np.zeros(LANES, dtype=np.int64)
            if base.bit_count() == 1:
                base_register = accumulate_sources(base, 0, counts, 'base')
                counts['base_onehot_raw_reads'] += 1
            elif base:
                key = ('base', base, 0)
                found = cache.read(key)
                if found is not None:
                    base_register, _ = found
                    counts['base_exact_hits'] += 1
                elif variant == VARIANTS[0]:
                    counts['external_pwp_vector_requests'] += 1
                    base_register = vector_value(base)
                    cache.write(key, base_register)
                else:
                    parent = parents[base]
                    parent_value = cache.read(('base', parent, 0)) if parent else None
                    if parent_value is not None:
                        counts['base_graph_parent_hits'] += 1
                        base_register = accumulate_sources(base ^ parent, 0, counts,
                                                           'base', parent_value[0])
                    else:
                        counts['base_direct_builds'] += 1
                        counts['base_graph_parent_misses_direct_build'] += bool(parent)
                        counts['base_graph_root_builds'] += not parent
                        base_register = accumulate_sources(base, 0, counts, 'base')
                    counts['base_graph_builds'] += 1
                    assert np.array_equal(base_register, vector_value(base))
                    cache.write(key, base_register)

            # Register 2: only completed resident pure residuals may initialize R.
            size = positive.bit_count() + negative.bit_count()
            counts['residual_source_terms_without_reuse'] += size
            residual_register = np.zeros(LANES, dtype=np.int64)
            if size:
                key = ('residual', positive, negative)
                found = cache.read(key) if residual_enabled and size >= 2 else None
                if found is not None:
                    residual_register, origin_base = found
                    counts['residual_exact_hits'] += 1
                    counts['residual_reads_with_changed_origin_base'] += origin_base != base
                    counts['residual_reads_across_distinct_nonzero_bases'] += bool(
                        origin_base and base and origin_base != base)
                else:
                    parent_key = (cache.signed_parent(positive, negative)
                                  if variant == VARIANTS[3] and size >= 2 else None)
                    if parent_key is not None:
                        parent_value, origin_base = cache.read(parent_key)
                        counts['residual_subset_hits'] += 1
                        counts['residual_reads_with_changed_origin_base'] += origin_base != base
                        counts['residual_reads_across_distinct_nonzero_bases'] += bool(
                            origin_base and base and origin_base != base)
                        residual_register = accumulate_sources(positive ^ parent_key[1],
                                                               negative ^ parent_key[2],
                                                               counts, 'residual', parent_value)
                    else:
                        counts['residual_direct_builds'] += 1
                        residual_register = accumulate_sources(positive, negative, counts, 'residual')
                    counts['residual_builds'] += 1
                    assert np.array_equal(residual_register, vector_value(positive, negative))
                    if residual_enabled and size >= 2:
                        cache.write(key, residual_register, origin_base=base)
            if base and size:
                base_register += residual_register
                counts['binary_vector_adds'] += 1
                counts['base_residual_join_adds'] += 1
                result = base_register
            else:
                result = base_register if base else residual_register

        expected = vector_value(mask)
        if not np.array_equal(result, expected):
            raise AssertionError((variant, capacity, row, mask, descriptor,
                                  result.tolist(), expected.tolist()))
        counts['all_original_destination_commits'] += 1
        counts['nonzero_destination_commits'] += bool(mask)
        counts['diagnostic_rational_lane_values_verified'] += LANES
    counts['final_shared_cache_vectors'] = len(cache.entries)
    assert counts['all_original_destination_commits'] == len(masks)
    assert counts['source_weight_reads'] == sum(counts[f'{p}_source_reads']
                                                for p in ('base', 'residual', 'bypass'))
    assert counts['binary_vector_adds'] == counts['base_residual_join_adds'] + sum(
        counts[f'{p}_source_adds'] for p in ('base', 'residual', 'bypass'))
    for operation in ('reads', 'writes', 'evictions'):
        assert counts[f'cache_{operation}'] == sum(counts[f'cache_{kind}_{operation}']
                                                  for kind in ('base', 'residual'))
    assert counts['cache_writes'] == counts['cache_evictions'] + len(cache.entries)
    assert counts['cache_base_reads'] == counts['base_exact_hits'] + counts['base_graph_parent_hits']
    assert counts['cache_residual_reads'] == counts['residual_exact_hits'] + counts['residual_subset_hits']
    assert counts['residual_source_reads'] <= counts['residual_source_terms_without_reuse']
    assert counts['peak_shared_cache_vectors'] <= capacity
    return dict(counts)


def add_counts(target, source):
    for key, value in source.items():
        if key.startswith('peak_'):
            target[key] = max(target[key], value)
        else:
            target[key] += value


def directed_checks(capacities):
    # Separate conformance cases, never pooled into held-out workload statistics.
    # Across unrelated bases: the same negative/positive delta, then its superset.
    bases = [0, 1, 0b111001, 0b110000001, 0b11000000001]
    dplus, dminus = 0b110, 1
    masks = [0, 1, bases[2] ^ dminus | dplus,
             bases[3] ^ dminus | dplus,
             bases[4] ^ dminus | dplus | (1 << 12)]
    # Force LRU churn with many distinct nontrivial base objects and revisits.
    pressure_centers = sorted({(1 << i) | (1 << j)
                               for i in range(8) for j in range(i + 1, 8)})
    pressure_masks = pressure_centers + pressure_centers[:5]
    # Shared signed-parent logic should never treat opposite signs as compatible.
    result = {}
    for label, rows, centers in (
            ('signed_cross_base_and_bypass', masks, sorted(bases)),
            ('cold_lru_eviction_and_revisit', pressure_masks, pressure_centers),
            ('dense_zero_and_sign_extremes', [0, MASK, 1, MASK ^ 1, 3, 7, MASK],
             [0, 1, 3, 7, MASK])):
        desc, structure = assign_rows(rows, centers)
        points = {}
        for capacity, variant in itertools.product(capacities, VARIANTS):
            points[f'{variant}/cap{capacity}'] = simulate(
                rows, desc, compile_bases(centers), capacity, variant)
        result[label] = {'supports': rows, 'centers': centers, 'structure': structure,
                         'variants': points}
    for capacity in capacities:
        point = result['cold_lru_eviction_and_revisit']['variants'][f'{VARIANTS[0]}/cap{capacity}']
        assert point['cache_evictions'] > 0
        signed = result['signed_cross_base_and_bypass']['variants']
        assert signed[f'{VARIANTS[2]}/cap{capacity}']['residual_reads_across_distinct_nonzero_bases'] > 0
        assert signed[f'{VARIANTS[3]}/cap{capacity}']['residual_subset_hits'] > 0
    # Binary mean ties, excluded input supports, and q larger than the data.
    tied_centers, tied_model = calibrate(Counter({0: 5, 1: 7, 3: 2, 12: 2}), 1)
    assert tied_centers == [0] and tied_model['included_nontrivial_rows'] == 4
    few_centers, few_model = calibrate(Counter({3: 2, 12: 2}), 128)
    assert few_centers == [3, 12] and few_model['actual_q'] == 2
    result['calibration_boundary_checks'] = {'tie_to_zero': tied_model,
                                            'fewer_supports_than_q': few_model}
    # Independent algebraic signed-mask extremes, including the +2048 INT8 sum.
    assert int((-WEIGHTS[:, 0]).sum()) == 2048
    for positive, negative in ((0, MASK), (MASK, 0), (0xAAAA, 0x5555)):
        counts = Counter()
        actual = accumulate_sources(positive, negative, counts, 'residual')
        assert np.array_equal(actual, vector_value(positive, negative))
    return result


def main():
    assert not OUT.exists(), 'Do not overwrite an existing research receipt.'
    plan = json.loads(PLAN.read_text())
    plan_sha = digest(PLAN)
    assert plan['calibration_samples'] == [0, 1, 2]
    assert plan['evaluation_samples'] == [4, 9]
    assert not set(plan['calibration_samples']) & set(plan['evaluation_samples'])
    assert plan['basis_counts'] == [8, 32, 128]
    assert plan['shared_wide_cache_capacities'] == [8, 16]
    assert plan['calibration_rows_per_partition_per_sample'] == 3000
    assert digest(LEDGER) == LEDGER_SHA
    print('Ledger SHA verified; running conformance checks.', flush=True)
    conformance = directed_checks(plan['shared_wide_cache_capacities'])
    models = {}
    model_records = []
    calibration_hash = hashlib.sha256()
    with LEDGER.open('rb') as stream:
        for operator, partition in itertools.product(plan['operators'], plan['k_partitions']):
            hist = Counter()
            for sample in plan['calibration_samples']:
                masks, raw = read_rows(stream, sample, operator, partition)
                hist.update(masks)
                calibration_hash.update(raw)
            for q in plan['basis_counts']:
                centers, metadata = calibrate(hist, q)
                parents = compile_bases(centers)
                key = (operator, partition, q)
                models[key] = (centers, parents)
                model_records.append({'operator': operator, 'k_partition': partition,
                                      **metadata,
                                      'compiled_base_parent': {str(k): v for k, v in parents.items()}})
            print('Calibrated operator/K', operator, partition,
                  'actual q', [len(models[(operator, partition, q)][0])
                               for q in plan['basis_counts']], flush=True)

    tiles, aggregate = [], {}
    evaluation_hash = hashlib.sha256()
    unique_eval_tiles = 0
    unique_eval_rows = 0
    with LEDGER.open('rb') as stream:
        for sample, operator, partition, chunk in itertools.product(
                plan['evaluation_samples'], plan['operators'], plan['k_partitions'],
                plan['evaluation_chunks']):
            n = min(64, 3000 - chunk * 64)
            masks, raw = read_rows(stream, sample, operator, partition, chunk * 64, n)
            evaluation_hash.update(raw)
            unique_eval_tiles += 1
            unique_eval_rows += n
            for q in plan['basis_counts']:
                centers, parents = models[(operator, partition, q)]
                descriptors, structure = assign_rows(masks, centers)
                points = {}
                for capacity, variant in itertools.product(
                        plan['shared_wide_cache_capacities'], VARIANTS):
                    key = f'{variant}/cap{capacity}'
                    counts = simulate(masks, descriptors, parents, capacity, variant)
                    points[key] = counts
                    total = aggregate.setdefault(f'q{q}/{key}', Counter())
                    add_counts(total, counts)
                    total['tiles'] += 1
                    total['raw_binary_vector_adds'] += structure['raw_binary_vector_adds']
                    total['unique_touched_pwp_prefetch_lower_bound'] += structure[
                        'unique_touched_nontrivial_bases_prefetch_lower_bound']
                tiles.append({'sample': sample, 'operator': operator,
                              'k_partition': partition, 'spatial_chunk': chunk,
                              'rows': n, 'requested_q': q, 'actual_q': len(centers),
                              'structure': structure, 'variants': points})
            if unique_eval_tiles % 16 == 0:
                print('Verified held-out support tiles', unique_eval_tiles, flush=True)

    assert unique_eval_tiles == 128 and unique_eval_rows == 7936
    comparisons = {}
    for q, capacity in itertools.product(plan['basis_counts'], plan['shared_wide_cache_capacities']):
        prefix = f'q{q}/'
        points = {variant: aggregate[f'{prefix}{variant}/cap{capacity}'] for variant in VARIANTS}
        comparisons[f'q{q}/cap{capacity}'] = {
            'note': 'Each ratio compares ONLY the named operation count; none is latency, traffic bytes, or energy.',
            'on_demand_over_pwp_binary_adds': points[VARIANTS[1]]['binary_vector_adds'] / max(1, points[VARIANTS[0]]['binary_vector_adds']),
            'exact_over_on_demand_binary_adds': points[VARIANTS[2]]['binary_vector_adds'] / max(1, points[VARIANTS[1]]['binary_vector_adds']),
            'subset_over_exact_binary_adds': points[VARIANTS[3]]['binary_vector_adds'] / max(1, points[VARIANTS[2]]['binary_vector_adds']),
            'subset_over_on_demand_binary_adds': points[VARIANTS[3]]['binary_vector_adds'] / max(1, points[VARIANTS[1]]['binary_vector_adds']),
            'subset_over_pwp_binary_adds': points[VARIANTS[3]]['binary_vector_adds'] / max(1, points[VARIANTS[0]]['binary_vector_adds']),
        }
    assert digest(PLAN) == plan_sha, 'Predeclared plan changed during execution.'
    report = {
        'date': '2026-09-07',
        'status': 'PREDECLARED_COHORT_EXECUTED_EXACT_DIAGNOSTIC_REFERENCE',
        'plan': plan, 'plan_sha256': plan_sha,
        'resolved_plan_details': RESOLVED,
        'script_sha256': digest(Path(__file__)),
        'imported_screen_joint_residual_sha256': digest(BASE / 'screen_joint_residual.py'),
        'ledger_path': str(LEDGER), 'ledger_sha256': LEDGER_SHA,
        'selected_calibration_bytes_sha256': calibration_hash.hexdigest(),
        'selected_evaluation_bytes_sha256': evaluation_hash.hexdigest(),
        'python_version': sys.version, 'numpy_version': np.__version__,
        'cohort': {'unique_calibration_rows': 144000, 'unique_evaluation_tiles': unique_eval_tiles,
                   'unique_evaluation_rows': unique_eval_rows, 'calibrated_models': len(models),
                   'diagnostic_lanes': LANES,
                   'held_out_lane_values_verified_all_variants': sum(
                       v['diagnostic_rational_lane_values_verified'] for v in aggregate.values()),
                   'diagnostic_mismatches': sum(v['diagnostic_mismatches'] for v in aggregate.values())},
        'variant_labels': dict(zip(VARIANTS, plan['variants'])),
        'state_contract': {'shared_wide_cache_capacities': plan['shared_wide_cache_capacities'],
                           'extra_reserved_vector_registers': 2,
                           'cache_initialization': 'cold per variant per evaluation tile',
                           'wide_value_unit': 'one complete diagnostic vector; not a mapped SRAM word or iso-area proof',
                           'numeric_reference': 'Full Y for every original row, not residual-only equality'},
        'diagnostic_weights': WEIGHTS.tolist(),
        'diagnostic_theta_numerators': [999883 + 17*c for c in range(16)],
        'diagnostic_common_denominator': 1000000,
        'limits': [
            'No RTL, EDA, physical SRAM admission, full-network speed, or AEE claim.',
            'Operation/request counts do not model Phi eight-input parallel reduction, PWP prefetch scheduling, memory-word widths, ports, pipeline stalls, or cycle latency.',
            'Clustering follows the predeclared independent calibration, not the official Phi artifact; all settings are fixed before evaluating held-out sample IDs.',
            'Same-sequence held-out samples do not establish cross-sequence generalization.',
            'Pattern matching, offline compilation, LRU tags and subset comparisons are necessary work; their hardware latency and area are not folded into arithmetic counters.',
            'External PWP request values are not free runtime source-weight reads; full precomputation arithmetic is separately recorded per calibrated model.',
            'Cache budget is shared across base and residual values. Both explicit vector registers, final cross-K accumulators, descriptors and pattern storage are additional state.',
            'Completed residuals are immutable cached copies. Auditing provenance does not grant future knowledge or change admission/replacement.',
            'Wide signed residual range differs from natural binary subset sums; integer-numerator equality does not establish finite hardware word fit or frozen floating-point reassociation equivalence.',
            'No complete TA-on-residual or Comperity cycle comparator is implemented here.'
        ],
        'directed_conformance_checks': conformance,
        'calibration_models': model_records,
        'aggregate': {k: dict(v) for k, v in aggregate.items()},
        'operation_count_comparisons': comparisons,
        'tiles': tiles,
        'claim_boundary': {'rtl_speedup': False, 'ppa': False, 'full_network': False,
                           'frozen_fp32_equivalence': False, 'new_AEE': False},
    }
    with OUT.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Completed', str(OUT), 'bytes', OUT.stat().st_size, flush=True)
    print(json.dumps({'cohort': report['cohort'], 'comparisons': comparisons}, indent=2), flush=True)


if __name__ == '__main__':
    main()
