"""Predeclared C1 materialization experiment. Counts are NOT cycles or PPA."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import Counter, OrderedDict
import hashlib
import itertools
import json
import numpy as np

BASE = Path(__file__).resolve().parent
PRIOR = BASE.parent / 'prosperity_fusion_20260906'
sys.path.insert(0, str(PRIOR))
import screen_basis_residual as old
from screen_joint_residual import LEDGER, LEDGER_SHA, digest
HW = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
sys.path.insert(0, str(HW / 'system_simulator/scripts'))
from analyze_m504_h67_single_port_parent_scratch import cleanroom_subset


class Cache:
    def __init__(self, mode, counts):
        self.mode, self.counts = mode, counts
        self.limit = 128 if 'c128' in mode else 16
        self.split = mode.startswith('split_')
        self.entries = OrderedDict()
        self.events, self.write_records = [], []

    def read(self, key):
        # Charge a tag comparison against every currently valid slot of this type.
        self.counts['wide_exact_tag_comparisons'] += sum(k[0] == key[0] for k in self.entries)
        if key not in self.entries:
            return None
        value, origin, wid = self.entries[key]
        self.entries.move_to_end(key)
        self.write_records[wid]['reads'] += 1
        self.events.append(('read', wid))
        self.counts['wide_reads'] += 1
        self.counts[f'wide_{key[0]}_reads'] += 1
        return value.copy(), origin

    def write(self, key, value, origin=0):
        assert key not in self.entries
        eligible = [k for k in self.entries if not self.split or k[0] == key[0]]
        limit = 8 if self.split else self.limit
        if len(eligible) == limit:
            victim = eligible[0]
            self.entries.pop(victim)
            self.counts['evictions'] += 1
            self.counts[f'{victim[0]}_evictions'] += 1
        wid = len(self.write_records)
        self.write_records.append({'key': key, 'reads': 0})
        self.events.append(('write', wid))
        self.entries[key] = (value.copy(), origin, wid)
        self.counts['wide_writes'] += 1
        self.counts[f'wide_{key[0]}_writes'] += 1
        self.counts['peak_wide_vectors'] = max(self.counts['peak_wide_vectors'], len(self.entries))
        assert len(self.entries) <= self.limit

    def parent(self, positive, negative):
        found = []
        for key in self.entries:
            if key[0] != 'residual':
                continue
            self.counts['wide_subset_tag_comparisons'] += 1
            if key[1] & positive == key[1] and key[2] & negative == key[2]:
                found.append(key)
        return min(found, key=lambda k: (-(k[1].bit_count()+k[2].bit_count()), k[1], k[2])) if found else None

    def finalize(self):
        for record in self.write_records:
            if record['reads'] == 0:
                self.counts['unused_writes_after_trace'] += 1
                self.counts[f'unused_{record["key"][0]}_writes_after_trace'] += 1
        self.counts['wide_read_write_events'] = self.counts['wide_reads'] + self.counts['wide_writes']
        self.counts['final_wide_vectors'] = len(self.entries)
        assert self.counts['wide_writes'] == self.counts['evictions'] + len(self.entries)
        if self.limit == 128:
            assert self.counts['evictions'] == 0
            self.counts['diagnostic_future_pruned_writes'] = self.counts['wide_writes'] - self.counts['unused_writes_after_trace']
            self.counts['diagnostic_future_pruned_read_write_events'] = self.counts['wide_reads'] + self.counts['diagnostic_future_pruned_writes']


class Shadow:
    def __init__(self, counts):
        self.counts = counts
        self.entries = OrderedDict()

    def exact_prior(self, key):
        self.counts['shadow_exact_comparisons'] += len(self.entries)
        return key in self.entries

    def compatible_prior(self, key):
        found = []
        for other in self.entries:
            self.counts['shadow_subset_comparisons'] += 1
            if other != key and other[0] & key[0] == other[0] and other[1] & key[1] == other[1]:
                found.append(other)
        return min(found, key=lambda k: (-(k[0].bit_count()+k[1].bit_count()), k)) if found else None

    def touch(self, key, compatible=False):
        assert key[0].bit_count() + key[1].bit_count() >= 2
        if compatible:
            assert key in self.entries
        if key in self.entries:
            self.entries[key] = min(3, self.entries[key] + 1)
            self.entries.move_to_end(key)
            self.counts['shadow_updates'] += 1
        else:
            if len(self.entries) == 32:
                self.entries.popitem(last=False)
                self.counts['shadow_evictions'] += 1
            self.entries[key] = 1
            self.counts['shadow_inserts'] += 1
        self.counts['peak_shadow_entries'] = max(self.counts['peak_shadow_entries'], len(self.entries))


def simulate(masks, descriptors, parents, mode):
    counts = Counter()
    cache, shadow = Cache(mode, counts), Shadow(counts)
    residual_enabled = 'independent' not in mode
    second = 'second' in mode
    compatible = 'compatible_second' in mode
    pwp = mode.startswith('pwp_')
    touched = 0

    def accum(positive, negative, label, initial=None):
        nonlocal touched
        touched |= positive | negative
        return old.accumulate_sources(positive, negative, counts, label, initial)

    for row, (mask, (base, positive, negative)) in enumerate(zip(masks, descriptors)):
        if mask.bit_count() <= 1:
            result = accum(mask, 0, 'bypass') if mask else np.zeros(old.LANES, dtype=np.int64)
        else:
            reg1 = np.zeros(old.LANES, dtype=np.int64)
            if base.bit_count() == 1:
                reg1 = accum(base, 0, 'base')
            elif base:
                bkey = ('base', base, 0)
                found = cache.read(bkey)
                if found is not None:
                    reg1 = found[0]
                    counts['base_exact_hits'] += 1
                elif pwp:
                    reg1 = old.vector_value(base)
                    counts['external_pwp_requests'] += 1
                    cache.write(bkey, reg1)
                else:
                    parent = parents[base]
                    found = cache.read(('base', parent, 0)) if parent else None
                    if found is None:
                        reg1 = accum(base, 0, 'base')
                        counts['base_direct_builds'] += 1
                    else:
                        reg1 = accum(base ^ parent, 0, 'base', found[0])
                        counts['base_graph_hits'] += 1
                    cache.write(bkey, reg1)

            size = positive.bit_count() + negative.bit_count()
            reg2 = np.zeros(old.LANES, dtype=np.int64)
            narrow_key = (positive, negative)
            prior_exact = shadow.exact_prior(narrow_key) if second and size >= 2 else False
            prefix = None
            if size:
                rkey = ('residual', positive, negative)
                found = cache.read(rkey) if residual_enabled and size >= 2 else None
                if found is not None:
                    reg2, origin = found
                    counts['residual_exact_hits'] += 1
                    counts['residual_cross_base_reads'] += bool(base and origin and base != origin)
                else:
                    parent_key = cache.parent(positive, negative) if residual_enabled and size >= 2 else None
                    if parent_key is not None:
                        initial, origin = cache.read(parent_key)
                        reg2 = accum(positive ^ parent_key[1], negative ^ parent_key[2], 'residual', initial)
                        counts['residual_subset_hits'] += 1
                        counts['residual_cross_base_reads'] += bool(base and origin and base != origin)
                    else:
                        if compatible and size >= 2 and not prior_exact:
                            prefix = shadow.compatible_prior(narrow_key)
                        if prefix is None:
                            reg2 = accum(positive, negative, 'residual')
                        else:
                            # Serialized write samples the old register before it is reused.
                            reg2 = accum(prefix[0], prefix[1], 'residual')
                            assert np.array_equal(reg2, old.vector_value(*prefix))
                            cache.write(('residual', *prefix), reg2, base)
                            counts['compatible_prefix_materializations'] += 1
                            reg2 = accum(positive ^ prefix[0], negative ^ prefix[1], 'residual', reg2)
                    assert np.array_equal(reg2, old.vector_value(positive, negative))
                    if residual_enabled and size >= 2 and (not second or prior_exact):
                        assert prefix is None
                        cache.write(rkey, reg2, base)
                        counts['complete_residual_materializations'] += 1
            # Query the OLD shadow snapshot first. These updates cannot influence the decision above.
            if second and size >= 2:
                if prefix is not None:
                    shadow.touch(prefix, compatible=True)
                shadow.touch(narrow_key)
            if base and size:
                reg1 += reg2
                counts['binary_vector_adds'] += 1
                counts['base_residual_join_adds'] += 1
                result = reg1
            else:
                result = reg1 if base else reg2
        assert np.array_equal(result, old.vector_value(mask)), (mode, row, mask)
        counts['destination_commits'] += 1
        counts['diagnostic_lane_values_verified'] += old.LANES

    cache.finalize()
    counts['per_tile_unique_source_coefficient_identities'] = touched.bit_count()
    counts['source_coefficient_reads'] = counts.pop('source_weight_reads', 0)
    assert counts['source_coefficient_reads'] == sum(counts[f'{p}_source_reads'] for p in ('base', 'residual', 'bypass'))
    assert counts['binary_vector_adds'] == counts['base_residual_join_adds'] + sum(counts[f'{p}_source_adds'] for p in ('base', 'residual', 'bypass'))
    assert counts['wide_writes'] == counts['wide_base_writes'] + counts['wide_residual_writes']
    assert counts['wide_reads'] == counts['wide_base_reads'] + counts['wide_residual_reads']
    counts['diagnostic_mismatches'] = 0
    return dict(counts)


def prosperity(masks):
    delta, parents = cleanroom_subset(np.array(masks, dtype=np.uint16))
    remaining = Counter(int(p) for p in parents if p >= 0)
    order = sorted(range(len(masks)), key=lambda i: (masks[i].bit_count(), i))
    counts, resident, touched = Counter(), {}, 0
    for row in order:
        parent, residue = int(parents[row]), int(delta[row])
        touched |= residue
        counts['source_coefficient_reads'] += residue.bit_count()
        if parent >= 0:
            assert parent in resident
            result = resident[parent].copy()
            counts['wide_reads'] += 1
            counts['binary_vector_adds'] += residue.bit_count()
            remaining[parent] -= 1
            if not remaining[parent]:
                del resident[parent]
        else:
            result = np.zeros(old.LANES, dtype=np.int64)
            counts['binary_vector_adds'] += max(0, residue.bit_count() - 1)
        result += old.vector_value(residue)
        assert np.array_equal(result, old.vector_value(masks[row]))
        if remaining[row]:
            resident[row] = result.copy()
            counts['wide_writes'] += 1
        counts['peak_live_parent_vectors'] = max(counts['peak_live_parent_vectors'], len(resident))
        counts['destination_commits'] += 1
        counts['diagnostic_lane_values_verified'] += old.LANES
    assert not resident and not any(remaining.values())
    counts['per_tile_unique_source_coefficient_identities'] = touched.bit_count()
    counts['wide_read_write_events'] = counts['wide_reads'] + counts['wide_writes']
    counts['diagnostic_mismatches'] = 0
    return dict(counts)


def add(target, source):
    for k, v in source.items():
        if k.startswith('peak_'):
            target[k] = max(target[k], v)
        else:
            target[k] += v


def directed(variants):
    # Same-sign proper prefix, then repeats, opposite signs, cross-base values, and cache churn.
    cases = {
        'prefix_then_exact': ([3, 7, 7, 15, 3], [0]),
        'signed_cross_base': ([0, 1, 62, 390, 5638, 390], [0, 1, 57, 385, 1537]),
        'replacement_and_dense': ([3 | (1 << i) for i in range(2, 16)] * 3 + [0, 65535, 65534], [0, 3, 65535]),
    }
    out = {}
    for name, (rows, centers) in cases.items():
        descriptors, _ = old.assign_rows(rows, centers)
        out[name] = {v: simulate(rows, descriptors, old.compile_bases(centers), v) for v in variants}
    exact = out['prefix_then_exact']['split_exact_second_b8r8']
    compat = out['prefix_then_exact']['split_compatible_second_b8r8']
    assert compat['compatible_prefix_materializations'] > 0
    assert compat['residual_subset_hits'] > exact.get('residual_subset_hits', 0)
    # First touch never writes a residual in either second-touch policy.
    for v in ('split_exact_second_b8r8', 'split_compatible_second_b8r8'):
        c = simulate([7], [(0, 7, 0)], {}, v)
        assert c.get('wide_residual_writes', 0) == 0
    # Opposite signs must never satisfy the same-sign containment test.
    c, key = Counter(), ('residual', 3, 4)
    cache = Cache('split_all_b8r8', c)
    cache.write(key, old.vector_value(3, 4))
    assert cache.parent(4, 3) is None
    assert cache.parent(11, 4) == key
    return out


def main():
    out = BASE / 'materialization_r1.json'
    assert not out.exists(), 'Never overwrite the first receipt.'
    plan_path = BASE / 'materialization_plan.json'
    plan = json.loads(plan_path.read_text())
    plan_sha = digest(plan_path)
    archive = (BASE / plan['calibration_source']).resolve()
    assert digest(archive) == plan['calibration_source_sha256']
    assert digest(LEDGER) == LEDGER_SHA
    models = {}
    for model in json.loads(archive.read_text())['calibration_models']:
        models[(model['operator'], model['k_partition'], model['requested_q'])] = (
            model['centers'], {int(k): v for k, v in model['compiled_base_parent'].items()})
    variants = plan['variants']
    conformance = directed(variants)
    print('Plan, frozen calibration and ledger SHA verified; directed checks passed.', flush=True)
    totals, tiles = {}, []
    selected = hashlib.sha256()
    baseline = Counter()
    ntiles = nrows = 0
    with LEDGER.open('rb') as stream:
        for sample, op, part, chunk in itertools.product(plan['evaluation_samples'], plan['operators'], plan['k_partitions'], plan['evaluation_chunks']):
            rows, raw = old.read_rows(stream, sample, op, part, chunk*64, min(64, 3000-chunk*64))
            selected.update(raw)
            ntiles += 1
            nrows += len(rows)
            strong = prosperity(rows)
            add(baseline, strong)
            for q in plan['basis_counts']:
                centers, parents = models[(op, part, q)]
                descriptors, structure = old.assign_rows(rows, centers)
                points = {}
                for variant in variants:
                    c = simulate(rows, descriptors, parents, variant)
                    points[variant] = c
                    total = totals.setdefault(f'q{q}/{variant}', Counter())
                    add(total, c)
                    total['tiles'] += 1
                tiles.append({'sample': sample, 'operator': op, 'k_partition': part,
                              'chunk': chunk, 'q': q, 'rows': len(rows),
                              'structure': structure, 'strong_baseline': strong, 'variants': points})
            if ntiles % 32 == 0:
                print('New evaluation tiles verified:', ntiles, flush=True)
    assert (ntiles, nrows) == (192, 11904)
    assert digest(plan_path) == plan_sha
    result = {
        'date': '2026-09-07', 'status': 'PREDECLARED_REFERENCE_EXECUTED',
        'plan': plan, 'plan_sha256': plan_sha, 'script_sha256': digest(Path(__file__)),
        'imported_script_sha256': digest(PRIOR / 'screen_basis_residual.py'),
        'strong_baseline_script_sha256': digest(HW / 'system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py'),
        'ledger_sha256': LEDGER_SHA, 'selected_evaluation_bytes_sha256': selected.hexdigest(),
        'cohort': {'unique_tiles': ntiles, 'unique_rows': nrows, 'sample_ids': plan['evaluation_samples'],
                   'calibration_retrained': False, 'new_diagnostic_lane_values_verified':
                   sum(v['diagnostic_lane_values_verified'] for v in totals.values()) + baseline['diagnostic_lane_values_verified'],
                   'diagnostic_mismatches': 0},
        'python_version': sys.version, 'numpy_version': np.__version__,
        'shadow_minimum_payload_bits_excluding_LRU_and_popcount': 32*(32+2+1),
        'strong_baseline': dict(baseline), 'aggregate': {k: dict(v) for k, v in totals.items()},
        'directed_checks': conformance, 'tiles': tiles,
        'limits': ['Native stream versus full-tile planning has different scheduling state; no iso-area or cycle claim.',
                   'Tag comparisons are logical valid-entry comparisons, not energy or latency. Physical CAM comparisons can be larger.',
                   'Unique coefficient IDs are counted independently per tile under its fixed weight/threshold identity; neither coefficient reads nor unique IDs are external bytes.',
                   'Future-unused-write elision is only an offline diagnostic for the same C128 trace.',
                   'PWP offline construction and Phi parallel eight-input units remain separate costs, not a scalar runtime proxy.',
                   'Non-1 rational diagnostic theta is not a measured checkpoint amplitude distribution or frozen FP32 equivalence.'],
        'claim_boundary': {'rtl_speedup': False, 'ppa': False, 'full_network': False, 'frozen_fp32_equivalence': False, 'AEE': False}}
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    fields = ['binary_vector_adds', 'source_coefficient_reads', 'per_tile_unique_source_coefficient_identities',
              'external_pwp_requests', 'wide_reads', 'wide_writes', 'unused_writes_after_trace', 'base_evictions']
    print(json.dumps({'cohort': result['cohort'], 'strong_baseline': dict(baseline),
                      'points': {k: {f: v.get(f, 0) for f in fields} for k, v in totals.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
