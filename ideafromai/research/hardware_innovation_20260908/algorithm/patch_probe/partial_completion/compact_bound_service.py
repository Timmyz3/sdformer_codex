"""Finite-service tradeoff for compact conservative bound operands.

Same machine and time-major trace as finite_service. Small shared tables reuse
the existing 48-byte bound scratch after eta generation. Extra selection/decode
is charged to the common vector issue path. Nominal storage is not mapped area.
"""
import json
from pathlib import Path

import numpy as np

from compact_bound_probe import upper_power2
from finite_service import Engine, Slice, GROUPS, H_GROUPS, aggregate, build_nrvs
from source_count_bound_probe import N_UP

HERE = Path(__file__).resolve().parent
MODES = ('channel_dyadic_cached', 'H8_exact_resident', 'H8_dyadic_resident', 'maximum_count_resident')


class BoundEngine(Engine):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.resident_loaded = False

    def state(self, kind, dep, beats):
        if kind != 'bound_LU_H8_read64':
            return super().state(kind, dep, beats)
        if self.mode == 'channel_dyadic_cached':
            dep = super().state('bound_exponent_H8_read64', dep, 2)
        elif not self.resident_loaded:
            # Exact shared table: 12 bins * 2 endpoints * 16 bits = 48 B.
            # Log table or per-H8 max constants: <=16 B including padding.
            dep = super().state('bound_resident_table_fill64', dep,
                                6 if self.mode == 'H8_exact_resident' else 2)
            self.resident_loaded = True
        return self.conv('bound_register_select_or_decode', dep)


class CompactSlice(Slice):
    def __init__(self, params, hg, mode):
        super().__init__(params, hg)
        self.e = BoundEngine(mode)
        if mode == 'channel_dyadic_cached':
            self.lo = -upper_power2(-self.lo)
            self.hi = upper_power2(self.hi)
        elif mode.startswith('H8'):
            self.lo = np.repeat(self.lo.min(1)[:, None], 8, axis=1)
            self.hi = np.repeat(self.hi.max(1)[:, None], 8, axis=1)
            if 'dyadic' in mode:
                self.lo = -upper_power2(-self.lo)
                self.hi = upper_power2(self.hi)
        elif mode == 'maximum_count_resident':
            self.lo = -N_UP[:, None]*upper_power2(np.maximum(-self.w.min(1), 0))
            self.hi = N_UP[:, None]*upper_power2(np.maximum(self.w.max(1), 0))


def main():
    params = dict(np.load(HERE/'integer_deployment/common3_diagonal_34.npz'))
    baseline = json.loads((HERE/'finite_service.json').read_text())
    result = dict(
        scope=baseline['scope'], machine=baseline['machine'],
        policies=['same candidate_time_major prefix, eta, ordered source/W issue, actual endpoint/compare/count fees and live feedback',
                  'all bounds outward enclose the exact INT16 per-channel table, validated by compact_bound_probe.py',
                  '48-byte bound scratch may be repurposed only after all core/tau/eta generation is done; eta still has its original separate 16-byte register',
                  'resident shared tables are reloaded for every independent H8 context; no free inter-context lifetime',
                  'an additional select/decode instruction occupies the common Conv issue path whenever a new bin is selected',
                  'decoded bounds feed the existing sequential low/high additions; there is no second bound-result bank',
                  'table muxes and exponent decoding need physical logic; same nominal capacity/issue count does not establish equal mapped area'],
        table_layouts=dict(channel_dyadic_cached='per-H8 per-bin: 80 payload bits padded to 128, one bin cached; 2304 B layer table',
                           H8_exact_resident='one H8 group: twelve INT16 endpoint pairs, 48 B; 576 B layer table',
                           H8_dyadic_resident='one H8 group: twelve two-5bit exponent codes, padded to16 B; 192 B layer table',
                           maximum_count_resident='one H8 group: eight positive/negative maximum exponents, padded to16 B; 192 B layer constants'),
        exclusions=['not RTL timing, area, power, full layer or full network',
                    '48-byte table reinterpretation and single-step decode are explicit unimplemented hardware assumptions'],
        controls={m:baseline['axes'][m]['totals'] for m in ('full10Y_time_major', 'candidate_time_major')},
        axes={m:dict(contexts=[]) for m in MODES}, source_preparation=[])
    for path in sorted((HERE/'integer_valid10').glob('capture_*.npz')):
        with np.load(path) as cap:
            golden = cap['gate_common3_diagonal_34_exact'].reshape(64, 12, 10, 8, 4).transpose(0, 1, 2, 4, 3)
            for g in GROUPS:
                words = cap['source_gate_words'][g].astype(np.int64)
                nrvs, counts, prep = build_nrvs(words, True)
                result['source_preparation'].append(dict(capture=path.name, group=g, **prep))
                for hg in H_GROUPS:
                    for mode in MODES:
                        sl = CompactSlice(params, hg, mode)
                        service = sl.run(words, nrvs, counts, 'candidate_time_major')
                        mismatch = int(np.count_nonzero(sl.answer != golden[g, hg]))
                        if mismatch:
                            raise RuntimeError(f'{mode} changed {mismatch} gates')
                        result['axes'][mode]['contexts'].append(dict(capture=path.name, group=g, h_group=hg,
                            service=service, gate_mismatches=mismatch, checked_gates=320))
                print(path.name, g, 'complete', flush=True)
    prep_ticks = sum(x['model_ticks'] for x in result['source_preparation'])
    ordinary = result['controls']['full10Y_time_major']['with_source_preparation_ticks']
    original = result['controls']['candidate_time_major']['with_source_preparation_ticks']
    for mode in MODES:
        total = aggregate(result['axes'][mode]['contexts'])
        total['source_preparation_ticks'] = prep_ticks
        total['with_source_preparation_ticks'] = total['model_ticks'] + prep_ticks
        total['change_vs_ordinary'] = total['with_source_preparation_ticks']/ordinary-1
        total['change_vs_per_channel_exact'] = total['with_source_preparation_ticks']/original-1
        result['axes'][mode]['totals'] = total
    (HERE/'compact_bound_service.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print({m:v['totals']['with_source_preparation_ticks'] for m,v in result['axes'].items()}, flush=True)


if __name__ == '__main__':
    main()
