"""Directed source-only hit checks; NOT natural cache performance samples."""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cache_candidate', HERE/'execute.py')
x = importlib.util.module_from_spec(spec); spec.loader.exec_module(x)


def run():
    q = x.common.read_npz(HERE.parent/'capture_owned/deployed_constants.npz')
    original = x.common.read_npz(HERE.parent/'capture_owned/000_zurich_city_09_a_0001.npz')
    identity = original['interior_I24'][:, :, :1, :2].copy()
    checks = []
    for mode in ('exact', 'certified'):
        for stress in (False, True):
            m = x.CacheMachine(stress); m.cache_mode = mode; m.source_q = q
            rows = []
            for label, delta in (('cold_real_crop', 0), ('exact_repeat', 0), ('synthetic_plus_one', 1)):
                current = identity+delta
                gate = x.matched.source_gold(current, q, 'dense')
                dot = (q['As_q16'].astype(np.int64)@current.reshape(10, -1)).reshape(current.shape)
                data = {'interior_I24':current, 'interior_sn1_gate':gate, 'interior_source_S48':dot}
                report = x.source(m, data, 'dense', 'interior')
                rows.append(dict(input=label, hits=report['hits'], nonzero_change_hits=report['nonzero_change_hits'],
                                 gate_check=report['checks'], actual_dot_values_checked=report['actual_dot_values_checked']))
                if label == 'exact_repeat':
                    assert report['hits'] == 24 and report['nonzero_change_hits'] == 0
                if label == 'synthetic_plus_one':
                    if mode == 'exact':
                        assert report['hits'] == 0
                    else:
                        assert report['nonzero_change_hits'] > 0
            checks.append(dict(mode=mode, stress=stress, cases=rows))
    result = dict(scope='12 directed source-only cases, separate from natural four-frame timing',
                  synthetic=True, full_consumer=False, natural_performance=False, checks=checks)
    (HERE/'directed_hits.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
