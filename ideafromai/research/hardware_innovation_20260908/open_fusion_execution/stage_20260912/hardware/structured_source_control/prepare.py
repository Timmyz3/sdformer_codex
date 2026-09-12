from pathlib import Path
import json
import numpy as np
from source34_adapter import support, reference

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[1]/'algorithm/hardware_exports/ordinary'


def main():
    parent = dict(np.load(PARENT/'deployed_constants.npz'))
    candidate = {k: v.copy() for k, v in parent.items()}
    candidate['As_q16'] *= support()
    assert np.count_nonzero(candidate['As_q16']) == 34
    np.savez_compressed(HERE/'deployed_constants.npz', **candidate)
    capture = dict(np.load(PARENT/'000_zurich_city_09_a_0001.npz', allow_pickle=True))
    fields, rows = {}, []
    for label in ('corner', 'interior'):
        i24 = capture[label+'_I24']
        parent_q, parent_gate = reference(i24, parent)
        assert np.array_equal(parent_gate, capture[label+'_sn1_gate'])
        q, gate = reference(i24, candidate)
        for suffix, value in [('I24', i24), ('Q24', q), ('gate', gate), ('parent_Q24', parent_q), ('parent_gate', parent_gate)]:
            fields[label+'_'+suffix] = value
        rows.append(dict(window=label, shape=list(i24.shape), gate_elements=int(gate.size),
            parent_reference_vs_actual_GPU_gate_differences=0,
            parent_gate_ones=int(parent_gate.sum()), source34_gate_ones=int(gate.sum()),
            gate_differences=int(np.count_nonzero(gate != parent_gate)),
            new_source_Q24_min=int(q.min()), new_source_Q24_max=int(q.max())))
    np.savez_compressed(HERE/'fixture.npz', **fields)
    info = dict(parent='ordinary original_ordered24 + onepass; current final-combination deployment',
        parent_directory=str(PARENT), pattern='contiguous3/3/4', groups=[[0,1,2],[3,4,5],[6,7,8,9]],
        changed_fields=[k for k in parent if not np.array_equal(parent[k], candidate[k])],
        support_slots=34, retained_nonzero=34, As_q16=candidate['As_q16'].tolist(),
        As_exponent=int(candidate['As_exponent']), source_threshold=candidate['source_threshold'].tolist(),
        source_direction=candidate['source_direction'].tolist(),source_constant=candidate['source_constant'].tolist(),
        source_theta=float(candidate['source_theta']), rows=rows,
        function='As_q16 @ I24 -> signed48 -> RNE15 -> saturate24 -> original inclusive source comparison -> theta*g',
        validation_scope='CPU source-only new function on actual parent I24; no old downstream gold or AEE inherited.',
        AEE='pending new ordinary diverse10; no training or automatic825')
    (HERE/'parameters.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__': main()
