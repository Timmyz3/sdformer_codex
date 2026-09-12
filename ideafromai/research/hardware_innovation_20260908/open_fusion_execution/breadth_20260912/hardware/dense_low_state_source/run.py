"""Fixed two-accumulator CSD control; no ROM expansion or strategy sweep."""
from pathlib import Path
from collections import Counter
import json, os, sys
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
BREADTH = HERE.parents[1]
OPEN = BREADTH.parent
BASE = OPEN.parent
LIFT = BASE / 'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40'
sys.path.insert(0, str(LIFT / 'schedule_compare_same_port/two_stage_writeback'))
import run_compare as sched
sys.path.insert(0, str(BREADTH / 'source_execution'))
import run as source


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def csd(value):
    sign = 1 if value >= 0 else -1
    n, shift, terms = abs(int(value)), 0, []
    while n:
        if n & 1:
            digit = 2 - (n & 3)
            terms.append((shift, sign * digit))
            n -= digit
        n //= 2
        shift += 1
    assert sum(sign * (1 << shift) for shift, sign in terms) == value
    return terms


def compile_fixed(p):
    program, ready, tag = [], {}, {}
    next_node = 10
    rows = []

    def emit(kind, operands=None, dst=None, **kwargs):
        nonlocal next_node
        operands = [] if operands is None else operands
        needed = max([ready[x['reg']] for x in operands] + ([ready.get(dst, 0)] if dst is not None else [0]))
        while len(program) < needed:
            program.append(dict(kind='nop', reason='RAW_wait', issue_slot=len(program)))
        ins = dict(kind=kind, operands=operands, logical_node=next_node, issue_slot=len(program), **kwargs)
        next_node += 1
        if dst is not None:
            ins['dst'] = dst
            ready[dst] = len(program) + sched.LATENCY
            tag[dst] = ins['logical_node']
        if kind == 'gate':
            ready[95] = len(program) + sched.LATENCY
        program.append(ins)
        return ins

    def operand(reg, shift=0, sign=1):
        return dict(reg=reg, node=tag[reg], shift=shift, sign=sign)

    for t in range(10):
        program.append(dict(kind='load', dst=t, source_t=t, logical_node=t, issue_slot=t))
        ready[t], tag[t] = t + sched.LATENCY, t
    for t, row in enumerate(p['As_q16']):
        terms = [operand(c, shift, sign) for c, q in enumerate(row) for shift, sign in csd(int(q))]
        chains = [terms[0::2], terms[1::2]]
        assert all(len(x) >= 2 for x in chains)
        start = len(program)
        for chain, terms_in_chain in enumerate(chains):
            emit('addsub', terms_in_chain[:2], dst=10 + chain)
        for k in range(2, max(map(len, chains))):
            for chain, terms_in_chain in enumerate(chains):
                if k < len(terms_in_chain):
                    emit('addsub', [operand(10 + chain), terms_in_chain[k]], dst=10 + chain)
        emit('addsub', [operand(10), operand(11)], dst=12)
        threshold, direction, constant = (int(p['source_' + key][t]) for key in ('threshold', 'direction', 'constant'))
        threshold, constant = sched.base.folded_cutoff(threshold, direction, constant, int(p['As_exponent']))
        emit('gate', [operand(12)], output_t=t, threshold=threshold, direction=direction, constant=constant)
        rows.append(dict(row=t, CSD_terms=len(terms), addsub=len(terms) - 1,
            program_words=len(program) - start,
            RAW_nops=sum(ins['kind'] == 'nop' for ins in program[start:])))
    while len(program) < max(ready.values()):
        program.append(dict(kind='nop', reason='pipeline_drain', issue_slot=len(program)))
    program.append(dict(kind='commit', issue_slot=len(program)))
    return program, rows


def actual_window(program, x, p):
    # Independent payload replay on all actual vectors. The scalar validator below
    # separately verifies RF logical tags and two-stage read timing.
    v = x.reshape(-1, 10, 8).transpose(1, 0, 2).reshape(10, -1).astype(np.int64)
    work = np.zeros((13, v.shape[1]), np.int64)
    dots, gates = np.zeros_like(v), np.zeros(v.shape, np.uint8)
    for ins in program:
        kind = ins['kind']
        if kind == 'load':
            work[ins['dst']] = v[ins['source_t']]
        elif kind == 'addsub':
            a, b = ins['operands']
            value = ((work[a['reg']] << a['shift']) * a['sign']
                     + (work[b['reg']] << b['shift']) * b['sign'])
            assert value.min() >= -(1 << 47) and value.max() < (1 << 47)
            work[ins['dst']] = value
        elif kind == 'gate':
            t = ins['output_t']; a = ins['operands'][0]
            dots[t] = (work[a['reg']] << a['shift']) * a['sign']
            if ins['constant'] >= 0:
                gates[t] = ins['constant']
            else:
                gates[t] = dots[t] >= ins['threshold'] if ins['direction'] > 0 else dots[t] <= ins['threshold']
    expected_dots = p['As_q16'].astype(np.int64) @ v
    expected_gate = source.literal_gate(v, p, 'dense')
    assert np.array_equal(dots, expected_dots)
    assert np.array_equal(gates, expected_gate)
    return dict(vectors=int(v.shape[1]), original_I24_values=int(v.size),
        completed_integer_dot_values=int(dots.size), completed_dot_differences=0,
        output_gate_values=int(gates.size), output_gate_differences=0, gate_ones=int(gates.sum()))


def main():
    endpoint = BREADTH / 'algorithm/matched_training/dense/stage320'
    p = dict(np.load(endpoint / 'deployed_constants.npz'))
    program, rows = compile_fixed(p)
    checks = sched.validate_program('ordinary', program, {k: v.tolist() for k, v in p.items()})
    dump(HERE / 'program_not_admitted.json', program)
    source.fields(program, HERE / 'program_not_admitted.txt')
    manifest_root = OPEN / 'stage_20260912/hardware/source_rtl_inputs'
    cases = [x for x in json.loads((manifest_root / 'manifest.json').read_text())['cases'] if x['axis'] == 'ordinary']
    actual = []
    for case in cases:
        inputs = np.fromfile(manifest_root / case['input_file'], dtype='<i4')
        actual.append(dict(window=case['window'], **actual_window(program, inputs, p)))
    kinds = dict(Counter(ins['kind'] for ins in program))
    capacity = sched.CONTRACT['instruction_ROM']['words']
    report = dict(status='ADMITTED' if len(program) <= capacity else 'NOT_ADMITTED_ROM_CAPACITY',
        identity='New stage320 dense; exact same As, RNE/sat, threshold and original I24 as the current CSE control.',
        evidence='CPU fixed-program arithmetic and RF-tag/two-slot schedule validation; not an executable 512-ROM RTL result.',
        instructions=len(program), ROM_words=capacity, excess_words=max(0, len(program) - capacity),
        instruction_counts=kinds, base_without_nops=len(program) - kinds.get('nop', 0),
        RAW_nops=sum(ins.get('reason') == 'RAW_wait' for ins in program),
        drain_nops=sum(ins.get('reason') == 'pipeline_drain' for ins in program),
        work_RF_vectors=13, reserved_gate_RF=95, gate_RF_vectors=1,
        work_RF_layout=dict(inputs=[0, 9], alternating_chains=[10, 11], row_sum=12),
        per_row=rows, scalar_checks=checks, actual_window_checks=actual,
        RTL_executed=False, downstream_replayed=False,
        policy='Only this deterministic per-row two-chain CSD point. No truncation, ROM expansion, ISA change or alternative sweep.')
    dump(HERE / 'results.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
