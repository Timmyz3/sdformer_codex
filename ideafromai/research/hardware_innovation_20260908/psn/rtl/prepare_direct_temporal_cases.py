"""Create independent NumPy-dot references and static CSD programs for the leaf."""
from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np


def csd(value):
    sign = -1 if value < 0 else 1
    magnitude, shift = abs(int(value)), 0
    terms = []
    while magnitude:
        if magnitude & 1:
            digit = 2-(magnitude & 3)
            terms.append((shift, sign*digit))
            magnitude -= digit
        magnitude //= 2
        shift += 1
    assert sum(digit*(1 << shift) for shift, digit in terms) == value
    assert all(0 <= shift <= 7 for shift, _ in terms)
    return terms


def program_for(weight, tau, variable):
    program, starts, lengths = [], [], []
    bounds = []
    for row in range(3):
        starts.append(len(program))
        coefficients = np.zeros(10, dtype=np.int64)
        low = high = -int(tau[row]) if variable[row] else 0
        row_min, row_max = low, high
        if variable[row]:
            for index, value in enumerate(weight[row]):
                for shift, digit in csd(int(value)):
                    program.append(index | (shift << 4) | (int(digit < 0) << 7))
                    coefficients[index] += digit*(1 << shift)
                    low = int(np.where(coefficients >= 0, coefficients*(-2048), coefficients*2047).sum())-int(tau[row])
                    high = int(np.where(coefficients >= 0, coefficients*2047, coefficients*(-2048)).sum())-int(tau[row])
                    row_min, row_max = min(row_min, low), max(row_max, high)
            assert np.array_equal(coefficients, weight[row])
        lengths.append(len(program)-starts[-1])
        assert -(1 << 23) <= row_min and row_max < (1 << 23)
        bounds.append([row_min, row_max])
    assert len(program) <= 128 and all(length < 64 for length in lengths)
    return np.array(program, dtype=np.uint8), starts, lengths, bounds


def reference(weight, tau, variable, fixed, mapping, inputs):
    dot = inputs.astype(np.int64) @ weight.astype(np.int64).T
    gates = np.where(variable, dot >= tau, fixed)
    address = (gates.astype(np.uint8) << np.arange(3, dtype=np.uint8)).sum(1).astype(np.uint8)
    return address, mapping[address]


def make_group(name, record, inputs, real=False):
    weight = np.asarray(record['weight_int8'], dtype=np.int8)
    tau = np.asarray(record['threshold_int32'], dtype=np.int32)
    variable = np.asarray(record['variable_rows'], dtype=np.uint8)
    fixed = np.asarray(record['constant_gates'], dtype=np.uint8)
    mapping = np.asarray(record['mapping'], dtype=np.uint8)
    inputs = np.asarray(inputs, dtype=np.int16)
    assert inputs.shape[1] == 10 and inputs.min() >= -2048 and inputs.max() <= 2047
    address, expected = reference(weight, tau, variable.astype(bool), fixed.astype(bool), mapping, inputs)
    program, starts, lengths, bounds = program_for(weight, tau, variable)
    return dict(name=name, real=real, weight=weight, tau=tau, variable=variable, fixed=fixed,
                mapping=mapping, inputs=inputs, address=address, expected=expected,
                program=program, starts=starts, lengths=lengths, bounds=bounds)


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    data = root/'algorithm/direct_code_integer'
    parameters = json.loads((data/'parameters.json').read_text())
    captures = sorted((data/'deployment/capture10').glob('*.npz'))
    assert len(captures) == 60, f'Expected fixed diverse10 x six sources; found {len(captures)} captures'
    groups, numeric_checks = [], 0
    for path in captures:
        name = path.stem.rsplit('_', 1)[-1]
        record = parameters[name]
        with np.load(path) as captured:
            group = make_group(path.stem, record, captured['input_int12'], real=True)
            assert np.array_equal(group['expected'], captured['expected_class']), path.name+' GPU class mismatch'
            assert np.array_equal(group['address'], captured['expected_address']), path.name+' GPU address mismatch'
            for key, captured_key in [('weight', 'weight_int8'), ('tau', 'threshold_int32'),
                                      ('mapping', 'address_to_class')]:
                assert np.array_equal(group[key], captured[captured_key]), path.name+' parameter mismatch'
        groups.append(group)
        numeric_checks += len(group['inputs'])
    inputs = np.array([[0]*10, [-2048]*10, [2047]*10, list(range(-5, 5)),
                       [3, -2048]+[0]*8, [-2048, 2047]*5], dtype=np.int16)
    directed = [
        ('directed_extreme_and_equality', [[-128, 127]+[0]*8, [1]+[0]*9, [0, -1]+[0]*8],
         [0, 3, 2048], [1, 1, 1], [0, 0, 0], [2, 7, 0, 6, 5, 1, 3, 4]),
        ('directed_constant_rows', [[-128]*10, [127]*10, [1, -1]+[0]*8],
         [0, 0, 0], [0, 0, 1], [1, 0, 0], [7, 6, 5, 4, 3, 2, 1, 0]),
        ('directed_zero_program', [[0]*10]*3,
         [0, 1, -1], [1, 1, 1], [0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 0]),
        ('directed_all_constant', [[-128]*10, [127]*10, [85]*10],
         [0, 0, 0], [0, 0, 0], [1, 0, 1], [3, 0, 7, 4, 1, 6, 2, 5]),
    ]
    for name, weight, tau, variable, fixed, mapping in directed:
        record = dict(weight_int8=weight, threshold_int32=tau, variable_rows=variable,
                      constant_gates=fixed, mapping=mapping)
        groups.append(make_group(name, record, inputs))
    target = here/'direct_temporal_cases.bin'
    with target.open('wb') as stream:
        stream.write(b'DTC1')
        stream.write(struct.pack('<I', len(groups)))
        for group in groups:
            name = group['name'].encode()
            stream.write(struct.pack('<H', len(name)))
            stream.write(name)
            stream.write(group['weight'].tobytes())
            stream.write(group['tau'].astype('<i4').tobytes())
            for key in ('variable', 'fixed', 'mapping'):
                stream.write(group[key].tobytes())
            stream.write(bytes(group['starts']))
            stream.write(bytes(group['lengths']))
            stream.write(struct.pack('<H', len(group['program'])))
            stream.write(group['program'].tobytes())
            stream.write(struct.pack('<I', len(group['inputs'])))
            for values, address, expected in zip(group['inputs'], group['address'], group['expected']):
                stream.write(struct.pack('<10hBB', *map(int, values), int(address), int(expected)))
    summary = {'reference': 'NumPy int64 dense dot, independent of CSD generation; checked against GPU captures',
               'real_vectors': numeric_checks, 'directed_vectors': sum(len(g['inputs']) for g in groups if not g['real']),
               'groups': len(groups), 'gpu_class_and_address_mismatches': 0,
               'csd_prefix_bounds': 'exact interval of cumulative input coefficients at each CSD prefix over signed12 domain, including minus tau; all fit Acc24',
               'uop_encoding': {'input_index': 'bits3:0', 'shift': 'bits6:4', 'negative': 'bit7'},
               'groups_detail': [{'name': g['name'], 'vectors': len(g['inputs']), 'real': g['real'],
                                  'uops': len(g['program']), 'row_uops': g['lengths'], 'prefix_bounds': g['bounds'],
                                  'expected_unstalled_handshake_latency': len(g['program'])+5,
                                  'expected_serial_vector_cycles': len(g['program'])+6,
                                  'configuration_write_cycles': len(g['program'])+11} for g in groups]}
    (here/'direct_temporal_cases.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'groups_detail'}, indent=2))
    print('real module uops:', {name: next(g['lengths'] for g in groups if g['name'].endswith(name)) for name in parameters})


if __name__ == '__main__':
    main()
