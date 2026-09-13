"""Read-only conversion of five existing programs and two shared I24 halos."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
OPEN = HERE.parents[1]
INPUTS = OPEN/'stage_20260912/hardware/source_rtl_inputs'
FAMILIES = (
    ('dense', 'source_execution/dense'),
    ('contiguous34', 'source_execution/contiguous34'),
    ('lifting40', 'source_execution/lifting40'),
    ('dense_two_term', 'source_constant_probe/dense'),
    ('lifting40_two_term', 'source_constant_probe/lifting40'),
)
FIELDS = ((0, 3), (3, 7), (10, 7), (17, 7), (24, 6), (30, 6),
          (36, 1), (37, 1), (38, 6), (44, 4), (48, 48), (96, 1), (97, 2))


def encode_program(path):
    result = []
    for line in path.read_text().splitlines():
        values = list(map(int, line.split()))
        assert len(values) == len(FIELDS)
        assert values[0] in range(6)
        assert 0 <= values[1] < 96
        assert all(0 <= values[i] < 96 for i in (2, 3))
        if values[0] == 4:
            assert values[1] == 95
        bits = sum((value & ((1 << width)-1)) << offset
                   for value, (offset, width) in zip(values, FIELDS))
        # Decode independently to catch signed threshold or field-order loss.
        back = [(bits >> offset) & ((1 << width)-1) for offset, width in FIELDS]
        if back[10] & (1 << 47):
            back[10] -= 1 << 48
        assert back == values
        result.append(bits)
    assert 0 < len(result) <= 512
    return result


def main():
    fixture = HERE/'fixtures'
    fixture.mkdir(exist_ok=True)
    input_cases = {x['window']: x for x in json.loads((INPUTS/'manifest.json').read_text())['cases']
                   if x['axis'] == 'ordinary'}
    manifest = dict(scope=__doc__, families=[], cases=[],
        original_RTL=str(OPEN/'stage_20260912/hardware/rtl_source/temporal_source.sv'),
        original_cpp_TB=str(OPEN/'stage_20260912/hardware/rtl_source/tb.cpp'),
        generated_parameters=False, recomputed_gold=False, modified_original_files=False)
    for label, case in input_cases.items():
        array = np.fromfile(INPUTS/case['input_file'], '<i4')
        assert array.size == case['tiles']*80
        assert np.all(array >= -(1 << 23)) and np.all(array < (1 << 23))
        path = fixture/(label+'_input32.hex')
        path.write_text(''.join(f'{int(value) & 0xffffffff:08x}\n' for value in array))
        back = np.asarray([int(line, 16) for line in path.read_text().splitlines()], '<u4').view('<i4')
        assert np.array_equal(back, array)
    for name, relative in FAMILIES:
        parent = OPEN/'breadth_20260912'/relative
        program = encode_program(parent/'program.txt')
        program_path = fixture/(name+'_program128.hex')
        program_path.write_text(''.join(f'{value:032x}\n' for value in program))
        prior = json.loads((parent/'results.json').read_text())
        manifest['families'].append(dict(name=name, parent=str(parent),
            program_source=str(parent/'program.txt'), program_length=len(program)))
        for label in ('corner', 'interior'):
            ic = input_cases[label]
            gold = np.fromfile(parent/(label+'_gates.bin'), '<u2')
            assert gold.size == ic['tiles']*8 and np.all(gold <= 1023)
            gold_path = fixture/(name+'_'+label+'_gold16.hex')
            gold_path.write_text(''.join(f'{int(value):04x}\n' for value in gold))
            back = np.asarray([int(line, 16) for line in gold_path.read_text().splitlines()], '<u2')
            assert np.array_equal(back, gold)
            for mode in ('ready', 'stress'):
                expected = next(row for row in prior['rows'] if row['window'] == label and row['mode'] == mode)
                manifest['cases'].append(dict(name=name+'_'+label+'_'+mode, family=name,
                    window=label, mode=mode, tiles=ic['tiles'], program_length=len(program),
                    program=str(program_path), inputs=str(fixture/(label+'_input32.hex')),
                    gold=str(gold_path), original_input=str(INPUTS/ic['input_file']),
                    original_gold=str(parent/(label+'_gates.bin')), prior_results=str(parent/'results.json'),
                    expected={key: expected[key] for key in ('mode', 'tiles', 'cycles', 'SR64_reads',
                        'SW64_writes', 'read_stall_cycles', 'write_stall_cycles',
                        'RF_vector_writebacks_checked', 'input_values', 'gate_bits_checked', 'differences')}))
    assert len(manifest['families']) == 5 and len(manifest['cases']) == 20
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('PREPARED', len(manifest['families']), 'families;', len(manifest['cases']), 'fixed cases')


if __name__ == '__main__':
    main()
