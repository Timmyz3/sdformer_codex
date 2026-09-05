#!/usr/bin/env python3
"""Measured gated-pair execution energy, with negative low-density point."""
import json
from pathlib import Path
import re

HW = Path(__file__).resolve().parents[2]


def main():
    rows = []
    physical = {}
    directories = dict(ordinary='m2255_hold_buffers_wq4h9t91', tsbg='m2255_hold_buffers_c_ygiy_s')
    for axis, directory in directories.items():
        path = HW / 'results' / directory
        physical[axis] = dict(path=str(path), **json.loads((path / 'result.json').read_text()))
    for window, oc, tc in [('low',4717,4717), ('median',6733,4465), ('high',22294,7554)]:
        point = dict(window=window)
        for axis, cycles in [('ordinary',oc), ('tsbg',tc)]:
            out = HW / 'results/m2256_gated_real_weight_power' / f'{axis}_{window}_direct'
            assert (out / 'COMPLETE.txt').is_file()
            report = (out / 'power.rpt').read_text()
            power = float(re.search(r'Total Power\s*=\s*([\deE+.-]+)', report).group(1))
            point[axis] = dict(cycles=cycles, power_mW=power, duration_ns=cycles*3,
                energy_nJ=power*cycles*3/1000, report=str(out/'power.rpt'))
        point['logic_energy_reduction'] = 1-point['tsbg']['energy_nJ']/point['ordinary']['energy_nJ']
        rows.append(point)
    result = dict(rows=rows, physical=physical,
        clock_gating='Common automatic clock gating settings; independent mapped clock groups',
        scope='3ns prelayout TT0.9V25C logic energy; direct zero-delay gate SAIF with checkpoint-derived candidate INT8 FC weights',
        excluded=['SRAM bank energy', 'preload', 'CTS and extracted routing', 'FC quantization AEE claim', 'population/frame energy'],
        observation='Low reuse loses energy despite equal cycles; medium/high win via shorter completion despite higher average power')
    output = HW / 'results/m2256_gated_real_weight_power/summary.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'physical'}, indent=2))


if __name__ == '__main__':
    main()
