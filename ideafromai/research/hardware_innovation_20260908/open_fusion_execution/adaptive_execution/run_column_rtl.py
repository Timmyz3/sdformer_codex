"""Bounded Verilator test of the NRV column subset matcher and residual leaf.

All P2 groups of the four saved gate windows, plus directed boundary cases.
Gold independently reconstructs full-K row bitsets in C++; no network/EDA run.
"""
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
FULL = BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
WORK = Path('/tmp/tcasii_column_rtl_20260912')


def actual_cases():
    cases = []
    captures = []
    for axis in ['ordinary', 'lifting_raw']:
        source = FULL/'capture'/axis/'000_zurich_city_09_a_0001.npz'
        captures.append(str(source))
        with np.load(source) as z:
            geometry = json.loads(str(z['window_geometry_json']))
            for window in ['corner', 'interior']:
                geo = geometry[window]
                gates = z[window+'_sn1_gate']
                words = np.zeros(gates.shape[1:], np.uint16)
                for t in range(10):
                    words |= gates[t].astype(np.uint16) << t
                height,width = geo['gate_shape']
                oy,ox = geo['gate_origin']
                sy0,sx0 = geo['source_origin']
                for y in range(height):
                    for x in range(0,width,2):
                        positions = min(2,width-x)
                        columns = []
                        for k in range(864):
                            c,offset = divmod(k,9)
                            ky,kx = divmod(offset,3)
                            mask = 0
                            for pos in range(positions):
                                sy,sx = oy+y+ky-1,ox+x+pos+kx-1
                                if 0 <= sy < 240 and 0 <= sx < 320:
                                    ly,lx = sy-sy0,sx-sx0
                                    assert 0 <= ly < words.shape[1] and 0 <= lx < words.shape[2]
                                    mask |= int(words[c,ly,lx]) << (pos*10)
                            if mask:
                                columns.append((k,mask))
                        cases.append(dict(name=f'{axis}_{window}_y{y}_x{x}',kind='capture',
                            positions=positions, columns=columns, axis=axis, window=window,
                            origin=[oy+y,ox+x], full_k_scanned=864))
    return cases,captures


def from_rows(name,rows,k_count):
    columns = [(k,sum((k in row)<<i for i,row in enumerate(rows))) for k in range(k_count)]
    return dict(name=name,kind='directed',positions=2,columns=columns)


def directed_cases():
    allbits = (1<<20)-1
    return [
        dict(name='all_equal_max864',kind='directed',positions=2,columns=[(k,allbits) for k in range(864)]),
        dict(name='empty_after_max_no_reset',kind='directed',positions=2,columns=[]),
        from_rows('nested_subsets', [set(range(i%7)) for i in range(20)],6),
        from_rows('equal_popcount_different_support',
            [{0,1},{2,3},{0,1},{1,2},set(),{0},{0,1,2},{0,1,2},
             {4,5},{6,7},{4,5},{5,6},set(),{4},{4,5,6},{4,5,6},
             {0,2},{1,3},{0,2},{1,3}],8),
        from_rows('identical_singletons_no_parent', [{0} for _ in range(20)],1),
        dict(name='last_column_changes_relation',kind='directed',positions=2,
             columns=[(k,allbits if k<863 else sum(1<<i for i in range(0,20,2))) for k in range(864)]),
        dict(name='zero_dense_max864_after_nonzero',kind='directed',positions=2,
             columns=[(k,0) for k in range(864)]),
    ]


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    actual,captures = actual_cases()
    cases = directed_cases()+actual
    test_file = WORK/'cases.txt'
    with test_file.open('w') as f:
        f.write(str(len(cases))+'\n')
        for case in cases:
            f.write(f"{case['name']} {case['kind']} {case['positions']} {len(case['columns'])}\n")
            for k,mask in case['columns']:
                f.write(f'{k} {mask}\n')
    build = WORK/'build'
    cmd = ['verilator','--cc','--exe','-Wno-fatal',
        '--top-module','column_relation_matcher','-Mdir',str(build),
        '-CFLAGS','-std=c++17 -O2',str(HERE/'column_relation_matcher.sv'),str(HERE/'column_tb.cpp')]
    compiled = subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (WORK/'build.log').write_text(compiled.stdout)
    if compiled.returncode:
        print(compiled.stdout[-12000:])
        raise SystemExit(compiled.returncode)
    made = subprocess.run(['make','-C',str(build),'-f','Vcolumn_relation_matcher.mk','-j2'],
                          text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (WORK/'make.log').write_text(made.stdout)
    if made.returncode:
        print(made.stdout[-12000:])
        raise SystemExit(made.returncode)
    output = WORK/'tb_results.json'
    tested = subprocess.run([str(build/'Vcolumn_relation_matcher'),str(test_file),str(output)],
                            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(tested.stdout,flush=True)
    if tested.returncode:
        raise SystemExit(tested.returncode)
    result = json.loads(output.read_text())
    aggregates = defaultdict(lambda: defaultdict(int))
    per_case = {c['name']:c for c in cases}
    for row in result['cases']:
        case = per_case[row['name']]
        key = (case['axis']+'_'+case['window'] if case['kind']=='capture' else 'directed')+'_'+row['profile']
        aggregates[key]['runs'] += 1
        for metric in ['leaf_cycles','accepted_columns','parent_comparisons','selected_parents',
                       'accepted_replay_columns','residual_comparisons','parent_stall_cycles',
                       'residual_stall_cycles','column_bubble_cycles','replay_bubble_cycles','busy_start_rejections']:
            aggregates[key][metric] += row[metric]
    capture_runs = [r for r in result['cases'] if r['kind']=='capture']
    result.update(scope=__doc__,simulation='Verilator leaf RTL functional/protocol check; not VCS/FM/DC/PT admission or complete layer timing.',
        verilator_version=subprocess.check_output(['verilator','--version'],text=True).strip(),
        captures=captures,actual_input_cases=len(actual),directed_input_cases=len(cases)-len(actual),
        profile_runs=len(result['cases']),actual_profile_runs=2*len(actual),
        actual_single_position_cases=sum(c['positions']==1 for c in actual),
        actual_full_k_scanned=len(actual)*864,
        input_layout='k=c*9+ky*3+kx; mask bit=position*10+t; source_y=gate_y+ky-1, source_x=gate_x+kx-1. Native image padding only. Actual input stream omits zero-mask columns.',
        gold='C++ bitset<864> per row rebuilt from physical k; full support inclusion, max-popcount, original-index tie and nnz<2 rule; independent from RTL counterexample matrix. Every residual column and reconstructed row checked.',
        borrowed_rule_source='/home/zhumd/work/literature_artifacts/Prosperity/kernels/prosparsity_cuda.cu:79',
        protocol=f'Ready and deterministic input-bubble/parent-backpressure/residual-backpressure profiles; finish standalone and on last accepted column; start rejected while residual pending; all {len(result["cases"])} runs share one initial reset.',
        capture_totals_both_profiles={key:sum(r[key] for r in capture_runs) for key in
            ['accepted_columns','parent_comparisons','residual_comparisons','leaf_cycles','mismatches']},
        finish_with_last_runs=sum(r['finish_with_last_column'] for r in result['cases']),
        finish_separate_runs=sum(not r['finish_with_last_column'] for r in result['cases']),
        build_warning_lines=[line for line in compiled.stdout.splitlines() if line.startswith('%Warning')],
        aggregates={k:dict(v) for k,v in aggregates.items()},
        cycle_definition='Counts accepted start through final residual handshake, includes detector/parent/replay handshakes and injected leaf bubbles; excludes reset, input SRAM generation/reload, arithmetic payload execution and all downstream consumers.',
        PPA=False,whole_chain_RTL=False,production_modified=False)
    (HERE/'column_rtl_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases',)},indent=2))


if __name__=='__main__':main()
