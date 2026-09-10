"""Build and run only the isolated single-PE functional test, without EDA."""
from pathlib import Path
import csv
import json
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    subprocess.run([sys.executable,'-B',str(HERE/'prepare_gp_temporal_cases.py')],check=True)
    commands=[['verilator','--cc','--exe','--top-module','gp_temporal_pe','-Wno-fatal',
               '--Mdir','gp_temporal_obj','-CFLAGS','-std=c++17 -O2',
               'gp_temporal_pe.sv','gp_temporal_scoreboard.cpp'],
              ['make','-C','gp_temporal_obj','-f','Vgp_temporal_pe.mk','-j1'],
              [str(HERE/'gp_temporal_obj/Vgp_temporal_pe'),str(HERE/'gp_temporal_cases.bin'),
               str(HERE/'gp_temporal_results.tsv')]]
    output=[]
    for cmd in commands:
        run=subprocess.run(cmd,cwd=HERE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output.append('$ '+' '.join(cmd)+'\n'+run.stdout)
        (HERE/'gp_temporal_run.log').write_text('\n'.join(output))
        if run.returncode:
            (HERE/'gp_temporal_result.json').write_text(json.dumps(dict(status='FAIL',command=cmd,
               returncode=run.returncode,output=run.stdout),indent=2)+'\n')
            print(run.stdout);raise SystemExit(run.returncode)
    rows=list(csv.DictReader((HERE/'gp_temporal_results.tsv').open(),delimiter='\t'))
    counts=json.loads((HERE/'gp_temporal_cases.json').read_text())
    groups={}
    for row in rows:
        if row['real']!='1':continue
        route='onehot7' if '_onehot7_' in row['case'] else 'time_rows'
        key=(('P4F2' if row['p4f2']=='1' else 'P8F1')+'_'+route+
             ('_member' if row['reduce']=='1' else '_scalar')+('_stress' if row['stress']=='1' else '_unstalled'))
        group=groups.setdefault(key,dict(tasks=0,start_to_complete_beats=0,issue_beats=0,commits=0,
                                        scalar_members=0,packets=0,source_gap_beats=0,
                                        input_backpressure_beats=0,output_backpressure_beats=0,read_responses=0))
        group['tasks']+=1
        for column in group:
            if column!='tasks':group[column]+=int(row[column])
    for group in groups.values():
        group['mean_start_to_complete_beats']=group['start_to_complete_beats']/group['tasks']
    passed=re.search(r'PASS tasks=(\d+) cancellation_locations=(\d+) cancellation_followup_tasks=(\d+) cycles=(\d+)',output[-1])
    assert passed
    result=dict(status='PASS',kind='VERILATOR_SINGLE_PHYSICAL_PE_FUNCTIONAL_TEST',
        tool=subprocess.check_output(['verilator','--version'],text=True).strip(),
        cases=len(rows)//4,regular_tasks=int(passed[1]),cancellation_locations=int(passed[2]),
        cancellation_followup_tasks=int(passed[3]),simulation_beats=int(passed[4]),
        real_capture_count=counts['real_capture_count'],real_cases=counts['real_cases'],
        complete_real_source_extent=384,numeric_mismatches=0,overflow_events=0,
        reference=counts['reference'],secondary_reference='C++ per-packet int64 sum, exact member/commit accounting and full state reread',
        theta_source_values=sorted(set(c['theta_source'] for c in counts['cases'] if c['real'])),
        precision=counts['precision'],
        common_storage=dict(S_data_bits=56*15,S_valid_bits=56,source_package_bits=96+64+3+1,
            pending_destination_bits=56,decode_table_bits=56,scalar_work_bits=1+6+4+10,
            read_response_bits=1+6+15+1+16,task_tag_bits=16,
            note='Plus task mode/rank/FSM/overflow control; no implicit second packet, W SRAM or source bank.'),
        arithmetic='one scalar W10 partial adder; actual four-W10 two-level 3:2 compressor plus final W10 CPA; one shared S16 guarded update adder; common configurable superset, no minimum-area baseline/PPA claim',
        state_interface='One explicit asynchronous register-array read address multiplexed between commit and consumer. One S write/beat. One-entry elastic response. Wide words are not reset.',
        timing_scope='Accepted start through state_complete for this PE, including NR4 acceptance and supplied input gaps. Consumer hold/read/backpressure and release are counted separately. No upstream source decode generation, W bank/refill/arbitration, 64PE multicast or PSN included.',
        protocol_coverage=['normal/stalled source input','scalar/member with the same destinations and write count',
            'P8F1/P4F2, class/time decode','negative/zero W and nonzero operands cancelling to valid zero',
            '0/1/2/3/4-source packets and poisoned inactive p/F/rows','both S15 limits',
            '112 non-destructive state reads/task, same-cycle response replace and output backpressure',
            'state held until explicit consumer_done, premature reads blocked',
            'consecutive configurations without reset','cancel waiting/executing/held read response then run another full task'],
        boundary='Common source NR4 windows, not Gustav private W=0 compaction. Register-array functional implementation; no SRAM macro mapping, DC/PT/FM, PPA, complete-layer acceleration or original FP32 equivalence.',
        real_group_totals=groups)
    (HERE/'gp_temporal_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(output[-1].split('\n',1)[1].strip())
    for key,value in groups.items():
        if key.endswith('_unstalled'):print(key,round(value['mean_start_to_complete_beats'],2),'beats/task',value['issue_beats'],'issues')


if __name__=='__main__':main()
