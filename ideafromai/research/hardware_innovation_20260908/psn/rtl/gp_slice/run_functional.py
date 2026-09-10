"""Build the bounded slice and run actual-address behavioural-memory tests."""
from pathlib import Path
import csv
import json
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    commands=[
        [sys.executable,'-B','prepare_cases.py'],
        ['verilator','--cc','--exe','--top-module','gp_slice','-Wno-fatal',
         '--Mdir','obj','-CFLAGS','-std=c++17 -O2','gp_slice.sv','gp_slice_pe.sv','scoreboard.cpp'],
        ['make','-C','obj','-f','Vgp_slice.mk','-j1'],
        [str(HERE/'obj/Vgp_slice'),str(HERE/'cases.bin'),str(HERE/'results.tsv')]]
    log=[]
    for cmd in commands:
        p=subprocess.run(cmd,cwd=HERE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log.append('$ '+' '.join(cmd)+'\n'+p.stdout)
        (HERE/'run.log').write_text('\n'.join(log))
        if p.returncode:
            (HERE/'result.json').write_text(json.dumps(dict(status='FAIL',command=cmd,
                returncode=p.returncode,message=p.stdout),indent=2)+'\n')
            print(p.stdout,flush=True);raise SystemExit(p.returncode)
    records=list(csv.DictReader((HERE/'results.tsv').open(),delimiter='\t'))
    case_summary=json.loads((HERE/'cases.json').read_text())
    match=re.search(r'PASS tasks=(\d+) gate_bits=(\d+) cycles=(\d+) response_backpressure=(\d+) nr_peak=(\d+) W_accept_peak=(\d+)',log[-1])
    assert match
    groups={}
    for row in records:
        if row['real']!='1':continue
        key='/'.join(('time' if row['time']=='1' else 'class',
                      'member' if row['reduce']=='1' else 'scalar',
                      'stalled' if row['stress']=='1' else 'ready'))
        group=groups.setdefault(key,dict(tasks=0))
        group['tasks']+=1
        for field,value in row.items():
            if field not in ('case','real','time','reduce','stress'):
                group[field]=(max(group.get(field,0),int(value)) if field.endswith('_peak')
                              else group.get(field,0)+int(value))
    for g in groups.values():g['mean_start_to_last_gate']=g['last_gate']/g['tasks']
    # Fair routes use identical actual source and W memory demand per capture.
    paired={}
    for row in records:
        stem=row['case'].rsplit('_packed_',1)[0]
        key=stem,row['reduce'],row['stress']
        counts=(int(row['source_reads']),int(row['w_reads']))
        if key in paired:assert paired[key]==counts
        else:paired[key]=counts
    result=dict(status='PASS',kind='VERILATOR_2_TILE_X_4_PE_CONNECTED_FUNCTIONAL_SLICE',
        tool=subprocess.check_output(['verilator','--version'],text=True).strip(),
        tasks=int(match[1]),gate_bits_compared=int(match[2]),simulation_beats=int(match[3]),
        numeric_mismatches=0,overflow_events=0,real_captures=case_summary['real_captures'],
        real_cases=case_summary['real_cases'],directed_cases=case_summary['total_cases']-case_summary['real_cases'],
        complete_C=384,positions_per_case=16,hidden_rows_per_case=2,T=10,
        source_reference=case_summary['reference'],
        address_validation='every source bank word0..71 once; each live decoded C row exactly one addressed W request/tile/ID; no C++ NRV/bundle supply',
        completed_interfaces=['RTL 64-bit addressed source fetch and 128-bit cross-word code extraction',
            'two actual W8 read ports/tile with independent request/response ownership and stalls',
            'private zero-W compaction; two bounded NR4 packets/PE plus explicit per-port returned-word holding',
            'P4/R destination selection, scalar or physical CSA member reduction, one S commit port',
            'complete C384 S->same-ID two-tile completion barrier->scalar CSD/tau->all T10 gates',
            'paired gate output backpressure with theta payload and task tag; explicit last accepted output before release',
            'consecutive full tasks and changing rank/program/weights without global reset'],
        arithmetic='W8, scalar partial W10, actual four-input W10 CSA, S15; ONE Acc24 guarded carry chain/PE shared by S commit and PSN. No MAC. Member reducer is extra arithmetic.',
        common_resources=dict(tiles=2,PEs_per_tile=4,P=4,F_live=1,C=384,
            physical_S_words_per_PE=56,S_bits=15,S_valid_bits_per_PE=56,active_S_words_per_PE='4*actual rank (24 time /28 class for s2b3)',
            private_NR4_packets_per_PE=2,packet_payload_bits_per_PE=2*4*(12+8),
            source_cache_words_per_PE=7,source_cache_word_bits=15,U_bits_per_PE=24,
            gate_bits_per_PE=10,source_banks=4,source_bank_bytes=1024,source_used_bytes_per_bank=576,
            source_read_width=64,source_extract_bits_per_ID=128,source_bridge_bits_per_ID=12,
            W_banks=2,W_bank_bytes=512,W_used_bytes_per_bank=384,W_read_ports_per_bank=2,W_read_width=8,
            W_transactions_per_port=1,W_return_hold_bits_per_port=8,
            W_transaction_metadata_per_port='2-bit source ID,9-bit actual W address,12-bit code row; request/response phase',
            program_copies=4,program_depth=256,program_word_bits=16,
            threshold_bits_per_tile=240,threshold_read_muxes_per_tile=4,theta_payload_bits_per_tile=32,
            note='Register arrays in RTL and explicit C++ single-entry behavioural-memory responses. These are raw declarations, not foundry SRAM/PPA. Only first28 of declared56 S slots are addressable in this P4/F1 slice; synthesis may trim unused slots.'),
        precision=dict(real='actual integer direct-code student W and trained B/tau; same-student class/time exactly equal; actual source/output theta in these captures is1',
            diagnostic='source theta=[0.5,1.5,2], quantum0.5, Wq=2*theta_c*baseW; output theta=[1.25,0.75], tau separately compiled; all legal S15 and Acc24 CSD-prefix bounds',
            theta_interface='output carries both full original32-bit theta payloads plus ten gate bits, no assumption theta=1'),
        coverage=dict(W_response_backpressure_beats=int(match[4]),private_packet_peak=int(match[5]),
            simultaneous_W_requests_accepted=int(match[6]),
            both_tiles_finish_first=all(sum(int(r[f'first_tile{i}']) for r in records)>0 for i in (0,1)),
            cases=['real twelve full-C captures','zero source after live task','all zero W','private1/3-row tails',
                   'signed -128/127 and valid zero cancellation','S15 positive/negative limits',
                   'nonunit theta rational compile','private buffers full and response held',
                   'source/W request gaps, unequal memory return latencies, output/done valid held']),
        timing='Accepted start to last accepted20-bit paired gate packet includes addressed local source/W service and all T10 consumers; program/tau config is separately explicit and outside these counts. Done handshake includes five held beats.',
        excluded=['source producer/quantizer and external code/W cold refill','II1 continuous W-address acceptance: current single-transaction ports have turnaround bubbles',
                  'F_cache multi-row strips, compressed-W two-pointer intersection and pruning',
                  '8x8 array and whole H1536/P1200 layer','FC2, current-domain BN2 and shortcut','VCS/DC/PT/Formality and mapped SRAM/PPA'],
        claim='Connected bounded functional hardware slice; not official GustavSNN artifact, full GustavSNN, complete FFN, or a qualified RTL speedup.',
        real_group_totals=groups)
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(log[-1].split('\n',1)[1],end='')
    print(json.dumps({k:v['mean_start_to_last_gate'] for k,v in groups.items()},indent=2))


if __name__=='__main__':main()
