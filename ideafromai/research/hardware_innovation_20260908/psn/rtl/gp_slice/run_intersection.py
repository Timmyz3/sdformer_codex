"""Run the same complete slice with addressed dense or compressed W images."""
from pathlib import Path
import csv
import json
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def summarize(log):
    rows=list(csv.DictReader((HERE/'intersection_results.tsv').open(),delimiter='\t'))
    cases=json.loads((HERE/'intersection_cases.json').read_text())
    metadata={c['name']:c for c in cases['cases']}
    match=re.search(r'PASS tasks=(\d+) gate_bits=(\d+) cycles=(\d+) response_backpressure=(\d+) nr_peak=(\d+) W_accept_peak=(\d+)',log)
    assert match
    groups={};pairs={}
    identifiers={'case','real','time','intersection','reduce','stress'}
    invariants=('source_reads','packets','events','commits','psn_issues')
    for row in rows:
        stem=row['case'].rsplit('_packed_',1)[0]
        family=metadata[row['case']].get('weight_variant','original_integer' if row['real']=='1' else 'directed')
        if row['real']=='1':
            key='/'.join((family,'time' if row['time']=='1' else 'class',
                          'intersection' if row['intersection']=='1' else 'read_then_filter',
                          'member' if row['reduce']=='1' else 'scalar',
                          'stalled' if row['stress']=='1' else 'ready'))
            group=groups.setdefault(key,dict(tasks=0));group['tasks']+=1
            for field,value in row.items():
                if field not in identifiers:
                    group[field]=(max(group.get(field,0),int(value)) if field.endswith('_peak')
                                  else group.get(field,0)+int(value))
        # Changing W representation must leave the complete arithmetic exactly
        # unchanged; pressure trajectories need not have the same cycle count.
        key=(row['case'],row['reduce'],row['stress'])
        counts=tuple(int(row[f]) for f in invariants)
        if key in pairs:assert pairs[key]==counts
        else:pairs[key]=counts
    for g in groups.values():
        g['mean_start_to_last_gate']=g['last_gate']/g['tasks']
        g['weight_density']=(g['nnz_tile0']+g['nnz_tile1'])/(g['tasks']*768)
    result=dict(status='PASS',kind='VERILATOR_BOUNDED_TWO_POINTER_VS_DENSE_W_ADDRESS_SERVICE',
        tool=subprocess.check_output(['verilator','--version'],text=True).strip(),
        tasks=int(match[1]),gate_bits_compared=int(match[2]),simulation_beats=int(match[3]),
        numeric_mismatches=0,overflow_events=0,
        original_real_captures=cases['real_captures'],
        additional_real_sparse_captures=cases['additional_actual_sparse_captures'],
        real_cases=cases['real_cases'],directed_cases=cases['total_cases']-cases['real_cases'],
        extent='2 tiles x4 PEs, P4/F1, complete C384, 16 positions x2 actual output rows, all T10 per task',
        borrowed_mechanism='GustavSNN HPCA2026 VII-B sorted NRV/W-index intersection; static per-task mode selection. Not a new mechanism or official artifact.',
        memory_format=dict(common_W_bytes_per_tile=2048,ports_per_tile=2,read_bits=8,
            dense='384 consecutive signed W8 bytes',
            intersection='uint16 nnz count followed by sorted {uint16 C index,int8 W}; 2+3*nnz bytes, max1154B',
            memory_response='Only actual RTL byte address selects memory data. Header/index/value share the same two ports. One transaction and one W8 response holder per port.',
            source='Four1KiB banks; actual64bit reads plus RTL128bit extractor; same72reads/bank (576B) in both modes. No free precomputed NRV or mask.',
            parser='Each of8 PEs retains count9,pointer9,index9,phase3,inflight1; four ports add kind3 and11bit byte address. Compared with the old top:269 added declared control bits including mode; no SRAM/PPA claim.'),
        functional_checks=['Independent NumPy and C++ dense C384 S/gate golden',
            'Exact per-PE compressed-memory address/kind/source trace including skipped W indices',
            'Dense values cover each live source row; intersection values cover exactly live source AND W!=0',
            'Private NR4 packet boundaries, events and S commits unchanged across W layouts',
            'Source physical addresses, W request and return ownership, valid-zero S, barriers, all gates and theta payload',
            'No global reset between tasks; request/return/output/done backpressure; opposite tile completion order',
            'Real dense and two actually trained sparse W variants plus zero W, empty source, C383 tails, signed cancellation and nonunit theta'],
        coverage=dict(response_backpressure_PE_port_beats=int(match[4]),private_NR4_packet_peak=int(match[5]),
            simultaneous_W_port_requests=int(match[6]),
            both_tiles_finish_first=all(sum(int(r[f'first_tile{i}']) for r in rows)>0 for i in (0,1))),
        timing='Accepted start to last accepted paired gate; includes local source/index/value reads, private compaction, S and all T10 PSN. Config/program/tau and cold imports are outside. Only ready uses matched fixed memory service; stalled modes have different RNG/tag histories and are functional stress only.',
        index_counter='w_reads counts all byte transactions; w_metadata_reads counts header and index bytes; w_value_reads counts W8 values. compare_active_PE_beats includes repeated head comparisons while waiting for arbitration, not unique comparisons.',
        boundaries=['Original paper does not specify this3B record or two byte-wide port timing; these are explicit implementation choices.',
            'V-A shares a resident W row across tile PEs; IV-E still has per-PE local reads and VI-B reports shared-buffer conflicts. VII-B does not specify index-head caching, same-address request coalescing or broadcast. Private byte-index re-reading here is not a demonstrated original-paper requirement.',
            'Ordinary stronger controls remain: paid same-address read coalescing/broadcast, shared index heads or a bounded index cache/bitmap using the same capacity. Adding them would complete the baseline, not create a new mechanism.',
            'Common source row bridge waits for both tiles to consume/skip before advancing; conservative extra ownership constraint, not an exact copy of all Gustav staging.',
            'W indices are static imported data with full capacity cost; runtime NRV rows are produced from actual source reads. Cold W/index DMA and compression construction are not timed.',
            'No source read reduction: full dense-code C scan remains even after a W list ends.',
            'PE last row and end handshakes are separated by the top; standalone same-cycle use remains unsupported.',
            'Existing resource limits remain: single-transaction memory ports, F1, no8x8/full layer/producer/FC2/BN2/shortcut or EDA.'],
        claim='Functional and port-transaction comparison within this slice; not a qualified RTL acceleration or PPA result.',
        real_group_totals=groups)
    (HERE/'intersection_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:dict(last_gate=v['mean_start_to_last_gate'],W_bytes=v['w_reads']/v['tasks'],
                           metadata=v['w_metadata_reads']/v['tasks'],density=v['weight_density'])
                      for k,v in groups.items() if k.endswith('/scalar/ready')},indent=2))


def main():
    commands=[
        [sys.executable,'-B','prepare_intersection_cases.py'],
        ['verilator','--cc','--exe','--top-module','gp_slice','-Wno-fatal',
         '--Mdir','obj_intersection','-CFLAGS','-std=c++17 -O2','gp_slice.sv','gp_slice_pe.sv','intersection_scoreboard.cpp'],
        ['make','-C','obj_intersection','-f','Vgp_slice.mk','-j1'],
        [str(HERE/'obj_intersection/Vgp_slice'),str(HERE/'intersection_cases.bin'),str(HERE/'intersection_results.tsv')]]
    log=[]
    for cmd in commands:
        p=subprocess.run(cmd,cwd=HERE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log.append('$ '+' '.join(cmd)+'\n'+p.stdout)
        (HERE/'intersection_run.log').write_text('\n'.join(log))
        if p.returncode:
            (HERE/'intersection_result.json').write_text(json.dumps(dict(status='FAIL',command=cmd,
                returncode=p.returncode,message=p.stdout),indent=2)+'\n')
            print(p.stdout,flush=True);raise SystemExit(p.returncode)
    print(log[-1].split('\n',1)[1],end='')
    summarize('\n'.join(log))


if __name__=='__main__':main()
