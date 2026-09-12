"""Summarize actual new endpoint results; never borrow missing825/RTL rows."""
from pathlib import Path
import json,csv

HERE=Path(__file__).resolve().parent


def main():
    rows=[]
    for structure in ('dense','contiguous34','lifting40'):
        path=HERE/structure/'results.json'
        if not path.exists():continue
        d=json.loads(path.read_text());ready=[r for r in d['rows'] if r['mode']=='ready']
        per_tile={r['cycles']/r['tiles'] for r in ready}
        assert len(per_tile)==1
        row=dict(structure=structure,AEE10=d['quality']['diverse10_AEE'],
            AEE9=d['quality']['holdout9_AEE'],source_ready_cycles_per_H8=per_tile.pop(),
            compiled_instructions=d['accounting']['instructions'],
            peak_live_RF_vectors=d['accounting']['peak_live_words'],
            checked_gate_bits=sum(r['gate_bits_checked'] for r in d['rows']),
            checked_RF_vector_writes=sum(r['RF_vector_writebacks_checked'] for r in d['rows']),
            differences=sum(r['differences'] for r in d['rows']))
        full=HERE.parent/'algorithm/valid825'/structure/'quality.json'
        row['new_AEE825']=json.loads(full.read_text())['summary']['AEE_frame_mean'] if full.exists() else None
        rows.append(row)
    baseline=next(r for r in rows if r['structure']=='dense')['source_ready_cycles_per_H8']
    for r in rows:r['source_cycle_reduction_vs_matched_dense']=1-r['source_ready_cycles_per_H8']/baseline
    result=dict(complete_source=len(rows)==3,complete_825=all(r['new_AEE825'] is not None for r in rows) and len(rows)==3,
        rows=rows,scope='Same common Verilator source RTL with newly trained constants. Different student functions with matched GT budgets. Not full-chain/whole-network service or PPA.',
        resource=dict(RF_vectors=96,lanes=8,lane_bits=48,ROM_words=512,ROM_bits=128,
            pipeline_stages=2,state_read_bits=64,state_write_bits=64),
        input='Tracked ordinary parent corner/interior I24, unchanged upstream; every student has fresh integer gate gold.',
        controls='Common full da4ml CSE and fixed last_use_pressure; identical ready/stress input/output handshakes.',
        required_next='Reexecute changed consumers and complete upstream/native/BN under the same resources; no old student service or825 inherited.')
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (HERE/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
