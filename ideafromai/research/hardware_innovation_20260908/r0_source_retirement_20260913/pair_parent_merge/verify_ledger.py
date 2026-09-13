#!/usr/bin/env python3
"""Independent source-geometry prediction of the fixed parent-merge interface."""
from pathlib import Path
import importlib.util,json
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('source_work',HERE.parent/'rtl/verify_ledger.py')
source_work=importlib.util.module_from_spec(spec);spec.loader.exec_module(source_work)

def main():
    rows=json.loads((HERE/'results.json').read_text());cache={};facts={};checked=0
    for r in rows:
        path=Path(r['fixture_path'])
        if path not in cache:
            old,detail=source_work.expected(path)
            mode5=dict(old[5],merge_issues=0)
            mode6=mode5.copy()
            h=detail['saved_psum_update_beats']
            terminal_savings=detail['nonzero_pair_services']-detail['c4_destination_services']
            mode6['cycles']-=2*h+terminal_savings
            for field in ('update_issues','psum_reads','psum_writes'):mode6[field]-=h
            mode6['sum_issues']+=h;mode6['merge_issues']=h
            detail.update(expected_cycle_savings=2*h+terminal_savings,
                eliminated_terminal_scans=terminal_savings,extra_merge_beats=h,
                parent_builds=mode5['sum_issues'])
            cache[path]=({5:mode5,6:mode6},detail)
        counts,detail=cache[path]
        for field,expected in counts[r['mode']].items():
            actual=r[field]
            if field=='cycles':actual-=r['source_stalls']+r['weight_stalls']+r['output_stalls']
            assert actual==expected,(r['fixture'],r['mode'],field,actual,expected)
            checked+=1
        facts[r['fixture']]=detail
    assert facts['all_supports_extreme']['support_codes']==list(range(16))
    result=dict(passed=True,runs=len(rows),scalar_counter_checks=checked,
        all_support_codes_covered=True,scope='Configured native source/mask/origin; no DUT trace or simulated state oracle; complete mode5 cycle prediction and mode6 interface delta; stalls added from actual handshake counters.',fixtures=facts)
    (HERE/'ledger_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='fixtures'},indent=2))
if __name__=='__main__':main()
