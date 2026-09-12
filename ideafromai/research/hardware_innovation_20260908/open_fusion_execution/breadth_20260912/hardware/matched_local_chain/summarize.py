from pathlib import Path
import json
HERE=Path(__file__).resolve().parent

def main():
    result=dict(scope='New stage320 literal parameters, two original real-I24 windows, complete local same-Machine source/preview/integer double consumer.',
        evidence='CPU payload service, not whole-chain RTL/PPA/full-frame; no new GPU snippets matched yet.',rows=[])
    for label in ('corner','interior'):
        baseline=json.loads((HERE/f'dense_{label}.json').read_text())
        for structure in ('dense','contiguous34','lifting40'):
            r=json.loads((HERE/f'{structure}_{label}.json').read_text())
            quality=json.loads((HERE.parents[1]/'algorithm/matched_training'/structure/'stage320/quality.json').read_text())
            checks=[r['source_program']['checks'],*r['producer']['checks'].values(),*r['consumer']['checks'].values()]
            assert all(x['differences']==0 for x in checks)
            result['rows'].append(dict(structure=structure,window=label,service_slots=r['service_slots'],
                reduction_vs_matched_dense=1-r['service_slots']/baseline['service_slots'],
                source_slots=r['source_program']['service_slots'],preview_end=r['integration']['producer_end'],
                consumer_slots=r['consumer']['service_slots'],physical_port_bytes=r['physical_port_bytes'],
                independent_checked_values=sum(x['values'] for x in checks),independent_differences=0,
                gate_activity=r['gate_activity'],new_literal_diverse10_AEE=quality['diverse10_AEE'],
                quality_scope='Already-measured matching literal GPU function; CPU scalar preview has not yet been compared with a new GPU endpoint. Does not inherit825.',
                trained_constant_package=f'../../algorithm/matched_training/{structure}/stage320/deployed_constants.npz'))
    result['checked_values_sum']=sum(r['independent_checked_values'] for r in result['rows'])
    result['complete_local_cases']=6
    alignment=HERE/'gpu_alignment.json'
    if alignment.exists():
        a=json.loads(alignment.read_text())
        result['GPU_alignment']='gpu_alignment.json'
        result['GPU_alignment_status']=a['status']
        result['evidence']='CPU local service with actual GPU finite-window integer endpoint comparison; FP preview differences retained. Not full-frame/whole-chain RTL/PPA.'
        for row in result['rows']:
            axis=a['axes'][row['structure']]
            row['GPU_integer_endpoints_match']=axis['windows'][row['window']]['fixed_integer_endpoints_match']
            row['quality_scope']='Actual trained GPU student with finite-window integer endpoint/parameter comparison; no whole-network CPU equivalence proof.'
            row['fresh_valid825']=axis.get('quality',{}).get('valid825')
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['checked_values_sum'], 'new parameter/gold values checked',flush=True)

if __name__=='__main__':main()
