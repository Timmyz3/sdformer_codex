from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
BREADTH=HERE.parents[1]


def main():
    report=dict(scope='Four ready real-I24 same-Machine complete local chains with fixed PoT2 source functions.',
        evidence='CPU payload/port service, no whole-chain RTL/PPA or inheritedAEE.',new_X=False,rows=[])
    for label in ('corner','interior'):
        dense=json.loads((HERE/f'dense_{label}.json').read_text())
        for structure in ('dense','lifting40'):
            r=json.loads((HERE/f'{structure}_{label}.json').read_text())
            parent=json.loads((HERE.parent/'matched_local_chain'/f'{structure}_{label}.json').read_text())
            checks=[r['source_program']['checks'],*r['producer']['checks'].values(),*r['consumer']['checks'].values()]
            assert all(x['differences']==0 for x in checks)
            report['rows'].append(dict(structure=structure,window=label,service_slots=r['service_slots'],
                own_parent_service=parent['service_slots'],reduction_vs_own_parent=1-r['service_slots']/parent['service_slots'],
                reduction_vs_same_permission_dense=1-r['service_slots']/dense['service_slots'],
                source_slots=r['source_program']['service_slots'],source_parent_slots=parent['source_program']['service_slots'],
                preview_end=r['integration']['producer_end'],consumer_slots=r['consumer']['service_slots'],
                physical_port_bytes=r['physical_port_bytes'],gate_activity=r['gate_activity'],
                parent_gate_activity=parent['gate_activity'],independent_checked_values=sum(x['values'] for x in checks),
                independent_differences=0,new_GPU_endpoint_capture=False,new_AEE10=None,new_AEE825=None,
                function=str(BREADTH/'source_constant_probe'/structure/'deployed_constants.npz')))
    report['checked_values_sum']=sum(x['independent_checked_values'] for x in report['rows'])
    report['strong_dense_source_controls']=dict(
        common_CSE='Actual complete chain here uses dense PoT2 common CSE:319 source-only ready cycles/tile,40workRF+gate.',
        fixed_low_RF='Same PoT2 dense function has admitted two-chain CSD source:472cycles/tile,13workRF+gate,229ROM. Not integrated or scheduled concurrently here.',
        inference='Do not call40RF a dense lower bound. This serial complete-chain cost uses the faster common CSE; future state-sharing comparisons must retain the legal low-RF control.')
    report['limits']=['Source-only ~35% cycle reduction contracts to measured7.6–9.2% local service reduction.',
        'Only fixed two local windows, completeK864; no native/globalBN/join or whole network cycles.',
        'PoT2 is common prior art. Relative lifting/dense differences include separately trained source/consumer activity, not all attributed to structure.',
        'New GPU endpoint captures and matching AEE remain separate pending tasks.']
    alignment=HERE/'gpu_alignment.json'
    if alignment.exists():
        a=json.loads(alignment.read_text())
        report['GPU_alignment']='gpu_alignment.json'
        report['GPU_integer_endpoints_and_parameters_match']=a['all_fixed_integer_endpoints_and_parameters_match']
        for row in report['rows']:
            axis=a['axes'][row['structure']]
            row['new_GPU_endpoint_capture']=True
            row['new_AEE10']=axis['fresh_diverse10_quality']['summary']['AEE_frame_mean']
            row['GPU_integer_endpoints_match']=axis['windows'][row['window']]['fixed_integer_endpoints_match']
        report['limits'][-1]='Fresh diverse10 and actual finite-window GPU integer endpoints now matched; floating preview differences retained, no new825 or whole-network equivalence.'
    (HERE/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
