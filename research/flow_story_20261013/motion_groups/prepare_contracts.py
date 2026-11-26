"""Build explicit remote contracts from an archived, verified loader invocation.

Runs locally, never connects to a server or starts a model. Deployment verifies
every declared remote file against these locally computed hashes.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = Path('/data/flow_a/code/research_20261013/motion_groups')
RUN = Path('/data/flow_a/runs/motion_groups_20261013')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    if path.exists():
        raise ValueError('Contract already exists: '+str(path))
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    launch = json.loads((HERE/'borrowed_launch.json').read_text())
    argv = launch['argv'][2:]
    argv.remove('--run-inference')
    args = dict(zip(argv[::2], argv[1::2]))
    assert len(args)*2 == len(argv)
    source_pins = {}
    for name in ('run_pair_probe.py', 'intervention_core.py', 'group_policy.py'):
        source_pins[str(REMOTE/name)] = sha(HERE/name)
    for name in ('evaluate_breadth.py', 'evaluate_breadth_extended.py', 'run_bn_scope.py',
                 'run_same_shape_reference.py', 'probe_actual_skip.py'):
        source_pins[str(REMOTE.parent/'breadth'/name)] = sha(HERE.parent/'breadth'/name)
    cases = HERE/'cases_preflight_dense.json'
    save(cases, {'schema': 'flow_breadth_cases_v1', 'cases': [
        {'id': 'native_provenance_placeholder', 'method': 'dense', 'bn': 'D'}]})
    args['--cases'] = str(REMOTE/cases.name)
    args['--cases-sha256'] = sha(cases)
    for mode, device, count, stages, pairs in (
            ('cpu_smoke', 'cpu', 1, [0], 1), ('gpu_diagnostic', 'cuda:0', 8, [0, 2], 2)):
        a = dict(args)
        a['--device'] = device; a['--wall-minutes'] = '20'; a['--output'] = str(RUN/(mode+'.json'))
        contract = {'schema': 'flow_motion_pair_probe_contract_v1', 'mode': mode,
                    'breadth_root': str(REMOTE.parent/'breadth'), 'source_pins': source_pins,
                    'base_args': [x for pair in a.items() for x in pair],
                    'target_count': count, 'stages': stages, 'pairs_per_layer': pairs,
                    'min_motion_pixels': 1.0, 'gpu_lock': '/data/flow_a/runs/.flow_a_gpu0.lock',
                    'queue_status': str(RUN/(mode+'_status.json')),
                    'intervention_origin': 'our diagnostic output mask; not a completed fast/slow method',
                    'population_rule': 'round-robin sequences in existing ordered gate-dev32',
                    'runtime_predictor_implemented': False, 'flow_source': 'offline same-target dense prediction',
                    'automatic_baseline_pause_or_resume': False,
                    'maximum_queue_wait_seconds': 14*3600, 'maximum_run_minutes': 20}
        save(HERE/(mode+'_contract.json'), contract)
    manifest = {p.name: sha(p) for p in (HERE/'run_pair_probe.py', HERE/'intervention_core.py',
        HERE/'group_policy.py', cases, HERE/'cpu_smoke_contract.json', HERE/'gpu_diagnostic_contract.json')}
    save(HERE/'deployment_manifest.json', manifest)
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
