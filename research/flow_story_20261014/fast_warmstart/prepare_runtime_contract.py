#!/usr/bin/env python3
"""Create the fixed CPU7-arm contract after the predeclared fit endpoint exists."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-result', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    a = parser.parse_args()
    root = Path(__file__).resolve().parent
    motion = root.parents[1]/'flow_story_20261013/motion_groups'
    inherited = json.loads((motion/'runtime_group_cpu_contract_v1.json').read_text())
    fitted = json.loads(a.fit_result.read_text())
    assert fitted['status'] == 'FAST_WARMSTART_COMPLETE'
    assert fitted['recipe']['steps'] == 500 and fitted['recipe']['seed'] == 20261014
    assert fitted['provenance']['fit_script_sha256'] == sha(root/'fit_fast_cache.py')
    remote = '/data/flow_a/code/research_20261014/fast_warmstart'
    c = {key: inherited[key] for key in ('breadth_root', 'base_args', 'source_pins',
        'head_weights', 'head_weights_sha256', 'head_fit_cache_sha256', 'flow_contract')}
    c.update(schema='flow_fast_runtime_contract_v1', target_count=4,
        motion_root='/data/flow_a/code/research_20261013/motion_groups', adapter_dir=remote+'/borrowed_dynamicswin',
        fast_weights='/data/flow_a/runs/fast_warmstart_20261014/fit_v1/weights.pt',
        fast_weights_sha256=fitted['weights_sha256'], fast_cache_sha256=fitted['provenance']['cache_sha256'],
        fast_fit_result_sha256=sha(a.fit_result), inherited_contract_sha256=sha(motion/'runtime_group_cpu_contract_v1.json'))
    c['base_args'][c['base_args'].index('--output')+1] = '/data/flow_a/runs/fast_warmstart_20261014/runtime_v1/results.json'
    c['source_pins'].pop(c['motion_root']+'/run_runtime_group_pilot.py')
    for name in ('run_fast_runtime.py', 'fast_tile_adapter.py', 'fit_fast_cache.py'):
        c['source_pins'][remote+'/'+name] = sha(root/name)
    borrowed = root.parents[1]/'flow_story_20261012/transfer_experiments/dynamicswin'
    for name in ('adapter.py', 'source_provenance.json'):
        c['source_pins'][c['adapter_dir']+'/'+name] = sha(borrowed/name)
    assert not a.output.exists()
    a.output.write_text(json.dumps(c, indent=2)+'\n')
    print(sha(a.output))


if __name__ == '__main__':
    main()
