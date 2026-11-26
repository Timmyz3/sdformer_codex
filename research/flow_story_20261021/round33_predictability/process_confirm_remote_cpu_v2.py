"""Bound CPU processing after confirmation completion; no training or GPU calls."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess

B = Path('/data/flow_a/analysis_round33_mirror/research/flow_story_20261021/round33_predictability')
PYTHON = '/data/flow_a/envs/v2/bin/python'
SOURCE = '/data/flow_a/data/DSEC_native_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    result = json.loads((B/'actual/confirm_v1/results.json').read_text())
    assert result['status'] == 'R33_COMPLETE', 'Collector must finish before CPU processing'
    output = B/'CONFIRM_REMOTE_CPU_EXECUTION_RECEIPT_V2.json'
    assert not output.exists()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    calls = [
        [PYTHON, '-B', str(B/'independent_review/audit_collector_confirm_v2.py'),
         '--run', str(B/'actual/confirm_v1'), '--source', SOURCE,
         '--output', str(B/'independent_review/CONFIRM_REMOTE_EXECUTION_AUDIT_V1.json'),
         '--heads-freeze', str(B/'confirmation_freeze_v1/HEADS_FREEZE_V1.json'),
         '--launch-receipt', str(B/'CONFIRM_LAUNCH_RECEIPT_V1.json')],
        [PYTHON, '-B', str(B/'make_pack_confirm_v2.py'), '--run', str(B/'actual/confirm_v1'),
         '--source', SOURCE, '--output', str(B/'packs/confirm_128_v1'),
         '--predictors-sha256', '35663dd6b999621652d9d185d72b967e33c380e1064a09ec75540d2759cd61c3'],
        [PYTHON, '-B', str(B/'augment_directional_pack.py'), '--pack', str(B/'packs/confirm_128_v1'),
         '--source', SOURCE, '--output', str(B/'packs/confirm_160_v1'),
         '--predictors-sha256', 'bba3733c6d4f63e8a71d075d0c73d3c696956b5c746ede142f79a4d9b7543cb0'],
    ]
    receipt = dict(schema='flow_r33_confirm_cpu_execution_v1', runner_sha256=sha(__file__),
                   source_result_sha256=sha(B/'actual/confirm_v1/results.json'), GPU_used=False,
                   source_location=SOURCE, calls=[], prior_attempt='V1 stopped during source audit before numerical quality: remote mirror missing unchanged algorithm/run_predictors.py; source copied, no GPU rerun or data changes.')
    for args in calls:
        began = datetime.datetime.now(datetime.timezone.utc).isoformat()
        c = subprocess.run(args, env=env, capture_output=True, text=True)
        receipt['calls'].append(dict(args=args, script_sha256=sha(args[2]), started_utc=began,
                                    finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    returncode=c.returncode, stdout=c.stdout, stderr=c.stderr))
        output.write_text(json.dumps(receipt, indent=2)+'\n')
        print(Path(args[2]).name, c.returncode, flush=True)
        if c.returncode:
            print(c.stdout, c.stderr, flush=True)
            c.check_returncode()
    receipt['status'] = 'COMPLETE'
    receipt['artifacts'] = {str(p.relative_to(B)): sha(p) for folder in [B/'packs/confirm_128_v1', B/'packs/confirm_160_v1'] for p in folder.iterdir()}
    output.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
