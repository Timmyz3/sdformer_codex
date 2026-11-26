"""Finite A100 queue: isolated two-step smoke, then two paired replications."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3] / 'SDformer'
CONFIG = ROOT / 'replication_config.yml'
CHECKPOINT = REPO / 'neuron_experiments/H9_bipolar_self_attention/results/dsec_c12_alpha0125_ep29_resume5_20260830/checkpoint_epoch34.pth'
SEEDS = [20260922, 20260923]


def main():
    status = dict(started=datetime.datetime.now().astimezone().isoformat(), complete=False,
                  machine='223.109.239.65:30163 / A100-SXM4-80GB', jobs=[])

    def save():
        (ROOT / 'QUEUE_STATUS.json').write_text(json.dumps(status, indent=2) + '\n')

    jobs = [('smoke', SEEDS[0], 2, True)] + [(f'seed_{seed}', seed, 256, False) for seed in SEEDS]
    environment = dict(os.environ, SDFORMER_USE_MLFLOW='0', SDFORMER_MLFLOW_MODEL_LOGGING='0',
                       SDFORMER_SNN_BACKEND='cupy', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    for name, seed, steps, smoke in jobs:
        directory = ROOT / name
        if directory.exists():
            raise RuntimeError(f'Refusing to overwrite an existing run: {directory}')
        directory.mkdir()
        command = [sys.executable, '-u', str(ROOT / 'run_replication.py'), '--seed', str(seed),
                   '--steps', str(steps), '--arms', 'official_eps,normalized', '--repo', str(REPO),
                   '--config', str(CONFIG), '--checkpoint', str(CHECKPOINT), '--output', str(directory)]
        if smoke:
            command.append('--skip-full-validation')
        job = dict(name=name, seed=seed, steps_per_arm=steps, command=command,
                   started=datetime.datetime.now().astimezone().isoformat(), status='running')
        status['jobs'].append(job)
        save()
        with (ROOT / f'{name}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        job.update(exit_code=result.returncode, ended=datetime.datetime.now().astimezone().isoformat(),
                   status='complete' if result.returncode == 0 else 'failed')
        save()
        if result.returncode:
            raise SystemExit(result.returncode)
        if not smoke:
            from summarize_replication import summarize
            summarize(directory)
    from summarize_replication import combine
    combine(ROOT, SEEDS)
    status['complete'] = True
    status['ended'] = datetime.datetime.now().astimezone().isoformat()
    save()


if __name__ == '__main__':
    main()
