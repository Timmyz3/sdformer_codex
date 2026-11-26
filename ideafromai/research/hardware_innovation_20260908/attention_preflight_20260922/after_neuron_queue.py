"""One bounded inference probe; neuron jobs have priority. No training here."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
NEURON = ROOT.parent / 'atlif_psn_replication_20260922'
REPO = ROOT.parents[3] / 'SDformer'


def main():
    status = dict(state='waiting_for_neuron_replication', complete=False,
                  priority='neuron repair first; this job is inference-only')
    def save():
        (ROOT / 'QUEUE_STATUS.json').write_text(json.dumps(status, indent=2)+'\n')
    save()
    while True:
        current = json.loads((NEURON / 'QUEUE_STATUS.json').read_text())
        if any(j.get('status') == 'failed' for j in current.get('jobs', [])):
            status['state'] = 'not_started_neuron_job_failed'
            save()
            return
        if current.get('complete'):
            break
        pid = int((NEURON / 'QUEUE_PID.txt').read_text())
        proc = Path(f'/proc/{pid}/cmdline')
        if not proc.exists() or b'run_queue.py' not in proc.read_bytes():
            status['state'] = 'not_started_neuron_queue_exited_without_complete'
            save()
            return
        time.sleep(30)
    # Fixed corrected reference, first predeclared seed. NOT selected by best
    # validation score; normalized remains a separate neuron candidate.
    checkpoint = NEURON / 'seed_20260922/official_eps_256successful.pth'
    output = ROOT / 'fullres10'
    if output.exists():
        raise RuntimeError(f'Existing preflight output: {output}')
    command = [sys.executable, '-u', str(ROOT / 'run_preflight.py'),
        '--repo', str(REPO), '--config', str(NEURON / 'replication_config.yml'),
        '--checkpoint', str(checkpoint), '--output', str(output)]
    status.update(state='running_inference_only', started=datetime.datetime.now().astimezone().isoformat(), command=command)
    save()
    env = dict(os.environ, SDFORMER_USE_MLFLOW='0', SDFORMER_MLFLOW_MODEL_LOGGING='0',
               SDFORMER_SNN_BACKEND='cupy', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    with (ROOT / 'preflight.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    status.update(state='complete' if result.returncode == 0 else 'failed',
                  exit_code=result.returncode, complete=result.returncode == 0,
                  ended=datetime.datetime.now().astimezone().isoformat())
    save()


if __name__ == '__main__':
    main()
