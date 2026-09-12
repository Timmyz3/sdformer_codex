"""Run the already fixed three representation arms after the active combination process."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--combination-pid', type=int, required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--next-script', default='run_representations.py')
    parser.add_argument('--prerequisite', default='combinations/run.json')
    parser.add_argument('--log', default='representations.log')
    args = parser.parse_args()
    while True:
        try:
            os.kill(args.combination_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(20)
    assert json.loads((HERE / args.prerequisite).read_text())['complete']
    with (HERE / args.log).open('w') as log:
        subprocess.run([sys.executable, '-u', str(HERE / args.next_script),
                        '--root', args.root], stdout=log, stderr=subprocess.STDOUT, check=True)
    print('QUEUE_STAGE_COMPLETE', args.next_script, flush=True)


if __name__ == '__main__':
    main()
