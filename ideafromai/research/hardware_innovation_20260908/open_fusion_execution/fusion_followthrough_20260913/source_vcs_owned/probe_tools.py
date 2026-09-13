"""One read-only VCS/version/license probe under the shared EDA mutex."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
LOCK = '/tmp/date_dual_synopsys_same_uid_eda_queue.lock'
LICENSE = '27030@ic.ismd-nemo'


def main():
    result = dict(vcs=shutil.which('vcs'), lmutil=shutil.which('lmutil'),
                  verilator=shutil.which('verilator'), license_server=LICENSE, lock=LOCK)
    env = os.environ.copy()
    env['SNPSLMD_LICENSE_FILE'] = LICENSE
    env['LM_LICENSE_FILE'] = LICENSE
    with open(LOCK, 'a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result['lock_acquired'] = False
        else:
            result['lock_acquired'] = True
            if result['vcs']:
                proc = subprocess.run([result['vcs'], '-full64', '-ID'], env=env,
                    cwd=HERE, capture_output=True, text=True, timeout=30)
                result['vcs_version'] = dict(returncode=proc.returncode,
                    output='\n'.join(line for line in (proc.stdout+proc.stderr).splitlines()
                                     if 'host ID' not in line))
            if result['lmutil']:
                proc = subprocess.run([result['lmutil'], 'lmstat', '-a', '-c', LICENSE],
                    env=env, cwd=HERE, capture_output=True, text=True, timeout=30)
                # Keep server/feature availability, not other users' checkout records.
                lines = [line.strip() for line in (proc.stdout+proc.stderr).splitlines()
                         if 'license server' in line.lower() or 'license manager' in line.lower()
                         or 'UP ' in line or ('Users of ' in line and 'vcs' in line.lower())
                         or 'error' in line.lower() or 'Cannot' in line]
                result['license_status'] = dict(returncode=proc.returncode, summary=lines)
    (HERE/'tool_probe.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
