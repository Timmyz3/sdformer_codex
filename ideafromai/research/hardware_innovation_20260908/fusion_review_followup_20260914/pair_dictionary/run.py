from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess

H = Path(__file__).resolve().parent
with (H / 'build.log').open('w') as log:
    subprocess.run(['verilator', '-Wall', '--cc', '--exe', '--top-module', 'decomp_core', '--Mdir', 'obj', 'decomp_core.sv', 'tb.cpp', '-CFLAGS', '-O3'], cwd=H, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(['make', '-C', 'obj', '-f', 'Vdecomp_core.mk', '-j2'], cwd=H, stdout=log, stderr=subprocess.STDOUT, check=True)
cases = json.loads((H / 'fixtures.json').read_text())
def run(job):
    case, mode, stall = job
    p = subprocess.run([str(H / 'obj/Vdecomp_core'), str(H / 'fixtures' / case['name']), str(mode), str(stall)], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError((job, p.returncode, p.stdout, p.stderr))
    return [dict(json.loads(line), fixture=case['name']) for line in p.stdout.splitlines()]
rows = []
with ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(run, [(c, m, s) for c in cases for m in [14, 15] for s in [0, 1]]):
        rows.extend(result)
(H / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
summary = {}
for mode in [14, 15]:
    for stall in [0, 1]:
        a = [r for r in rows if r['fixture'].startswith('real_') and r['mode'] == mode and r['stall'] == stall and r['command'] == 0]
        summary[f'mode{mode}_stall{stall}'] = {key: sum(r[key] for r in a) for key in rows[0] if key not in ['fixture', 'mode', 'stall', 'command', 'state_cycles']}
(H / 'SUMMARY.json').write_text(json.dumps(dict(commands=len(rows), outputs_compared=sum(r['outputs'] for r in rows), real_first_commands=summary), indent=2) + '\n')
print(json.dumps(summary, indent=2))
