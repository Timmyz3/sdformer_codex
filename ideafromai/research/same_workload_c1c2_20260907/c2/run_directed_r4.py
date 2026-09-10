import sys
assert sys.version_info[:2] == (3, 12)
from pathlib import Path
import hashlib, json, shutil, subprocess, time

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / 'functional_r4c'
OUT = ROOT / 'functional_r4_directed'
OUT.mkdir(exist_ok=False)
start = time.time()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

history = []
try:
    admitted = json.loads((CAMPAIGN / 'result.json').read_text())
    assert admitted['status'] == 'PASS_BOUNDED_RTL_BEHAVIORAL_IO_NO_PPA'
    for name in ['c2_ftp_bank_parallel.sv', 'scoreboard_r4.cpp']:
        assert sha(ROOT / name) == admitted['input_sha256'][str(ROOT / name)]
    plan = json.loads((ROOT / 'plan_directed_r4.json').read_text())
    fixture = OUT / 'fixture'
    fixture.mkdir()
    weights = [int(x) for x in (ROOT / 'weights.txt').read_text().split()]
    assert len(weights) == 384 * 96
    assert sha(ROOT / 'weights.txt') == admitted['input_sha256'][str(ROOT / 'weights.txt')]
    shutil.copyfile(ROOT / 'weights.txt', fixture / 'weights.txt')
    patterns = [
        [0] * 384,
        [0x3ff] * 384,
        [0x140 if c % 16 < 3 else 0 for c in range(384)],
        [0x140 if c % 17 == 0 else (0x155 if c % 2 == 0 else 0x2aa) for c in range(384)]
    ]
    masks = [patterns[p % 4] for p in range(32)]
    with (fixture / 'support.txt').open('x') as f:
        f.write('32 10 384 96 256\n')
        for row in masks:
            f.write('\n'.join(format(x, 'x') for x in row) + '\n')
    truth = []
    for row in patterns:
        truth.append([sum(weights[c * 96 + h] for c in range(384) if (row[c] >> t) & 1)
                      for t in range(10) for h in range(96)])
    with (fixture / 'golden.txt').open('x') as f:
        for p in range(32):
            f.write('\n'.join(str(x) for x in truth[p % 4]) + '\n')
    exe = CAMPAIGN / 'ftp_obj/c2_sim'
    inputs = [ROOT / 'plan_directed_r4.json', ROOT / 'run_directed_r4.py',
              ROOT / 'c2_ftp_bank_parallel.sv', ROOT / 'scoreboard_r4.cpp',
              CAMPAIGN / 'result.json', exe,
              fixture / 'support.txt', fixture / 'weights.txt', fixture / 'golden.txt']
    identities = {str(p): sha(p) for p in inputs}
    (OUT / 'input_sha256.json').write_text(json.dumps(identities, indent=2) + '\n')
    snapshot = OUT / 'source_snapshot'
    snapshot.mkdir()
    for name in ['plan_directed_r4.json', 'run_directed_r4.py', 'c2_ftp_bank_parallel.sv', 'scoreboard_r4.cpp']:
        shutil.copyfile(ROOT / name, snapshot / name)
    results = {}
    for mode in [0, 1]:
        axis = 'shared' if mode else 'direct'
        cmd = [str(exe), str(fixture), str(mode), str(OUT / (axis + '.json'))]
        record = {'axis': axis, 'argv': cmd, 'cwd': str(ROOT)}
        history.append(record)
        (OUT / 'commands.json').write_text(json.dumps(history, indent=2) + '\n')
        with (OUT / (axis + '.log')).open('xb') as log:
            run = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        record['returncode'] = run.returncode
        (OUT / 'commands.json').write_text(json.dumps(history, indent=2) + '\n')
        assert run.returncode == 0, axis
        result = json.loads((OUT / (axis + '.json')).read_text())
        assert result['integer_checks'] == 30720 and result['errors'] == 0
        assert max(q['max_temporal_vectors_written_per_bank_cycle'] for q in result['quads']) == 10
        expected_requests = 6 * sum(m != 0 for row in masks for m in row)
        assert result['logical_vector_requests'] == expected_requests
        results[axis] = result
        print(axis, 'PASS T10', flush=True)
    for p in inputs:
        assert sha(p) == identities[str(p)], str(p)
    (OUT / 'result.json').write_text(json.dumps({
        'status': 'PASS_DIRECTED_FUNCTIONAL_ONLY_NO_PERFORMANCE_CLAIM',
        'elapsed_seconds': time.time() - start, 'captured_data': False,
        'plan': plan, 'input_sha256': identities, 'axes': results,
        'cycle_numbers': 'Raw driver diagnostics only; not admitted to captured performance comparisons'
    }, indent=2) + '\n')
except Exception as error:
    (OUT / 'FAILED.json').write_text(json.dumps({
        'status': 'FAILED_DO_NOT_CITE', 'error': repr(error),
        'elapsed_seconds': time.time() - start
    }, indent=2) + '\n')
    raise
