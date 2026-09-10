"""Build an isolated Verilator scoreboard; never invokes synthesis or a license.

Use Python 3.12. Every run requires a fresh output directory. All compared
values are synthetic finite-width interval interfaces, not ep34 arithmetic.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
CONFIGS = [(32, 4, 32, 10), (4, 0, 4, 2), (4, 2, 4, 2), (4, 4, 4, 2), (5, 2, 8, 3)]
SOURCES = sorted((BASE / 'rtl').glob('*.sv')) + [BASE / 'sim/packet_test_top.sv']


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Use Python 3.12')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in SOURCES + [BASE / 'sim/packet_scoreboard.cpp', Path(__file__).resolve()]}
    plan = {'status': 'PREDECLARED_FUNCTIONAL_CONFIGS', 'configs': CONFIGS,
            'source_sha256': hashes,
            'scope': 'Synthetic interval RTL, no network execution or PPA',
            'verilator_version': subprocess.check_output(['verilator', '--version'], text=True).strip()}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    lint = ['verilator', '--lint-only', '--Wall', '--top-module', 'packet_test_top'] + list(map(str, SOURCES))
    with (out / 'lint.log').open('w') as f:
        subprocess.run(lint, check=True, stdout=f, stderr=subprocess.STDOUT)
    results = []
    for b, k, w, t in CONFIGS:
        tag = f'b{b}_k{k}_w{w}_t{t}'
        build = out / tag
        cmd = ['verilator', '--cc', '--exe',
               '--top-module', 'packet_test_top', '-Mdir', str(build),
               f'-GB={b}', f'-GK={k}', f'-GW={w}', f'-GT={t}',
               '-CFLAGS', f'-std=c++17 -DB_TEST={b} -DK_TEST={k} -DW_TEST={w} -DT_TEST={t}']
        cmd += list(map(str, SOURCES)) + [str(BASE / 'sim/packet_scoreboard.cpp')]
        with (out / f'{tag}_build.log').open('w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
            subprocess.run(['make', '-C', str(build), '-f', 'Vpacket_test_top.mk', '-j', '2'],
                           check=True, stdout=f, stderr=subprocess.STDOUT)
        run = subprocess.run([str(build / 'Vpacket_test_top')], text=True, capture_output=True)
        (out / f'{tag}_run.log').write_text(run.stdout + run.stderr)
        run.check_returncode()
        result = json.loads(run.stdout)
        assert result['status'] == 'PASS_FUNCTIONAL_RTL_ONLY'
        results.append(result)
        print(json.dumps(result), flush=True)
    assert hashes == {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in hashes}
    summary = {'status': 'PASS_FIVE_CONFIGURATIONS', 'results': results,
               'source_sha256': hashes, 'lint': 'PASS_WALL_DEFAULT_CONFIG',
               'limitations': ['Synthetic interval endpoints supplied by testbench',
                              'External full-T repair result supplied by independent dense comparator',
                              'No BN/PSN interval producer or external SRAM implemented',
                              'No ep34 FP32 equivalence, VCS/DC/PT/Formality closure, AEE or PPA'],
               'PPA_ADMISSION': 0, 'RTL_SPEEDUP_ADMISSION': 0, 'FROZEN_FP_EQUIVALENCE': 0}
    (out / 'result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
