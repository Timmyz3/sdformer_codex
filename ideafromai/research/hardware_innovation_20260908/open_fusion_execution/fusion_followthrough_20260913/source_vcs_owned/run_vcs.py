"""One isolated full64 VCS build and the fixed twenty source cases.

Uses the shared Synopsys mutex; exits on the first compile/simulation/check
failure. Does not retry an attempted campaign or touch any earlier directory.
"""
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
LOCK = '/tmp/date_dual_synopsys_same_uid_eda_queue.lock'
LICENSE = '27030@ic.ismd-nemo'
RESULT = HERE/'results.json'


def save(result):
    RESULT.write_text(json.dumps(result, indent=2)+'\n')


def main():
    if RESULT.exists():
        raise SystemExit('This isolated campaign has already been attempted; inspect results/logs, no automatic retry.')
    manifest = json.loads((HERE/'manifest.json').read_text())
    tool = json.loads((HERE/'tool_probe.json').read_text())
    vcs = shutil.which('vcs')
    result = dict(scope=__doc__, status='waiting_for_mutex', complete=False,
                  VCS_completed=False, tool_probe=tool, rows=[], required_cases=20,
                  lock=LOCK, license_server=LICENSE, original_RTL=manifest['original_RTL'],
                  original_RTL_modified=False, synthesis=False, STA=False, formal=False,
                  PPA=False, full_chain=False, simulator_comparison_is_not_speedup=True)
    save(result)
    if vcs is None:
        result['status']='VCS_missing'; save(result)
        raise SystemExit('VCS absent; reviewable SV TB and fixtures remain available.')
    build = HERE/'build'; build.mkdir()
    env = os.environ.copy()
    env['SNPSLMD_LICENSE_FILE']=LICENSE
    env['LM_LICENSE_FILE']=LICENSE
    env['VCS_HOME']=str(Path(vcs).resolve().parents[1])
    command = [vcs, '-full64', '-sverilog', '-timescale=1ns/1ps',
               '-top', 'tb_source_vcs', '-Mdir='+str(build/'csrc'),
               '-o', str(build/'simv'), '-l', str(HERE/'compile.log'),
               manifest['original_RTL'], str(HERE/'tb_source_vcs.sv')]
    result['compile_command']=command
    print('EDA_MUTEX_WAIT', LOCK, flush=True)
    with open(LOCK, 'a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        result['status']='compiling'; save(result)
        print('EDA_MUTEX_ACQUIRED; VCS_COMPILE_START', flush=True)
        start=time.monotonic()
        try:
            proc=subprocess.run(command, env=env, cwd=HERE, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            result['status']='compile_timeout'; result['error']=str(exc); save(result)
            raise SystemExit('Compile timed out; stopped with no retry.')
        (HERE/'compile_driver.log').write_text(proc.stdout+proc.stderr)
        result['compile_returncode']=proc.returncode
        result['compile_wall_seconds']=time.monotonic()-start
        if proc.returncode != 0:
            result['status']='compile_failed'; save(result)
            raise SystemExit('Compile failed; stopped with no retry. See compile_driver.log.')
        print('VCS_COMPILE_DONE', flush=True)
        result['status']='running'; save(result)
        for case in manifest['cases']:
            folder=HERE/'runs'/case['name']; folder.mkdir(parents=True)
            cmd=[str(build/'simv'), '+PROGRAM='+case['program'], '+INPUT='+case['inputs'],
                 '+GOLD='+case['gold'], '+MODE='+case['mode'],
                 '+PROGRAM_LENGTH='+str(case['program_length']), '+TILES='+str(case['tiles'])]
            print('CASE_START', case['name'], flush=True)
            start=time.monotonic()
            try:
                proc=subprocess.run(cmd, env=env, cwd=folder, capture_output=True, text=True, timeout=180)
            except subprocess.TimeoutExpired as exc:
                result['status']='simulation_timeout'; result['failed_case']=case['name']
                result['error']=str(exc); save(result)
                raise SystemExit('Simulation timed out; stopped with no retry.')
            output=proc.stdout+proc.stderr
            (folder/'simulation.log').write_text(output)
            parsed=[line.split('SOURCE_VCS_RESULT ', 1)[1] for line in output.splitlines()
                    if line.startswith('SOURCE_VCS_RESULT ')]
            if proc.returncode != 0 or len(parsed) != 1:
                result['status']='simulation_failed'; result['failed_case']=case['name']
                result['returncode']=proc.returncode; save(result)
                raise SystemExit('Simulation failed; stopped with no retry. See case simulation.log.')
            actual=json.loads(parsed[0])
            expected=case['expected']
            comparison={k:dict(VCS=actual.get(k), archived_Verilator=v)
                        for k,v in expected.items() if actual.get(k)!=v}
            row=dict(name=case['name'], family=case['family'], window=case['window'], mode=case['mode'],
                     command=cmd, measured=actual, archived_Verilator=expected,
                     reference_file=case['prior_results'], field_mismatches=comparison,
                     gate_and_RF_exact=actual['differences']==0,
                     all_cycle_port_stall_counts_equal=not comparison,
                     simulation_wall_seconds=time.monotonic()-start,
                     log=str(folder/'simulation.log'))
            result['rows'].append(row); save(result)
            if comparison:
                result['status']='cross_simulator_count_mismatch'; result['failed_case']=case['name']; save(result)
                raise SystemExit('VCS values passed but counters differ; stopped for review, no retry.')
            print('CASE_DONE', case['name'], actual['cycles'], 'cycles; gates/RF/counts exact', flush=True)
        result['status']='complete'; result['complete']=True; result['VCS_completed']=True
        result['totals']=dict(cases=len(result['rows']),
            cycles=sum(x['measured']['cycles'] for x in result['rows']),
            RF_vector_writebacks_checked=sum(x['measured']['RF_vector_writebacks_checked'] for x in result['rows']),
            RF_lane_values_checked=8*sum(x['measured']['RF_vector_writebacks_checked'] for x in result['rows']),
            gate_bits_checked=sum(x['measured']['gate_bits_checked'] for x in result['rows']),
            gate_or_RF_differences=0, counter_field_mismatches=0)
        save(result)
        print('VCS_SOURCE_COMPLETE', json.dumps(result['totals']), flush=True)
    print('EDA_MUTEX_RELEASED', flush=True)


if __name__ == '__main__':
    main()
