"""Verilator only. Each attempt gets a new immutable directory, including failures."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='New run name under records; never reused')
    args = ap.parse_args()
    assert args.run.replace('_','').isalnum()
    result_dir = ROOT/'records'/args.run
    result_dir.mkdir(parents=True,exist_ok=False)
    plan = json.loads((ROOT/'plan.json').read_text())
    fixture = ROOT/'fixtures/fc2_sample0_diagnostic.txt'
    fixture_manifest = json.loads((ROOT/'fixtures/manifest.json').read_text())
    assert sha(fixture)==fixture_manifest['fixture_sha256']
    protected = ['plan.json','contract.md','rtl/c2_temporal_shared_sum.sv','sim/scoreboard.cpp',
                 'sim/make_fixture.py','sim/run_functional.py','fixtures/manifest.json']
    hashes = {p:sha(ROOT/p) for p in protected}
    log = {'status':'RUNNING','scope':'Verilator functional protocol only; simulation cycles are not an RTL speedup',
           'input_sha256':hashes,'fixture_sha256':sha(fixture),
           'tool_version':subprocess.check_output(['/usr/bin/verilator','--version'],text=True).strip(),
           'configs':[],'claim_boundary':plan['claim_boundary']}
    (result_dir/'plan_snapshot.json').write_text(json.dumps(log,indent=2)+'\n')
    try:
        for cfg in plan['configs']:
            name=f"t{cfg['T']}_d{cfg['D']}"
            folder=result_dir/name
            folder.mkdir()
            obj=folder/'obj'
            commands = [
                ['/usr/bin/verilator','--cc','--exe','-Wall','--top-module','c2_temporal_shared_sum',
                 '--Mdir',str(obj),f"-GT={cfg['T']}",f"-GSLOTS={cfg['D']}",
                 '-CFLAGS',f"-std=c++17 -O2 -DTEST_T={cfg['T']} -DTEST_D={cfg['D']}",
                 str(ROOT/'rtl/c2_temporal_shared_sum.sv'),str(ROOT/'sim/scoreboard.cpp')],
                ['/usr/bin/make','-C',str(obj),'-f','Vc2_temporal_shared_sum.mk','-j2'],
                [str(obj/'Vc2_temporal_shared_sum')]+([str(fixture)] if cfg['fixture'] else [])]
            last=''
            for stage,command in zip(['verilate','compile','simulate'],commands):
                start=time.monotonic()
                process=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                (folder/f'{stage}.log').write_text(process.stdout)
                (folder/f'{stage}_command.json').write_text(json.dumps({'argv':command,'returncode':process.returncode,
                    'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
                if process.returncode:
                    raise RuntimeError(f'{name} {stage} failed; see {folder}/{stage}.log')
                last=process.stdout
            counts=json.loads(last.strip())
            assert counts['errors']==0 and counts['status']=='PASS_FUNCTIONAL_ONLY'
            log['configs'].append({'config':cfg,'counts':counts})
            print(name,counts['status'],'sources',counts['sources'],flush=True)
        assert all(sha(ROOT/p)==h for p,h in hashes.items()),'Inputs changed during simulation'
        log['status']='PASS_FUNCTIONAL_ONLY'
    except Exception as exc:
        log['status']='FAILED_DO_NOT_CITE';log['error']=str(exc)
        raise
    finally:
        with (result_dir/'result.json').open('x') as f:
            json.dump(log,f,ensure_ascii=False,indent=2);f.write('\n')

if __name__=='__main__':
    main()
