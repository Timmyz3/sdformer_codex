#!/usr/bin/env python3
"""Serial bounded launch template; defaults to preflight, never provisions GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ARMS=('native','FF','CC','fine_s2b2')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True,help='JSON: runtime_manifest,membership,bn_buffers,bn_receipt absolute paths and *_sha256')
    p.add_argument('--inputs-sha256',required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','recover'),required=True)
    p.add_argument('--conditions',nargs='+',choices=('reference','diversity'),default=['reference','diversity'])
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();assert sha(a.inputs)==a.inputs_sha256,'Input binding JSON differs'
    assert len(a.conditions)==len(set(a.conditions)),'Duplicate condition'
    value=json.loads(a.inputs.read_text())
    args=[]
    for name in ('runtime_manifest','membership','bn_buffers','bn_receipt'):
        assert sha(value[name])==value[name+'_sha256'],'Input bytes differ: '+name
        args.extend(['--'+name.replace('_','-'),value[name],'--'+name.replace('_','-')+'-sha256',value[name+'_sha256']])
    a.output_root.mkdir(parents=True,exist_ok=True)
    for condition in a.conditions:
        for arm in ARMS:
            output=a.output_root/condition/a.stage/arm
            cmd=[sys.executable,'-B',str(HERE/'run_coverage_recovery.py')]+args+[
                '--condition',condition,'--mode',a.stage,'--arm',arm,'--output-dir',str(output),'--wall-minutes','30']
            if a.stage=='recover':
                for smoke_arm in ARMS:cmd+=['--smoke-receipt',str(a.output_root/condition/'smoke'/smoke_arm/'results.json')]
            if a.execute:cmd+=['--execute']
            else:cmd+=['--preflight-output',str(a.output_root/(condition+'_'+a.stage+'_'+arm+'_preflight.json'))]
            print(json.dumps({'condition':condition,'arm':arm,'execute':a.execute,'argv':cmd}),flush=True)
            subprocess.run(cmd,check=True)


if __name__=='__main__':main()
