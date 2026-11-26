#!/usr/bin/env python3
"""Bind completed3 R27 artifacts; metadata/bytes only, no GT/model access."""
import argparse
import json
from pathlib import Path
import reduce_raw_v1 as r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actual-root',required=True,type=Path)
    p.add_argument('--audit',required=True,type=Path)
    p.add_argument('--source-input-root',type=Path,default=r.HERE.parent/'dev64_source_snapshot')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();r.require(not a.output.exists(),'Fresh input binding')
    audit=r.read(a.audit)
    r.require(audit['schema']==r.AUDIT_SCHEMA and audit['status']=='PASS' and audit['quality_analysis_allowed'] is True,
              'All formal runs independently complete')
    runs={}
    for arm in r.ARMS:
        path=(a.actual_root/('formal_'+arm+'_v1')/'results.json').resolve()
        h=r.sha(path);r.require(audit['source_results'][arm]==h,'Matching audited original result')
        runs[arm]=dict(path=str(path),sha256=h)
    out=dict(schema='flow_r27_raw_quality_inputs_v1',runs=runs,
             execution_audit=dict(path=str(a.audit.resolve()),sha256=r.sha(a.audit)),
             source_input_root=str(a.source_input_root.resolve()))
    r.write(a.output,out);print(json.dumps({'sha256':r.sha(a.output),'actual_arrays_loaded':False}))

if __name__=='__main__':main()
