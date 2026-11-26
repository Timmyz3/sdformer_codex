#!/usr/bin/env python3
"""Completed result/audit metadata binding only; no array reads."""
import argparse
from pathlib import Path
import reduce_joint_raw_v1 as r

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--result',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r.require(not a.output.exists(),'Fresh independent input manifest')
    audit=r.read(a.audit)
    r.require(audit['schema']==r.AUDIT_SCHEMA and audit['status']=='PASS'
        and audit['quality_analysis_allowed'] is True and audit['engineering_smoke'] is False,
        'Actual formal audit only')
    r.require(audit['source_result_sha256']==r.sha(a.result),'Bound genuine new result')
    r.write(a.output,dict(schema='flow_r30_joint_quality_inputs_v1',
        run=dict(path=str(a.result.resolve()),sha256=r.sha(a.result)),
        execution_audit=dict(path=str(a.audit.resolve()),sha256=r.sha(a.audit))))
    print(r.sha(a.output))
if __name__=='__main__':main()
