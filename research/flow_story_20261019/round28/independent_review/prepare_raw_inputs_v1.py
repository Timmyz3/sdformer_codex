#!/usr/bin/env python3
"""Only completed Q4 metadata binding; no original/prediction array access."""
import argparse,json
from pathlib import Path
import reduce_raw_v1 as r

def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r.require(not a.output.exists(),'New input manifest')
 audit=r.read(a.audit);r.require(audit['schema']==r.AUDIT_SCHEMA and audit['status']=='PASS' and audit['quality_analysis_allowed'] is True,'Actual complete audit')
 h=r.sha(a.result);r.require(audit['source_results']=={'Q4':h},'Bound one Q4 result')
 r.write(a.output,dict(schema='flow_r28_raw_quality_inputs_v1',run=dict(path=str(a.result.resolve()),sha256=h),execution_audit=dict(path=str(a.audit.resolve()),sha256=r.sha(a.audit))))
 print(json.dumps(dict(sha256=r.sha(a.output),GT_values_loaded=False)))
if __name__=='__main__':main()
