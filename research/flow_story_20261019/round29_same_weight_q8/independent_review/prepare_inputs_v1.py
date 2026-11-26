import argparse,json
from pathlib import Path
import reduce_same_weight_v1 as r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r.require(not a.output.exists(),'Fresh input manifest');j=r.read(a.audit)
 r.require(j['schema']==r.AUDIT_SCHEMA and j['status']=='PASS' and j['quality_analysis_allowed'] is True and j['source_result_sha256']==r.sha(a.result),'Actual complete64 audit/result')
 r.write(a.output,dict(schema='flow_r29_same_weight_quality_inputs_v1',run=dict(path=str(a.result.resolve()),sha256=r.sha(a.result)),execution_audit=dict(path=str(a.audit.resolve()),sha256=r.sha(a.audit))))
 print(json.dumps(dict(sha256=r.sha(a.output),GT_values_read=False)))
