import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import io_common as c
import release_fit_inputs_v1 as release
import reduce_projection_raw_v1 as r

class Checks(unittest.TestCase):
    def test_full_projection_bind_and_bad_role_or_sequence_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);result=p/'native_results.json';audit=p/'audit.json';result.write_text('synthetic');audit.write_text('synthetic')
            state=dict(members=[],native={},paths={},evidence={},result_path=result,audit_path=audit)
            for i in range(24):
                tid=str(i);native=p/(tid+'_native.npz');native.write_bytes(b'metadata-only native fixture')
                state['members'].append(dict(target_id=tid,sequence='seq'+str(i%8)))
                state['native'][tid]=(native,{})
                state['paths'][tid]={'gt':p/(tid+'_gt.npy'),'mask':p/(tid+'_mask.npy')}
                state['evidence'][tid]={'gt':'a'*64,'mask':'b'*64}
            members=release.release(state,'freeze')['members'];pins=[];ext={}
            for name in ('projection_fitter','projection_core'):
                src=p/(name+'.py');src.write_text('#synthetic source pin\n');pins.append(dict(path=str(src),sha256=c.sha(src)))
                ext[name]=dict(relative_path=src.name,sha256=c.sha(src))
            fit=dict(schema='flow_r22_motion_projection_fit_inputs_v1',graph_id=c.GRAPH,members=members,roles={'sha256':c.ROLE_SHA},source_pins=pins)
            fp=p/'fit.json';fp.write_text(json.dumps(fit))
            pred=dict(schema='flow_r22_motion_projection_predictions_v1',status='FROZEN_BEFORE_DEVELOPMENT_QUALITY',inputs_sha256=c.sha(fp),
              source_result_sha256=c.sha(result),execution_audit_sha256=c.sha(audit),graph_id=c.GRAPH,fit_ids=[str(i) for i in range(16)],development_ids=[str(i) for i in range(16,24)],selector_training=False,original_model_training=False,CUDA_initialized=False,device='CPU',predictions=[])
            for row in members:
                tid=row['target_id'];arr={k:np.zeros((30,40,2),np.float32) for k in r.core.METHODS};z=p/(tid+'_proxy.npz');np.savez_compressed(z,**arr)
                pred['predictions'].append(dict(target_id=tid,role=row['role'],file=z.name,sha256=c.sha(z),payloads={k:c.payload(v) for k,v in arr.items()}))
            pp=p/'predictions.json';pp.write_text(json.dumps(pred));args=SimpleNamespace(predictions=pp,predictions_sha256=c.sha(pp),fit_inputs=fp)
            with patch.object(c,'REPO',p):
                _,values=r.bind_projection(args,{'external_sources':ext},state);self.assertEqual(len(values),24)
                for corruption in ('sequence','dev_GT'):
                    bad=copy.deepcopy(fit)
                    if corruption=='sequence':bad['members'][0]['sequence']='forged_sequence'
                    else:bad['members'][16]['fit_GT']={'path':'forbidden','sha256':'a'*64}
                    fp.write_text(json.dumps(bad));pred['inputs_sha256']=c.sha(fp);pp.write_text(json.dumps(pred));args.predictions_sha256=c.sha(pp)
                    with self.assertRaisesRegex(ValueError,'canonical sequences'):r.bind_projection(args,{'external_sources':ext},state)
    def test_fixed_source_metrics_available(self):
        self.assertEqual(c.sha(c.MATH),c.MATH_SHA);self.assertEqual(c.sha(c.ROLES),c.ROLE_SHA)
        self.assertEqual(c.sha(c.HERE/'ANALYSIS_FREEZE_V1.json'),c.ANALYSIS_SHA)
if __name__=='__main__':unittest.main()
