import ast,tempfile,unittest
from pathlib import Path
import io_common as c
import release_fit_inputs_v1 as r
class Checks(unittest.TestCase):
    def test_plan_fixed_replay_roles(self):
        rows=[{'target_id':str(i)} for i in range(24)];p=c.plan(rows)
        self.assertEqual(len(p),26);self.assertEqual(sum(v[3] for v in p),24)
        self.assertEqual(p[1],('0','fit','native_repeat1',False));self.assertEqual(p[2],('0','fit','native_repeat2',False))
        self.assertEqual([v[1] for v in p if v[3]],['fit']*16+['development']*8)
    def test_fit_only_map_no_dev_labels(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);rp=p/'results.json';ap=p/'audit.json';rp.write_text('synthetic');ap.write_text('synthetic')
            state=dict(members=[],native={},paths={},evidence={},result_path=rp,audit_path=ap)
            for i in range(24):
                tid=str(i);n=p/(tid+'.npz');n.write_bytes(b'synthetic metadata-only archive')
                state['members'].append(dict(target_id=tid,sequence='s'+str(i%8)))
                state['native'][tid]=(n,{})
                state['paths'][tid]={'gt':p/(tid+'_gt.npy'),'mask':p/(tid+'_mask.npy')}
                state['evidence'][tid]={'gt':'a'*64,'mask':'b'*64}
            out=r.release(state,'freeze')
            self.assertEqual(len(out['members']),24);self.assertEqual(out['fit_GT_references'],16)
            self.assertTrue(all('fit_GT' not in v and 'fit_mask' not in v for v in out['members'][16:]));self.assertFalse(out['GT_array_values_loaded'])
            self.assertEqual(out['development_GT_references'],0);self.assertEqual(out['sealed_data_references'],0)
    def test_no_array_loader_in_source_release(self):
        for p in (Path(c.__file__),Path(r.__file__)):
            tree=ast.parse(p.read_text())
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load' for n in ast.walk(tree)))
if __name__=='__main__':unittest.main()
