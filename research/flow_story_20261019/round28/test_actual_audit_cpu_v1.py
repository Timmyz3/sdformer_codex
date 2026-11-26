#!/usr/bin/env python3
"""Synthetic audit rejection checks; no model forward, checkpoint, or GT read."""
import copy,hashlib,importlib.util,json,math,os,sys,time,unittest
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
a=load(HERE/'audit_actual_recovery_v1.py','r28_audit_cpu')

def fixture(geometry):
    calls={};work={}
    for name in sorted(a.TARGETS):
        if name in a.LINEARS:
            stage=int(name.split('.layers.')[1].split('.')[0]);c=96*2**stage;h,w=geometry[0]//(4*2**stage),geometry[1]//(4*2**stage)
            ci,co=(c,c*4) if name.endswith('.fc1') else (c*4,c)
            ins,outs=[10,1,h,w,ci],[10,1,h,w,co];k=ci
        else:
            h,w=geometry[0]//2,geometry[1]//2;ci=co=96;k=864;ins,outs=[10,1,ci,h,w],[10,1,co,h,w]
        calls[name]=[dict(input_shape=ins,output_shape=outs,module_class='STEAffine')]
        # Independent per-pixel classification; do not call implementation or auditor helpers.
        low=10*sum(y//9<h//9 and x//9<w//9 for y in range(h) for x in range(w));rows=10*h*w;tail=rows-low
        chunk=min(65536,(8*1024*1024)//co)//8*8;kernel=[]
        for bit in range(8):
            for start in range(0,tail if bit in (0,1,2,3) else rows,chunk):
                n=min(chunk,(tail if bit<4 else rows)-start)
                kernel.append(dict(kind='binary_weight_plane',bit=bit,rows=n,padding_rows=max(24,((n+7)//8)*8)-n,
                                   scope='tail' if bit<4 else 'all',coefficient=-128 if bit==7 else 2**bit))
        work[name]=dict(mode='Q4',reference=False,rows=rows,tail_rows=tail,complete_rows=low,Cin=k,Cout=co,
            binary_verified=True,int32_accumulation=True,scale_applications=1,logical_row_plane_work=4*rows+4*tail,
            actual_integer_kernel_calls=len(kernel),explicit_padded_rows=sum(v['padding_rows'] for v in kernel),kernel_calls=kernel,
            low_bit_rule='arithmetic_shift_drop4_no_reclamp_no_offset',software_plane_dtype='int8',quantization_device='cpu',
            spatial_rows_retained=True,full_T_and_channels=True,cycle_claim=False)
    return calls,work

class Tests(unittest.TestCase):
    def test_full_and_crop_domains(self):
        calls,work=fixture((480,640));total=a.check_calls(calls,work,'Q4',(480,640))
        self.assertEqual(total['row_planes'],17731680)
        calls,work=fixture((288,384));total=a.check_calls(calls,work,'Q4',(288,384))
        self.assertEqual(total['row_planes'],sum(v['logical_row_plane_work'] for v in work.values()))
        a.check_calls(calls,{},'native',(288,384))
        with self.assertRaises(ValueError):a.check_calls(calls,work,'Q6',(288,384))

    def test_detailed_signed_drop4_rejections(self):
        calls,work=fixture((480,640));name=sorted(work)[0]
        mutations=[('low_scope',lambda w:w['kernel_calls'][0].update(scope='all')),
                   ('low_rows',lambda w:w['kernel_calls'][0].update(rows=w['kernel_calls'][0]['rows']+1)),
                   ('sign',lambda w:w['kernel_calls'][-1].update(coefficient=128)),
                   ('padding',lambda w:w['kernel_calls'][0].update(padding_rows=999)),
                   ('reordered_planes',lambda w:w['kernel_calls'].reverse()),
                   ('Q6_work',lambda w:w.update(logical_row_plane_work=6*w['rows']+2*w['tail_rows'])),
                   ('nonbinary',lambda w:w.update(binary_verified=False)),
                   ('wrong_T',lambda w:w.update(full_T_and_channels=False))]
        for label,mutate in mutations:
            broken=copy.deepcopy(work);mutate(broken[name])
            with self.assertRaises(ValueError,msg=label):a.check_calls(calls,broken,'Q4',(480,640))

    def test_dense_and_parity_rejections(self):
        calls,work=fixture((480,640));dense=copy.deepcopy(work)
        for item in dense.values():
            item.update(reference=True,logical_row_plane_work=None);chunk=min(65536,(8*1024*1024)//item['Cout'])//8*8;k=[]
            for scope,count in [('complete',item['complete_rows']),('tail',item['tail_rows'])]:
                for start in range(0,count,chunk):
                    n=min(chunk,count-start);k.append(dict(kind='dense_integer_reference',bit=None,rows=n,padding_rows=max(24,((n+7)//8)*8)-n,scope=scope,coefficient=None))
            item.update(kernel_calls=k,actual_integer_kernel_calls=len(k),explicit_padded_rows=sum(v['padding_rows'] for v in k))
        a.dense_reference_work(dense,work)
        broken=copy.deepcopy(dense);broken[sorted(dense)[0]]['kernel_calls'][0]['scope']='tail'
        with self.assertRaises(ValueError):a.dense_reference_work(broken,work)
        names={'PSN::synthetic'+str(i) for i in range(105)};keys=names|{'integer::'+n for n in a.TARGETS}|{'head'+str(i) for i in range(4)}
        pair={'pass':True,'checks':{n:dict(bitwise_equal=True,different_bytes=0,actual_sha256='0'*64,reference_sha256='0'*64) for n in keys}}
        a.parity(pair,True,names)
        del pair['checks']['head3']
        with self.assertRaises(ValueError):a.parity(pair,True,names)

    def test_recipe_source_and_actual_membership(self):
        smoke=load(HERE/'implementation/run_smoke_v2.py','audit_test_smoke');formal=load(HERE/'implementation/run_recovery_v2.py','audit_test_formal')
        self.assertEqual(a.EXPECTED_RECIPE,formal.recipe(smoke));self.assertEqual(formal.GRAPH,a.GRAPH)
        members=smoke.load_membership(a.R27/'independent_review/membership_v1.json');formal.validate_order(members)
        self.assertEqual(a.sha(HERE/'implementation/run_recovery_v2.py'),a.RUNNER_SHA)
        self.assertEqual(a.sha(HERE/'implementation/run_smoke_v2.py'),a.SMOKE_SHA)
        self.assertEqual(a.sha(HERE/'implementation/recovery_integer_core.py'),a.CORE_SHA)
        self.assertEqual(a.EXPECTED_RECIPE['crop'],[288,384]);self.assertEqual(512*2+64*2+3,1155)

if __name__=='__main__':
    out=Path(sys.argv[1]);assert not out.exists();start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    payload=dict(schema='flow_r28_actual_auditor_cpu_v1',status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                 failures=len(result.failures),errors=len(result.errors),wall_seconds=time.monotonic()-start,
                 sources={n:a.sha(HERE/n) for n in ('audit_actual_recovery_v1.py','test_actual_audit_cpu_v1.py')},
                 GPU_used=False,model_forwards=0,GT_values_read=False,real_recovery_results_read=False)
    out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload));sys.exit(0 if result.wasSuccessful() else 1)
