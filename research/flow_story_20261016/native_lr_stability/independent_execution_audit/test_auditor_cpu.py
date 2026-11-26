#!/usr/bin/env python3
"""Synthetic independent audit checks; no new run result or quality reads."""
import copy, importlib.util, json, unittest
from pathlib import Path
P=Path(__file__).parent
spec=importlib.util.spec_from_file_location('audit_native_lr',P/'audit_native_lr.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class AuditChecks(unittest.TestCase):
    def test_geometry_and_fixed_order(self):
        m=a.read(a.ROOT/'data_protocol/manifest_v1.json');a.memberships(m)
        for mode,n in [('smoke',9),('recover',228)]:
            schedule=a.expected_forwards('diversity',mode,'native',m)
            self.assertEqual(len(schedule),n)
            self.assertEqual(sum(a.expected_layer(layer,arm,dual)[0] for _,_,arm,dual in schedule for layer in a.GEOMETRY),n*558000)
        formal=a.expected_forwards('diversity','recover','native',m)
        self.assertEqual([r[1] for r in formal if r[0]=='training'],[r['target_id'] for r in m['train_order_diversity']])
        self.assertEqual([r[2] for r in formal if r[0]=='evaluation_endpoint'],['native','FF']*8)
    def test_actual_geometry_is_recomputed(self):
        ex={'all12_observed_fc_rows':{n:{'fc1_rows':a.expected_layer(n,'native')[0],'fc2_rows':a.expected_layer(n,'native')[0]} for n in a.GEOMETRY},'layers':{},'summed_fc1_rows':558000,'summed_fc2_rows':558000}
        self.assertEqual(a.check_execution(ex,'native',False,'training')['FC1_rows'],558000)
        damaged=copy.deepcopy(ex);damaged['all12_observed_fc_rows'][next(iter(a.GEOMETRY))]['fc1_rows']-=1
        with self.assertRaises(ValueError):a.check_execution(damaged,'native',False,'training')
        damaged=copy.deepcopy(ex);damaged['summed_fc1_rows']-=1
        with self.assertRaises(ValueError):a.check_execution(damaged,'native',False,'training')
    def test_fp64_drift_and_zero_norm(self):
        import torch
        names={'x.mlp.fc1.weight','x.mlp.bn1.norm_layer.weight','x.mlp.sn1.spiking_neuron.weight'};old=a.ALLOWED;a.ALLOWED=names
        initial={n:torch.tensor([0.]) for n in names};state={n:torch.tensor([1.]) for n in names}
        one={'initial_squared_L2':0.,'delta_squared_L2':1.,'max_abs_delta':1.,'tensors':1,'relative_L2':None}
        r={'categories':{k:dict(one) for k in ('FC','PSN','BN_affine')},'layers':{'x.mlp':dict(one,delta_squared_L2=3.,tensors=3)}}
        try:
            self.assertTrue(a.check_drift(state,initial,r))
            bad=copy.deepcopy(r);bad['categories']['FC']['relative_L2']=0.
            with self.assertRaises(ValueError):a.check_drift(state,initial,bad)
            bad=copy.deepcopy(r);bad['layers']['x.mlp']['delta_squared_L2']=2.
            with self.assertRaises(ValueError):a.check_drift(state,initial,bad)
        finally:a.ALLOWED=old
    def test_resource_nested_scopes(self):
        s={'max_allocated_bytes_so_far':10,'max_reserved_bytes_so_far':20,'forward_seconds':1.,'backward_optimizer_seconds':2.,'synchronized_train_step_seconds':3.1}
        d={'steps':[s,dict(s)],'max_allocated_bytes':10,'max_reserved_bytes':20,'training_seconds':6.3,'wall_seconds':7.,'initial_audit_peak_bytes':{'allocated':11,'reserved':20},'full_entry_peak_bytes':{'allocated':11,'reserved':20},'runtime':{'device_name':'synthetic','torch':'synthetic'}}
        self.assertEqual(a.check_resources(d)['training_step_count'],2)
        bad=copy.deepcopy(d);bad['full_entry_peak_bytes']['allocated']=10
        with self.assertRaises(ValueError):a.check_resources(bad)
        bad=copy.deepcopy(d);bad['steps'][1]['max_allocated_bytes_so_far']=9
        with self.assertRaises(ValueError):a.check_resources(bad)
        bad=copy.deepcopy(d);bad['steps'][0]['synchronized_train_step_seconds']=2.
        with self.assertRaises(ValueError):a.check_resources(bad)
    def test_frozen_source_chain(self):
        pins=a.source_pins();self.assertEqual(pins[str(a.ROOT_FREEZE)],a.ROOT_FREEZE_SHA)
        self.assertEqual(pins[str(a.OLD_AUDIT)],a.OLD_AUDIT_SHA)
    def test_no_runner_import_or_quality_selection(self):
        import ast
        tree=ast.parse((P/'audit_native_lr.py').read_text())
        imports=[x.module if isinstance(x,ast.ImportFrom) else alias.name for x in ast.walk(tree) if isinstance(x,(ast.Import,ast.ImportFrom)) for alias in (x.names if isinstance(x,ast.Import) else [None])]
        self.assertFalse(any('run_native_lr' in x or 'execute_recovery' in x for x in imports if x))
        self.assertFalse(any(isinstance(x,ast.Subscript) and isinstance(x.slice,ast.Constant) and x.slice.value in ('metrics','summaries','AEE','aee') for x in ast.walk(tree)))
if __name__=='__main__':unittest.main()
