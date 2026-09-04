#!/usr/bin/env python3
from __future__ import annotations
import ast, copy, importlib.util, inspect, sys, unittest
from pathlib import Path

HW=Path(__file__).resolve().parents[2]
SRC=HW/'system_simulator/scripts/run_m1086_c1_zero_work_exact_1rw_source.py'
spec=importlib.util.spec_from_file_location('m1086_test_source',SRC)
if spec is None or spec.loader is None:raise RuntimeError('cannot load M1086')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)

class M1086Tests(unittest.TestCase):
 def test_zero_work_exact_empty_semantics(self):
  state={(0,1):9};before=dict(state);p=M.M1056.TaskPlan(207,146,0,15)
  r=M.schedule_task(p,500,state)
  self.assertEqual(state,before);self.assertEqual(r.events,[]);self.assertEqual(r.grants,{})
  self.assertEqual((r.queue_peak,r.nominal_excess_accesses,r.delayed_accesses,r.maximum_read_write_lifetime),(0,0,0,0))
  self.assertTrue(r.raw_dependencies_pass);self.assertEqual(r.nominal_work_end,500);self.assertEqual(r.effective_work_end,500)
 def test_positive_path_byte_behavior_equivalent(self):
  for work in (8,15,16,31,224,280,4096):
   a={};b={};p=M.M1056.TaskPlan(1,7,work,3)
   self.assertEqual(M.schedule_task(p,101,a),M.M1056.schedule_task(p,101,b));self.assertEqual(a,b)
 def test_production_work_domain_exact(self):
  for value in (0,15,16,224):self.assertEqual(M.validate_production_work(value),value)
  for value in (True,False,-1,*range(1,15),1.0,'15'):
   with self.assertRaises(RuntimeError):M.validate_production_work(value)
 def test_dependency_empty_bool_negative_rejected(self):
  valid=M.M1056.PortEvent('r',0,0,0,0,0,'READ',0)
  M.validate_dependencies([valid])
  for dep in (M.M1056.Dependency('',0),M.M1056.Dependency('r',True),M.M1056.Dependency('r',-1)):
   event=M.M1056.PortEvent('w',0,1,0,0,0,'WRITE',0,(dep,))
   with self.assertRaises(RuntimeError):M.validate_dependencies([event])
 def test_stream_zero_then_positive_deterministic(self):
  s=M.DesignStream(last_write={(0,16):1000});z=s.consume_internal(M.M1056.TaskPlan(207,146,0,15));state=dict(s.last_write)
  n=s.consume_internal(M.M1056.TaskPlan(208,158,224,16))
  self.assertEqual(state,{(0,16):1000});self.assertEqual(z.effective_work_end,z.work_start)
  self.assertEqual(n.work_start,max(z.effective_work_end,z.work_start+158)+2)
  self.assertGreaterEqual(n.grants['t208:b0:R'].cycle,1001)
 def test_real_task207_next_regression(self):
  value=M.real_task207_next_regression();self.assertEqual(value['status'],'PASS_M1086_REAL_TASK207_NEXT_RAW_REGRESSION')
  self.assertEqual(value['task207_coordinate'],[0,0,0,207]);self.assertFalse(value['production_iterator_called'])
 def test_preflight_zero_argument_and_no_cycle_scheduler(self):
  self.assertEqual(len(inspect.signature(M.canonical_work_domain_preflight).parameters),0)
  source=inspect.getsource(M.canonical_work_domain_preflight)
  self.assertNotIn('schedule_task(',source);self.assertNotIn('DesignStream(',source)
  self.assertNotIn('cycles_after_commit',source)
 def test_full_iterator_zero_argument_repaired_stream(self):
  self.assertTrue(inspect.isgeneratorfunction(M.iter_canonical_full_replay_results))
  self.assertEqual(len(inspect.signature(M.iter_canonical_full_replay_results).parameters),0)
  source=inspect.getsource(M.iter_canonical_full_replay_results)
  self.assertIn('DesignStream()',source);self.assertIn('validate_production_work',source)
 def test_frozen_raw_and_cascade_anchors(self):
  o=M.M1056.small_oracle();self.assertTrue(o['same_address_raw_enforced'])
  self.assertEqual(o['cascade']['nominal_cycles'],20);self.assertEqual(o['cascade']['arbitrated_cycles'],22)
 def test_capacity_and_frozen_authorities(self):
  self.assertEqual(M.M1064.derive_physical_capacity()['derived_total_bytes'],214912)
  self.assertEqual(M.validate_frozen_authorities()['m1085_outer'],M.M1085_OUTER)
 def test_row_provenance_forgery_still_rejected(self):
  source=inspect.getsource(M.iter_canonical_full_replay_results)
  self.assertIn('ProvenanceCoverage()',source);self.assertIn("proof['full_coverage_pass']",source)
  self.assertEqual(M.M1072.EXPECTED_CANDIDATE_PARENT['work_cycles'],409734336)
 def test_no_frozen_source_mutation_or_full_execution(self):
  self.assertEqual(M.sha256(M.M1072_PATH),M.M1072_SHA)
  oracle=M.source_small_oracle();self.assertFalse(oracle['full_replay_executed'])
  self.assertEqual(oracle['zero_events'],0)

if __name__=='__main__':unittest.main()
