#!/usr/bin/env python3
"""Reproducible CPU service/work report. Never runs model inference."""
import argparse, csv, io, json, time, unittest
from pathlib import Path
import service_model as m
import test_service_model as tests

def write(path,value):
 with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
 args.output_dir.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();log=io.StringIO()
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
 receipt={'schema':'flow_current_bn_r16_hardware_cpu_checks_v1','status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'log':log.getvalue(),
  'source_sha256':{str(p):m.sha(p) for p in (Path(m.__file__),Path(tests.__file__),Path(__file__))},'neural_forward_calls':0,'GPU_calls':0}
 write(args.output_dir/'CPU_CHECKS.json',receipt)
 if not result.wasSuccessful():raise SystemExit(1)
 value=m.suite(m.Config());write(args.output_dir/'SERVICE_WORK_REPORT.json',value)
 _,small=m.layer('tiny_trace',(10,1,3,4,4),[[False]],'spill',m.Config(activation_sram_bytes=8192,reserved_statistics_and_metadata_bytes=4096),True)
 write(args.output_dir/'TINY_TRANSACTION_TRACE.json',{'config':m.asdict(small.cfg),'events':small.events,'actual_DMA_beats':small.validate(True)})
 with (args.output_dir/'SUMMARY.csv').open('x',newline='') as stream:
  fields=['arm_policy','nominal_FC_rows_each','executed_FC1_rows','executed_FC2_rows','producer_issue_slots_hypothetical','producer_blocking_schedule_ticks','traffic_bytes','optimistic_known_work_saved_fraction']
  writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
  for k,r in value['strategies'].items():
   writer.writerow(dict(arm_policy=k,**{f:r[f] for f in fields[1:6]},traffic_bytes=sum(r['producer_traffic_bytes'].values()),optimistic_known_work_saved_fraction=value['comparisons'][k]['optimistic_savings_fraction_of_known_work']))
 print(json.dumps({'status':'CPU_SCREEN_COMPLETE','tests':result.testsRun,'elapsed_seconds':time.monotonic()-started,'output':str(args.output_dir),'no_full_network_or_RTL_claim':True}))
if __name__=='__main__':main()
