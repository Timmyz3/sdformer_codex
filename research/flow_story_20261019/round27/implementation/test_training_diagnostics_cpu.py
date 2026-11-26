#!/usr/bin/env python3
import importlib.util,json,os,sys,time,unittest
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import torch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('diagnostics',HERE/'training_diagnostics.py');d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
class Tests(unittest.TestCase):
 def test_displacement_and_aggregation(self):
  x=torch.tensor([3.,4.]);y=torch.tensor([6.,8.]);r=d.displacement(x,y)
  self.assertEqual(r['absolute_delta_L2'],5.);self.assertEqual(r['relative_delta_L2'],1.);self.assertEqual(r['changed_numeric_elements'],2)
  zero=d.displacement(torch.zeros(2),torch.ones(2));self.assertIsNone(zero['relative_delta_L2'])
  aggregate=d.aggregate_parameters([r,zero]);self.assertAlmostEqual(aggregate['relative_delta_L2'],(27/25)**.5)
  self.assertFalse(d.displacement(torch.tensor([-0.]),torch.tensor([0.]))['bytewise_unchanged'])
  with self.assertRaises(ValueError):d.displacement(torch.zeros(2),torch.tensor([float('nan'),0.]))
 def test_training_pass_statistics_and_clip(self):
  rows=[dict(target_id=str(i),gradients_finite=True,gradient_parameter_count=453,supervised_loss_px=float(i),
   teacher_mean_L1_px=2.,combined_loss=i+.2,preclip_grad_norm=100.1 if i==255 else float(i%100),wall_seconds=.5) for i in range(256)]
  summary=d.summarize_pass(rows)
  self.assertEqual(summary['distributions']['supervised_loss_px']['mean'],127.5)
  self.assertEqual(summary['distributions']['supervised_loss_px']['median'],127.5)
  self.assertEqual(summary['norm_gt_100_count'],1);self.assertEqual(summary['reconstructed_FP32_clip_coefficient_lt_one_count'],1)
  self.assertEqual(summary['wall_seconds_sum'],128.)
  rows[-1]['target_id']='0'
  with self.assertRaises(ValueError):d.summarize_pass(rows)
 def test_exclusive_group_priority(self):
  bn={'sttmultires_unet.decoders.0.norm.weight'}
  examples=[('sttmultires_unet.decoders.0.norm.weight',torch.zeros(8),'BN'),
   ('sttmultires_unet.decoders.0.sn.spiking_neuron.weight',torch.zeros(10,10),'PSN'),
   ('sttmultires_unet.decoders.0.deconv.0.weight',torch.zeros(8,8,3,3),'decoder'),
   (next(iter(d.FC)),torch.zeros(8,8),'FC'),(next(iter(d.CONV)),torch.zeros(8,8,3,3),'conv'),
   (d.PREFIX+'layers.0.swin_blocks.0.attn.positional_encoding',torch.zeros(8,8),'other')]
  for name,x,expected in examples:self.assertEqual(d.classify(name,x,bn),expected)
  self.assertEqual(len(d.FC),24);self.assertEqual(len(d.CONV),4);self.assertEqual(len(d.NONE),24)
if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1);start=time.monotonic()
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));p=Path(sys.argv[1]);assert not p.exists()
 p.write_text(json.dumps({'schema':'flow_r27_training_diagnostic_cpu_v1','status':'PASS' if r.wasSuccessful() else 'FAIL','tests':r.testsRun,
  'failures':len(r.failures),'errors':len(r.errors),'source_sha256':{n:d.sha(HERE/n) for n in ('training_diagnostics.py','test_training_diagnostics_cpu.py')},
  'GPU_used':False,'real_GT_read':False,'actual_final_weights_read':False,'model_forwards':0,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
 sys.exit(0 if r.wasSuccessful() else 1)
