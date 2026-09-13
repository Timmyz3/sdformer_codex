"""Complete layer finite psum service replay. All outputs remain required."""
import sys
sys.dont_write_bytecode=True
import json,subprocess,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
d=np.load(HERE.parent/'channel_major_numeric_plan.npz')
blob=b''.join(np.concatenate([d[f'{kind}{i}'].reshape(-1) for i in range(12)]).tobytes() for kind in ('p','r','o'))
path=HERE/'plan.bin';path.write_bytes(blob)
subprocess.run(['g++','-O3','-march=native','-std=c++17',str(HERE/'schedule.cpp'),'-o','/tmp/prosperity_psum_probe'],check=True)
t=time.monotonic();p=subprocess.run(['/tmp/prosperity_psum_probe',str(path)],capture_output=True,text=True)
(HERE/'run.log').write_text(p.stderr)
if p.returncode:raise RuntimeError(p.stderr)
rows=json.loads(p.stdout)
for row in rows:
 row['host_seconds_shared_run_not_cycles']=time.monotonic()-t
 baseline=next(v for v in rows if v['scalar_bytes']==row['scalar_bytes'] and v['policy']=='ordinary_unified_LRU')
 strong=next(v for v in rows if v['scalar_bytes']==row['scalar_bytes'] and v['policy']=='ordinary_resident_set')
 row['cycle_delta_vs_LRU_pct']=(row['cycles']/baseline['cycles']-1)*100
 row['cycle_delta_vs_resident_set_pct']=(row['cycles']/strong['cycles']-1)*100
 print(row['scalar_bytes'],row['policy'],row['cycles'],row['dram_r'],row['dram_w'],flush=True)
result=dict(scope='Complete sample1 M3000/K6912/N768; original M256/K16/N128 forest; no convolution source transformation.',
 counter_correction='Original 995328000-byte g_psum read/write counts are on-chip logical access counts, not DRAM traffic. Original m,n,k schedule is already output-stationary.',
 resource=dict(PEs=128,psum_total_bytes=98304,SRAM_read_ports=1,SRAM_write_ports=1,
   SRAM_vector_lanes=128,ALU_vector_latency=1,DDR_bits_per_cycle=1024,weight_buffer_bytes=32768,
   double_weight_buffers=True,double_TCAM=True,parent_last_use_release_all_controls=True,
   same_three_temporary_psum_vectors_all_controls=True),
 widths=dict(INT8='Original paper-style width capacity sanity only; no trained FP32 numeric claim.',
   INT64='Exact trained-FP32 dyadic integer reference from parent numeric_results.json. Iso-resource CPU service point, not original ASIC PPA.'),
 candidate='T10 coherent resident-set admission and final-K completion. Generic fixed resident-set plus dead-parent eviction is an ordinary output-stationary/cache control, not X.',
 limits=['Feasible deterministic one-row-at-a-time controller with double-buffer lookahead, not optimal hardware throughput.',
 'Logical128-lane SRAM access percycle, unit-latency ALU and128B DDR transactions are declared model parameters, not measured macros.',
 'TCAM preload and query, full source key fetch, every repeated N/K weight fill, SRAM and shared DDR resource conflicts are charged.',
 'No generic admission decisions use unseen future-K activity.48 parent slots are a fixed architectural reserve, not a tuned optimum.',
 'T10 completion is tracked for all1800 spatial/output-group cohorts; following dense PSN/PED computation is not modeled.',
 'Static theta preserves linear sum reuse and creates no extra numeric sparsity or cache entitlement.'],points=rows)
(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
