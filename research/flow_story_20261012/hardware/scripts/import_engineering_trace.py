#!/usr/bin/env python3
"""Import observed geometry/FC rows from the one-input engineering GPU probe.
No GPU or model is executed here. Masks are synthetic policies, not learned gate
decisions. FP32 execution observations DO NOT validate int8/int32 hardware AEE.
"""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from cycle_sim import RES,digest,dump,simulate,load_mask
source=R.parent/'deployment_contract/whole_model_tile_probe.json'
e=json.loads(source.read_text());assert e['status']=='ENGINEERING_PROBE_COMPLETE'
records=[]
for case in e['cases']:
 layer=case['arms']['actual_skip']['layers'][0]
 T,B,H,W,C=layer['shape_TBHWC'];assert [T,B,C]==[10,1,96]
 assert layer['spans_complete_T'] and layer['tile_hw']==[9,9] and layer['mode']=='actual_skip'
 keep=[v for row in layer['keep'] for v in row]
 m={'schema':'flow-mlp-mask-v1','origin':'synthetic','batch_size':B,'T':T,'height':H,'width':W,
    'channels':C,'tile_size':9,'stage_index':0,'block_index':0,'keep':keep,'predictor_cycles_per_tile':0,
    'provenance':{'checkpoint_sha256':e['checkpoint_sha256'],'frame_id':e['target_id'],
     'BN_contract':e['BN_policy'],'source_trace':str(source),'source_trace_sha256':digest(source),
     'pattern':case['pattern'],'layer_name':layer['layer_name'],'input_sha256':e['input_sha256'],
     'note':'36-step engineering checkpoint, one TRAIN input; actual FP32 FC rows; synthetic mask policy; no AEE or quantized arithmetic evidence'}}
 path=R/'masks'/('engineering_'+case['pattern']+'.json');dump(path,m);load_mask(path)
 row={'pattern':case['pattern'],'mask_path':str(path.relative_to(R)),'observed_native_fc_rows':layer['observed'],
      'FP32_geometry_only_not_quantized_accuracy':True,'cycle_models':[]}
 for outstanding in (1,2):
  result=simulate(H,W,keep,r=dict(RES,dot_outstanding=outstanding),name='engineering_'+case['pattern'])
  assert result['kept_valid_points']==layer['kept_real_pixels']
  assert result['kept_valid_points']*T==layer['observed']['fc1_rows']==layer['observed']['fc2_rows']
  row['cycle_models'].append(result)
 records.append(row)
dump(R/'results/engineering_trace_cycle_results.json',{'status':'passed','source_trace_sha256':digest(source),
 'cycle_script_sha256':digest(R/'cycle_sim.py'),'import_script_sha256':digest(__file__),
 'scope':'first stage0 MLP only; imported FP32 geometry/row observations; synthetic masks and hypothetical int8/int32 cycle model; not whole-net or accuracy evidence',
 'records':records})
print(json.dumps([{'pattern':r['pattern'],'fc_rows':r['observed_native_fc_rows']['fc1_rows'],
                  'cycles':[v['clock_cycles'] for v in r['cycle_models']]} for r in records],indent=2))
