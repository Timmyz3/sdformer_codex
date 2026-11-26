"""Post-result geometric coverage diagnostic; no prediction or causal mapping."""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[2]/'flow_story_20261016/coverage_recovery_round13'
METRIC=OLD/'independent_quality_review/reduce_raw_quality.py'
REGION=OLD/'preparation/regions_v1.json'
SNAPSHOT=OLD/'next_action_headroom/h8_input_snapshot_v1'
H8=OLD/'next_action_headroom/implementation_h0/h8_manifest.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(METRIC)=='26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088'
assert sha(REGION)=='2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b'
assert sha(SNAPSHOT/'receipt.json')=='7f0cc5d74ce171aaa82cc602ec877e8fd58456e7481a9a158bc397cb8efa7b5d'
assert sha(H8)=='1083d2a3f9b940aee0ebf271cb30b062eeed7bc4bcbf51429d573172573569f5'
spec=importlib.util.spec_from_file_location('_r19_support_metric',METRIC)
metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
ids=[r['target_id'] for r in json.loads(H8.read_text())['members']]
cache,evidence=metric.load_snapshot(SNAPSHOT,SNAPSHOT/'receipt.json',ids)
region=json.loads(REGION.read_text())
yy,xx=np.indices((480,640));tail=(yy>=432)|(xx>=576)
assert int(tail.sum())==480*640-432*576
frames=[]
for tid in ids:
    s=cache[tid]
    masks=metric.masks_from_original(s['gt'],s['mask'],s['events'],region['boundary_threshold_pixels'],region['large_displacement_threshold_pixels'])
    counts={}
    for name,m in masks.items():
        n=int(m.sum());nt=int((m&tail).sum());nf=int((m&~tail).sum())
        assert nt+nf==n
        counts[name]={'total':n,'full_tile_coordinate_region':nf,'tail_coordinate_region':nt}
    frames.append({'target_id':tid,'regions':counts})
out={'schema':'flow_r19_output_coordinate_tail_support_v1','status':'POST_RESULT_GEOMETRY_ONLY',
     'mapping':'Output coordinate (y,x) -> nominal feature cell (floor(y/16),floor(x/16)); full tiles y<432,x<576.',
     'scope':'All8 H8 and all16 regions; output-coordinate coverage only, not receptive fields, intervention effects or causal attribution. Full-domain BN and downstream mixing remain. No prediction, model forward or new gate.',
     'source_sha256':{str(p):sha(p) for p in [METRIC,REGION,H8,SNAPSHOT/'receipt.json',Path(__file__)]},
     'input_evidence':evidence,'frames':frames}
p=HERE/'OUTPUT_TAIL_SUPPORT_V1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'output_sha256':sha(p),'L_counts':[{'target_id':f['target_id'],**f['regions']['L']} for f in frames]}))
