"""Append the predeclared directional input; preserve all original labels exactly."""
import argparse,hashlib,json,sys,shutil
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'directional_branch'));import predictors as p
def sha(q):return hashlib.sha256(Path(q).read_bytes()).hexdigest()
def read(q):return json.loads(Path(q).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--predictors-sha256',required=True);a=ap.parse_args()
    assert not a.output.exists() and sha(p.__file__)==a.predictors_sha256
    assert p.N_FEATURES==160 and p.SCHEMA=='flow_r33_directional_small_predictors_v1'
    mp=HERE/'data/MEMBERSHIP_V1.json';assert sha(mp)=='d860612a3ae3e4620fa2a9a616df9f8e93774efadcf7dc564881e27b7e7da8cc'
    member={r['target_id']:r for r in read(mp)['records']};pf=read(a.pack/'PACK_FREEZE_V1.json')
    for n,h in pf['files'].items():assert Path(n).name==n and sha(a.pack/n)==h
    roles=['fit','tune'] if pf['phase']=='fit_tune' else ['confirm'];newpacks={};checked=[]
    for role in roles:
        with np.load(a.pack/(role+'.npz'),allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
        assert set(old['roles'])=={role};features=[]
        for i,tid in enumerate(old['ids']):
            r=member[str(tid)];assert r['role']==role
            f=next(x for x in r['source_files'] if x['category']=='input');rel=Path(f['relative_path']);assert not rel.is_absolute() and '..' not in rel.parts
            path=a.source/rel;assert sha(path)==f['sha256'] and path.stat().st_size==int(f['bytes'])
            x=p.voxel_features(np.load(path,allow_pickle=False));assert x.shape==(160,) and np.array_equal(x[:128],old['x'][i])
            features.append(x);checked.append(dict(target_id=str(tid),input_sha256=f['sha256'],first128_bitwise_equal=True))
        new=dict(old);new['x']=np.stack(features)
        for k in old:
            if k!='x':assert np.array_equal(old[k],new[k])
        newpacks[role]=new
    a.output.mkdir(parents=True)
    for role,data in newpacks.items():np.savez_compressed(a.output/(role+'.npz'),**data)
    shutil.copyfile(a.pack/'QUALITY_V1.json',a.output/'QUALITY_V1.json')
    receipt=dict(schema='flow_r33_directional_pack_v1',phase=pf['phase'],runner_sha256=sha(__file__),predictors_sha256=sha(p.__file__),membership_sha256=sha(mp),original_pack_freeze_sha256=sha(a.pack/'PACK_FREEZE_V1.json'),source_checks=checked,labels_unchanged=True,original_quality_json_sha256=sha(a.pack/'QUALITY_V1.json'),GPU_used=False,model_forwards=0)
    (a.output/'DIRECTIONAL_AUGMENT_RECEIPT_V1.json').write_text(json.dumps(receipt,indent=2)+'\n')
    freeze=dict(schema='flow_r33_pack_freeze_v1',phase=pf['phase'],files={f.name:sha(f) for f in a.output.iterdir()},model_forwards=0,GPU_used=False)
    (a.output/'PACK_FREEZE_V1.json').write_text(json.dumps(freeze,indent=2)+'\n');print('DIRECTIONAL_PACK_COMPLETE',pf['phase'])
if __name__=='__main__':main()
