"""Read-only checkpoint comparison; copy the explicitly handed-off /tmp assets.

Run on sd5ai with Python 3.12. Never writes a checkpoint or traverses other jobs.
No new hash/seal ledger is created.
"""
from pathlib import Path
import json
import shutil

BASE=Path('/root/private_data/work/hardware_innovation_20260908')
REPO=Path('/root/private_data/work/sdformer_codex/SDformer')
OUT=BASE/'handoff_review_20260919/remote_assets'
OUT.mkdir(parents=True,exist_ok=True)
paths=[Path('/tmp')/x for x in ['dead_neuron_probe.py','verify_deadskip.py',
                              'binary_deadmask.json','motion_parity.py']]
paths+=sorted(Path('/tmp').glob('deadprobe_*.json'))
copied=[]
missing=[]
for p in paths:
    if not p.is_file():
        missing.append(str(p));continue
    shutil.copy2(p,OUT/p.name)
    copied.append(dict(source=str(p),destination=str(OUT/p.name),bytes=p.stat().st_size))

stem=REPO/'neuron_experiments/H9_bipolar_self_attention/results'
checkpoints={
    'restored_live':stem/'dsec_c12_alpha0125_ep29_resume5_20260830/checkpoint_epoch34.pth',
    'warmstart_backup':stem/'t49_ternary_smoke/warmstart/checkpoint_epoch34.pth',
    'prior_incoming_copy':REPO/'hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs/checkpoint_epoch34.pth',
}
def equal_file(a,b):
    if a.stat().st_size!=b.stat().st_size:return False
    with a.open('rb') as f,b.open('rb') as g:
        while True:
            x,y=f.read(4*1024*1024),g.read(4*1024*1024)
            if x!=y:return False
            if not x:return True

metadata={k:dict(path=str(p),resolved=str(p.resolve()),size=p.stat().st_size,
                inode=p.stat().st_ino,device=p.stat().st_dev,mtime=p.stat().st_mtime)
          for k,p in checkpoints.items()}
same={k:equal_file(checkpoints['restored_live'],p) for k,p in checkpoints.items()
      if k!='restored_live'}
result=dict(copied=copied,missing=missing,checkpoint_files=metadata,
            whole_file_byte_equal_to_live=same,
            distinct_files=len({(v['device'],v['inode']) for v in metadata.values()})==3,
            checkpoint_mutated=False,
            scope='All bytes compared, including temporal tensors; no new AEE or claim about changed overlay semantics.')
(OUT/'PRESERVATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(copied=len(copied),missing=missing,
                     whole_file_byte_equal_to_live=same,distinct_files=result['distinct_files'])))
