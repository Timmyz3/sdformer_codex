"""Strictly reuse archived flags; diagnostic theta amplitudes are not checkpoint W."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARSER_DIR = ROOT.parent/'mechanism_rebuild_gh_20260906/scripts'
sys.path.insert(0, str(PARSER_DIR))
from screen_threshold_packets import sources, EXPECTED

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan = json.loads((ROOT/'plan.json').read_text())
    out = ROOT/'fixtures'
    out.mkdir(exist_ok=True)
    target = out/'fc2_sample0_diagnostic.txt'
    record = out/'manifest.json'
    assert not target.exists() and not record.exists(), 'No silent fixture overwrite'
    capture = sources()
    rows = []
    frames = []
    for stage in plan['stages']:
        key = f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.fc2'
        spec, flags = capture[key]
        C = int(spec['input_channels'])
        T = 10
        P = len(flags)//T
        positions = [i*(P-1)//15 for i in range(16)]
        channels = np.arange(C, dtype=np.int64)[:,None]
        lane = np.arange(8, dtype=np.int64)[None,:]
        weight = ((channels+3)*(lane+5)*13)%256-128
        weight[:,0] = -128
        weight[:,1] = 127
        phi = weight*(999883+17*channels)
        bits = flags.reshape(T,P,C)
        for i,p in enumerate(positions):
            signature = np.tensordot(1 << np.arange(T,dtype=np.int64),bits[:,p,:],axes=(0,0))
            expected = bits[:,p,:].astype(np.int64) @ phi
            # Bound every possible subset sum, not only the final sums.
            bound = np.abs(phi).sum(axis=0)
            assert np.all(bound < (1 << 47))
            frame_id = 1000+stage*100+i
            frames.append((frame_id,signature,phi))
            rows.append({'frame_id':frame_id,'module':key,'position':p,'P':P,'C':C,
                         'active_sources':int(np.count_nonzero(signature)),
                         'source_firings':int(bits[:,p,:].sum()),
                         'absolute_subset_sum_bound_per_lane':bound.tolist(),
                         'direct_integer_outputs':expected.tolist()})
    with target.open('x') as f:
        f.write(f'C2PROTO1 10 8 {len(frames)}\n')
        for fid,sig,phi in frames:
            f.write(f'{fid} {len(sig)}\n')
            for q,values in zip(sig,phi):
                f.write(str(int(q))+' '+' '.join(str(int(x)) for x in values)+'\n')
    receipt = {'status':'ARCHIVED_FLAGS_WITH_DIAGNOSTIC_NONUNIT_THETA_COEFFICIENTS',
               'capture_sha256':EXPECTED,'parser_sha256':sha(PARSER_DIR/'screen_threshold_packets.py'),
               'plan_sha256':sha(ROOT/'plan.json'),'generator_sha256':sha(Path(__file__)),
               'fixture_sha256':sha(target),'frames':rows,
               'frozen_checkpoint_weight_test':False,'new_capture':False}
    with record.open('x') as f:
        json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'frames':len(frames),'sources':sum(len(x[1]) for x in frames),
                      'fixture_sha256':receipt['fixture_sha256']}))

if __name__ == '__main__':
    main()
