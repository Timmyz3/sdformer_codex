from pathlib import Path
import ast
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parent
LOCKED = ROOT.parent
NEW = LOCKED.parent
old_f = np.load(NEW / 'spatial_winograd_inputs/factors.npz')
old_profiles = json.loads((LOCKED / 'profiles.json').read_text())
definitions = [n for n in ast.parse((LOCKED / 'prepare.py').read_text()).body
               if isinstance(n, ast.FunctionDef)]

def rd(path):
    return np.array([int(v, 16) for v in path.read_text().split()], np.uint32)

def link(dst, src):
    if not dst.exists():
        dst.symlink_to(os.path.relpath(src, dst.parent))
    else:
        assert dst.is_symlink() and dst.resolve() == src.resolve()

for arm in ['moment', 'native_tap']:
    H = ROOT / arm
    H.mkdir(exist_ok=True)
    f = np.load(NEW / 'spatial_winograd_pruning' / arm / 'factors.npz')
    exported = np.load(NEW / 'spatial_winograd_pruning' / arm / 'gold_tiles.npz')
    for key in ['q1', 'a_q40', 'b_q20']:
        assert np.array_equal(f[key], old_f[key]), (arm, key)
    q1 = f['q1'].astype(np.int64)
    q2 = f['q2'].astype(np.int64)
    W = np.einsum('orx,rcy->ocyx', q2, q1)
    assert np.array_equal(W, f['expanded_int32'])
    U = f['winograd_q2'].astype(np.int64)
    assert np.array_equal(U, np.stack([2*q2[:,:,0], q2.sum(2),
                          q2[:,:,0]-q2[:,:,1]+q2[:,:,2], 2*q2[:,:,2]], 2))
    profiles = {}
    records = []
    env = dict(H=H, np=np, f=f, q1=q1, q2=q2, W=W, profiles=profiles, records=records)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'locked-oracle-definitions', 'exec'), env)
    native, profile, rawlayout, hexfile = [env[k] for k in ['native', 'profile', 'rawlayout', 'hexfile']]
    (H / 'parameters').mkdir(exist_ok=True)
    link(H / 'parameters/q1.hex', LOCKED / 'parameters/q1.hex')
    link(H / 'parameters/consumer.hex', LOCKED / 'parameters/consumer.hex')
    hexfile(H / 'parameters/q2.hex', q2.reshape(12,8,2,8,3).transpose(2,0,3,4,1).reshape(576,8))
    hexfile(H / 'parameters/wq2.hex', U.reshape(12,8,2,8,4).transpose(2,0,3,4,1).reshape(768,8))
    a = f['a_q40'].astype(np.int64)[None,:,None,None]
    b = f['b_q20'].astype(np.int64)[None,:,None,None]
    export_index = {int(t): i for i, t in enumerate(exported['tile_ids'])}
    exported_matches = 0
    changed_parent_p = changed_parent_i24 = 0
    for old_path in old_profiles:
        old = Path(old_path)
        d = H / 'fixtures' / old.name
        d.mkdir(parents=True, exist_ok=True)
        words = rd(old / 'source.hex').reshape(96,4,4).astype(np.uint16)
        origin = rd(old / 'origin.hex').view(np.int32).astype(np.int64) + 1
        ev, z, p = native(words, origin)
        dv = np.stack([z[:,:,:,0]-z[:,:,:,2], z[:,:,:,1]+z[:,:,:,2],
                       z[:,:,:,2]-z[:,:,:,1], z[:,:,:,1]-z[:,:,:,3]], 3)
        assert abs(dv).max() < 2**15
        parts = []
        for stripe in range(2):
            M = np.einsum('tryi,ori->toyi', dv[:,stripe*8:stripe*8+8], U[:,stripe*8:stripe*8+8])
            sums = np.stack([M[:,:,:,0]+M[:,:,:,1]+M[:,:,:,2],
                             M[:,:,:,1]-M[:,:,:,2]-M[:,:,:,3]], 3)
            assert not np.any(sums & 1)
            parts.append(sums // 2)
        assert np.array_equal(p, parts[0]+parts[1])
        bits = rd(old / 'identity.hex').reshape(480,8)
        j_layout = np.clip(np.rint(bits.view(np.float32).astype(np.float64)*2**20),
                           -2**31,2**31-1).astype(np.int64)
        assert np.array_equal(j_layout, rd(old / 'j.hex').view(np.int32).reshape(480,8))
        a_layout = np.repeat(f['a_q40'].reshape(12,1,8),40,1).reshape(480,8).astype(np.int64)
        b_layout = np.repeat(f['b_q20'].reshape(12,1,8),40,1).reshape(480,8).astype(np.int64)
        pl = rawlayout(p)
        wide = pl*a_layout + (b_layout+j_layout)*2**20
        out = wide >> 26
        rem = wide-(out << 26)
        out += (rem > 2**25) | ((rem == 2**25) & ((out & 1) != 0))
        i24 = np.clip(out, -2**23, 2**23-1)
        if old.name.startswith('tile_'):
            ix = export_index[int(old.name[5:])]
            for key, value in [('p_int',pl),('wide_int64',wide),('i24',i24)]:
                assert np.array_equal(value,rawlayout(exported[key][ix])),(arm,old.name,key)
            assert np.array_equal(z,exported['z_halo_int'][ix])
            exported_matches += 1
        for name in ['source','origin','identity','j','z','d']:
            link(d / f'{name}.hex', old / f'{name}.hex')
        hexfile(d / 'gold.hex',pl)
        hexfile(d / 'wide.hex',np.stack([wide&0xffffffff,(wide>>32)&0xffffffff],-1))
        hexfile(d / 'i24.hex',i24)
        changed_parent_p += int(np.count_nonzero(pl != rd(old/'gold.hex').view(np.int32).reshape(480,8)))
        changed_parent_i24 += int(np.count_nonzero(i24 != rd(old/'i24.hex').view(np.int32).reshape(480,8)))
        e = profile(ev,z,origin)
        wm = wv = 0
        for ss in range(2):
            dd = dv[:,ss*8:ss*8+8]
            rl = np.any(dd != 0,axis=(0,2,3))
            for og in range(12):
                live = np.any(U[og*8:og*8+8,ss*8:ss*8+8] != 0,axis=0) & rl[:,None]
                wv += int(live.sum())
                wm += int(((dd != 0) & live[None,:,None,:]).sum())
        e.update(winograd_mac=wm, winograd_weight=wv)
        profiles[str(d)] = e
    for stage in ['small','held','disjoint','sequences']:
        names = (LOCKED / f'{stage}.txt').read_text().splitlines()
        (H / f'{stage}.txt').write_text('\n'.join(str(H/'fixtures'/Path(n).name) for n in names)+'\n')
    (H/'profiles.json').write_text(json.dumps(profiles,separators=(',',':'))+'\n')
    report = dict(passed=True, arm=arm, fixtures=len(profiles), export_gold_matches=exported_matches,
                  independently_recomputed_raw_J20_wide_I24_each=len(profiles)*3840,
                  per_stripe_exact_Winograd=True, changed_parent_raw=changed_parent_p,
                  changed_parent_I24=changed_parent_i24,
                  new_sequence_outputs='source/identity reused, all raw/wide/I24 recomputed with this arm')
    (H/'admission.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    print(json.dumps(report),flush=True)
