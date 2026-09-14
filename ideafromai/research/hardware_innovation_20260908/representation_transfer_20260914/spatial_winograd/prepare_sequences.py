from pathlib import Path
import ast
import json
import numpy as np

H = Path(__file__).resolve().parent
Q = H.parent / 'quality/q11'
f = np.load(H.parent / 'spatial_winograd_inputs/factors.npz')
g = np.load(Q / 'sequence_tiles.npz')
metadata = json.loads((Q / 'sequence_tiles.json').read_text())
q1 = f['q1'].astype(np.int64)
q2 = f['q2'].astype(np.int64)
W = np.einsum('orx,rcy->ocyx', q2, q1)
profiles = json.loads((H / 'profiles.json').read_text())
records = []
# Reuse the source-derived native/expanded-W oracle and fixture layout only.
# No old same-frame outputs enter this calculation.
definitions = [n for n in ast.parse((H / 'prepare.py').read_text()).body
               if isinstance(n, ast.FunctionDef)]
exec(compile(ast.Module(body=definitions, type_ignores=[]), 'prepare-definitions', 'exec'))

a = f['a_q40'].astype(np.int64)[None, :, None, None]
b = f['b_q20'].astype(np.int64)[None, :, None, None]
rows = []
for i, meta in enumerate(metadata):
    name = f"seq{i // 2:02d}_tile{meta['tile_id']}"
    words, origin = g['source_words'][i], g['output_origin_yx'][i]
    ev, z, p = native(words, origin)
    bits = g['identity_fp32_bits'][i]
    j = np.clip(np.rint(bits.view(np.float32).astype(np.float64) * 2**20),
                -2**31, 2**31-1).astype(np.int64)
    wide = p * a + (b + j) * 2**20
    rounded = wide >> 26
    remainder = wide - (rounded << 26)
    rounded += (remainder > 2**25) | ((remainder == 2**25) & ((rounded & 1) != 0))
    i24 = np.clip(rounded, -2**23, 2**23-1)
    for key, value in [('z_halo_int', z), ('p_int', p), ('J_q20', j),
                       ('wide_int64', wide), ('i24', i24)]:
        assert np.array_equal(value, g[key][i]), (name, key)
    emit(name, words, origin, p, z, bits, j, i24)
    layout = rawlayout(wide)
    hexfile(H / 'fixtures' / name / 'wide.hex',
            np.stack([layout & 0xffffffff, (layout >> 32) & 0xffffffff], axis=-1))
    rows.append(dict(index=i, name=name, **meta, output_origin_yx=origin.tolist(),
                     source_origin_yx=(origin - 1).tolist(), fixture=records[-1]))

(H / 'sequences.txt').write_text('\n'.join(records) + '\n')
(H / 'profiles.json').write_text(json.dumps(profiles, separators=(',', ':')) + '\n')
(H / 'sequence_fixture_metadata.json').write_text(json.dumps(rows, indent=2) + '\n')
summary = dict(passed=True, sequences=len(metadata)//2, tiles=len(metadata),
               raw_J20_wide_I24_each=len(metadata)*3840,
               native_Z_values=len(metadata)*1280,
               mismatches=0, source=str(Q / 'sequence_tiles.npz'),
               selection='first captured frame of each sequence; fixed tile128 and tile9664',
               oracle='source -> Q1/Q2 and independently expanded W -> raw; FP bits -> J20 -> wide -> RNE26/I24')
(H / 'sequence_admission.json').write_text(json.dumps(summary, separators=(',', ':')) + '\n')
print(json.dumps(summary))
