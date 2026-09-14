from pathlib import Path
import json
import numpy as np

H = Path(__file__).resolve().parent
SRC = H.parent/'rr_modular'
BASE = H.parents[1]
FIX = BASE/'fusion_ten_trials_20260914/phase_borrow'
DATA = BASE/'r8_consumer_fusion_20260914/data'


def obligations(words, origin, fixture, raw_gold, identity_fp32, j_gold, i_gold):
    q1 = np.fromfile(fixture/'param4.bin', '<i4').astype(np.int64).reshape(864, 8)
    w = np.fromfile(fixture/'param5.bin', '<i4').astype(np.int64).reshape(12, 8, 8).transpose(0, 2, 1).reshape(96, 8)
    co = np.fromfile(fixture/'param7.bin', '<i4').astype(np.int64).reshape(12, 2, 8)
    klive = np.any(q1 != 0, axis=1)
    assert np.array_equal(klive, np.fromfile(fixture/'param6.bin', '<i4').reshape(864, 8)[:, 0])
    good = bool(np.all(np.maximum(q1, 0).sum(0) <= 511) and np.all(np.minimum(q1, 0).sum(0) >= -512))
    z = np.zeros((4, 10, 8), np.int64)
    active_columns = U1 = U2 = merged1 = merged2 = repairs = fields = source = 0
    for y in range(4):
        for x in range(4):
            source += 96 * (0 <= origin[0]+y < 240 and 0 <= origin[1]+x < 320)
    for k in range(864):
        if not klive[k]:
            continue
        channel, tap = divmod(k, 9)
        masks = []
        for p in range(4):
            y, x = p//2+tap//3, p%2+tap%3
            masks.append(int(words[channel, y, x]) if 0 <= origin[0]+y < 240 and 0 <= origin[1]+x < 320 else 0)
        active_columns += bool(any(masks))
        issue1 = ((masks[0] | masks[1] | masks[2]).bit_count()+masks[3].bit_count() if good else
                  (masks[0] | masks[1]).bit_count()+(masks[2] | masks[3]).bit_count())
        issue2 = (masks[0] | masks[1] | masks[2] | masks[3]).bit_count()
        U1 += issue1; U2 += issue2
        events = sum(v.bit_count() for v in masks)
        merged1 += events-issue1; merged2 += events-issue2
        for t in range(10):
            active = np.array([(v >> t) & 1 for v in masks], bool)
            if not active.any():
                continue
            low = (z[:, t]+128) % 256 - 128
            candidate = low[active]+q1[k]
            overflow = (candidate < -128) | (candidate > 127)
            repairs += bool(overflow.any()); fields += int(overflow.sum())
            z[active, t] += q1[k]
    zz = z.reshape(40, 8)
    raw = np.concatenate([zz @ w[8*o:8*o+8].T for o in range(12)])
    assert np.array_equal(raw, raw_gold)
    jf = np.asarray(identity_fp32, float)
    j = np.clip(np.rint(jf*(1 << 20)), -(1 << 31), (1 << 31)-1).astype(np.int64)
    assert np.array_equal(j, j_gold)
    a = np.repeat(co[:, 0], 40, axis=0)
    b = np.repeat(co[:, 1], 40, axis=0)
    expected = []
    for pv, av, bv, jv in zip(raw.flat, a.flat, b.flat, j.flat):
        q, r = divmod(int(pv)*int(av)+(int(bv)+int(jv))*(1 << 20), 1 << 26)
        q += r > 1 << 25 or (r == 1 << 25 and q % 2)
        expected.append(max(-(1 << 23), min((1 << 23)-1, q)))
    assert np.array_equal(np.array(expected).reshape(480, 8), i_gold)
    vlive = np.any(w.reshape(12, 8, 8) != 0, axis=1)
    M = int(((zz != 0)*vlive.sum(0)).sum())
    V = int((vlive & np.any(zz != 0, axis=0)).sum())
    result = {}
    for mode, U, merged in ((1, U1, merged1), (2, U2, merged2)):
        R = repairs if mode == 2 else 0
        N = 10 if mode == 2 else 0
        result[mode] = dict(core_first_issues=U, core_merged_updates=merged, core_repair_issues=R,
            core_repair_fields=fields if mode == 2 else 0, core_normalization_issues=N,
            core_source_words=source, core_weight_words=active_columns+V, core_second_weight_words=V,
            core_local_source_reads=int(klive.sum()), core_mac_issues=M, core_z_scalar_reads=M,
            core_z_vector_reads=40+U+R+N, core_z_writes=10+U+R+N,
            core_psum_reads=480, core_psum_writes=480, range_fallback_tiles=int(mode == 1 and not good),
            intrinsic=5427+int(klive.sum())+2*active_columns+3*U+M+2*R+2*N)
    return result


def ordered(a):
    return np.asarray(a).reshape(10, 12, 8, 2, 2).transpose(1, 3, 4, 0, 2).reshape(480, 8)


def main():
    predictions = {}
    cases = json.loads((FIX/'fixtures.json').read_text())
    for c in cases:
        p = FIX/'fixtures'/c['name']
        pred = obligations(np.fromfile(p/'source.bin', '<u2').reshape(96, 4, 4), c['input_origin'], p,
                           np.fromfile(p/'raw.bin', '<i4').reshape(480, 8),
                           np.fromfile(p/'identity_fp32.bin', '<f4').reshape(480, 8),
                           np.fromfile(p/'identity.bin', '<i4').reshape(480, 8),
                           np.fromfile(p/'gold.bin', '<i4').reshape(480, 8))
        predictions[c['name']] = pred
    arrays = {name: np.load(DATA/f'{name}.npy', mmap_mode='r') for name in
              ['first_source_words', 'identity_fp32_full', 'raw_p_full', 'identity_q20_full', 'i24_new_full']}
    rows = []
    files = ['results.json', 'results_small.json']
    if (SRC/'results_64.json').exists():
        files.append('results_64.json')
    for name in files:
        rows.extend(json.loads((SRC/name).read_text()))
    for r in rows:
        if 'fixture' not in r:
            for tile in range(r['first_tile'], r['first_tile']+r['tiles']):
                key = f'tile_{tile}'
                if key in predictions:
                    continue
                oy, ox = 2*(tile//160)-1, 2*(tile%160)-1
                words = np.zeros((96, 4, 4), np.int64)
                for y in range(4):
                    for x in range(4):
                        if 0 <= oy+y < 240 and 0 <= ox+x < 320:
                            words[:, y, x] = arrays['first_source_words'][:, oy+y, ox+x]
                predictions[key] = obligations(words, (oy, ox), FIX/'fixtures/real_0',
                    ordered(arrays['raw_p_full'][tile]), ordered(arrays['identity_fp32_full'][tile]),
                    ordered(arrays['identity_q20_full'][tile]), ordered(arrays['i24_new_full'][tile]))
        terms = ([predictions[r['fixture']][r['mode']]]*r['tiles'] if 'fixture' in r else
                 [predictions[f'tile_{tile}'][r['mode']] for tile in range(r['first_tile'], r['first_tile']+r['tiles'])])
        pred = {k: sum(t[k] for t in terms) for k in terms[0]}
        for k, v in pred.items():
            if k != 'intrinsic':
                assert r[k] == v, (r.get('fixture', r['first_tile']), r['mode'], k, r[k], v)
        assert r['core_cycles'] == pred['intrinsic']+r['core_source_stalls']+r['core_weight_stalls']+r['core_arbitration_stalls']+r['core_output_stalls']
        assert r['shared_source_grants'] == r['core_source_words']
        assert r['shared_weight_grants'] == r['core_weight_words']
        assert r['shared_z_grants'] == r['core_z_vector_reads']+r['core_z_scalar_reads']+r['core_z_writes']
        assert r['shared_psum_grants'] == r['core_psum_reads']+r['core_psum_writes']
        assert r['shared_alu_grants'] == r['proof_issues']+r['core_first_issues']+r['core_mac_issues']+r['core_repair_issues']+r['core_normalization_issues']
        assert r['conflict_cycles'] == r['core_arbitration_stalls']
        assert r['proof_issues'] == (0 if r['command'] else 864)
        assert r['static_words'] == (0 if r['command'] else 1848)
        assert r['total_cycles'] == r['window_cycles']+r['launch_cycles']+r['static_words']+r['parameter_stalls']+r['source_load_words']+r['origin_words']+r['source_load_stalls']+1
    summary = dict(command_count=len(rows), author_result_files=files, independent_fixture_count=len(cases),
        independent_native_tiles=len(predictions)-len(cases), independent_values_per_stage=len(predictions)*3840,
        per_command_obligations_and_grant_equations=True,
        repair_denied_cycles=sum(r['core_repair_arbitration_stalls'] for r in rows),
        normalization_denied_cycles=sum(r['core_normalization_arbitration_stalls'] for r in rows))
    (H/'predictions.json').write_text(json.dumps(predictions, indent=2)+'\n')
    (H/'math_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
