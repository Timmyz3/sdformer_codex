"""Independent raw-input math and selected fresh Verilator runs; old tree is read-only."""
from pathlib import Path
from collections import Counter
import json
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OLD = BASE / 'fusion_ten_trials_20260914'
RNG = np.random.default_rng(914062)

def readhex(path):
    a = np.array([int(v, 16) for v in path.read_text().split()], dtype=np.uint32)
    return a.view(np.int32).astype(np.int64)

def writehex(path, a):
    path.write_text(''.join(f'{int(v) & 0xffffffff:08x}\n' for v in np.asarray(a).flat))

def native_z(words, q1, origin):
    # Direct convolution-index accumulation, independent of original im2col reshape.
    z = np.zeros((4, 10, 8), np.int64)
    for p in range(4):
        for ky in range(3):
            for kx in range(3):
                y, x = p // 2 + ky, p % 2 + kx
                if 0 <= origin[0] + y < 240 and 0 <= origin[1] + x < 320:
                    event = ((words[:, y, x, None] >> np.arange(10)) & 1)
                    z[p] += event.T @ q1[:, :, ky, kx].T
    return z

def raw_output(z, q2):
    out = np.empty((480, 8), np.int64)
    for og in range(12):
        for p in range(4):
            for t in range(10):
                out[og * 40 + p * 10 + t] = q2[og * 8:og * 8 + 8] @ z[p, t]
    return out

def dictionary(q1):
    rows = q1.reshape(8, 864).T
    keys = [sum((int(v) & 7) << (3 * i) for i, v in enumerate(row)) for row in rows]
    freq = Counter(k for k in keys if k)
    selected = sorted((k for k in freq if freq[k] >= 2), key=lambda k: (-freq[k], k))[:32]
    cls = np.array([selected.index(k) + 1 if k in selected else 0 for k in keys])
    rep = np.zeros(32, np.int64)
    for i, key in enumerate(selected):
        rep[i] = keys.index(key)
    return cls, rep, len(selected)

def emit_decomp(name, words, q1, q2, origin):
    d = HERE / 'fixtures' / name
    d.mkdir(parents=True, exist_ok=True)
    z = native_z(words, q1, origin)
    writehex(d / 'source.hex', words)
    writehex(d / 'origin.hex', origin)
    writehex(d / 'q1.hex', q1.reshape(8, 864).T)
    writehex(d / 'q2.hex', q2.reshape(12, 8, 8).transpose(0, 2, 1))
    writehex(d / 'k_live.hex', np.any(q1.reshape(8, 864), axis=0))
    writehex(d / 'gold.hex', raw_output(z, q2))
    cls, rep, ng = dictionary(q1)
    writehex(d / 'class.hex', cls)
    writehex(d / 'representative.hex', rep)
    writehex(d / 'ngroups.hex', [ng])
    return d

def emit_temporal(name, words, q1, q2, origin):
    d = HERE / 'fixtures' / name
    d.mkdir(parents=True, exist_ok=True)
    raw = raw_output(native_z(words, q1, origin), q2)
    # Unit alpha coefficient 2**26 makes I24 = clip(raw); exact FP32 identities zero.
    ab = np.zeros((12, 2, 8), np.int64)
    ab[:, 0] = 1 << 26
    words.astype('<u2').tofile(d / 'source.bin')
    raw.astype('<i4').tofile(d / 'raw.bin')
    np.zeros((480, 8), '<i4').tofile(d / 'identity.bin')
    np.zeros((480, 8), '<f4').tofile(d / 'identity_fp32.bin')
    np.clip(raw, -(1 << 23), (1 << 23) - 1).astype('<i4').tofile(d / 'gold.bin')
    q1.reshape(8, 864).T.astype('<i4').tofile(d / 'param4.bin')
    q2.reshape(12, 8, 8).transpose(0, 2, 1).astype('<i4').tofile(d / 'param5.bin')
    np.repeat(np.any(q1.reshape(8, 864), axis=0)[:, None], 8, axis=1).astype('<i4').tofile(d / 'param6.bin')
    ab.astype('<i4').tofile(d / 'param7.bin')
    tile = ((origin[0] + 1) // 2) * 160 + (origin[1] + 1) // 2
    return d, tile

def check_original():
    stats = {'decomp_fixtures': 0, 'decomp_values': 0, 'temporal_fixtures': 0, 'temporal_values_each_checkpoint': 0}
    decomp_inputs = {}
    for c in json.loads((OLD / 'decompositions/q1_bitplanes/definition.json').read_text())['fixtures']:
        p = OLD / 'decompositions/q1_bitplanes/fixtures' / c['name']
        w = readhex(p / 'source.hex').reshape(96, 4, 4)
        q1 = readhex(p / 'q1.hex').reshape(864, 8).T.reshape(8, 96, 3, 3)
        q2 = readhex(p / 'q2.hex').reshape(12, 8, 8).transpose(0, 2, 1).reshape(96, 8)
        origin = readhex(p / 'origin.hex')
        z = native_z(w, q1, origin)
        assert np.array_equal(raw_output(z, q2).ravel(), readhex(p / 'gold.hex'))
        assert np.array_equal(np.any(q1.reshape(8, 864), axis=0), readhex(p / 'k_live.hex'))
        dp = OLD / 'decompositions/q1_dictionary/fixtures' / c['name']
        cls, rep, ng = dictionary(q1)
        assert np.array_equal(cls, readhex(dp / 'class.hex'))
        assert np.array_equal(rep, readhex(dp / 'representative.hex'))
        assert ng == readhex(dp / 'ngroups.hex')[0]
        stats['decomp_fixtures'] += 1
        stats['decomp_values'] += 3840
        decomp_inputs[c['name']] = (w, q1, q2, origin)
    temporal_inputs = {}
    for c in json.loads((OLD / 'temporal_direction/fixtures.json').read_text()):
        p = OLD / 'temporal_direction/fixtures' / c['name']
        w = np.fromfile(p / 'source.bin', '<u2').astype(np.int64).reshape(96, 4, 4)
        q1 = np.fromfile(p / 'param4.bin', '<i4').astype(np.int64).reshape(864, 8).T.reshape(8, 96, 3, 3)
        q2 = np.fromfile(p / 'param5.bin', '<i4').astype(np.int64).reshape(12, 8, 8).transpose(0, 2, 1).reshape(96, 8)
        raw = raw_output(native_z(w, q1, c['input_origin']), q2)
        assert np.array_equal(raw.ravel(), np.fromfile(p / 'raw.bin', '<i4'))
        fp = np.fromfile(p / 'identity_fp32.bin', '<f4').astype(np.float64).reshape(480, 8)
        j = np.clip(np.rint(fp * (1 << 20)), -(1 << 31), (1 << 31) - 1).astype(np.int64)
        assert np.array_equal(j.ravel(), np.fromfile(p / 'identity.bin', '<i4'))
        ab = np.fromfile(p / 'param7.bin', '<i4').astype(np.int64).reshape(12, 2, 8)
        a, b = (np.repeat(ab[:, k], 40, axis=0) for k in [0, 1])
        # Python integer divmod avoids NumPy overflow and separately states RNE.
        out = []
        for pv, av, jv, bv in zip(raw.flat, a.flat, j.flat, b.flat):
            quotient, remainder = divmod(int(pv) * int(av) + (int(jv) + int(bv)) * (1 << 20), 1 << 26)
            quotient += remainder > (1 << 25) or (remainder == (1 << 25) and quotient % 2 == 1)
            out.append(min((1 << 23) - 1, max(-(1 << 23), quotient)))
        assert np.array_equal(out, np.fromfile(p / 'gold.bin', '<i4'))
        stats['temporal_fixtures'] += 1
        stats['temporal_values_each_checkpoint'] += 3840
        temporal_inputs[c['name']] = (w, q1, q2, c['input_origin'])
    stats['cross_real_fixture_equal'] = {
        name: {label: bool(np.array_equal(a, b)) for label, a, b in zip(['source', 'q1', 'q2', 'origin'], item, temporal_inputs[name])}
        for name, item in decomp_inputs.items() if name.startswith('real_')
    }
    stats['cross_real_source_word_differences'] = {
        name: int(np.count_nonzero(item[0] != temporal_inputs[name][0]))
        for name, item in decomp_inputs.items() if name.startswith('real_')
    }
    rows = decomp_inputs['real_0'][1].reshape(8, 864).T
    stats['real_Q1_nonzero_R2_key_counts'] = [
        len({tuple(row) for row in rows[:, start:start + 2] if np.any(row)})
        for start in range(0, 8, 2)
    ]
    # DA signed18 subset sum + signed13 bit reconstruction, including -4 Q1 extension.
    for width in range(1, 14):
        low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
        for trial in range(40):
            z = RNG.integers(low, high + 1, size=8, dtype=np.int64)
            q = RNG.choice([-32768, -32767, -1, 0, 1, 32767], size=(8, 8))
            got = np.zeros(8, np.int64)
            for bit in range(width):
                for group in range(2):
                    v = q[:, group * 4:group * 4 + 4] @ ((z[group * 4:group * 4 + 4] >> bit) & 1)
                    assert np.all((v >= -131072) & (v <= 131071))
                    got += v * (-(1 << bit) if bit == width - 1 else 1 << bit)
                    assert np.all((got >= -(1 << 31)) & (got < (1 << 31)))
            assert np.array_equal(got, q @ z)
    stats['DA_random_extreme_vectors'] = 520
    return stats

def build(leaf, top, names):
    src = OLD / leaf
    dest = HERE / ('build_' + src.name)
    dest.mkdir(exist_ok=True)
    obj = dest / 'obj'
    cmd = ['verilator', '-Wall', '--cc', '--exe', '--top-module', top, '--Mdir', str(obj)]
    cmd += [str(src / name) for name in names] + ['-CFLAGS', '-O3']
    with (dest / 'build.log').open('w') as log:
        subprocess.run(cmd, cwd=dest, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['make', '-C', str(obj), '-f', f'V{top}.mk', '-j2'], stdout=log, stderr=subprocess.STDOUT, check=True)
    return obj / ('V' + top)

def run(binary, fixture, mode, stall, tile=None):
    cmd = [str(binary), str(fixture), str(mode), str(stall)]
    if tile is not None:
        cmd.append(str(tile))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError((cmd, proc.returncode, proc.stdout, proc.stderr))
    return [dict(json.loads(line), fixture=fixture.name, audit_leaf=binary.parent.parent.name) for line in proc.stdout.splitlines()]

def main():
    stats = check_original()
    full = np.full((96, 4, 4), 1023, np.int64)
    qmax = np.full((8, 96, 3, 3), -3, np.int64)
    qmin = np.full_like(qmax, -4)
    q2max = np.full((96, 8), -32768, np.int64)
    # Nonidentical spatial halves, K=15/16 block boundary, signed planes, and poison padding.
    words = RNG.integers(0, 1024, size=(96, 4, 4), dtype=np.int64)
    qrandom = RNG.integers(-3, 4, size=(8, 96, 3, 3), dtype=np.int64)
    q2random = RNG.choice([-32768, -1, 0, 1, 32767], size=(96, 8)).astype(np.int64)
    q2random[8:16] = 0
    q2random[:, 6:] = 0
    decomp_cases = [
        emit_decomp('random_padding_poison', words, qrandom, q2random, [237, 317]),
        emit_decomp('count_864_signed3', full, qmax, q2max, [79, 119]),
        emit_decomp('signed3_min_extension', full, qmin, q2max, [79, 119]),
    ]
    # Signed3 minimum plus zero before first anchor and mid-sequence zeros.
    dynamic = np.zeros_like(full)
    for t, channels in enumerate([0, 96, 1, 0, 48, 96, 0, 1, 96, 48]):
        dynamic[:channels] |= 1 << t
    temporal_case = emit_temporal('temporal_min_zero_gaps', dynamic, qmin, q2random, [79, 119])
    records = []
    for leaf in ['q1_bitplanes', 'q2_da', 'q1_dictionary']:
        binary = build('decompositions/' + leaf, 'decomp_core', ['decomp_core.sv', 'tb.cpp'])
        cases = decomp_cases + [OLD / 'decompositions' / leaf / 'fixtures/real_2']
        for p in cases:
            for mode in [14, 15]:
                records += run(binary, p, mode, 1)
        print('PASS fresh RTL', leaf, flush=True)
    binary = build('temporal_direction', 'consumer_stream', ['consumer_stream.sv', 'i24_consumer.sv', 'temporal_core.sv', 'tb.cpp'])
    metas = {c['name']: c for c in json.loads((OLD / 'temporal_direction/fixtures.json').read_text())}
    cases = [temporal_case] + [(OLD / 'temporal_direction/fixtures' / name, metas[name]['tile_id']) for name in ['real_2', 'direction_residual', 'range_guard']]
    for p, tile in cases:
        for mode in [0, 1, 2]:
            records += run(binary, p, mode, 1, tile)
    print('PASS fresh RTL temporal_direction', flush=True)
    stats.update(fresh_rtl_commands=len(records), fresh_raw_values=sum(r.get('outputs', 0) for r in records), fresh_temporal_checkpoint_values=sum(r.get('raw_outputs', 0) for r in records))
    (HERE / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
    (HERE / 'summary.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
