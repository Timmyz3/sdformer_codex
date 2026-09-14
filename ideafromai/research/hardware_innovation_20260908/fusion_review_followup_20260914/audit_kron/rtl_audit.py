"""Own snapshots/TB only; invoke with Python 3.12. No source-tree mutation."""
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
from audit import H, SRC, raw_z, inspect_fixture, build


def large_s_fixture():
    p = H / 'fixtures' / 'large_signed_S'
    p.mkdir(parents=True, exist_ok=True)
    q1 = np.full((8, 96, 3, 3), -4, np.int64)
    words = np.full((96, 4, 4), 1023, np.int64)
    a = np.zeros((12, 4), np.int64)
    for o in range(12):
        a[o, o % 4] = -1 if o % 2 else 1
    b = np.repeat(np.array([-32, 31]*4, np.int64)[:, None], 2, axis=1)
    w = np.kron(a, b)
    mask = (1 << 1) | (1 << 10)
    for o in (1, 10):
        w[8*o:8*o+8] = 0
        w[8*o:8*o+8, 0] = [131071, -131072]*4
    z = raw_z(words, q1, (79, 119))
    raw = np.concatenate([z @ w[8*o:8*o+8].T for o in range(12)])
    assert np.max(np.abs(raw)) < (1 << 31)
    params = {
        4: q1.reshape(8, 864).T,
        5: w.reshape(12, 8, 8).transpose(0, 2, 1),
        6: np.repeat(np.any(q1.reshape(8, 864) != 0, axis=0)[:, None], 8, axis=1),
        7: np.stack([np.full((12, 8), 1 << 26), np.zeros((12, 8), np.int64)], axis=1),
        8: np.pad(a, ((0, 0), (0, 4))),
        9: b.T,
        10: np.array([mask]+[0]*7),
    }
    for kind, data in params.items():
        np.asarray(data, '<i4').tofile(p / f'param{kind}.bin')
    words.astype('<u2').tofile(p / 'source.bin')
    raw.astype('<i4').tofile(p / 'raw.bin')
    np.zeros((480, 8), '<i4').tofile(p / 'identity.bin')
    np.zeros((480, 8), '<f4').tofile(p / 'identity_fp32.bin')
    np.clip(raw, -(1 << 23), (1 << 23)-1).astype('<i4').tofile(p / 'gold.bin')
    prediction, bounds = inspect_fixture(p, (79, 119))
    (p/'description.json').write_text(json.dumps(dict(bounds, origin=[79, 119], tile=6460,
         S_values=[221184, -214272], prediction=prediction), indent=2)+'\n')
    return p


def run(folder, fixture, mode, stall, tile, expected=0):
    result = subprocess.run([str(folder/'obj/Vconsumer_stream'), str(fixture), str(mode), str(stall), str(tile)],
                            cwd=folder, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    name = f'{fixture.name}_mode{mode}_stall{stall}'
    (folder/f'{name}.log').write_text(result.stdout+result.stderr)
    assert result.returncode == expected, (name, result.returncode, result.stderr)
    rows = [dict(json.loads(line), fixture=fixture.name) for line in result.stdout.splitlines() if line.startswith('{')]
    if not expected:
        assert len(rows) == 2
        for row in rows:
            row['actual_mode'] = (row['command'] if mode == 2 else 1-row['command'] if mode == 3 else mode)
    return rows


def main():
    pre = H / 'pre_fix'
    if not (pre / 'obj/Vconsumer_stream').exists():
        build(pre)
    pre_rows = run(pre, SRC/'fixtures/real_0', 2, 0, 0, expected=20)
    post = H / 'post_fix'
    post.mkdir(exist_ok=True)
    for name in ('consumer_stream.sv', 'kron_core.sv', 'i24_consumer.sv'):
        shutil.copyfile(SRC/name, post/name)
    assert 'factor_resident' in (post/'consumer_stream.sv').read_text()
    shutil.copyfile(pre/'tb.cpp', post/'tb.cpp')
    build(post)
    records = []
    fixture_metadata = {f['name']: f for f in json.loads((SRC/'fixtures.json').read_text())}
    names = ['real_0', 'real_2', 'all_factor', 'all_exact', 'factor_negative_extreme', 'factor_cancellation', 'padding_poison', 'identity_ties_sat']
    for name in names:
        for mode in (0, 1):
            records += run(post, SRC/'fixtures'/name, mode, 1, fixture_metadata[name]['tile_id'])
    for name in ('real_0', 'all_factor', 'all_exact'):
        for mode in (2, 3):
            for stall in (0, 1):
                records += run(post, SRC/'fixtures'/name, mode, stall, fixture_metadata[name]['tile_id'])
    corner = large_s_fixture()
    for mode in (0, 1, 2, 3):
        for stall in (0, 1):
            records += run(post, corner, mode, stall, 6460)
    for row in records:
        p = corner if row['fixture'] == corner.name else SRC/'fixtures'/row['fixture']
        origin = (79, 119) if p == corner else fixture_metadata[row['fixture']]['input_origin']
        pred = inspect_fixture(p, origin)[0][row['actual_mode']]
        for key, value in pred.items():
            if key != 'base_cycles':
                assert row[key] == value, (row['fixture'], row['actual_mode'], key, row[key], value)
        assert row['core_cycles'] == pred['base_cycles']+row['core_source_stalls']+row['core_weight_stalls']+row['core_output_stalls']
    (H/'rtl_results.json').write_text(json.dumps(records, indent=2)+'\n')
    summary = dict(pre_fix_expected_failure='mode0 cold -> mode1 warm: raw mismatch, exit20',
                   pre_fix_first_command=pre_rows[0], post_fix_successful_commands=len(records),
                   raw_J_I24_values_each=len(records)*3840,
                   switching_commands=sum(r['mode'] >= 2 for r in records),
                   large_S_commands=sum(r['fixture'] == corner.name for r in records),
                   verified_per_command_counter_and_cycle_obligations=True)
    (H/'rtl_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
