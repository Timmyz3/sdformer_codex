"""Reconstruct class counters and scheduling obligations from raw inputs, independently of RTL."""
from pathlib import Path
import json
import numpy as np
from prepare import readhex

H = Path(__file__).resolve().parent
cases = json.loads((H / 'fixtures.json').read_text())
rows = json.loads((H / 'results.json').read_text())
checks = 0
predictions = {}
max_counter = 0
direct_fallback_pairs = 0
def eq(a, b):
    global checks
    checks += 1
    assert a == b, (a, b)

for c in cases:
    p = H / 'fixtures' / c['name']
    source = readhex(p / 'source.hex').reshape(96, 4, 4)
    origin = readhex(p / 'origin.hex')
    q = readhex(p / 'q1.hex').reshape(864, 8)
    v = readhex(p / 'q2.hex').reshape(12, 8, 8).transpose(0, 2, 1).reshape(96, 8)
    classes = (readhex(p / 'class.hex')[:, None] >> (np.arange(4) * 6)) & 63
    reps = readhex(p / 'representative.hex').reshape(32, 8)
    ngroups = int(readhex(p / 'ngroups.hex')[0])
    eq(bool(np.all((classes <= 32) | (classes == 63))), True)
    eq(bool(np.all((q >= -3) & (q <= 3))), True)
    livek = np.any(q != 0, axis=1)
    eq(bool(np.array_equal(livek, readhex(p / 'k_live.hex'))), True)
    events = np.zeros((864, 40), np.int64)
    for k in range(864):
        channel, tap = divmod(k, 9)
        for pos in range(4):
            y, x = pos // 2 + tap // 3, pos % 2 + tap % 3
            if 0 <= origin[0] + y < 240 and 0 <= origin[1] + x < 320:
                events[k, pos*10:pos*10+10] = (source[channel, y, x] >> np.arange(10)) & 1
    z = events.T @ q
    raw = np.concatenate([z @ v[og*8:og*8+8].T for og in range(12)])
    eq(bool(np.array_equal(raw.ravel(), readhex(p / 'gold.hex'))), True)
    counts = np.zeros((4, 32, 40), np.int64)
    qdirect = np.zeros_like(q)
    for group in range(4):
        for k in range(864):
            code = int(classes[k, group])
            pair = q[k, 2*group:2*group+2]
            if code == 0:
                eq(bool(np.all(pair == 0)), True)
            elif code == 63:
                qdirect[k, 2*group:2*group+2] = pair
            else:
                eq(bool(np.array_equal(pair, reps[code-1, 2*group:2*group+2])), True)
                eq(code <= ngroups, True)
                counts[group, code-1] += events[k]
    reconstructed = events.T @ qdirect
    for group in range(4):
        reconstructed[:, 2*group:2*group+2] += counts[group].T @ reps[:, 2*group:2*group+2]
    eq(bool(np.array_equal(z, reconstructed)), True)
    eq(bool(np.all(counts <= 864)), True)
    max_counter = max(max_counter, int(counts.max()))
    direct_fallback_pairs += int(np.count_nonzero(classes == 63))
    lo = events[:, np.r_[0:10, 20:30]] != 0
    hi = events[:, np.r_[10:20, 30:40]] != 0
    support = lo | hi
    active = np.any(support, axis=1) & livek
    U = int(support[livek].sum())
    A = int(active.sum())
    K = int(livek.sum())
    vlive = np.any(v.reshape(12, 8, 8) != 0, axis=1)
    rankcost = vlive.sum(axis=0)
    M = int(((z != 0) * rankcost).sum())
    V = int((vlive & np.any(z != 0, axis=0)).sum())
    valid = sum(0 <= origin[0] + y < 240 and 0 <= origin[1] + x < 320 for y in range(4) for x in range(4))
    common = dict(outputs=3840, source_words=valid*96, local_source_reads=K,
                  second_weight_words=V, z_scalar_reads=M, mac_issues=M,
                  psum_reads=480, psum_writes=480)
    predictions[c['name'], 14] = dict(common, weight_words=A+V, first_issues=U,
        dual_updates=int((lo[livek] & hi[livek]).sum()), z_vector_reads=U+40, z_writes=U+20,
        metadata_reads=0, count_checks=0, count_bank_reads=0, count_bank_writes=0,
        aux_reads=0, aux_writes=0, aux_issues=0, aux_weight_words=0, aux_events=0,
        base_cycles=5437+K+2*A+3*U+M)
    grouped = (classes > 0) & (classes <= 32)
    needs_count = np.any(grouped, axis=1) & active
    needs_direct = np.any(classes == 63, axis=1) & active
    blocks = np.any(support.reshape(864, 10, 2), axis=2).sum(axis=1)
    updates = int(blocks[needs_count].sum())
    group_bank_updates = int((blocks[needs_count] * grouped[needs_count].sum(axis=1) * 2).sum())
    slots = int(np.any(counts != 0, axis=(0, 2)).sum())
    retires = int(np.any(counts != 0, axis=0).sum())
    clear = ngroups * 10
    direct_A = int(needs_direct.sum())
    direct_U = int(support[needs_direct].sum())
    count_checks = int(needs_count.sum()) * 10
    extra = A + clear + count_checks + 2*updates
    if ngroups:
        extra += ngroups + 1 + 21*slots + 3*retires
    predictions[c['name'], 15] = dict(common, weight_words=direct_A+slots+V,
        first_issues=direct_U+retires, dual_updates=int((lo[needs_direct] & hi[needs_direct]).sum()),
        z_vector_reads=direct_U+retires+40, z_writes=direct_U+retires+20,
        metadata_reads=A, count_checks=count_checks,
        count_bank_reads=group_bank_updates+80*slots,
        count_bank_writes=group_bank_updates+8*clear,
        aux_reads=updates+10*slots, aux_writes=updates+clear, aux_issues=updates,
        aux_weight_words=slots, aux_events=retires,
        base_cycles=5437+K+2*direct_A+3*direct_U+M+extra)
for r in rows:
    expected = predictions[r['fixture'], r['mode']]
    for key, value in expected.items():
        if key != 'base_cycles':
            eq(r[key], value)
    eq(r['cycles'], expected['base_cycles'] + r['source_stalls'] + r['weight_stalls'] + r['output_stalls'])
    eq(sum(r['state_cycles']), r['cycles'])
    eq(r['configuration_cycles'], 0 if r['command'] else 3361 if r['mode'] == 14 else 4258)
old = json.loads((H.parents[1] / 'fusion_ten_trials_20260914/decompositions/q1_bitplanes/results.json').read_text())
lookup = {(r['fixture'], r['mode'], r['stall'], r['command']): r for r in old}
matched = 0
for r in rows:
    key = r['fixture'], r['mode'], r['stall'], r['command']
    if r['mode'] == 14 and key in lookup:
        for field, value in lookup[key].items():
            eq(r[field], value)
        matched += 1
eq(max_counter, 864)
eq(direct_fallback_pairs > 0, True)
result = dict(complete=True, checks=checks, native_gold_values=len(cases)*3840,
              rtl_commands=len(rows), rtl_values=sum(r['outputs'] for r in rows),
              original_control_records_equal=matched, maximum_counter=max_counter,
              configured_direct_fallback_pairs=direct_fallback_pairs,
              method='Independent raw-source class counts, factor reconstruction, and FSM obligation counts; actual RTL outputs checked by C++ TB')
(H / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
