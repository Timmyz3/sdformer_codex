"""Native-value/obligation reconstruction plus shared-resource receipt audit."""
from pathlib import Path
import json
import numpy as np

H = Path(__file__).resolve().parent
B = H.parents[1]
P = B / "fusion_ten_trials_20260914/phase_borrow"
D = B / "r8_consumer_fusion_20260914/data"
predictions = {}

def predict(words, q1, q2, source_words):
    spike = ((words[None] >> np.arange(10)[:, None, None, None]) & 1).astype(np.int64)
    patch = np.stack([spike[:, :, p // 2:p // 2 + 3, p % 2:p % 2 + 3].reshape(10, 864) for p in range(4)])
    kl = np.any(q1 != 0, axis=0)
    a = patch[:, :, kl].astype(bool)
    u0 = int((a[0] | a[1]).sum() + (a[2] | a[3]).sum())
    range_ok = bool(np.all(np.maximum(q1, 0).sum(1) <= 511) and np.all(np.minimum(q1, 0).sum(1) >= -512))
    u1 = int((a[0] | a[1] | a[2]).sum() + a[3].sum()) if range_ok else u0
    u2 = int(a.any(0).sum())
    exact = patch @ q1.T
    low = np.zeros((4, 10, 8), dtype=np.int64)
    high = np.zeros_like(low)
    prefix = np.zeros_like(low)
    repairs = fields = 0
    for k in range(864):
        if not kl[k]:
            continue
        delta = patch[:, :, k, None] * q1[:, k][None, None, :]
        prefix += delta
        unwrapped = low + delta
        overflow = (unwrapped < -128) | (unwrapped > 127)
        repairs += int(overflow.any(axis=(0, 2)).sum())
        fields += int(overflow.sum())
        high += (unwrapped > 127).astype(np.int64) - (unwrapped < -128).astype(np.int64)
        low = ((unwrapped + 128) & 255) - 128
        assert np.array_equal(prefix, low + 256 * high)
    assert np.array_equal(exact, prefix)
    assert np.all(high >= -16) and np.all(high < 16)
    canonical = (((high - (low < 0)) & 31) << 8) | (low & 255)
    canonical = np.where(canonical >= 4096, canonical - 8192, canonical)
    assert np.array_equal(canonical, exact)
    zl = exact != 0
    vl = np.any(q2.reshape(12, 8, 8) != 0, axis=1)
    return dict(U1=u1, U2=u2, repairs=repairs, fields=fields, fallback=int(not range_ok),
                A=int(a.sum()), K=int(kl.sum()), Q=int(a.any((0, 1)).sum()),
                V=int((vl & zl.any((0, 1))).sum()), M=int((zl.sum((0, 1)) * vl.sum(0)).sum()),
                source=source_words), exact @ q2.T

fixture_rows = json.loads((H / "results.json").read_text())
for name in sorted(set(r["fixture"] for r in fixture_rows)):
    f = P / "fixtures" / name
    meta = json.loads((f / "meta.json").read_text())
    oy, ox = meta["input_origin"]
    word = np.fromfile(f / "source.bin", "<u2").reshape(96, 4, 4).copy()
    valid = np.array([[0 <= oy + y < 240 and 0 <= ox + x < 320 for x in range(4)] for y in range(4)])
    word *= valid
    q1 = np.fromfile(f / "param4.bin", "<i4").astype(np.int64).reshape(864, 8).T
    q2 = np.fromfile(f / "param5.bin", "<i4").astype(np.int64).reshape(12, 8, 8).transpose(0, 2, 1).reshape(96, 8)
    pred, raw = predict(word, q1, q2, 96 * int(valid.sum()))
    gold = np.fromfile(f / "raw.bin", "<i4").reshape(12, 4, 10, 8).transpose(1, 2, 0, 3).reshape(4, 10, 96)
    assert np.array_equal(raw, gold)
    predictions[name] = pred

stream_rows = json.loads((H / "results_small.json").read_text())
if (H / "results_64.json").exists():
    stream_rows += json.loads((H / "results_64.json").read_text())
source = np.load(D / "first_source_words.npy", mmap_mode="r")
gold_raw = np.load(D / "raw_p_full.npy", mmap_mode="r")
coeff = np.load(D / "consumer_coefficients.npz")
q1, q2 = coeff["q1"].astype(np.int64), coeff["q2"].astype(np.int64)
for first, count in sorted(set((r["first_tile"], r["tiles"]) for r in stream_rows)):
    total = None
    for tile in range(first, first + count):
        oy, ox = 2 * (tile // 160) - 1, 2 * (tile % 160) - 1
        word = np.zeros((96, 4, 4), dtype=np.uint16)
        valid_words = 0
        for y in range(4):
            for x in range(4):
                if 0 <= oy + y < 240 and 0 <= ox + x < 320:
                    word[:, y, x] = source[:, oy + y, ox + x]
                    valid_words += 96
        pred, raw = predict(word, q1, q2, valid_words)
        assert np.array_equal(raw.transpose(1, 2, 0).reshape(10, 96, 2, 2), gold_raw[tile])
        if total is None:
            total = dict.fromkeys(pred, 0)
        for key, value in pred.items():
            total[key] += value
    predictions[f"native_{first}_{count}"] = total

for r in fixture_rows + stream_rows:
    mode, n = r["mode"], r["tiles"]
    if "fixture" in r:
        p = {k: v * n for k, v in predictions[r["fixture"]].items()}
    else:
        p = predictions[f"native_{r['first_tile']}_{n}"]
    u = p[f"U{mode}"]
    repair, norm = (p["repairs"], 10 * n) if mode == 2 else (0, 0)
    expect = dict(core_first_issues=u, core_merged_updates=p["A"] - u,
                  core_repair_issues=repair, core_repair_fields=p["fields"] if mode == 2 else 0,
                  core_normalization_issues=norm, core_z_vector_reads=u + 40 * n + repair + norm,
                  core_z_writes=u + 10 * n + repair + norm, core_z_scalar_reads=p["M"], core_mac_issues=p["M"],
                  core_source_words=p["source"], core_weight_words=p["Q"] + p["V"], core_second_weight_words=p["V"],
                  core_local_source_reads=p["K"], core_psum_reads=480 * n, core_psum_writes=480 * n,
                  proof_issues=0 if r["command"] else 864, range_fallback_tiles=p["fallback"] if mode == 1 else 0)
    expect["core_cycles"] = 5427 * n + p["K"] + 2 * p["Q"] + 3 * u + p["M"] + 2 * (repair + norm) + sum(r[k] for k in ["core_source_stalls", "core_weight_stalls", "core_output_stalls", "core_arbitration_stalls"])
    for key, value in expect.items():
        assert r[key] == value, (r.get("fixture", r.get("first_tile")), mode, key, r[key], value)
    assert r["shared_alu_grants"] == r["proof_issues"] + u + p["M"] + repair + norm
    assert r["shared_z_grants"] == r["core_z_vector_reads"] + r["core_z_scalar_reads"] + r["core_z_writes"]
    assert r["shared_source_grants"] == p["source"] and r["shared_weight_grants"] == p["Q"] + p["V"]
    assert r["shared_psum_grants"] == 960 * n and r["conflict_cycles"] == r["core_arbitration_stalls"]
    assert r["consumer_cycles"] == 3385 * n + r["consumer_join_wait_cycles"] + r["consumer_output_stalls"]
    assert r["window_cycles"] == r["consumer_cycles"] + n + n // 2
    assert r["launch_cycles"] == r["batches"] == (n + 1) // 2
    assert r["total_cycles"] == r["window_cycles"] + r["launch_cycles"] + sum(r[k] for k in ["static_words", "parameter_stalls", "source_load_words", "origin_words", "source_load_stalls"]) + 1
    assert r["source_load_words"] == 1536 * n and r["origin_words"] == n
    assert r["outputs"] == r["raw_outputs"] == r["J_outputs"] == 3840 * n
    assert r["output_beats"] == r["consumer_identity_words"] == 480 * n
    assert r["consumer_add_issues"] == 960 * n and r["consumer_coefficient_words"] == 24 * n

small = {(r["first_tile"], r["mode"]): r for r in stream_rows if r["tiles"] == 3 and not r["stall"] and not r["command"]}
gate = all(small[first, 2]["total_cycles"] < small[first, 1]["total_cycles"] for first in [159, 19197])
all_rows = fixture_rows + stream_rows
out = dict(passed=True, commands=len(all_rows), values_each_stage=sum(r["outputs"] for r in all_rows),
           gate_64=gate, native_predictions={k: v for k, v in predictions.items() if k.startswith("native_")},
           dual_context_repair_waits=sum(r["core_repair_arbitration_stalls"] for r in fixture_rows if r["tiles"] == 2),
           dual_context_normalization_waits=sum(r["core_normalization_arbitration_stalls"] for r in fixture_rows if r["tiles"] == 2))
(H / "checks.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
