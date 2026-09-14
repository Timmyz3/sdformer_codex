"""Reconstruct Q1 partial sums and exact centered carry obligations independently."""
from pathlib import Path
import json
import numpy as np

H = Path(__file__).resolve().parent
OLD = H.parents[1] / "fusion_ten_trials_20260914/phase_borrow"
rows = json.loads((H / "results.json").read_text())
predictions = {}
for name in sorted(set(r["fixture"] for r in rows)):
    fixture = OLD / "fixtures" / name
    meta = json.loads((fixture / "meta.json").read_text())
    oy, ox = meta["input_origin"]
    src = np.fromfile(fixture / "source.bin", "<u2").reshape(96, 4, 4).copy()
    valid = np.array([[0 <= oy + y < 240 and 0 <= ox + x < 320 for x in range(4)] for y in range(4)])
    src *= valid
    spikes = ((src[None] >> np.arange(10)[:, None, None, None]) & 1).astype(np.int64)
    q1 = np.fromfile(fixture / "param4.bin", "<i4").astype(np.int64).reshape(864, 8)
    q2 = np.fromfile(fixture / "param5.bin", "<i4").astype(np.int64).reshape(12, 8, 8).transpose(0, 2, 1).reshape(96, 8)
    patches = np.stack([spikes[:, :, p // 2:p // 2 + 3, p % 2:p % 2 + 3].reshape(10, 864) for p in range(4)])
    live_k = np.any(q1 != 0, axis=1)
    a = patches[:, :, live_k].astype(bool)
    positive = np.maximum(q1, 0).sum(0)
    negative = np.minimum(q1, 0).sum(0)
    range_ok = bool(np.all(positive <= 511) and np.all(negative >= -512))
    unions = {0: int((a[0] | a[1]).sum() + (a[2] | a[3]).sum()),
              1: int((a[0] | a[1] | a[2]).sum() + a[3].sum()),
              2: int(a.any(0).sum())}
    if not range_ok:
        unions[1] = unions[0]
    exact = np.zeros((4, 10, 8), dtype=np.int64)
    low = np.zeros_like(exact)
    high = np.zeros_like(exact)
    repairs = fields = 0
    for k in range(864):
        if not live_k[k]:
            continue
        for t in range(10):
            active = patches[:, t, k].astype(bool)
            if not active.any():
                continue
            delta = active[:, None] * q1[k][None, :]
            exact[:, t] += delta
            unwrapped = low[:, t] + delta
            overflow = (unwrapped < -128) | (unwrapped > 127)
            repairs += bool(overflow.any())
            fields += int(overflow.sum())
            high[:, t] += (unwrapped > 127).astype(np.int64) - (unwrapped < -128).astype(np.int64)
            low[:, t] = ((unwrapped + 128) & 255) - 128
            assert np.array_equal(exact[:, t], low[:, t] + high[:, t] * 256)
            assert np.all(high[:, t] >= -16) and np.all(high[:, t] < 16)
            if range_ok:
                assert np.all(exact[:, t] >= -512) and np.all(exact[:, t] < 512)
    canonical_high = high - (low < 0).astype(np.int64)
    canonical = ((canonical_high & 31) << 8) | (low & 255)
    canonical = np.where(canonical >= 4096, canonical - 8192, canonical)
    assert np.array_equal(canonical, exact)
    independent_raw = exact @ q2.T
    raw = np.fromfile(fixture / "raw.bin", "<i4").reshape(12, 4, 10, 8).transpose(1, 2, 0, 3).reshape(4, 10, 96)
    assert np.array_equal(independent_raw, raw)
    z_live = exact != 0
    v_live = np.any(q2.reshape(12, 8, 8) != 0, axis=1)
    rank_live = z_live.any((0, 1))
    predictions[name] = dict(unions=unions, repairs=repairs, repair_fields=fields, range_ok=range_ok,
                             positive=positive.tolist(), negative=negative.tolist(), A=int(a.sum()), K=int(live_k.sum()),
                             Q=int(a.any((0, 1)).sum()), V=int((v_live & rank_live).sum()),
                             M=int((z_live.sum((0, 1)) * v_live.sum(0)).sum()), source=96 * int(valid.sum()))

for r in rows:
    p = predictions[r["fixture"]]
    mode = r["mode"]
    u = p["unions"][mode]
    repair = p["repairs"] if mode == 2 else 0
    norm = 10 if mode == 2 else 0
    expected = dict(core_first_issues=u, core_merged_updates=p["A"] - u,
                    core_repair_issues=repair, core_repair_fields=p["repair_fields"] if mode == 2 else 0,
                    core_normalization_issues=norm, core_z_vector_reads=u + 40 + repair + norm,
                    core_z_writes=u + 10 + repair + norm, core_z_scalar_reads=p["M"], core_mac_issues=p["M"],
                    core_weight_words=p["Q"] + p["V"], core_second_weight_words=p["V"],
                    core_source_words=p["source"], core_local_source_reads=p["K"],
                    proof_issues=0 if r["command"] else 864, proof_range_ok=int(p["range_ok"]),
                    range_fallback_tiles=int(mode == 1 and not p["range_ok"]),
                    core_psum_reads=480, core_psum_writes=480)
    expected["core_cycles"] = 5427 + p["K"] + 2 * p["Q"] + 3 * u + p["M"] + 2 * (repair + norm) + sum(r[k] for k in ["core_source_stalls", "core_weight_stalls", "core_output_stalls"])
    for key, value in expected.items():
        assert r[key] == value, (r["fixture"], mode, key, r[key], value)
    assert r["consumer_cycles"] == 3385 + r["consumer_join_wait_cycles"] + r["consumer_output_stalls"]
    assert r["total_cycles"] == r["consumer_cycles"] + sum(r[k] for k in ["static_words", "parameter_stalls", "source_load_words", "origin_words", "source_load_stalls"]) + 3

old = json.loads((OLD / "results.json").read_text())
for r in rows:
    if r["mode"] != 0:
        continue
    prior = next(x for x in old if x["fixture"] == r["fixture"] and x["mode"] == 2 and x["stall"] == r["stall"] and x["command"] == r["command"])
    # The original BP suite also stalls the independent consumer-wide permit.
    # Only no-BP mode0 is expected to preserve its exact whole-cycle receipt.
    if not r["stall"]:
        for key in ["total_cycles", "core_cycles", "consumer_cycles", "core_first_issues", "core_mac_issues"]:
            assert r[key] == prior[key]

real = {m: {k: sum(r[k] for r in rows if r["fixture"].startswith("real_") and r["mode"] == m and r["stall"] == 0 and r["command"] == 0)
            for k in ["total_cycles", "core_first_issues", "core_repair_issues", "core_normalization_issues"]} for m in [0, 1, 2]}
output = dict(passed=True, commands=len(rows), reconstructed_raw_values=len(predictions) * 3840,
              real8=real, fixture_predictions=predictions,
              gate_64=real[2]["total_cycles"] < real[1]["total_cycles"])
(H / "checks.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps({k: v for k, v in output.items() if k != "fixture_predictions"}))
