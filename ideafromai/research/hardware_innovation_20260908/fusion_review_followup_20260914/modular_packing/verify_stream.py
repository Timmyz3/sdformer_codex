from pathlib import Path
import json
import numpy as np

H = Path(__file__).resolve().parent
D = H.parents[1] / "r8_consumer_fusion_20260914/data"
rows = json.loads((H / "results_64.json").read_text())
src = np.load(D / "first_source_words.npy", mmap_mode="r")
raw = np.load(D / "raw_p_full.npy", mmap_mode="r")
coeff = np.load(D / "consumer_coefficients.npz")
q1, q2 = coeff["q1"].astype(np.int64), coeff["q2"].astype(np.int64)
live_k = np.any(q1 != 0, axis=0)
v_live = np.any(q2.reshape(12, 8, 8) != 0, axis=1)
totals = dict(U0=0, U1=0, U2=0, repairs=0, repair_fields=0, A=0, K=0, Q=0, V=0, M=0, source=0)
for tile in range(128, 192):
    oy, ox = 2 * (tile // 160) - 1, 2 * (tile % 160) - 1
    block = np.zeros((96, 4, 4), dtype=np.uint16)
    for y in range(4):
        for x in range(4):
            if 0 <= oy + y < 240 and 0 <= ox + x < 320:
                block[:, y, x] = src[:, oy + y, ox + x]
                totals["source"] += 96
    spike = ((block[None] >> np.arange(10)[:, None, None, None]) & 1).astype(np.int64)
    patch = np.stack([spike[:, :, p // 2:p // 2 + 3, p % 2:p % 2 + 3].reshape(10, 864) for p in range(4)])
    a = patch[:, :, live_k].astype(bool)
    totals["U0"] += int((a[0] | a[1]).sum() + (a[2] | a[3]).sum())
    totals["U1"] += int((a[0] | a[1] | a[2]).sum() + a[3].sum())
    totals["U2"] += int(a.any(0).sum())
    totals["A"] += int(a.sum())
    totals["K"] += int(live_k.sum())
    totals["Q"] += int(a.any((0, 1)).sum())
    exact = np.zeros((4, 10, 8), dtype=np.int64)
    low = np.zeros_like(exact)
    high = np.zeros_like(exact)
    for k in range(864):
        if not live_k[k]:
            continue
        delta = patch[:, :, k, None] * q1[:, k][None, None, :]
        exact += delta
        unwrapped = low + delta
        overflow = (unwrapped < -128) | (unwrapped > 127)
        totals["repairs"] += int(overflow.any(axis=(0, 2)).sum())
        totals["repair_fields"] += int(overflow.sum())
        high += (unwrapped > 127).astype(np.int64) - (unwrapped < -128).astype(np.int64)
        low = ((unwrapped + 128) & 255) - 128
        assert np.array_equal(exact, low + 256 * high)
    assert np.all(high >= -16) and np.all(high < 16)
    canonical = (((high - (low < 0)) & 31) << 8) | (low & 255)
    canonical = np.where(canonical >= 4096, canonical - 8192, canonical)
    assert np.array_equal(canonical, exact)
    assert np.array_equal((exact @ q2.T).transpose(1, 2, 0).reshape(10, 96, 2, 2), raw[tile])
    z_live = exact != 0
    totals["V"] += int((v_live & z_live.any((0, 1))).sum())
    totals["M"] += int((z_live.sum((0, 1)) * v_live.sum(0)).sum())

for r in rows:
    m, n = r["mode"], r["tiles"]
    u = totals[f"U{m}"]
    repair, norm = (totals["repairs"], 10 * n) if m == 2 else (0, 0)
    expect = dict(core_first_issues=u, core_merged_updates=totals["A"] - u,
                  core_repair_issues=repair, core_repair_fields=totals["repair_fields"] if m == 2 else 0,
                  core_normalization_issues=norm, core_z_vector_reads=u + 40 * n + repair + norm,
                  core_z_writes=u + 10 * n + repair + norm, core_z_scalar_reads=totals["M"], core_mac_issues=totals["M"],
                  core_source_words=totals["source"], core_local_source_reads=totals["K"],
                  core_weight_words=totals["Q"] + totals["V"], core_second_weight_words=totals["V"],
                  proof_issues=0 if r["command"] else 864, range_fallback_tiles=0, proof_range_ok=1)
    expect["core_cycles"] = 5427 * n + totals["K"] + 2 * totals["Q"] + 3 * u + totals["M"] + 2 * (repair + norm) + sum(r[k] for k in ["core_source_stalls", "core_weight_stalls", "core_output_stalls"])
    for key, value in expect.items():
        assert r[key] == value, (m, key, r[key], value)
    assert r["source_load_words"] == 1536 * n
    assert r["consumer_cycles"] == 3385 * n + r["consumer_join_wait_cycles"] + r["consumer_output_stalls"]
    assert r["total_cycles"] == r["consumer_cycles"] + sum(r[k] for k in ["static_words", "parameter_stalls", "source_load_words", "origin_words", "source_load_stalls"]) + 2 * n + 1

cold = {r["mode"]: r for r in rows if r["stall"] == 0 and r["command"] == 0}
saved = cold[1]["total_cycles"] - cold[2]["total_cycles"]
assert saved == 3 * (totals["U1"] - totals["U2"]) - 2 * (totals["repairs"] + 640)
out = dict(passed=True, commands=len(rows), values_each_stage=sum(r["outputs"] for r in rows),
           native_totals=totals, cold_noBP_cycles={m: cold[m]["total_cycles"] for m in cold},
           candidate_saved_vs_triple10=saved, candidate_saved_fraction=saved / cold[1]["total_cycles"])
(H / "checks_64.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
