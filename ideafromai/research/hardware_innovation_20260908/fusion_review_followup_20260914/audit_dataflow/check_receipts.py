"""Independent receipt arithmetic and representative native-input gold reconstruction."""
import json
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
B = H.parents[1]
OLD = B / "fusion_ten_trials_20260914"
D = B / "r8_consumer_fusion_20260914/data"
families = ["dataflow/d1_forward", "dataflow/d2_halo", "dataflow/d3_interleave", "phase_borrow", "joint_selected"]
receipts = []
for family in families:
    for path in (OLD / family).glob("results*.json"):
        for r in json.loads(path.read_text()):
            n = r["retired_tiles"]
            assert r["outputs"] == r["raw_outputs"] == r["J_outputs"] == n * 3840
            assert r["output_beats"] == r["consumer_identity_words"] == n * 480
            assert r["origin_words"] == n
            assert r["static_words"] == (0 if r["command"] else 1848)
            assert r["source_load_words"] == r["external_source_words"] + r["padding_words"]
            assert r["consumer_conversion_issues"] == r["consumer_mul_issues"] == r["consumer_round_issues"] == n * 480
            assert r["consumer_add_issues"] == n * 960
            assert r["consumer_coefficient_words"] == n * 24
            overhead = sum(r[k] for k in ["static_words", "parameter_stalls", "source_load_words", "origin_words", "source_load_stalls"])
            assert r["total_cycles"] == r["consumer_cycles"] + overhead + 2 * n + 1
            if family.endswith("d3_interleave"):
                assert r["shared_alu_grants"] == r["core_first_issues"] + r["core_mac_issues"]
                assert r["shared_z_grants"] == r["core_z_vector_reads"] + r["core_z_scalar_reads"] + r["core_z_writes"]
                assert r["shared_source_grants"] == r["core_source_words"]
                assert r["shared_weight_grants"] == r["core_weight_words"]
                assert r["shared_psum_grants"] == r["core_psum_reads"] + r["core_psum_writes"]
                assert r["conflict_cycles"] == r["core_arbitration_stalls"]
            if path.name == "results_full.json":
                receipts.append(dict(family=family, mode=r["mode"], cycles=r["total_cycles"],
                                     source_load_words=r["source_load_words"], values_each_stage=r["outputs"]))
    fixture = OLD / family / "fixtures/real_0"
    q1 = np.fromfile(fixture / "param4.bin", "<i4").reshape(864, 8)
    metadata = np.fromfile(fixture / "param6.bin", "<u4").reshape(864, 8)
    assert np.array_equal(metadata[:, 0], np.any(q1 != 0, axis=1))

src = np.load(D / "first_source_words.npy", mmap_mode="r")
raw = np.load(D / "raw_p_full.npy", mmap_mode="r")
identity = np.load(D / "identity_fp32_full.npy", mmap_mode="r")
jgold = np.load(D / "identity_q20_full.npy", mmap_mode="r")
igold = np.load(D / "i24_new_full.npy", mmap_mode="r")
coeff = np.load(D / "consumer_coefficients.npz")
q1, q2 = coeff["q1"].astype(np.int64), coeff["q2"].astype(np.int64)
a, b = coeff["a_q40"].astype(np.int64), coeff["b_q20"].astype(np.int64)
tiles = sorted(set(range(128, 192)) | {19197, 19198, 19199})
for tile in tiles:
    y, x = 2 * (tile // 160) - 1, 2 * (tile % 160) - 1
    block = np.zeros((96, 4, 4), dtype=np.uint16)
    for dy in range(4):
        for dx in range(4):
            if 0 <= y + dy < 240 and 0 <= x + dx < 320:
                block[:, dy, dx] = src[:, y + dy, x + dx]
    spikes = ((block[None] >> np.arange(10)[:, None, None, None]) & 1).astype(np.int64)
    actual = np.empty((10, 96, 2, 2), dtype=np.int64)
    for py in range(2):
        for px in range(2):
            patch = spikes[:, :, py:py + 3, px:px + 3].reshape(10, 864)
            actual[:, :, py, px] = (patch @ q1.T) @ q2.T
    assert np.array_equal(actual, raw[tile])
    j = np.clip(np.rint(identity[tile].astype(np.float64) * (1 << 20)), -(1 << 31), (1 << 31) - 1).astype(np.int64)
    assert np.array_equal(j, jgold[tile])
    wide = actual * a[None, :, None, None] + (j + b[None, :, None, None]) * (1 << 20)
    floor = wide >> 26
    remainder = wide & ((1 << 26) - 1)
    rounded = floor + ((remainder > (1 << 25)) | ((remainder == (1 << 25)) & ((floor & 1) != 0)))
    assert np.array_equal(np.clip(rounded, -(1 << 23), (1 << 23) - 1), igold[tile])

result = dict(old_receipt_arithmetic_passed=True, static_k_live_exact=True,
              native_reconstruction_tiles=len(tiles), native_reconstruction_values_each_stage=len(tiles) * 3840,
              old_full=receipts)
(H / "receipt_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
