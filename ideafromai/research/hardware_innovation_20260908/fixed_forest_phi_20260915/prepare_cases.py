"""Export 8+8 strips and 54+54 full-K strips, with no PWP/gold payload.

Selections are fixed by address before inspecting activity. Run with Python/NumPy.
Only cases.npz and CASES.md beside this script are written. Existing sources are
read-only; no importing the old prepare.py (which has side effects).
"""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REF = BASE / "shared_execution_20260915/r8_reference"
DATA = BASE / "r8_consumer_fusion_20260914/data"
K16_IDS = (0, 7, 15, 23, 30, 38, 46, 53)
CALIBRATION_TILES = (0, 159, 19040, 19199, 2584, 4840, 9680, 14520)
HELD_TILES = (128, 136, 144, 152, 4000, 4016, 4032, 4048)


def make_forest(bits):
    """Deterministic local-author-kernel policy: max popcount, first index tie.

    Preserve zero/onehot bypass. Equal rows may inherit only earlier row IDs.
    Proper subset parents have fewer bits. No synthesized node or future values.
    """
    masks = [sum(int(v) << k for k, v in enumerate(row)) for row in bits]
    pc = [v.bit_count() for v in masks]
    parents = np.full(40, -1, np.int16)
    for i, mask in enumerate(masks):
        if pc[i] < 2:
            continue
        best_count = 0
        for j, candidate in enumerate(masks):
            if pc[j] <= best_count or candidate & mask != candidate:
                continue
            if candidate == mask and j >= i:
                continue
            parents[i] = j
            best_count = pc[j]
    order = np.asarray(sorted(range(40), key=lambda i: (pc[i], i)), np.uint8)
    delta = np.asarray([mask ^ (masks[int(parents[i])] if parents[i] >= 0 else 0)
                        for i, mask in enumerate(masks)], np.uint16)
    seen = set()
    for i in map(int, order):
        p = int(parents[i])
        if p >= 0:
            assert p in seen and (masks[p] & masks[i]) == masks[p]
        seen.add(i)
    return np.asarray(masks, np.uint16), parents, order, delta


def source_footprint(origin):
    """Valid absolute spatial coordinates for the complete C96 x 4 x 4 halo."""
    iy, ix = map(int, origin)
    return {(iy + y, ix + x) for y in range(4) for x in range(4)
            if 0 <= iy + y < 240 and 0 <= ix + x < 320}


def main():
    records = json.loads((REF / "fixtures.json").read_text())
    q1_loaded = np.fromfile(REF / "parameters/param4.bin", dtype="<i4").reshape(864, 8)
    with np.load(DATA / "factors.npz", allow_pickle=False) as factors:
        q1_factor = factors["q1"].reshape(8, 864).T.astype(np.int32)
    assert np.array_equal(q1_loaded, q1_factor)
    assert ((q1_loaded >= -4) & (q1_loaded <= 3)).all()
    q1 = q1_loaded.astype(np.int8)
    full_source = np.load(DATA / "first_source_words.npy", mmap_mode="r", allow_pickle=False)
    assert full_source.shape == (96, 240, 320)
    assert set(CALIBRATION_TILES).isdisjoint(HELD_TILES)

    # First geometrically interior pair in the already fixed small-case list.
    # This selection never inspects source bits, dot products, or activity.
    def interior(tile):
        y, x = records[f"tile_{tile}"]["input_origin_yx"]
        return 0 <= y and y + 3 < 240 and 0 <= x and x + 3 < 320
    full_cal, full_held = next((a, b) for a, b in zip(CALIBRATION_TILES, HELD_TILES)
                              if interior(a) and interior(b))
    assert (full_cal, full_held) == (4840, 4016)
    specifications = (("calibration", list(zip(CALIBRATION_TILES, K16_IDS))),
                      ("held", list(zip(HELD_TILES, K16_IDS))),
                      ("full_calibration", [(full_cal, k) for k in range(54)]),
                      ("full_held", [(full_held, k) for k in range(54)]))

    output = {}
    all_rows = []
    footprints = {}
    verified_q1_tiles = set()
    checked_source_words = checked_scalar_gates = checked_q1_values = 0
    for split, cases in specifications:
        arrays = {key: [] for key in ("S", "W", "case_name", "tile_id", "k16_id",
                  "global_k", "source_words", "input_origin_yx", "masks", "parents",
                  "order", "delta_masks", "fixture_relative_path")}
        footprints[split] = set()
        for tile, k16 in cases:
            fixture = REF / "fixtures" / f"tile_{tile}"
            meta = records[f"tile_{tile}"]
            assert meta["frame_index"] == 0 and meta["file"] == "zurich_city_09_a_0001.npy"
            words = np.fromfile(fixture / "source.bin", dtype="<u2").reshape(96, 4, 4)
            origin = np.fromfile(fixture / "origin.bin", dtype="<i4")
            assert np.array_equal(origin, meta["input_origin_yx"])
            assert (words < 1024).all()
            footprints[split] |= source_footprint(origin)
            # Independently check the actual raw halo words against the full-frame
            # source; padding is real zero here, not an activity-selected mask.
            for y in range(4):
                for x in range(4):
                    yy, xx = int(origin[0]) + y, int(origin[1]) + x
                    expected = full_source[:, yy, xx] if 0 <= yy < 240 and 0 <= xx < 320 else np.zeros(96, np.uint16)
                    assert np.array_equal(words[:, y, x], expected)
                    checked_source_words += 96
            global_k = np.arange(k16 * 16, k16 * 16 + 16, dtype=np.int32)
            S = np.zeros((40, 16), np.uint8)
            for row in range(40):
                p, t = divmod(row, 10)
                py, px = divmod(p, 2)
                for j, gk in enumerate(map(int, global_k)):
                    c, tap = divmod(gk, 9)
                    ky, kx = divmod(tap, 3)
                    S[row, j] = (int(words[c, py + ky, px + kx]) >> t) & 1
                    # A separate path starts at absolute full-frame coordinates.
                    yy, xx = int(origin[0]) + py + ky, int(origin[1]) + px + kx
                    direct_word = int(full_source[c, yy, xx]) if 0 <= yy < 240 and 0 <= xx < 320 else 0
                    assert S[row, j] == ((direct_word >> t) & 1)
                    checked_scalar_gates += 1
            W = q1[global_k].copy()
            masks, parents, order, delta = make_forest(S)
            # CPU validation only: no dot-product answer is saved in cases.npz.
            expected = S.astype(np.int64) @ W.astype(np.int64)
            reconstructed = np.zeros_like(expected)
            for row in map(int, order):
                if parents[row] >= 0:
                    reconstructed[row] = reconstructed[int(parents[row])]
                for j in range(16):
                    if (int(delta[row]) >> j) & 1:
                        reconstructed[row] += W[j]
            assert np.array_equal(expected, reconstructed)
            # Verify the complete Q1 ordering against existing latent gold, not
            # merely against another spelling of the selected-strip dot product.
            latent = np.fromfile(fixture / "latent.bin", dtype="<i4").reshape(10, 8, 2, 2)
            if tile not in verified_q1_tiles:
                ev = (words[None] >> np.arange(10)[:, None, None, None]) & 1
                for py in range(2):
                    for px in range(2):
                        patch = ev[:, :, py:py + 3, px:px + 3].reshape(10, 864)
                        got = patch.astype(np.int64) @ q1.astype(np.int64)
                        assert np.array_equal(got, latent[:, :, py, px])
                        checked_q1_values += got.size
                verified_q1_tiles.add(tile)
            name = f"{split}_tile_{tile}_k16_{k16:02d}"
            values = dict(S=S, W=W, case_name=name, tile_id=np.int32(tile),
                          k16_id=np.int32(k16), global_k=global_k, source_words=words,
                          input_origin_yx=origin, masks=masks, parents=parents,
                          order=order, delta_masks=delta,
                          fixture_relative_path=str(fixture.relative_to(BASE)))
            for key in arrays:
                arrays[key].append(values[key])
            all_rows.append(dict(split=split, tile=tile, k16=k16, source_ones=int(S.sum()),
                                 live_rows=int(np.any(S, axis=1).sum()), parents=int((parents >= 0).sum()),
                                 delta_multi=sum(int(x).bit_count() > 1 for x in delta)))
        for key, vals in arrays.items():
            output[f"{split}_{key}"] = np.asarray(vals)
        assert output[f"{split}_S"].shape == (len(cases), 40, 16)
        assert output[f"{split}_W"].shape == (len(cases), 16, 8)
    # No calibration pixel can reappear in any held source halo, even at another
    # channel/K partition. They remain an intra-frame spatial split, not new data.
    assert footprints["calibration"].isdisjoint(footprints["held"])
    assert np.array_equal(output["calibration_k16_id"], output["held_k16_id"])
    assert np.array_equal(output["calibration_W"], output["held_W"])
    assert footprints["full_calibration"].isdisjoint(footprints["full_held"])
    assert footprints["full_calibration"] <= footprints["calibration"]
    assert footprints["full_held"] <= footprints["held"]
    assert np.array_equal(output["full_calibration_k16_id"], np.arange(54))
    assert np.array_equal(output["full_held_k16_id"], np.arange(54))
    assert np.array_equal(output["full_calibration_W"], output["full_held_W"])
    full_stats = {}
    for split, tile in (("full_calibration", full_cal), ("full_held", full_held)):
        S, W = output[f"{split}_S"], output[f"{split}_W"]
        assert np.array_equal(output[f"{split}_global_k"].reshape(-1), np.arange(864))
        # Sum all 54 strip results: a real full-K Q1 output tile, not a full layer.
        total = np.einsum("kmj,kjn->mn", S.astype(np.int64), W.astype(np.int64))
        latent = np.fromfile(REF / "fixtures" / f"tile_{tile}" / "latent.bin", dtype="<i4").reshape(10, 8, 2, 2)
        assert np.array_equal(total, latent.transpose(2, 3, 0, 1).reshape(40, 8))
        parents = output[f"{split}_parents"]
        residual_count = np.asarray([[int(v).bit_count() for v in mask]
                                    for mask in output[f"{split}_delta_masks"]])
        original_count = S.sum(axis=2)
        full_stats[split] = dict(tile=tile, strips=54, row_partition_instances=2160,
            original_source_ones=int(S.sum()), original_popcount_histogram=np.bincount(original_count.astype(int).ravel(), minlength=17).tolist(),
            root_delta_popcount_histogram=np.bincount(residual_count.ravel(), minlength=17).tolist(),
            original_multi_rows=int((original_count > 1).sum()), parents=int((parents >= 0).sum()),
            multi_roots=int(((parents < 0) & (residual_count > 1)).sum()),
            multi_parent_deltas=int(((parents >= 0) & (residual_count > 1)).sum()),
            residual_multi_rows=int((residual_count > 1).sum()),
            residual_terms=int(residual_count.sum()),
            strips_with_multi_residual=int(np.any(residual_count > 1, axis=1).sum()))
    for key in ("S", "W", "case_name", "tile_id", "k16_id", "global_k", "masks", "parents", "order", "delta_masks"):
        output[key] = np.concatenate([output[f"calibration_{key}"], output[f"held_{key}"]], axis=0)
    output["split"] = np.asarray(["calibration"] * 8 + ["held"] * 8)
    output["row_p"] = np.repeat(np.arange(4, dtype=np.uint8), 10)
    output["row_t"] = np.tile(np.arange(10, dtype=np.uint8), 4)
    output["model_id"] = np.asarray("deployed_flatR8_Q1_signed3")
    output["forest_policy"] = np.asarray("max_subset_popcount_then_min_row_index; zero/onehot_no_parent; EM_earlier_index; stable_popcount_index_order")
    if (HERE / "cases.npz").exists():
        with np.load(HERE / "cases.npz", allow_pickle=False) as previous:
            for key in previous.files:
                if not key.startswith("full_"):
                    assert np.array_equal(previous[key], output[key]), ("old_case_changed", key)
    # Atomic replacement also permits the root agent to read the old archive
    # while this preparation executes. The temporary is removed by replace().
    temporary = HERE / ".cases.tmp.npz"
    np.savez_compressed(temporary, **output)
    temporary.replace(HERE / "cases.npz")
    with np.load(HERE / "cases.npz", allow_pickle=False) as archive:
        for key, values in output.items():
            assert np.array_equal(archive[key], values), key

    report = ["# 真实 R8 Q1：8＋8 条带与一对 full-K 小输出 tile", "",
        "固定地址选择；没有查看活动率后换点。只生成原始源/权重及确定森林，不保存 PWP、代码本、点积或输出 gold；CPU 点积仅用于本导出脚本断言。", "",
        "来源为 `../shared_execution_20260915/r8_reference/fixtures/tile_*/source.bin`、`origin.bin` 与 `parameters/param4.bin`。W 另与 `../r8_consumer_fusion_20260914/data/factors.npz:q1` 逐值相等；源 halo 另与该 data 目录的 `first_source_words.npy` 逐词相等。模型身份沿用部署 flatR8，不新增量化、训练或质量结论。", "",
        "输入为同一 `zurich_city_09_a_0001.npy`、frame_index=0 的不同位置；校准/held 的 tile 身份和**完整有效 4×4 source halo 坐标并集均不相交**。这是同帧空间留出，不是独立序列验证。8 对 K16 地址相同，W 完全相同；每地址目前只有 40 条校准 row，不能夸大代码本统计量。", "",
        "`calibration_S` / `held_S`: uint8[8,40,16]，`calibration_W` / `held_W`: int8[8,16,8]。`row=p*10+t`，`p=2*py+px`，`K=c*9+ky*3+kx`；各 strip 全部16位原样保留，没有按 W 的 k-live 削源。真实全部 W 范围为 [%d,%d]，满足 signed3；没有凭空加入 −4。" % (int(q1.min()), int(q1.max())), "",
        "同前缀还含 `case_name/tile_id/k16_id/global_k/source_words/input_origin_yx/masks/parents/order/delta_masks/fixture_relative_path`。`source_words` 为未改 C96×4×4 原始低10门字。合并视图 `S[16,40,16]`、`W[16,16,8]`、`case_name` 等按 calibration 在前、held 在后，必须使用 `split` 区分；默认 `allow_pickle=False` 可读。", "",
        "确定森林：popcount<2 不设父；其余取非零最大子集，平局选最小原 row index；相同 mask 只能取更早 index。稳定顺序为 `(popcount,index)`，parent=-1 为 root。该规则匹配已读本地作者 kernel 的 first-index 口径，区别于论文并列时最大 index；本轮只固定一种，不择优。`delta_masks=mask XOR parent_mask`；森林只读原始 S，不读 W 或 held 校准信息。", "",
        "本导出不生成 Phi 代码本，后续 prepare_runs.py 单独完成校准：每个 K16 只允许对应 calibration 数据参与选中心；Phi-alone 可用原始 S 独立校准，固定森林臂可用 root/delta 校准，同 q/样本预算。held 数据只供运行时精确查询，不选最佳离线中心。禁止把 CPU 断言中的乘积作为 RTL 配置。", "",
        "| split | tile | K16编号 | 起始K | 原始1数/640 | 非零行/40 | 有父行 | 多项root/delta |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in all_rows[:16]:
        report.append(f"| {r['split']} | {r['tile']} | {r['k16']} | {16*r['k16']} | {r['source_ones']} | {r['live_rows']} | {r['parents']} | {r['delta_multi']} |")
    report += ["", "## 新增 full-K 覆盖：全部 54 个 K16，保留原小集", "",
        f"固定校准 tile{full_cal} / held tile{full_held}，按既有配对列表选择首个双方4×4 halo均在图内的几何配对，选择不检查活动率。两侧 halo 不交；小集全部旧字段逐值保持。这里新增的是 **P4×T10×N8 的一个小输出 tile、完整 K864**，不是完整空间层、全网、64 tile 测试，也没有新增独立帧。", "",
        "`full_calibration_S` / `full_held_S` 为 uint8[54,40,16]，对应 W 为 int8[54,16,8]；同前缀提供 case_name、tile_id、k16_id、global_k、source_words、input_origin_yx、masks、parents、order、delta_masks、fixture_relative_path。编号0..53全部保留，未按活动筛选。两侧同地址 W 完全相等；每 K 的校准仍只有该位置40条row，held 不参与选中心。", "",
        "下表分母均为54×40=2160个 **row/K分区实例**；多项指该root或parent-delta的popcount≥2，零项也包含零源root和EM空delta。这个统计不是可省周期或可省W字节。", "",
        "| split / tile | 原始1数/34560 | 原始多项row | 有父row | 多项root | 多项parent-delta | 多项总数/2160 | 有多项的K16数/54 | 剩余term数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, stats in full_stats.items():
        report.append(f"| {split} / {stats['tile']} | {stats['original_source_ones']} | {stats['original_multi_rows']} | {stats['parents']} | {stats['multi_roots']} | {stats['multi_parent_deltas']} | {stats['residual_multi_rows']} | {stats['strips_with_multi_residual']} | {stats['residual_terms']} |")
    report += ["", "完整popcount直方图，数组index=0..16；保留两侧所有空条带：", ""]
    for split, stats in full_stats.items():
        report.append(f"- `{split}` 原始 S：`{stats['original_popcount_histogram']}`；root/delta：`{stats['root_delta_popcount_histogram']}`。")
    report += ["", f"导出自检通过：按124条带共进行了 {checked_source_words:,} 次原始 halo word 比较（包含full-K重复读同一tile）；{checked_scalar_gates:,} 次独立绝对坐标门位比较；全部124条带的固定森林局部 N8 重构；16个不同tile的 {checked_q1_values:,} 个完整 Q1 latent 与旧 fixture gold。新增两侧54条带的结果逐K相加，另核对640个完整Q1输出值。NPZ写回读取逐字段相同；旧小集全部非full字段逐值不变。", "",
        "此处给的是已展开 K16 叶输入；原始 source 的存取/im2col、检测/排序仍须在完整系统比较中收费，不能以这份 NPZ 视为免费前端。`cases.npz` 为再生本地数据，不加入 Git。", "",
        "复现：`/opt/anaconda3/bin/python3.12 prepare_cases.py`。脚本只写本目录 `cases.npz` 和 `CASES.md`，不改旧 fixture。"]
    (HERE / "CASES.md").write_text("\n".join(report) + "\n")
    print(json.dumps(dict(cases=124, calibration=8, held=8, full_calibration=54, full_held=54,
                         q1_range=[int(q1.min()), int(q1.max())],
                         raw_source_words_checked=checked_source_words, gate_bits_checked=checked_scalar_gates,
                         full_Q1_values_checked=checked_q1_values, source_halos_disjoint=True,
                         archive_bytes=(HERE / "cases.npz").stat().st_size, full_stats=full_stats), indent=2))


if __name__ == "__main__":
    main()
