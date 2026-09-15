"""Real projected-source FC1/PSN fixtures; no GPU, training, RTL, or code-ID input.

Only cases.npz and CASES.md beside this file are generated. Y/U/gold are explicitly
testbench-only expectations. Static PWP values and source code IDs are not saved.
"""
from pathlib import Path
import json
import runpy
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CAP = BASE / "algorithm/support_training"
PREFIX = "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp."
POSITION_TILES = (0, 199, 399, 599)
P, T, C, H = 32, 10, 96, 96


def pack_n12(values):
    """One 128bit word: 12 signed10 lanes at bits[0:119], bits[120:127]=0."""
    assert len(values) == 12
    packed = 0
    for lane, value in enumerate(values):
        value = int(value)
        assert -512 <= value <= 511
        packed |= (value & 1023) << (10 * lane)
    assert packed >> 120 == 0
    raw = packed.to_bytes(16, "little")
    recovered = []
    for lane in range(12):
        bits = (int.from_bytes(raw, "little") >> (10 * lane)) & 1023
        recovered.append(bits - 1024 if bits & 512 else bits)
    assert np.array_equal(recovered, values)
    return raw


def gates(U, tau, positive, constant, constant_gate):
    signed = np.where(positive[None, None, :], U >= tau[None], U <= tau[None])
    return np.where(constant[None, None, :], constant_gate[None], signed).astype(np.uint8)


def calculate(S, W, A, tau, positive, constant, constant_gate, dictionary, dictionary_words):
    """Independent dense and exact-LUT/escape numerical routes."""
    assert S.shape == (P, T, C) and W.shape == (C, H)
    x, w, a = S.astype(np.int64), W.astype(np.int64), A.astype(np.int64)
    Y = x @ w
    U = np.einsum("ts,psh->pth", a, Y)
    gold = gates(U, tau, positive, constant, constant_gate)
    lookup = [{int(mask): k for k, mask in enumerate(group)} for group in dictionary_words]
    tables = np.einsum("gdc,gch->gdh", dictionary.astype(np.int64), w.reshape(6, 16, H))
    through_lut = np.zeros_like(Y)
    match_count = zero_count = onehot_escapes = multi_escapes = 0
    for p in range(P):
        for t in range(T):
            for g in range(6):
                group = S[p, t, g * 16:g * 16 + 16]
                mask = sum(int(v) << k for k, v in enumerate(group))
                if mask in lookup[g]:
                    match_count += 1
                if mask == 0:
                    zero_count += 1
                elif mask.bit_count() >= 2 and mask in lookup[g]:
                    through_lut[p, t] += tables[g, lookup[g][mask]]
                else:
                    for k in range(16):
                        if mask & (1 << k):
                            through_lut[p, t] += w[g * 16 + k]
                    onehot_escapes += mask.bit_count() == 1
                    multi_escapes += mask.bit_count() >= 2
    assert np.array_equal(Y, through_lut)
    # A different temporal loop order and scalar channel branch check the
    # orientation and >= / <= / constant semantics without any float operation.
    second_U = np.zeros_like(U)
    for s in range(T):
        for t in range(T):
            second_U[:, t] += int(A[t, s]) * through_lut[:, s]
    assert np.array_equal(U, second_U)
    second_gate = np.empty_like(gold)
    for h in range(H):
        if constant[h]:
            second_gate[:, :, h] = constant_gate[:, h][None]
        elif positive[h]:
            second_gate[:, :, h] = second_U[:, :, h] >= tau[:, h][None]
        else:
            second_gate[:, :, h] = second_U[:, :, h] <= tau[:, h][None]
    assert np.array_equal(gold, second_gate)
    lo, hi = np.minimum(w, 0).sum(0), np.maximum(w, 0).sum(0)
    ulo = np.minimum(a[:, :, None] * lo[None, None], a[:, :, None] * hi[None, None]).sum(1)
    uhi = np.maximum(a[:, :, None] * lo[None, None], a[:, :, None] * hi[None, None]).sum(1)
    assert (Y >= lo).all() and (Y <= hi).all()
    assert (U >= ulo[None]).all() and (U <= uhi[None]).all()
    assert min(int(lo.min()), int(ulo.min()), int(tau.min())) >= -(1 << 31)
    assert max(int(hi.max()), int(uhi.max()), int(tau.max())) < (1 << 31)
    return Y.astype(np.int32), U.astype(np.int64), gold, dict(
        source_ones=int(S.sum()), groups=P * T * 6, exact_groups=match_count,
        zero_groups=zero_count, onehot_escape_groups=onehot_escapes,
        multi_escape_groups=multi_escapes, Y_range=[int(Y.min()), int(Y.max())],
        U_range=[int(U.min()), int(U.max())], gates_one=int(gold.sum()))


def main():
    # Reuse only the existing archive reader, not its numerical calculation or
    # service model. run_name is not __main__, and the module performs no writes.
    read_torch = runpy.run_path(str(BASE / "bn_state/support_service_model.py"))["read_torch"]
    D = np.load(CAP / "dictionary.npy", allow_pickle=False).astype(np.uint8)
    assert D.shape == (6, 16, 16) and np.isin(D, (0, 1)).all()
    dwords = (D.astype(np.uint32) * (1 << np.arange(16))).sum(2).astype(np.uint16)
    assert all(len(set(map(int, row))) == 16 for row in dwords)
    Wfull = read_torch(CAP / "forced_code_weight_int8.pt")
    source_params = read_torch(CAP / "forced_code_parameters.pt")
    assert Wfull.shape == (384, 96) and Wfull.dtype == np.int8
    qat = np.rint(np.clip(source_params["W"] * source_params["theta"] /
                          source_params["scale"][:, None], -127, 127)).astype(np.int8)
    assert np.array_equal(Wfull, qat)
    assert np.array_equal(D, source_params["dictionary"])
    consumer = read_torch(BASE / "algorithm/integer_s0_valid825/integer_parameters.pt")[PREFIX]
    A = consumer["temporal_int16"].copy()
    assert A.shape == (10, 10) and A.dtype == np.int16
    assert consumer["temporal_fractional_bits"] == 14
    assert np.array_equal(consumer["weight_row_scale"].astype(np.float32), source_params["scale"])

    # Verify *all* full-H table values, including the six zero codes. No tables
    # are saved as DUT fixtures; the runtime data-service implementation owns them.
    terms = np.einsum("gdc,hgc->gdch", D.astype(np.int64), Wfull.astype(np.int64).reshape(384, 6, 16))
    table = terms.sum(2)
    ordered_prefix = terms.cumsum(2)
    any_prefix_lo, any_prefix_hi = np.minimum(terms, 0).sum(2), np.maximum(terms, 0).sum(2)
    assert min(int(table.min()), int(any_prefix_lo.min())) >= -512
    assert max(int(table.max()), int(any_prefix_hi.max())) <= 511
    word_count = 0
    for row in table.reshape(6, 16, 4, 96).reshape(-1, 96):
        for chunk in row.reshape(8, 12):
            pack_n12(chunk)
            word_count += 1
    assert word_count == 3072
    nonzero_codes = int((D.sum(2) > 0).sum())
    assert nonzero_codes == 90
    assert np.array_equal(table[:, 0], np.zeros((6, 384), np.int64))

    ylo = np.minimum(Wfull.astype(np.int64), 0).sum(1)
    yhi = np.maximum(Wfull.astype(np.int64), 0).sum(1)
    aa = A.astype(np.int64)
    ulo = np.minimum(aa[:, :, None] * ylo[None, None], aa[:, :, None] * yhi[None, None]).sum(1)
    uhi = np.maximum(aa[:, :, None] * ylo[None, None], aa[:, :, None] * yhi[None, None]).sum(1)
    run = json.loads((CAP / "run.json").read_text())
    frames = run["validation_files"][:2]
    assert frames == ["zurich_city_09_a_0001.npy", "zurich_city_07_a_0001.npy"]
    assert set(frames).isdisjoint(run["train_files"])
    records = []
    fullframe_groups = 0

    def add(name, S, hblock, *, is_real, file="", frame_index=-1, tile_index=-1,
            positive=None, constant=None, constant_gate=None, tau=None):
        hslice = slice(hblock * 96, (hblock + 1) * 96)
        W = Wfull[hslice].T.copy()
        tau = consumer["threshold_int64"][:, hslice].copy() if tau is None else tau
        positive = consumer["positive_gain"][hslice].copy() if positive is None else positive
        constant = consumer["constant_channels"][hslice].copy() if constant is None else constant
        constant_gate = consumer["constant_gate"][:, hslice].copy() if constant_gate is None else constant_gate
        Y, U, gold, stats = calculate(S, W, A, tau, positive, constant, constant_gate, D, dwords)
        if is_real:
            assert stats["exact_groups"] == stats["groups"]
        records.append(dict(case_name=name, S=S.copy(), W=W, tau=tau.astype(np.int64),
            positive_gain=positive.astype(bool), constant_channels=constant.astype(bool),
            constant_gate=constant_gate.astype(bool), Y=Y, U=U, gold=gold,
            is_real=is_real, file=file, validation_index=frame_index, tile_index=tile_index,
            hblock=hblock, stats=stats))

    for frame_index, file in enumerate(frames):
        stem = Path(file).stem
        with np.load(CAP / f"forced_code_{stem}_source.npz", allow_pickle=False) as z:
            assert tuple(z["shape"]) == (10, 19200, 96)
            packed = z["gate_bits"]
            assert packed.shape == (10, 19200, 12) and packed.dtype == np.uint8
            words = np.ascontiguousarray(packed).view("<u2").reshape(10, 19200, 6)
            for g in range(6):
                exact = (words[:, :, g, None] == dwords[g]).sum(-1)
                assert (exact == 1).all()
                fullframe_groups += exact.size
            for tile_index in POSITION_TILES:
                first = tile_index * 32
                selected = packed[:, first:first + 32]
                S = np.unpackbits(selected, axis=-1, bitorder="little").transpose(1, 0, 2).copy()
                assert S.shape == (32, 10, 96)
                assert np.array_equal(np.packbits(S.transpose(1, 0, 2), axis=-1, bitorder="little"), selected)
                for hblock in range(4):
                    add(f"real_v{frame_index}_tile{tile_index}_h{hblock}", S, hblock,
                        is_real=True, file=file, frame_index=frame_index, tile_index=tile_index)
    assert len(records) == 32 and fullframe_groups == 2304000

    # Five explicit synthetic source/protocol cases. D, A and W are still real;
    # only the last case changes threshold/comparison/constant metadata.
    add("diagnostic_zero", np.zeros((P, T, C), np.uint8), 0, is_real=False)
    onehot = np.zeros((P, T, C), np.uint8)
    for p in range(P):
        for t in range(T):
            onehot[p, t, (p * T + t) % C] = 1
    add("diagnostic_onehot_escape", onehot, 0, is_real=False)
    assert records[-1]["stats"]["onehot_escape_groups"] == 320
    escape_masks = []
    for g in range(6):
        escape_masks.append(next(mask for mask in range(1, 65536)
                                 if mask.bit_count() == 2 and mask not in set(map(int, dwords[g]))))
    escaping = np.zeros_like(onehot)
    for p in range(P):
        for t in range(T):
            g = (p + t) % 6
            for k in range(16):
                escaping[p, t, g * 16 + k] = (escape_masks[g] >> k) & 1
    add("diagnostic_multibit_escape", escaping, 0, is_real=False)
    assert records[-1]["stats"]["multi_escape_groups"] == 320
    dense = np.fromfunction(lambda p, t, c: ((p + t) % 3 == 0) | (((p + t) % 3 == 1) & (c % 2 == 0)), (P, T, C), dtype=int).astype(np.uint8)
    add("diagnostic_signed_dense", dense, 0, is_real=False)
    assert records[-1]["U"].min() < 0 < records[-1]["U"].max()
    # Force both comparator senses, equality at p0, and constants that contradict
    # the live comparison. This is a diagnostic metadata contract, not real BN.
    diag_y = dense.astype(np.int64) @ Wfull[:96].T.astype(np.int64)
    diag_u = np.einsum("ts,psh->pth", aa, diag_y)
    pos = (np.arange(H) % 2 == 0)
    const = (np.arange(H) % 8 == 0)
    tau = diag_u[0].copy() + ((np.arange(H) % 3) - 1)[None]
    const_gate = ((np.arange(T)[:, None] + np.arange(H)[None]) % 2 == 0)
    add("diagnostic_negative_gain_constant", dense, 0, is_real=False,
        positive=pos, constant=const, constant_gate=const_gate, tau=tau)
    last = records[-1]
    tie = (last["U"][0] == tau) & ~const[None]
    assert (tie & pos[None]).any() and (tie & ~pos[None]).any()
    assert last["gold"][0][tie].all()  # Both <= and >= must accept equality.
    live_decision = np.where(pos[None, None], diag_u >= tau[None], diag_u <= tau[None])
    assert np.any(live_decision[:, :, const] != const_gate[:, const][None])
    assert np.array_equal(last["gold"][:, :, const], np.broadcast_to(const_gate[:, const], (P, T, int(const.sum()))))

    arrays = {key: np.asarray([r[key] for r in records]) for key in
        ("S", "W", "tau", "positive_gain", "constant_channels", "constant_gate", "Y", "U", "gold",
         "case_name", "is_real", "validation_index", "tile_index", "hblock")}
    arrays["frame_file"] = np.asarray([r["file"] for r in records])
    arrays.update(D=D, A=A, temporal_fractional_bits=np.asarray(14, np.int32),
        theta_source=np.asarray(consumer["theta_source"]), theta_output=np.asarray(consumer["theta_output"]),
        S_packed=np.packbits(arrays["S"].reshape(37, 320, 96), axis=-1, bitorder="little"),
        row_p=np.repeat(np.arange(P, dtype=np.uint8), T), row_t=np.tile(np.arange(T, dtype=np.uint8), P),
        dut_input_fields=np.asarray(["S", "W", "D", "A", "tau", "positive_gain", "constant_channels", "constant_gate"]),
        tb_only_fields=np.asarray(["Y", "U", "gold"]),
        source_boundary=np.asarray("post-projection g_prime; original pre-projection g unavailable in capture"))
    assert arrays["S"].shape == (37, 32, 10, 96)
    assert arrays["W"].shape == (37, 96, 96)
    temporary = HERE / ".cases.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(HERE / "cases.npz")
    with np.load(HERE / "cases.npz", allow_pickle=False) as z:
        for key, value in arrays.items():
            assert np.array_equal(z[key], value), key

    report = ["# Forced-code 真实 FC1→PSN 输入与整数接口", "",
        "固定前2个validation文件×位置tile[0,199,399,599]×Hblock0..3，共32个真实病例；追加5个明确标识的诊断。选点只依据既定索引，没有按活动率筛选。32病例只有8个不同源P32 tile，H384的四块共享同一源，不能称32个独立源样本。", "",
        "来源：`../algorithm/support_training/run.json`、`dictionary.npy`、`forced_code_weight_int8.pt`、`forced_code_parameters.pt` 和两个 `forced_code_*_source.npz`；后级参数来自 `../algorithm/integer_s0_valid825/integer_parameters.pt` 的 `" + PREFIX + "`。", "",
        "已核 `train_exact_support_probe.py::Student.source/quantized_weight`：捕获是强制最近码投影后的g′，未保存投影前g；真实W重建 round(clamp(W_float×theta/scale)) 与saved INT8逐值相等，D与学生state_dict相等。**forced_code_parameters中的A属于前级源PSN，不能误当FC1后的时间A**；导出后级A/tau/flags严格沿`bn_state/support_service_model.py`使用冻结integer consumer参数，不替换它们。", "",
        "DUT边界从原始bit表示的**已投影g′**开始；未实现最近码前级，也不以g′再次恰好命中证明前级编码器已验证。没有提供code ID或PWP表。Y/U/gold只是TB期望值，不能接入DUT数据路径。当前theta_source/output均为1.0仍显式保存，未另外把权重乘一次theta。", "",
        "## 字段与精确算术", "",
        "- `S`: uint8[37,32,10,96]；每病例按[p,t,c]，row=p*10+t。`S_packed`: uint8[37,320,12]，每row原96位little-bit序。",
        "- `W`: int8[37,96,96]，按[c,h]；`D`: uint8[6,16,16]，按[group,code,c_local]，为所有病例共同常量。",
        "- `A`: int16[10,10]，按[t_out,t_in]，共同Q14系数；`tau`: int64[37,10,96]；`positive_gain/constant_channels`: bool[37,96]；`constant_gate`: bool[37,10,96]。",
        "- `Y`: int32[37,32,10,96]；`U`: int64同shape；`gold`: uint8同shape，仅TB。`case_name/is_real/frame_file/validation_index/tile_index/hblock`标识来源，前32real、后5diagnostic。",
        "- 直接Y[p,t,h]=Σc S[p,t,c]W[c,h]，U[p,t,h]=Σs A[t,s]Y[p,s,h]；**不在中间右移14位或RNE**。非constant且positive时U≥tau，negative时U≤tau；constant优先使用对应[t,h]固定gate。这是已有整数学生的阈值合同，不是原始FP32 BN等价声明。", "",
        "## 完整数值与INT10 / N12约束", "",
        f"真实W范围[{int(Wfull.min())},{int(Wfull.max())}]，A范围[{int(A.min())},{int(A.max())}]，tau范围[{int(consumer['threshold_int64'].min())},{int(consumer['threshold_int64'].max())}]。真实384个通道全部positive_gain=true、constant_channels=false；负gain与常量仅在诊断标志中实际覆盖。", "",
        f"全D×W表为[6,16,384]，含零码 **{table.size:,}值**；排除6个零码为90×384={nonzero_codes*384:,}值，不能混淆旧表的34560与全表36864。全值范围[{int(table.min())},{int(table.max())}]；按c顺序build前缀[{int(ordered_prefix.min())},{int(ordered_prefix.max())}]，任意部分子集的安全范围[{int(any_prefix_lo.min())},{int(any_prefix_hi.max())}]，均满足signed10。此为固定真实W/D的admission，任意INT8 W表不自动满足。", "",
        f"已逐值验证3072个128bit物理字的精确pack/unpack：**一个word低120位=12×signed10，高8bit=0 padding**，lane j起始bit=10j，负数按10bit二补码还原。每H96系数行8word=128B；零码也存时384行=49152B，省零码时360行=46080B。每word插入8pad与旧逐整行连续位流内容布局不同，虽字数相同不能沿用旧解码地址。没有把表本体或打包word保存为DUT答案。", "",
        f"对任意二值source，固定W的Y及任意累加前缀包络[{int(ylo.min())},{int(yhi.max())}]；固定A的U及任意时间累加前缀包络[{int(ulo.min())},{int(uhi.max())}]。因Y区间含0，各时间乘积区间也含0，任意前缀包含于该包络。signed13 Y与signed28 U足够，本导出保留Y32/U64；既有Y24/U48硬件合同当然也覆盖，不需要新增舍入。", "",
        "## 固定病例与诊断", "",
        "位置tile j覆盖展平p=[32j,32j+31]；四项分别是[0,31]、[6368,6399]、[12768,12799]、[19168,19199]。两个源文件在run.json顺序分别为zurich_city_09_a_0001.npy、zurich_city_07_a_0001.npy，均不在train_files中；未改字典/训练。", "",
        "| case | g′ 1数/30720 | exact组/1920 | onehot逃逸 | 多项逃逸 | Y范围 | U范围 |", "|---|---:|---:|---:|---:|---|---|"]
    for r in records:
        s = r["stats"]
        report.append(f"| {r['case_name']} | {s['source_ones']} | {s['exact_groups']} | {s['onehot_escape_groups']} | {s['multi_escape_groups']} | {s['Y_range']} | {s['U_range']} |")
    report += ["", "五诊断分别为零源、遍历96列onehot逃逸、各group预定最小非字典2bit码逃逸、交替dense/偶列/空源的signed计算、同dense源的负gain＋常量覆盖。D/A/W均取真实参数；最后一项tau/sign/constant是合成协议参数，不能算成真实BN统计。它让两种比较方向在p0均出现等于tau，同时让部分constant gate与实时比较相反，实际核对常量优先级。", "",
        f"两个完整捕获帧共 **{fullframe_groups:,}个16bit group**逐一验证恰好命中对应D一次；全部32真实病例亦全命中，包括零码。37病例的Y/U/gate各{37*P*T*H:,}值，由NumPy直接点积与独立查表/逐bit逃逸、按s循环时间累加和逐通道符号/常量分支逐值核等；NPZ写回再读逐字段相同。没有新的RTL、GPU、训练或网络质量结果。", "",
        "复现：`/opt/anaconda3/bin/python3.12 prepare_cases.py`。只写本目录cases.npz与CASES.md；NPZ为再生数据，不加入Git。"]
    (HERE / "CASES.md").write_text("\n".join(report) + "\n")
    print(json.dumps(dict(real_cases=32, diagnostic_cases=5, source_shape=list(arrays["S"].shape),
        fullframe_exact_groups=fullframe_groups, Y_U_gold_each=37*P*T*H,
        table_values=int(table.size), table_range=[int(table.min()),int(table.max())],
        Y_bound=[int(ylo.min()),int(yhi.max())], U_bound=[int(ulo.min()),int(uhi.max())],
        archive_bytes=(HERE / "cases.npz").stat().st_size), indent=2))


if __name__ == "__main__":
    main()
