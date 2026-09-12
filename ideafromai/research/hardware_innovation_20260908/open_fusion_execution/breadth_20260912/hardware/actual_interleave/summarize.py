"""Summarize only the existing fixed-P2 16 ready and six stress results."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
BREADTH = HERE.parents[1]
AXES = ("dense_twopot", "lifting_twopot", "contiguous34")
LABELS = dict(dense_twopot="dense 两项", lifting_twopot="lifting 两项", contiguous34="matched34")
BASE_MODES = ("serial_CSE_P2", "serial_CSE_P1_Z", "serial_same_CSE", "joint_CSE")
LOW_MODES = ("serial_same_low_CSD", "joint_low_CSD")


def read(path):
    return json.loads(path.read_text())


def traffic(counts):
    sr, sw, cr = (counts.get(key, 0) for key in ("SR64_reads", "SW64_writes", "CR256_reads"))
    dma_in, dma_out = (counts.get(key, 0) for key in ("DMA_input_slots", "PED_DMA_output_slots"))
    assert dma_in % 5 == dma_out % 5 == 0
    return dict(SR64_reads=sr, SW64_writes=sw, CR256_reads=cr,
                SR_bytes=sr * 8, SW_bytes=sw * 8, CR_bytes=cr * 32,
                CW256_writes=counts.get("CW256_writes", 0),
                coefficient_fill_bytes=counts.get("CW256_writes", 0) * 32,
                external_input_bytes=dma_in // 5 * 32,
                external_output_bytes=dma_out // 5 * 32,
                external_total_bytes=(dma_in + dma_out) // 5 * 32,
                DMA_input_slots=dma_in, DMA_output_slots=dma_out,
                source_ROM128_fetches=counts.get("source_ROM128_fetches", 0),
                source_ROM_fetch_bytes=counts.get("source_ROM128_fetches", 0) * 16)


def case(axis, mode, stress=False):
    name = axis + "_" + mode + ("_stress" if stress else "") + ".json"
    raw = read(HERE / name)
    prefix = raw["common_prefix"]
    checks = dict(A_source=prefix["source"], **prefix["source_and_preview"],
                  A_updated=dict(values=prefix["updated_values"], differences=prefix["updated_differences"]),
                  A_projection_gate=dict(values=prefix["projection_gate_values"], differences=prefix["projection_gate_differences"]),
                  **raw["checks"])
    assert all(check["differences"] == 0 for check in checks.values())
    assert raw["service_slots"] == prefix["prefix_slots"] + raw["tail_slots"] == sum(raw["stages"].values())
    full, tail = traffic(raw["counts"]), traffic(raw["tail_counts"])
    assert tail["external_input_bytes"] == 348480 and tail["external_output_bytes"] == 5760
    assert raw["tail_counts"]["input_store"] == 43560
    assert raw["tail_counts"]["source_gate_word_store"] == 2904
    assert tail["source_ROM128_fetches"] == 121 * 12 * raw["source_program_words"]
    assert raw["combined_source_ROM_words"] <= raw["resource"]["source_ROM_words"]
    assert raw["SRAM_high_water"] <= raw["resource"]["state_bytes"]
    return dict(axis=axis, mode=mode, condition="stress" if stress else "ready", source=name,
                service_slots=raw["service_slots"], prefix_slots=prefix["prefix_slots"], tail_slots=raw["tail_slots"],
                full_traffic=full, tail_traffic=tail,
                prefix_traffic={key: full[key] - tail[key] for key in full},
                source_work_RF=raw["source_work_RF"], source_offset=raw["source_offset"],
                PED_hblock=raw["PED_hblock"], source_program_words=raw["source_program_words"],
                combined_source_ROM_words=raw["combined_source_ROM_words"],
                SRAM_high_water=raw["SRAM_high_water"], resource=raw["resource"], checks=checks,
                checked_values=sum(row["values"] for key, row in checks.items() if key != "A_PED_actual_DMA_bytes"),
                checked_external_bytes=checks["A_PED_actual_DMA_bytes"]["values"],
                scheduling=raw["scheduling"], stages=raw["stages"])


def comparison(serial, joint):
    assert serial["axis"] == joint["axis"] and serial["prefix_slots"] == joint["prefix_slots"]
    saved = serial["service_slots"] - joint["service_slots"]
    return dict(axis=serial["axis"], condition=serial["condition"],
                serial_mode=serial["mode"], joint_mode=joint["mode"],
                serial_source=serial["source"], joint_source=joint["source"],
                prefix_slots=serial["prefix_slots"], serial_service_slots=serial["service_slots"],
                joint_service_slots=joint["service_slots"], serial_tail_slots=serial["tail_slots"],
                joint_tail_slots=joint["tail_slots"], saved_slots=saved,
                full_reduction=saved / serial["service_slots"], tail_reduction=saved / serial["tail_slots"],
                joint_minus_serial_tail_traffic={key: joint["tail_traffic"][key] - serial["tail_traffic"][key]
                                               for key in joint["tail_traffic"]})


def render(report):
    lines = [
        "# 真实共 RF 交织：固定 A-PED P2 + 完整 B 源 halo",
        "",
        "16 个 ready 与既定 6 个 stress 结果均完成。相对同函数最强已测串行 `serial_CSE_P1_Z`，`joint_CSE` 的含前缀服务减少 **0.277–0.589%（ready）/ 0.290–0.638%（stress）**。这是当前工作组合的有限收益；既没有证明系统倍率，也不能凭此裁决整个交织家族。",
        "",
        "共同前缀实际执行 A=corner 的完整源 halo、K864 preview、非因果 sn2，以及首对 anchor `(0,0)`、`(0,2)` 的 Conv2/merge/投影门。尾段才将这一 P2 的完整 PED U24/V96 与同帧 B=interior 的完整源 halo 交织。终点包含 116,160 个 B 门位、1,920 个 A PED signed24 值及其 5,760 字节实际外送；不含 A 其余整数位置、B 后续 preview、完整双窗口、全域 BN/native/join 或整网。",
        "",
        "**同函数最强已测串行与交织。** 全服务为共同前缀加尾段；减少比例分别以串行全服务和串行尾段为分母。stress 固定为每 32 槽 SR 末 8 槽、SW 末 4 槽阻塞，沿用 ready 选定的两项控制，没有在压力结果上再选型。",
        "",
        "| 条件 / 函数 | 共同前缀 | 串行全服务 | 交织全服务 | 少用槽数 | 全服务减少 | 串行尾段 | 交织尾段 | 尾段减少 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["strongest_comparisons"]:
        lines.append("| {} / {} | {:,} | {:,} | {:,} | {:,} | {:.3%} | {:,} | {:,} | {:.3%} |".format(
            row["condition"], LABELS[row["axis"]], row["prefix_slots"], row["serial_service_slots"],
            row["joint_service_slots"], row["saved_slots"], row["full_reduction"],
            row["serial_tail_slots"], row["joint_tail_slots"], row["tail_reduction"]))
    lines += ["", "**同布局串行消融与普通 13RF 控制。** `serial_same_*` 与对应 `joint_*` 使用相同源程序、RF 映射、PED 分块及两 RF 供数，SR/SW/CR 和源 ROM 访问次数相同；此处隔离的是调度。CSD 的低源活跃集同时带来更多源程序工作。", "",
              "| 函数 / 源程序 | 同布局串行全服务 | 交织全服务 | 同布局减少 | 同布局尾段减少 | 交织相对最强 CSE 串行增减 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in report["same_layout_comparisons"]:
        lines.append("| {} / {} | {:,} | {:,} | {:.3%} | {:.3%} | {:+.3%} |".format(
            LABELS[row["axis"]], row["program"], row["serial_service_slots"], row["joint_service_slots"],
            row["full_reduction"], row["tail_reduction"], row["joint_change_vs_strongest_serial"]))
    lines += ["", "dense/34 的 13RF 两链 CSD 都能从同布局交织获得部分回收，但本次全服务仍慢于各自最快 CSE 串行；低 RF 本身不足以证明净服务收益。原 P2/H32 四 RF 供数串行也已实际执行，全部 ready 项如下。SR/SW 是 64 位事务，CR 是 256 位事务，单元格为**全服务 / 尾段**次数；stress 对应项的访问量与 ready 相同。", "",
              "| 函数 | 模式 | 全服务 / 尾段槽数 | SR64 | SW64 | CR256 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in report["cases"]:
        if row["condition"] != "ready":
            continue
        f, t = row["full_traffic"], row["tail_traffic"]
        lines.append("| {} | [{}]({}) | {:,} / {:,} | {:,} / {:,} | {:,} / {:,} | {:,} / {:,} |".format(
            LABELS[row["axis"]], row["mode"], row["source"], row["service_slots"], row["tail_slots"],
            f["SR64_reads"], t["SR64_reads"], f["SW64_writes"], t["SW64_writes"], f["CR256_reads"], t["CR256_reads"]))
    lines += ["", "相对最强 P1 留 Z 串行，CSE 交织的尾段 SR 分别多 6,000 / 1,800 / 2,400 次，SW 均多 180 次，CR 分别少 96 / 多 48 / 多 144 次（dense / lifting / 34）。槽数收益没有自动变成更少的总端口访问。逐项字节换算、前缀计数和源 ROM 取指已写入 [summary.json](summary.json)。", "",
              "**外部传输与内部存储分开计数。** 所有 22 项的外部载荷相同：", "",
              "| 边界 | 外部输入字节 | 外部输出字节 | 合计 |",
              "|---|---:|---:|---:|"]
    for label, key in (("共同前缀", "prefix"), ("尾段", "tail"), ("含前缀全服务", "full")):
        row = report["external_bytes"][key]
        lines.append("| {} | {:,} | {:,} | {:,} |".format(label, row["input"], row["output"], row["total"]))
    lines += ["", "全外部输入中，165,568 字节用于 preview/整数系数填充与替换（5,174 次 CW256），其余 587,520 字节为状态输入；尾段 B 输入 348,480 字节对应 54,450 个 DMA 槽及 43,560 次 SW64。B 门的 23,232 字节计为 2,904 次内部 SW64，不是外部输出。A 的 5,760 字节真实外送对应 900 个 DMA 槽。每个 32B DMA 事务为五槽；端口字节不等于外部载荷，重复读取照常计数。", "",
              "**共同资源与合法映射。** 单 96×8×48 RF、单 ready/pending 与 issue，SR64/SW64/CR256，状态和系数各 128KiB，512×128 位源 ROM。所有结果状态地址高水位为 121,536 字节，未缩小物理资源。交织布局如下；两个 CSE 串行强控制可复用原址源程序。", "",
              "| 函数 / 源程序 | 源工作 RF | 源 RF 区间 | PED 块 | B 程序字 | A+B ROM 字 |",
              "|---|---:|---|---:|---:|---:|"]
    for row in report["cases"]:
        if row["condition"] != "ready" or not row["mode"].startswith("joint_"):
            continue
        lines.append("| {} / {} | {} | {}–{} | H{} | {} | {} |".format(
            LABELS[row["axis"]], row["mode"][6:],
            row["source_work_RF"], row["source_offset"], row["source_offset"] + row["source_work_RF"] - 1,
            row["PED_hblock"], row["source_program_words"], row["combined_source_ROM_words"]))
    lines += ["", "源门固定 RF95，PED 标量供数 RF93/94。普通 P1 控制保留 30 个 U/Z RF，再以 RF30–89 执行 V/H48；交织 PED 使用低地址输出区间。源 ROM 尾段实际取指为 `121×12×B程序字数`，低 CSD 的额外程序字按实际取指收费。共享 64B 分阶段暂存的已审阅有效载荷上界为 59B；SR/CR 仍各单响应，gather24、标量及源门 collector 沿用既定状态。", "",
              "共享 `DMA32` 锁覆盖输入的五个 DMA 槽到四次 SW，以及输出的四次 SR 到五个 DMA 槽，避免隐含第二缓冲。就绪 issue 使用固定 round-robin，但**锁释放后可被同一流立即重获，不保证 32B DMA 事务公平**。当前只有 PED 访问 CR；这不是任意两流的通用 CR 调度器。", "",
              "**共同数值检查。** 22 项均逐项通过 A 源、preview Z/raw/BN1/sn2、P2 updated/投影门、完整 B 源门、P2 PED 与实际 DMA 外送。每项检查 404,480 个数值/门位及另列的 5,760 外送字节，所有 differences=0；重复配置不新增独立窗口样本。原 RNE/signed24 饱和、PED V 后偏置再饱和以及 lifting 的逐阶段 norm 保留。代码审阅、边界修补和独立争用/饱和小测见 [REVIEW.md](REVIEW.md)，不把这些 CPU 载荷检查称为新增交织 RTL 或 PPA。", "",
              "**各函数独立质量证据。** 以下为各自实际 fresh valid825，均为 825 帧 / 48,152,523 有效像素；NB0 帧均 1.445352534681，判据仅为严格更低。质量与本地服务并列，不相乘。", "",
              "| 函数 | 825 帧均 AEE | 像素加权 AEE | 相对 NB0 |",
              "|---|---:|---:|---:|"]
    for row in report["quality825"]:
        lines.append("| [{}]({}) | {:.9f} | {:.9f} | {:+.9f} |".format(
            LABELS[row["axis"]], row["source"], row["AEE_frame_mean"], row["AEE_pixel_mean"], row["delta_NB0"]))
    lines += ["", "dense/lifting 的两项新函数没有继承父 825：分别比自身未量化父增加 0.000724059 / 0.009748630。lifting 的实际 `lift2b` 饱和 12 项已包含在该 AEE 中；AT-LIF `{0,theta}` 的固定幅度仍可静态折权。NB0 保留原最终头，新函数使用既定粗头协议，区别见各质量说明。", "",
              "三种函数的源/消费者参数与活动不同，跨函数绝对槽数差不能全部归因于交织或 lifting 结构。当前只覆盖第一对 A-PED anchor 对完整 B 源 halo 这一固定工作量比例，以及一个固定周期背压形状；未搜索分块/队列或声称最优调度，也不能据此否定其他交织边界。完整 A 窗口边界属于另项工作。", "",
              "数据来自本目录既有 22 个结果；[summarize.py](summarize.py) 仅读取并重建本 [summary.json](summary.json) 和 README，不训练、不运行载荷或 GPU。"]
    return "\n".join(lines) + "\n"


def main():
    rows = [case(axis, mode) for axis in AXES
            for mode in BASE_MODES + (() if axis == "lifting_twopot" else LOW_MODES)]
    rows += [case(axis, mode, True) for axis in AXES for mode in ("serial_CSE_P1_Z", "joint_CSE")]
    assert len(rows) == 22
    by_key = {(row["axis"], row["mode"], row["condition"]): row for row in rows}
    strongest, ablations = [], []
    for axis in AXES:
        ready = [row for row in rows if row["axis"] == axis and row["condition"] == "ready"]
        serial = min((row for row in ready if row["mode"].startswith("serial_")), key=lambda row: row["service_slots"])
        joint = min((row for row in ready if row["mode"].startswith("joint_")), key=lambda row: row["service_slots"])
        assert serial["mode"] == "serial_CSE_P1_Z" and joint["mode"] == "joint_CSE"
        for condition in ("ready", "stress"):
            s, j = (by_key[axis, mode, condition] for mode in (serial["mode"], joint["mode"]))
            strongest.append(comparison(s, j))
            if condition == "stress":
                for mode in (serial["mode"], joint["mode"]):
                    assert by_key[axis, mode, condition]["full_traffic"] == by_key[axis, mode, "ready"]["full_traffic"]
        for program in ("CSE",) + (() if axis == "lifting_twopot" else ("low_CSD",)):
            s, j = (by_key[axis, mode + program, "ready"] for mode in ("serial_same_", "joint_"))
            for key in ("source_work_RF", "source_offset", "PED_hblock", "source_program_words", "combined_source_ROM_words"):
                assert s[key] == j[key]
            assert s["full_traffic"] == j["full_traffic"]
            row = comparison(s, j)
            row.update(program=program, joint_change_vs_strongest_serial=j["service_slots"] / serial["service_slots"] - 1)
            ablations.append(row)
    strongest.sort(key=lambda row: (row["condition"] == "stress", AXES.index(row["axis"])))
    quality = []
    for axis in AXES:
        structure = dict(dense_twopot="dense", lifting_twopot="lifting40", contiguous34="contiguous34")[axis]
        directory = "valid825" if axis == "contiguous34" else "source_constant_valid825"
        path = BREADTH / "algorithm" / directory / structure / "quality.json"
        q = read(path)
        assert q["complete"] and q["same_frame_set"] and q["same_per_frame_valid_pixels"]
        assert q["summary"]["frames"] == 825 and q["summary"]["valid_pixels"] == 48152523
        quality.append(dict(axis=axis, source="../../algorithm/{}/{}/quality.json".format(directory, structure),
                            AEE_frame_mean=q["summary"]["AEE_frame_mean"], AEE_pixel_mean=q["summary"]["AEE_pixel_mean"],
                            NB0_AEE=q["NB0_AEE"], delta_NB0=q["delta_NB0"], better_than_NB0=q["better_than_NB0"]))
    external = {}
    for part in ("prefix", "tail", "full"):
        key = part + "_traffic"
        triples = {(row[key]["external_input_bytes"], row[key]["external_output_bytes"], row[key]["external_total_bytes"]) for row in rows}
        assert len(triples) == 1
        inp, out, total = triples.pop()
        external[part] = dict(input=inp, output=out, total=total)
    report = dict(
        complete=True, ready_cases=16, stress_cases=6,
        scope="Actual A corner full source/preview prefix, first A anchor P2 Conv2/merge/gate and full PED, plus complete B interior source halo only.",
        fixed_positions=[[0, 0], [0, 2]], new_experiments=False, new_training=False, new_GPU=False, new_RTL=False, new_PPA=False,
        strongest_comparisons=strongest, same_layout_comparisons=ablations, cases=rows, external_bytes=external,
        resource=rows[0]["resource"], SRAM_high_water=121536,
        shared_staging=dict(capacity_bytes=64, reviewed_payload_upper_bound_bytes=59, source_gate_RF=95, PED_input_RF=[93, 94]),
        numerical_checks=dict(all_zero_differences=True, cases=22, per_case_values=rows[0]["checked_values"],
                              per_case_external_bytes=5760, repeated_cases_are_not_independent_inputs=True, review="REVIEW.md"),
        quality825=quality, quality_and_service_are_not_multiplied=True,
        limits=["Fixed first A-PED P2 plus whole B halo, not full A or two-window inference.",
                "Same-function service comparisons; cross-function activity and trained consumers differ.",
                "Fixed ready round-robin and one periodic stress shape; DMA32 lock reacquisition is not transaction-fair.",
                "Only the PED flow uses CR; this is not an arbitrary two-flow CR scheduler.",
                "Low-RF results neither prove a physical state lower bound nor kill the interleaving family.",
                "CPU payload service only, without new interleaver RTL, PPA, or system throughput."])
    (HERE / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "README.md").write_text(render(report))
    print("Summarized {} ready + {} stress cases; all existing endpoint checks have zero differences.".format(report["ready_cases"], report["stress_cases"]))


if __name__ == "__main__":
    main()
