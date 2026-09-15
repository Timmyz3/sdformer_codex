# 真实 R8 Q1：8＋8 条带与一对 full-K 小输出 tile

固定地址选择；没有查看活动率后换点。只生成原始源/权重及确定森林，不保存 PWP、代码本、点积或输出 gold；CPU 点积仅用于本导出脚本断言。

来源为 `../shared_execution_20260915/r8_reference/fixtures/tile_*/source.bin`、`origin.bin` 与 `parameters/param4.bin`。W 另与 `../r8_consumer_fusion_20260914/data/factors.npz:q1` 逐值相等；源 halo 另与该 data 目录的 `first_source_words.npy` 逐词相等。模型身份沿用部署 flatR8，不新增量化、训练或质量结论。

输入为同一 `zurich_city_09_a_0001.npy`、frame_index=0 的不同位置；校准/held 的 tile 身份和**完整有效 4×4 source halo 坐标并集均不相交**。这是同帧空间留出，不是独立序列验证。8 对 K16 地址相同，W 完全相同；每地址目前只有 40 条校准 row，不能夸大代码本统计量。

`calibration_S` / `held_S`: uint8[8,40,16]，`calibration_W` / `held_W`: int8[8,16,8]。`row=p*10+t`，`p=2*py+px`，`K=c*9+ky*3+kx`；各 strip 全部16位原样保留，没有按 W 的 k-live 削源。真实全部 W 范围为 [-3,3]，满足 signed3；没有凭空加入 −4。

同前缀还含 `case_name/tile_id/k16_id/global_k/source_words/input_origin_yx/masks/parents/order/delta_masks/fixture_relative_path`。`source_words` 为未改 C96×4×4 原始低10门字。合并视图 `S[16,40,16]`、`W[16,16,8]`、`case_name` 等按 calibration 在前、held 在后，必须使用 `split` 区分；默认 `allow_pickle=False` 可读。

确定森林：popcount<2 不设父；其余取非零最大子集，平局选最小原 row index；相同 mask 只能取更早 index。稳定顺序为 `(popcount,index)`，parent=-1 为 root。该规则匹配已读本地作者 kernel 的 first-index 口径，区别于论文并列时最大 index；本轮只固定一种，不择优。`delta_masks=mask XOR parent_mask`；森林只读原始 S，不读 W 或 held 校准信息。

本导出不生成 Phi 代码本，后续 prepare_runs.py 单独完成校准：每个 K16 只允许对应 calibration 数据参与选中心；Phi-alone 可用原始 S 独立校准，固定森林臂可用 root/delta 校准，同 q/样本预算。held 数据只供运行时精确查询，不选最佳离线中心。禁止把 CPU 断言中的乘积作为 RTL 配置。

| split | tile | K16编号 | 起始K | 原始1数/640 | 非零行/40 | 有父行 | 多项root/delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| calibration | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration | 159 | 7 | 112 | 14 | 9 | 3 | 1 |
| calibration | 19040 | 15 | 240 | 0 | 0 | 0 | 0 |
| calibration | 19199 | 23 | 368 | 1 | 1 | 0 | 0 |
| calibration | 2584 | 30 | 480 | 2 | 2 | 0 | 0 |
| calibration | 4840 | 38 | 608 | 0 | 0 | 0 | 0 |
| calibration | 9680 | 46 | 736 | 74 | 19 | 7 | 6 |
| calibration | 14520 | 53 | 848 | 22 | 13 | 6 | 1 |
| held | 128 | 0 | 0 | 0 | 0 | 0 | 0 |
| held | 136 | 7 | 112 | 7 | 6 | 1 | 0 |
| held | 144 | 15 | 240 | 1 | 1 | 0 | 0 |
| held | 152 | 23 | 368 | 3 | 3 | 0 | 0 |
| held | 4000 | 30 | 480 | 27 | 13 | 8 | 0 |
| held | 4016 | 38 | 608 | 24 | 16 | 8 | 0 |
| held | 4032 | 46 | 736 | 9 | 7 | 1 | 1 |
| held | 4048 | 53 | 848 | 9 | 6 | 2 | 0 |

## 新增 full-K 覆盖：全部 54 个 K16，保留原小集

固定校准 tile4840 / held tile4016，按既有配对列表选择首个双方4×4 halo均在图内的几何配对，选择不检查活动率。两侧 halo 不交；小集全部旧字段逐值保持。这里新增的是 **P4×T10×N8 的一个小输出 tile、完整 K864**，不是完整空间层、全网、64 tile 测试，也没有新增独立帧。

`full_calibration_S` / `full_held_S` 为 uint8[54,40,16]，对应 W 为 int8[54,16,8]；同前缀提供 case_name、tile_id、k16_id、global_k、source_words、input_origin_yx、masks、parents、order、delta_masks、fixture_relative_path。编号0..53全部保留，未按活动筛选。两侧同地址 W 完全相等；每 K 的校准仍只有该位置40条row，held 不参与选中心。

下表分母均为54×40=2160个 **row/K分区实例**；多项指该root或parent-delta的popcount≥2，零项也包含零源root和EM空delta。这个统计不是可省周期或可省W字节。

| split / tile | 原始1数/34560 | 原始多项row | 有父row | 多项root | 多项parent-delta | 多项总数/2160 | 有多项的K16数/54 | 剩余term数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_calibration / 4840 | 384 | 90 | 80 | 10 | 20 | 30 | 16 | 257 |
| full_held / 4016 | 1876 | 489 | 440 | 49 | 90 | 139 | 38 | 1018 |

完整popcount直方图，数组index=0..16；保留两侧所有空条带：

- `full_calibration` 原始 S：`[1929, 141, 49, 27, 7, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`；root/delta：`[1944, 186, 19, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`。
- `full_held` 原始 S：`[1225, 446, 259, 111, 63, 31, 12, 6, 5, 2, 0, 0, 0, 0, 0, 0, 0]`；root/delta：`[1320, 701, 108, 23, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`。

导出自检通过：按124条带共进行了 190,464 次原始 halo word 比较（包含full-K重复读同一tile）；79,360 次独立绝对坐标门位比较；全部124条带的固定森林局部 N8 重构；16个不同tile的 5,120 个完整 Q1 latent 与旧 fixture gold。新增两侧54条带的结果逐K相加，另核对640个完整Q1输出值。NPZ写回读取逐字段相同；旧小集全部非full字段逐值不变。

此处给的是已展开 K16 叶输入；原始 source 的存取/im2col、检测/排序仍须在完整系统比较中收费，不能以这份 NPZ 视为免费前端。`cases.npz` 为再生本地数据，不加入 Git。

复现：`/opt/anaconda3/bin/python3.12 prepare_cases.py`。脚本只写本目录 `cases.npz` 和 `CASES.md`，不改旧 fixture。
