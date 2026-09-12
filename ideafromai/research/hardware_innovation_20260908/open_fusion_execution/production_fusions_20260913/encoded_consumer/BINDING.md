# 旧 R24 导出绑定真实前段与消费者

`binding.py` 只克隆 `stage_20260912/hardware/consumer_ranked.py::run` 的完整局部消费者，替换 anchor 的 U/V 块。原始 I24 输入、源门程序、完整 K864 preview/sn2、完整 K864 Conv2/F/merge、projection 门、PED SRAM 外送和 projection 门外送均实际执行。未包括 native projection、global BN 或 join。

## API

```python
data, live, q, params = binding.load(axis, label)
preview, producer_report, machine = binding.run_prefix(
    machine_class, data, live, axis, label, stress=False)
value, consumer_report = binding.consumer_run(
    data, q, label, callback, machine=machine, stress=False,
    gold=None, handoff_native=False)
```

`axis` 为 `ordinary` 或 `lifting_raw`，`label` 为 `corner` 或 `interior`。`data/live/q` 分别读取旧 R24+onepass 的 `stage_20260912/algorithm/hardware_exports/{axis}/000_zurich_city_09_a_0001.npz`、`live_parameters.npz`、`deployed_constants.npz`。`params` 仅包含 `fixed_q8/affine_q8/full_g_q8`，每项 `{D,c,step}` 是 `breadth_20260912/representation/parameters/{axis}.npz` 原整数参数，不拟合。

回调签名为 `callback(m, base, q, positions, geo) -> (actual_u, actual_ped)`，数组形状为 `(count,10,24)` 和 `(count,10,96)`，`count<=2`；位置顺序与 SRAM 的 P/T/C 顺序一致。回调进入时真实 UPDATED 和该位置的真实 projection gate 已写好。回调必须把实际 PED 写到 `PED_V`；绑定核验返回的 PED 与 SRAM 相同，再经原 `read_word` 和 DMA 输出槽外送。U 可从真实 RF 捕获，不强制落盘。`m.encoded_u_ready` 可由回调给出有定义的 U 完成时刻，否则行报告写 `null`，不捏造 P1 交错路径的聚合 U 完成点。

`gold` 只接受最终检查用的 `{'continuous': ..., 'U_ped': ...}`，`U_ped` 可省略；它不传入回调，且只能在执行结束后参与比较。`updated` 与 `gate` 始终检查旧导出。未给 `gold` 时检查旧 R24 原 PED，并在结束后独立计算原 U 参考。输出 `value` 键与原消费者相同：`updated/continuous/gate/U_ped/PED_spill`。

## 已核对的父参数

两父各 34 项 live 参数与原 `FULL/capture/{axis}` 逐项相同。部署常量仅 `U_ped_q16/V_ped_q16` 与旧 R32 不同；源、preview、Conv2、门和所有原尺度常量相同。窗口 geometry、I24、sn1、preview Z/raw/BN1、sn2、updated 和 projection gate 也逐项相同。因此原 `two_stage_writeback/{axis}_fused_program.json` 仍对应这两套旧 R24 导出，实际源程序执行另要求整数门 0diff。

preview 的历史 FP32 raw/BN1 相对 CUDA 捕获差分仍按原统计核验；仅 `rms` 汇总容许 NumPy 运行时末位至多 2ULP，不允许改变差分计数、最大误差或整数端点。

## 地址与寿命

| 空间 | 地址与约束 |
|---|---|
| 原 consumer coefficient | 0..41023；`U_conv2_theta=0,F=27648,U_ped=30720,V_ped=35328,BN2_constant=39936,PED_bias=40320,compare=40704` |
| 新 coefficient | 回调懒加载至 >=65536，低地址原镜像和 base 不改；总池 131072B |
| 原 state | `I=0,DIR=8192,LAT16=16384,UPDATED=20480,PED_U=32768,PED_V=40960,GATE=49152,PROJ=65536` |
| PROJ 最大范围 | 当前 interior 的 9×9×96×2B，止于 81088；原包装器 high-water 只描述原缓冲，回调另报 scratch 最大地址 |
| source 临时输入 | `90112`，prefix 结束后已死；>=98304 可供本阶段 callback scratch，总池仍 131072B |
| RF | 96×8×48；原 sparse-U/F/projection 会重用 RF0..79，不能把回调 Uc 缓存假设为跨 callback 永活；既有 gather 为 RF88..91 |

系数冷 fill 可持久复用，RF 内容需按实际寿命恢复。普通原 P1 smoke 使用已有 `ordinary_dense`，分别 count=1 执行 U/V 并落盘 U；它只证明绑定，不作为新 kernel 的 P1 keep-Z 性能分母。

## 一次固定 smoke

命令：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.12 binding.py --smoke`。

固定 ordinary/interior/ready，完整真实前段后同 Machine 接 16 个 anchor 的原 P1 U/V，已完成；未扩窗口、压力或参数配置。

| 测量 | 结果 |
|---|---:|
| 真实 producer | 2,126,474 slots |
| 原 P1 consumer | 636,418 slots |
| 同 Machine 合计 | 2,762,892 slots |
| SR64 字节 | 4,064,904 |
| SW64 字节 | 1,362,504 |
| CR256 字节 | 4,224,864 |
| CW256 字节 | 165,568 |
| 源门 | 116,160 值，0diff |
| preview Z / sn2 | 25,920 / 77,760 值，均 0diff |
| updated / projection gate | 各 77,760 值，均 0diff |
| U / PED | 3,840 / 15,360 值，均 0diff |

174,720 个消费者整数端点全精确，真实 PED SRAM 外送字节与位置顺序复核通过，sn2 无外送/重入。BN1 RMS 为 `4.7971686718922096e-05`，历史汇总为 `4.79716867189221e-05`，相差 1ULP；其余历史差分统计逐项相同。该运行不证明压缩接口性能，后续 kernel 必须用共同 P1 keep-Z 权限的同函数控制。
