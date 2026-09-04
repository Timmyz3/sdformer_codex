# M1123C：M1122C Path-C common-charge 独立静态打铁

裁决：**GO，但只准入 Path-C 的会计边界，不准入新的性能或物理结果。** M1122C 可以合法地把相同 `214,912 B` 外部容量费用施加给 candidate、strongest-zero 和 same-coordinate-bit；它没有证明 `214,912 B` 已物理实现，也没有把现有 `1.759×` raw CPU 机会升级为 RTL、PPA 或系统加速。

## 独立复算

三轴容量均为 `214,912 B`，预算均为 `245,760 B`，余量均为 `30,848 B`。已知几何只覆盖：

```text
(9 parent + 60 psum + 24 weight) × 2,048 B
= 93 × 2,048 B
= 190,464 B
```

剩余量为：

```text
214,912 B - 190,464 B = 24,448 B
```

这 `24,448 B` 只能标作相同的保守外部容量费用 `[model]`。它不是 live storage、没有实例化、没有已知 macro count，也没有端口、延迟、面积、漏电或动态能耗数字。`ceil(24448/2048)=12` 只是舍入诊断，不能写成“存在 12 个宏”。同样，93 个 macro-equivalent 是容量几何，不是已证明端口、时序、面积和能量的完整 93 宏 top。

## 公平性与禁止双计

Path C 的共同部分是技术、容量组织、端口、延迟、面积/漏电系数和读写能耗系数。实际 address-timed access count 可以因轴不同而不同，必须用共同系数分别收费，不能强制成相同动态能耗。

外部费用已经含 9 个 parent macro-equivalent 时，logic-only top 内允许的 parent macro 实例数必须为 0。旧的 parent-macro-inclusive top 不能直接再加外部费用；只能重新综合零宏 matched top，或先用独立封存的精确贡献做减法。每个 physical/model storage row 的 internal charge 与 external charge 之和必须恰好为一次。

## 未来表格公式

- 面积：`A_total_axis = A_logic_axis + A_ext_common`；
- 时间：必须联合重放 axis logic 与同一 external port/latency model，不能把彼此重叠的周期独立相加；
- 吞吐面积效率：`(work_units/T_axis)/(A_logic_axis+A_ext_common)`；
- 外部动态能量：每轴实际读写数乘相同读写系数；
- 外部漏电能量：共同漏电功率乘该轴联合执行时间；
- 总能量：logic、external dynamic、external leakage 和对称 residual model term 分项相加；
- 倍速：`T_baseline/T_candidate`。

这些公式目前只是未来测量合同。数值外部模型尚未冻结，matched logic DC/PT、joint RTL cycle、SAIF/PTPX、throughput/mm²、power 和 energy 仍全部为空。

## 负向打铁

独立 hammer 完成 247 项检查，拒绝 34 个攻击，包括：伪造 residual macro count；把 residual 写成 live/instantiated；三轴端口、延迟、面积/漏电/动态系数不一致；强制实际 access 相同；旧 top 保留 9 个 parent 宏再外加 9 个；把逻辑结果叫 total；把 `1.759×` 升为 RTL/PPA/system；破坏未来面积、时间或能量公式；重复 JSON key、NaN、Infinity、live extra 和 sealed symlink。

冻结 CPU 数重新核对为 `763,908,050 / 434,242,823 = 1.7591725402×`，但合法标签仍仅是“冻结 H67 四层 bottleneck Conv、十样本、raw CPU same-ledger opportunity”。如果外部端口、调度或延迟改变，必须重新联合重放。

下一步若要执行，必须另建并独立打铁的新合同，冻结一个数值化 external-memory model 和零内部宏的三套 matched logic boundary。M1123C 本身不授权 RTL、filelist/Tcl、DC/PT/SAIF/PTPX、GPU、远端任务或新性能数字。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
