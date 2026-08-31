# M459 对 M458 exact destination multicast 的独立打铁结论

评分 **94/100**，P0=0、P1=2、P2=2。结论同意 M458：**B>1 destination multicast RTL 全部 NO-GO，保留 B1 M430。**

独立审计没有导入或调用 M458 analyzer。它从冻结 M40 packed support 和 M430 q32 catalog 重建了 40 records、17,280 phases、51.84M destination；九个 M430 sealed phase 字段、十二个 per-phase B 字段、全部 per-B 字段均为 0 mismatch。独立 group ledger 对 M458 做了 108 个 scalar 映射检查，group/issue/full/tail/capacity/waste/utilization/remainder occupancy 全部 0 mismatch。

## 公平性能

| B | strong-zero cycle | catalog cycle | equal-B catalog/zero | 相对 B1 目录优势 | zero 自身加速 | catalog 自身加速 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 742,148,386 | 517,041,352 | 1.435375x | 1.000000x | 1.000x | 1.000x |
| 2 | 372,144,211 | 303,343,776 | 1.226807x | 0.854694x | 1.994x | 1.704x |
| 4 | 187,306,040 | 196,536,768 | 0.953033x | 0.663961x | 3.962x | 2.631x |
| 8 | 97,676,162 | 143,835,128 | 0.679084x | 0.473106x | 7.598x | 3.595x |

目录单独看确实随 B 加速，但 strong zero 获得相同 accumulator banks/update ports 后加速更快。equal-B 目录优势从 1.435x 严格降到 1.227x、0.953x、0.679x；B4/B8 已直接输给 equal-B zero。这个 NO-GO 在把新增 B 路硬件面积视为免费的情况下仍成立，不依赖 `throughput/B` proxy。

## exact 与 tail 守恒

每个 group 严格在 `sample/operator/partition/output-block` 内，输出块不融合。per-block useful contribution 对所有 B 不变：zero 92,640,472；PWP 15,909,646；correction 38,055,489。其中 plus 33,054,253、minus 5,001,236 独立成组且精确相加。

B2/B4/B8 的 catalog wasted slots 分别为 540,741 / 1,631,261 / 3,836,097；PWP utilization 从 98.30% 降到 95.06%/89.16%，correction 从 99.31% 降到 97.93%/95.24%。所有 `ceil(n/B)`、tail occupancy、capacity=useful+waste 均通过，destination contribution、PWP reconstruction、plus/minus overlap 与 old_psum update-count mismatch 全为 0。总计检查 414,720,000 个 destination/output-block context。

## 资源与证据边界

`throughput/B` 的 0.852/0.658/0.449 只是悲观 proxy，不能叫 throughput/area。真实新增代价包括 B 路 accumulator bank/update port、gather/list storage 与 broadcast/crossbar；目前没有 matched DC/PTPX 或物理 SRAM/interconnect。但即使完全忽略这些代价，equal-B 结果也已经足够关停 B>1。

M458 原始 marker 和 80-file exact-once audit 有效。M459 R1/R2 因评审器输出字段问题 fail-closed，各自保留 `DO_NOT_CITE`；R3 在 raw 前先通过 two-row CSV/JSON/double-seal micro，再由独立 recovery contract 授权最终只读 pass。三轮 recovery 没有改变 grouping、cycle、B、catalog 或 decision 公式。machine result 中继承的 `second_pass` 字段名仅是旧标签，不能解释成整个评审历史只额外读取一次。

M430 train/heldout、M435、M458 双封均 0 mismatch；M458 目录、M430 catalog 未改；`docs/359` 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

