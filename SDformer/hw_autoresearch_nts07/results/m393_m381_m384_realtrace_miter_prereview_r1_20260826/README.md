# M393：M381↔M384 真实 trace 性能闭环预评审

结论：**现在应立即做冻结真实 trace 的 controller↔cycle-simulator miter**，但
当前 `1.0763828768x` 仍不能称为 RTL 实测性能。评分 78/100，
P0/P1/P2 = 0/4/3；四个 P1 是“性能准入阻塞”，不是说现有证据造假。

M381 与 M384 的功能边界是对得上的：zero-only compaction、strict PWP rule、
48-bit canonical descriptor、maximal direct-address run、L1..8/D8 credit、tagged
tile1 prefetch 和双 replay 都已实现验证。真正没闭环的是周期：

- M384 只跑过 4 个 synthetic phase，没有跑支撑 M381 的 17,280 个真实 phase；
- q32 matcher、pattern/tile0 DMA、descriptor/PWP SRAM 和 O4 backend 都在 RTL 外；
- M384 没有 tile0-payload-done 输入，环境必须保证 tile0 完成后才启动 replay0；
- M381 的 one-cycle seal、two-cycle tail、tile1 overlap 尚未与实际 FSM 逐拍对账；
- M387/M391 的 3 ns 只证明 controller logic 在 ideal-clock/no-SPEF/zero-macro
  边界可行，不能替 matcher/DMA/SRAM/backend 定频。

本目录 JSON 已给出最小可执行合同：使用 exact M248 trace 与 q32 catalog，
由独立 generator 逐相生成 51.84M rows，在 Synopsys VCS 中驱动未修改的 M384；
外部模型冻结为 cmd32、32 B/cycle、SRAM L8/II1/D8，并按
fallback=`4*popcount`、PWP=`4*distance+8` 驱动 backend ready。必须完成
17,280 phases、34,560 replays、42,943,778 次双 replay descriptor 事务，并把
每一个 M381↔RTL cycle delta 归入命名组件。

执行后的门很简单：零结构/数值/event mismatch、零未归因周期，且重新计算的
四层 Conv 模块 speedup 仍需 `>=1.05x`。若实际总周期不是 505,195,832，应封存
新结果并 supersede 旧数，不能调参抹平。无论结果如何，都不得升级为 system
speedup、DATE headline、energy、physical SRAM 或 paper PPA。

`docs/359` 与所有既有证据均未修改。
