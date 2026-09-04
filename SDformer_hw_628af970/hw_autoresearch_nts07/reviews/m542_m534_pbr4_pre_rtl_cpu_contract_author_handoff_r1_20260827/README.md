# M542｜M534/PBR4 pre-RTL CPU 合同 author handoff r1

日期：2026-08-27  
状态：`SOURCE_ONLY_PRE_RTL_CPU_CONTRACT_COMPLETE__FRESH_STATIC_HAMMER_REQUIRED__NO_RUN_AUTHORIZED`

本里程碑只创建一份 `run_authorized=false` 的 CPU execution contract 和一份 future runner schema；没有
创建或运行 analyzer，没有创建结果目录，没有修改 RTL、M534 r1--r4、`docs/524` 或 `docs/359`。

合同冻结四个同资源点 `A1-SC8/A1-ISO8/A1-OSG/PBR4`。三种 A1 必须在 PBR4 不可见时先跑完并封存，
随后按完整 S10 cycle sum 选择一个固定 `A1-STRONG`；所有 sample、layer、traffic 与 gate 都使用这一点。
PBR4 的唯一性能门仍是 ratio-of-sums `>=1.30x` 且每个 sample `>=1.10x`，同时要求功能和守恒 0 mismatch、
weight read/refill 不增加、资源相同且 group/RMW/commit 序列不等价于 A1-OSG。

r4 hammer 的三个 P2 已落到可执行规格：

- `slice3=beat_index(0..2)`、`beats_remaining=2-beat_index`；每拍在 outbound ready/valid 时恰退休一次，
  第三拍是唯一 transfer retire，不存在独立或免费 sink ACK；反压只保持，重复拍 sticky fault；
- final-output 采用唯一 layer-specific `directory_index`，向量地址
  `0x20000000+directory_index*384`，beat 地址再加 `beat_index*128`；96 个 Acc24 以小端三字节排列，
  后 96 byte 必须为零 padding；
- 每个 output block 重读完整 dense bitplane。active read/bit、logical/padded byte、base/stall cycle 和
  symbolic energy term 都逐层逐样本报告，并对四点共同收费。

`222,736 B` 旧 r2 数已明确 superseded；唯一当前坐标是 `239,636 B modeled logical`，foundry macro、
CACTI、area、power、energy、PPA 均为 false。M511 payload 与 decoder signed-INT8 weight package 尚未
在本合同 author 时准入，future runner source 也不存在，因此静态 hammer 即使通过也不能执行。
