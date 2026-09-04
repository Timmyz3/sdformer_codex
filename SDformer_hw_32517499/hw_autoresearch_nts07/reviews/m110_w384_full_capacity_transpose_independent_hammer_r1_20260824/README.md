# M110 W384 standalone transpose 独立打铁评审 r1

日期：2026-08-24  
评分：**88/100**  
严重度：**P0=0，P1=5，P2=4**  
结论：**W384 standalone controller geometry/full-capacity 功能 GO；accumulator、2.535× scheduled/RTL-measured、PPA/system/headline 全部 NO-GO。**

本评审只写 `reviews/m110_w384_full_capacity_transpose_independent_hammer_r1_20260824/`。production RTL、SVA、TB、contract、results 与 sealed run 均未修改；`docs/359_DATE终局冻结_20260813.md` SHA256 前后均为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 结论

没有发现新的 RTL P0。production sealed VCS 与独立 sealed VCS 均通过。独立 campaign 不照抄 production 的 ascending ingress，而是对两组完整窗口都执行 **reverse key × reverse row** 输入，再逐 token 检查升序输出，从而真正验证 transpose、bank identity、direction、destination、last-for-key 与 stall 保持。

M110 可以把 M109-r2 的 `W384 controller_geometry_vcs` 从 false 提升为 true；不能把 M109-r2 的 `2.535462×` 提升为 RTL measured 或 scheduled speedup。M110 明确不含 accumulator，而 M109 W384 点假定的 signed24 accumulator 单窗口就有 7,077,888 bit（884,736 byte）。

## 冻结身份

| 项目 | SHA256 |
|---|---|
| M110 contract | `4f2b5c329ea552742c55a362739f032272fb510cc3c659b0c73f52eced9f5253` |
| production RTL | `61a2c18f3b0a350bfc57193b9573f3d0ed5ea68f68ae4fc982ec1908054dbd6c` |
| production SVA | `daf98af5808c58d90b7428eeb42061a956bbe6b4889a52dadf5b47d4f83bc8cf` |
| production TB | `1a59afc90e2a3e6c4b6edb233951e9811c89765ed610b4af4a80f5a85d7f70d4` |
| production sealed RUN_COMPLETE | `2b73e6e29fcd176ab17d479fa33c0d0d785d3e2b90719ec7047b9513f5acfef7` |
| M109-r2 result | `ee61b90ee894c6e6c778b815a52f1d8b6edc9c877227bc4987e4b135aa16c321` |
| M109-r2 independent review | `423a53a9d65cc274dad2deedad8e41f28afe08178506f31f234624ccb0e24f9f` |
| independent SVA | `aa38c8e3f1440f7593ccede7c27eda14a8a0a1bd3d0fc509a19648f0379886b9` |
| independent TB | `e8afd15d2f39732380508beab8b708b16159c86b2456421e847361da50069b3e` |
| independent runner | `5a662b1d5a70c4d0bdc41b00735340754d34532eae3277732f6df0cf12ac3299` |

完整输入身份见 `input_manifest.sha256`，sealed 运行前全部通过 `sha256sum -c`。

## 商业 VCS 结果

工具：Synopsys VCS V-2023.12-SP1。独立 compile/sim RC 均为 0，compile warning/error pattern=0，assertion failure=0。

| 指标 | production sealed | independent sealed |
|---|---:|---:|
| full windows | 2 | 2 |
| keys × rows/window | 128 × 384 | 128 × 384 |
| ingress events | 98,304 | 98,304 |
| load / event / service tokens | 768 / 98,304 / 99,072 | 768 / 98,304 / 99,072 |
| changed-legal event II1 pairs | 98,302 | 98,302 |
| service stalls | 9,952 | 17,708 |
| fill/drain overlap cycles | 49,152 | 49,153 |
| exact close grace | 2 | 2 |
| cross-bank exact close grace | covered | 1 explicit cover |
| ingress ordering | ascending | reverse key, reverse row |

独立 scoreboard 检查每一个 service token：

- 每 active key 恰好先发三个 load beat，再发 384 个 event；
- 无论 ingress 顺序如何，drain 均按 key 0..127、row 0..383 排序；
- source/block、row、destination、direction、last、context 全部逐项匹配；
- 17,708 个 service stall 中 payload 保持稳定；
- 两个 full window 的 exact final event 都不重收，首个 full close 在 alternate fill bank 可用时仍不重收；
- 两个 changed-legal empty close 可相邻两拍接受，验证 close II=1。

非法 campaign 另做五次 reset-isolated 攻击：第一个非法 row code `384`、duplicate、grace 后 changed direction/context、event-close collision、两 bank 占用时第三次 ingress。全部同拍 fail-closed、sticky quarantine，且只能 reset 恢复。

## 196,608 bit 核算

该数字准确，但只代表 raw bitmap：

- presence：2 banks × 128 keys × 384 rows = 98,304 bit；
- direction：98,304 bit；
- raw total：196,608 bit = 24,576 byte；
- active-key、base、context、identity-valid 至少另加 314 bit；
- control/grace、ECC、macro rounding 前 controller state 至少 196,922 bit，向上取整 24,616 byte。

`196,608 bit` 不得称为 total state、SRAM macro area 或物理存储。RTL 对两组 bitmap 做全阵列同步 reset并支持任意 bit set/clear，当前 `macro_count=0`，尚无 DC 或目标 SRAM 映射证据。

## M109-r2 的 2.535× 如何解释

W384 冻结数据为：

| 项目 | 数值 |
|---|---:|
| exact heldout events | 188,148,490 |
| active groups | 8,271,296 |
| PWP tokens | 226,222,255 |
| candidate cycles | 439,708,199 |
| conditional fixed8 baseline | 1,114,863,448 |
| same-clock precompacted service-island ratio | 2.5354620417× |
| controller bitmap + signed24 accumulator lower bound | 909,352 byte |

M109-r2 的 raw work 与 dual-timeline 软件递推已独立复现。M110 现在只补上 **standalone W384 controller geometry** 的商业 VCS 证据。以下仍未进入 M110：

1. 384 × 8 output blocks × 96 lanes × signed24 的 full-lane accumulator；
2. finite-width numeric equivalence、bank/address/RMW/forwarding、clear/epoch、flush/commit；
3. actual heldout ordered descriptor 的 integrated RTL replay；
4. PWP/correction shared-weight SRAM 地址、端口、延迟与仲裁；
5. precompaction scan、有限队列和 delivery bandwidth；
6. macro/DC/STA/power/equal-area。

因此 paper-safe 说法只能是：**W384 standalone transpose controller 在商业 VCS 中通过 full-capacity、II=1、stall 和 fail-closed 验证；相关 2.535× 仍是 precompacted same-clock software projection。**

## Findings

### P1

- **ACCUMULATOR-ABSENT**：M110 没有实现 M109 W384 点所需的 884,736-byte signed24 accumulator 或其数值/RMW/commit miter。
- **ACTUAL-INTEGRATED-CYCLE-REPLAY-ABSENT**：无 actual heldout 的 precompaction/PWP/correction/controller/accumulator integrated replay，2.535× 不是 RTL measured。
- **PRE-DC-PRE-MACRO**：196,608-bit bitmap 没有 SRAM mapping、DC、timing、area、power；全阵列 reset 与任意 bit update 可能阻止直接宏推断。
- **FAILFAST-NOT-GENERAL-BACKPRESSURE**：两 bank unavailable 时 assert valid 会 sticky fault，集成必须使用 credit/ready look-ahead 或 adapter。
- **CONTEXT/BASE/PARTIAL-WINDOW SCHEMA OPEN**：operator/partition identity、base alignment/overflow 与 partial-window legal extent 尚未冻结。

### P2

- production TB 使用与 drain 相同的 ascending ingress；本独立 reverse-order campaign 才真正压力验证 transpose sorting。
- production SVA 未明确断言 exact grace、changed-legal streaming、semantic illegal 与 range boundary；其 stall property 也未允许同拍 protocol fault 终止 service。
- 196,608 bit 只是 raw payload；314 bit 也只是 minimum bank metadata，均不包含完整控制/ECC/宏取整。
- M109 baseline 未匹配 descriptor/controller ingress，candidate 又假定 lossless precompaction；2.535× 必须保持 conditional service-island qualifier。

## GO / NO-GO

| 项目 | 决定 |
|---|---|
| production sealed directed VCS | **GO** |
| independent reverse-order full 128×384 | **GO** |
| changed-legal event/close II=1 | **GO** |
| exact event / cross-bank close grace | **GO** |
| stall stability、lossless sorted drain | **GO** |
| range/duplicate/context/collision/unavailable quarantine | **GO** |
| raw bitmap 196,608 bit | **GO，必须带 metadata/macro qualifier** |
| W384 controller geometry VCS | **GO** |
| full-lane accumulator VCS | **NO-GO，未实现** |
| M109 W384 2.535× software projection | **GO，projection label only** |
| 2.535× RTL-measured/scheduled | **NO-GO** |
| actual heldout integrated replay | **NO-GO** |
| DC/macro/physical/equal-area/system/headline | **NO-GO** |

复跑：

```bash
cd hw_autoresearch_nts07/reviews/m110_w384_full_capacity_transpose_independent_hammer_r1_20260824
./run_vcs_m110_w384_independent.sh
```

runner 拒绝覆盖已有 `vcs_sealed/`；复跑必须使用新目录，不能破坏本次收据。
