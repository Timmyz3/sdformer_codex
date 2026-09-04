# M112 W384 lane-sliced accumulator 独立打铁评审 r1

日期：2026-08-24  
评分：**89/100**  
严重度：**P0=0，P1=6，P2=5**  
结论：**wrapper mapping、behavioral sync-1R1W protocol 与 directed numeric 功能 GO；foundry macro、physical timing/energy、integrated cycle speedup 与 system/headline NO-GO。**

本评审只写本目录，未修改 M112/M111 production RTL、SVA、TB、contract 或 sealed evidence。`docs/359_DATE终局冻结_20260813.md` SHA256 前后均为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 结论

M112 的功能 reshape 是正确的。独立 commercial VCS 同时实例化冻结 M111 作为 cycle reference，并使用完全独立的软件整数 scoreboard：在 reverse/permuted update、flatten 两端、one-gap RMW、full commit、长 backpressure、lazy stale memory、RDW/range/正负 overflow/collision 攻击上，M112 的外部行为与 M111 逐周期一致，`block×384+row` 和 96 个 lane slice 均未发现错位。

96×3072×24 比 8×384×2304 更接近 SRAM compiler 的正常形状，这个架构方向成立。但 M112 没有减少每次 update 的总数据运动：96 个宏仍同时读 2,304 bit、写 2,304 bit，并驱动 96 个 signed25 adder。它只是把“不可能的单个超宽字”变为“96 个同时活动的窄宏”。没有真实 compiler/DB、floorplan、STA/PTPX 前，不能说它已经物理合理，更不能说 2.535× 已被硬件测得。

## Exact-SHA 与 sealed 可复核性

| 项目 | SHA256 |
|---|---|
| M112 contract | `8eb2d82c329bd1612d2808a1edfb13345eddaa770156adf7da172a008f981f44` |
| adapter RTL | `ee5a2a84c8c28e113340c73195fc08eec4c975eed27622ea8eee654b3f25226e` |
| frozen M111 core | `354e0de95ee4380098c09fac67af3e137b3ab8bb9f88ac706d62fe201179b43a` |
| production SVA | `938373f712ef925d08fdad9aeeac4040e66b01c541f7f41416cafd76c1f4d874` |
| production TB | `7cbfa75bbe408fa080580dbe1037b04ef7c93db87e58efa68a349d154cfbee5e` |
| production runner | `ccf661e829613e9140f4ea750738af2899b4e71e754a6a3bb688373fb0447993` |
| production sealed RUN_COMPLETE | `458dc8af156165bf726d36a57813d2d476ec25dded82ffdee077c186f63bba26` |
| independent SVA | `c1dc84b2285e4780a03dea3ed0404f1a735edf7b38a3854c4d9e1bc8ca7c5394` |
| independent TB | `b0333842c2e4b7521a82cf8d5d218b2485f234bd8a8860da465157ba9356bf1c` |
| independent runner | `ae618be30b5cdd4a408ac457412aae7e74d2d09ba9fcb323686b124f11c6c4b7` |

Production sealed 的 8 个输入、4 个输出与 runner self-hash 全部复核通过；compile/sim RC=0，compile warning/error signature=0，assertion failure=0，exact PASS line 匹配。独立 sealed 也先执行完整 `input_manifest.sha256` 校验并拒绝覆盖已有收据。

## 独立 VCS 结果

工具：Synopsys VCS V-2023.12-SP1。

| 项目 | 独立结果 |
|---|---:|
| reverse/permuted updates | 256 |
| nonconflicting II=1 pairs | 255 |
| read/write overlap | 255 |
| commit vectors | 6,144 |
| signed lane comparisons | 589,824 |
| exact flattened read checks | 518 |
| exact flattened write checks | 261 |
| lane write slice checks | 25,056 |
| commit stall cycles | 2,242 |
| full commit windows | 2 |

攻击均为 reset-isolated：

- consecutive same-address：第二项被拒绝，第一项的 96-lane write 仍完整落存；
- row `384`：首个非法 9-bit row code，无 read/write，sticky fail-close；
- signed24 `max+1` 与 `min-1`：整条 vector write 被抑制，旧值不损坏；
- start+update collision：无 accept/read，sticky fail-close；
- 第二窗口不清 data memory，只清 3,072 valid bits；旧物理数据仍非零，但 3,072 个 commit 全为零，且 invalid row 不发 macro read。

## Flatten 与 lane mapping

RTL 使用：

```text
flat = (block << 8) + (block << 7) + row
     = block × (256 + 128) + row
     = block × 384 + row
```

因此：

- `(block=0,row=0) -> 0`；
- `(block=1,row=0) -> 384`；
- `(block=7,row=383) -> 3,071`；
- legal range 精确为 `0..3071`。

每个 lane `i` 的双向映射为 `lane_word[i] <-> vector[i*24 +: 24]`。独立 miter 检查了 25,056 次 lane write slice，并逐周期对照 M111 的 onehot block command。该证据是强 directed equivalence，不是形式化穷举 equivalence。

## Sync-read、lazy-valid、commit、RDW 与 overflow

一条合法 update 在接受拍发同步 read，同时缓存 `(block,row,delta,prior_valid)`；下一周期用 96 个 25-bit 扩展加法形成 write。不同地址可 II=1；`A,B,A` 可正确累加，而连续 `A,A` 因第二个 read 与旧 write 的宏 RDW 语义未冻结而 fail-close。

window start 只清 3,072 valid bits，不扫 884,736-byte data。首次写 invalid row 时 base 强制为零；commit 对 valid row 发 macro read，对 invalid row 直接发零。one-deep commit pipe 在 `commit_ready=0` 时冻结所有 data/sideband，最后项固定 `(block7,row383)`，接受后才产生 `window_done`。

overflow guard 在 25-bit 扩展和上检查任一 lane 的 sign extension；任一 lane overflow 会抑制整条 96-lane write并 sticky fault。独立测试同时覆盖正、负两端。

这些行为在 behavioral one-cycle sync 1R1W 模型上成立。若 foundry macro 具有两周期/registered output、只有 1RW、或不同的 cross-port semantics，当前 pipeline 和 II=1 都必须重做。

## 96 个窄宏是否真的更物理合理

| 组织 | M111 logical wide banks | M112 lane sliced |
|---|---:|---:|
| 实例数 | 8 | 96 |
| depth/instance | 384 | 3,072 |
| width/instance | 2,304 bit | 24 bit |
| logical data | 7,077,888 bit | 7,077,888 bit |
| 每 vector read | 2,304 bit | 96×24 = 2,304 bit |
| 每 vector write | 2,304 bit | 96×24 = 2,304 bit |

优势是真实的：24-bit word 比 2,304-bit word 更接近常规 SRAM compiler，block 维被折进深度，每个 lane read 直接进入一个 lane adder。

但尚未证明的代价同样真实：

1. 96 个宏都要独立外围、clock/enable、测试/修复接口；实例开销可能很大。
2. 12-bit read/write address 与 enables 需要扇出到 96 个宏。
3. 每次 update 仍有 4,608 data-bit 的 macro read+write 活动；delta 与 commit 另有 2,304-bit 全局路径。
4. 96 个宏的输出要围绕 96 个 signed25 adder floorplan，并在目标周期内读→加→overflow reduce→写。
5. compiler 若只提供 4,096-depth，数据阵列会从 884,736 B 膨胀到 1,179,648 B，增加 33.3%；width rounding/dual-port peripheral 还会继续增加。
6. 真实 3072×24 1R1W 是否存在、PVT latency、RDW、ECC、redundancy、power gating、congestion 与 PTPX 均未知。

因此结论是：**M112 比 M111 的超宽 logical port 更有物理可实现方向，但不是已验证的物理实现。** 下一步必须用目标 SRAM compiler 做 width/depth/port DSE，至少比较 `96×24b`、较少实例的 `48/96b` 等切片，并用 macro abstract/DB 跑 floorplan-aware STA/PTPX。

## 存储口径

- accumulator data：`384×8×96×24 = 7,077,888 bit = 884,736 B`；
- lazy valid：`3,072 bit = 384 B`；
- M109 descriptor raw payload：24,576 B；
- minimum descriptor metadata：314 bit；
- M109 的 909,352 B 明确未含 valid；加上 M112 实际 valid 后，逻辑小计至少为 **909,736 B**。

这个小计仍不含 controller、宏 depth/width rounding、1R1W peripheral、ECC 和 redundancy。

## 2.535462× claim boundary

M109-r2 的 `439,708,199 candidate cycles / 1,114,863,448 conditional fixed8 cycles = 2.5354620417×` 是 independently reproduced 的 same-clock precompacted software projection。

M110 已验证 standalone transpose geometry；M112 已验证 standalone lane-sliced accumulator directed 功能。两者尚未用 actual heldout descriptor 串接，也没有 shared-weight SRAM、precompaction delivery、foundry macro timing、频率归一或 matched physical baseline。因此：

- `2.535462× software projection`：GO，必须保留 qualifier；
- `2.535462× scheduled RTL / measured / physical / system / headline`：全部 NO-GO。

## Findings

### P1

- **FOUNDRY-1R1W-MACRO-UNPROVEN**：没有真实 3072×24 1R1W compiler macro、DB、PVT、RDW、ECC 或 rounding。
- **AGGREGATE-WIRING-AND-FANOUT-UNCHANGED**：96 宏同时工作，2,304-bit 总读写与全局 delta/commit 并未消失。
- **SRAM-ADDER-SRAM-TIMING-UNPROVEN**：one-cycle SRAM read→96 adders→next-edge write 无 STA/P&R 证据。
- **INTEGRATED-ACTUAL-REPLAY-ABSENT**：M110/M112/PWP/shared SRAM/commit 未串 actual heldout commercial cycle miter。
- **SAME-ADDRESS-LIVENESS-EXTERNAL**：连续同址会 reset-only quarantine，实际 workload spacing 未集成证明。
- **MATCHED-PHYSICAL-BASELINE-ABSENT**：无同库、同频、等面积 baseline，不能报 physical speedup/energy。

### P2

- production SVA 只查 address range，未查 exact flatten、onehot source 和 96 lane slice；独立 miter已补 directed 证据。
- production regression 未覆盖 negative overflow、row384 和 start/update collision。
- M109 909,352 B 漏计实际 3,072 valid bits；最低逻辑小计应为 909,736 B。
- 独立 wrapper/M111 miter 是 directed，不是 formal exhaustive equivalence。
- invalid rows 虽不读 macro，但仍固定输出 3,072 个 commit vector；zero bus/consumer energy 与 sparse commit 方案未评估。

## GO / NO-GO

| 项目 | 决定 |
|---|---|
| production exact-SHA sealed VCS | **GO** |
| independent commercial VCS | **GO** |
| flatten address / 96 lane mapping | **GO directed** |
| wrapper 对冻结 M111 cycle equivalence | **GO directed，非 formal** |
| sync-read、nonconflicting II=1 | **GO behavioral directed** |
| lazy-valid、commit backpressure | **GO directed** |
| same-address preserve/fail-close | **GO directed** |
| 正负 overflow、range、collision | **GO directed** |
| lane slicing 比 2304-bit macro 更合理 | **GO，architecture direction only** |
| foundry 96×3072×24 1R1W fabric | **NO-GO** |
| same-clock physical timing/energy | **NO-GO** |
| actual heldout integrated cycle miter | **NO-GO** |
| M109 2.535× software projection | **GO，qualifier mandatory** |
| 2.535× scheduled RTL / physical / system / headline | **NO-GO** |

复跑：

```bash
cd hw_autoresearch_nts07/reviews/m112_w384_lane_sliced_accumulator_independent_hammer_r1_20260824
./run_vcs_m112_independent_hammer.sh
```

runner 会拒绝覆盖现有 `vcs_sealed/`，复跑必须新建收据目录。
