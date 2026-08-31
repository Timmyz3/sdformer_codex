# M146 四 bank age-queue 独立打铁评审

结论：**86/100，条件通过；P0=0、P1=1、P2=4。** M146 的核心机制成立：两个深度 4 的 2-bit bank-ID FIFO 能跨 32-bit sequence wrap 保持服务年龄顺序，综合结构中没有 sequence 大小/年龄比较器。封存 VCS/DC 均通过身份校验，独立 Synopsys 复跑也逐项复现。

但它暂时不能无条件宣称“所有 stale response 都 fail-closed”，也不能把相对 M142 的 `68/18=3.78×` logic-level 比和 `2.52/0.90=2.80×` QoR path 比写成任何 speedup。M142 是集成 controller，M146 是 standalone scheduler，功能、资源和关键端点都不匹配。

## 独立结果

| 项目 | 独立结果 | 判定 |
|---|---:|---|
| Production VCS exact rerun | 40 jobs / 36 reuse / 3 attacks / 0 assertion fail | 复现封存 PASS |
| Independent VCS | 64 wrap jobs / 60 reuse / 两个 FIFO 均到 4 / 4 attacks | FIFO 顺序和 quarantine 通过 |
| Reset-midflight | 1 次，复位后清空并恢复 | 通过 |
| Reset release quietness | `release_valid` 漏出 1 次 | 发现 P2 |
| Independent DC area | 1536.696002 µm²，1827 cells，373 sequential | 精确复现 |
| Independent DC timing | 18 levels，0.90 ns path，setup/hold 1.6265/0.0002 ns | 精确复现 |
| DC scope | ideal clock / ZeroWireload / 0 macro | 仅 pre-macro logic-only |

DC resource report保留 4 个 32-bit 完整身份等值比较器：fill identity 两处、PWP completion 一处、correction completion 一处；另有 1 个 32-bit sequence incrementer。没有 32-bit `<`/`>` 年龄比较器。当前关键路径从 `correction_done_sequence[18]` 到 `bank_state_q_reg[0][0]`，属于 correction identity equality + quarantine/state-update cone。

## 必须修改后再 admission

### P1 — 32-bit ABA 窗口

FIFO 顺序跨 wrap 没问题，但身份在 `2^32` 次 fill 后会复用。如果一个极老 completion 恰好与当前 active job 的 bank/tag/sequence 全相同，RTL 无法区分。要么在合同中冻结并断言 response lifetime 严格小于 identity reuse interval，要么增加不会在 stale response 仍可能存活时复用的 epoch。否则“unconditional stale rejection”不成立。

### P2 — reset 期间 release_valid 未门控

`pwp_valid`、`correction_valid`、`fill_ready` 和 `protocol_error` 都有 reset 门控，但 `release_valid` 没有。独立 VCS 在 `rst_core=1`、同步复位边沿到来前输入匹配 correction completion，稳定观察到一次 release。建议给 `release_valid` 增加 `!rst_core`，并加入 reset 期间所有 transaction valid 必须为 0 的 assertion。

### 其余 P2

- Production directed VCS 没有把 wrong-bank、wrong-tag、wrong-sequence、wrap、两个 FIFO-full、midflight reset 和 reset quietness 分别封口；独立测试已经提供可移植用例。
- 封存 DC receipt 的 `critical_path_length_ns` 因 awk field 错误为空；现有 identity-pinned overlay 对 frozen QoR 的 0.90 ns 修正正确。下一 revision 应修 runner，并对空值 fail-closed。
- 当前是 standalone、ideal-clock、ZeroWireload、0-macro DC；没有 Formality、物理实现、matched integration 或 executable same-workload cycle comparison。因此不是 paper PPA，也不是 speedup admission。

## 评分与处置

评分拆分：协议与顺序 25/30，验证强度 20/25，综合可复现性 19/20，证据与 claim discipline 17/20，admission boundary 5/5。

建议保留 M146 作为可信的“FIFO-age replacement removes age-comparison cone”模块创新。先修 P1/P2，再重封 VCS；相对 M142 的 3.78×/2.80× 只能作为结构观察，禁止写成 cycle、frequency、physical 或 system speedup。

机器可读结论见 `m146_independent_hammer_review_r1.json`；`manifest.sha256` 覆盖本目录全部保留证据（manifest 自身除外）。评审未修改 production RTL、contracts、sealed runs 或 `docs/359`，也未 commit/push。
