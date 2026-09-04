# M514 C2-D directed VCS receipt-blind 独立打铁 r1

日期：2026-08-27  
证据目录：`results/m514_c2d_directed_vcs_r1_20260827`  
结论：`PASS_DIRECTED_FUNCTIONAL_COMPLETENESS_ONLY__NO_PERFORMANCE_OR_PPA_ADMISSION`  
评分：**98/100**  
P0：**0**  
P1：**3**  
本次运行 VCS / DC / PT / Formality / DSE：**否（只读复核已有封包）**

## 独立结论

M514 exact-SHA VCS 证据足以准入 C2-D K3/S2/P1/OP1 ConvTranspose2d polyphase
address mapper 的 **directed functional completeness**。本审阅先不读取 receipt，从 runner、
冻结 RTL/TB、compile/sim/assert 日志、coverage database 与封存结构独立恢复结论；最后才读取
receipt 交叉比对。两阶段结论完全一致。

可准入的事实只有：在冻结的 3 ns directed test 中，43 个 source-to-tap tuple 全部与独立
scoreboard 精确一致，四个 phase bank 的计数为 `6/10/10/17`，经历 8 个 stall cycle、4 次
same-edge successor replacement 和 1 次 stalled-tap illegal-successor attack；已接受事件在
fault 锁住新事件后仍完整 drain。compile 与 simulation 均 rc=0，恰好一条 exact PASS，未见
assertion failure、tuple mismatch、unexpected tap、fatal 或 timeout。

不得据此主张 full decoder trace、cycle speedup、area、timing、Formality、energy、system
speedup、paper PPA ready 或 DATE headline。

## receipt-blind 方法

### 阶段 A：不读取 receipt

1. 从 filelist 独立确认 VCS 输入只有冻结 RTL 与 TB；
2. 从 runner 恢复 exact-SHA gate、VCS 命令、seed、负日志模式、exact PASS 正则与 seal 流程；
3. 从 TB scoreboard 与 fatal gates 恢复 43 taps、phase、stall、replacement、maximum coordinate
   与 fault-drain 合同；
4. 从 RTL/TB 恢复 embedded SVA/immediate assertions；本设计没有独立 SVA 文件；
5. 从 compile/sim/assert/coverage artifacts 独立判断实际执行和覆盖是否非空；
6. 验证 canonical 内层 `SHA256SUMS`、outer seal、实际 regular-file population 与 one-shot
   attempt identity。

### 阶段 B：读取 receipt

receipt 的 schema/status、seed、43/8/4、`6/10/10/17`、protocol attack=1、四个 coverage
布尔值及十项 claim boundary 与阶段 A 完全相同，没有新增或放大主张。

## exact identity

| 对象 | SHA256 / 状态 |
|---|---|
| exact runner | `fd39ed72a13ceec74a95c5959b90bad24ccce7788e9bb37b16d46ef07f38558b` |
| VCS binary | `0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287` |
| RTL（含 immediate assertions） | `90c44fc9bde839c3cf325ccc8f45c153bf5d30e18de7f39b26d7a4456b017a9a` |
| TB（含 stall stability concurrent SVA） | `6c283bf94d6933e6aa866428f63d6a8b9a2066da2deb39220301f781ec3df47a` |
| two-line filelist | `0a0dbfb33d429566e695afbdbcf48b5081e25fac30d925956a5e96804658adbc` |
| VCS contract | `60e4fe5921a374f399bef82fd1902718428bb8f9d6f3d86dc5d03bda7953ab5b` |
| pre-run static review SHA256SUMS | `20eb76fa32976d4789581c921fae6247c7cee254c090665b922e09609751177e` |
| receipt | `aa6fb4d68c0ec43147481ec3355d8bfdd84777a151cf6f985f20dd763e24d8ee` |
| result SHA256SUMS | `4a77e9d980715cfa6ed2c672b1b9f13f4b8e5c3cc23c95cffa2700ccdb210eaf` |
| result outer-seal file | `98fdb9c3c74f2e27e8ed6094267d3610caad82c6bc7f3eb1cfe7bedadb6609d3` |
| docs/359 | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |

`input_sha256.txt` 与当前所有冻结输入逐项一致；one-shot
`.m514_c2d_directed_vcs_r1_attempt_consumed` 的 exact-four population、member seal、outer seal
与 identity 也全部通过。compile log 明确解析 exact RTL/TB 并选择 exact TB top。

## 覆盖与正确性

### tuple 与边界

- 前五个事件覆盖 fanout `4/6/6/9`，并增加第二个 interior event；阶段一 expected/observed
  都必须为 34，否则 fatal。
- 最后事件覆盖最大合法 `height=width=32, source=(31,31)`；它产生 9 taps，包含最大
  destination coordinate 63。最终 expected/observed 都必须为 43。
- scoreboard 对 tag/time/channel/source、kernel y/x/index、destination y/x、phase、event-last
  与 stream-last 做逐 tuple 四态不等比较；任一 mismatch 立即 fatal。
- 实际 phase 计数 `6+10+10+17=43`，四 bank 均非空。

### backpressure、replacement 与 fault drain

- `stalls=8`，因此 TB 的 `p_tap_stable_under_stall` 前件真实触发，不是 vacuous pass；
- `replacements=4`，same-edge retire/accept 覆盖非空；
- protocol attack=1 在已广告 tap 被强制 stall 时注入非法 successor；最终仍观察完 43 taps，
  且要求 `protocol_error=1, event_ready=0, tap_valid=0, busy=0`。

### assertion 与 VCS artifacts

- compile 使用 `-assert svaext`、`-cm assert`，simulation 使用 `-cm assert` 与 assertion report；
- compile.rc=0、sim.rc=0；compile log 没有 runner 所禁止的 warning/error/fatal；
- `assert.report` 为 0 bytes，表示没有 failure 条目；`assert.report.disablelog` 三个分类标题下
  都没有 disabled assertion instance；
- assertion coverage database 实际存在 8 个 regular files、3036 bytes；
- sim log 只有一条 exact PASS，negative scan 对 assertion failure、mismatch、unexpected tap、
  fatal、watchdog 与 timeout 全部为零。

因此“无 assertion/mismatch”与“覆盖非空”均有独立证据，而不是只相信 receipt 布尔值。

## 封存审计

canonical 的 member seal 与 outer seal 均通过。当前目录的 94 个 regular files（排除两个 seal
文件）与 `SHA256SUMS` 的 sealed set 精确相等；RUN_COMPLETE、receipt、RC、日志、VCS binary
与 assertion coverage database 均在 seal 内。

目录同时包含两个 VCS 生成 symlink：coverage shape link 与 csrc shared-object link。它们没有
进入 `find -type f` 生成的 member seal；二者都不是本结论的唯一证据，其 target 对应的实际
regular files 已封存，所以不构成 P0，但属于 evidence-topology P1。

## P1

1. canonical seal 精确覆盖当下所有 regular files，但没有 exact entry/type manifest；两个 VCS
   生成 symlink 与空目录拓扑未封。后续应封 `find -P` 的 population/type/link-target manifest。
2. receipt 本身只记录 tool/seed/measurement/claims，不内嵌 runner/RTL/TB/filelist SHA；身份依靠
   同一 sealed package 中的 `input_sha256.txt` 交叉连接。可在下一版 receipt 内显式引用该文件
   SHA 与 result outer seal。
3. assertion DB 非空、stall SVA 前件非空且无失败，但没有导出 named assertion hit/attempt
   summary；若追求更强 artifact presentation，可只读导出 URG assertion summary 后另封，不能
   回写本 canonical。

## 决策与下一门

结论是：`PASS_M514_DIRECTED_FUNCTIONAL_COMPLETENESS_ONLY`。M514 可作为 C2 decoder
ConvTranspose2d completeness adapter 的功能证据；下一门应按既定合同做 standalone 3 ns DC，
并把面积作为 admitted C2 K8 logic 的 **additive upper bound**。在 DC/STA/必要的 Formality、
full decoder workload 与 cycle model 之前，不得把 M514 写成速度贡献或 PPA headline。

`docs/359` 未修改，SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
