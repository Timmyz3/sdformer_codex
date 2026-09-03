# M1999｜M1998/M1995 TSBG-B4 keyword-legal VCS 结果独立 hammer

日期：2026-09-02

裁决：**PASS；100/100，P0=0，P1=0，P2=0。仅准入 keyword-legal 组件级 directed behavioral VCS 功能证据。**

## 身份、seal 与执行预算

M1998 result、相邻 consumed attempt、M1995 失败诊断、M1997 source review 和 M1990 旧功能结果 review 的内外 seal 全部通过。实际编译依次读取 M803 adapter、SHA 为 `2c1a8a7644b359a153decdc3106a8718992d37d54809007b61e184121fcc14fd` 的 M1995、M1880 SVA 和 M1984 TB；top 为 `tb_m1880_c2_tsbg_b4_real_channel_signed_frontend`。M1997 把该 M1995 身份绑定为只含 16 个 standalone `context` 到 `ctx` 的 alpha rename。

namespace 中只有一个 M1998 结果和一个相邻 consumed attempt。执行账本严格为 1 次 license query、1 次 VCS compile、1 次 simv、0 retry；SVA 以 `-assert svaext` 编译、运行期启用 `global_finish_maxfail=1`，并有 180 秒外层 timeout。

## 唯一机器回执与有界进展

以下 13 字段完整行逐字出现且只出现一次：

`PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED rows=48 issues=576 products=9216 commits=24 bundles_base=576 bundles_tsbg=144 scalar_base=4608 scalar_tsbg=1152 stale=1 retired_replay=1 replay_accept=0 reset=2 recovery=1`

10 个 phase 的 begin/complete 均各一次，共 20 token；load 为 52 begin、52 complete、0 timeout。独立 parser 对错误 rows、截断、额外字段、重复行、字段重排、缺字段、错误 prefix、错误 stale、重复字段和非数值共 10 种变异全部拒绝。

日志没有 compile error、fatal、原生 SVA failure、`SVAA-RNF`、ignored/unknown runtime assertion option、watchdog 或 directed timeout。TSBG 的 11 个 required SVA cover 全部非零；base 侧没有注入 stale/reset attack，因此对应两个 cover 为预期 0。

## 与 M1990 的行为一致性

M1998 与封存 M1990 的完整 PASS 行、20 个 phase token、104 个 load ledger token以及 base/TSBG 全部 cover 计数逐项相同。这证明 keyword-legal alpha rename 没有改变该 directed 行为账本；它不把旧 CPU premodel 或局部 ratio 升格为新性能结果。

## 准入边界

可以引用：**精确 M1995 keyword-legal 源在该 directed workload 下通过 VCS 功能、逐 lane 算术、work conservation、LRU4、typed-signed corner、stall/reorder、stale/replay 攻击和两次 reset recovery 验证。**

不得声称：DC、面积、PPA、同面积、精确周期加速、时序、hold、功耗、能量、全系统或论文 headline；M1866 的 `2.533808x` CPU premodel 不因本结果升级。

本 hammer 没有启动 EDA 或 license query；`docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
