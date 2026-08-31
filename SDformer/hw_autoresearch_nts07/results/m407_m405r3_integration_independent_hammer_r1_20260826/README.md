# M407：M405R3 selected-slice integration 独立打铁

结论：**PASS，M406 两个 P1 均关闭；允许进入 M408 全量静态 codec 与真实 q32
VCS。** 评分 95/100，P0/P1/P2=`0/0/5`。没有新速度。

我用 Synopsys VCS 做了两组独立执行：

- 原 R3 四项 exact-SHA 全量复跑：elastic、prefix、integration、q32 M384 均
  compile/sim rc=0，零 assertion/fatal/offending；
- 新写反例 TB，重新攻击 M406 的 configuration lifetime 和 global fail-closed，
  同时检查两 pass、tie、narrow/wide 算术、padding 原子性和 release 边界。

## M406 P1 关闭

配置在 last-row 后仍 live。pass1 pending 时 release 关闭；最后 result 未退休时也
关闭。result 退休后，配置继续供合法 PWP 使用；wide assembly/FIFO 未 drain 时
release 仍关闭。独立 TB 模拟两次外部 M384 replay，drain adapter 后再给
`phase_release_valid`，release 合法接受并清除配置。

global fail-closed 反例中，我在 matcher result pending 时同时拉高 config/release/
row/PWP 输入与 result/contribution ready，并注入 wrong-tag PWP。`protocol_error`
同拍出现，所有导出的 config/release/row ready/accept、result valid/accept、PWP
ready/accept、contribution valid/accept 均为零；随后连续 6 拍 sticky quiescence，
总共 9 组 accept 等式检查，ready/accept split=0。

同拍语义需准确限定：wrong-tag low 和非法 release 是 shell combinational violation。
adapter/matcher 内部 padding/config/row 错误在 leaf 注册 error 边界汇总；padding
攻击仍在任何 contribution 可见前被隔离，atomic leak=0，错误可见后全局持续静默。

## leaf 与 M384 没有被 R3 破坏

独立小 miter：matcher source/pass0/pass1/early/results=`2/2/1/1/2`，pass0 重复
center 取 ID0、pass1 重复 center 取 global ID16；无 source/descriptor scratch。
adapter low/high/narrow/wide/contribution=`2/1/1/1/3`；narrow `0x80→0xf80`，wide
`0x080+0xf00=-128`，非法 padding leak=0。

exact-SHA leaf 复跑保持原账本：elastic 386 blocks、100 narrow、286 wide、672
contributions、4 attacks、0 atomic leak；prefix 64 rows、64 pass0、61 pass1、1
early、64 outputs、3 attacks，tie=最低 global center ID。

新版 M384 RTL/SVA/TB 收据有效：tile0 PWP base=6240、tile1=38912、stride/run=640，
bounds=26720/59392；4 phases、8 replays、10,804 bundles、14 runs、10 attacks、
0 mismatch，D8/L8 最大 FIFO/outstanding/credit 均到 8。

## 必须降级的 phase-release 边界

R3 shell **没有实例化 M384，也没有 M384 replay_done/phase_done 输入**。其内部
release 条件只是 matcher release-ready、adapter empty 和 global-safe。因此最后
result 退休且 adapter 为空时，即使外部两次 replay 尚未发生，shell 的
`phase_release_ready` 已经为 1。

这符合 R3 合同的“外部 controller 驱动”定义，但不能写成已经集成证明。真实系统
必须把“两次 replay 后的 M384 phase_done”接到 `phase_release_valid`，不得仅看到
shell ready 就提前释放配置。当前准入的是 leaf/shell directed handshake 和独立
M384 directed VCS，不是完整 M384→shell phase lifecycle。

仍未完成：442,368-block/42,467,328-lane static codec VCS、全真实 q32 miter、M384
物理集成、DC/Formality/PT、SRAM/能耗和 RTL-measured/system speedup。`docs/359`
未修改。
