# M831：C1 R19 wall-clock fail-closed source author handoff

## 交付结论

M831/R19 source-only package 已完成。它只修复 R18 在 Verilog 时间之前被 Synopsys runtime 外部服务阻塞且没有 wall-clock 上界的问题；未改 RTL、TB、SVA、foundry UNIT_DELAY 模型、13 个 normal cover、P2、held-final、六攻击、资源门或双封门。

生产命令唯一新增语义为：

`/usr/bin/timeout --signal=TERM --kill-after=30s 300s ./simv -no_save`

300 s 只是一条 fail-closed wall-clock 基础设施门，不是仿真周期、RTL 性能或 timeout 归因。rc 124（TERM 退出）或 rc 137（TERM 不退后 KILL）且 `sim.log` 无 HDL token 时，终态必须分类为 `infrastructure_timeout_before_verilog_time` 并双封 `FAILED_DO_NOT_CITE`。`-no_save` 只按 VCS 自身提示避免 ASLR re-exec；没有加入、猜测或宣称任何 telemetry-disable 环境变量。

## 冻结身份

- Runner SHA：`2db504cfe68d58fa6735cdb4a438ac01c0cb5dddb8a8858307af5350f116ceca`
- Source contract SHA：`d8e4fcd8575b9837facbbfd362ebd96927c2a01df5e3e08f1b74e0c39dbd6f04`
- Closed candidate SHA：`6189fe1abda8c856897726d1ec2bbd279afaec7b2654808729997494a4325373`
- `/usr/bin/timeout` SHA：`2d5662f0e08f558aa842d7bc99aa00ea534b4fb46c7e72f6a4c834220cafbf02`
- Top/SVA/TB/foundry model SHA 分别保持 `726039...` / `b9f66f...` / `cd0cf9...` / `8343ac...`。

R18 result、M827 和 M829 已作为 exact-SHA 直接前件绑定。R18 永久不可重跑、恢复、重标或引用；R18 失败不能归因 RTL。

## Source-only 验证

- Python 3.6 TB r8 static：PASS；六行 witness、13-cover 顺序、P2、held-final、六攻击保持。
- 完整 closure：34 个 custom definition、266 个 custom call、21 个外部命令，0 未定义、0 duplicate、0 whitelist 漂移；delete/rename/inject-stale 三个负变异均被拒。
- `require_regular_sha`：94 条 lower-hex exact edge；runner 自己的 pre-mkdir dry-run 已实际走过所有前件/资产校验。
- fake-simv：fast `(0,0)`、TERM timeout `(124,0)`、TERM 不退→KILL `(137,0)`、tee failure `(0,7)`；TERM/KILL 后 fake pid 均不存在；timeout failure receipt 的成员与外层 seal 均通过。
- pre-mkdir：rc86，事件严格为 collision_initial → cgroup → resource → collision_final → live_probe_boundary_stop；VCS identity、license、compile、simv 和 result mkdir 均为 0。

## 边界与下一门

本 author 没有运行 VCS、simv、许可证查询或任何 EDA，没有创建正式 result/attempt/release，也没有自做 source hammer。Candidate 的 `launch_now=false` 且 `authorization_effective_now=false`。

下一步只能由 fresh independent reviewer 按 M833 request 重跑 source-static、closure 三负变异、timeout fake-simv 与 pre-mkdir dry-run，并核对全部双封及 R18/M827/M829 前件。只有 PASS100、P0/P1/P2=0/0/0 才能继续 candidate hammer；仍不能直接 launch。

`docs/359` 未修改，SHA 为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
