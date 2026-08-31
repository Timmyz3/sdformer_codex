# M695｜M691 Table-A production registry r9 fresh hammer

## 裁决

`PASS_CANONICAL_ZERO__NO_GO_PRODUCTION_ADMISSION__R10_REPAIR_REQUIRED`，评分
`48/100`，`P0=0 / P1=5 / P2=1`。

r9 的两项基础纪律成立：canonical 仍为
`production_runs=0 / authority=0 / bundle=0 / eligible=0 / headline=false /
analytical=false`，且 `-5e-13 mW` 微负功耗严格拒绝。但它还不能成为未来 Table-A
production PPA 的入口。

## 独立攻击结论

作者正路径 fixture 用同一个 `/bin/true` 冒充五个新思工具和 `simv`，用
ELF 后拼字节冒充六个 `.db`，十个算子是 `wire alive` 空壳，mapped netlist 与 RTL
逐字节相同，SAIF 只有一个 `TC`。在生成前把该 fixture 的旧配置身份修正为冻结的
M527 `b0_dense96_fixed_t10` 后，整条伪造链不但通过 extractor，还被 registry 计为
`validated_production_run_count=1`。

另外四个独立假阳性同样成立：

1. `execution_steps.dc.executable` 可换成不同的 `/bin/false`，而 `argv[0]` 仍指向
   alleged `dc_shell`；只重写自报日志即可通过。
2. 十个 module/instance 名称正则不能证明真实十算子实现或 mapped std-cell netlist。
3. SAIF annotation 的 `99/100` nets、`198/200` pins 完全由 JSON 自报，未从
   PrimeTime annotation report 导出。
4. PTPX 报告没有 17 个宏实例名，DC area 报告没有 logic/macro 拆分；两项仍靠 JSON
   自报完成“一致性”。

## r10 必修门

- 每一步 `entry.executable == argv[0] == expected tool`，版本 stdout、命令、退出码与
  输出由受信 runner 捕获；工具与 compiler DB 绑定批准的安装/hash inventory。
- 冻结真实 H67 source/elaboration identity；从 VCS/DC/Formality/netlist 报告导出十算子
  hierarchy、正 std-cell/FF/combinational census，拒绝空壳和 RTL-copy netlist。
- 从 PrimeTime PX annotation report 导出 SAIF top、duration、toggle、net/pin 覆盖率。
- 从 DC/PTPX/netlist 的 rooted hierarchy/reference 报告分别导出同一 `8+8+1` 宏实例，
  以及 logic/macro area 和 per-memory power；不接受 production JSON 自证。
- 正向测试必须真正走到 mandatory M527 production map，同时证明 synthetic fixture
  没有独立 native-run authority 时会被拒绝。

只有 fresh r10 hammer 达到 `P0=P1=0` 才可 GO。M695 未运行 EDA/GPU，也未修改
`docs/359`。
