# M1288｜C3 M917 Fixed-T10 inert PTSTA source 作者收据

## 裁决

**PASS SOURCE 100/100；PT/EDA 未运行，等待异作者静态打铁与独立 M1289 launch admission。**

本里程碑新增且只新增一条 M917 exact mapped identity 的 PrimeTime slow-max/
fast-min pre-layout STA source DAG。Tcl 读取冻结 mapped Verilog/SDC，绑定
`ssg0p9v125c` max 与 `ffg1p05vm40c` min，输出 setup、hold、coverage、constraint、
clock、library 与 scope 报告。Tcl 没有 `set_fix_hold`、`fix_eco_timing`、
`read_parasitics`、power 或网表写出命令。

Runner 默认 inert：未来 M1289 admission 缺失时，在创建 run/work/attempt namespace、
碰撞扫描、工具/version/license 调用前以 rc=3 退出。负控已执行，三个 namespace
仍全部不存在；未启动 PT、EDA、VCS、GPU 或远端任务，未查询 license。

若将来 PT 报告 hold<0，terminal receipt 只能发布
`DIAGNOSTIC_STOP...HOLD_NEGATIVE`，并要求新的 netlist-only hold-fix identity、
Formality 与重复 PT；M1288 自身不会自动修 hold，也不会改变 M917 mapped SHA。

## 机械门

静态测试 13/13 PASS：

- M917/M928/M1285 双 seal 与 mapped.v/SDC、slow/fast DB、PT SHA 均冻结；
- 双角 max/min 和 100-path 报告入口存在；
- hold-fix、ECO、parasitics、power 命令均不存在；
- future admission 位于任何 namespace mutation 和 PT 启动之前；
- fresh namespace、one-shot、same-UID collision、精确 isolated-job exclusion、
  private 0700 HOME、descendant drain、failure quarantine 均存在；
- `docs/359` 既是冻结输入又在工具结束后重新校验；SHA 未变化。

## Claim boundary

当前仅准许写“inert PT source prepared and statically checked”。不得写 PT completed、
setup/hold closed、full STA、PPA、power、energy、speedup、system 或 headline。

`docs/359_DATE终局冻结_20260813.md` SHA256 保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
