# M1128r5｜M1122r4 elaboration failure 只读取证

结论：高置信根因不是 DC selector、license、include 路径、`SYNTHESIS` 保护或库加载，而是 r2 的宏别名写法没有产生 r2 模块。

## 已封失败事实

- attempt outer：`8a012c8638c2e8a8da743cbf570a13f5c8bc8d85716b433882d03405e12e5e37`，`dc_attempts=1`。
- quarantine outer：`2f9173b1e988b1f639e6c3d683fdf720fa9debfeaca8caf27bf5845a36527f83`，状态为 `FAILED_DIAGNOSTIC_DO_NOT_CITE`，`m1122r4_retry=false`。
- selector runtime receipt 仍为 PASS：实际捕获的是冻结 `common_shell_exec` 和精确七 token argv。这证明 selector 修复成功，但不证明 HDL top 存在。
- DC 明确打开了 `rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv`，Presto analyze 成功；随后 `elaborate m1112r2...` 报 LBR-0，`current_design` 报 UID-4，并由 Tcl 显式退出 35。

## 根因

r2 文件执行：

```systemverilog
`define m1112_c2_k1_async_observation_shadow_wrapper m1112r2_c2_k1_async_observation_shadow_wrapper
`include "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"
```

但被 include 文件的声明是裸 token：

```systemverilog
module m1112_c2_k1_async_observation_shadow_wrapper #(...)
```

Verilog/SystemVerilog 宏只有以反引号调用时才展开；裸 identifier 不会因为存在同名 ``define`` 而替换。因此 analyze 合法编译了旧模块，WORK 中没有请求的 r2 top。M872 和 M917 的成功点都直接声明了与 `DESIGN_NAME` 一致的真实 module，随后各自 `Elaborated 1 design`；这是关键调用差异。

同一缺陷还潜伏在 r2 mapped-VCS TB：TB top 和 DUT type 都是裸旧 token。只修 DC RTL 后，mapped VCS 仍会在下一阶段失败。

## 最小 additive r5 边界

推荐机械复制重命名，而非本轮引入真 wrapper：

1. 从冻结 base RTL 产生新 r5 文件，只替换一次真实 module declaration 名；其余字节保持等价。
2. 从冻结 base TB 产生新 r5 文件，只替换 TB top 和 DUT type 两处真实 identifier。
3. 新建 r5 filelist、design/TB top、engine、contract、launcher/hammers 与全部 namespace；旧 r2/r4 文件和失败 namespace 不动。
4. 静态门必须检查“真实直接 module 声明”而非只检查 ``define``；正式 fresh attempt 前另行授权一次 directed VCS compile/elaboration。

真 wrapper 实例化旧模块理论上可行，但需要重新审计完整参数/端口绑定、层级 flatten 和 337-bit reset census，并且无法省掉 TB 修复，当前窗口风险更高。

## 仍然有效 / 已失效

仍有效：M1122r4 selector runtime 诊断回执、attempt/quarantine 永久不可重试证据、M872 C2 三轴 logic-only DC、M917 C3 Fixed logic-only DC。M1125r4 静态 hammer 本身仍有效，但它授权的唯一 r4 attempt 已消费。

不存在：M1122r4 mapped functionality、M1122r4 PPA/activity/power/performance。禁止从这次失败升级任何论文指标。

本审计未运行 EDA/VCS、未修改被审源、未触碰 docs/359。旧 r4 永久 `NO_RETRY`，只允许另行 author additive r5 源与合同。
