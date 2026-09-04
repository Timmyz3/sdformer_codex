# M2128｜M2127/M2125 RTL-SAIF 诊断失败独立锤审

## 裁决

**M2127 已消费且失败，禁止重跑；没有 VCS 编译、仿真、SAIF、功耗或论文可引用结果。**

attempt 目录和 failure quarantine 均为穷尽双封。执行计数严格为一次
license query、零次 VCS compile、零次 `simv`、零份 SAIF、零次 DC/PT。
唯一失败记录是 `Failure: timing contamination`，发生在 license preflight
成功之后、VCS compile 计数和进程启动之前。canonical result 不存在，launch
lock 已释放。

## 根因

失败不是实际的 UNIT_DELAY 或 SDF 污染，而是 runner 的 fail-closed 字符串检查
产生了假阳性：

```python
any("UNIT_DELAY" in item or "sdf" in item.lower()
    for item in compile_command)
```

runner 在 `hw_autoresearch_nts07/results` 下创建绝对 work path；仓库路径包含目录名
`SDformer`。重构完全相同形状的 compile argv 后，只有三个动态 pathname 命中：

- argument 9：`-Mdir=/.../SDformer/.../csrc`
- argument 11：`/.../SDformer/.../sources.absolute.f`
- argument 15：`/.../SDformer/.../simv`

三者都不是 SDF option。compile literal option 中没有 `-sdf*`、`+sdf*` 或
`+define+UNIT_DELAY`；冻结的六个 active source/filelist 内容中也没有
`$sdf_annotate` 或 UNIT_DELAY。因此本次失败没有触及 RTL、窗口对齐、ledger 或
SAIF 的正确性，只说明 runner 在进入 VCS 前误拒绝了合法路径。

## M2126 漏检

M2126 的 `no_unit_delay_or_sdf` 检查只遍历 AST 中的 literal constants，并扫描
active source 文本。`-Mdir` f-string、`resolved_filelist` 和 `build/simv` 都是
动态表达式，不在 `compile_constants` 中；审阅也没有构造包含 `/SDformer/` 的
完整 argv mutation。因此 M2126 正确证明了“源码和 literal option 无 SDF”，但
没有证明 runner 的 broad substring predicate 不会与路径名碰撞。这是 source
review 的 P1 覆盖缺口。

## 唯一允许的后继

只能建立全新的 source/result/attempt/lock identity；不得修改后重跑 M2125，更
不得复用 M2127。新 source 需要：

1. 只检查显式 option token（真实 `-sdf*`/`+sdf*`、
   `+define+UNIT_DELAY`），不得把任意 pathname 中的 `sdf` 当作选项；
2. 继续独立扫描 frozen active source/filelist 中的 `$sdf_annotate`、UNIT_DELAY
   或同类显式注释；
3. source hammer 必须新增四类 mutation：`/SDformer/` 合法路径必须通过；真实
   SDF option、UNIT_DELAY define、源码 `$sdf_annotate` 必须分别失败；
4. 保持 M2125 的 RTL/TB/UCLI、slot42 两轴、compile/runtime initreg 两阶段、
   settled-negedge 窗口、完整 ledger、一次 compile/两次串行 simv 预算和
   diagnostic-only 边界。

本锤只授权新 source 的编写，不授权 VCS、DC、PT、license query 或任何论文
数字。

## 身份与边界

- M2125 runner：`6021c4a9...e658815`
- M2126 review：`9949b7f7...bcb910f`
- attempt manifest：`4f210d30...fa8b1d85`
- failure manifest：`d4cd1a06...23c3ab6f`
- docs/359：`dedde7ce...bdfc4`

本次独立锤审未调用 EDA、license 或 GPU；没有修改 M2125、M2126 或 M2127。
