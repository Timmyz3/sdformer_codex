# M1669｜M1668 TSBG runtime-closed entity-rebind source 独立审阅

## 裁决

**PASS 98/100**：`PASS_M1669_M1668_RUNTIME_CLOSED_ENTITY_REBIND_SOURCE__AUTHORIZE_RELEASE_AUTHORING__NO_CAPTURE`。P0=0、P1=0、P2=1。

M1668 可授权不同作者编写 M1670 release，但本回执不授权 capture、GPU、远端写、attempt 消费、EDA 或自动重试。M1670 及其不同作者 launch hammer 仍必须把远端目标、M1257 安装顺序和一次 attempt/no-retry 约束写成可执行 authority。

## 已闭合证据

- source、test、selection、contract、author receipt 及其 sidecar/outer seal 全部精确重算；作者回执递归人口为 7 个文件，无未封成员。
- 作者回执内的远端只读 `build_runtime` 证据绑定 `ssh.sd5ai.scnet.cn:10037`、`root`、`/root/private_data/work/sdformer_codex/SDformer`，checkpoint/config 路径与 SHA 精确一致；该观察产生 0 GPU run、0 attempt write。
- parent 顺序为 predecessor/runtime identity 检查 → `build_runtime` → parent budget/delegate；注入 `build_runtime` 失败会在 parent budget 前 fail closed。
- child 在 GPU/attempt delegate 前重复 predecessor、entity 与 `build_runtime` 检查；下层 clean child 的顺序固定为 `build_runtime → GPU lease → O_EXCL attempt → checkpoint/model`。
- `_bound_exact_m1647` 只在受控上下文中将 lower loader 绑定到 rebound runtime，退出后恢复原 loader；因此不会永久污染共享模块。
- lower M1624 使用 fresh result/attempt/work/failure namespace，attempt 以 `O_EXCL` 消费，失败后不自动重试。

## 独立打铁

- CPython 3.6.8 与 3.12.13 的规范化输出逐字段一致。
- 两个解释器各通过 18/18 个 M1668 源测试。
- `source_self_check()` 返回 `PASS_M1668_SOURCE_SELF_CHECK__RUNTIME_HANDOFF_CLOSED__NO_CAPTURE`。
- 11 类突变全部被拒：缺失 M1257 runtime tar、config inode 漂移、config SHA 漂移、`build_runtime` 失败、打开 retry、双 parent、双 GPU、checkpoint 重选、config 语义改变、remote host 漂移、remote root 漂移。
- 本审阅未连接远端、未读取 checkpoint payload、未启动 capture/GPU/EDA、未写 attempt，也未 commit/push。

## P2 与 M1670 硬门

远端 `build_runtime` 是被双封的只读瞬时观察，不是 launch authority。M1670 必须同时满足：

1. 精确固定 host=`ssh.sd5ai.scnet.cn`、port=`10037`、user=`root`、repo=`/root/private_data/work/sdformer_codex/SDformer`；
2. 在 M1668 source preflight 前安装并核验 M1257 canonical runtime；
3. parent 与 child 各自重复 checkpoint/config entity 和 `build_runtime` 闭包；
4. 一次 attempt、一次 child、一次 GPU、一次 capture，任何失败均禁止自动 retry；
5. 由不同作者 release hammer PASS 后，方可授权真正远端 launch。

本里程碑仍是 source-only admission；没有产生 TSBG AEE、cycle、traffic、energy、RTL 或 system-speedup 数字，禁止进入论文性能表。
