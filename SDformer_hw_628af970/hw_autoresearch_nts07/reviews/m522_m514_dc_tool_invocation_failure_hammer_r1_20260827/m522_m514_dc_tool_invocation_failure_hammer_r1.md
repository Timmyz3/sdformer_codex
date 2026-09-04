# M522/M514 r3 DC 工具调用失败独立审阅

裁决：**失败证据完整，根因唯一，当前没有任何可引用的 DC/STA 结果。** r5 的一次正向授权已经消耗，禁止原样重跑；只允许建立新的 r6 身份，在独立静态审阅 P0=0 后执行一次。

## 授权链与输入身份均未漂移

失败目录中的 runner expected/observed SHA 都是 `375e7602106e46d13520e3e1301254c61e489002030004bac751d7f5fb921a88`，与 r5 独立静态审阅授权的 runner 完全相同。合同本体、失败目录内合同副本和静态审阅记录的 SHA 都是 `203b3f6b6f3820e2d6266366af3b2b473bb5ba5a8573b1eb7ac82001340ede56`。r5 静态审阅的 inner manifest 和 outer seal 均重新校验通过；合同 16/16 个冻结输入、失败目录 `input_sha256.txt` 的 17/17 条身份均由本审阅重新计算一致。

错误 runner-SHA 的负向预检仍返回 10。资源门也确实通过：commit headroom 为 68,330,304 KiB（门槛 33,554,432 KiB），MemAvailable 为 423,412,712 KiB（门槛 67,108,864 KiB）。因此本次失败不是输入漂移、未授权执行或资源不足。

## 根因是启动器 basename 被解析掉

合同冻结的工具关系本身正确：`/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell` 是 raw link text 为 `snps_shell` 的符号链接，解析到 `/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell`；解析目标 SHA 为 `23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2`。

问题在于 runner 第 544 行执行的是解析后的普通文件：

```text
"${m522_dc}" -f "${m522_hw}/${m522_tcl}"
```

其中 `m522_dc=/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell`，而不是 `m522_dc_link=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell`。

安装的 `snps_shell` 不是可忽略调用名的二进制，而是按调用 basename 分派后端的 POSIX wrapper：第 11 行把 `script_name` 初始化为空；只有第 29–40 行的符号链接解引用循环才在第 33 行把它设为原始启动名；第 191–200 行的 `dc_shell` 分支才构造 `common_shell_exec -shell dc_shell`；第 398–400 行的默认分支输出 unsupported 错误并退出 1。直接调用普通文件时不进入符号链接循环，所以 `script_name` 保持为空，产生与 `dc.log` 字节级一致的唯一一行：

```text
Error: The  script is not supported.
```

因此没有进入 `common_shell_exec` 或 Design Compiler 后端，Tcl 和库都没有机会被读取。直接写 `snps_shell -shell dc_shell ...` 也不是合法修复：这个 wrapper 先按 basename 选择 case，`-shell` 参数不会设置 `script_name`。合法的最小修复是用 `dc_shell` 符号链接路径执行 `-f`，同时继续独立冻结并核对 link text、resolved path 和 resolved-target SHA。

## 失败结果与隔离完整

`dc.rc` 为 1，`dc.log` 只有上述一行（SHA `db8e7da6d428906db65cc813663d2345dfeb5a5cacd5ab04e2f80c439af56f39`）。失败目录没有 `reports/`、没有 `netlist/`、没有 DC receipt、没有 `RUN_COMPLETE.txt`；r3 canonical 和残留 staging 均不存在。故本次结果不能支持面积、时序或任何性能主张。

EXIT trap 成功完成 no-follow 隔离：`FAILED_SYMLINK_INVENTORY.json` 状态为 `PASS_NOFOLLOW_INVENTORY_UNLINK_ZERO_SYMLINK`，移动前记录 0 个链接；本审阅对 quarantine 再做不跟随链接的遍历，结果仍是 0。隔离目录含 14 个普通文件、2 个目录（含根），失败标记为 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`，runner exit code 为 1。

## r6 只允许这一处代码变化

新身份建议为：

- 合同：`contracts/m522_m514_c2d_logic_only_dc_contract_r6_20260827.json`，schema `m522_m514_c2d_logic_only_dc_contract_v6`；
- 新 runner SHA；
- canonical：`dc_handoff/runs/m522_m514_c2d_logic_only_dc_3p000ns_r4_20260827`；
- receipt：`m522_m514_c2d_logic_only_dc_receipt_r4.json`。

唯一允许的代码变化是：正向执行使用 `m522_dc_link` 的 `dc_shell` 路径；`m522_dc` 只保留为 resolved executable 身份和 SHA 的检查对象。RTL、filelist、SDC、DC Tcl、双库、M514 VCS 证据、资源/进程/封存根/quarantine 门和 3 ns pre-macro logic-only 主张边界全部不变。

r6 静态打铁必须明确核对正向 `argv[0]` 为精确的 `dc_shell` 路径，静态证明它进入安装 wrapper 的 `dc_shell` 分支，并拒绝任何直接 `snps_shell` 形式；本失败审阅及双 seal 必须成为 r6 冻结输入。P0=0 后仅授权一次正向执行，失败仍消费授权。即使成功，也必须再做独立 receipt-blind DC 审阅，才能引用 additive decoder-support area/timing。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
