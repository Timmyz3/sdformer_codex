# M1059 独立打铁：M1058 C2 K1 reset-hygiene source/release

结论：**GO：源修复与当前非启动 production candidate 通过，允许 M1068 撰写一次性 release。**

## 源修复与独立 VCS

独立核对 exact SHA/侧封后，对模块名做归一化，并仅删去新增的同步 reset 块：service 与旧 M519 逐字符相同；standalone、K1 top 与 matched shell 同样归一化相同。新增的状态只有 `fifo_tag_q`、`fifo_block_q`、`fifo_bank_id_q`、`fifo_channel_q` 四组 FIFO payload 在 reset 分支清零。K8 与 K1x8 实例及源文件保持冻结。

本评审未复用 M1058 `simv`，在私有 `mktemp` 目录重新编译三个 VCS 二进制，独立重放：

- 新 K1 五个 case：`259/737/3153/7569/14` cycle，全部 mismatch 为 0；
- 旧 K1 五个 case：同一周期锦标，全部 mismatch 为 0；
- 默认初态下 5 组 reset 长度/相位攻击 PASS；
- `+vcs+initreg+random` 只作验证攻击，5 组冻结 seed/reset 矩阵 PASS。

## Production anchor 与 launch 边界

`tb_m1058_c2_k1_reset_hygiene_mapped_gate_case.sv` 的 `expected_cycle()` 对 axis 0（K1）已明确冻结 `259/737/3153/7569/14`，`final_checks()` 会强制逐 case exact equality。末尾 `return -1` 只是未知 axis 的防御性默认，不会绕过 K1。

当前 candidate 保持 `status=PREPARED_NOT_RELEASED` 与 `launch_now=false`，production assets 不含 `+vcs+initreg`。本锤只授权 M1068 撰写 exact 一次性 release，本身不直接启动 DC/mapped VCS/SAIF/PTPX。

## Fail-closed 与红线

删 reset、改数据路径、改周期锦标、允许 initreg、错 SHA/status、`launch_now=true`、改 K8/K1x8 实例等 10 类攻击全部被拒。M1046 继续是 consumed/failed/DO_NOT_RETRY，完成 gate case=0、production SAIF=0，M1068 不得复用旧 namespace。

本评审未运行 DC、mapped-gate VCS、SAIF 或 PTPX；不主张 mapped X 已修复、功耗、能量、系统加速或 paper-PPA ready。`docs/359` 未修改。
