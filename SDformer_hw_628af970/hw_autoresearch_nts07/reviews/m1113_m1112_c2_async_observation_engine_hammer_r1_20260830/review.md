# M1113：M1112 C2 async-observation source/engine 独立打铁

## 裁决

**NO-GO。固定 M1112 身份需要重新作者化和重新封存；不得创建 launcher、launch receipt、attempt，也不得运行 EDA。**

本轮严格限定为 source/static/变异审阅。未调用 VCS、DC、simv、任何 launcher，也未消费 production attempt。固定 wrapper、TB、engine、contract、作者回执与 M1109 历史权限链均核验通过，`docs/359` 保持冻结 SHA。

## 已通过的部分

- wrapper 有且仅有 13 个独立 shadow accounting 寄存器，共 337 bit；统一采用 `posedge clk_core or posedge rst_core`，并在 `header_accept` 开启新诊断 epoch。
- 13 个 frozen synchronous-debug 输出只接入 `unused_frozen_debug_*`，未进入 `obs_*`；shadow/observation 也未反馈进 ready/valid/payload/memory/accumulator 功能锥。
- TB 在同一个组合块中原子计算 22 个 `$isunknown` predicate，保存 first-X，同时持续累积 union，到第 128 cycle 才 fail closed。
- 删除第 22 个 predicate、删除 union、在 first-X 分支提前 `$fatal`、在窗口关闭前提前 `$fatal`，四类变异均被独立检查拒绝。
- mapped 结构 gate 能拒绝 D/CP-only flop 和 336-bit 少一位 census；M1109 的唯一历史目录内 symlink 例外仍能按封存字节验证。
- 固定 engine argv、未来 zero-argument parent、一次 attempt consume 和新 namespace 未创建边界均存在。

## 两个 P0

### M1113-P0-01：async reset 只验端口名，未验来源

固定 engine 的 `structural_reset_gate()` 仅要求每条 shadow 实例语句含 `.CDN/.CN/.SDN/.SN`。独立合成的 337-bit 假 netlist 把所有 `.CDN` 接到 `unrelated_fake_reset`，现有 gate 仍返回 ACCEPTED。

这不满足 contract 的“复位 pin 必须来自 `rst_core` 或其单次反相”。修复必须解析 async pin 网络来源，并拒绝常量、无关 net、数据逻辑、多级或重汇合 reset cone。

### M1113-P0-02：live seal 元数据可为 symlink

`verify_double()` 没有对 `.sha256` sidecar 调用 `verify_regular()`；`verify_flat()` 没有对 `SHA256SUMS` 调用 `verify_regular()`，也不核对实际成员与 manifest 的精确集合。

独立变异证明：symlink sidecar 能被 `verify_double()` 接受；symlink manifest 加一个未列出的 `manifest.real` 能被 `verify_flat()` 接受。这违反“历史 quarantine 内唯一成员 symlink 例外、所有 live input 严格 regular”的边界。

修复必须对所有 live primary/sidecar/manifest/outer 做 `lstat + regular + !symlink`，并要求 sealed directory 精确成员覆盖。历史例外仅保留 M1109 已封存 quarantine 内的那一个成员链接。

## 后续唯一合法顺序

1. 以新 namespace 作者化 M1112r2 engine/contract/author receipt，补齐 reset provenance traversal 和 live seal metadata 检查。
2. 由不同作者重新做 source/engine hammer。
3. 只有新 hammer 为 GO，才允许作者化 zero-argument launcher；仍不能直接消费 attempt。
4. launcher 还需 `env -i` 等价的固定最小环境，并接受下一轮独立 launch hammer；之后才可能授权唯一一次 fresh attempt。

当前没有 mapped functionality、性能、功耗、系统倍速或 paper PPA 准入。
