# M1228 — Motion 跨 run 最终 checkpoint 选择与硬件重绑定 binder（source-only）

## 结论

M1228 source-only 版本完成，作者侧 15 项测试全部通过。它没有执行生产
binder，也没有读取远端或实际 checkpoint，因此**当前仍未选择最终
checkpoint**。合法下一步是不同作者 source hammer；只有 ep29、ep30、ep32、
ep34 四个真实 strict-valid825 候选全部存在并通过门槛后，才可另行授权一次
生产选择。

## 相对 M1167 的实质修正

M1167 假设所有候选位于同一 run 目录并使用同一配置。M1228 不复用该假设，
而是明确建模两组身份：

| 候选 | epoch | run | config SHA |
|---|---:|---|---|
| legacy_ep29 | 29 | `date_two_contribution.../c12_binary_motion_ttx` | `c7b5b994...` |
| resume_ep30 | 30 | `dsec_c12_alpha0125_ep29_resume5_20260830` | `630e735c...` |
| resume_ep32 | 32 | 同上 | `630e735c...` |
| resume_ep34 | 34 | 同上 | `630e735c...` |

旧 ep29 checkpoint 额外固定 SHA `2144dfd6...`。新 run manifest 必须以严格
非布尔整数、有序列表声明 `evaluation_epochs=[30,32,34]`。缺任何一个候选均
停止，不会从当前已存在的子集提前选 winner。

## 准入与选择

每个候选均要求：

- `samples` 是精确非布尔整数 825；
- 4 个 checkpoint load-audit counter 是精确非布尔整数零；
- 模块集合精确为 105 ATLIF 与 12 attention；
- profile 内嵌 checkpoint 路径/SHA/size/mtime 与 config 路径/SHA 精确匹配；
- binder 独立稳定计算并输出 profile、checkpoint、config 各自的
  SHA/size/mtime；
- 文件必须是普通非符号链接，且哈希前后 size/mtime/inode/device 不变。

四者全部准入后才按 exact AEE 最小选择；AEE 相等时只按 epoch 较低者优先。
输出中既包含 selected checkpoint，也包含 selected config，避免把新 checkpoint
错误绑定到旧配置。

## E0–E8 收口规则

输出覆盖 E0–E8 九个 activation/weight-dependent target。统一规则是：只有既有
artifact 的独立 seal 精确绑定 selected checkpoint 与 selected config 的
SHA/size/mtime 时才允许复用，否则必须 invalidation 后重新 capture/replay。
因此如果新 ep30/32/34 胜出，旧 ep29 的 capture、Conv/decoder ledger、ATLIF/FC/
patch/BN 活动、RQTB、SAIF/PTPX 与 weight/range 证据不会被静默继承；若 ep29
胜出，也可以通过 exact identity proof 有条件复用，而不是无条件全部重跑。

## 测试与边界

15 项 unittest 覆盖跨-run winner、跨-config 输出、低 epoch tie-break、双 seal，
以及缺候选、候选集合漂移、manifest 缺失/额外/浮点/重排、重复 JSON key、配置
SHA、旧 checkpoint SHA、全部 artifact identity 字段、四个 load-audit 零类型、
模块数、非有限 AEE、symlink 与输出覆盖攻击。全部只使用临时合成文件。

本包不声明 valid825 已完成，不声明最终 checkpoint，不授权 E1–E8 重绑定，也不
提供 accuracy、cycle、speedup、energy 或 PPA 结果。`docs/359` 未修改。
