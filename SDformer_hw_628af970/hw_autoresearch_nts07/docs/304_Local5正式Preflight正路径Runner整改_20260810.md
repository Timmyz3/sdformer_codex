# Local5 正式 Preflight 正路径 Runner 整改

## 1. 第三次复审结论

第三次独立 DATE 复审给出 `3/5 Reject`。H×H topology `[契约审计]` 继续 ACCEPT，
但正路径 packaging 被拒绝，原因是 runner 在执行 preflight 前先跑单测，而单测无条件
断言正式 manifest 不存在。正式 manifest 一旦到达，代码本体虽然能走正路径，runner
却会先因该断言退出。

这是有效 P1。它说明“函数支持两个状态”不等于“可交付 runner 支持两个状态”。

## 2. 双状态不变量

固定入口测试改为只冻结共同不变量：

- window、head-group、H×H task 数量和 digest 不变；
- `admission_generated` 永远为 false；
- manifest 缺失时必须为 `DENY_FORMAL_MANIFEST_ABSENT`；
- manifest 存在时必须为 `PREFLIGHT_PASS_NOT_G0`，五项 artifact binding 齐全，
  canonical key coverage 为 true；
- 任一分支的状态被篡改后，独立输入重放必须拒绝报告。

preflight 增加只用于隔离验证的 `LOCAL5_EREP_PROFILE_DIR` 环境入口。默认值仍是正式
profile 目录；任务 digest、selection SHA、projection contract 和全部严格合同均未放宽。
测试 fixture 必须位于仓库内独立目录，不能覆盖或注入运行中的正式 producer。

## 3. Manifest-Present Runner 集成回归

新增：

```text
tests/build_local5_erep_formal_preflight_v4_positive_fixture.py
sim_qfit/run_local5_erep_formal_preflight_v4_positive_fixture.sh
```

builder 从冻结 selection plan 展开 1200 window 和 13800 个 canonical all-head group，
复制冻结 projection JSON/NPZ，并生成隔离的 ordered payload、cohort 和 manifest。该
fixture 是 `[契约审计]` 正路径夹具，不含真实网络 term payload，禁止写成 `[prof]`。

集成脚本实际调用与正式流程相同的：

```text
sim_qfit/run_local5_erep_formal_preflight_v4.sh
```

而不是绕过 runner 直接调用 Python 函数。runner 的 source-input 列表也改为读取当前
preflight 已冻结的 profile 路径，确保隔离正路径与默认正式路径使用同一打包逻辑。

## 4. 双分支结果

### 4.1 默认正式目录

```text
results/local5_erep_formal_preflight_v4_runnerfix_20260810
```

| 项 | 结果 |
|---|---:|
| Python 单测 | 10/10 PASS |
| status | `DENY_FORMAL_MANIFEST_ABSENT` |
| admission generated | false |
| result SHA / receipt SHA | PASS / PASS |

### 4.2 隔离 manifest-present fixture

```text
results/local5_erep_formal_preflight_v4_positive_runnerfix_20260810
```

| 项 | 结果 |
|---|---:|
| runner 内 Python 单测 | 10/10 PASS |
| status | `PREFLIGHT_PASS_NOT_G0` |
| formal group | 13800 |
| canonical key coverage | true |
| artifact binding | 5/5 |
| source-input 正式文件名 | 5/5 均出现 |
| admission generated | false |
| runner result/receipt SHA | PASS / PASS |
| 外层集成 result/receipt SHA | PASS / PASS |

正路径实际写入 source-input 收据的五项夹具为：

```text
ordered_term_manifest.json
ordered_term_items.npz
ordered_cohort.json
checkpoint_projection_contract.json
checkpoint_projection_contract.npz
```

## 5. 裁决边界

本轮关闭的是“manifest-present runner 是否可达、是否逐文件绑定、是否能形成完整收据”
这一 P1。它不证明隔离 fixture 是真实 workload，也不把 `PREFLIGHT_PASS_NOT_G0`
提升成 G0 admission。

当前正式 producer 的 manifest 仍缺失，所以 formal G0 继续 DENY。真实 13800-group
manifest 到达后，仍需用同一 runner 对真实 ordered payload 执行一次；随后才可进入
head-phase/window-schedule/command 底层账本重放和 T450/OUT_DIM32 Acc32 miter。

Git untracked 属可重建提交问题，当前由逐文件 SHA 和 receipt 保证本机字节身份；未获用户
提交授权前不擅自创建 commit。ASIC DC/STA/SAIF/PPA 也不在本轮证据范围内。
