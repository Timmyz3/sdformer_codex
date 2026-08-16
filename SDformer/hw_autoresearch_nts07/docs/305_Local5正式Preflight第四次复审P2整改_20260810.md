# Local5 正式 Preflight 第四次复审 P2 整改

## 1. 复审裁决

第四次独立 DATE 复审给出 `4/5 Accept`：

- H×H topology preflight：ACCEPT，证据仅为 `[契约审计]`；
- manifest-present packaging：ACCEPT，依据隔离正 fixture 的同 runner 端到端结果；
- formal G0：DENY，正式 manifest 仍缺失；
- P0/P1：0/0。

复审还指出两个可修 P2：projection JSON 只冻结结构和内部 payload SHA，未预注册
JSON 本身的字节 SHA；正路径 runner 最后一行日志仍硬编码 `status=DENY`，容易把
preflight 状态与 formal G0 状态混淆。

## 2. Projection 字节冻结

正式 preflight 新增两个预注册常量：

```text
projection JSON SHA-256
c2bf6f406345d1bcc0f8a883318f59dc63116a96c96cd4138af83ce495ce9669

projection NPZ SHA-256
81edeefa16d2177c8739579f42485a58f2a70581a078e9ea367d7422f73446f4
```

环境变量只能切换 profile 目录以执行隔离 fixture，不能改变这两个 SHA、selection SHA、
210600 task 数量或 task digest。修改 `topology_contract`、block module 或任一未逐字段
解析的 JSON 内容都会先因文件 SHA 不同而 fail closed。

## 3. 状态日志消歧

runner 末行现在分别打印：

```text
preflight_status=<DENY_FORMAL_MANIFEST_ABSENT|PREFLIGHT_PASS_NOT_G0>
formal_g0=DENY
```

因此正 fixture 不再显示成模糊的 `status=DENY`。`PREFLIGHT_PASS_NOT_G0` 只表示输入
包装合同通过，仍明确不生成 admission。

## 4. 证据边界

本次只是对已 ACCEPT 的 runner 契约补强，未新增 `[prof]` 或 `[rtl]` 证据。真实
producer manifest、底层账本重放、T450/OUT_DIM32 Acc32 miter、候选 RTL 和 ASIC
PPA 仍未由本轮放行。相关源文件的 Git untracked 状态维持 P2；未获用户提交授权前不
擅自创建 commit，本机字节身份继续由 source/result/receipt SHA 保证。
