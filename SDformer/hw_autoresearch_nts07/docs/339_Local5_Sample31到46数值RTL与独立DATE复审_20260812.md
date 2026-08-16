# Local5 Sample31 到 46 数值 RTL 与独立 DATE 复审

> 日期：2026-08-12  
> 正证据：`results/local5_numeric_samples31_46_batch_v2_20260812/`  
> 证据等级：`[rtl]+[软件整数金参考]+[rtl-build-provenance]`  
> formal G0：**DENY**

## 1. 本轮结论

sample31 至 sample46 已用同一个 sealed v5 Verilator release 完成真实运行，并通过独立
DATE 证据审阅：

| 项 | 结果 |
|---|---:|
| 样本 | 16，全部 `execution=RUN` |
| canonical block-window | 192，即每个样本 12 个 |
| Acc32 | 31,795,200 |
| mismatch | 0 |
| 共同 release manifest SHA256 | `c620cf6a33f1c9bbdb1c7d85ba0fa485580f8f578287850d08b7c6ee52939bf9` |
| 独立裁决 | **4.5/5，Conditional Accept** |

本轮允许新增 16 个 numeric sample 证据。由于 sample15 至 sample30 的并行批次在本轮
审阅时仍未全部封存，正式已独立审计覆盖暂时只能从 15/100 提升到 **31/100**，禁止
提前写成 47/100。

## 2. 独立重算

独立审稿代理没有只读取 batch summary，而是完成以下重算：

- 16 份 NPZ 的 `expected_acc32` 与 `actual_acc32` 全量逐元素比较；
- sample31、38、39、46 的 schema、dtype、shape、拓扑和 offset 单独复核；
- 以固定随机种子 `20260812` 抽取四个归档切片，与窗口 `actual.memh` 比较：
  sample32/S3B0、sample46/S2B5、sample35/S3B1、sample43/S2B2；
- 20/20 batch receipt、16 份 sample receipt、16 份 shard complete/report/archive、
  192 份窗口 executable/filelist 绑定全部通过；
- 192 个窗口均与预冻结 `joint_window_selection_plan.json` 一致；日志均恰好包含
  12 条 `PASS_WINDOW`，无 `RESUME` 或 `SKIP`。

四份重点 NPZ 均为 schema version 4，单样本包含 1,987,200 个 `int32` Acc32，公共
offset 为：

```text
0, 43200, 86400, 172800, 259200,
432000, 604800, 777600, 950400,
1123200, 1296000, 1641600, 1987200
```

## 3. 审稿问题与修复

### P1：连续覆盖尚未成立

独立审阅时 sample23 至 sample30 尚未形成完整批次，因此只能声明：

```text
已独立审计：sample0-14 + sample31-46 = 31/100
禁止声明：sample0-46 连续闭合 = 47/100
```

sample15 至 sample30 必须等待批次 complete、独立 NPZ/`actual.memh` 复核和来源审计，
通过后再更新累计覆盖。

### P2：外层 launcher 未进入本批 source snapshot

本批封存了 batch runner 和其单测，但真正调用的工作树
`sim_qfit/run_local5_erep_numeric_sample_shard.sh` 未作为 batch source snapshot 绑定。
工作树 launcher SHA 与 v5 release 内历史副本 SHA 不同：

```text
live launcher: cc507bc8ca6dbf67273f06b3f9fe38bc8586fbf89a94a64b87ee3e8fbdcea293
release copy : 472ccf65dbeab386b93258a4d8de811cd46f2906a135654c41b5b34df03e4492
```

差异位于 release 创建和 service-mode 参数；本批实际数值执行继续使用 v5 release 内
封存的 runtime 与 executable，所以不推翻数值结论，但 provenance 不够完整。

后续批处理器已经旁路修复：启动前将 live launcher 复制为只读 source snapshot，写入
plan/receipt，并在每个 sample 执行前后校验 live SHA；变更后 6/6 单元测试和 Python
语法检查通过。该修复只约束后续新批次，不倒签本批证据。

### P2：缺外部不可变 trust root

哈希清单与被哈希文件仍位于同一本地可写文件系统；最终投稿归档应再用 Git commit、
只读归档 SHA 或外部对象存储锚定。绝对路径也应在最终可复现包中转换为归档相对路径。

## 4. 证据边界

- `verification_regression_cycles=1,883,215,151` 仅表示验证环境记账，不能作为部署
  latency、throughput、speedup 或硬件性能；
- 每个样本只覆盖预选的 12 个 canonical block-window，不是所有 full-resolution
  空间窗口；
- 本轮没有生成 formal phase ledger、admission receipt、DC/STA/SAIF 或 ASIC PPA；
- numeric coverage 增长不改变 formal G0，仍为 **DENY**；
- 本轮是验证证据扩展，不是新的 DATE 架构创新。

## 5. 下一步

1. 收口并独立复审 sample15 至 sample30；
2. 通过后才把连续 numeric coverage 更新为 47/100；
3. 后续 sample47 起使用含 launcher fail-closed 封存的新批处理合同；
4. numeric 证据继续与 H24 phase telemetry、formal archive/admission 分账。
