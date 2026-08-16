# Local5 共享 RTL Release 与 Sample1 数值分片

## 1. 本轮结论

Local5 数值 adapter 已从“每个 sample 复制并编译一棵 build tree”改为“一份 sealed
release 服务多个 sample”。sample1 使用该共享 release 完成 12 个真实窗口、四种 head
规模的 pre-bias/pre-BN/pre-requant/pre-residual Acc32 miter：

```text
status = PASS_NUMERIC_SAMPLE_SHARD_NOT_G0
window = 12/12
Acc32 scalar = 1,987,200
mismatch = 0
max_abs_error = 0
regression cycle = 118,036,260
formal G0 = DENY
```

证据等级是 `[rtl]+[软件整数金参考]`。它证明第二个真实样本在当前数值边界上逐整数相等，
不证明 phase ledger、full encoder、ASIC PPA 或 Local5 的最终网络精度。

## 2. 共享 Release

release 目录：

```text
results/local5_erep_numeric_rtl_release_v2_20260811
```

manifest SHA256：

```text
ee1bf0d6001dd963284680faf67be4103f96e8910144def48350ca8136293676
```

release 绑定：

1. 41 个 RTL/TB/verification/adapter 源文件的 SHA256 清单和 tar source bundle；
2. H3、H6、H12、H24 四个 Verilator executable 及 SHA256；
3. 每个 executable 的实际 compile argv 和 log；
4. Python、Verilator、C++ 与 Make 的版本文本；
5. 后续每个窗口的实际 run argv、输入/权重/actual receipt 和 release manifest SHA。

共享 release 的工程意义是消除每个 sample 重复编译约 565 MiB build tree 的开销，并让
不同样本复用完全相同的 DUT binary。该机制是验证基础设施，不是 DATE 架构贡献。

## 3. Sample1 实测覆盖

输出目录：

```text
results/local5_erep_numeric_sample1_shard_v2_release_20260811
```

| stage | block 数 | head | 窗口 RTL cycle 范围 | mismatch |
|---:|---:|---:|---:|---:|
| 0 | 2 | 3 | 687,183--687,600 | 0 |
| 1 | 2 | 6 | 2,274,421--2,291,101 | 0 |
| 2 | 6 | 12 | 8,145,759--8,309,405 | 0 |
| 3 | 2 | 24 | 31,308,215--31,390,642 | 0 |

`result_sha256.txt` 已逐项校验通过，12 个窗口 receipt 均绑定同一个 release manifest。
后续 v5 release 下 sample2 也已通过，当前 formal 数值分片完成度为 3/100；不能因为
sample0/1/2 都通过而宣称 profile100 已闭环。sample2 详见 `docs/325`。

## 4. 独立 DATE 风格复审

共享 release v2 得分 `4.0/5`，裁决为“可作为受限的 RTL build provenance 使用”，P0=0。
复审认可实际执行 argv 和 release SHA 绑定，但给出四类 P1：

1. source bundle 验证仍依赖当前工作目录，verifier 未自动逐成员核对 tar 与 source
   manifest；
2. 工具只绑定版本文本，尚未绑定 executable 绝对路径和 binary SHA；
3. run argv 的验证只检查 plusarg 前缀，未逐值绑定 task plan、vector、actual 路径与
   H3/H6/H12/H24 executable；
4. 缺 executable/tool/tar 篡改、错误 H binary、plusarg value、CWD 漂移、并发创建和
   中断恢复等负向测试。

因此 v2 不作为 100-sample 批量运行的最终 release。下一版必须先关闭以上 P1，再用一个
受限窗口 canary 证明升级没有改变数值合同。

## 5. 下一步

1. 构建 CWD-independent 的 v3 release verifier；
2. 自动解包并逐成员校验 source bundle；
3. 绑定工具路径/SHA 和精确 compile/run argv；
4. 为初次发布增加互斥锁与原子目录发布；
5. 增加负向 mutation tests；
6. v3 通过独立复审后，才恢复剩余 98 个 sample 的批量回放。
