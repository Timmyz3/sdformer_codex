# Local5 Sample0 正式 Numeric Shard 与恢复链路收口

> 日期：2026-08-11  
> 证据：`[rtl]`、`[软件整数金参考]`、`[prof]`  
> 数值边界：theta-folded INT8 的 pre-bias/pre-BN/pre-requant/pre-residual Acc32  
> 裁决：sample0 numeric shard 通过；`formal G0 = DENY`

## 1. 为什么重做

旧 H3/H6/H12/H24 smoke 虽然数值零失配，但独立 DATE 复审发现：

1. TB 内部 stage/block/window tag 固定为 0；
2. H6/H12/H24 只覆盖各 stage 的 block0；
3. 四个结果不是同一冻结源码和同等级 provenance；
4. 软件 expected 依赖 manifest 行顺序；
5. Acc32 的 bias/requant 边界表述不够明确。

因此旧结果只保留为 head 数和容量 smoke，本轮从 sample0 的 12 个真实 block 重新
建立 numeric shard。

## 2. 合同整改

### 2.1 真实控制坐标

验证 TB 新增运行时：

```text
+STAGE_ID
+BLOCK_ID
+WINDOW_ID
```

TB 检查 stage-to-H 映射、block 范围和 9-bit window 范围，并把真实坐标送入
`tile_start_*` 和 `head_job_*`。终止日志必须唯一打印同一坐标，actual adapter 再与
task plan 比较；任一不一致直接失败。

### 2.2 Manifest 行序独立

软件 expected 不再要求 manifest 行本身按 head 排序，而是按 head canonical key
建表、拒绝重复和缺失，再按 `head=0..H-1` 拼 projection 输入通道。新增乱序正测和
重复 head 负测。

### 2.3 数值边界

checkpoint contract 的原始 scope 已明确：

```text
K_binary event
  -> theta_K 离线折入 W
  -> per-output-channel dyadic INT8 weight
  -> cross-head integer Acc32
```

本轮 miter 停在 pre-bias/pre-BN/pre-requant/pre-residual Acc32。bias 存在且未被忽略
为零，而是明确位于本次验证边界之外。当前结果不能称为完整部署输出或网络逐 bit
等价。

## 3. 三方来源隔离

```text
software expected
  producer destination-major item_*
  + theta-folded checkpoint INT8 weight
              |
              v
read-only shard merge <--- DUT actual
                         descriptor -> Q/K/mask
                         -> score/Shiftmax5
                         -> relation transpose
                         -> source-major term
                         -> TCFM5 projection
                         -> cross-head Acc32
```

- expected 不使用 descriptor 方向反演；
- actual 在 `NO_ACC_CHECK` 下不读取软件 expected；
- merge 不读取原 profile，只读取 12 组 task plan、expected、actual 和 receipt；
- 两条路径仍共享同一个 producer lineage，因此只能称“adapter 路径隔离”，不能称
  “完全独立模型 oracle”。

## 4. 可恢复 Runner

入口：

```bash
SAMPLE=0 \
OUT_DIR=results/local5_erep_numeric_sample0_shard_v1_reviewfix_20260811 \
bash sim_qfit/run_local5_erep_numeric_sample_shard.sh
```

一个 release 编译四个 executable：H3/H6/H12/H24。每个窗口按以下顺序完成：

```text
expected -> vectors -> Verilator actual -> actual receipt
         -> window_complete.json.tmp -> atomic replace
```

没有 `window_complete.json` 的目录在重启时会被清理重跑；已有完成标志的窗口必须
通过 artifact SHA、tool SHA、executable SHA 和源码闭包复验后才能复用。

本轮真实经历一次恢复负结果：旧 runner 在恢复时覆盖了带 UTC 时间戳的
`tool_versions.txt`，导致所有 actual receipt 在 merge 时被正确拒绝。整改后：

1. 有已完成窗口时保留原工具文件，只比较当前工具版本正文；
2. 四个 executable 按 `build_sha256.txt` 复验后复用；
3. 工具或二进制变化时拒绝恢复并要求新 `OUT_DIR`；
4. 12 个窗口全部 `RESUME`，约 13 秒内重新完成只读 merge。

该负结果证明恢复路径是 fail-closed，而不是只验证正常路径。

## 5. Sample0 结果

结果目录：

```text
results/local5_erep_numeric_sample0_shard_v1_reviewfix_20260811
```

总结果：

| 指标 | 结果 |
|---|---:|
| 真实 block/window | 12/12 |
| input-head group | 138 |
| pre-bias Acc32 | 1,987,200 |
| mismatch | 0 |
| max abs error | 0 |
| 固定验证 service 总周期 | 118,052,500 |
| 12 窗纯 Verilator 墙钟合计 | 1,019.75 s（16.996 min） |
| miter NPZ | 15.16 MiB |
| 含四份 build/向量/日志的试运行目录 | 约 641 MiB |

逐 block：

| stage/block | H | window | Acc32 | cycle | mismatch |
|---|---:|---:|---:|---:|---:|
| 0/0 | 3 | 94 | 43,200 | 695,572 | 0 |
| 0/1 | 3 | 304 | 43,200 | 687,324 | 0 |
| 1/0 | 6 | 54 | 86,400 | 2,273,821 | 0 |
| 1/1 | 6 | 44 | 86,400 | 2,274,825 | 0 |
| 2/0 | 12 | 21 | 172,800 | 8,333,096 | 0 |
| 2/1 | 12 | 18 | 172,800 | 8,181,005 | 0 |
| 2/2 | 12 | 9 | 172,800 | 8,149,717 | 0 |
| 2/3 | 12 | 6 | 172,800 | 8,145,973 | 0 |
| 2/4 | 12 | 23 | 172,800 | 8,277,737 | 0 |
| 2/5 | 12 | 13 | 172,800 | 8,482,025 | 0 |
| 3/0 | 24 | 3 | 345,600 | 31,213,193 | 0 |
| 3/1 | 24 | 0 | 345,600 | 31,338,212 | 0 |

这些 cycle 是 transaction-indexed 验证服务、完整 result serialization 和文件输出
条件下的回归延迟，不是部署吞吐或 EREP 性能。

## 6. Archive 与 SHA 闭包

只读 merge 生成：

```text
shard/acc32_miter_shard.npz
shard/numeric_shard_report.json
```

miter archive 使用 v4 冻结成员、dtype 和坐标顺序：

```text
stage/block/window/output_tile/source/out
```

写盘后重新检查 ZIP member 顺序/编码，再以 `allow_pickle=False` 读回并执行
`parse_miter_archive`。结果为：

```text
12 windows
1,987,200 Acc32
0 mismatch
```

从仓库根目录执行 `receipt_sha256.txt`、`result_sha256.txt`、12 个窗口 receipt 和四个
build receipt 的 `sha256sum -c` 均通过。

## 7. 回归

联合执行 formal preflight、archive replay、ledger replay、旧 canary、新 tag canary
和 numeric shard 单测：

```text
44 tests / 44 PASS
```

H3 新 tag canary 还以同一源码通过 Icarus 与 Verilator：

```text
stage=0 / block=0 / window=94
43,200 Acc32 / simulator
cross-simulator raw SHA 相同
mismatch=0
```

## 8. 能证明和不能证明的内容

### 已证明 `[rtl]`

- sample0 的 12 个真实 stage/block/window 坐标全部执行；
- 同一源码下四个 H-class executable 覆盖 H={3,6,12,24}；
- 12 个 block 各自的 descriptor、权重和跨头 Acc32 逐项零失配；
- 单 sample miter archive 可写、可读、可恢复、可按 SHA 审计；
- 软件 expected 不依赖 manifest 行顺序。

### 未证明 `[待验证]`

- 其余 99 个 sample 和总计 1,200 个窗口；
- 462,600 条正式 phase/event ledger；
- relation memo 的目标性能路径，本轮仍为 `USE_MEMO=0` recompute baseline；
- 多反压种子、Icarus 的 H6/H12/H24 分层交叉验证；
- bias/BN/requant/residual/ATLIF 的部署数值闭环；
- EREP 性能、full encoder 性能和 ASIC PPA。

## 9. Formal G0 裁决

```text
Gate A producer              PASS
Gate B formal preflight      PASS_NOT_G0
Gate C sample0 numeric       PASS_NOT_G0（1/100）
Gate C phase ledger          DENY
Gate D full archive          DENY
formal G0                    DENY
EREP candidate RTL           DENY
```

下一最高优先级不是立即跑剩余 99 个 sample，而是先决定并实现 phase/event capture：
numeric archive 已可扩展，formal G0 当前真正的结构性缺口变成 462,600 条 phase ledger
与三层只读重放。没有这条链，即使 100/100 numeric 全过也不能生成 admission receipt。
