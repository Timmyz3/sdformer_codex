# Local5 Phase Array Store 流式扩展性与 H24 资源准入

> 日期：2026-08-12  
> 范围：Local5 identity-service phase trace 的验证归档基础设施  
> 证据：`[rtl-trace-derived]`、`[独立软件逐行验证]`、`[资源实测]`、`[模型]`  
> formal G0：**DENY**

## 1. 本轮关闭的问题

H12 参数化 identity/phase canary 已有 12,244,663 行真实 RTL trace，但旧
`generate_local5_h3_phase_template_patch_v1.py` 使用 `rows=list(reader)`，把全部 CSV
行保存为 Python 字典。H12 运行期间曾观察到约 10.9 GB RSS；该数字没有
`/usr/bin/time` 密封，只作为问题定位记录，不能作为论文主表。

本轮目标不是新增 RTL 或架构机制，而是回答：

1. H12/H24 phase trace 能否在有限内存下生成和验证；
2. 新表示是否与旧 NPZ、原始 RTL trace 逐值等价；
3. H24 能否不依赖旧高内存生成器；
4. identity、源码和篡改负例是否进入可审计信任链。

## 2. 设计

### 2.1 两遍生成与字段分片

第一遍只顺序提取边界、事件/来源/payload 字典和计数；第二遍按 65,536 行批量写入
27 个有类型 `.npy` 字段文件。逐行 patch 字段采用 mmap，避免常驻全部 row object。

```text
candidate_trace.csv
  -> pass 1: boundary + dictionary + exact counts
  -> segment plan: prefix/head/tile/suffix 七类
  -> pass 2: 65,536-row vector batch
  -> 27 typed .npy arrays + manifest SHA
```

新增四个只读 identity array：

- `identity_sample`
- `identity_stage`
- `identity_block`
- `identity_window`

`heads` 继续作为独立 array。验证器必须同时比较 runner 参数、identity-service v4
manifest、独立 verification receipt、trace 内 manifest/receipt 双 SHA、store manifest
和 typed arrays。因此只改 manifest，或同时改 store manifest/array 并重绑 SHA，均不能通过。

### 2.2 页驻留负结果与修复

初版虽然使用 mmap，但没有主动释放已处理页。H12 正确性通过后，legacy/source-only
verifier 峰值仍分别为 459,908/405,720 KB，已经接近 512 MiB 门槛；直接按文件容量
外推 H24 不成立。

最终版采用：

- generator：每 1,048,576 行 flush，并对 patch mmap 执行 `MADV_DONTNEED`；
- verifier：每 262,144 行释放所有已触碰 mmap 页；
- legacy array 比较后再次释放新 store 映射；
- 平台不支持 `MADV_DONTNEED` 时 fail-closed。

这不是硬件页面管理机制，只是验证工具的内存扩展性修复。

### 2.3 独立验证

验证器不导入生成器，独立执行：

1. 27 个 array 的文件集合、dtype、shape、nbytes、file bytes 和 SHA；
2. manifest/array 的完整 identity；
3. 七类 instance 顺序、offset、模板长度和派生统计；
4. 逐行展开并与原始 CSV tuple 比较；
5. 重建完整 CSV 字节流 SHA；
6. 与外部 identity-service manifest/receipt 和 trace 内双 SHA 交叉验证；
7. 可选地与旧 NPZ 的 23 个公共数组逐值比较，仅允许 `schema_version 1 -> 2`。

旧 NPZ 已降为可选交叉证据。H12 同一包额外执行一次不传旧 NPZ 的 source-only
全量重放，证明 H24 不需要先运行旧生成器。

## 3. H12 正确性结果

正式包：

`results/local5_h12_phase_array_store_v2_20260812/`

| 项 | 结果 |
|---|---:|
| identity | sample1/stage2/block2/window21/H12 |
| 原始/展开行 | 12,244,663 |
| 新 store arrays | 27 |
| 旧 NPZ 公共 arrays | 23 |
| 公共数组 mismatch | 0 |
| legacy 模式展开 SHA mismatch | 0 |
| source-only 模式展开 SHA mismatch | 0 |
| store 文件字节 | 318,936,649 |
| 旧 NPZ 文件字节 | 318,939,163 |
| formal G0 | DENY |

新格式目标是可流式访问和可独立验证，不是文件压缩；两者文件大小约相等是预期结果，
不得写成存储节省。

## 4. 篡改回归

10/10 负例被拒绝：

| 负例 | 是否重绑 SHA | 拒绝来源 |
|---|---|---|
| patch offset 改写 | 是 | instance/template 语义 |
| payload code 改写 | 是 | 逐行 trace 等价 |
| template event 改写 | 是 | 逐行 trace 等价 |
| identity relabel | manifest 改写 | manifest/array identity |
| identity 双边改写 | manifest+array 改写并重绑 | frozen manifest/receipt expected identity |
| source trace 替换 | 参数替换 | trace SHA/bytes |
| verifier source 替换 | 是 | 当前/归档 verifier SHA |
| manifest 派生统计改写 | manifest 改写 | 独立重算统计 |
| 额外 array | 否 | exact member set |
| 缺失 array | 否 | exact member set |

准确口径是“10/10 篡改拒绝，其中 patch offset、payload、template 和 identity 双边改写
包含 array SHA 重绑”，不能写成“10 项全部是 SHA 重绑数组篡改”。

## 5. 资源实测

环境与限制：Python 3.12.3、NumPy 1.26.4、Parastor 文件系统、swap=0；单次运行，
主机负载前后有变化。因此 RSS 可直接比较数量级，wall-time 不做优化加速声明；
user CPU 仅作为工具预算。

### 5.1 H12 page-drop 前后

| 阶段 | pre-drop RSS | 最终 RSS | 下降 |
|---|---:|---:|---:|
| generator | 211,204 KB | 154,968 KB | 26.63% |
| legacy verifier | 459,908 KB | 149,040 KB | 67.59% |
| source-only verifier | 405,720 KB | 144,688 KB | 64.34% |
| tamper runner | 173,160 KB | 143,520 KB | 17.12% |

最终 generator/legacy/source-only 均低于 512 MiB。page-drop 前后 27/23 数组、
12,244,663 行、CSV SHA 和 10/10 负例完全相同。

wall-time 受共享机器负载影响，不可作公平 speedup；user CPU 的变化为 generator
`+1.02%`、legacy verifier `+2.37%`、source-only verifier `-6.03%`，说明内存降低没有
引入数量级计算开销。

### 5.2 H3 到 H12 扩展

| 项 | H3 | H12 | 倍率 |
|---|---:|---:|---:|
| trace rows | 862,507 | 12,244,663 | 14.20x |
| unique payload | 56,467 | 385,057 | 6.82x |
| generator RSS | 78,772 KB | 154,968 KB | 1.97x |
| source-only verifier RSS | 56,392 KB | 144,688 KB | 2.57x |

RSS 不再与 trace 行数或 array store 文件大小线性增长；主要剩余常驻项是唯一 payload
字符串字典。

## 6. H24 资源预算

机器可读/中文报告：

`results/local5_phase_array_store_h24_budget_v1_20260812/`

七类模板长度在 H3/H12 相同。对 H24：

| 指标 | 预算 |
|---|---:|
| phase rows | 47,941,735 |
| instances | 1,177 |
| unique payload | 1,194,625 |
| array store 文件字节 | 1,227,490,117 |
| 保守原始 trace（64 B/row） | 3,068,271,105 |

H3/H12 两点模型加 20% 保护后的 RSS：

| 项 | 预测 | 20% 保护 | 512 MiB |
|---|---:|---:|---|
| generator | 334.7 MiB | 401.6 MiB | PASS |
| source-only verifier | 353.7 MiB | 424.5 MiB | PASS |

运行前裁决是 `CONDITIONAL_ADMIT_H24_RESOURCE_ONLY`：只允许在资源门槛下尝试 H24。

H24 后续已完成真实运行：47,941,735 行候选 trace、345,600 个 Acc32 零失配、
10/10 负例；generator/verifier 实测 RSS 为 377,836/407,636 KB，均低于 512 MiB 和
20% 保护值。详见 `docs/333`。这只把 H24 单窗从 `[模型]` 晋级为
`[rtl]+[资源实测]`，不改变 formal G0。

## 7. DATE 证据边界

本轮：

- 关闭验证基础设施的 H12 内存扩展性风险；
- 提高 Local5 H12/H24 正确性证据可复现性；
- 不增加架构 novelty 分数；
- 不证明 Local5 性能、能量、面积或 full encoder 收益；
- 不改变 formal G0=`DENY`；后续 `docs/334`、`docs/336` 已把 numeric shard 推进到
  15/100，但
  462,600-phase ledger 和 admission 仍未完成。

同目录 SHA 与 `chmod 444/555` 是内容绑定，不是外部不可篡改信任根。投稿冻结时仍需
Git commit 或独立 release ledger 锚定最终包。

## 8. 当前裁决

首轮独立终审给出 `4/5 Conditional Accept / Minor Revision`，确认 H12 内存扩展性 P1
已关闭，并要求补独立 expected identity。该项已用 identity-service manifest/receipt
和双边重绑负例关闭。针对性复审裁决为：`P0=0`、`P1=0`、`P2=外部不可篡改信任根
未建立`，本轮验证基础设施可信度 `5/5 Accept`。

该 Accept 仅覆盖 H12 Phase Array Store 验证基础设施，不代表 Local5 formal G0、整篇
DATE 论文或架构创新已通过。资源模型已经由 H24 实跑验证；rows、instance、unique
payload 和 store bytes 预测与实测一致，RSS 保护值覆盖实测。该晋级只覆盖一个真实
H24 窗口，formal G0 继续 DENY。
