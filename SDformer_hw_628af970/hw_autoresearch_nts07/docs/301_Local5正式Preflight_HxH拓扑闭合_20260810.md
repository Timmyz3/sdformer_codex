# Local5 正式 Preflight：HxH 拓扑闭合

> 后续状态：首次独立评审已 4/5 接受 HxH P0，但发现 JSON 浮点宽松相等、manifest
> shadow 字段、task digest 未作断言和 runner 只支持 DENY 路径。整改结果见
> `docs/302_Local5正式Preflight首次复审整改_20260810.md`，本文件结果路径为历史版。

## 1. 本轮目标

正式 adapter 评审指出一个 P0：软件文档虽然写明每个窗口有 `H` 个输入 head 和
`H` 个 OUT_DIM32 输出 tile，但旧 schedule/统计代码没有强制 `O=H`。因此错误的
`H×1` 或 `H×O` 工作量仍可能进入周期模型。

本轮只关闭这个拓扑缺口，不修改运行中的 GPU producer、不生成 admission PASS、
不实现 EREP 候选 RTL。新增：

```text
scripts/local5_erep_formal_preflight_v4.py
tests/test_local5_erep_formal_preflight_v4.py
sim_qfit/run_local5_erep_formal_preflight_v4.sh
```

## 2. 机器化合同

### 2.1 Window 与 head

固定 selection plan 必须满足：

- 100 个 sample；
- 每 sample 的 block 拓扑为 `2/2/6/2`；
- stage head 为 `3/6/12/24`；
- sample-major、stage/block canonical 顺序恰有 1200 行；
- 每行 window 落在 `440/120/30/10` 范围内；
- probability 与 analysis weight 分别为 `1/N` 与 `N`。

每个 window 展开全部 `head=0..H-1`，必须得到 13800 个唯一 key：

```text
(sample, stage, block, selected_window, input_head)
```

正式 manifest 的 group 行允许换序，但 key multiset 必须精确相等；缺一行、重复
一行、head 越界或 full-resolution 字段错误均 fail closed。

### 2.2 Projection contract

12 个 block 必须严格按拓扑顺序出现。对每个 stage：

```text
C = H * 32
weight shape = [C,C]
input-head count = H
output-tile count = H
```

每个 block 的 theta、原始/effective float weight、int8 weight、scale-exp2 和 bias
六类 NPZ 数组必须名称、shape、dtype 全部精确，额外或缺失数组均失败。

### 2.3 HxH 任务

每个 canonical window 显式枚举：

```text
(sample,stage,block,window,input_head,output_tile)
```

计数为：

| stage | 每 sample/block 结构 | 100 sample 任务数 |
|---|---:|---:|
| 0 | `2×3²` | 1,800 |
| 1 | `2×6²` | 7,200 |
| 2 | `6×12²` | 86,400 |
| 3 | `2×24²` | 115,200 |
| 合计 |  | **210,600** |

任务 canonical SHA-256：

```text
5e894781aaca24b307fc0c33ddb116b28082694f484e3bb15784b8da7a6b07c6
```

## 3. 当前结果

结果目录：

```text
results/local5_erep_formal_preflight_v4_hxhfix_20260810
```

| 检查 | 结果 |
|---|---:|
| selection window | 1200，PASS |
| expected input-head group | 13800，PASS |
| projection block | 12，PASS |
| HxH task | 210600，PASS |
| 单测 | 5/5 PASS |
| result hash / complete receipt | PASS / PASS |
| formal manifest | 缺失 |
| admission generated | false |

最终状态严格为：

```text
DENY_FORMAL_MANIFEST_ABSENT
```

## 4. 证据边界

本轮证据是 `[契约审计]`，证明现有 selection 与 projection 输入能展开为完整 HxH
拓扑，并证明 formal manifest 到达后应如何拒绝 key 缺失/重复。它没有证明：

- 13800 个正式 group 已生成；
- 210600 个任务已做 RTL replay；
- T450/OUT_DIM32 Acc32 miter 已完成；
- C0--C4 周期来自底层 ledger 重放；
- formal G0 或 EREP candidate RTL 已放行。

下一个 P0 是防聚合自报：统计器必须读取 head phase/window event ledger，重新执行
schedule 得到 C0--C4，不能比较两份同源 scalar JSON。
