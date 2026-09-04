# Local5 EREP 正式 Archive 内容重放合同

## 1. 本轮结论

本轮关闭的是 Local5 EREP 的 archive 内容解析合同，不是正式 G0：

```text
[synthetic-contract]+[代码审计]：PASS
联合回归：48/48 PASS
formal G0：DENY
```

结果目录：

```text
results/local5_erep_ledger_replay_v4_reviewfix_v3_20260810
```

结果文件和 receipt 的 SHA-256 已全部通过 `sha256sum -c`。正式入口因
`local5_erep_g0_admission_v4_20260810/admission_receipt.json` 不存在而拒绝，符合
fail-closed 预期。

## 2. 为什么需要内容重放

只把 `rtl_trace_archive.npz` 和 `acc32_miter_archive.npz` 的文件 SHA 写进 receipt，
只能证明“读取的是同一份字节”，不能证明其中保存了规定的 trace、相序和 Acc32。
本轮新增：

```text
scripts/local5_erep_archive_replay_v4.py
tests/test_local5_erep_archive_replay_v4.py
```

统计器现在按以下顺序工作：

```text
文件 basename/SHA 绑定
  -> NPZ 安全解析（allow_pickle=False）
  -> 成员/dtype/shape/规模检查
  -> raw event 重建 phase/head/window ledger
  -> expected/actual Acc32 逐元素比较
  -> 与 head_phase_ledger 做 canonical SHA 对照
  -> 独立重放 C0--C4
  -> 统计 G0 指标
```

任何前级失败都不能进入后级统计。

## 3. RTL Trace NPZ 合同

成员集合必须精确，禁止缺项和额外影子成员。主要数组如下：

| 类别 | 数组 | dtype | 语义 |
|---|---|---:|---|
| schema | `schema_version` | `uint16[1]` | 固定为 4 |
| window | `window_sample/stage/block/token/weight/heads` | 固定整数类型 | 规范窗口元数据 |
| phase | `phase_window_index/input_head/role/output_tile/duration` | 固定整数类型 | phase 描述符 |
| offset | `phase_event_offsets` | `int64` | 每个 phase 的 event 边界 |
| event | `event_resource/event_cycle/event_identity` | `uint8/uint32/S64` | 原始资源事件 |

phase 次序固定为：

```text
window
  -> prepare[output_tile=0..H-1]
  -> drain[output_tile=0..H-1]
  -> input_head=0..H-1
       -> fill
       -> direct[output_tile=0..H-1]
       -> execute[output_tile=0..H-1]
```

每个 phase 内 event 固定按“角色资源序号、cycle、identity”排序。identity 必须是
非空、无 NUL、最长 64 字节的 ASCII；fixture encoder 在转换为 `S64` 前先检查长度，
不允许 NumPy 静默截断。逐资源 cycle 和 identity 均不得重复；fill/execute 的
relation、epoch、FIFO 记录 identity 序列必须一致。

在 `np.load` 之前还会直接检查 ZIP container 的 `infolist()`：原始 `.npy` 成员名、
顺序和数量必须精确，禁止重名成员、目录项、加密成员、archive comment 和非
stored/deflated 编码；member comment、extra 和非零 flag bits 也一律拒绝。这样同名
成员不能先被 Python 字典折叠后再逃过集合检查。

## 4. Acc32 Miter NPZ 合同

成员同样必须精确：

| 类别 | 数组 | dtype | 语义 |
|---|---|---:|---|
| schema/window | 与 trace 对应成员 | 固定整数类型 | 与 trace 窗口逐项一致 |
| offset | `window_value_offsets` | `int64` | 每个窗口 Acc32 边界 |
| value | `expected_acc32` | `int32` | 独立软件金参考 |
| value | `actual_acc32` | `int32` | RTL 实际输出 |

Acc32 属于 window 最终输出，不属于单个 input head。每个窗口必须包含
`H * 450 * 32` 个标量，隐式坐标次序固定为：

```text
output_tile=0..H-1 -> source=0..449 -> out=0..31
```

parser 逐元素重算 mismatch，非零立即拒绝；随后用窗口坐标、expected 字节和 actual
字节重算摘要。这里不另存 multiset digest：固定坐标 canonical array 比 multiset 更强，
不会把两个 source/output 坐标交换误判为等价。

## 5. 正式规模

正式常量由 100 sample、12 block 和 stage head 拓扑推导，不由 adapter 自报：

| 指标 | 正式值 | 推导 |
|---|---:|---|
| window | 1200 | `100 * (2+2+6+2)` |
| input-head | 13800 | `100 * sum(blocks_s * H_s)` |
| phase | 462600 | `100 * sum(blocks_s * [2H + H(1+2H)])` |
| Acc32 标量 | 198720000 | `13800 * 450 * 32` |

stage 的 `H={3,6,12,24}` 和 `weight={440,120,30,10}` 也在 archive 入口逐窗口锁定。
正式窗口 token 还必须在后续 ledger replay 中与冻结 selection plan 逐项相等。

## 6. 本轮验证

联合 runner：

```bash
OUT_DIR=results/local5_erep_ledger_replay_v4_reviewfix_v3_20260810 \
  ./sim_qfit/run_local5_erep_ledger_replay_v4_checks.sh
```

48 项测试覆盖：

- NPZ 成员、dtype、维度、schema 和 canonical phase 次序；
- raw event identity 修改后不能匹配旧 head ledger；
- expected/actual 任一元素不等时重算得到非零 mismatch；
- 非 NPZ、损坏 ZIP、pickle 禁止路径；
- stage/H/weight、正式规模和 `S64` 边界；
- 三层账本 anti-self-report、int/float 宽松相等绕过、archive 三方 SHA；
- 原始 ZIP 同名、乱序、目录、archive comment、member extra 和 BZIP2 编码拒绝；
- C0--C4 调度、容量基线与统计阈值原有回归。

synthetic fixture 的实际重放规模是 1 window、3 head、27 phase、91 event 和
43200 个 Acc32 标量，mismatch 为 0。它只证明 parser 合同可执行，不能证明正式
Local5 workload 或 RTL 正确。

## 7. 仍未关闭的边界

1. producer 当前尚未生成正式 manifest，formal preflight 仍为 DENY；
2. 正式 adapter、1200-window RTL trace 和约 1.99 亿标量的 miter archive 尚不存在；
3. parser 能证明 expected 与 actual 是否相同，但 expected 是否来自冻结独立软件金参考、
   actual 是否来自目标 RTL，仍需正式 adapter 源码绑定与生成流程审计；
4. NPZ 采用隐式 Acc32 坐标顺序，因此 adapter 必须先按固定坐标落盘；若将
   expected/actual 同步错误重排，单靠 equality 无法识别，必须由独立来源与坐标生成测试
   防止；
5. 当前没有新增 `[rtl]`、`[prof]`、PPA、吞吐或精度结果，也没有 EREP candidate RTL。

runner 另生成确定性的 `source_bundle.tar.gz`，将本轮脚本、测试、runtime contract 和
本文档纳入结果 SHA 与 receipt。它关闭“untracked 源码无法从结果包恢复”的本地复现
问题，但不替代正式 Git commit/tag 或外部签名信任根。

因此本轮只把“任意 NPZ 字节可冒充正式 archive”的漏洞降为已关闭，下一正式 P0 是
producer 完成后实现并审计 adapter 的双来源生成链，再运行 formal G0。
