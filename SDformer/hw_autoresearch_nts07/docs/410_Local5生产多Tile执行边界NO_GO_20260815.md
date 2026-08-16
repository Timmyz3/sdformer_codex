# Local5 生产多 output-tile 执行边界 NO-GO

## 1. 裁决

- 状态：`NO_GO_AS_NEW_ARCHITECTURE_KEEP_EXISTING_COMPLETENESS_PATHS`。
- 证据：`[rtl-bound-profile]`。
- 不新增 term-stationary、term-tape 或多 tile 贡献名，不写新 RTL。
- Local5 创新维持 `3.1/5`、完整度维持 `3.2/5`、综合维持
  `3.1/5 Borderline Reject`。
- `docs/359_DATE终局冻结_20260813.md` 未修改。

## 2. 三种精确实现边界

### A. term 保持并访问全部 output tile

每个 source-owned term 只接收一次，然后依次更新多个输出 tile，必须同时保留这些
tile 的 Acc 上下文。现有 OUT2 Acc payload 为 `450x2x32=28,800 bit`；OUT32 为
`460,800 bit`，正好扩大 `16x`。这就是已有 packed OUT32 的空间宽度交易，不是
新的固定面积数据流。

### B. 固定 OUT2 Acc，物化后重放 term

生产 100-group 中有 74,131 个 source-owned term 和 11,245 个 active source
descriptor。按当前端口所需的最小字段计数：

| replay 对象 | 字段宽度 | 100-group payload | 相对值 |
|---|---:|---:|---:|
| expanded term | 29 bit | 2,149,799 bit | 2.078x |
| factorized source descriptor | 92 bit | 1,034,540 bit | 1.000x |

expanded term tape 即使不计控制、SRAM 粒度和路由，也比 factorized descriptor 大
`2.078x`。若重新因子化 term tape，就恢复为现有 Relation Memo/source descriptor
对象，不是新物化对象。

### C. 固定 OUT2 Acc 且不存 replay 对象

该条件下只能按输出 tile 重算 score、relation 或 source descriptor 前端。它是当前
系统完整度缺口的基线，不产生新的跨算子复用机制。

## 3. 强基线结果

已有真实多 tile Relation Memo 对两个真实窗口的组件 speedup 分别为：

- sample0/window94：`1.019232x`；
- sample25/window374：`0.999837x`。

前者未过预注册 `1.05x` 独立贡献门槛，后者为负收益。packed OUT32 与 Memo 继续
分别作为空间宽度和固定面积 replay 的完整度证据，但不提高 Local5 架构创新分。

机器可读报告：

```text
results/local5_production_multitile_boundary_audit_20260815/report.json
```

## 4. 后续边界

生产多 output-tile/cross-head scheduler 仍可为完整度实现，但只能点名为系统调度，
不能包装成新执行对象。Local5 下一项真正影响投稿完整度的证据仍是目标宏下的
1R1W/1RW matched SAIF/DC/STA/PTPX，以及完整 block/encoder 边界。
