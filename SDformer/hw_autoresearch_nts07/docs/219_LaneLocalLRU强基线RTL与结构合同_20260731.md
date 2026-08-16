# Lane-Local LRU强基线RTL与结构合同

## 1. 为什么先做强基线

GS-TTB的潜在贡献不是“缓存相同gate的product”。该能力普通lane-local LRU
已经具备。只有先实现公平的LRU，后续才能回答：

> 把gate-to-slot解析前移到term producer，并把稳定slot随TTB传递，是否真的
> 减少投影端关联查找、替换状态和weight/product活动，而不是更换命名。

本轮因此冻结B1/B2：

- B1：4-way lane-local LRU product cache；
- B2：6/8-way lane-local LRU product cache。

## 2. 已实现文件

```text
rtl_qfit/qfit_lane_product_cache_leaf.sv
rtl_qfit/qfit_sync_1rw_bank.sv
tb_qfit/tb_qfit_lane_product_cache_leaf.sv
verif_qfit/qfit_lane_product_cache_assertions.sv
sim_qfit/run_qfit_lane_product_cache_checks.sh
scripts/check_qfit_lane_product_cache_structure.py
```

结果目录：

```text
results/qfit_lane_product_cache_20260731/
```

详细数据见该目录的`report.md`。

## 3. 已冻结的不变量

1. 一个合法输入term恰好产生一个输出；
2. hit、miss严格划分全部accepted term；
3. hit只读一个68-bit product bank，operand isolation关闭结果乘法通路；
4. miss只启动4个`10×8→17`结果乘法器，并写一个way；
5. victim选择invalid-first，否则确定性true-LRU；
6. lane之间不共享tag、valid或LRU状态；
7. epoch切换失效全部cache entry；
8. 输出stall期间product和全部payload保持稳定；
9. 同步product SRAM的read-valid必须在hit输出首周期成立。

## 4. 当前证据

| 证据 | 状态 | 边界 |
|---|---|---|
| Icarus W4/W6/W8定向等价 | PASS | 小参数、定向向量 |
| Verilator + SVA | PASS | W4/W6/W8及W6真实trace |
| Verilator lint | PASS | 仅TB时钟提示 |
| Yosys W4/W6/W8 | PASS | 综合可读，不是PPA |
| product bank结构合同 | PASS | 68×32同步1RW |
| 真实1494-term RTL replay | W4 66.60%，W6/W8 89.56% | 单W6 trace |

## 5. 对创新性的影响

这一基线把后续论文主张收窄为三个可证差分：

1. **Producer-resolved slot**：term producer在跨阶段边界一次解析gate，投影端
   不再做每term关联查找；
2. **Stable no-replacement residency**：window/weight epoch内slot不替换，
   消除LRU metadata写和误驱逐；
3. **Slot-bearing TTB**：常见路径只传slot id，codebook外或满表项走exact
   bypass，保持bit-exact。

“4-slot首次绑定优于4-way LRU”目前只在单W6 trace成立，不能当普遍结论。
“PF 4-code优于LRU”更必须通过held-out profile，否则只是对当前trace过拟合。

## 6. 下一门槛

G0未完全关闭前不进入候选RTL。必须先完成：

1. 独立ASIC代码审阅；
2. 独立DATE新颖性审阅；
3. 修复审阅中的P0/P1；
4. 给weight epoch、row/window和product SRAM latency写清接口合同；
5. 补多sample/fullres held-out、随机LRU与目标工艺物理证据。

达到以上条件后，才实现共用`HIT/FILL/BYPASS`操作格式的GS-TTB叶模块，并
保证LRU与GS-TTB复用同一product bank和4乘法器后端。
