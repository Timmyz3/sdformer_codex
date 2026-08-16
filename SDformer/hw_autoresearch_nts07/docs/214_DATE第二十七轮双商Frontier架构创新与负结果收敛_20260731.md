# DATE 第二十七轮：双商Frontier架构创新与负结果收敛

> **实现合同更新：** 本文初版“必须等row complete后才emit”存在容量满导致
> 循环等待的风险。`docs/215`已改为可提前封口的row-owned segment，并用
> 反向指针满足单写口append。本文保留为发现DQFS的过程记录，RTL合同以
> `docs/215`为准。

## 1. 本轮结论

第八轮DATE复评指出了一个关键数学错误：`source`不影响
`gate×W[lane,:]`，因此`(source,lane,gate)`不是product等价键。本轮已：

1. 撤回source-product下界；
2. 将执行关系拆成destination/schedule商与value商；
3. 用真实有序RTL term流验证简单memo；
4. 发现简单memo收益不足，但row-bounded重排有明显潜力；
5. 提出可落RTL的**双商Frontier调度器**。

当前证据仍是W6定向向量，不是post-G0或目标工艺PPA。新架构只进入
“候选”，尚不能写成论文已实现贡献。

---

## 2. 正确的两级等价关系

### 2.1 Destination/Schedule Quotient

现有FCSR和term builder形成：

```text
(source, lane, gate) -> destination_mask
```

`source`决定五点拓扑中的destination集合、边界与退休生命周期。这个键适合
描述“向哪里提交”，但不决定product数值。

### 2.2 Value Quotient

真正决定投影product的是：

```text
(lane, gate, weight_epoch) -> gate * W[lane, output_tile]
```

在同一权重epoch和output tile内，不同source只要`lane/gate`相同，product
向量就完全相同。由此形成第二个商：

```text
多个source term -> 一个value key -> 一次product生成 -> 多次destination提交
```

本轮将二者合称为：

**DQFS：Dual-Quotient Frontier Scheduler，双商Frontier调度器。**

---

## 3. 为什么普通缓存不够

共同Local5 producer的W6有序trace包含：

| 指标 | 数值 |
|---|---:|
| term | 1494 |
| 唯一`(lane,gate)`值键 | 156 |
| 全窗口复用理论上界 | 89.56% |

但原始顺序的时间局部性很弱：

| 在线结构 | 总项数 | 命中率 |
|---|---:|---:|
| 每lane 1项 | 32 | 9.30% |
| 每lane 2项 | 64 | 21.69% |
| 全相联32项 | 32 | 7.23% |
| 全相联64项 | 64 | 49.13% |
| 全相联128项 | 128 | 88.76% |

全局16项direct-mapped在当前流上为0命中。这个负结果说明：

- “加一个小product cache”不能作为主创新；
- 大全相联表的比较、数据阵列与wide-product存储成本可能抵消收益；
- 需要利用Local5的有界关系生命周期主动制造局部性。

---

## 4. Row-Bounded Value-Quotient Frontier

将同一有序trace按可证明完整的范围重排：

| 重排范围 | product生成次数 | 减少 | 最大缓冲term |
|---|---:|---:|---:|
| 单source | 1494 | 0.00% | 64 |
| 固定16项chunk | 1494 | 0.00% | 16 |
| 固定64项chunk | 1289 | 13.72% | 64 |
| 固定128项chunk | 967 | 35.27% | 128 |
| source row frontier | 607 | 59.37% | 307 |
| plane frontier | 267 | 82.13% | 813 |
| 整窗口 | 156 | 89.56% | 1494 |

row frontier位于收益和状态之间的中间点。W6中：

- 每row平均101.17个值键，最大124个；
- 每lane每row gate基数p95=6，最大6；
- 每值键平均连接2.46条term，p95=5，最大6；
- 最大term缓冲为307项。

这组数据支持“每lane小目录+共享term SRAM”，不支持全窗口CAM。

---

## 5. DQFS微架构

### 5.1 数据流

```text
Local5 score/Shiftmax
  -> FCSR-RX：destination-major转source-major
  -> Source Quotient：形成(source,lane,gate,destination_mask)
  -> DQFS Collect Context
       - Lane-Gate Directory
       - Term Payload SRAM
       - Next-Pointer SRAM
  -> Row Complete
  -> DQFS Emit Context
       - 逐(lane,gate)取出链
       - 一次生成gate*W[lane,output_tile]
       - 对链内source term复用product
  -> TCFM-5：五色无冲突destination提交
  -> Acc SRAM
```

### 5.2 两上下文重叠

DQFS使用两个row context：

- Context A收集下一row；
- Context B按值键发射上一row；
- row边界交换角色；
- 若B未完成，FCSR通过ready/valid自然反压。

双context不是独立创新，而是把row-bounded重排吞吐化的必要实现。

### 5.3 Lane-Gate Directory

每个lane只比较该lane的gate目录，不做全局CAM：

```text
directory[lane][way] = {
    valid,
    gate,
    head_pointer,
    tail_pointer,
    term_count
}
```

W6观测支持6-way候选，但部署参数必须由post-G0多样本的
`per-row/per-lane gate cardinality`锁定。超出容量时必须走exact fallback：

1. 关闭该row的值商重排并按原顺序发射；或
2. 将溢出项送入RAW FIFO。

禁止丢弃、合并不同gate或覆盖未退休目录项。

### 5.4 Term Payload SRAM

term SRAM不重复存product，只存：

```text
source_plane/source_y/source_x
destination_mask
next_pointer
window/row边界元数据
```

`lane/gate`驻留在目录项中。相比wide-product cache，DQFS将存储开销从
“每cache项一个output-tile宽product向量”改成“窄term链+单个活动product
寄存器”。

### 5.5 Product生成与提交

进入一个目录项时：

1. 读取`W[lane,output_tile]`；
2. 生成一次`gate×W`向量；
3. product向量驻留；
4. 沿term链逐项向TCFM-5提交；
5. 链结束后切换下一个值键。

因此乘法器激活次数从term数变成row内唯一值键数。destination update数量
不变，TCFM-5的写端口合同不变。

---

## 6. Bit-Exact条件

DQFS只重排同一row内的整数累加顺序，不改变：

- gate数值；
- weight；
- destination集合；
- 每个destination收到的product多重集；
- term边界与非法term原子拒绝规则。

当前Acc使用固定宽度二进制补码回绕。模`2^ACC_W`加法满足交换律和结合律，
因此重排后最终bit pattern相同。

若后续改为饱和累加或中途可见的partial sum，则该证明不再自动成立，必须：

- 保持原顺序；或
- 在软件数值合同中证明饱和不触发；或
- 扩宽内部Acc并在最终一次量化。

`weight_epoch`必须进入目录生命期；权重切换前所有旧context必须排空。

---

## 7. 与既有工作的差分

| 工作 | 借鉴 | DQFS本土化差异 |
|---|---|---|
| Prosperity | exact product reuse、先profile复用距离 | 不匹配binary row相似性；值键由精确`lane/gate/epoch`给出，并利用row frontier主动重排 |
| Phi | pattern/residual分层 | static部分是五点关系生命周期，dynamic部分是gate值商；不复制其pattern表 |
| Bishop | TTB、metadata-first work unit | work unit改为frontier-complete row；不使用ECP，不需要dense/sparse双核 |
| Sanger | stationary数据流 | 驻留对象是一个值键的product向量，直到其source链全部提交 |
| FireFly-T | bitmap事件抽取 | K next-set-bit位于上游；DQFS解决跨source product复用，不把事件抽取本身列为创新 |
| StreamTensor类stream IR | 迭代空间描述 | source商和value商由硬件在线切换，不依赖离线完整稀疏矩阵 |

不可宣称：

- 发明cache、linked list、双缓冲或stationary dataflow；
- 复用了Prosperity/Bishop的PPA数字；
- W6的59.37%等于部署能耗降低；
- DQFS已经完成RTL或DC。

可争取的架构主张是：

> 利用Local5固定关系的可证明row-complete时刻，在同一流中先形成
> destination/schedule商，再形成跨source的value商，并将value-stationary
> product直接提交到拓扑编译的Acc银行。

这是“关系生命周期、算术等价类和物理写端口”的联合设计，不是独立缓存优化。

---

## 8. 下一步最小实现

### P0

1. 为post-G0 profiler增加每row值键、每lane gate基数、链长与overflow统计；
2. 实现参数化DQFS叶模块：双context、lane-local目录、term SRAM、RAW fallback；
3. 与原始有序term后端做bit-exact对照；
4. 统计product生成次数、目录比较、term SRAM访问、反压和完整wall-time；
5. 接入TCFM-5，验证重排后的五bank Acc完全一致。

### P1

1. 同步1R1W SRAM latency、随机TCFM反压、连续row/window；
2. `TIME_PLANES=3`、非2幂尺寸、目录溢出和weight epoch切换；
3. 统一`OUT_DIM=32`或论文真实tile宽度；
4. 与wide-product cache、无重排、plane重排做同约束PPA。

### P2

1. post-G0多sample/window mean、p95、p99；
2. 同SRAM compiler、同SDC的DC/STA/SAIF；
3. 消融：edge流、source商、source商+FCSR、+DQFS、+TCFM；
4. 将Role-Sharded完整readback归约计入wall-time。

---

## 9. 本轮证据

- `results/qfit_local5_projection_tile_yosys_20260731/ordered_term_trace.csv`
- `results/qfit_local5_projection_tile_yosys_20260731/report.md`
- `results/qfit_value_quotient_trace_20260731/value_quotient_stats.json`
- `results/qfit_value_quotient_trace_20260731/report.md`
- `scripts/analyze_qfit_value_quotient_trace.py`

本轮完成的是架构候选与数据约束，不是论文accept证据。是否晋级取决于真实
Local5 full-resolution trace、可综合RTL和同宏EDP。
