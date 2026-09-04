# DG-TTB与异常拆分Gate-Slot跨阶段数据流提案

> 状态更新：后续DATE复审发现DG-TTB被原始descriptor直传和因子化延迟展开
> 支配，DG-TTB已封存为负结果。最新架构见
> `docs/222_DGTTB负结果与双Stationary延迟展开架构迭代_20260731.md`。

## 1. 从真实代码得到的架构观察

Local5当前数据流为：

```text
5-role gate descriptor
  -> source-multicast term builder
  -> 每descriptor去重gate
  -> 对每个active K lane逐term重复发送：
     source metadata + lane + 9-bit gate + destination mask
  -> projection
```

`qfit_source_multicast_term_builder`已经把五角色相同非零gate合并成最多5个
`unique_gate + destination_mask`。但去重结果只保留在builder内部，输出term
仍重复携带完整source metadata和9-bit gate。

W6 ordered trace：

| 指标 | 值 |
|---|---:|
| source descriptor | 36 |
| term | 1,494 |
| gate字典项总数 | 91 |
| 单descriptor字典max | 4 |
| 原term流 | 55,278 bit |
| 理想变长header/body下界 | 22,455 bit |
| 带body_count安全变长格式 | 22,743 bit |
| 单拍固定5-entry header/body | 23,544 bit |
| 固定header逻辑位数减少 | 57.41% |

这是单trace逻辑字段计数，不含FIFO指针、valid-ready、布线、时钟树和toggle，
也不是fullres结果。

## 2. DG-TTB定义

候选名称：

> **DG-TTB：Descriptor-Gate Token-Term Bundle**

它把source descriptor拆成有序header和body：

```text
DG header {
    source_id,
    source_y/x,
    unique_count,
    body_count[7:0],
    unique_gate[0:4]
}

DG body {
    lane,
    gate_ref[2:0],
    destination_mask[4:0],
    descriptor_last
}
```

当前宽度口径：

- 原term：`source(17)+lane(5)+gate(9)+mask(5)+last(1)=37 bit`；
- body：`lane(5)+gate_ref(3)+mask(5)+last(1)=14 bit`；
- 第一版固定header：每descriptor一次
  `source(17)+unique_count(3)+body_count(8)+5×gate(45)=73 bit`。

22,455 bit只是不含`body_count`的理想变长信息下界，不能作为物理FIFO位数。
第一版采用73-bit单拍固定header，总计23,544 bit。

## 3. 协议与无死锁合同

### 3.1 最小实现

第一版只允许一个descriptor context处于body发射状态：

1. builder先完成header握手；
2. header在projection端驻留；
3. body按原term顺序发射；
4. `descriptor_last`退休后才接收下一header。

该版本没有header ID、重排或多context join，风险最低，语义与现有builder一一
对应。

### 3.2 双FIFO扩展

若header和body使用独立FIFO：

- body不得在对应header提交前可见；
- downstream只有在header context valid时拉高body ready；
- `descriptor_last`必须恰好出现一次；
- header不能在前一descriptor body未退休时覆盖；
- reset/abort必须同时清除header context与body FIFO。

多descriptor in-flight是后续吞吐优化，不是G1首版要求。

## 4. 与GS-TTB的关系

策略矩阵表明：

| W4策略 | product start |
|---|---:|
| LRU | 499 |
| FIFO | 454 |
| SRRIP | 429 |
| projection-side first-bind | 397 |
| same-trace global top4 | 262 |
| admission-aware Belady oracle | 242 |

因此动态GS的397次product start并不来自“跨阶段slot”本身；projection-side
first-bind也能得到相同结果。GS真正需要证明的只剩：

1. producer解析slot后，projection端取消关联tag比较；
2. stable slot取消LRU写；
3. bundle传slot而非完整gate；
4. exact bypass不污染驻留项。

## 5. 固定宽GS-TTB为什么被淘汰

W4固定包同时携带`op(2)+slot(2)+gate(9)`，key字段为13 bit，比原9-bit gate
更宽。W6单trace：

| 编码 | key bit | 相对原gate流 |
|---|---:|---:|
| 原9-bit gate | 13,446 | 基线 |
| 固定宽动态GS | 19,422 | 增加 |
| exception-split动态GS | 9,549 | 减少28.98% |

因此固定宽GS-TTB为NO-GO。

## 6. ES-GS-TTB

保留候选：

> **ES-GS-TTB：Exception-Split Gate-Slot TTB**

```text
primary stream {
    payload,
    op = HIT/FILL/BYPASS,
    slot
}

exception stream {
    gate或DG gate_ref
}
```

只有FILL/BYPASS产生exception项。两条流分别保持顺序，下游看到FILL/BYPASS
primary时才pop下一exception，因此不需要全局sequence ID。

### 6.1 原子发射

producer采用原子双入队：

```text
needs_exc = FILL || BYPASS
source_ready = primary_ready && (!needs_exc || exception_ready)
primary_push = source_valid && source_ready
exception_push = primary_push && needs_exc
```

FILL/BYPASS的primary与exception必须同拍成功，禁止先发primary再补exception。
两条FIFO均为标准ready-valid、ready不依赖valid，避免组合死锁。

downstream按primary头部联合出队：

```text
join_valid = primary_valid && (!needs_exc || exception_valid)
primary_pop = join_valid && product_ready
exception_pop = primary_pop && needs_exc
```

### 6.2 Read-after-fill

primary流严格有序：

```text
FILL(slot, gate) -> 后续HIT(slot)
```

projection按序执行，FILL先写同步1RW product bank，下一term最早下一周期读，
因此无需跨stage fill-ack。若未来允许重排或多个consumer，则必须恢复
`EMPTY/RESERVED/VALID`及ack，不能沿用此简化。

## 7. DG与ES的融合顺序

推荐顺序：

1. 先实现DG-TTB single-context header/body；
2. 与原37-bit flat term在同producer/TCFM下做逐位等价和总线toggle对照；
3. 再实现projection-side first-bind强基线；
4. 然后实现ES-GS W4/W6；
5. 最后才考虑PF模式。

原因：DG只改变无损表示和跨阶段传输，风险低；ES-GS同时引入状态驻留、
exception join和product cache，验证面更大。

## 8. Motion与Local5双线

### Local5

header粒度天然是五角色source descriptor，gate字典已在RTL中存在，DG-TTB可
直接落地。

### Motion

Motion H67已有SCS row级occupied class/gate目录，历史profile显示每行活跃
gate class的`p50/p95/p99/max=1/3/3/6`。可评估row-header版本：

```text
row header = occupied gate/class dictionary
row body   = token/lane + dictionary ref
```

但Motion当前GateStack/NMF可能已直接消费目录，不一定存在与Local5同等的
重复flat term传输。因此必须先从真实H67 trace重建“原流 vs row-header流”
位数、周期和FIFO活动，不能套用Local5的59.38%。

## 9. 与相关工作的边界

| 工作 | 借鉴 | 本工作差分 | 不宣称 |
|---|---|---|---|
| Bishop | TTB、metadata-first、header/body纪律 | exact source-gate dictionary，不做ECP和异构双核 | 发明TTB |
| Prosperity | exact reuse的评估纪律 | gate-slot product驻留与Local5 source quotient | 发明product reuse |
| StreamTensor | 迭代空间随stream描述 | 硬件ordered header/body与descriptor-last合同 | 编译器itensor |
| FLAT/LoAS | 数据流与时间/驻留优先 | 五角色gate字典和event term流 | 通用attention dataflow首次提出 |

DG-TTB可辩护的新意是：

> 将Local5五角色gate等价关系从builder内部临时状态提升为跨阶段header/body
> ISA，使source quotient、gate dictionary和lane-term在硬件中保持同一个
> exact迭代空间，同时消除term级source/gate重复传输。

## 10. DATE晋级门槛

DG-TTB进入贡献列表前必须满足：

1. flat-term与DG-TTB最终TCFM Acc逐位一致；
2. 真实W6 trace和fullres多sample均无协议失配；
3. 相同FIFO容量下报告周期、stall、mean/p95/p99；
4. 统计header/body bit、FIFO读写和总线toggle；
5. DC/STA证明header context与gate_ref mux不破坏频率；
6. SAIF证明传输减少大于header/FIFO/context开销；
7. 与通用header/body dictionary和仅source-header消融公平比较。

ES-GS还必须额外证明：

- exception FIFO不形成新瓶颈；
- FILL/BYPASS无丢失、无错配；
- W4/W6相对projection-side first-bind及W6 LRU仍有PPA净收益；
- PF使用独立sample-held-out。

在这些证据完成前，DG-TTB和ES-GS都是`[prof]+[模型]`候选，不是已成立的
DATE贡献。
