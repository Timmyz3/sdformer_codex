# 精确68位Pareto修订与GS-TTB跨阶段Gate-Slot架构冻结

## 1. 为什么本轮必须换方向

第十轮独立评审后复核发现，DQFS文档曾用`OUT_DIM=32`估算1024-bit
product cache。随后第十二轮审阅进一步指出：即使当前Local5/TCFM接口为
4×32-bit，cache也不必保存完整Acc接口宽度。真实数值参数是：

```text
OUT_DIM = 4
gate    = unsigned 9 bit
weight  = signed 8 bit
product = exact 4 × 17 = 68 bit
Acc接口 = 4 × 32 = 128 bit
```

按精确68-bit缓存格式重新计入cache data/tag/LRU、DQFS双context term/directory/
metadata后，W6 ordered trace上的16个DQFS候选全部在“存储bit、product计算
减少”二维上被更简单结构支配。DQFS因此封存为负结果，不进入TCFM集成。

完整数据见：

`results/qfit_product_storage_pareto_20260731/report.md`

## 2. 新的本土化观察

W6 trace有1494个source-quotient term、156个全局`(lane,gate)`值键。

关键不是把任意值键放入通用LRU，而是gate出现顺序具有明显偏置：

| 结构 | 存储bit下界 | product计算 | product减少 |
|---|---:|---:|---:|
| lane-local 4-way LRU | 10308 | 499 | 66.60% |
| 4-slot首次绑定、无替换 | 10148 | 397 | 73.43% |
| lane-local 6-way LRU | 15620 | 156 | 89.56% |
| 6-slot首次绑定、无替换 | 15140 | 165 | 88.96% |
| profile-frozen 4-code | 8900 | 262 | 82.46% |

4-slot无替换表比4-way LRU少160 bit，并少102次product计算。原因不是更好的
通用替换算法，而是Local5 gate vocabulary的高频值较早出现：稳定保留首次
gate，比随后被低频gate冲掉更适合该term流。

这是W6单trace观察，必须等待post-G0多样本验证，不能先称为普遍规律。

进一步的profile-frozen codebook把全局高频gate
`{15,29,31,32}`直接映射为2-bit slot，其他gate走exact bypass。它不需要
per-lane tag和替换状态，在W6上优于4-way LRU及动态4-slot表。但Local5
Shiftmax的合法输出并不理论受限于这四个或七个码；随机合法向量可产生更多
gate，因此该结构只能写成网络profile驱动的ASIC专用化，不能写成代数定理。

## 3. GS-TTB定义

本轮候选命名为：

> **GS-TTB：Gate-Slot Token-Term Bundle**

它借鉴Bishop的TTB“把稀疏执行信息随bundle传递”思想，但bundle内容改为
Local5的source-quotient term和稳定gate slot，不使用Bishop的ECP近似。

第一版固定宽逻辑格式：

```text
GS-TTB {
    source_plane/y/x,
    lane,
    destination_mask,
    slot_valid,
    slot_id,
    slot_fill,
    bypass,
    gate_if_fill_or_bypass,
    row/window boundary
}
```

策略矩阵随后证明该固定宽包为NO-GO：每term同时携带`op+slot+gate`时，
key字段从原9 bit增至W4的13 bit。后续只保留
**ES-GS-TTB（Exception-Split GS-TTB）**：

```text
primary stream   = payload + op + slot
exception stream = gate，仅FILL/BYPASS产生
```

两条FIFO保持同一exception子序，primary中的op决定是否pop exception，因此
不携带sequence。W6单trace上，动态W4的key流量由13446 bit降至9549 bit
（28.98%）；固定宽包
则增至19422 bit。该数字尚未计入双FIFO、sequence和join控制，必须以RTL总线
toggle及SAIF复核。

三种精确路径：

| 路径 | 条件 | 行为 |
|---|---|---|
| slot hit | gate已绑定 | term只携带slot，投影端直接读product |
| slot fill | 有空slot的新gate | 写tag，计算一次product并写slot |
| exact bypass | slot满且gate未绑定 | 当前term直接计算product，不替换已有slot |

`exact bypass`只损失性能，不改变输出，不丢弃term，也不触发近似。

附加候选`PF-GS-TTB`（Profile-Frozen GS-TTB）使用编译期codebook：

- codebook gate命中时直接编码固定slot；
- 每个lane仅保存slot valid和68-bit精确product；
- codebook外gate携带原9-bit值并走exact bypass；
- codebook由训练/profile集合选择，测试集合只用于评估，不得反向调参。

## 4. 跨阶段微架构

```text
FCSR descriptor
    |
Source Multicast Term Builder
    |
Gate-Slot Allocator
  - lane-local tag slots
  - first-bind/no-replacement
  - exact overflow classifier
    |
GS-TTB FIFO
    |
Direct-Indexed Product Table
  - hit: product[lane][slot]
  - fill: gate * W -> product[lane][slot]
  - bypass: gate * W -> transient product
    |
TCFM-5 destination multicast
    |
five conflict-free Acc banks
```

与普通product cache的差分必须严格写成：

1. gate-slot绑定在上游完成，投影端不做per-term CAM；
2. slot在一个window/weight epoch内稳定，不写LRU、不搬移product；
3. 满表新gate精确旁路，不污染高复用slot；
4. GS-TTB把9-bit gate在稳定路径压缩为2/3-bit slot；
5. TCFM只消费直接索引product和destination mask。

不能声称：

- 发明cache；
- 4-slot对任意网络优于LRU；
- 当前位模型已经证明能耗或面积优势；
- 单W6 trace可以外推fullres数据集。

## 5. 公平基线

必须实现以下四条路径：

| 编号 | 路径 |
|---|---|
| B0 | 无复用：每term计算一次product |
| B1 | lane-local 4-way LRU product cache |
| B2 | lane-local 6/8-way LRU product cache |
| C0 | 4-slot GS-TTB + exact bypass |
| C1 | 6-slot GS-TTB + exact bypass |
| C2 | 4/5-code PF-GS-TTB + exact bypass |

共同约束：

- 同一ordered term producer；
- 同一4个乘法器和乘法流水；
- 同一68-bit内部product和128-bit符号扩展输出；
- 同一TCFM-5后端；
- 同一ready-valid与SRAM读延迟；
- 同一weight端口；
- 同一SDC和SRAM宏规则。

## 6. 必须统计的活动

除wall-time外，逐window输出：

- accepted/retired term；
- product starts；
- slot/cache hit；
- cold fill；
- overflow bypass；
- tag compare；
- LRU metadata write；
- product SRAM read/write；
- weight read；
- output/TCFM stall；
- slot occupancy；
- 每lane distinct gate；
- final Acc mismatch。

SAIF阶段必须分别给出tag、replacement、product SRAM、multiplier、weight SRAM和
TCFM的动态功耗，不能只报总toggle。

## 7. RTL分解与签核顺序

### G0：强基线

实现`qfit_lane_product_cache_leaf`：

- 参数化4/6/8 way；
- lane-index set、gate tag；
- 确定性LRU；
- 同步读、一拍hit；
- miss fill；
- 完整反压和计数。

### G1：GS-TTB allocator

实现`qfit_gate_slot_allocator`：

- first-bind/no-replacement；
- hit/fill/bypass三态；
- slot稳定；
- window/epoch原子清空；
- 输出bundle在stall下稳定。

### G2：direct product table

实现`qfit_gate_slot_product_leaf`：

- 直接`lane×slot`读；
- fill与bypass共用4个乘法器；
- hit不读weight、不启动乘法；
- 输出符号扩展后的128-bit Acc接口product和原destination payload。

### G3：TCFM集成

把TCFM接口拆成：

```text
product_valid/product_ready/product_vector
destination payload
```

避免TCFM再次用gate和weight重复计算。

## 8. Exact不变量

1. 每个输入term恰好输出一次；
2. hit时读取的slot tag必须等于输入gate；
3. fill只写空slot，不替换有效slot；
4. bypass不修改任何slot；
5. weight epoch切换前所有输出已退休；
6. output product等于`gate * W[lane,:]`；
7. 反压期间bundle与product保持稳定；
8. 最终TCFM Acc与无复用B0逐位一致。

## 9. 当前DATE判定

GS-TTB比DQFS更适合当前4-wide硬件，但现在仍只是`[prof]+[模型]`候选。

它可能形成的DATE架构主张是：

> 利用all-binary Local5 attention产生的低基数、早期偏置gate vocabulary，
> 将值复用解析从投影端关联cache前移到term编译阶段，以稳定slot bundle、
> 无替换product驻留和exact bypass构成跨阶段语义数据流。

只有在多样本profile、RTL同producer对照和同宏SAIF均成立后，这条主张才可
进入论文贡献列表。

## 10. 下一步

1. 先由独立ASIC/DATE reviewer审阅本合同；
2. 通过后实现B1/B2，不先实现候选；
3. 再实现G1/G2并做同trace比较；
4. 若4-slot优势跨样本不稳定，保留6-slot或退回LRU；
5. 每完成一个阶段重新进行DATE评分。
