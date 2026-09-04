# 完整C0片上Builder与分段精确Walker架构迭代

## 1. 结论先行

上一轮只完成三格式Serializer后端。本轮已经把前端canonical workspace接入，形成完整C0：硬件直接接收162条final-gate/K记录，自动构建term和destination，自动选择IPD/FADC/RAW并逐word写入typed slot。

因此，“完整C0片上三格式builder缺失”已经在RTL证据等级关闭。但论文主架构仍未签核：当前C0存在bitmap写mux过重、私有payload复制和单workspace串行化，且没有目标库PPA与扩大trace。

## 2. 模块层次

```text
gatestack_onchip_typed_builder_c0_top
  |- gatestack_canonical_head_workspace_c0
  |    |- RAW scratch
  |    |- 4x32x162 destination bitmap
  |    |- fanout/class directory
  |    `- linear或16-bit segmented destination walker
  |- gatestack_typed_format_policy
  `- gatestack_typed_builder_commit_top
       |- gatestack_typed_payload_serializer
       `- gatestack_head_slot_sram_adapter
```

## 3. 相序

```text
IDLE
  -> CAPTURE 162 token
  -> ANALYZE：按gate升序、lane升序建立term directory
  -> METADATA/POLICY
  -> DESCRIPTOR + DESTINATION，或RAW replay
  -> SERIALIZE
  -> SLOT COMMIT
  -> DONE
```

C0在done前不接受下一head。raw capture协议错误禁止启动Serializer；class容量溢出不破坏RAW scratch，policy会选择RAW并正常commit。

## 4. 新增架构点：Segmented Exact Destination Walker

真实RTL数据显示线性扫描有效率只有8.69%。16-bit segmented walker以11个segment nonzero加segment内first-one实现精确token定位，只在输出握手后清除remaining bitmap中的对应bit。

它与approximate pruning不同：事件集合、term顺序、token顺序和最终payload逐bit不变。真实S0/S3向量下scan cycle减少91.31%，开放结构generic cell增加5.38%。这可作为算子微架构贡献，但仍需8/16/32分段宽度的DC时序、面积和功耗消融后才能冻结16-bit。

## 5. 当前主架构命名

建议暂用：

**TA-GateStack：Typed-Atomic Gate-Class Stack**

其中三层贡献分别为：

1. **Canonical Tri-Format Builder**：同一final-gate/K canonical workspace支持IPD优先、FADC容量救援和RAW精确回退；
2. **Segmented Exact Destination Walker**：将稀疏bitmap的逐零扫描变为事件比例发射；
3. **Typed Atomic Residency Path**：格式身份、tag、payload边界、IPD选择性驻留和末字原子发布组成统一生命周期。

当前不应把“ping-pong”“Shiftmax”或“Yosys cell减少”单独写成主创新。

## 6. 下一轮

P0顺序：

1. 对4x32 bitmap做显式class-bank和16-bit segment-bank重构，降低动态mux；
2. 生成四stage全部45 head的无随机空拍RTL latency ledger；
3. 用真实head序列建立C0与C1双workspace共享Serializer的周期模型；
4. 只有模型显示C1有稳定净收益，才实现C1 RTL；
5. 删除私有payload副本，做buffered与streaming direct-commit同约束消融；
6. GPU空闲后扩大bit trace；并行补valid825与目标库DC/STA/SAIF/LEC。

下一次独立DATE复审至少应等待第2、3项完成；否则只能确认功能闭环，不能显著提高架构评分。
