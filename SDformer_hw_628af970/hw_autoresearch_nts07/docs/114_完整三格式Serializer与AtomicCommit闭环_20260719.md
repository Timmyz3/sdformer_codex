# 完整三格式Serializer与Atomic Commit闭环

## 1. 本轮关闭了什么

第五轮DATE独立审稿把“片上三格式builder只有metadata/policy，没有payload serializer和atomic commit”列为致命问题。本轮实现并验证了后端一半：

```text
冻结格式与metadata
  -> RAW41/IPD32W/FADC24 payload serializer
  -> 完整性与计数复核
  -> typed head-slot commit
  -> inspect/replay/release
```

逐word真实金参考、随机反压、双模拟器、SVA、Yosys结构检查与Erie lint已经通过。尚未实现的是前端一半：从final-gate/K token自动构建RAW scratch、term directory和destination bitmap。

## 2. RTL层次

```text
gatestack_typed_builder_commit_top
  |- gatestack_typed_payload_serializer
  |    |- 128x24 term memory
  |    |- 128-bit append reservoir
  |    `- 128x64 private payload buffer
  `- gatestack_head_slot_sram_adapter
       `- typed context/head slot memory
```

集成顶层只做连线，不插入数据或控制逻辑。Serializer先在私有buffer完成payload并验证，再把commit begin/word流送入slot adapter；slot adapter只在正确末字握手后发布slot metadata。

## 3. 接口相序

### 3.1 RAW41

```text
begin(format=RAW,payload_bits=6642)
  -> token0..161: {token_id,gate9,K32}
  -> internal check/pack
  -> commit_begin
  -> 104 commit_word
  -> done
```

token ID必须严格从0递增至161。最后一字只有低50 bit有效。

### 3.2 IPD32W/FADC24

```text
begin(frozen metadata)
  -> ordered descriptor stream
  -> per-term ordered destination stream
  -> descriptor/event/header cross-check
  -> commit_begin
  -> payload word stream
  -> done
```

term必须按`gate_code`升序、再按`lane_id`升序；单term destination必须按token ID严格升序且无重复。FADC仅在`fanout>21`时切换bitmap。

## 4. Atomic语义

这里的atomic是“对consumer不可见的原子发布”，不是物理SRAM回滚：

1. Serializer在commit前检查格式、上下文/head范围、payload尺寸、descriptor计数、event计数与排序；
2. 任一错误在commit前产生`done_error`，不发起slot commit；
3. slot adapter在末字前不置valid；
4. replay只能看到完整且metadata一致的head。

## 5. 真实向量结果

| Stage/Head | 格式 | term/event | payload/word | 结果 |
|---|---|---:|---:|---|
| S0/H0 | IPD32W | 32/127 | 2168 bit/34 | 逐word PASS |
| S3/H4 | FADC24 | 61/814 | 6520 bit/102 | 逐word PASS |
| S0/H0 | RAW41 | 32/127 | 6642 bit/104 | 逐word PASS |
| S1/H0 empty | IPD32W | 0/0 | 128 bit/2 | 逐word PASS |

集成总计4 commit、4 replay、4 release、242 word，零mismatch和零protocol error。非法129-bit空IPD在commit前原子拒绝。

## 6. 验证签核

| Gate | 结果 |
|---|---|
| Icarus 12.0 | PASS |
| Verilator 5.020 + SVA | PASS，0 warning/error |
| Yosys 0.33 `memory_collect; check` | PASS |
| Erie独立lint | 3个RTL均0 warning/error |

复现入口：

```bash
sim_hitflow/run_gatestack_typed_payload_serializer_checks.sh
sim_hitflow/run_gatestack_typed_builder_commit_checks.sh
```

## 7. 为什么还不能冻结为论文主架构

当前C0实现为确保commit前发现全部错误，增加了128x64私有payload buffer。这与下游104x64 head slot形成存储复制，也使每个payload至少经历一次额外写和读。它适合作为功能基线，但不应包装成主创新。

主架构应采用：

```text
RAW scratch + canonical term bitmap directory
  -> metadata/policy冻结
  -> 只读所选格式
  -> 直接流式写slot
  -> 正确末字原子发布
```

由于RAW scratch始终存在，压缩分析失败可在commit前回退RAW；commit后的内部错误通过slot协议abort并保持slot invalid。这样可以删除私有payload副本。随后再用两个workspace重叠下一head capture/analyze和当前head emit，共享一套serializer。

## 8. 下一阶段验收条件

1. 从162个final-gate/K token自动建立按`gate_code/lane`规范排序的term directory；
2. 生成`4x32x162` destination bitmap、fanout和全部policy metadata；
3. 不依赖TB外部descriptor/destination流，自动选择IPD/FADC/RAW并逐word匹配金参考；
4. 实现无私有payload副本的直接流式commit变体；
5. 比较C0 buffered、C0 streaming、C1 dual-workspace的真实trace周期、存储访问和目标库PPA；
6. 扩大bit trace并补齐valid825与DC/STA/SAIF/mapped LEC。

完成上述第1至3项后，才可声称“完整C0片上三格式builder”；完成第4至6项后，才具备重新请求DATE接收级审稿的证据基础。
