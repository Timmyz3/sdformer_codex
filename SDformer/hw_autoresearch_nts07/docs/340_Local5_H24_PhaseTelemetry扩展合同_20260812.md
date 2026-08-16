# Local5 H24 Phase Telemetry 扩展合同

> 日期：2026-08-12  
> 状态：实现前冻结合同  
> 证据等级：`[模型]+[既有H3/H24 rtl校准]`  
> formal G0：**DENY**

## 1. 决策

H24 不照搬 H3 的“每个资源事件写一行 CSV”。已有 H24 canary 的 candidate trace 为
47,941,735 行、约 3.0 GB；完整双 trace 加 Phase Store 已占约 6.7 GB。若再把
cross-head 1RW 命令逐条复制到 telemetry，会重复保存规则事件，不能形成真正独立的
第二 oracle，也无法扩展到 1,200 个窗口。

冻结为三级证据：

| 事件类别 | H24 保存方式 | 校验方式 |
|---|---|---|
| group/tile/head/phase 边界 | 逐条 semantic record | 与完整 trace 独立重建的边界集合比较 |
| relation/weight/final accepted | 计数 + 全序双 64-bit digest + 首尾锚点 | 与 identity trace 流式摘要比较 |
| cross-head Acc / TCFM5 term-bank | 主/副观察器共同投影 + 独立协议 oracle | 摘要、闭式计数、相序、不变量和 Acc32 联合比较 |

逐条 CSV 只保留低基数边界；高基数规则事件保存
`count + dual-64 ordered digest + first/last anchors`。SV 内摘要明确不是 SHA256 或密码学
承诺；Python 另对 summary 文件计算 SHA256。摘要输入必须包含 cycle 和完整共同事件
投影，不能只哈希计数。

## 2. H 参数化闭式合同

令 `H=HEADS=OUTPUT_TILES`，token=450，head_dim=32，out_dim=32，则：

```text
tile_start/tile_done          = H
head_start/head_done          = H^2
relation req/rsp accepted     = 450 H^2
weight req/rsp accepted       = 1024 H^2
final accepted                = 14,400 H
semantic phase records        = 1 + 2H + 5H^2
Acc32                          = 14,400 H
```

H3 已实测校准为：

```text
phase=52
relation req/rsp=4,050/4,050
weight req/rsp=9,216/9,216
final=43,200
Acc32=43,200
```

H24 的冻结期望为：

| 项 | H24 |
|---|---:|
| semantic phase | 2,929 |
| relation req/rsp | 259,200 / 259,200 |
| weight req/rsp | 589,824 / 589,824 |
| final accepted | 345,600 |
| 五类 aligned accepted 合计 | 2,043,648 |
| Acc32 | 345,600 |

scalar、non-inplace 合同下，cross-head Acc 命令可由协议独立推导。对每个 output
tile：首 head 写 14,400 条；后续 `H-1` 个 head 各读写 14,400 条；最后 drain 读
14,400 条。因此：

```text
cross-head command = 28,800 H^2
read command       = 14,400 H^2
write command      = 14,400 H^2
```

H3 为 259,200 条，和既有实测一致；H24 冻结为 16,588,800 条，read/write 各
8,294,400 条。该公式、地址秩序和 `first-head write -> later-head read/write -> drain
read` 相序共同构成独立协议 oracle，不能只比较两个观察器对同一网线的摘要。

## 3. 第二观察器边界

主 monitor 观察 executor 聚合接口；第二观察器必须绑定更低层的结构，不得简单复制
主 monitor 的输入：

1. cross-head Acc：一个观察路径取 executor 聚合接口，另一路绑定目标
   `qfit_single_port_acc_memory.command_*`；两者共同投影冻结为
   `{cycle,rw,addr,write_data}`，read 时 `write_data=0`；除摘要一致外，还必须满足上一节
   的独立命令计数、地址遍历和相序 ledger；
2. TCFM5：绑定 `qfit_tcfm5_projection_top` 的 public term input 与五个 bank update
   接口；共同投影冻结为
   `{cycle,source,lane,expected_mask,actual_mask,bank_addr[0..4]}`。低层五 bank 事件必须
   在一个 term commit 上聚合后比较，不能把五条 bank update 与一条 term 错当同一
   count；`expected_mask` 从五邻域边界和颜色拓扑独立重建；
3. 两个观察器都不得输出控制信号，不得驱动 ready/valid，不得读取主 monitor 状态；
4. 第二观察器的源码、bind、枚举规格和摘要器必须独立封存。

一个通用模块会出现多个实例，bind 结果必须带 `%m` instance path。verifier 要求
allowlist 中目标实例基数严格等于 1，且 elaboration 中出现任何额外 observer 实例都
fail-closed，不能静默忽略；每个实例必须使用独立文件和独立累计状态。

类型级 bind 只在 observer 与 DUT 共同重新 elaboration 的 telemetry executable 中有效，
不得事后附着到既有 sealed executable，也不得外推到综合后网表。

## 4. 摘要字节合同

两条 64-bit rolling digest 分别采用冻结且不同的更新函数；输出必须写算法名，禁止标为
SHA256。每条事件先序列化为以下无歧义 frame：

```text
domain_tag_len:u16 || domain_tag:ASCII
schema_version:u16 || resource_code:u16
instance_path_len:u16 || instance_path:UTF-8
sequence:u64 || cycle:u64 || payload_len:u16 || payload
```

- 所有整数固定宽度、无符号、little-endian；有符号数据先按 two's-complement 同宽
  bit pattern 编码；
- payload 字段顺序严格采用上一节的共同投影顺序，不允许省略全零字段；
- 每种 resource 分开累计，空流初值和 final value 必须写入 summary；
- sequence 先按 accepted cycle，再按冻结 subevent order 单调递增；同 cycle 的不同
  resource 不得混入同一摘要；
- instance path 属于 domain，不同实例不能共享累计状态；
- verifier 必须有交换、重复、删除、跨 resource 重绑、大小端和字段宽度篡改负例。

Python 对最终 summary 文件做 SHA256，只证明封存文件未变化，不把双 64-bit rolling
digest 宣传成密码学无碰撞承诺。

## 5. 枚举独立性

H3 v1 的 `HEAD_WEIGHT/FRONTEND/READOUT/RELEASE` 状态分组在 monitor 和 Python
verifier 中重复硬编码，属于共同错误风险。H24 前必须：

- 用一份只读 JSON 规格冻结状态号到 phase role 的映射；JSON 必须包含 module、配置、
  enum width、symbol、value、role 和 RTL SHA；
- monitor 在编译时仍采用本地常量，但 runner 记录 RTL source SHA；
- verifier 从 JSON 规格读取，不再复制 monitor case；
- 独立审阅从 `qfit_local5_tagged_t450_job_engine` 的 enum 定义提取状态号，与 JSON 逐项
  核对。当前 Direct pilot 不得用 memo engine 的不同状态编码代替；若扩 memo 路径，
  必须新增独立 module/config 条目。

该 JSON 是验证合同，不是硬件输入，也不能作为 DATE 架构创新。

phase 比较必须是带 sequence 和基数的有序 interval ledger，逐项检查 identity、role、
start/end/duration、重复、缺失和相邻关系；禁止用普通集合比较吞掉重复项。

## 6. 封存与磁盘闸门

H24 pilot 运行前必须满足：

- 运行前冻结 plan、identity、release、monitor、second observer、verifier、runner SHA；
- `requested_identity == actual_identity`，不一致时 exit nonzero，禁止创建 PASS 文件；
- `evidence_payload_bytes` 上限 512 MiB：统计 output 下除 `build/`、`source/` 外所有
  regular file 的逻辑 `st_size`；硬链接按 `(dev,ino)` 只计一次，压缩包按存储文件大小
  计，symlink 一律拒绝；外部 SHA 引用不计入。超过上限 fail-closed，不自动降级；
- v1/v2 身份错误包按不可变 `complete/package digest` 进入 machine-readable denylist；
  正向准入仍优先要求 schema/status 白名单、`identity=MATCH` 和全部 receipt 通过，
  denylist 只是附加保险；
- 所有 digest 在 `complete.json` 写入前生成，complete 只最后写一次；
- 验证后源文件或摘要变化必须导致 receipt 校验失败。

H24 不再保存第二份完整 identity trace。完整 trace 可引用既有封存 canary，通过 SHA
绑定；新运行只保存 compact boundary、摘要和 Acc32。

## 7. 准入判据

H24 pilot 只有同时满足以下条件才可标记 `PASS_H24_PHASE_TELEMETRY_PILOT_NOT_G0`：

1. 身份 exact match；
2. 2,929 条 semantic phase 完整，必要 role 数量满足 H 参数公式；
3. 五类 aligned accepted event 的 count 和 ordered digest 与完整 trace 一致；
4. cross-head Acc 与 TCFM5 的主/第二观察器对共同事件投影的 count、digest、首尾锚点
   一致；cross-head 另满足 `28,800H^2` 精确命令数及读写/地址相序，TCFM5 另满足
   expected/actual mask 和 bank address 拓扑 oracle；
5. 345,600 个 Acc32 与独立软件整数金参考零失配；
6. 至少包括 identity tamper、摘要篡改、事件丢失、事件交换、instance path 重绑和
   bank-mask 错误六类 fail-closed 负例；
7. package 不超过冻结磁盘限额。

## 8. 独立审阅裁决

本合同首轮独立 DATE 审阅为：

```text
3.8/5，Conditional GO for implementation
H24 PASS = NO-GO，直到上述 P0 关闭
Formal G0 = DENY
```

审阅确认 H 参数公式和 H24 数值正确，也认可摘要方案可替代第二份逐事件 CSV 的长期
存储；但“换位置观察同一网线”不是独立 oracle，必须加入闭式协议 ledger 与拓扑重建。

## 9. 明确边界

H24 pilot 即使全部通过，也只证明“四种 H 中最大 H 的单窗口 telemetry 可扩展且与既有
trace/数值一致”。它仍不等于：

- 1,200-window / 100-sample formal archive；
- 462,600 条正式 phase schema 全覆盖；
- EREP candidate RTL admission；
- full encoder 性能、DC/STA/SAIF 或 ASIC PPA；
- DATE 架构创新本身。

因此 H24 pilot 通过后 formal G0 仍保持 **DENY**，下一闸门是有限多窗口/多 stage
抽样，再决定是否用参数化证明和分层 digest 取代全量逐行 archive。
