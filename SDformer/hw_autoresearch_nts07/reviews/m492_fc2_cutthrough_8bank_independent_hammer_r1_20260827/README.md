# M492 cut-through FC2 独立打铁评审

## 裁决

**95/100，GO 到 matched 3.0 ns DC；NO-GO 到正周期加速、系统加速或论文 PPA 主张。**

本评审按 receipt-blind 执行：未读取
`results/m492_fc2_cutthrough_8bank_equal_bandwidth_vcs_r1_exact_20260827`
中的生产回执或日志。评审只读生产合同、runner、RTL、SVA、TB，并在本目录
独立重编译/仿真。工具为 Synopsys VCS V-2023.12-SP1；未使用开源仿真、
DC、GPU，也未修改生产文件或 `docs/359`。

## 关键结论

### 1. M490 未发现组合环

- cut-through 完成判定只依赖已登记 slot 状态、银行 `valid`、身份检查和输入
  payload，不依赖 `bank_rsp_ready` 或 `bank_rsp_accept`。
- `core_rsp_accept` 只进入同 slot 退休复用的 `req_slot_open`；请求合法性不依赖
  slot 是否空闲，因此该旁路不会回灌 `protocol_error`。
- 冻结 M218 的 `mem_req_valid` 来自登记 FIFO/scoreboard 状态和合法响应释放，
  `mem_rsp_ready` 来自登记 skid 状态；未发现从 M490 `core_req_ready` 返回
  `core_rsp_accept` 的闭环。
- 独立 VCS 编译无 warning/error，也没有组合环诊断。仍存在从银行最终响应到
  core response payload 的长组合路径；这不是功能环，必须由 matched DC 决定
  3.0 ns 是否成立。

### 2. 同拍 slot reuse 未观察到旧响应污染

M490 同一时钟边沿先退休旧 slot，再以非阻塞赋值登记新 generation；当旧响应
在该边沿被 core 消费时，其最终银行 beat 被显式禁止写回 slot。独立定向用例完成：

- generation 1、mask `ff` 的响应在 stall 后保持稳定；
- 同拍退休 slot 0 并登记 generation 2、mask `03`；
- generation 2 的 mask、tag、两银行权重全部精确，未携带 generation 1 权重；
- 最终计数为 2 bundle request/response、10 bank request/response，0 fault。

### 3. stall payload 稳定，但生产集成覆盖有一处缺口

生产 M492 集成复跑中 `cp_core_response_stall=0`，所以该次集成测试里的 response
stability assertion 是空通过；`result_stalls=44` 是最终结果端，不能代替 adapter
response stall。独立定向补测把 adapter core response 连续 stall 三拍，
`cp_core_response_stall=3`，payload/tag/mask 保持稳定且 SVA 0 fail。

类似地，生产集成复跑的 `cp_partial_request_distribution=0`；独立补测用 `0f` 后
`f0` 的银行 ready 分两批发送同一 bundle，覆盖 partial distribution 1 次、pending
stall 1 次，没有重复请求。建议下一版生产合同把这两个 cover 也改成强制非零；
当前独立证据足以放行物理诊断，不应把原集成覆盖写成“全部非零”。

### 4. M491 与 M349 的等带宽比较成立

两边都连接八个完全相同的 `m349_fc2_scalar_bank_memory_model`：每银行 128 bit、
L4、每周期至多一个 word，合计峰值 8 word/1024 bit 每周期。两边共享相同的
`request_allow`、`response_allow`、result/done backpressure；每次架构运行均复位
edge ordinal，所以顺序执行没有调度相位偏置。候选为一份 M218 Acc24 服务加
M490，基线为八份 M219 Acc24 服务，这是要由 matched DC 定价的状态组织差异。

请求/响应的 `(block,slice,channel)` 多重集在两边完全一致，银行权重和最终
Acc24 数值均为 0 mismatch；因此比较的是相同工作，不只是相同峰值端口。

### 5. 周期采样公平，但只有四个非零工作点

周期定义为 `header_accept` 到 `token_done_accept` 首尾均含，起止由同一个
posedge monitor 记录，消除了 active-region 竞态。独立复跑得到：

| output blocks | events | M491 K8 | M349 K1x8 | ratio |
|---:|---:|---:|---:|---:|
| 1 | 20 | 51 | 51 | 1.000000 |
| 2 | 41 | 131 | 131 | 1.000000 |
| 4 | 90 | 486 | 486 | 1.000000 |
| 8 | 110 | 1231 | 1231 | 1.000000 |
| 1 | 0 | 14 | 14 | 1.000000 |

最后一行是零事件边界测试，不是第五个性能 workload。四个非零点只能证明
定向 standalone replay 上周期持平，不能证明冻结 120-record trace、完整 FC2、
FFN 或系统性能。

### 6. 独立 VCS/SVA 结果

主复跑：10 clean、2 reset、4 protocol attack；numeric/transaction multiset/
weight mismatch 均为 0。request/result/raw stall 分别为 376/44/1153；候选和
基线 younger-before-older 分别为 1080/7024；所有合同强制 cover 非零，0 assertion
fail。

额外定向复跑：partial request distribution 1、pending request stall 1、adapter
response stall 3、same-cycle slot reuse 1、cut-through response 1，0 assertion fail。

## 评分

| 维度 | 分数 | 说明 |
|---|---:|---|
| RTL/协议正确性 | 24/25 | 未见组合环或污染；零周期 SRAM response 未验证 |
| 同资源公平性 | 24/25 | 端口、模型、调度、事务工作量一致；仍是合成定向负载 |
| 验证充分性 | 23/25 | 主复跑和补测均过；两个关键 cover 需补进生产合同 |
| 主张纪律 | 15/15 | 明确 cycle parity，不宣称正加速/PPA/系统指标 |
| 可复现与封存 | 9/10 | exact SHA 输入明确；物理门和冻结 trace 尚未完成 |
| **总分** | **95/100** | **GO 到 matched DC** |

## matched DC 放行条件

1. M491 与 M349 使用同一 TSMC28 library、3.0 ns 约束、相同 I/O delay/drive/load。
2. 候选面积必须包含 M490 全部 slot/weight/hold 状态；不得只报 M218 core。
3. 八个 SRAM 宏若 black-box，必须在两边同量排除并把结论限定为 logic-only；
   若计入宏，则两边使用同一宏、同一数量、同一 PVT。
4. 周期已经持平，所以 DC 后唯一合法派生是 throughput/area；只有候选总逻辑面积
   小于 K1x8 才是 GO。能量效率需同一工作量 SAIF/PTPX 后另行裁定。
5. 在 frozen 120-record trace 复跑前，不得写“完整 FC2 加速”；不得把 1.0 parity
   包装为 cycle speedup。

## 论文主张边界

现在可写：`Under a directed equal-bandwidth eight-bank replay, the shared-state
K8 organization preserves the K1x8 cycle count exactly.`

现在不可写：正周期加速、物理加速、完整 FC2/FFN、系统加速、能量优势、
paper-PPA-ready 或 headline。DATE 主贡献只能在 matched DC/PTPX 与冻结 trace
闭合后升格。

