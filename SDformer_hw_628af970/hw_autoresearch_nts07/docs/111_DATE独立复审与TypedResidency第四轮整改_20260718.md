# DATE独立复审与Typed Residency第四轮整改

## 一、结论

本轮在`docs/110_TypedSlotMetadata与IPD选择性驻留架构闭环_20260718.md`完成后，委托独立子代理按DATE硬件架构稿标准进行两次只读复审。

第一次复审评分为`3.0/5，Weak Reject`。审稿人确认重复PEEK、Adaptive与residency格式冲突已经关闭，但指出cache release缺少payload tag，所谓“tag-coherent lifecycle”仍依赖串行时序假设；同时实验报告没有RTL/向量/runner哈希和工具版本。

完成第四轮整改后，第二次复审确认上述两项关闭，最新评分为：

| 项目 | 结论 |
|---|---|
| 推荐意见 | **Weak Reject，接近Borderline** |
| 综合评分 | **3.2/5** |
| 技术状态 | typed dispatch与IPD-only residency控制闭环成立 |
| 投稿状态 | 仍缺目标PPA、builder/policy、扩展workload和部署精度 |

该评分不是DATE接收概率，也不是正式外审结果，只用于内部成熟度门槛。当前硬件部分较第三轮`2.8/5`有实质提升，但仍不能标记为可投稿或ASIC签核。

## 二、独立复审确认已关闭的缺口

### 2.1 提交期格式元数据与重复PEEK

- slot commit一次性解析RAW41/IPD32W/FADC24格式身份；
- PLAN、atomic commit、router、projection共享同一format；
- metadata有效时Adaptive decoder直接启动对应child；
- word0 PEEK只作为legacy fallback保留。

复审结论：`[RTL+SVA]`关闭。

### 2.2 Adaptive与descriptor residency冲突

- 只有IPD执行cache lookup/fill；
- FADC与RAW从word0完整回放；
- resident route强制绑定IPD；
- warm offset、cache ownership和route/format在atomic commit统一检查。

四个真实窗口与两个合成mixed用例均为零mismatch；两个mixed用例的11个IPD head各跨23个warm tile，得到`253`次命中和`11`次释放。

复审结论：正常串行执行路径下的格式污染与错误offset问题已关闭。

### 2.3 Stale release/refill竞争

第一次复审发现descriptor cache release只有`context/head`，延迟的旧release理论上可能清除同一slot的新tag line。

整改后release携带`payload_tag`：

| cache状态 | release tag | 行为 |
|---|---|---|
| line存在 | 匹配 | 清line，增加release计数 |
| line不存在 | 任意 | 幂等no-op，不阻塞生命周期 |
| line存在 | 不匹配 | 保留line，置protocol error，增加tag-mismatch计数 |

定向TB执行“释放旧line -> 新tag refill -> 注入旧tag延迟release -> 读取新line -> 正确tag释放”，确认旧release不删除新line。descriptor cache、dualtag lifecycle、output-tile residency、control-plane、single-context、四stage和mixed回归全部通过。

复审结论：`[RTL+TB+SVA]`关闭。

### 2.4 实验可复现性

`results/gatestack_typed_residency_fulltop_20260718/report.json`新增：

- single-context RTL filelist bundle SHA-256；
- runner SHA-256；
- 六个向量目录全部文件的bundle SHA-256；
- 无驻留对照报告SHA-256；
- Icarus、Verilator和Yosys版本。

报告明确区分四个真实窗口与两个合成mixed功能用例，不再把后者描述成独立真实workload。

复审结论：当前实验达到可复核水平。

## 三、审稿人要求收紧的Claim

### 3.1 Commit检查边界

commit只验证word0中的格式身份、version、保留位和tag，并决定slot是否可以发布。第二header、term count、descriptor长度和destination完整性仍由decoder检查。

允许写：

> 提交期验证格式身份并生成tag-coherent typed slot metadata。

不允许写：

> 提交期已经证明完整压缩payload合法。

### 3.2 External descriptor fill边界

外部fill会经过IPD-only资格和容量检查，但没有逐entry证明内容等于同tag slot中的IPD descriptor。它是trusted prefill接口，不是端到端自验证路径。主实验使用decoder自动fill。

### 3.3 Adaptive命名边界

当前格式由上游预编码，硬件只做typed dispatch。没有on-chip builder和成本policy前，不得写“硬件自主自适应选择格式”。更稳妥名称是：

> Runtime Typed Heterogeneous Sparse Dispatch

“Adaptive CSR”只能作为RTL历史模块名或描述header携带格式，不应成为算法自主决策claim。

## 四、开放结构证据更新

tag-qualified release加入比较器和错误计数后，三种配置重新生成网表并完成开放LEC：

| 配置 | cells | `$mem_v2` | LEC |
|---|---:|---:|---|
| 静态IPD + residency | 4191 | 13 | 4832/4832 |
| Typed Adaptive + no residency | 4958 | 11 | 4762/4762 |
| Typed Adaptive + IPD-only residency | 5249 | 14 | 4832/4832 |

选择性驻留相对Typed无驻留增加291 generic cells，约5.87%；完整Typed候选相对静态IPD+驻留增加1058 cells，约25.2%。这组数字只作结构代理，不能替代面积、频率、功耗或EDP。

## 五、最新剩余P0

### P0-1 目标库物理证据

需要在同一目标`.db`、PVT、SDC和SRAM macro下比较Direct RAW、IPD-only、FADC-only、Typed、Typed+Residency，并给出DC/STA、mapped SAIF、SRAM读写能量和mapped-netlist LEC/Formality。

### P0-2 On-chip builder/policy或降级命名

必须二选一：

1. 实现并计入IPD/FADC/RAW payload builder与确定性成本policy；
2. 正式把主张降级为runtime typed dispatch，并把格式生成放到系统边界之外。

### P0-3 Workload与full encoder

扩展到多sample、多block、多window，报告真实格式分布、p50/p95/p99/worst周期和fallback；将attention、projection、ATLIF、skip SRAM与外存纳入Amdahl/FPS/带宽/能量分账。

### P0-4 部署精度与覆盖收敛

完成INT8 valid825精度合同；增加随机backpressure/reset/abort、坏header1/长度、tag alias、cache release/refill长序列和多context覆盖，并报告功能覆盖率。

## 六、论文贡献定位

当前可以支持的硬件表述是：

> GateStack使用运行时类型化slot，在上游预编码的IPD32W、FADC24与RAW41精确表示之间执行格式限定分派，并通过IPD-only descriptor residency、共享multicast后端和output-tile-stationary累加形成统一稀疏投影数据流。

Typed metadata与tag-qualified residency是使组合架构可验证、可异常恢复的控制闭环，不是独立主创新。DATE主贡献仍必须落在：

1. final-gate等价类驱动的workload映射；
2. 低/高扇出异构表示与共享multicast执行；
3. output-tile-stationary避免head-major partial-sum spill；
4. 统一事务合同带来的系统级正确性与可实现性。

## 七、下一阶段决策

在没有目标库与valid825结果时，不继续扩展第四种descriptor格式，也不把0.6%的residency周期收益当主线。CPU侧优先补异常/长序列覆盖和builder/policy成本模型；获得库后立即执行同边界物理消融。软件侧优先完成valid825部署合同与更大范围真实trace。
