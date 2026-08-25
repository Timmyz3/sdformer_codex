# M79 Precision-Elastic PWP VCS 独立打铁 R1

日期：2026-08-23  
角色：独立审阅；不采用生产脚本的 `PASS` 作为结论前提  
对象：`rtl_m79/precision_elastic_pwp_beat_assembler.sv`、配套 SVA/TB/filelist/contract、生产 sealed run，以及 M78 `SHARED_32B` 周期假设  
结论：**隔离定向功能 GO；M78 shared32 周期闭环、有限队列/宏集成、DATE 性能与 PPA 主张均 NO-GO。**

## 1. 独立复核结论

生产 exact-SHA 输入六项均与合同一致。我在新的隔离 `RUN_DIR` 重跑原 VCS/SVA runner，得到：

```text
PASS M79 directed transactions=136 beats=512 escapes=8 stalls=12 lanes=96 protocol_attacks=2 widths=8,9,10,11,12
```

此外使用独立 TB（没有复用生产 TB 的 pass counter）重测 signed 极值、padding 和连续事务启动间隔，得到：

```text
M79_INDEPENDENT_II width=8  beats=3 command_ii_cycles=4
M79_INDEPENDENT_II width=9  beats=4 command_ii_cycles=5
M79_INDEPENDENT_II width=10 beats=4 command_ii_cycles=5
M79_INDEPENDENT_II width=11 beats=5 command_ii_cycles=6
PASS M79 independent hammer checks=12 signed_extremes=4 widths=4 padding_attacks=3 missing_last=1
```

因此，M79 已经证明的事实是：

- 96 lane 的 8/9/10/11-bit little-bitstream 解包和 12-bit signed extension 正确；独立测试覆盖每种位宽的负极值、正极值、`-1` 和 `0`。
- payload/beat/padding 几何独立复算为：8-bit=`768b/3 beat/0 pad`，9-bit=`864b/4 beat/160 pad`，10-bit=`960b/4 beat/64 pad`，11-bit=`1056b/5 beat/224 pad`。
- 生产向量对 8/9/10-bit 穷尽全部 codeword；11-bit 命中 2028/2048 个 codeword，并命中两端 signed 极值。
- 12-bit 是零 beat、零 payload 的 escape **控制 token**；它没有实现 bit-sparse fallback 权重读取或执行。
- 生产 VCS 覆盖 output stall、premature last、9-bit nonzero padding；独立 VCS 又覆盖 9/10/11-bit nonzero padding 和 final beat 缺失 `last`，均 fail closed。

完整机器可读结果见 `m79_independent_hammer.json`。

## 2. P0：shared32 性能模型没有被当前 RTL 支撑

### P0-1：M78 按 beat 数计费，M79 的连续启动间隔为 beat+1

M78 的 `pwp_service_cycles()` 只计算 `ceil(PWP_bytes / 32B)`，即 8/9/10/11-bit 分别为 3/4/4/5 cycle；随后直接把这个数乘以 PWP 使用次数计入 candidate compute。

M79 是单 transaction buffer：command 被接受后才能进入 collecting；最后一个 beat 后产生 output；下一 command 最早在 output retirement 同拍接受。独立 VCS 在 `output_ready=1`、没有人为 stall 的最佳条件下实测最小 command II 为 4/5/5/6 cycle。换言之，beat 带宽本身正确，但当前 RTL 没有证明 M78 假定的无气泡可持续服务率。

Cap11/SHARED_32B 的 M78 原数字为：

| 项目 | 数值 |
|---|---:|
| bit-sparse cycles | 1,114,402,488 |
| candidate cycles | 790,689,185 |
| reported speedup vs bit-sparse | 1.4094065x |
| regular PWP uses | 58,969,374 |

若每个 PWP 的这 1 cycle 都无法隐藏，敏感性结果变为 `849,658,559 cycles / 1.3115886x`。这只是**无隐藏敏感性**，不是替代 headline：真实值需要 ordered PWP/correction 序列、有限队列和共享端口冲突仿真确定。当前证据既不能证明所有额外周期都暴露，也不能证明它们都能隐藏。

修复门槛：至少实现 ping-pong assembler 或 command/data skid queue，使连续事务的 RTL II 等于 3/4/4/5；并用真实 ordered 序列在有限 queue、同一 256-bit SRAM 端口和 consumer backpressure 下重放。

### P0-2：没有 consumer、有限 buffer 或真实宏接口

当前模块含一个 1280-bit `buffer_q` 和一个 1152-bit canonical output register，输出接口本身是 1152-bit；没有 SRAM address/CE/WE、banking、read latency、macro DB、ECC、跨 bank 边界或 consumer accumulate 接口。它是正确的协议适配器，不是 M78 所需的完整 shared32 datapath。

修复门槛：把 256-bit macro read、assembler、96-lane consumer 和 correction-weight 共享仲裁接在同一 top；冻结 SRAM compiler shape，并用 DC/STA/Formality/SAIF/PTPX 给出 area、Fmax 和 energy。

### P0-3：12-bit escape 只发 token，fallback 路径没有闭合

M78 Cap11 的 catalog 仅 1 个 12-bit outlier，但 heldout 中实际触发 362 次。M79 对 width12 不读 PWP beat，只输出 `escape=1`；没有证明 tag 能驱动正确 partition/pattern/output-block 的 bit-sparse 权重回退，也没有证明 fallback 与正在使用的 shared32 port/queue 不冲突。生产 TB 的 8 个 escape 是合成 tag，不是 M78 这 362 次真实有序回放。

修复门槛：将真实 362 次 escape 注入 integrated top，检查地址身份、bit-sparse 数值等价、端口仲裁和完成顺序。

### P0-4：不能把 isolated 1.409x 包装成 DATE 系统 headline

M78 自身声明 valid825-internal、isolated-module、accuracy=false、full-system=false、PPA=false。`12.588x vs dense` 主要继承 bit-sparse 相对 dense 的约 8.93x 基础收益；本轮精度弹性相对强 bit-sparse baseline 的增量是 1.409x（且尚未由 M79 周期闭合）。DATE 主结果必须给同工艺、同频率/面积/带宽约束的强基线和全网络端到端数据。

## 3. P1：协议与 evidence 仍需加固

- `ap_legal_accepted_width` 要求任何 accepted command 都合法，但 RTL 的 `command_ready` 不按 width gating；非法 width 会先 `command_accept=1`，随后 fault。若将非法 width 纳入 negative regression，该 SVA 会与 RTL 的“接受后 fail-closed”语义冲突。应选择并冻结一种协议：非法命令不接受，或允许接受后报错并修改 assertion。
- 生产 TB 没有 reset-mid-transaction、illegal width、orphan beat、command held while busy、随机 simultaneous handshake、长时间随机 output backpressure；现有 SVA 也没有 beat index/beat count、padding 和 command-to-output tag conservation 的端到端 property。
- `cp_*` 是 cover property 命中，不等于 assertion 完整性；尤其不能据此推出 bounded liveness 或无死锁。
- 名为 sealed 的生产目录 mode 是 `775`，顶层文件仍为 `664/775`；没有 producer output SHA manifest。runner 本身 SHA 为 `19c7f864...`，但 runner 没有自我 pin。输入身份不错，输出封存和执行器身份仍是 P1。

## 4. 评分

| 维度 | 分数 / 10 | 判断 |
|---|---:|---|
| 隔离 RTL 数值功能 | 8.5 | 四种位宽、signed 极值、padding 和 fail-closed 基本可信 |
| 协议/反压完整度 | 6.0 | output stall 已测；随机并发、liveness、非法 width 合同不足 |
| 硬件创新性 | 6.5 | precision-elastic PWP + block escape 有组合价值，但关键 novelty 仍在 M78 算法/DSE，M79 只是单缓冲适配器 |
| 性能优势证据 | 4.0 | beat 几何成立；持续 II 实测与模型差 1 cycle，且没有 ordered finite-resource replay |
| 宏/PPA/物理可实现性 | 2.0 | 无 macro top、DC/STA/Formality/SAIF/PTPX |
| DATE 论文证据完整度 | 3.0 | isolated internal screen，非 accuracy/full-system、非强基线同约束 silicon/PPA |
| 综合里程碑 | 4.8 | 功能子模块可保留，性能 headline 不准入 |

创新性判断是“有潜力但不够独立”：精度弹性存储（相对 fixed12 约 25.35%）和稀有 block escape 是好的 architecture ingredient；要达到 DATE 强稿，需要将它提升为可持续无气泡、宏感知、端到端可测的 **elastic PWP supply engine**，而不是只停在 unpack assembler。

## 5. 下一里程碑 GO 条件

1. 双缓冲/两 context RTL 在 VCS 中实测连续事务 II=`3/4/4/5`，同时覆盖 mixed width、consumer stall 和共享端口 correction 插入。
2. 用 M78 的真实 ordered PWP/correction/362 escape 序列重放，至少报告 queue depth sweep、stall cycles、port utilization、p50/p95 sample cycles。
3. 集成真实或可替换 macro wrapper；DC/STA 证明 3ns（或论文冻结频率）无违例，Formality PASS，SAIF/PTPX 报告动态功耗。
4. 在相同 SRAM 容量、端口和频率下对 Fixed12、bit-sparse、Phi-like fixed12、precision-elastic Cap11 做 A/B；headline 只采用 full-network/equal-rate/equal-accuracy 结果。
5. 修复非法 width 的 RTL/SVA 合同，并扩展 constrained-random protocol regression。

在以上条件满足前，本里程碑应表述为：**“M79 isolated precision-elastic PWP beat assembler passed directed and independent VCS; system speedup and PPA remain unadmitted.”**
