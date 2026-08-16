# VL-GS-TTB：DVCO、ABIC 与 Local5 阈值幅度纠错

> 日期：2026-08-01  
> 证据边界：本文包含 `[prof-ordered]`、`[bounded-model]`、`[rtl-directed]`、
> `[checkpoint-static]` 和 `[open-synth]`；不包含 DC、STA、SAIF 或目标工艺 PPA。

## 1. 本轮结论

本轮完成了两项架构控制机制和一项会改变 Local5 硬件边界的语义纠错：

1. **DVCO（Dual-Vocabulary Context Overlap）**：Motion 使用两个物理词表
   bank，在消费 context i 的 term body 时构建 context i+1 的 eager header；
2. **ABIC（Atomic Bind-and-Issue Coalescing）**：Local5 在 slot 首次绑定时，
   将同拍 update 转发给对应 primary，并允许输出同拍退休/再装载；
3. fullres post-G0 首次真实运行证伪了“`k_orig`严格0/1”的旧合同。正确合同
   是 `K = event * theta_block`，其中 theta 是每个 attention block 的 ATLIF
   标量阈值。

阶段判定：

- DVCO 最小 RTL：**GO**，但收益仅相对串行 VL，不是相对 raw-gate 的 PPA；
- ABIC 最小 RTL：**GO**，证明消除了实现气泡，不单独宣称 DATE 创新；
- Local5 旧 projection bit-exact：**撤销**；relation、term 多重集和事件支持集
  证据仍有效，数值 projection 必须加入 theta 折叠后重签；
- Motion/Local5 fullres 算法主线：**当前均不具备论文主结果资格**，因为
  AEE 明显落后同协议 NB0。

## 2. Fullres 结果改变了主线判断

统一 paper-W15 协议为 480x640、无 crop、`T2x15x15`、valid825。最佳浮点
结果为：

| 模型 | epoch | AEE | benchmark AAE | spikes | firing | energy proxy |
|---|---:|---:|---:|---:|---:|---:|
| NB0 | 29 | **1.4454** | 6.5128 | 126.1156G | 9.7927% | 107137.62 uJ |
| H67 Motion | 29 | 2.0730 | 7.9029 | 87.9821G | 6.8450% | 77853.53 uJ |
| H66d Local5 | 29 | 2.0912 | 7.9574 | 89.8206G | 6.9880% | 79361.90 uJ |

相对 NB0：

- Motion spikes 减少 30.24%，energy proxy 减少 27.33%，但 AEE 恶化 43.42%；
- Local5 spikes 减少 28.78%，energy proxy 减少 25.93%，但 AEE 恶化 44.68%。

hardware-order 部署复核为 Motion AEE 2.088046、Local5 AEE 2.109147。两条线
都没有出现“只因定点导致失败”的情况；主要差距在 fullres/W15 算法适配本身。
因此不能继续依据 crop/W9 的 H67 AEE 1.46 把 Motion 写成唯一正式主线，也不能
因 Local5 架构更丰富而忽略 fullres 精度。

## 3. Motion DVCO

### 3.1 数据流

```text
SCS class enumeration
  -> header builder -> bank A/B commit queue
                         |
projection term body <- bank A/B ordered consume
```

bank 状态为 `EMPTY -> BUILD -> COMMITTED -> ACTIVE -> EMPTY`。两个 bank
交替分配；header builder 只有在同 bank 的前前 context 已消费完后才能覆盖。
raw fallback 只提交 mode，fast context 提交 mode/count 和 slot-to-gate 表。

### 3.2 profile100 有界模型

| slots | active context | header cycles | serial VL | dual-bank | header hidden | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| S2 | 339689 | 816362 | 7917396 | 7159749 | 92.81% | 1.1058x |
| S4 | 339689 | 939874 | 8040908 | 7161420 | **93.58%** | **1.1228x** |
| S6 | 339689 | 940355 | 8041389 | 7161476 | 93.57% | 1.1229x |
| S8 | 339689 | 940355 | 8041389 | 7161476 | 93.57% | 1.1229x |

S4 仍是键位数 Pareto 点。1.1228x 只比较“串行 VL”与“双 bank VL”，不能
与 9-bit raw 链路直接比较，也不能写成端到端加速。

### 3.3 RTL

新增 `qfit_vl_gs_ttb_motion_dvco.sv`，定向覆盖 fast/raw/fast 三 context、
输出随机反压、bank-full 禁止覆盖和上下文顺序。结果为 5 个 term 零失配、
4 个 build/body overlap 周期和 3 个 bank-full wait 周期。

默认 S4 的 Yosys 结构代理为 143 cells、88 memory bits。该数字只证明没有
异常展开，不是面积。

## 4. Local5 ABIC

### 4.1 为什么需要 ABIC

旧 decoder 有两个纯实现气泡：

1. 输出寄存器不能同拍退休并装入下一项，因此上限接近一项两拍；
2. fill 的 update 写入后，primary 必须再等一拍才能读取 slot。

ABIC 增加 elastic refill 和同地址 commit-forward：

```text
update(set,slot,gate) ----+----> slot table commit
                          +----> same-cycle primary gate forwarding
primary ------------------------> elastic output register
exception ----------------------> ordered exact join
```

### 4.2 有限 FIFO 模型

1498-term 定向 trace、无下游反压时：

| S | decoder | D | cycles | producer stalls | fill blocks |
|---:|---|---:|---:|---:|---:|
| 4 | registered | 1 | 2998 | 1497 | 1 |
| 4 | elastic-only | 1 | 1596 | 96 | 96 |
| 4 | elastic-only | 4 | 1501 | 0 | 1 |
| 4 | ABIC | 1 | **1500** | **0** | **0** |
| 6 | elastic-only | 1 | 1644 | 144 | 144 |
| 6 | ABIC | 1 | **1500** | **0** | **0** |

S4 的 288 个 raw bypass 在 D=1 下不会增加无反压总周期。8-ready/4-stall
敏感性中，S4/S6 ABIC 都是 2248 拍；周期由下游服务率决定，而非 exception
流失控。该反压模式是模型，不是真实 backend trace。

### 4.3 RTL

新增 `qfit_vl_gs_ttb_abic_decoder.sv`，对真实 1498-term CSV 回放：

- 1498 项 gate 和 payload 顺序零失配；
- 96 次 fill 全部走同拍 commit-forward；
- 生命周期总计 1502 拍，除 1498 term 外只有 start、首装载、末退休和 end；
- Icarus、Verilator `--assert`、lint 和 Yosys check 均 PASS。

将 slot valid 改为 packed bitmap、gate 数据取消无意义 reset 后，默认 S6
从 Yosys 910 cells 降为 152 cells，gate table 保留为 1728-bit 单 memory、
单写端口。该变化说明综合形态已修正，不能换算 ASIC 面积。

## 5. Local5 阈值幅度纠错

### 5.1 被证伪的旧假设

Local5 score 使用：

```text
q_event = q_orig > 0
k_event = k_orig > 0
```

但 value 路径实际使用：

```text
attn = sum(gate * k_orig)
```

旧 profiler 要求 `k_orig in {0,1}`，真实 fullres callback 因此 fail-closed。
这证明旧 Local5 projection RTL 的 one-bit K 输入不能直接宣称软件数值等价。

### 5.2 正确的 FAED 因子化

checkpoint 静态审计得到 12/12 个 `sn_k` 阈值均为标量，范围
0.9999954104 至 1.0，9/12 精确等于 1，最大偏差 4.59e-6。于是：

```text
K = E_K * theta_block
gate * K * W = gate * E_K * (theta_block * W)
```

据此提出双线共享的 **FAED（Factorized Amplitude-Event Dataflow，幅度-事件
因子化数据流）**：

1. score、稀疏 relation、TTB 和 K buffer 只传 1-bit 支持集 `E_K`；
2. 每个 attention block 只保留一个 theta descriptor；
3. weight epoch 开始时一次性生成 `W_theta = theta * W`；
4. Motion term 和 Local5 gate-equivalence term 均复用 `gate * W_theta`；
5. 不为每个 spike 传输重复的多位幅度。

这比“所有数据严格二值”更准确，也更接近 PHI 的 primary/side-information
分解，但本工作没有近似 residual：若 theta 定点合同冻结，分解在该数值域内
是代数等价的。Prosperity 的 exact reuse 用于复用 `gate*W_theta`；Bishop 的
TTB header 纪律用于携带 block/weight epoch，而不是逐 term 携带 theta。

### 5.3 当前证据边界

- `[checkpoint-static]` 已证明参数是 block 标量；
- runtime post-G0 profile 尚未完成，当前 GPU 被其他任务占用；
- profiler 已改为 fail-closed 检查有限、非负、支持集一致和单非零幅度；
- watcher 已加 4096 MiB GPU 占用门槛，PID 由系统后台维护，空闲后自动重跑；
- theta 的定点位宽、`theta*W` 舍入次序和 valid825 增量误差尚未冻结。

## 6. DATE 标准自评

本轮无法调用新的独立 subagent：第三方审稿代理额度仍不可用。因此以下是作者
侧审阅，不能冒充独立评分。

| 维度 | 自评 | 判断 |
|---|---:|---|
| workload 动机 | 4.0/5 | Motion profile100 和 Local 定向 trace 扎实 |
| 架构抽象 | 3.0/5 | VL 生命周期、DVCO、ABIC、FAED 已形成统一合同 |
| 单机制新颖性 | 2.2/5 | 双缓冲、elastic refill、forward 单看都常规 |
| 组合可辩护性 | 3.0/5 | 必须绑定 gate vocabulary 与 ATLIF 标量幅度特征 |
| RTL 可信度 | 3.8/5 | 定向回放、双仿真器、SVA、综合可读 |
| 系统/物理证据 | 2.0/5 | 无真实 backend trace、DC、STA、SAIF |
| 算法主结果 | 1.5/5 | fullres AEE 比 NB0 差 43% 至 45% |

**总体自评：2.6/5，Weak Reject。** 相比上一轮，架构控制闭环更完整，且
发现并纠正了关键 K 语义；但 fullres 精度已经成为比硬件创新更高优先级的
拒稿原因。不能继续靠增加 RTL 行数把当前版本迭代到 accept。

## 7. 下一阶段门槛

1. 等 GPU 空闲后完成 Local5 100-sample runtime theta/support profile；
2. 软件侧补 theta 保留、定点折叠和 theta=1 三组 valid825；
3. Motion/Local5 至少一条 fullres 线把 AEE 恢复到可接受 Pareto 区域；
4. 将 FAED weight loader 和 block descriptor 接入相同 projection 边界；
5. 用真实 backend ready trace 重放 ABIC，而非周期性敏感性模式；
6. 形成 raw、串行 VL、DVCO/ABIC、FAED 的同约束 DC/STA/SAIF 表。

在第 3 项未满足前，硬件继续保留双线 RTL 研究原型，但不冻结论文唯一主线。
