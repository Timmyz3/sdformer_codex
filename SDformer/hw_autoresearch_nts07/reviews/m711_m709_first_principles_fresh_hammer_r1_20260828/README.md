# M711｜M709 第一性原理新鲜打铁评审

日期：2026-08-28  
模式：独立、只读、fail-closed；未运行 EDA、GPU、训练或远端任务；未修改任何既有文件。  
被审对象：`reviews/m709_first_principles_hardware_innovation_audit_r1_20260828`。

## 1. 结论先行

M709 的证据卫生总体良好，C1/M528、C2/M519 与 C3/M518 的大方向保留；但三个新候选的 admission gate 不能原样执行。独立裁决为：

| 对象 | M711 裁决 | 可写边界 |
|---|---|---|
| M528 dead-write-only 1RW product capture | **KEEP** | `435,293,339` cycle；相对 M468 strongest-zero `1.746753×`、同坐标 bit `1.741232×`；仍是单序列四层 Conv 的 CPU 同账本点，不是系统或 PPA headline。 |
| M519 typed signed K8 | **KEEP + REVISE 数字** | 等服务 K1×8 相对 K8 为 `1.012185–1.039216×`；共享状态/面积/energy/source 才是合法主张。M709 的 K1/K8 `4.89–6.32×`不能由它引用的 M519 四个 active row 复算。 |
| M518 Fixed T10 | **KEEP AS BASELINE** | directed VCS 的 `17` issue cycle/tile、clean N1/N4=`29/80`；不是 workload、面积或系统结果。 |
| PIDP | **REVISE，禁止作为当前独立 novelty** | 只保留为 decoder 的“descriptor-scatter 与 bitmap destination-pull”实现消融；必须先建立新的可执行同资源 baseline。 |
| TDA | **REVISE，保留 pre-RTL fast-kill** | 没有偷用 rank-3，计算对象差成立；但必须把 45 表的驻留/装载、32-read 复制、transpose、配置与完整 service latency 收全。 |
| RS-BN | **KILL AS CYCLE CANDIDATE** | 在现有 32/64/128 B/cycle 坐标上，乐观上界也过不了 `1.15×`；只可在来源 lifetime 已证明、总能量全收费后作为 memory/energy future study。 |

总判决：**FAIL_M709_RECOMMENDATION_GATES__KEEP_C1_C2_C3__REVISE_PIDP_TDA__KILL_RSBN_CYCLE**。M709 不是伪造结果；问题是它把三个“待测 idea”写成了仍可能一日过门的候选，而第一性原理账本已能提前关闭一个，并要求另外两个重写门。

## 2. 证据完整性

- M709 自身 `SHA256SUMS` 3/3 与 outer seal 均通过。
- 本评审绑定的 17 个 canonical source SHA 全部重算一致。
- `docs/359_DATE终局冻结_20260813.md` 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- `recompute_m711_metrics.py` 在本机 Python 3.6 只读执行通过；输出冻结于 `recompute_stdout.txt`。

## 3. C1/C2/C3 复核

### 3.1 C1/M528：KEEP

M528 原始结果的同账本除法可重复：

- `760,350,133 / 435,293,339 = 1.7467534301×`；
- `757,946,784 / 435,293,339 = 1.7412322131×`；
- macro-rounded coordinate=`213,376 B <= 245,760 B`。

M709 对 C1 的数字和 scope 判断成立。唯一不能跨越的边界仍是：一条序列、十样本、四层 bottleneck Conv、CPU cycle model；无完整 RTL recurrence、真实集成宏、PTPX、全网或系统 headline。

### 3.2 C2/M519：KEEP，但 M709 的范围不可复算

M519 引用 receipt 的四个 active row 给出：

| events | K1 | K8 | K1×8 | K1/K8 | K1×8/K8 |
|---:|---:|---:|---:|---:|---:|
| 20 | 259 | 51 | 53 | 5.078431 | 1.039216 |
| 41 | 737 | 131 | 133 | 5.625954 | 1.015267 |
| 90 | 3153 | 486 | 499 | 6.487654 | 1.026749 |
| 110 | 7569 | 1231 | 1246 | 6.148660 | 1.012185 |

因此 cited active range 是 `5.078431–6.487654×`，不是 M709 的 `4.89–6.32×`。这不改变 M709 的核心判断——单 K1 不是稀疏收益的公平分母——但论文与收口表必须使用可追溯行或明确另一个数据源。该 M519 receipt 自身仍有 `P1=4`，且 claim boundary 明确为 directed component VCS，完整 FC2、DC、power、PPA、system 均为 false。

### 3.3 C3/M518：KEEP AS BASELINE

M518 的 sealed VCS 只承认 `17` issue cycle/tile、N1/N4 clean=`29/80`。M709 把它当 exact fairness anchor 是正确的；把它当完整 ATLIF 性能点则不成立。

## 4. PIDP：不是完全重复，但 novelty delta 被 M709 写宽了

### 4.1 与既有机制逐项对照

| PIDP 声称对象 | 既有证据已经覆盖 | 尚可能成立的对象差 |
|---|---|---|
| K3/S2 parity/polyphase 地址 | M523 已做 K3/S2/P1/OP1 tap expansion 与 phase bundling，并用 directed VCS 闭合 | 从 source-scatter loop 反转为 destination-pull loop |
| typed K8 bundle | M523 已产生 K8 tap bundle；M519 已有 typed signed K8 service | bitmap probe 直接形成 C2 command，避免 descriptor 物化 |
| single destination owner / Acc24 | A1-OSG 已是 output-stationary destination-keyed context；PBR4 已有四 phase×四 context、partial-RMW backing | 若能证明不用 PBR4 的 global frontier/directory 仍能 exact close，才是协议简化 |
| close/commit | M534/PBR4 已冻结 frontier、owner、dense commit、persistent directory 与 terminal FSM | 按 destination enumeration 得到静态 close，减少动态 descriptor retirement 检测 |

因此 PIDP **不是 byte-identical duplicate**：destination-pull 的遍历方向与 bitmap-word access 是新实现选择。但 parity、bundling、destination ownership、phase context 与 close 不是新空白。论文只能写成“将已知 polyphase/bitmap pull 映射到 H67 decoder，并与现有 scatter descriptor path 做同资源 A/B”，不能作为第三个从零发明的架构。

### 4.2 当前 gate 不可执行

M709 指定 strongest A1-SC8/A1-ISO8/A1-OSG 为 denominator；然而这些点的正式 runner 在 M596 被判 `FAIL_SOURCE_STATIC`，有 `P0/P1/P2=3/2/1`：terminal state 不持久、shared external resource 不统一、descriptor duplicate/replace 可 false-pass、occupancy 和 aggregate ledger 未闭。M596 的 synthetic `18/18/22/21` 只是 golden smoke，不是 S10 performance。

所以 PIDP 不能直接对这组“best A1”跑 ratio gate。修订路径二选一：

1. 先修完 PBR4/A1 executable baseline；或
2. 更适合 DATE Accept：建立一个最小 destination-owner slice，与 M523 descriptor-scatter slice 在同 bitmap/source、weight、Acc24、commit 和 SRAM 端口下 A/B，不再复活全局 PBR4 调度器。

M705 density=`0.2331877882` 还带来一个必须收费的先验：忽略边界时，dense pull bitmap probe 数相对 active-scatter contribution 数约为 `1/density=4.288389×`。pull 可以用窄 bitmap read 换掉宽 descriptor/RMW，但不能把 probe 当免费；若没有 bytes/energy 或完成泡收益，cycle 很可能不升。

**裁决：REVISE。**保留“scatter-vs-pull 的同资源实现消融”，杀掉“当前独立第三 novelty”和“直接使用 M596 denominator”的写法。

## 5. TDA：没有 rank-3 偷换，但门隐藏了三类资源

### 5.1 数学重算成立的部分

两个 5-input group 的表位数为：

`2 × 32 × 10 × 11 = 7,040 bit = 880 B`。

11 bit 足以容纳五个 signed INT8 常数之和。16-bit physical slot 下 unique table=`1,280 B`；16 spatial lane、每 lane 同拍读两组的简单复制为 `20,480 B`。160 个 Acc24 再加 `480 B`，合计 `20,960 B`，在 M709 的 24 KiB 门下只余 `3,616 B`。

M709 明确以 full-rank T10 为对象、signed MSB 作减法、禁止 rank-3 分母；所以“偷用 rank-3”指控不成立。

### 5.2 被 gate 隐藏的部分

1. **45 组配置身份。**45 个 unique table 即使不复制也要 `57,600 B`；若全部按 16 lane 常驻复制则是 `921,600 B`。只能二选一：常驻并收费，或在 layer/context 切换时收费 table load、broadcast/write、barrier 与 bytes。
2. **完整 service，而非 issue-only。**M518 Fixed 是 `17` issue，但 clean N1/N4=`29/80`，即各自还有 `12` 个非 issue cycle。即使 TDA issue 达到 10，保留相同非 issue 税的上界只有 N1 `29/(12+10)=1.318182×`、N4 `80/(12+40)=1.538462×`，不是 issue-only `1.7×`。理想 8 issue 的对应上界为 `1.45×/1.818182×`，仍未收费 table load、transpose、ROM response、result backpressure。
3. **端口与物理实现。**32 vector read/cycle 不是 32 个免费 lookup。ROM replication、多路选择、fanout、bitplane transpose、配置存储、output queue、bias/threshold 与 signed exact ordering 都必须进入 area/power/latency；`throughput/mm² >=1.25×` 只能由 matched DC/PTPX 后判定，不能由 table bits 预测为通过。

修订 gate 必须同时要求：所有 45 config 的 resident-or-load ledger；N1/N4 accepted-start-to-done；配置切换；32-read 实现；transpose/output/control；exact modulo/round/threshold miter；matched area。只做 one-lane RTL 前仍可先 CPU/static DSE。

**裁决：REVISE / KEEP PRE-RTL ONLY。**它是三个候选中唯一对象差足够且未被现有实现覆盖的一条，但当前没有任何 performance admission。

## 6. RS-BN：文字承认了税，硬门没有把税变成总账

M709 的 prose 并非完全忘记第二遍 sparse FC、source rewind 和 weight read；真正的问题是 one-day gate 没有显式、不可绕过地把以下项放入同一式子：第二遍 FC1 与 FC2、第二遍 weight/descriptor/psum、moments 顺序、global barrier、为 rewind 延长的 source lifetime，以及全部 compute/memory energy。

独立做一个对 candidate 极度有利的上界：

- M480 Q24 fused raw write+read=`2,626,560,000 B`；
- 取 **未准入** 的 M481 最快 full-width projection，FC1=`61,357,094.12` cycle；
- 再不公平地给 FC2 使用 single-K1 `4.7642×` 理想比，FC2=`8,692,749.46` cycle；
- producer 合计仅 `70,049,843.59` cycle；忽略 moments、barrier、source retention、第二遍额外控制与能量，已是乐观上界。

串行模型 `speedup <= (Cproducer + Craw)/(2*Cproducer)` 得：

| BW (B/cycle) | raw write+read cycle | 乐观 speedup 上界 |
|---:|---:|---:|
| 32 | 82,080,000 | **1.085869×** |
| 64 | 41,040,000 | **0.792934×** |
| 128 | 20,520,000 | **0.646467×** |

即便在最慢 32 B/cycle，也过不了 M709 的 `1.15×`。再收费只会更差。

峰值存储也未闭：exact-binary FC1 bit-packed source payload 已是 `103,680,000 B`，Q24 raw peak=`221,184,000 B`。若 rewind 必须延长这批 source 的 lifetime，则在 descriptor 之前最多只有 `2.133333×`，不是 `8×`。若作者主张 source 本来就常驻且可免费重放，必须证明 baseline/candidate 相同 lifetime、地址、端口和 overwrite schedule，不能把既有 storage 当无限期免费。

**裁决：KILL AS CYCLE CANDIDATE。**只在 full address/lifetime ledger 与总能量模型表明 memory-energy 确有收益时，降级为 C2 memory-support；不做新 RTL。

## 7. P0 / P1 / P2

### P0（3）

1. **PIDP denominator 不存在。**M709 允许对 M596 已判 source-static FAIL 的 A1/PBR4 架构做 performance gate，可能用 false-pass baseline 产生 ratio。
2. **TDA gate 可 false-admit。**`issue<=10` 与 active table+acc `<=24 KiB` 没有冻结 45-table resident/load、完整 N1/N4 service 和 32-read 物理端口，能把隐藏资源换成表面 throughput。
3. **RS-BN gate 违反自身 lower bound。**cycle 条件在乐观 producer 上界下已不可能过 `1.15×`；energy/peak OR 分支未收费 source lifetime、第二遍 FC/weight/moments/barrier，可错误放行。

### P1（2）

1. M709 的 K1/K8 `4.89–6.32×` 不可由引用的 M519 rows 复算；正确 active range=`5.078431–6.487654×`。
2. “PIDP/TDA 任一形成第三点即可到约 4.0/5”的 forecast 过早：PIDP novelty 与 denominator 未闭，TDA 无 RTL/PPA，C1/C2 自身的 macro/matched physical closure 也仍是硬缺口。

### P2（2）

1. M709 `source_sha256` 使用短标签而非完整 path，且多项绑定的是 Markdown review，不是 machine receipt/raw row；不影响本次人工定位，但降低脚本复现性。
2. “ep35 的 45 个矩阵全部满秩”在 M709 source map 中没有指向包含 45-rank ledger 的明确 path；本评审仅接受其用于禁止 rank-3，不把它提升为新性能证据。

## 8. DATE 模拟评分

| 维度 | /5 | 独立判断 |
|---|---:|---|
| Novelty | 3.1 | C1/C2 有对象/协议差；PIDP 与既有机制重叠较大；TDA 只是候选。 |
| Soundness | 3.7 | 已有 C1/VCS/负结果纪律强，但 M709 三个候选门均需修。 |
| Significance | 3.1 | C1 1.75×局部可信；C2 等服务优势尚无物理数字；decoder/ATLIF 无新 admitted ratio。 |
| Implementation | 3.0 | 有多个 functional slice；C1 macro integration、C2 matched PPA、TDA/PIDP 均未闭。 |
| Evaluation | 2.8 | 单序列 C1、directed C2/C3；统一 S3/full-network direct table 未形成。 |
| Reproducibility | 4.2 | seal/contract 很强；M709 source path 与数值范围有小缺口。 |

综合：**3.15/5，Borderline Reject / Weak-Accept boundary，当前约 30–40% accept**。这低于 M709 的 3.4，因为它把两个需重写 gate 的候选和一个已可先验杀掉的候选计入了 readiness。若只闭合 M528 真实宏/RTL/PTPX 与 M519 matched K8-vs-K1×8 PPA，并形成统一 direct table，可到约 `3.6–3.8/5`；不需要依赖三个新候选全部成功。

## 9. 推荐执行顺序

1. **P0：**修论文/表中的 C2 ratio source；C1/C2 物理闭环继续优先。
2. **P0：**RS-BN cycle line 关闭，不开 RTL；energy future study 先写 source-lifetime ledger。
3. **P1：**TDA 重写 full-service/static resource gate；过门才做 one-lane RTL。
4. **P1：**PIDP 改成最小 scatter-vs-pull A/B，不等待也不复活 PBR4 全局 scheduler。
5. **P2：**统一表只收 admitted 数；M523、M705、M518 Fixed 作为网络完整性/support。

本评审没有运行任何新思流程，也没有授权后续 EDA/GPU/训练。
