# Motion 多分辨率 Bundle Held-out 负结果与架构否决

## 1. 本轮结论

本轮按照 docs/271 的独立 DATE 评审要求，只做 B4/B8/B16/B32 全量周期模型
筛选，没有修改 RTL。结果明确否决“多分辨率 metadata + 12-block 静态粒度”
作为 DATE 架构贡献：

- `[模型-heldout]` sample 0--49 用于冻结选择，sample 50--99 只作 held-out；
- `[模型-heldout]` calibration 的最佳全局、4 个 stage 和 12 个 block 全部选择
  B32；
- `[模型-heldout]` held-out 的 per-stage、12-block 和候选集内逐 row oracle
  与固定 B32 完全相同，增量均为 0；
- `[模型-heldout]` 固定 B32 相对固定 B8 的 held-out 含 preload 周期只减少
  2.6118%，row p95/p99 均无改善；
- `[模型-heldout]` 672000 条 row 上，B32 相对 B8 为 535711 条更快、
  136289 条相同、0 条更慢；
- `[模型-heldout]` B4→B8→B16→B32 的粗粒度单调性违例为 0；
- `[待验证]` 上述 2.6118% 尚未计 B32 的 32-input priority selector、256-bit
  padded metadata、mask mux 和控制扇出，不能直接当成可实现收益；
- 架构门槛状态为 `REJECT_AS_PARAMETER_DSE`，不进入 canonical RTL、OpenROAD
  或论文贡献列表。

## 2. 预划分合同

本轮在执行前冻结：

```text
calibration = sample 0..49
held-out    = sample 50..99
candidate   = {B4, B8, B16, B32}
```

calibration 只用于选择：

1. 最佳全局固定 B；
2. 每 stage 静态 B；
3. 每 block 静态 B。

held-out 同时评估四个固定 B、全局静态、stage 静态、block 静态和冻结
`{B4,B8,B16,B32}` 候选集内逐 row oracle。
这里的 held-out 只表示没有参与本轮 bundle 选择；此前 docs/270 已经统计过同一
profile 的总体分布，因此不能写成完全未观察的独立数据集。

## 3. 周期模型合同

四种候选只改变 active bitmap 的分组：

```text
front(B) = depth-1 ordered bundle scanner(active mask, B)
row(B)   = front(B) + common SCS/gated-K backend + 225-cycle preload
```

共同保持：

- exact three-class zero-K quotient injection；
- active pair、score、class histogram、Shiftmax 和 gated-K；
- 同一个 depth-1 descriptor 相序；
- 同一 225 拍 Q/K preload。

B8 在全部 672000 行上与 docs/270 已校准模型残差为 `0..0`。因此本轮只比较
bundle 前端，不重复归因 zero-K gating 收益。

## 4. Calibration 冻结结果

| B | calibration 含 preload 周期 |
|---:|---:|
| 4 | 171880944 |
| 8 | 165181386 |
| 16 | 162210579 |
| 32 | 160936701 |

全局选择 B32。每个 stage 与每个 block 也全部选择 B32：

```text
stage histogram = {B32: 4}
block histogram = {B32: 12}
```

这已经表明 workload 没有产生可供静态粒度描述符利用的 block 异质性。

## 5. Held-out 结果

| 策略 | 含 preload 周期 | 相对 RQTB2S 加速 | row p95 | row p99 |
|---|---:|---:|---:|---:|
| fixed B4 | 171357169 | 1.263498x | 1268 | 1479 |
| fixed B8 | 164632825 | 1.315105x | 1268 | 1479 |
| fixed B16 | 161633823 | 1.339506x | 1268 | 1479 |
| fixed B32 | 160332966 | 1.350374x | 1268 | 1479 |
| global static | 160332966 | 1.350374x | 1268 | 1479 |
| stage static | 160332966 | 1.350374x | 1268 | 1479 |
| block static | 160332966 | 1.350374x | 1268 | 1479 |
| candidate-set row oracle | 160332966 | 1.350374x | 1268 | 1479 |

关键增量为：

| 对比 | 总周期减少 | sample p95 | sample p99 | row p95/p99 |
|---|---:|---:|---:|---:|
| B32 vs B8 | 2.6118% | 2.4999% | 2.4797% | 0/0 |
| stage vs global B32 | 0 | 0 | 0 | 0/0 |
| block vs global B32 | 0 | 0 | 0 | 0/0 |
| oracle vs global B32 | 0 | 0 | 0 | 0/0 |

## 6. 为什么粗粒度在该模型中单调不劣

当前 depth-1 scanner 对一个非空 bundle 仍逐 active pair 发射，因此改变 B 不会
减少或增加 active pair 工作；粗粒度只会合并 header。全活动时，所有粒度均为
225 个 active pair 加一次最终交接，共 226 拍；存在空洞时，较粗分组可能减少
header，而不会引入额外 active pair。

这解释了 672000 行上的经验单调性：

```text
front(B32) <= front(B16) <= front(B8) <= front(B4)
```

该结论针对当前单槽、有序、mask 内逐 pair 发射的模型，不应外推到多槽、banked
selector 或每拍多 pair 发射的其他微架构。

## 7. 结构代价与否决理由

| B | group 数 | padded metadata bit | flat selector 输入 |
|---:|---:|---:|---:|
| 4 | 57 | 228 | 4 |
| 8 | 29 | 232 | 8 |
| 16 | 15 | 240 | 16 |
| 32 | 8 | 256 | 32 |

若只把现有参数从 8 改成 32，得到的是：

- 模型周期减少 2.6118%；
- metadata padding 相对 B8 增加 24 bit；
- priority selector 从 8 input 扩到 32 input；
- 没有 stage/block 自适应收益；
- row p95/p99 不改善。

因此即使后续物理结果为正，它也只能作为工程参数优化，不能单独列为 DATE
贡献。按预注册的 5% 周期门槛，本轮在物理评估前已经失败；继续写多分辨率 RTL
只会增加 mux、mode、验证和论文解释负担。

## 8. 对论文和代码的处理

论文允许把本轮作为负结果：

> We evaluated stage- and block-static multi-resolution TTB issue on a
> pre-split 100-sample trace. All policies collapsed to the same global B32
> choice, leaving no workload-adaptive gain beyond a 2.61% fixed-granularity
> cycle reduction before selector cost.

不得写：

- “提出多分辨率 TTB 架构”；
- “12-block descriptor 自适应提升性能”；
- “B32 已经通过 RTL/PPA”；
- “2.6118% 是 ASIC 吞吐或能量收益”。

现有 B8 RTL 保持不变。B32 参数点不进入主回归，除非未来数据流改成多 pair/拍或
metadata 物理读取合同发生变化，使当前单调模型不再适用。

## 9. 复现入口

```bash
sim_h67/run_h67_multires_bundle_heldout.sh
```

产物：

```text
results/h67_multires_bundle_heldout_20260809/report.{md,json}
```

脚本执行 9 项单元测试、100 sample/1200 record 全量解码、B8 零残差校准、
calibration 冻结、held-out 消融和单调支配审计。

## 10. 双线下一步

Motion 本轮否决一个新机制，不代表停止。下一候选必须改变当前“一个 active
pair/拍”的核心瓶颈或扩大系统调度边界，不能继续枚举 bundle 参数。

Local5 的 profile watcher 仍在等待 GPU。等待期间更高价值的本机工作是审计并
补齐其通用 T450 顶层之外的 12-block 时间复用调度与外部 SRAM latency/反压
合同；最终 checkpoint 到达后再绑定真实 theta-folded payload 和多窗口 trace。

## 11. 独立 DATE 复审与整改

独立子代理只读复算脚本、JSON 和文档，未发现改变负结果的算术错误。整条 Motion
线评分为：

| 维度 | 分数 |
|---|---:|
| 总推荐 | 3.3/5，Weak Reject / Major Revision |
| 新颖性 | 3.0/5 |
| 架构完整度 | 2.9/5 |
| 实现可信度 | 3.9/5 |
| 实验完整度 | 3.4/5 |

复审确认“不做多分辨率 RTL、不把固定 B32 纳入主回归”是正确决策。它还用同一
模型只读检查 B64/B128/B225，相对 B8 的理想周期减少为 2.9992%/3.1670%/
3.2421%；即使不计 selector 代价，单槽粗粒度扫描的渐近空间仍低于 5%。这些
扩展点没有进入本轮冻结候选和主结果，只用于判断不值得开启物理研究轮次。

复审指出并已整改：

1. DSE 现在重新 fail-close 检查 1200 record 的 active/K/motion 三类 ordered
   TTB trace，共 3600 项；
2. 所有“绝对 oracle”措辞改为“冻结 `{B4,B8,B16,B32}` 候选集合内逐 row
   oracle”；
3. 明确 B8 `0..0` 只是对已校准公式的自洽检查，独立可信度仍来自此前 138 条
   real-bit 和 5570 条 canonical RTL；
4. 单元测试从 9 项增到 13 项，新增随机中间 mask 单调性、calibration policy
   freeze、held-out 不泄漏、三种 gate 分支和 trace-exact fail-closed。

复审选择下一轮唯一优先级为 Local5 系统完整度，而不是继续枚举 Motion
scanner。首版不依赖最终 checkpoint 的接口范围冻结为：

- 单实例、单 in-flight job，按 12-block 几何发出 6720 个
  `{stage,block,head,window}` T450 job；
- Q/K SRAM request/response 携带 `{job_tag,plane,y,x}`，支持 1/2/4/随机有界
  latency、双向 backpressure 和 in-order fail-closed tag；
- weight context 显式 load/commit/release，绑定
  `{stage,block,head,out_tile}`，不能继续依赖 reset-only `weights_loaded_q`；
- Acc32 result ready/valid 接外部 SRAM，全部结果接受且 tile 内部 drain 后才允许
  `job_done`；
- 先用通用确定性权重和每 block 至少一条非零 T450 oracle 闭合协议，最终
  checkpoint 到达后只替换 Q/K、mask、folded weight 与 SHA。

Motion 保持现有回归和 CPU 上界筛选；Local5 的系统优先级不等于停止 Motion 或
冻结其后续新机制。
