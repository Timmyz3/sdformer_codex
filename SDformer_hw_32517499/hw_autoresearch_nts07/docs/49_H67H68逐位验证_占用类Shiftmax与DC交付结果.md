# H67/H68逐位验证、占用类Shiftmax与DC交付结果

## 0. 结论

本轮已经把H67和H68从软件候选推进为可执行开放工具回归、可读取到Design Compiler的增量
attention row RTL。当前主线结论如下：

1. H67继续作为硬件主线。RTL逐位模型valid825为AEE `1.462688`、AAE `9.403994`、
   spikes `26.354366G`，相对原dyadic部署AEE只变化`+0.000055`。
2. H68部署态不是第二套矩阵attention。其训练期Castling辅助在eval中为零，部署RTL为无
   Motion-XOR的TTX；逐位模型AEE `1.472654`，相对原部署变化`+0.001167`。
3. 新增的核心硬件机制是占用类SCS-Shiftmax：零K token仍精确进入分母，但按最终定点score
   类聚合；只扫描真实出现的类，不固定扫描H67全部35类。
4. H67的35类扫描采用两拍“找类/读计数 -> exp2乘加”流水；H68只有3类，编译期特化为单拍。
   profile100周期模型显示H67 attention行核周期/帧下降`12.86%`，H68下降`0.37%`。
5. 当前包可以进入有工艺库的DC探索综合，但本机没有`dc_shell`、`.db/.lib`和SRAM宏，尚无
   可发表的WNS、面积、功耗或Formality结果。

因此，当前状态是：**算法和定点数值冻结通过，增量RTL开放回归通过，DC交付结构就绪；正式
工艺PPA和全加速器论文签核未完成。**

## 1. 软件部署图冻结

### 1.1 H67

冻结参数为`T=2`、`head_dim=32`、one-sided binary Q/K、`alpha0=1/64`、Motion权重`1/4`、
score步长`1/128`和gate步长`1/128`。每个token/head的Q7 score为：

```text
overlap = popcount(Q & Kcurrent)
same0   = popcount(~Q & ~Kcurrent)
motion  = popcount(Kcurrent XOR Kpeer)
scoreQ7 = RNE(4*overlap + motion + same0/16)
```

RNE表示最近偶数舍入。Motion项在Q7域直接加`motion`，不是右移两位。H67零K时score仍依赖
`Kpeer`，所以不能按`q_active`粗糙合并，只能按最终score code合并。

### 1.2 H68

H68训练时包含参数为零的binary matrix辅助输出，并在训练过程中把辅助权重退火到零。eval和
部署配置均强制辅助权重为零，因此部署score为：

```text
overlap = popcount(Q & K)
same0   = popcount(~Q & ~K)
scoreQ7 = RNE(4*overlap + same0/16)
```

H68零K可达score只有`0/1/2`三类。`h68_castling_deploy_top`不含矩阵score SRAM、矩阵
Shiftmax或`weights@K`，也不读取peer K。

### 1.3 H60无carrier执行图

H67/H68都使用：

```text
Q/K binary event -> token score -> integer Shiftmax -> gate*Kcurrent
```

软件安装覆盖包含105个ATLIF wrapper，但profile100中两条线都只执行93个。未执行的12个全部是
每个attention block的`sn2_q`原QKFormer carrier神经元；H60分支不调用它。硬件固定部署图应按
93个ATLIF逻辑调用点核算运行周期和状态访问，105只作为软件安装或回退兼容口径。

## 2. 增量RTL数据流

```text
descriptor(stage, block, head, window)
                    |
                    v
H67: Q[31:0] + {K1,K0}[63:0] + time_sel
H68: Q[31:0] + K[31:0]
                    |
                    v
overlap / same-zero / optional Motion-XOR popcount
                    |
                    v
Q7 score + row max
       |                         |
       | Kcurrent=0              | Kcurrent!=0
       v                         v
score histogram             active-entry bank
+ occupied bitmap          {score,K,token}=56 bit
       |                         |
       +----------+--------------+
                  v
          16项Q8 exp2 LUT
                  |
       integer denominator sum
                  |
        ceil-log2二次幂分母
                  |
      Q1.7 RNE gate，饱和到2.0
                  |
        sparse {token,K,gate,threshold}
```

### 2.1 SCS-Shiftmax精确代数

对零K token，输出`gate*K`恒为零，但其score不能从Shiftmax分母删除。RTL使用：

```text
den = sum_active exp2(score_i - row_max)
    + sum_class count[c] * exp2(c - row_max)
```

这不是token pruning，也不是近似。活动K保留score/K/token用于gate回放；零K只保留最终score
的multiplicity。H67合法类为`0..34`，H68为`0..2`。

### 2.2 占用类扫描

旧控制器只要存在折叠token，就固定扫描全部合法类。profile100中H67每行实际占用类均值只有：

| stage | 占用类/行 | 活动项/行 |
|---:|---:|---:|
| 0 | 2.75 | 31.47 |
| 1 | 1.36 | 3.63 |
| 2 | 2.34 | 10.88 |
| 3 | 2.13 | 24.43 |

新RTL在load阶段维护占用位图和剩余类计数，仅弹出非空类。H67把35路优先编码/直方图读和
exp2乘加分成两拍，避免形成单周期长路径；H68只有3路，保持单拍。位图popcount与“剩余类+
在途类”的关系由SVA持续检查。

### 2.3 存储组织

- active-entry bank：`162 x 56 bit`，合并score、K和token，求和/发射共享一个逻辑读口；
- H67 histogram：`35 x 8 bit`，另有35-bit占用位图；
- H68 histogram：`3 x 8 bit`，另有3-bit占用位图；
- histogram明确按小型寄存器bank实现，连续同类token要求单拍read-modify-write；
- active-entry当前是异步读RTL数组，DC可展开为触发器和mux，但正式SRAM宏需要同步读FSM重排。

精确物理深度消除了`MAX_TOKENS=162`被错误填充到256项的容量。存储消融只统计bank数据位，
不含位图、FSM和宏外围。

| 设计 | 深度配置 | bank数据位 | Yosys通用单元 | 触发器 | mux |
|---|---|---:|---:|---:|---:|
| H67 | 精确162/35 | 9,352 | 25,045 | 8,441 | 8,875 |
| H67 | 填充256/64 | 14,848 | 37,132 | 13,308 | 13,973 |
| H68 | 精确162/3 | 9,096 | 22,645 | 8,131 | 8,273 |
| H68 | 填充256/4 | 14,368 | 33,061 | 12,746 | 12,896 |

精确深度使H67/H68通用单元分别下降`32.55%/31.51%`。这些数字仅用于同一Yosys流程内的结构
对照，不是标准单元面积。

## 3. 接口合同

| 接口 | 主要字段 | 语义 |
|---|---|---|
| 帧控制 | `start_frame/busy/done` | 单实例按descriptor串行处理一帧 |
| 行请求 | stage2、block3、head5、window10、tokens8 | 12个block由descriptor时间复用 |
| H67输入 | valid/ready/last、time1、Q32、Kpair64 | 每拍一个token，payload 97 bit |
| H68输入 | valid/ready/last、Q32、K32 | 每拍一个token，payload 64 bit |
| 稀疏输出 | valid/ready/last、token8、K32、gate9、threshold8 | payload 57 bit，按token index散写 |
| 性能状态 | loaded、folded、emitted、classes、exp transactions | 逐行profiling和异常审计 |

`out_gate_q8`是历史端口名，实际格式为9-bit无符号Q1.7：`1.0=128`、`2.0=256`。全折叠行
没有输出beat，也没有`out_last`，只产生`done`；下游必须预清零或使用token index散写。

本轮正式支持的参数合同是`HEAD_DIM=32`、`MAX_TOKENS=162`、H67类深度至少35、H68类深度
至少3、active memory深度不小于162。深度参数只为存储消融暴露，当前RTL没有可综合的参数越界
断言；DC运行不得使用违反上述合同的覆盖值。

当前top不包含projection、ATLIF状态更新、残差、skip SRAM、DMA和片上网络。encoder的跨stage
skip只有S0/S1/S2三条downsample前skip；S3输出是bottleneck/首个decoder输入，不应称为第4条
encoder skip。block内部残差和decoder连接均未被本增量attention RTL删除。

## 4. 验证结果

| 层级 | 检查 | 结果 |
|---|---|---|
| H67 score | 35,937组计数组合+100,000随机32-bit向量 | 0不一致 |
| H68 score | 1,089组计数组合+100,000随机32-bit向量 | 0不一致 |
| gate量化 | 独立Python整数参考+100,000组Icarus向量 | 0不一致 |
| row RTL | H67/H68，8和162 token，fold开/关、反压 | 通过 |
| 极端类 | H67全部35类、H68全部3类、单token、全活动 | 通过 |
| 协议SVA | 输出反压稳定、done/busy、last/valid、类位图不变量 | 通过 |
| lint | H67/H68顶层Verilator | 通过 |
| 结构综合 | Yosys hierarchy、未驱动直接失败、check | 通过 |
| 网表回灌 | 行级通用映射网表使用同一scoreboard | 通过 |
| H68结构 | 无matrix auxiliary，Motion-XOR编译期关闭 | 通过 |
| ATLIF覆盖 | installed105、executed93、uncalled12均为sn2_q | 通过 |
| valid825 | H67/H68完整RTL Shiftmax软件模型 | 通过 |
| 全顶层顺序LEC | Yosys实验流程 | 超时，未关闭 |
| 正式LEC | DC SVF + Formality | 等待工具、工艺库和映射网表 |

valid825详细结果：

| 候选 | RTL AEE | RTL AAE | spikes(G) | firing | spike能耗代理(uJ) |
|---|---:|---:|---:|---:|---:|
| H67 | 1.462688 | 9.403994 | 26.354366 | 0.056954 | 23362.23 |
| H68 | 1.472654 | 9.471391 | 26.416394 | 0.057088 | 23414.83 |

spike能耗仅为活动代理，不含Motion-XOR、histogram、Shiftmax、SRAM、projection、控制和clock tree。

## 5. 周期与稀疏度指导

基于每帧6720行和profile100加权：

| 设计 | 旧固定扫描周期/帧 | 新占用扫描周期/帧 | 变化 | 500MHz行核帧率代理 |
|---|---:|---:|---:|---:|
| H67 | 1,591,065 | 1,386,424 | -12.86% | 360.64 |
| H68 | 1,376,202 | 1,371,097 | -0.37% | 364.67 |

该帧率只覆盖无外部停顿的attention row engine，不是SDformer端到端FPS。SRAM同步读、投影、
ATLIF、三条encoder skip、decoder和数据搬运都未计入。

H67/H68真实T=2 lane更新密度分别为`2.5090%`和`2.5523%`，零更新token/head分别约`74.00%`
和`74.30%`。这支持后续Exact Delta/ETCR研究，但当前RTL尚未实现previous-Q/K状态和Delta route，
不能把`48.7%`理论compare下降写成已实现硬件收益。

## 6. DC交付内容

`dc_handoff/`包含：

- 500MHz探索SDC，setup/hold uncertainty、全部同步输入延迟、输出延迟、transition/load和fanout；
- H67/H68双顶层`compile_ultra`脚本；
- mapped Verilog、DDC、SDC、SVF、QoR/area/power/timing/constraint报告生成；
- DC工件完整性审计；
- Formality交接脚本；
- 开放工具综合、lint、仿真、断言和网表回灌入口。

同步低有效`rst_n`已经纳入普通输入时序，未错误设置false path。开放综合脚本不再用
`setundef -undriven -zero`掩盖未驱动问题。

## 7. 当前GO/NO-GO

### 可以做

- 用当前H67/H68 top进入DC compile-explore；
- 比较同一库、同一SDC下H67与H68的增量逻辑；
- 以触发器行缓冲先获得逻辑关键路径和粗粒度面积；
- 用SVF和映射网表进入Formality。

### 不能做

- 不能把Yosys通用单元换算成um2、MHz或mW；
- 不能把当前attention row top称为全encoder/整网加速器；
- 不能声称SRAM PPA，当前没有macro模型；
- 不能声称LEC关闭，当前全顶层Yosys LEC超时；
- 不能用spike能耗代理替代芯片功耗；
- 不能保证DATE录用，正式PPA、对照和新颖性证据仍需下一阶段完成。

## 8. 证据路径

- `results/h67_h68_rtl_exact_valid825.md`
- `results/gate_quant_reference.md`
- `results/h67_score_reference.md`
- `results/h68_score_reference.md`
- `results/h67_h68_score_class_scan_cycle_model.md`
- `results/h67_h68_storage_ablation.md`
- `results/h67_h68_atlif_module_coverage.md`
- `docs/47_H67H68_DATE深度论文与开源调研.md`
- `docs/48_H67H68功能验证计划.md`
- `dc_handoff/README.md`
