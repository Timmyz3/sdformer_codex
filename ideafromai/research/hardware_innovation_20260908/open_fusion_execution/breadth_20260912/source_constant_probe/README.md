# 固定两项有符号幂和：源RTL、局部链与新十帧完成

**两套新函数均实际降低源周期；普通dense也同样获益，不能作为lifting独占X。** 无训练、无项数/位宽/指数扫描，只将新320步dense的As和lifting的q12分别投影到同一固定“至多两项signed power-of-two整数”集合。先写的[A/B/X和规则](PLAN.md)、[实际运行](run.py)、[完整汇总](summary.json)保留。

| 同结构、同真实I24 | 原dense | 两项dense | 原lifting | 两项lifting |
|---|---:|---:|---:|---:|
| ready周期/H8 | 491 | **319** | 446 | **290** |
| ready减少 | — | 35.0305% | — | 34.9776% |
| 加减节点 | 252 | 122 | 165 | 80 |
| 程序字 | 275 | 144 | 224 | 138 |
| 工作RF峰值（另有1个门RF） | 61 | 40 | 13 | 10 |
| 实际norm24 | 0 | 0 | 35 | 35 |
| 改变系数 | — | 98/100 | — | 39/40 |

固定压力下，合并两窗口的周期分别少29.4152%和32.2525%；它是原ready/stress服务合同，不是实际芯片或整网速度。四列都使用同da4ml0.6.0完整CSE、last_use_pressure、同公共RTL二级流水和96×8×48 RF/512×128 ROM，不缩资源。原普通34项源仍为303周期，应保留作结构强对照；不同结构的质量和函数不同，不能只比较源周期就宣布最终胜出。RF峰值也不是物理面积或状态下界。

每系数在424个合法signed16候选值中选最近整数，tie选较小绝对值；投影不是按验证质量择优。dense原指数15、lifting原指数12不变。dense最大系数整数偏差959，lifting为644；量化文件逐项记录原值、新值、差和所有保持字段。配对、排列、源/消费者cutoff、下游矩阵、BN2常量、PED bias均逐键核对未改。字面lifting仍执行40次原RNE/sat24，公共编译保留35个实际norm，最后5个纯门出口仍只做相同精确阈值折叠。

| 两真实halo，193,920门位/函数 | dense | lifting |
|---|---:|---:|
| corner改变门位/77,760 | 178 | 314 |
| interior改变门位/116,160 | 313 | 693 |
| 总改变门位 | 491（0.253197%） | 1,007（0.519286%） |

这是相对**各自未量化的新320步父函数**的门差；不是RTL错误，也不意味着任务质量已过门。输入仍是已入库ordinary上游I24两halo，77,760/116,160值，与原source hook一致；CPU先重算父门并逐位匹配原新参数gold，再生成新函数gold。每臂596组边界/随机向量通过原编译器的数值、RF逻辑tag和两槽readiness检查。公共RTL两臂×两窗×ready/stress共8例，**775,680个门位和1,342,896次向量RF写回检查0差**。这些重复压力模式不新增独立输入样本。

两种函数的SR64与SW64访问量均与原同结构相同，每次两窗共72,720个SR64字和4,848个SW64字；减少的是常量加减执行及相应调度费用。源边界不含外部冷DMA、preview/双消费者、native或全域BN；不把这张表与任何后段倍率相乘。

交给算法代理的两个完整NPZ可直接装入现有`LiteralForward`：

| constants | structure | CPUgold |
|---|---|---|
| [dense/deployed_constants.npz](dense/deployed_constants.npz) | `dense` | [dense/cpu_gold.npz](dense/cpu_gold.npz) |
| [lifting40/deployed_constants.npz](lifting40/deployed_constants.npz) | `lifting40` | [lifting40/cpu_gold.npz](lifting40/cpu_gold.npz) |

CPUgold字段为`corner/interior_I24`、`corner/interior_source_gate`及`corner/interior_parent_source_gate`，布局均T,C,H,W；输入来源、逐窗差和位数见各目录`cpu_checks.json`。算法代理已接收，排在三原新学生825及主halo之后，先核真实source hook两halo，再最多两臂diverse10。**本目录CPU/RTL结果没有新AEE，不能继承父10帧或825。** 最终网络质量以另行实际评估为准。

两项量化、PoT与CSE均是公共底座；本次只补一个此前未实际执行的固定常量接口，作为性能使能/普通强对照保留，没有新ISA、训练、PPA或新标题。复跑使用`hardware_innovation_20260908/psn/cmvm_20260909/.venv/bin/python`执行本目录`run.py`，随后用普通Python执行`summarize.py`；公共RTL二进制及源在上一阶段`hardware/rtl_source/`，没有修改它们。

另补了必要的[同函数低RF普通控制](dense_low_state/README.md)：不改既有逐行两链CSD策略，在新两项dense上得到229字、13工作RF＋门RF，原512字ROM可执行且四RTL例通过；ready为472周期，较同函数完整CSE319慢47.9624%。这是同一个新dense函数的额外编译布局，不是第三个量化/AEE臂；未来交织须保留此合法普通控制，不能拿dense40/61RF当下界。

## 新函数质量与消费者已实际接上

[root完成的两臂十帧](../algorithm/source_constant_aee/README.md)：dense AEE1.165900900，lifting1.152912916，均过NB0；不继承父825。四例完整局部CPU链已执行，约35%源叶降幅在链上成为7.61–9.19%；同量化权限lifting仅再少4.47–5.47%。真实GPU四halo的1,284,480项整数端点与参数匹配，中间FP差异单列。[局部链及核对](../hardware/source_constant_local_chain/README.md)。此更新关闭上文记录的队尾待测状态。
