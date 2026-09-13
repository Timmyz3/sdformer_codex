# Consumer enumeration 独立审阅

2026-09-13。审阅者为数据/质量分支；仅在 `data_and_quality/` 写入本审阅及独立计数核对脚本。没有修改被审 SV/TB，没有新GPU、训练或重跑RTL。

**发现并促成修复一处实际工作守恒缺陷；修正版功能、事务与周期独立核对通过。** 29个fixture、232条命令、890880个输出通过，全部实测周期差与独立状态差公式一致。初版中断结果不作为性能结论。

## 发现与修复

初版 [native_sparse.sv](../consumer_enumeration/native_sparse.sv) 的 `DEST_PICK` 设置 `og/p` 并清选中bit后，无条件进入 `READ_WA`。这绕开了原 `NATIVE_DEST → launch_weights()` 的 `need_a` 判定。若 `source_a_hold==0 && source_b_hold!=0`，mode1只读WB，初版mode3多取WA；整数输出仍可能完全正确，因为 `01` 消费者只用WB，但权重词数与背压工作已经不同。

已直接向作者指出位置和具体触发条件。作者将该状态改为共用 `launch_weights()`。此修正不需要增加状态：mode1/mode3的 `need_a/need_b/need_sum` 来自已稳定的两个 source hold，与同拍更新的 `og/p` 无关；权重地址直到后续 `READ_WA/READ_WB` 才使用，届时非阻塞赋值已生效。修后仍清零不需要的wa/wb，保留普通控制的完整取词权限。

## 逐项静态结论

| 关注点 | 审阅结论 |
|---|---|
| `mode_q==1` 改 `mode_q[0]` | mode寄存器为2bit，低位真恰好覆盖1和3；source地址、weight地址、hold读取、入口、SKIP_BLOCK后继均正确共享。mode0/2专用分支保留；mode3在ADVANCE先处理bitmap分支，不会落入旧扫描循环。 |
| C4掩码 | cp是0…47的两通道组；`cp/2`恰好对应C4。A/B两通道共享同一个结构组，与原native mask一致。 |
| 合法消费者 | `dest_new[g*4+q]` 为live O组与3×3感受野几何的交集。其条件与旧`native_dep`相同；对合法bit，计算出的ky/kx均为0…2。 |
| 选择顺序 | 从47递减到0、命中覆盖索引的组合循环实际选择最低置位bit；等价于旧og升序、p升序中保留合法项。时间pending同样按最低位枚举。 |
| 索引上界 | source最大1535；weight最大10367；psum最大479；模式3无新更宽访问范围。图像origin为signed16，以signed整型加本地位置，边界门在源读取前生效。 |
| pending清除 | DEST_PICK每次清一个destination bit，随后完成该消费者的所有时间项；ADD_WRITE清一个time bit。进入下一destination前time pending已空；进入下一source前destination pending已空。合法完整命令在重新开始前两者均空。 |
| 重新开始 | IDLE不显式清两个pending，但RESET清0，INIT重设循环变量；新的非零source CHECK在读destination之前重写dest_pending，零source从不读取旧bitmap。每消费者取词结束前会写time pending。对合法完整命令无需依赖额外复位。中途破坏状态/异步取消不在既有协议内。 |
| 背压 | 等待权重时state、source holds、og/p均稳定，已清的destination由当前正在执行的状态持有；不会重复取出或丢失。输出沿用原DRAIN_READ/SEND，ready为0时不推进row，data/addr保持。 |
| 数据算术与存储 | diff没有改变source/W/psum容量和端口，也没有改变八条`lhs+rhs`数据链；WA+WB仍用相同链并保存signed17。新增的是48bit待处理寄存器和有限组合选择/地址逻辑。 |

## 测试与范围核对

新 [tb.cpp](../consumer_enumeration/tb.cpp) 与原 [native TB](../native_sparse/tb.cpp) 逐文件diff无变化。TB仅装入原始10bit source、真实W、结构mask、origin与独立整数gold，未计算动态destination集合。每条命令检查全部480个输出beats/3840个值、输出顺序与背压稳定性，连续执行两次且不reset。这里两次是**相同mode、相同配置**的重新开始；没有动态切mode的实测用例，静态状态路径未显示跨合法完成命令切mode会读旧pending。

[run.py](../consumer_enumeration/run.py) 覆盖原native的全部控制和三套真实掩码8tile，mode1/3、两种背压、两个命令。它比较全部source/W/psum/sum/update/output工作计数，并要求新增SV里的mode1周期与原native逐用例相同。`masked_contexts`不作为相等事务要求是正确的，因为候选正好取消了这类控制扫描。源/权重/output stall随调度相位改变，可以不同；工作计数仍必须相同。

配置和计费沿用原接口：core从CLEAR开始至FINISH，包含初始化和末输出；每个新tile另加1536个source及1个origin装入拍，静态W/mask首次再加10368+288拍。无新的destination配置表，集合在SV的CHECK拍形成。第二次命令用于状态测试，不能把它当成零装入的新输入性能。

## 独立状态差公式

对每个实际图内、源对T10并集非空、至少有一个live O组的 `(cp,native_xy)`，令G为live O组数，P为几何有效空间消费者数。

- 旧mode1：live组有4次NATIVE_DEST和4次ADVANCE；dead组只用1次NATIVE_DEST跳过。合计 `8G+(12-G)=12+7G` 拍。
- 修正mode3：每个有效目的有1次DEST_PICK和1次ADVANCE，共 `2GP` 拍。
- 因两边有效消费者、取词、time枚举、psum更新完全相同，正常点核心周期减少应精确为各源词的 `Σ(12+7G−2GP)`。

[review_enumeration_counters.py](review_enumeration_counters.py) 从原fixture独立读source/mask/origin生成该差，核对作者已经实跑的周期及扣除实际stall后的差；不是用公式替代Verilator。初版无条件READ_WA的额外请求，也可独立计为每个A=0/B非零源词的G×P，避免仅依赖最终数值检查漏掉公平性错误。

## 创新与物理边界

bitmap、priority encoder、静态支持交集与普通稀疏消费者迭代都属于借入A。即使本次减少周期，也只能成为后续训练/结构目标的更强执行底座，不能单凭加速就升级为新标题机制。

新增48way priority及有限live/几何组合逻辑由mode1/3在同一SV共同保留；没有候选专享数据加法器或存储端口，但**同一个模块不等于同面积/Fmax证据**。组合控制路径、寄存器译码及布线时延未经综合或STA，本审阅不将cycle改善外推成同频物理时间、PPA或全网收益。

## 修正版实测闭合

最终 [SUMMARY.json](../consumer_enumeration/SUMMARY.json) 与 [results.json](../consumer_enumeration/results.json) 已完成：**29个fixture、232条命令、890880个输出逐值通过**。独立核对全部29个fixture（含零/一、边界毒值与三掩码8tile），正常点周期差以及全部背压/第二命令扣除实际stall后的周期差，均与上述公式精确一致；source/W/psum/sum/update/output工作量全部相同。新增SV中116条mode1命令的周期与旧native结果逐条相同。见 [review_enumeration_counters.json](review_enumeration_counters.json)。

| 掩码 | mode1 core | mode3 core | 精确减少 | 含新source/origin装入的mode1→mode3 |
|---|---:|---:|---:|---:|
| dense | 468,088 | 415,024 | 53,064 | 480,384 → 427,320 |
| physical25 | 378,992 | 334,373 | 44,619 | 391,288 → 346,669 |
| magnitude25 | 366,803 | 323,132 | 43,671 | 379,099 → 335,428 |

三臂总和均为8个校准tile，取每tile第一条命令，不扩大成232个独立输入样本。dense含source/origin的cycle降低约11.05%；core约11.34%。所有数据工作不变，节省精确来自取消非法/死消费者扫描；额外背压收益取决于请求相位，不单独解释为读词减少。

修正版结论：**功能与事务公平性审阅通过，周期账闭合；未发现未修复的具体bug**。这是一份更强普通稀疏iterator控制A，仍不据此声称新标题机制、Fmax/PPA或整网收益。
