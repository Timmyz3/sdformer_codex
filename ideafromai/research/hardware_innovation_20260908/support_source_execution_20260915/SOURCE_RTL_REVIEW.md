# Source producer / classifier 独立审阅

审阅最终 `source_classifier.sv`、`tb_source.cpp`、`export_source_tb.py`、`build_prefix_tables.py`、两份 prefix table 与 **2,680 行**稳定 `source_cycles.csv`，包括最终 static/full next-X 预取；只读 root 实现，未重建或重跑 RTL。另完成 Python 3.12 NumPy-only 源输入再生，见末节。

**结论：当前 source 确实由原始定点 X 做完整时间 MAC，图请求及预取走真实有限 bank 接口，未发现阻断性数值/协议 bug。** 固定静态消除 32 个共同距离位、按用途装参、static64/full96 的普通 next-X 预取均已落实。最强已测 static64+PF 为 83,008/88,256 拍，熵序 exact-code32+PF 为 71,509/87,055：ready 省 **13.8529%**，BP 仅省 **1.3608%**。响应 class 的实际增量须与同序、同格式、同预取的 exact-code 图相比较。

## 1. 原始 X、MAC 与共同 T10 义务

- DUT 输入 SRAM 每 channel 两个 128bit word，每 word 装 5×signed24 X 与 8bit padding。它没有接受预算 gate、code ID、节点分支或 MAC 答案。`xscalar` 解包源时间 s，十路 signed16×signed24→signed40 乘法，每拍同时计算十个目标时间，按 s=0..9 发出 **100 个 scalar MAC/channel**；两拍 product 流水后累加到十个 signed48 U。
- CLEAR→10 拍 MAC→DRAIN→DECIDE 的顺序正确，U 必须完成最终写回才判 `U>=τ`；没有中间 RNE。TB 对每个实际生产 channel 的全部十个 U 和 gate 逐值核对，不只是最终 class 碰巧相同。X 极值诊断也在 signed24 内；固定 A 对任意此类 X 的 prefix abs 界为 74,524,393,472，小于 2^47。
- 图的执行单位是一个 channel 的**完整 T10 门字**。每次从十个未终止消费者中选当前变量排名最小的 channel，真实生产一次，再推进所有恰好等待该变量的节点；其他消费者保留。固定有序图保证后继变量排名更晚，所以不会在以后重新需要已跳过/已生产的 channel。TB 禁止重复 producer channel；独立路径重建也核过没有重发。
- 这实现的是 `A16Q12/X24Q16/τQ28` 新定点函数。它对原 FP32 forced student 的差异仍为完整训练缓存 raw gate 1,093、projected gate 887；不能借原 AEE 或把整数逐值通过写成原 FP32 无损。

## 2. 强静态对照、排序与 class 类型

mode0 完整 96 channel 只是参考；mode1 删除 D 中对所有 code 都相同的列，实际信息列数 `[9,13,15,8,11,8]`，合计 **64**。当前 D 的共同列均为零，直接用 OR mask 正确；这些位对 Hamming 距离只加共同项，最小编号平局规则保持。不能将从 96 降到 64 归给动态图。

自然序按物理 channel；熵序只用 D 每列 16 个模式的 one-count，按 `abs(count-8)` 升序、物理编号打破平局。没有利用本轮 X/gate 活动率排序或逐 case 选序。RTL 实读 `order_rank[96]` 的 48B/3word 元数据，十消费者按此共同顺序选择；不是把十个时间面独立提前结束后的收益相加。

mode2 输出原 Hamming argmin code；mode3 输出 **whole-H384 response 的 canonical code**。后者可能不是原投影 code，也不承诺原 projected gate 身份；它仅在构图使用的 `../support_lut_execution_20260915/response_class_W.npy` 下具有相同完整 FC1 响应。该 W 是已标注的修改 W 探针，不能默认为原 forced W，亦没有新 AEE。TB 的 mode3 gold 明确先求 argmin 再映射 canonical，类型边界正确。接下游时必须绑定同一 D/W 响应合同；本叶没有实际接 FC1 或完整网络去验证该绑定。

决策图生成器对每组全部 65,536 输入核过 code/class、静态消除和平局；这是函数证明与合成覆盖，不是验证数据分布。自然 class/code 在当前真实 64 包都请求 3,786 channel，语义区分降低没有带来少生产；熵序为 code 3,359、class 3,353，**class 只额外取消 6 次完整 channel 生产**。

## 3. 图读取、格式与预取协议

- 内存为 8 bank×128bit、低三位选 bank、每 bank 一笔在途。需求 miss 只在握手后 pending，返回才填 cache/节点；同 word 对十个节点的广播利用该真实返回。`graph_hits` 计节点消费者命中次数，不能当作物理 word 数。cache 为 8 个 128bit word，共 **128B**，每 bank 一个直接映射条目，跨组保留、每命令清 valid。
- 64bit 格式每 word 两节点；32bit 格式每 word 四节点，lo/hi 各 12bit、变量 4bit，其余 padding。当前各图节点数最多 2,179，所有指针可放 12bit，32→36bit 解码零扩展正确；终端由 node_id<16 判定，因此不需要读取被省去的终端标志/label 字段。熵图虽只需 10bit 指针，RTL 仍用共同 12bit 格式，未假设进一步压缩收益。
- 预取只读取当前实际节点的两个 child record，不读取将来的 gate/分支答案。它在 MAC/DRAIN 空闲内存阶段通过相同 req/pending 接口请求，未选分支也真实计费；20bit `pf_done` 防止本次 MAC 反复预取相同义务，新增控制状态并非零面积。响应进入同一个 128B cache，可能冲突驱逐或被无用分支占用。
- 最终 static/full 也在当前 MAC 期间预取组内下一个已知 channel 的两个 X word，复用**同一个** 128B cache；当前 X 仍在独立 32B holding 中，不被下一 X 覆盖。XFETCH 对 cache hit 复制到 holding，对迟到返回按 `pending_addr/2==当前全局channel` 识别；未加新数据数组，未将 graph 模式改成同时额外缓存 X。物理 source 请求现按 `addr<192` 计数，包含预取；cache→holding 复制不再算一次外部请求。
- **DRAIN 的 `pv==0 && !(|req_valid)` 修正已在最终源码。** 当一个预取请求遭拒绝时，不能因乘法结束撤回它。已接受、尚在途的预取可越过 DECIDE，但 LOOK 必须在 `all_ready && pending==0` 后进入下一 XFETCH/OUT；因此图返回不会误写 X holding，也不会带未退休访问结束命令。
- MAC/DRAIN 内 node/child/relevance 稳定；同 bank 若有 pending 就不发第二笔，地址不因其他 bank 返回改变。TB 对拒绝期间请求 valid/address 保持、bank 单在途及输出 group/code 背压保持均有实际检查。producer 输出是内部观察事件，没有外部 ready；不能把它当作已经接入一个可背压下游的独立接口。

## 4. 参数与资源费用

最初所有模式都装 35word 并非逐模式必要配置；现已闭环修正：

| 模式 | 实际启动参数字 | 读取内容 |
|---|---:|---|
| full96 | 29 | A 13 + τ 4 + D 12 |
| static64 | 30 | 上述 + informative mask 1 |
| code/class graph | 21 | A 13 + τ 4 + 本类型 roots 1 + rank 3 |

每命令仍冷装这些参数，配置驻留/跨 P 批处理未实现。源 192 个 word（3,072B，含 padding）与图/参数共享同一接口；预取没有增加外部端口。片内镜像首次装填、构图、X 的定点编码、上游浮点 source 输入生产均在当前计时边界之前。

RTL 除外部 SRAM 镜像外，持有 A 200B、τ 60B、D 192B、info mask 12B、root 容量 24B、rank 48B、图 cache 数据 128B 加 tag/valid、pending 地址、十份 node/lo/hi/var、raw word 20B、X holding 32B、U 60B、两级 product 100B，以及队列/控制。算术是 **10 个 16×24 multiplier、10 个 48bit accumulator adder**。这些资源不是上一片 96 路 FC1/PSN 电路的等面积实现，没有 Fmax/功耗/PPA 证明。64→32 格式保留同一 cache 容量和接口宽度，可以比较请求与周期，不能把字节比率直接当周期比率。

## 5. 最终收据与独立检查

最终 67 cases（64 个真实训练位置、3 个定向 X）×2 order×各模式适配×2 BP，合计 **2,680 tasks 全 PASS**。每任务 6×10 个分类输出，共 **160,800 labels**；按实际生产数计，RTL 观察到 **1,750,820 个 U 和对应 gate**。没有把未生产通道的 CPU oracle 计成 RTL 已检查值。每 task 是 P1×T10×C96，64 个真实 task 源自首两训练帧各 32 个抽样位置，不是完整 P32 联合调度、留出验证或整层。

独立 Python 3.12 检查：从 `source.bin` 的原 X/A/τ 自己算整数 MAC、完整 Hamming，再按十消费者共同路径走图，先核对完整 2,144 条图/原供数记录及 16,080 个逻辑最终 label；随后核新增普通预取保持 mode0=96、mode1=64 个 channel，并对最终全部 **2,680 行**重做状态/费用检查。图的逻辑 channel 数与 pack/prefetch/BP 无关。全部最终 CSV 满足：

`scalar_mac=100*channels`；`xwords=2*channels`；`words=boot_words(mode)+xwords+graph_words`。

普通预取每组首 channel 仍需求读，故 `prefetch_words=2*(channels-6)`；所有普通 PF 记录均满足。图 PF 的该字段是图预取，普通 PF 则是 X 预取，不能统一误认成 graph 请求。

`cycles=sum(state0..state13)-1`：TB 把 DONE 可见状态也计入 state histogram，cycles 已含接受 start 的那拍；done_ready 恒零，没有测试 DONE 握手后无 reset 重启。

64 个真实训练位置累计，采用最终按用途配置：

| 臂 | ready cycles | BP cycles | 实际 channel 数 |
|---|---:|---:|---:|
| full96 | 125,504 | 150,464 | 6,144 |
| full96+next-X PF | 119,744 | 124,928 | 6,144 |
| static64 | 86,720 | 103,744 | 4,096 |
| **static64+next-X PF** | **83,008** | **88,256** | 4,096 |
| 自然 code64 | 84,279 | 112,636 | 3,786 |
| 自然 code32 | 83,691 | 109,534 | 3,786 |
| 自然 code32+PF | 80,580 | 98,170 | 3,786 |
| 熵序 code64 | 74,588 | 99,672 | 3,359 |
| 熵序 code32 | 73,956 | 97,498 | 3,359 |
| 熵序 code32+PF | **71,509** | **87,055** | 3,359 |
| 熵序 class32+PF | 71,385 | 86,537 | 3,353 |

相对最终 static64+PF，自然 code32+PF 在 ready 小正 2.9250%，BP **反慢 11.2332%**；不能用旧弱供数分母隐藏该负例，也不能继续用旧未经按需配置的 86,071/87,360 等数字。熵序 code32 开预取，ready 图请求 **5,268→7,574**（其中 6,934 为预取），周期却 73,956→71,509：这是多取数据换取与 MAC 重叠，不能写成图流量减少。class 对同条件 code 只再省 ready **124 拍**、BP **518 拍**；相对最终 static64+PF 的净省为 14.0023%/1.9477%，主要部分不是响应等价类的独立贡献。

## 6. 普通供数强控与 Python 3.12 闭环

本审阅指出的单一普通缺项 **next-X 预取已实际完成**，不能再列为待办。它保持 static64 的 8,192 个源 word 请求不变，其中 7,424 个在 MAC/DRAIN 阶段实际提前读取；ready/BP 服务从 86,720/103,744 降到 83,008/88,256，显著收紧 BP 分母。这是成熟供数适配，不需训练或换函数。当前可以保留条件性的 exact-code 部分生产改进与 class 小增量，不宣称所有可能流水、跨组预取、配置驻留和 P32 调度均已最优。

Python 环境违规已明确纠正：历史 GPU replay 实用了 Python 3.10，旧脚本/日志留作事实记录；`prepare_sources.py` 现要求 **Python 3.12**，走 `prepare_sources_numpy312.py`，从 tar BytesIO 经已有受限 `read_torch` 读取 X/参数，全程未导入 Torch。默认直接生成 root 需要的 `source_cases.npz`；历史 NPZ/统计仅是显式 `--compare-to` / `--compare-statistics` 可选项。已在空输出目录 `regeneration_check/` 无历史比较参数实际再生，全部 26 个字段随后与活动输入逐值同；全 32 帧机会/差异统计亦已复现。旧活动输入未被隔离检查覆盖。历史完整 FP32 raw 数组未保存，不把逐帧统计一致夸大成旧完整数组逐值比较；也没有再运行 3.10。
