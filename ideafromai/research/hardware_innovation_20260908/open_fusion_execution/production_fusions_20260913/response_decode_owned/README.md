# 响应内W8解码：往返减少，仍未超过expanded16

**C2固定两个条件、六臂已跑完。** 新响应内解码比旧RF84放置少6,144／7,680槽，但同函数expanded16仍更快。全部updated I24、projection gate与完整PED输出共1,025,280次值比较，0差；旧expanded16和RF84两臂均逐服务槽、逐count复现已有收据。按[固定计划](PLAN.md)停止这个放置的加速主张，不追加宽度、缓存或延迟扫描。

| ordinary/interior，完整已有局部边界 | expanded16 | 旧RF84解码 | 响应内解码 | 新放置相对expanded16 |
|---|---:|---:|---:|---:|
| ready | 2,800,716 | 2,817,422 | 2,811,278 | +0.3771% |
| 原固定stress | 3,095,578 | 3,113,466 | 3,105,786 | +0.3298% |

原压力条件仍为每32槽中SR在24–31暂停、SW在28–31暂停，没有重新挑压力轨迹。新放置局部消费者本身为684,804／765,368槽，相对expanded16的674,242／755,160仍多10,562／10,208槽。

使用的是`stage_20260912/weight_compensation/ordinary_lowbit_gpu_parameters.npz`中的既有W8字段，并逐字段核对原GPU部署导出。W8 code为32×96 signed8、范围−127至127；32个signed16行scale为110–183，恢复的U仍为signed16，V及bias不变。这不是上一C1的R24激活量化，也不是当前新320步学生或native_w8；同函数表示比较无需借用其他AEE。

每点真实执行共同I24生产者、preview及sn2，然后将完整Machine状态、时间、pending与仲裁复制给三臂。继续实际完整K864、Conv2/merge、projection门及U32/V96 PED，未导入现成门值、未把旧表相加。native投影、全域BN与最终join仍不在这个边界。

## 新放置具体付了什么

旧路径每H8通过`CR→RF84 signed16→RF读取→16B权重暂存`供给原MAC；新路径在同一个32B CR响应上选择8个code、符号扩展后写入**同一个**16B暂存。每个向量明确付**2槽**：一条共享decode issue和一槽显式等待，从issue起满2槽才可读，不与MAC重叠。两点各3,072次解码，均收费6,144槽；没有把解码假设成零延迟。

物理实现需要32B响应中的4选1×64bit选择、8路signed8→signed16扩展，以及既有暂存写使能和valid控制。2槽是CPU执行模型假设，未做STA/综合，不能据此声称相同面积或PPA已成立。所借打包和解码先验见[MiLo原文§3.3](https://arxiv.org/html/2504.02658v2)与[上轮原始来源核验](../literature_owned/PRIMARY_SOURCES.md)；C2只测试具体响应放置，不申领量化或打包本身的新颖性。

共享资源保持96×8×48 RF、SR64/SW64/CR256、128KiB状态/系数、原8192B源ROM及64B共同staging。source gather预留`[0:24)`，权重暂存为`[24:40)`；单CR256响应没有变成多端口。每次MAC读取实际暂存并核对真实系数体；每点61,440次向量MAC均发生在decode完成之后，H8向量保留到其全部20次P2×T10消费结束才覆盖。source collector写入与decode写入均检查对方区域不变；这是现有模型的分区审计，不是完整gather RTL证明。新路径不写RF84，也不新增RF；行scale、split、源与header的原RF分配继续占用。

## 为什么还慢

ready中，去掉3,072次RF84解码写回和3,072次RF→staging读取后，替换成显式响应解码；同时少6,144个原RF依赖等待槽，净少6,144槽。stress中另少1,536槽端口/写回等待，合计少7,680槽。不能将后者当成固定解码器自身每次都省更多。

新旧packed的SR/SW/CR/CW字节完全相同。相比expanded16，两packed臂每点都只少读23,808B系数、少冷填2,976B；SR为4,103,304B、SW为1,366,344B均未减少。原行scale仍保留**1,280次signed24拆分、1,280次16×24乘积、640次shift/add合并**及原RNE/sat，压缩权重未使这些执行义务消失。ready的U阶段从旧131,200降到125,056槽，仍高于expanded16的113,936槽。

仅在现有时间线上算术扣除全部6,144个新增decode槽，ready仍余4,418槽、stress仍余4,064槽差距。这个数字只是**不重排执行的算术余量**：没有重新计算压力仲裁，也不是严格性能下界或新执行配置；不据它否定其他W8路径。

## 收据与裁决

[汇总JSON](summary.json) · [六臂CSV](comparison.csv) · [ready完整结果](ready.json) · [stress完整结果](stress.json) · [代码](run.py)。逐向量可用/最后读取记录在[ready事件](ready_decoder_events.json)和[stress事件](stress_decoder_events.json)。每臂检查77,760个updated值、77,760个门值、15,360个PED值；六臂是同两个负载的重复对照，不是六组独立数据。

root已审代码和两份结果，无新增阻断；独立审阅由root另行汇总。首次环境兼容处理仅将Python3.6改为项目Python3.12，并对保存的BN1 RMS汇总允许2ULP（实际1ULP）；源门、差异数量、最大误差、所有整数输出和旧两臂计数仍精确核验。早期[rms诊断](ready_prefix_check_failure.json)保留，不当机制负结果。

决定：保留该有费响应接口的正确性与部分改善证据，**停止固定2槽放置的加速主张**。它改善已失败RF84布局的具体问题，尚未跨过expanded16，不承担标题X；这不等于W8、MiLo或响应内解码整个家族失败。无GPU、AEE重评、RTL/EDA、main.tex、hash或git操作，写入仅在本目录。

重跑：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.12 run.py`，压力点加`--stress`；`summarize.py`仅汇总同六臂，不执行新配置。
