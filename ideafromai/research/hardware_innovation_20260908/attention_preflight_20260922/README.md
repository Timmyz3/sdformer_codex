# 神经元优先；注意力替换的第一轮隔离探针

2026-09-22，用户授权审阅 Pro 的新包、纳入后续实验，仍先修好 AT-LIF+PSN。原包保留在 `ideafromai/注意力替换研究与实验包_20260922/`，本目录是新实现与复核，不覆盖原文。

## 当前执行顺序

1. **已完成，原最高优先级**：`../atlif_psn_replication_20260922/` 的两个随机种子、每臂256有效更新、各自完整valid825。保留PSN、Motion、原动态BN；只训练5247个神经元参数，尚非全训练集/全网络联合训练。
2. **已随后完成，01:09:20**：`after_neuron_queue.py` 等待神经元队列 `complete=true` 后，跑完三臂各10帧的全分辨率推理，没有并发占GPU或启动注意力长训。固定采用第一预定seed的 `official_eps_256successful.pth` 作为修复对照，而非按验证最高分挑模型。它仍是候选参考，不是已锁定投稿学生。见[本轮结果](RESULTS.md)。
3. **后续神经元训练优先**：复核两个seed结果，确定代理规则与精度设置，再用完整训练集做恢复训练。注意力同预算短训需在该学生固定之后开展；此处没有自动启动新训练。
4. **注意力短训队列的顺序**：A0匹配恢复对照 → A1简单门 → A2具备明确可训练导引的SAF适配；再单独做空间/存储组选择与TP整块保留残差。STAtten/DSSA留作表达能力对照，不开六臂长训矩阵。

推理预检查不因零样本AEE差就杀候选；它回答接线、数值变化及潜在计算粒度。晋级仍看恢复后的任务精度与净硬件收益。遵守用户现行精度政策：**不再强卡父模型+0.005**；同口径优于SDformerFlow可继续考虑。仍同时报告相对Motion的变化，不能只挑更弱基线。

## 已核对的论文与包内信息更新

- SAFformer正式来源：[IJCAI2026](https://www.ijcai.org/proceedings/2026/212)，[方法式4–8与算法1](https://arxiv.org/html/2605.08270v2)。原算法含DWConv、SGM、整数k与门神经元。SGM算法写有跨T/B/N的聚合；批次依赖及floor/top-k的反传须明确，不能默默发明STE后称原文复现。此次未核实作者可执行代码。
- **找到包内尚未确认的TP-Spikformer官方代码**：[xlzhou126/TP-Spikformer](https://github.com/xlzhou126/TP-Spikformer)。作者README标ICLR2026，含QKFormer/SDT适配，不托管基础权重。OpenReview页面本次仍遇浏览器验证，会议信息目前据作者仓库；[作者全文](https://arxiv.org/html/2603.00527v1)的IR-Arc保留未更新位置。后续优先读其QK实现，避免重写已有选择/回填。
- [SAST官方实现](https://github.com/Peterande/SAST)保留作窗口/token选择和稀疏MLP的强对照，不因借来事件光流就认领这些已有机制。

## 本轮实际函数（不要与完整SAF混淆）

只改 `layers.2.swin_blocks.0.attn` 的pre-projection carrier，保留全量Q/K/BN/PSN、位置编码、原映射、输出投影及proj_bn。使用sn_q/sn_k hooks取数，在attn_drop前替换carrier。原Motion仍被计算后丢弃，因此**脚本运行时间不代表部署加速**。

令 `qL=_qkformer_token_q(q_orig)`，`c=count(qL>0)`。每现有逻辑head为32通道；按原450个逻辑token组成一行。

|臂|函数|含义|
|---|---|---|
|motion|原函数原样执行|A0数值对照；hook必须逐位无影响|
|q_presence_probe|`Z=K*[c>0]`|无新神经元的单位硬门；不是QKFormer整套复现|
|saf_fixed_half_probe|`Z=K*[c>0]*stable_topk(c,floor(N/2))`|SAF思路的固定k支持选择；无DWConv/SGM/gate-LIF，不是完整SAF|

同分按原逻辑索引稳定选择。保留K实际θ幅度；不引入连续gate。硬支持/选择没有Q梯度，本轮**只推理**，K的保留路径可微；不会把它直接当成已完成SAF训练。k=N回到简单门，k=0只保证Z=0。

CPU测试已过：2004组整数选择、k=0/N、全零Q、θ≠1、源lane逆映射、预期梯度连接、动态BN反例。真实网络测试随后检查A0 hook逐位一致、三个完整网络输出有限、逐帧AEE与选择统计。

## 这次源码审阅识别的三个执行障碍

1. **动态BN**：`bn_k`当前依赖全域输入统计。选择后只生成部分K linear输出会改函数。报告 `legal_preBN_K_production_ratio_with_live_BN=1.0`；选择后门/PSN的潜在裁剪另算，不能从keep50直接报K节省50%。
2. **真实坐标与旧layout**：`window_partition_v2`先按B/window排列再直接view为Tw2；Q/K又有直接reshape。现有逻辑token不等于自然位置/时间/连续通道head。首轮严格保留身份，报告逻辑选择率与按真实reshape逆映射的K物理位置需求；统计包含padding。**不编造3×3 halo、真正T10 mask或bank请求计数。** 后续空间选择必须在外层[B,T,H,W,C]做有效坐标映射，并单独评价布局改动。
3. **消费者**：`sn2_q`未安装为活ATLIF且原h60不调用，不能直接启用当新gate；`attn_sn`返回未供给proj。Z为零后仍有proj bias/动态BN，不能据此跳过整块FFN。TP式保留原特征须在同分辨率残差块外层定义，绑定完整实际T10。

因此本轮返回的是算法质量/生产依赖的线索，没有RTL加速比，没有PPA。硬件候选X仍应来自合法取消生产/完整物理请求的机制；直方图、top-k和普通回填属于先验。

## 输出

- `FUNCTIONAL_CHECKS.json`：CPU功能边界。
- `QUEUE_STATUS.json`、`QUEUE_PID.txt`：新A100后置队列。
- `fullres10/PREFLIGHT.json`：10帧×3臂、A0逐位检查、选择/物理位置需求及AEE；不是完整valid825。
- `preflight.log`：真实推理进度。

训练/推理只在新A100；不改默认H9源码、原checkpoint、生产RTL或main.tex。实验包中的所有历史数字仍留在其原始范围。
