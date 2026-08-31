# M650｜M511 单次 capture 失败负结果 fresh hammer

最终结论：`PASS_NEGATIVE_MILESTONE__ALL_BINARY_DECODER_ASSUMPTION_DISPROVED`，99/100，P0=0、P1=0、P2=2。

本轮只读审计 M647 已授权且已经消耗的 M511 one-shot 失败现场；没有删除或修改 attempt/staging，没有重跑 GPU，没有运行 payload verifier、EDA 或远端任务，也没有修改 `docs/359`。M647 review 双 seal、attempt initial 双 seal和其中九项身份全部重新验证通过。

## 可封存事实

- M647 外层 seal 文件 SHA256 为 `4db2718e368ccbf1a1991efdc5fd92778c9d6b97aa1923c7fb7775eb75f29bd0`；其结论只授权 exact M646 one-shot capture，不授权任何性能结论。
- attempt 已在 `2026-08-28T11:22:51+08:00` 标为 `CONSUMED_IMMEDIATELY_BEFORE_M511_ONE_SHOT`。initial `SHA256SUMS` 和 outer seal 均通过，九项文件身份逐项 `sha256sum -c` 通过。
- 三次资源采样均在运行前：commit headroom 为 103,401,272 / 108,518,192 / 108,278,428 KiB，MemAvailable 为 419,766,260 / 423,506,640 / 423,207,120 KiB，GPU free 均为 24,048 MiB；cgroup failcnt/under_oom/oom_kill 均为 0。
- canonical capture `system_handoff/outgoing/m511_h67_ep35_convtranspose_binary_inputs_s10_r1_20260827` 不存在；canonical payload-verifier output `results/m511_h67_ep35_convtranspose_payload_verify_r1_20260827` 不存在。
- producer 留下的 staging `...staging.rn2m8c3a` 仍存在且只有 `FAILED.json` 与首个 d0 bitpack。`FAILED.json` SHA256=`343a29e2932345e83d9da2410eb070f22520b9c0e4302e4940a25633e2250863`，声明 `status=FAIL_CLOSED_NO_PASS_MANIFEST`、`completed_records=1`，失败原因为 `RuntimeError: M511 raw ConvTranspose2d input is not exact binary`。
- 首个 `s00_d0.activation.le.bitpack` SHA256=`ad2251f1fb8a470651044456e0b7182bd6db0e0a89fb63018efa3a9e6fcd6447`，大小 576,000 byte。按冻结 d0 shape `[10,1,1536,15,20]` 解码为 4,608,000 bit，其中 839,586 个 1、3,768,414 个 0，活动率 18.220182292%；元素数整除 8，无尾部填充歧义。

## 结构解释——明确标为推断

冻结合同与冻结源码给出 base channel 96、四级 encoder、倍率 2、输出 flow channel 2。decoder 构造式为 `2 * input_size + prediction_channels`，且 forward 只在 `i>0` 时把 `predictions[-1]` concat 到下一层输入。因此：

- d0：`2×768=1536`，没有 previous-prediction channel；
- d1：`2×384+2=770`；
- d2：`2×192+2=386`；
- d3：`2×96+2=194`。

据此，可把后三层的结构写作 **`C_feature + 2 previous-flow channels`**。结合 d0 成功、d1 随即失败，“两条 previous-flow channel 破坏 all-binary 假设”是很强的定位假设；若把 `C_feature` 记成 `C_binary`，必须显式标注为 **inference/hypothesis**。现有 `FAILED.json` 没有记录首个非二值元素的 channel，也没有保存 d1 原值，因此本审计不声称已经实测证明非二值值只来自那两个通道；后续需要 selective channel probe 才能闭合。

## 结论边界

本里程碑只证明：冻结 H67 ep35、首个 S10 样本执行中，“四个 ConvTranspose2d raw input 全部 exact binary”的 M511 前提为假；捕获器按合同 fail closed，未发布 PASS manifest。

它不证明 decoder 不能使用混合 binary/analog 数据通路，也不证明 d1/d2/d3 的 feature 部分不是 binary。它没有产生 40-record payload、cycle、speedup、能量、RTL/VCS、Synopsys/PPA、全网指标或 DATE headline。任何后续 decoder 性能工作必须重新定义 mixed-channel contract，不能把本次 1-record staging 当可准入完整 trace。

## P2 风险与现场处置

1. staging 没有 producer PASS seal，仍是可变失败现场；M650 只把其当前两个文件的 exact SHA/size 写入自己的双 seal，不把 staging 升格成 canonical artifact。
2. 本审计没有读取磁盘上的模型 stdout/stderr 日志，也没有从未提供的 exec transcript 补写 checkpoint-load、显存峰值或具体首个非二值数值。若主流程另有实时 exec 输出，必须单独封存后才能引用。

现场必须保留：one-shot 已消耗，canonical capture/verifier output 均 absent，失败 staging 不删除、不改名、不复用为 PASS 输入。`docs/359` SHA 复核仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
