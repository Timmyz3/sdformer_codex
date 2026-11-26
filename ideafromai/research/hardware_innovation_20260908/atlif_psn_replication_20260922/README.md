# 新 A100 上的 AT-LIF+PSN 独立恢复复验

2026-09-22。用户已授权新 A100 80GB 为今后的算法训练机，并继续现有神经元修复。新机完整数据、环境、三个身份的短训练检查已经迁移 agent 完成，本轮读取其报告并现场确认 GPU 空闲。

要回答：旧结果 normalized 的改善主要集中在一帧，改变训练顺序/增强/dropout 后，相对同预算 official_eps 的改善是否仍存在？不重做纯 AT-LIF、Q bridge、结构改造或新硬件实验。

训练设计在运行前固定：

- 两个 seed：20260922、20260923；每个 seed 独立进程，从同一原 Motion C12 ep34 重新加载。
- 每个 seed 两臂：official_eps 与 normalized；各 256 次成功更新，只训练 81 个神经元的 PSN weight/bias/θ，共 5,247 参数。
- 与旧实验相同的 256 个 B2 批次、共 512 帧；仅改变批次顺序，保留每批内部的两帧及次序。每个 seed 用 SeedSequence 的独立用途域派生顺序/增强/dropout，避免相邻 seed+step 导致随机流错位重合。保存实际文件列表与每步 seed。
- 两臂配对共享批次、增强和 dropout；fresh optimizer/scaler，原 PSN/θ 学习率、θ ε=1e-12、η=0、AMP/clip/retry 规则均保持。使用旧实验真实 diagnostic_config.yml，剔除会误导身份的旧 note/历史预算标签后另存新配置。
- 训练 B2、原动态 BN。卷积/linear AMP；PSN forward FP32 且 matmul TF32 off；backward FP32 matmul TF32 off。非神经元 matmul/cuDNN TF32 仍与旧策略一致。
- 每64步十帧只作诊断；固定第256成功步为终点，不选中途最好模型。
- 每 seed 完整比较 ep34+两终点的全部 valid825，B1/no_running、AMP off、严格 PSN 精度。两个 seed 各自新进程，避免完整验证清空 BN running buffers 后错误续跑。

主比较为每 seed 的 normalized−official_eps；同时列各自−ep34。负 ΔAEE 为改善。预先报告完整825、逐帧中位差、好坏帧/序列、像素加权AEE，以及排除已知敏感帧 zurich_city_09_a_0541.npy 后的敏感性值。正式指标仍是完整825，不根据结果挑 seed 或删帧。

两个 seed 只能评价这套固定训练集合和短预算下的随机过程稳定性，不能声称数据抽样泛化、统计显著胜出或全网长期训练收益。normalized 与 official 的比较包含代理形式和 mean→sum 的变化，不只归因于一个幅值项。旧 A800 的 AEE 仅作历史参考，新 A100 上重新配对计算基线。

先在独立 smoke 输出目录跑每臂2步检查脚本，再运行两个固定实验；smoke 不计正式预算。两个正式实验即使第一组无精度收益也执行完，不因中途诊断修改第二组。

实现复用 20260920 已审阅循环；新入口仅增加明确 seed、批次顺序和记录、显存峰值统计。完整验证继续调用旧 validate_candidates.py。默认 H9、third_party、原始 checkpoint 和生产硬件不修改。

运行：新服务器 `/root/private_data/work/envs/sdformer312/bin/python run_queue.py`。队列顺序执行，每个 seed 全验证后保存独立 SUMMARY.json；最终保存 REPLICATION_SUMMARY.json。所有输出在本目录的新 run 子目录，不向原模型路径写入。

2026-09-22 运行前独立审阅已指出并采纳：分域随机种子、保留 B2 配对、每 seed 新进程、以 normalized−official_eps 为主、对已知敏感帧预先列敏感性统计。两 seed 的 ep34 基线另做逐帧比较；若漂移，不把混合基线结果当稳定性结论。

## 已完成结果（2026-09-22）

两seed全部训练与valid825完成，每臂256次有效更新，均无AMP溢出；两次基线逐帧AEE完全一致。详见 `REPLICATION_SUMMARY.json` 与各seed `SUMMARY.json`。

|完整825的帧均AEE|seed20260922|seed20260923|
|---|---:|---:|
|ep34，同机严格PSN精度|1.201963093|1.201963093|
|official_eps|1.195747149|1.198072748|
|normalized|1.196409903|1.197103872|

normalized相对official一胜一负；排除预先指定敏感帧的辅助统计中，两次都更差（+0.001303/+0.001167）。正式指标仍含全部825，不能把这种小预算结果称为代理梯度稳定优胜。保留官方修复路径为后续参考，normalized仍是候选；尚未启动完整训练集/全网络训练。

按用户新授权，在本队列完成后另跑了三臂各10帧的注意力推理探针，使用固定第一seed official_eps，不变更神经元、不并行抢GPU。见[后续注意力结果](../attention_preflight_20260922/RESULTS.md)。
