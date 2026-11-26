# R13：固定更新预算的数据覆盖对照，身份协议

本目录只实施身份选择与隔离，未登录服务器、未读取任何事件/GT/预测/损失数组、未执行 GPU。用于 R13 从原始 135930 checkpoint 重新建立共同部署状态，再比较 reference40 与 diversity200。它不复用 R12 的恢复权重或旧 CMA176，不将 R12 的 fit20/check8 当作新的 reference。

## 冻结结果

权威入口为 [manifest_v1.json](manifest_v1.json)，SHA256：`eba4544fe99c082d7d099f90d03470d7abad22b01b68d3eb4d022e61e78151d1`。schema 为 `flow_round13_coverage_identity_v1`。所有成员均为原 `common_targets.json` 的完整 canonical row；训练顺序每行仅额外增加 `step`、`sequence_visit_index`，均从 1 开始。

| 新角色 | 成员/访问方式 | 序列限制 |
|---|---|---|
| reference recovery | 8×5=40 身份，每个访问5次，共200更新 | 仅新训练8序列 |
| diversity recovery | 8×25=200 身份，每个访问1次，共200更新 | 与 reference 每一步的序列相同 |
| selector reserve | 8×5=40 身份，仅预留，尚未采集或拟合 | 与完整 diversity200 身份不重叠 |
| BN calibration | 8×22=176 身份，原 `bn_calibration` 角色 | 完全排除两条新 check 序列；仅事件用于校准 |
| terminal check | 2×4=8 新身份 | 新恢复/校准/selector 流水线排除的两整条序列 |

十条 `gate_fit` 序列按 `(SHA256(sequence UTF8), sequence)` 排序，前两条固定为 **zurich_city_10_a、zurich_city_11_a**。不读取质量挑选序列。余八条的固定顺序为：07a、09a、02e、06a、11c、10b、05b、02c。

每条训练序列在排除历史164身份后，按 `SHA256("flow-r13-coverage-members-20261016:"+target_id)` 排序，前5为 reference、前25为 diversity、第26至30为 selector reserve。每轮按固定八序列顺序访问一遍，共25轮；reference 在本序列前5身份间循环，diversity 单遍访问本序列前25身份。这样固定更新数、每条序列访问数与序列课程，所改变的是身份覆盖/重复率这个联合因素。**不是样本独立性、训练计算相等或因果覆盖机制的证明。**

最小的 07a 原有35个 `gate_fit`，排除旧 fit20 的0021/0088后33个；25 recovery + 5 selector 可行，余3个。没有为凑配额放回旧身份或在观察质量后更换规则。

## 历史排除和曝光边界

选择器绑定10份原始元数据文件的 SHA。历史排除逐级重构自：dev32、check64、旧 fit20、R6 check8、R9 check16、R10 check8、R11 check8、R12 check8。并核对其并集等于 R12 权威 `check_membership_v1.json` 的 `excluded_ids ∪ fit ∪ check`，共 **164 身份**。旧 fit20 即使并非 audit，也保守排除。R7/R8及中间探索复用的身份由该既有身份链覆盖；本协议不主张所有原始训练身份未被使用过。

新 check：

| target_id | 距最近上述历史身份的 target index 差 | 距最近旧 BN176 身份的 index 差 |
|---|---:|---:|
| zurich_city_10_a_0475 | 133 | 22 |
| zurich_city_10_a_0168 | 174 | 55 |
| zurich_city_10_a_0509 | 114 | 6 |
| zurich_city_10_a_0554 | 69 | 39 |
| zurich_city_11_a_0070 | 11 | 3 |
| zurich_city_11_a_0176 | 78 | 6 |
| zurich_city_11_a_0209 | 111 | 4 |
| zurich_city_11_a_0016 | 43 | 5 |

这些是**文件目标索引差，不是秒数或时间独立性保证**。canonical 时间字段尚为 null。新 check 内预设 gap≥3，实际最小33。训练侧 BN/recovery/selector 按目标身份互斥，但有相邻索引，没有强制时间清洗；完整逐身份距离已保存在 [identity_diagnostics_v1.json](identity_diagnostics_v1.json)。OF 上下文文件不是本轮 v2 的输入；本协议没有建立跨模型多窗输入互斥证明。

原始骨干已见全部 train。两条新 check 序列此前各有2个旧 fit20 身份，且各有16个旧 CMA176 身份；因此它们只是**新恢复/BN/selector 流水线的序列留出**，不是骨干未见、研究历史未见、独立场景或官方 DSEC 测试。新的 BN176 与旧 BN176 有128身份相同，但本轮必须从原 checkpoint 重置 BN 并重新校准，不读取旧 CMA 的统计结果；其余48个校准身份来自八条合法训练序列的原 BN 角色池。

## 执行契约

- 权重起点为原135930 SHA `d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30`。各条件从相同新校准状态重新开始、fresh optimizer；禁止恢复 R12 终点。
- BN 校准仅使用 `bn_members` 的事件；两条 check 序列不能进入新的 BN/PCA/scaler/selector/位置先验等任何拟合。校准算法、native 78 BN、导出的24私有 BN、54外部 current-stat 合同由独立 preparation/runner 绑定，本身份清单本身不证明数值正确。
- `check_members` 仅用于所有比较条件和规则固定后的统一终点审计。初始锚点与终点可使用同8身份，但不能在候选/训练/超参尚未固定时打开检查质量并据此改配方。
- `selector_members` 只是40身份预留，尚未冻结 head 训练/折叠/标签合同，也不授权将 check 标签用于 selector。后续如果训练权重变化，旧特征、干预收益与评分头须重建。
- 原 canonical 的 GT SHA/时间字段为空并不等于缺少运行时数据验证；实际 arrays 的 acceptance inventory SHA、形状、数值和部署源校验由后续 preflight 完成。本目录不伪造这些证据。

## 复现与核验

```bash
python3 research/flow_story_20261016/coverage_recovery_round13/data_protocol/build_manifest.py --verify
```

`build_manifest.py` 无数组库依赖；新建输出时拒绝覆盖，`--verify` 从10份固定元数据重新计算并逐对象比较。另一个不导入选择器的 [audit_manifest.py](audit_manifest.py) 已独立重构排序、历史并集、角色约束和400步课程，比较 **864 个完整 canonical row**，结果 [metadata_audit_v1.json](metadata_audit_v1.json) 为 PASS。这是同一数据协议代理的第二条实现检查，不冒充外部独立评审；根代理/评审代理仍可复核。

数据隔离依据：[R12 后续分支协议](../../ordinary_recovery_round12/AFTER_RECOVERY_DECISION.md)，原身份源和历史 SHA 逐项见 manifest 的 `source_paths`、`source_sha256`。
