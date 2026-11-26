# A 线共同 DSEC target 清单与本机数据核验

2026-10-11；本轮已在 **ismd 主机 `ic.ismd-nemo`** 生成并检查真实清单。只做 CPU 文件/数组/来源核验；未启动 GPU、训练、模型权重加载或数据预处理，未改旧代码。证据范围是下列明确的数据根目录，不代表全机数据已穷尽搜索。[机器状态](artifacts/ready_status.json)、[数据根与数量](artifacts/availability_summary.json)

**已完成：共同训练6000／验证2152个 target、OF双片上下文和v2单片target的可用清单、辅助校准/开发角色、SHA256及缺失清单。当前 `membership_manifest_ready=true`，`common_training_data_ready=false`。** 默认本地v2数据根仅有9个共同训练target具备完整输入/GT/mask，验证为0；默认OF数据根没有预处理数组。不能据此启动长训练，也不能把清单生成成功当作协议/精度已复现。[状态](artifacts/ready_status.json)、[逐target可用性](artifacts/local_availability.csv)

**同日后续已找到完整源：** [旧服务器审计](ARCHIVE_DATA_AUDIT.md)确认共同6000/2152的v2三件套全部存在，4样本GT/PNG绑定通过；[取回记录](archived_sample_staging.json)证明20个小样本文件已复制到独立ismd缓存并逐文件核SHA。下文9/0及ready_status仅描述早期指定本地默认根，未覆盖这个新缓存或远端源；不再代表完整数据来源未知。

## 固定来源与实际产物

成员严格取自作者 `J-Cuadrado/OF_EV_SNN` 提交 `ff9406130acab6dcd3073cbe18f7f764806b0153` 的 `train_split_doubleseq.csv` 和 `valid_split_doubleseq.csv`。脚本先校验已核对的官方CSV SHA256；不匹配即拒绝生成，不会静默切到新版划分。保持原行顺序、所有6000／2152成员、13／5序列，以及OF两列关系。[来源与检查](artifacts/source_provenance.json)

| 产物 | 用途 |
|---|---|
| [common_targets.csv](artifacts/common_targets.csv)、[common_targets.json](artifacts/common_targets.json) | canonical清单：序列、target、OF前片上下文、两种模型输入合同、未知时间戳/GT绑定字段 |
| [of_train_pairs_noheader.csv](artifacts/of_train_pairs_noheader.csv)、[of_valid_pairs_noheader.csv](artifacts/of_valid_pairs_noheader.csv) | OF原loader需要的双列、无表头清单 |
| [v2_train_targets_noheader.csv](artifacts/v2_train_targets_noheader.csv)、[v2_valid_targets_noheader.csv](artifacts/v2_valid_targets_noheader.csv) | v2单列、无表头清单；与OF第二列逐行相同 |
| 同名 `*_pairs.csv`、`*_targets.csv` | 有表头的人读/通用程序导出；**不要直接交给原loader** |
| [train_auxiliary_roles.csv](artifacts/train_auxiliary_roles.csv)、[policy JSON](artifacts/train_auxiliary_policy.json) | `aux_policy_v2`：仅训练侧的BN校准、gate拟合、gate开发角色；不改变backbone训练成员 |
| [BN initial CSV](artifacts/bn_calibration_initial_targets.csv)、[BN expanded CSV](artifacts/bn_calibration_expanded_targets.csv)、[schedule JSON](artifacts/bn_calibration_schedule.json) | 176条初始校准、703条预留扩展池及确定执行顺序；JSON提供可直接传给校准工具的target ID列表 |
| [missing_artifacts.json](artifacts/missing_artifacts.json) | 指定根下缺失/不合格的数组绝对路径；另列每序列raw时间戳、PNG目录、events.h5、rectify_map缺失 |
| [existing_array_evidence.json](artifacts/existing_array_evidence.json) | 实际找到的必需数组的SHA256、shape、dtype、有限值和mask有效像素数 |
| [timestamp_gt_mapping_audit.json](artifacts/timestamp_gt_mapping_audit.json)、[raw_metadata_audit.json](artifacts/raw_metadata_audit.json) | 原始时间戳/PNG序号关联是否可核；本次未知项均为null/明确未验证 |
| [artifact_sha256.json](artifacts/artifact_sha256.json) | 23个生成产物的SHA256；不含自身，避免自引用 |

生成后做了独立检查：8152个唯一target；训练/验证target、序列及OF上下文clip均无交叉；两种模型无表头导出逐行对应canonical；辅助角色并集正好覆盖6000训练target；所有产物哈希一致。官方CSV原文保持不变。[源码](build_and_audit.py)、[成员检查](artifacts/source_provenance.json)

`common_targets.json` 中的时间戳、原PNG名、GT/mask哈希保持null，CSV对应空字段；**这些canonical字段没有从文件名猜数**。少量本机真实数组哈希记录在独立audit，避免“同名文件存在”被误作已绑定原始GT。`timestamp_row_0based_candidate=n−1` 只表达源代码的序号规则，字段名与状态明确为candidate；不是已验证时间映射。[canonical](artifacts/common_targets.json)、[audit](artifacts/timestamp_gt_mapping_audit.json)

## 训练侧校准与开发：不改变原生backbone划分

当前策略为 **`aux_policy_v2`**。保留 `zurich_city_01_a`、`zurich_city_05_a` 两个完整序列用于gate开发；在剩余11个官方训练序列内，分别按 `SHA256("flow-a-bn-pool-v2:" + target_id)` 升序排列（相同摘要时按target ID打破平局）。每序列前16条构成初始BN校准集；前 `min(64, N)` 条构成预留扩展池；剩余target用于gate拟合。这个规则不读取图像、AEE、事件密度或模型结果。[角色策略](artifacts/train_auxiliary_policy.json)

v2替代此前只用 `zurich_city_06_a` 的641条单场景规划，目的是降低单场景校准造成假失败的风险，并在实验前统一相冲突的规划。没有已训练或已做真实BN校准的旧结果被覆盖。**共同6000／2152清单及8份模型导出的SHA256均与调整前完全相同**；官方训练/验证成员没有改变。[独立复核](validation_record.json)

| 角色 | 序列 | target数 | 使用限制 |
|---|---|---:|---|
| BN calibration initial | 非gate开发的11个训练序列，各16条 | 176 | 输入统计初始集；是扩展池的严格子集及执行顺序前缀 |
| BN calibration expanded pool | 同11个序列；`zurich_city_02_a`仅63条，其余各64条 | 703 | 从一开始全部预留，不用于gate拟合/开发；扩展时开启新的capture，按冻结顺序重新收集 |
| Gate development | `zurich_city_01_a`、`zurich_city_05_a` | 724 | 选择gate阈值、预算、超参数及对照设置 |
| Gate fit | 非gate开发序列中排除整个BN扩展池后的target，具体见CSV | 4573 | 拟合风险评分/等预算学习式gate |
| 最终研究验证 | 官方5个valid序列 | 2152 | 阈值、BN、模型/预算选择完成后才正式评价；不用于调参 |

**所有6000条仍是backbone训练成员。** BN扩展池、gate开发、gate拟合的target ID两两互斥，并集正好6000；176条初始集不重复计入这个并集。BN与gate拟合共享部分序列，而且相邻OF两片输入共享1050个clip文件；不能称它们是序列或上下文独立的holdout。Gate开发与另两角色按序列隔离，上下文clip交叉均为0。[策略及交叉计数](artifacts/train_auxiliary_policy.json)

辅助角色的target隔离并不意味着gate开发数据对backbone是未见数据；不得用724条声称backbone独立泛化。若另需真正的backbone开发holdout，必须另立有标识的探索协议和训练运行，不能删改当前官方成员或把结果当原生复现。本轮没有强加这一额外训练。BN池只估计输入统计，不用GT选择校准成员或阈值；初始→扩展动作须预先约定，不能读取5个验证序列的AEE后再挑有利校准集。[策略](artifacts/train_auxiliary_policy.json)

重复使用gate开发集调参仍可能过拟合，因此应预先限制候选/轮数，最终只在冻结方案后打开5个valid序列。多预算工作点预先在开发集确定，再把全部工作点一起用于最终验证；不能在验证曲线上挑最有利预算。[计划]

## 本机数据实际边界与预处理问题

默认核验根：[availability_summary.json](artifacts/availability_summary.json)完整记录。v2为 `SDformer/data/Datasets/DSEC/saved_flow_data`；OF为只读官方快照的 `data/dataset/saved_flow_data`。

- v2目录共见GT/mask各10个，全部属 `zurich_city_09_a`。其中`0001`不是官方双片target，故共同清单可用的是9条，全部落在训练侧。其他8143个共同target的GT/mask/input仍缺失。[数量](artifacts/availability_summary.json)、[缺失清单](artifacts/missing_artifacts.json)
- v2事件目录有268个NPY：`zurich_city_09_a` 10个，`interlaken_01_a` 108个，`thun_01_b`75个，`zurich_city_12_a`75个。后三者不在此13／5研究清单中，不能用来填充2152验证target。[目录计数](artifacts/availability_summary.json)
- 本轮对指定raw目录的18个序列均未找到能绑定所需target的 `forward_timestamps.txt` 和forward PNG清单，因此所有跨模型目标时间映射尚未核实。清单有同名target不证明实际文件有同一时间范围。[raw audit](artifacts/raw_metadata_audit.json)
- 另查 `/data`、`/datasets`、`/home/zhumd/data`、`/mnt`：`/data`存在、`/mnt`为空，另两处不存在。在深度≤4、不跟随symlink、跳过通用线目录的**目录名**扫描中未找到DSEC命名目录；部分共享目录无读取权限。**这不是断言ismd全机没有完整DSEC**，更深目录、其他挂载或其他机器的数据仍需明确路径核验。[有界扫描](data_root_discovery.json)、[扫描脚本](discover_data_roots.py)

新发现的协议问题均来自固定源码，不能未经核对就修掉后继续称“原生复现”：

1. **OF事件路径不一致。** 固定官方loader读取 `event_tensors/11frames/<filename>`，生成器写 `event_tensors/11frames/left/<filename>`。本脚本默认依loader核验；可用 `--of-event-subdir event_tensors/11frames/left` 显式核验生成器布局，但这不修改或修复原loader，后续wrapper必须明确其路径适配。[来源](artifacts/source_provenance.json)
2. **rectification语义不能默认一致。** OF固定生成器计算了`x_rect/y_rect`后仍把原`x/y`送入count-frame生成函数；v2本地生成器把rectified坐标送入voxel并裁掉界外事件。这是源码事实，不足以认定作者已发布权重究竟用哪种输入。共同target不抹去这项差异；需要把原生复现、统一预处理变体和部署实验分别标识。[来源](artifacts/source_provenance.json)
3. **GT解码表面通道次序不同有库语义。** OF用imageio的RGB顺序读x/y/mask；v2用OpenCV BGR读对应的2/1/0通道。代码层面可对应，仍须拿实际同PNG解码并比较GT/mask内容，不能仅以源代码推断所有现有NPY已一致。[两份生成器的固定SHA与路径](artifacts/source_provenance.json)
4. **输入时间上下文不同。** OF保留前片＋当前片的21个帧，v2是单target时间段的10-bin输入。共同清单统一target时间/GT，不暗改各网络输入观察时间；这项差异须在主表注中声明。[canonical输入合同](artifacts/common_targets.csv)

## 可运行入口与后续完成条件

从任意工作目录运行本机核验：

```bash
python3 /home/zhumd/work/sdformer_codex/research/flow_story_20261011/preflight/data_protocol/build_and_audit.py
```

Python 3.6+；NumPy仅用于现有NPY内容/shape核查。脚本不导入torch，不触发CUDA，不加载checkpoint，不下载数据，不读取/写入旧实验模型代码；只在指定 `--output-dir` 写产物。缺少NumPy时成员清单仍可生成，但数组不能标ready。对其他明确数据根复核时使用新输出目录保存机器间证据：

```bash
python3 data_protocol/build_and_audit.py \
  --v2-saved-root /explicit/v2/saved_flow_data \
  --v2-raw-root /explicit/v2/DSEC \
  --of-saved-root /explicit/of/saved_flow_data \
  --of-raw-root /explicit/of/dataset \
  --of-event-subdir event_tensors/11frames/left \
  --output-dir data_protocol/audit_other_root
```

上面路径是待替换的模板，不是已确认存在的位置。原始布局按本地v2的 `train_optical_flow/<seq>/flow`、`train_events/<seq>/events/left`，以及OF生成器的 `train/<seq>/flow`、`train/<seq>/events/left` 读取；若实际布局不同，需要显式适配审计路径，不能改名后假设内容一致。

脚本若找到raw时间戳和数量匹配的PNG，会按原生成器序号规则给出target/context时间范围及PNG名，并比较两模型GT/mask数组内容。但它**不会自动把NPY和PNG的内容来源绑定判通过**，也不会自动抹平rectification差异；`gt_npy_to_raw_png_content_binding_ready` 和预处理一致性需在下一步实际PNG解码/输入来源核验后才可关闭。现阶段这些false是未取得证据，不是已判数据错误。

因此下一步首先定位完整数据根，运行同一审计；随后完成原PNG解码内容比对、时间戳和rectification合同，再启动CPU全模型数值预检。不要把论文原AEE复现结果作为“开始复现训练”的前置条件：**原文约10%精度门是后续独立证据门**；这里的 `gpu_training_ready=false` 是数据前置条件未齐，且不代替服务器资源/授权检查。[机器状态](artifacts/ready_status.json)

原文v2的官方测试1.602不能用作当前2152本地验证的10%复现参考。保留原生协议复现门与本共同划分的“我们重训＋同基线ΔAEE”两层记录，不能用同一个数字冒充两者。[前置主合同](../FLOW_A_PREFLIGHT.md)
