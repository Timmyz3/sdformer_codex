# M75 hard-binary-support STE / fail-closed pin 独立打铁评审 R1

## 判定

**算术与梯度修复 PASS；规定的三项负向攻击 PASS；整体 fail-closed FAIL；正式 PAFT 继续 BLOCK。**

M75 已经正确消除了上一轮最直接的幅值作弊：forward hardware proxy 只看 hard binary support，0.2 与 0.9 的非零幅值产生完全相同的 support、baseline 和 candidate cost，同时 identity STE 保留有限且非零的梯度。两个现有 PAFT 生成配置也都保持 `enabled=false`，M71 撤销合同仍有效。

但生产路径保留了一个普通 config 即可打开的 `unit_test_allow_unpinned_revoked_catalog`，它会绕过 catalog SHA、train-only role、train eligibility、train-list SHA、checkpoint identity 和 operator pin。隔离攻击已证明原始 revoked M71 catalog 可由该开关直接接受；更严重的是，把原 M71 JSON 的自报 role/eligibility/identity 字段重新贴标签并在 config 中匹配其 SHA 后，正常路径也会成功安装四个 PAFT hooks。当前没有 known-revoked catalog SHA denylist，也没有外部 revocation/overlap receipt 的不可伪造 pin。

因此，M75 不能被解释为“M71 不可能恢复训练资格”。目前安全的原因是正式配置关闭，而不是 admission 已经真正 fail-closed。

## 冻结身份

| 对象 | SHA256 |
|---|---|
| `pattern_paft.py` | `22292b265292b4d3c00cdeb1addd3020c7b2a417adc855aa043d1394735d3bf1` |
| production M71/M75 validator | `cb3eac62663fc3618e5b4019686fa0cf121bfad942992cd84bc12ffa7e79c4ba` |
| M75 r3 receipt | `e3abe4438884ec1764aab6d3660e5cc9413d414b75a24abd1efa90952d774c02` |
| revoked M71 catalog | `142e32f0d988721ce9edf25d4dcf3883d82f2604f2aee9c755cde87b2ef70cdd` |
| M71 revocation contract | `4a96226b35234366854c656db6f7443699f6d91131b8281c56a36039bf3a0238` |
| formal 5-epoch config | `9c045a4bd5d590d03295745772ce918a870b6458dac295a89006c075ab9a0170` |
| one-step smoke config | `49bcd8a92b1a4a6b198474dabfca4295eab065da4cf2f4388d4bb206304e8a46` |
| training entrypoint | `fccd1d05bbf73aac0061e604a9d199cf9c3fd4ba8e9cea175231a3ebc14e44ac` |

独立算术验证器没有 import `pattern_paft.py` 或 production validator。生产 admission 攻击在单独脚本中显式 import，并且所有变异文件只写入 `TemporaryDirectory`。本评审未修改生产文件。

### 评审期间的生产漂移边界

本 R1 的 admission 攻击严格针对表中 `pattern_paft.py` SHA `22292b...` 和 r3 receipt 所 pin 的 validator SHA `cb3eac...`。P0 报告给根线程后，根线程立即把 live `pattern_paft.py` 改成 M77 schema/external-contract/denylist 版本，并继续修改 validator；这些后续文件不属于 M75 r3 receipt 的身份，也未由本 R1 准入。下面的 bypass 结论是目标 SHA 的历史事实，不应因后续覆盖源码而抹除；后续实现必须用新 receipt 和新独立轮次复审。

这次漂移也直接印证 r3 缺 `pattern_paft.py` SHA 的问题：receipt 本身无法判断自己测试的是旧实现还是后续实现。

## hard-support 算术独立复算

独立实现使用：

```text
hard = detach(x != 0)
support_ste = x + detach(hard - x)
cost = min(popcount(support), 1 + min Hamming(support, pattern))
```

对两个向量、每个向量前四位非零，并以相同四位 pattern 为 center：

| 非零幅值 | binary support | baseline | candidate |
|---:|---|---:|---:|
| 0.2 | 相同 | 8 | 2 |
| 0.9 | 相同 | 8 | 2 |

两幅值的 support tensor bit-exact 相同且只含 `{0,1}`。对 `candidate + baseline` 反传：

- 0.2 gradient L1：`27,616`；
- 0.9 gradient L1：`27,616`；
- 两者 gradient tensor 相同、finite、非零。

为避免仅由 baseline 项“代替”实际 regularizer 提供梯度，另构造前三位 active、第四位缺失的 near-pattern，只对 candidate 反传：

- 两幅值 candidate 均为 `2`；
- candidate-only gradient L1 均为 `6,897`；
- 两者 gradient tensor 相同、finite、非零。

这部分判定为 **PASS**。M75 修复了 raw amplitude 直接参与 Hamming/popcount 的 P0。

## r3 receipt 审计

r3 receipt 的以下内容可复核：

- validator SHA 和 catalog SHA 与当前文件一致；
- amplitude 0.2/0.9 的 candidate=`2`、baseline=`8`；
- direct gradient L1=`27,616`；
- hook penalty=`0.0052001155`、hook gradient L1=`0.0008727789`，均 finite/nonzero；
- missing catalog SHA、wrong catalog SHA、当前 revoked role 三项均被拒绝；
- accuracy、heldout/cycle speedup、RTL/PPA 均没有 admission。

receipt 的 claim boundary 是克制且正确的。

但 receipt identity 只 pin 了 validator 和 catalog，**没有 pin 被 import、真正实现 hard-support/pin 逻辑的 `pattern_paft.py` SHA**。validator SHA 不能覆盖动态 import 的模块内容。结合上一轮已经实证的 source-SHA TOCTOU，这个 receipt 还不是可封存的 implementation identity。

## 规定的三项 fail-closed 攻击

隔离调用生产 `_load_catalog` 的结果：

| 攻击 | 结果 | 拒绝原因 |
|---|---|---|
| 缺 `catalog_sha256` | PASS，拒绝 | `catalog SHA pin absent or mismatched` |
| 错 `catalog_sha256` | PASS，拒绝 | `catalog SHA pin absent or mismatched` |
| 当前 revoked M71 role | PASS，拒绝 | `lacks the train-only role receipt` |

这三项规定攻击通过，但只覆盖“未经修改的当前 M71 JSON + 没有测试 override”的窄路径。

## P0：config-controlled override 可恢复 revoked M71

目标 SHA `22292b...` 的生产 `_load_catalog` 和 `install_pattern_paft` 都读取：

```text
unit_test_allow_unpinned_revoked_catalog
```

只要普通 config 将其设为 `true`，所有正式 pins 均被绕过。隔离攻击对原始 M71 catalog 设置该字段后，loader **接受**。production validator 自身正是靠该路径安装 revoked catalog 做 hook test，所以这不是静态猜测。

这是 P0：测试旁路位于生产模块、由训练配置控制，没有 test binary/build/environment 边界。任何错误配置或复制配置都能恢复被撤销 catalog。

修复要求：生产 loader/installer 完全删除该 config key。unit test 应在 test-only 文件中构造最小合法 synthetic catalog，或通过不可由 YAML 控制的私有测试 helper；生产入口不存在 bypass 分支。

## P0：自报 role/identity 可重贴标签

对目标 SHA 的第二个独立攻击复制 revoked M71 catalog，仅做以下修改：

```text
split.role = DSEC_TRAIN_ONLY_PAFT_CALIBRATION
split.train_catalog_eligible = true
identity.train_sequence_list_sha256 = config 自选 SHA
identity.checkpoint_sha256 = 临时 checkpoint 的真实 SHA
```

然后 config pin 修改后文件的真实 catalog SHA、相同 train-list/checkpoint SHA 和原四个 operator names。结果：

- `_load_catalog` 接受；
- `install_pattern_paft` 完整接受；
- 四个 revoked M71 hooks 全部安装成功。

原因是所有“来源证明”都由同一 catalog/config 自报。loader 没有：

- known-revoked catalog SHA denylist；
- M71 revocation contract SHA/status 检查；
- 外部、独立生成的 train/valid/test overlap receipt；
- dataset file receipt 或 calibration sample identity pin；
- catalog builder/algorithm SHA pin。

修复要求：launch contract 必须 pin 可信的 train-only catalog allowlist、builder SHA、train list SHA、checkpoint SHA、sample file receipts、valid825/test overlap=0 receipt，并显式 deny revoked M71 SHA。生产代码只消费这份外部 contract，不能让 catalog 自己证明自己。

## 正式配置与 M71 撤销状态

扫描 `configs/generated/*.yml`，只有两个配置包含 `pattern_paft`：5-epoch 与 one-step smoke。两者均满足：

- `pattern_paft.enabled=false`；
- `blocked_reason=M71_VALID825_CATALOG_REVOKED_USE_TRAIN_ONLY_SUCCESSOR`；
- `paft_catalog_split=REVOKED_M71_VALID825_INTERNAL_SAMPLES_0_TO_4`；
- `paft_heldout_split=REVOKED_NOT_AN_INDEPENDENT_HELDOUT`。

M71 revocation contract 仍为 `REVOKED_FOR_PAFT_TRAINING_VALID825_DATA_LEAKAGE`，`m71_catalog_train_eligible=false`。因此当前生成配置不会启动 PAFT，这一点 **PASS**。

但 formal config 的 `note` 仍把 M71 称为“frozen train-only catalog”，`resume_protocol` 仍写 five-epoch PAFT，属于 P1 语义残留；应改成 revoked/internal-screen only，防止人工复制时误启用。

## 其他 P1/P2

### P1：没有绑定 Phi/Lloyd catalog 方法

loader 仍只接受旧 M71 schema，没有要求 `calibration_algorithm=filtered weighted Hamming Lloyd` 或 builder SHA。即使将来来源确实为 train split，也可能重新使用已证明较弱的 top-frequency catalog，而非 M72 的 Phi-aligned 方法。

### P1：operator pin 太晚且失败不回滚

`expected_operator_names` 在 state 建立、pattern 加载、hooks 注册之后才检查。若 pin mismatch 或中途缺 module，函数抛错但 model 已被部分修改。应先完整验证 catalog/operator list，再一次性 attach；异常路径必须移除已注册 hooks 并删除 state。

### P1：缺 binary/nonnegative source contract

`vectors != 0` 会把负幅值也映射成正 support 1。当前 H67 配置预期 binary `{0,+theta}`，但 PAFT installer 没有 pin 上游 output mode/threshold module identity，也没有 `vectors >= 0` 断言。若未来接入 ternary/negative source，cost 会丢失 sign。应 fail-closed 断言 nonnegative binary-threshold source，或实现正负两 plane cost。

### P1：实际训练数据列表未在运行时核验

loader 只比较 catalog identity 与 config 中的 `train_sequence_list_sha256` 字符串，没有对训练 dataloader 实际使用的 sequence list 文件重新计算 SHA。需要在 train entrypoint 解析实际 dataset list 后计算并传入可信 receipt。

### P1：production test 覆盖仍不完整

production direct test 只对 low case 回传 `candidate+baseline`，未单独断言 high gradient 非零，也未单独测 candidate-only near-pattern gradient。本独立评审补测通过，但正式 validator/receipt 应补上这些字段。

### P2：训练效率

- 每个 hook 的 finite 检查调用 `.all().item()`，GPU 上会同步；
- pattern tensors 保持 CPU，并在每个 partition chunk/forward 重复 `.to(device,dtype)`；
- 固定 64 个位置仍可能采样别名。

这些不影响本次 directed correctness，但会降低 5-epoch 训练吞吐并扩大 proxy sampling noise。

## 评分（100 分）

| 维度 | 分数 | 判断 |
|---|---:|---|
| hard-support 数值正确性 | 94 | amplitude invariance、binary forward、总梯度及 candidate-only 梯度均独立通过 |
| 硬件/算法协同创新性 | 64 | 将硬件 support cost 正确接入 STE 是必要且有价值的 co-design，但 STE 本身不是 headline 新颖性 |
| 性能证据 | 24 | 没有新 heldout speedup、cycle、RTL 或 PPA；只证明 proxy 可训练且不再看幅值 |
| fail-closed / provenance | 31 | 三项浅攻击通过，但 override 和自报重贴标签均可完整安装 revoked hooks |
| PAFT 正式就绪度 | 37 | 当前配置关闭是安全的；移除两项 P0 并获得 training-only Lloyd catalog 后才可 smoke |
| DATE 硬件证据增量 | 28 | 提升训练机制可信度，没有提升现有 1.503x/1.259x 性能结论 |

## 解锁下一轮的最小修改

1. 从生产 `pattern_paft.py` 删除 `unit_test_allow_unpinned_revoked_catalog` 分支。
2. 用 test-only synthetic 合法 catalog 重写 unit test，不再装入 M71 revoked catalog。
3. 外部 launch contract pin allowlisted catalog、builder/algorithm、训练 list、sample receipts、checkpoint、operators、valid/test overlap receipt；显式 deny M71 SHA `142e32...`。
4. receipt 增加 `pattern_paft.py` start/end SHA、train entrypoint SHA、launch contract SHA，并要求运行期间不漂移。
5. 在 attach hooks 前完成全部身份/shape/operator 检查，异常路径零副作用。
6. 保持两个正式配置 `enabled=false`；只有新的 DSEC-train-only、Phi/Lloyd catalog 经独立 overlap 审计后，才生成新的 smoke config。

当前允许：继续开发、test-only directed unit。当前禁止：M71 PAFT smoke、五轮训练、checkpoint selection、valid825 accuracy、speedup/DATE claim。
