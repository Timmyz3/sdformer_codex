# Local5 theta 折叠生产链与 TCFM5 投影闭环

## 本轮结论

本轮只关闭 Local5 当前最高优先级的数值合同缺口，没有启动
12-block scheduler，也没有把已有 banking、FIFO 或定点化重新包装为
DATE 创新。

生产 profile、acceptance 和 projection vector 链现在强制使用
`local5_checkpoint_projection_contract_v2`：

```text
checkpoint K-ATLIF theta_block + projection W_float
                         |
                         v
              W_eff = theta_block * W_float
                         |
                         v
          per-output-channel dyadic INT8 quantization
                         |
                         v
            binary K event * gate_Q1.7 * W_eff_INT8
                         |
                         v
                 per-head Acc32 partial sum
```

运行时 K 仍然是 1-bit event，不增加 theta 乘法器或事件带宽。

## 生产链修改

1. `profile_local5_hardware_features.py` 从每个 attention block 显式读取
   `sn_k.spiking_neuron.thresh`，先折叠再量化，导出 12 block 的 v2
   manifest 和 72 个 payload 数组；
2. `validate_local5_postg0_acceptance.py` 拒绝 v1、缺 theta、非正 theta、
   错误量化顺序或错误 runtime contract；
3. `generate_local5_active_projection_postg0_vectors.py` 新增显式 v2 模式和
   可审计 contract override，同时保留旧 v1 重放能力；
4. checkpoint-bound follower 只接受
   `checkpoint_theta_folded_dyadic_int8_head_slice`；
5. TCFM5/Linear5 回归入口现可接受外部预生成向量目录，不再
   在回归前强制覆盖为默认 synthetic 向量。

## 独立导出器与生产导出器对拍

旧 fullres Local5 checkpoint 上：

| 检查 | 结果 | 证据 |
|---|---:|---|
| schema | v2 / v2 | `[checkpoint-static]` |
| attention block | 12 / 12 | `[checkpoint-static]` |
| payload array | 72 | `[checkpoint-static]` |
| 生产 vs 旁路数组失配 | 0 | `[checkpoint-static]` |
| 折叠导致 INT8 码改变 | 80 / 2,156,544 | `[checkpoint-static]` |
| dyadic exponent 改变 | 0 / 4,416 | `[checkpoint-static]` |

该对拍防止“独立旁路是对的，但生产 profiler 仍导出错误权重”。它不证明
new rank-1，因为新 checkpoint 尚未释放。

## 真实 T450 投影 RTL 回放

统一向量来自旧 checkpoint 的 qualified post-G0 profile100，固定按四个
stage 各抽取 25 组，合计 100 组。向量 manifest 绑定 checkpoint SHA、
trace SHA、v2 manifest SHA 和 v2 payload SHA。

| 对照 | 周期 | 结果 | 证据等级 |
|---|---:|---:|---|
| Direct-1RW | 121,501 | 100/100 组 Acc32 零失配 | `[rtl]` |
| DS-GASR-2C | 119,348 | 100/100 组 Acc32 零失配 | `[rtl]` |
| TCFM5，关系 SRAM L1 | 109,343 | 90,000 个 Acc32 零失配 | `[rtl]` |
| Linear5，关系 SRAM L1 | 158,476 | 90,000 个 Acc32 零失配 | `[rtl]` |
| TCFM5，关系 SRAM L2 | 112,418 | 90,000 个 Acc32 零失配 | `[rtl]` |
| Linear5，关系 SRAM L2 | 159,534 | 90,000 个 Acc32 零失配 | `[rtl]` |

TCFM5 相对等资源 Linear5 的聚合周期收益为 L1 `1.449x`、L2
`1.419x`。这与折叠前趋势相同，因为 theta 修正只改变权重数值，不改变
term 数、目的地或 bank 冲突。因此本轮不增加性能贡献，只将原有 TCFM5
性能证据从“错误 value 数值合同”修正为“theta-folded 数值合同”。

## 验证与产物

- 合同与 acceptance 相关单测：10/10 PASS；
- Direct/DS-GASR：确定性、随机空泡+SVA、lint、Yosys check 全部 PASS；
- TCFM5/Linear5：Icarus 小例、Verilator+SVA 真实 100 组、L1/L2、
  Yosys 结构检查全部 PASS。

主报告：

- `results/local5_theta_folded_deployment_closure_20260805/report.md`；
- `results/local5_theta_folded_qgasr2c_fivebank_rtl_20260805/report.md`；
- `results/local5_theta_folded_tcfm5_linear5_rtl_20260805/report.md`。

## DATE 贡献边界

本轮是必要的正确性与系统完整度工作，创新性增量为 0。不单独宣称：

- theta 折叠是 DATE 架构创新；
- INT8 定点化是架构创新；
- TCFM5 `1.449x` 是本轮新获得的收益；
- 100 组部分 Acc32 等于完整 attention、encoder 或网络 bit-exact；
- 同步 SRAM 端口合同等于 foundry SRAM macro 或 ASIC PPA。

它的价值是解除 Local5 端到端集成前的已知数值阻断，使“双向五色关系数据流
+ source-major term + TCFM5”不再建立在错误的 `theta=1` 假设上。

## 尚未完成

1. 正在训练的 Local5 新 rank-1 checkpoint/profile100/all12 尚未释放；
2. 本轮只到 per-head Acc32，不包含 cross-head reduction、bias、no-running BN、
   requant、残差、ATLIF 和 decoder；
3. `score/Shiftmax5 -> relation transpose -> source-major term -> TCFM5`
   仍未在一个顶层里用同一真实 trace 闭环；
4. 12-block descriptor scheduler 未集成；
5. 无 DC/STA/SAIF 与 foundry SRAM macro。

## 下一唯一门槛

等新 Local5 rank-1 释放，使用当前生产 v2 链原样重跑 profile100/all12、
checkpoint SHA acceptance、Direct/TCFM5 Acc32 零失配。新 checkpoint 通过前，
不把旧 checkpoint 结果写成正式 Local5 部署合同，也不把 12-block scheduler
提前接到尚未冻结的数值边界。

## DATE 独立评审（第一轮）

独立 DATE 审稿视角评分为 `3/5，Reject / Major Revision`。本轮对正确性和
系统完整度分别有增量，但创新性增量为 0。审稿人认可 theta 折叠已经进入
生产 exporter、acceptance 和真实 T450 投影回放；同时指出原 acceptance 只校验
manifest 元数据与 SHA，仍可能接受“payload 被修改，同时把 manifest SHA 一起
更新”的自洽错误。

该问题有道理，而且属于部署合同的 fail-closed 缺口，不是写作问题。

## 评审整改：checkpoint 独立数值重算

新增 `verify_local5_theta_folded_projection_contract.py`。该工具不信任 manifest
中的量化数组，而是独立完成以下步骤：

1. 按 manifest 绑定的 SHA 重新加载 checkpoint；
2. 对全部 12 个 attention block 重新读取原始 `proj.weight`、K-ATLIF theta
   和 `proj.bias`；
3. 重新计算 `W_eff=theta_K×W_float`；
4. 调用权威 dyadic 量化函数重新产生 INT8 weight 与 per-output scale；
5. 对生产 payload 的全部 72 个数组逐元素比较。

旧 checkpoint 的独立重算结果为：

| 检查项 | 数量 | 结果 | 证据 |
|---|---:|---:|---|
| attention block | 12 | PASS | `[checkpoint-static]` |
| payload array | 72 | PASS | `[checkpoint-static]` |
| INT8 weight | 2,156,544 | 零失配 | `[checkpoint-static]` |
| dyadic scale | 4,416 | 零失配 | `[checkpoint-static]` |
| bias | 4,416 | 零失配 | `[checkpoint-static]` |

acceptance 现在将该重算设为必需检查
`checkpoint_projection_payload_recomputed`。单元测试还构造了一个负例：修改
payload 后同步更新 manifest 中的 payload SHA；旧的 SHA-only 验证会接受，新的
checkpoint 重算会 fail-closed。9/9 单元测试全部通过。

为避免长驻 watcher 在源码修改前就 import 旧版本模块，旧硬件 follower 已单独
停止并使用最新源码重新启动；训练进程未被停止。生产 follower 不传 contract
override，显式 override 仅保留给离线审计重放，不能绕过正式 rank-1 入口。

整改后关闭了第一轮评审提出的 provenance/numeric-binding 缺口，但不改变本轮
创新性结论，也不提前宣称新 rank-1 已经闭环。

## DATE 独立评审（第二轮）

第二轮仍为 `3/5，Reject / Major Revision`。审稿人确认“修改 payload 并同步更新
SHA”已被 checkpoint 独立重算关闭，但指出 verifier 仍信任 manifest 自报的
`module/weight_name/theta_name/prefix`。如果攻击者同时交换 block 参数映射与
对应 payload，旧 verifier 仍可能得到一个数值自洽、但部署身份错误的合同。

这项意见成立。它说明数值正确性和拓扑身份正确性是两个独立条件。

## 评审整改：固定 12-block 拓扑 ABI

verifier 现在内建唯一 Local5 部署拓扑，而不是从 manifest 学习拓扑：

| stage | depth | channel | heads | head-dim |
|---:|---:|---:|---:|---:|
| S0 | 2 | 96 | 3 | 32 |
| S1 | 2 | 192 | 6 | 32 |
| S2 | 6 | 384 | 12 | 32 |
| S3 | 2 | 768 | 24 | 32 |

对每个 `(stage, block)`，verifier 独立推导并逐字段校验：

- 固定 row 顺序与 12 个唯一 block；
- `sttmultires_unet.encoders.swin3d.layers.S.swin_blocks.B.attn`；
- `proj.weight`、`sn_k.spiking_neuron.thresh`、`proj.bias` 参数名；
- `sS_bB` payload prefix；
- 权重形状、head 数、head-dim 与 bias presence。

固定合同标识为
`local5_swin_2_2_6_2_c96_192_384_768_h3_6_12_24_v1`。生产 exporter
会为新产物写入该标识与显式 `bias_name`；verifier 对历史 v2 manifest 仍通过
逐字段推导完成相同检查，不依赖该自报标识。

新增攻击负例同时执行三件事：交换 S0B0/S0B1 的 module/weight/theta/bias
映射、交换六类 payload 数组、更新 payload SHA。该产物在数值上仍可自洽，
但现在于 checkpoint 数值重算前即因固定拓扑映射不匹配而 fail-closed。

当前合同/acceptance 单元测试为 10/10 PASS。旧 checkpoint 的独立重算仍为
2,156,544 个 INT8 weight、4,416 个 scale、4,416 个 bias 零失配，并新增
`topology_mapping=PASS_FIXED_12_BLOCK_ABI`。

第二轮提出的 manifest 拓扑语义漏洞至此关闭；创新性增量仍为 0。正式进入
12-block scheduler 前仍必须等待新 rank-1 使用同一 ABI 完成生产导出、独立重算
和 Direct/TCFM5 Acc32 回放。

## DATE 独立评审（第三轮）

第三轮对“theta-folded 部署合同工作包”的评分提升为
`4/5，Conditional Accept`，但整篇 Local5 硬件论文仍是大修状态。审稿人确认：

- 固定 ABI 已实质关闭 manifest 参数映射与 payload 同步篡改；
- 10/10 测试包含 payload 自洽篡改和 block-remap 自洽篡改两个负例；
- 旧 checkpoint 的全量独立重算可作为可信的 checkpoint-bound、ABI-bound
  正确性证据；
- 本轮创新性增量仍为 0。

复审提出新生产产物必须显式携带 ABI 字符串。该 minor 已进一步收紧：正式
post-G0 acceptance 现在要求
`topology_contract=local5_swin_2_2_6_2_c96_192_384_768_h3_6_12_24_v1`，
并输出 `checkpoint_projection_topology_abi=true`。历史 manifest 仅允许被独立
verifier 审计，不再能通过新的正式 acceptance。

同时修复 checkpoint-bound follower 的收尾缺陷：最终 provenance 组装曾引用
未定义的 `projection_manifest`，现改为在读取 vector manifest 后立即解析并复用。
该问题不会改变已有周期或数值结果，但若不修会在所有 RTL 回归完成后阻止
`checkpoint_bound_scope.json` 落盘。9 个相关 unittest 与 6 个 watcher 定向检查
均通过。

截至本节，唯一剩余 major gate 是新 rank-1：必须用同一生产 ABI 完成 v2 导出、
checkpoint 独立重算、profile100/all12、Direct/TCFM5 Acc32 零失配和完整 SHA
贯通。在此之前仍不进入正式 12-block scheduler 集成。
