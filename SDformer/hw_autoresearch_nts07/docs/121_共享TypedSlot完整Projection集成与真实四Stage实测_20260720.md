# 共享 Typed-Slot 完整 Projection 集成与真实四 Stage 实测

## 1. 本轮结论

本轮第一次把片上 Builder 与既有 projection execution slice 通过同一份 typed-slot SRAM 生命周期直接接通，形成如下可执行链路：

```text
final-gate/K token
  -> C0或C1 canonical Builder
  -> IPD32W/FADC24/RAW41自动选择与序列化
  -> atomic typed-slot commit
  -> inspect/replay/release
  -> decoder/TDR/product/multicast
  -> AccTile/bias/final
```

Builder 的 payload SRAM 就是 execution 的 slot service，不再把 payload 复制到第二份私有 buffer。真实 `sample0/B0/window0` 的 S0-S3 共 45 个 head 已完成 C0/C1 双模式逐元素回放：每模式比较 233,280 项，双模式合计 466,560 项，失配为 0，`payload_copy_words=0`。

更重要的负面结论是：C1 不能作为默认架构。四 stage 中 Builder 局部加速为 1.408x，但在同步 bias SRAM 合同下 Builder 只占 C0 完整链路周期的 6.68%，因此系统总加速只有 1.020x。默认配置继续冻结为 C0+BPB，C1 仅保留为吞吐消融。

## 2. 集成边界与接口

新增顶层 `rtl_hitflow/gatestack_builder_projection_single_context_top.sv`，职责包括：

1. 按 `HEADS` 接收一批 head 的 final-gate/K token；
2. 先接纳并锁存 group tag/tile，再防止 head 重复、越界或批次未完成时启动 execution；
3. 将 Builder 的 inspect/replay/release 端口直接连接 execution 的 external-slot 接口；
4. 只在全部 head 已接受、已提交且全部 slot valid 时允许 group 启动；
5. group 完成后释放全部 slot 并重新开放下一批；
6. 通过显式 `batch_abort_valid/ready` 原子清理异常批次，并保持带 tag 的 `group_done_error` 直到上游握手；
7. 从 group 接纳到 execution 启动设置独立 Builder watchdog，覆盖 execution watchdog 尚未生效的阶段。

`gatestack_single_context_execution_top.sv` 增加了可静态选择的 external-slot service。启用后，旧私有 payload commit 端口保持 not-ready，execution 只消费外部 typed-slot 元数据和 payload word。旧内部 slot 模式仍通过原回归。

## 3. 小型端到端回归

缩参测试使用 `TOKENS=8、HEADS=2、OUT_TILE=2`，包含权重请求反压、同步 bias req/rsp、双 bank final 和两种 Builder：

| 模式 | Builder | Projection | 总周期 | 输出签名 | payload copy |
|---|---:|---:|---:|---:|---:|
| C0 | 115 | 137 | 252 | -112 | 0 |
| C1 | 72 | 137 | 209 | -112 | 0 |

局部 Builder 加速为 `115/72=1.597x`，系统加速为 `252/209=1.206x`。结果与 Amdahl 分账完全一致，说明 Builder 数字不能直接外推为完整链路数字。

该测试还执行以下异常恢复场景：

- 先提交 head0，再重复提交 head0；
- 重复 head 必须 not-ready，并锁存 `protocol_error`；
- wrapper 自动把 Builder 协议错误转换为原子 reset pulse 和带 tag 的 `group_done_error`；
- accepted/completed bitmap、typed-slot 与 execution 状态全部清零；
- 随后重新提交完整两 head 批次，仍得到原输出签名和周期。

同一回归还覆盖 host 主动 `batch_abort_valid/ready`，以及 group 已接纳但 head 永不到达时的 Builder watchdog；三条路径都必须返回且只返回一个带原 group tag 的 error completion。

C0/C1 的 Icarus 动态仿真、Verilator lint/elaboration 和 Erie 顶层 lint 均通过。

## 4. 真实 S0-S3 RTL 结果

输入来源：

- Builder：`tb_hitflow/vectors/gatestack_all45_builder_20260720`；
- projection：`tb_hitflow/vectors/real_sample0_s0_b0_capacity` 至 `s3`；
- 权重为 checkpoint 导出的逐输出通道 INT8 码，bias 与 expected output 来自同一整数金参考；
- execution 使用 adaptive CSR、residency enabled、32-lane output tile。

| Stage | Head | C0 总周期 | C1 总周期 | 系统加速 | C0 Builder | C1 Builder | Builder 加速 | C0 Builder 占比 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 3 | 4375 | 4055 | 1.079x | 1015 | 695 | 1.460x | 23.20% |
| S1 | 6 | 4652 | 4607 | 1.010x | 1037 | 992 | 1.045x | 22.29% |
| S2 | 12 | 28370 | 27385 | 1.036x | 3146 | 2161 | 1.456x | 11.09% |
| S3 | 24 | 173906 | 171166 | 1.016x | 8921 | 6180 | 1.444x | 5.13% |
| 合计 | 45 | 211303 | 207213 | **1.020x** | 14119 | 10028 | **1.408x** | **6.68%** |

数值和生命周期检查：

- 每模式 final 输出比较 233,280 项，双模式合计 466,560 项，失配 0；
- 两模式各 stage checksum 完全一致；
- 45 个 head 全部 atomic commit 和 release；
- projection head issue、term、bias/final 数量与 stage 金参考一致；
- payload copy 为 0；
- protocol、builder、error abort 和 timeout 均为 0；
- S1 的 6 个 head 全部为零 term，slot service 执行最小 replay，但仍完成 36 个 head×tile projection issue 和 31,104 项输出比较。

机器可读结果位于：

- `results/gatestack_builder_projection_real_allstages_20260720/report.json`；
- `results/gatestack_builder_projection_real_allstages_20260720/report.md`。

## 5. 对架构决策的修正

此前 C1 的 45-head Builder 周期为 `14078 -> 10035`，局部加速 1.403x。完整 projection 的真实结果表明该收益被后端稀释：

```text
C0 Builder占比 = 14119 / 211303 = 6.68%
C1 Builder加速 = 14119 / 10028 = 1.408x
Amdahl上限约   = 1 / ((1-0.0668) + 0.0668/1.408)
                = 1.020x
```

RTL 实测总加速也是 1.020x。因此：

1. C1 不再具备成为默认论文配置的定量依据；
2. 仅做跨 group 的 Builder/projection 重叠，理论上最多隐藏约 6.7% 当前周期，也不足以单独满足 1.20x 子系统门槛；
3. 下一轮架构必须直接优化 replay/decode、term×output-tile delivery、AccTile 和 bias/final；
4. C0+BPB 保持面积效率默认，C1 仅作“局部流水为何不能替代系统优化”的负面消融。

## 6. 当前验证边界

已完成的是一个真实 window、四 stage 的完整 projection 子系统 RTL，不是整网 ASIC：

- 尚无 100 帧、12 block、多 window 的逐 bit trace；
- 尚无 mapped SRAM、DC/STA、SAIF、功耗、布线或门级 LEC；
- Verilator 对真实大数组完成 lint/elaboration，动态长回归使用 Icarus；
- residual、skip、ATLIF 和 full encoder 调度不在本顶层；
- valid825 的最终量化部署合同仍需软件侧闭环。

## 7. 可复现入口

```bash
bash sim_hitflow/run_gatestack_builder_projection_single_context_small.sh
bash sim_hitflow/run_gatestack_builder_projection_real_s0.sh
PYTHONPATH=scripts python3 -m unittest \
  scripts/test_summarize_gatestack_builder_projection_allstages.py
python3 scripts/summarize_gatestack_builder_projection_allstages.py \
  --build-dir build_hitflow/gatestack_builder_projection_real_allstages \
  --out-dir results/gatestack_builder_projection_real_allstages_20260720
```
