# GateStack 活动 Token 与 R2/R4 Compactor RTL 验证（2026-07-15）

## 1. Profile 驱动结论

既有 `pair_k_count_ordered_trace` 保存 `[T=2,B,H,N]` 逐 token K-count。新增：

- `scripts/analyze_gatestack_compactor_profile.py`
- `scripts/test_analyze_gatestack_compactor_profile.py`
- `results/gatestack_compactor_profile_20260715.{json,md}`

重建的 active-lane 总和与 `projection_baseline_active_lanes_ordered_trace` 逐 row 完全一致。

| 指标 | mean | p99 | max |
|---|---:|---:|---:|
| 活动 token/head | 18.344 | 159 | 162 |
| 单 token 最大 K lane | 2.112 | 14 | 19 |
| R2 提取周期 | 35.818 | 470 | 862 |
| R4 提取周期 | 24.200 | 275 | 469 |

不允许跨 token 打包时，R2 精确周期是 `ceil(total/2)` 理想值的 1.1816 倍，R4 为 1.5817 倍。完整模型必须使用逐 token 求和，不能用理想总量除法。

## 2. 新增 RTL

| 文件 | 作用 |
|---|---|
| `rtl_hitflow/gatestack_active_token_iterator.sv` | 将 162-bit active-token mask 分为 9×18，按 token ID 顺序跳读 |
| `rtl_hitflow/gatestack_event_compactor.sv` | 参数化 WAYS=2/4，把单 token K bitmap 分批转成 lane event |
| `verif_hitflow/gatestack_event_compactor_assertions.sv` | stall 稳定、event count 范围、valid mask/count 一致 |
| `verif_hitflow/bind_gatestack_event_compactor_assertions.sv` | assertion bind |
| `tb_hitflow/tb_gatestack_active_token_iterator.sv` | 空/全满/随机 active-token mask 与随机反压 |
| `tb_hitflow/tb_gatestack_r4_event_compactor.sv` | 同一自检 TB 参数化验证 R2/R4 |
| `sim_hitflow/run_gatestack_compactor_checks.sh` | iverilog、Verilator assertion、Yosys 严格入口 |

## 3. 功能与反压结果

```text
PASS: active-token iterator loads=102 tokens=4254 stalls=1526
PASS: R2 event compactor tokens=502 events=4072 stalls=771
PASS: R4 event compactor tokens=502 events=4019 stalls=398
PASS: GateStack active-token + R2/R4 compactor；Verilator 0 warning/error
```

R2/R4 的随机 mask 序列不同是因为同一 PRNG 同时驱动反压，执行周期不同会改变后续随机状态；每档都由独立 expected mask 自检所有 event，无丢失、无重复、lane 顺序正确。

| 阶段 | Active-token | R2 | R4 |
|---|:---:|:---:|:---:|
| iverilog 自检 | PASS | PASS | PASS |
| Verilator assertions | N/A（复用已签 OBI 核） | PASS | PASS |
| Verilator warning/error | 0 | 0 | 0 |
| Yosys check | PASS | PASS | PASS |

## 4. 结构对照

Yosys `memory -nomap`：

| 模块/参数 | 通用 cell | mux | 说明 |
|---|---:|---:|---|
| active-token wrapper + OBI | 213 | 62 | 9×18 分段 token 选择 |
| event compactor R2 | **374** | **223** | 主线 |
| event compactor R4 | 726 | 459 | 消融 |

这不是工艺面积，但 R4 相比 R2 增加 94.1% 通用 cell 和 105.8% mux，而活动 token 跳读完整模型只从约 1.382x 提升到 1.386x。因此 R2 锁为首版，R4 不进入默认 DC 配置。

## 5. RTL 合同

### Active-token iterator

```text
load_active_token_mask[161:0]
  -> token_valid/ready
  -> {tag, token_id, token_last}
  -> done_valid/ready
```

mask 为空时不产生 token，直接 done。padding bit 固定为 0，输出 token ID 必须小于 162。

### Event compactor

```text
input : {tag, token_id, slot_id, K_bits[31:0]}
output: {tag, token_id, slot_id,
         lane_valid[WAYS], lane_id[WAYS], event_count, last_for_token}
```

同一 token 的 event 按 lane ID 递增。下游反压时整个 batch 稳定；最后一个 batch 握手后才接受下一 token。空 K token 可被消费但不产生 event。

## 6. 仍未闭合

1. event batch 到 packed slot 的多写 bank 映射尚未实现。
2. slot/lane 到 compact term ID 的映射和 prefix/base 生成尚未实现。
3. active-token scratch 随机读延迟尚未加入 RTL。
4. R2 priority 路径仍无目标库 500 MHz WNS。
5. 当前 TB 是随机合成 mask，不是把 672000 条真实 K bitmap 喂入 RTL；原 profile 只保存 K-count，没有保存完整 K bits。

## 7. 下一步

实现 `gatestack_capacity_mode_selector` 和 packed head-slot 元数据合同。先验证：

- `active_classes=4/5`；
- `CSR_bits=6642/6643`；
- CSR 与 RAW mode 在任意反压下只提交一次；
- RAW mode 不丢失 overflow 前已经捕获的 token。

完成后再实现 term prefix 和 slot SRAM adapter，避免先写完整顶层后才发现容量或端口合同错误。
